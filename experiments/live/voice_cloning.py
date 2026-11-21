"""Clone the operator's voice for the voice bot: Chatterbox, OpenVoice.

Synthetic only. A reference voice stands in for the enrolled operator:
a Piper voice that is not the bot's (``es_ES-sharvard-medium`` speaker
0, and ``es_ES-mls_9972-low`` as a second one) reads the Spanish
enrolment text (``transcriber/domain/enrolment.py``, about 30 s), as a
person does at enrolment. Each engine runs as the voice bot would run
it, the worker ``lab/runners/voice_clone.py`` in its own virtual
environment, and speaks ten short bot-like Spanish replies in each
reference's voice. Measured per engine:

* **first audio**: wall time from writing the sentence to the worker
  until its answer line (the WAV is written), warm; the cold load (start
  to ``ready``) and the first request of a new reference (its speaker
  conditioning) are reported apart;
* **similarity**: cosine of WavLM x-vectors (``microsoft/wavlm-base-
  plus-sv``, the enrolment voiceprints' encoder, one vector per whole
  clip) between each cloned sentence and the reference reading, and
  between each cloned sentence and the same sentence read by the
  reference's Piper voice itself (the "true" voice);
* **naturalness proxy**: word error rate of faster-whisper
  large-v3-turbo (int8) on the cloned audio against the text;
* **GPU memory**: peak ``nvidia-smi`` memory of the worker process.

OpenVoice is also run on the CPU (the GPU is shared with the live
recogniser); Chatterbox is not (it is already slow on the GPU).

Baselines with no cloning: the bot's Piper voice ``es_ES-davefx-medium``
(similarity to each reference, WER) and the reference voice itself (its
renderings of the sentences, and the two halves of the reference
reading).

Usage (the shared ``.venv``; take the GPU lease first)::

    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python \\
        experiments/live/voice_cloning.py --work DIR \\
        [--runs chatterbox@cuda openvoice@cuda openvoice@cpu] \\
        [--out experiments/live/voice_cloning_results.json]
"""

import argparse
import json
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from transcriber.application.ports import Audio
from transcriber.domain.comparison import normalise_text, word_error
from transcriber.domain.enrolment import enrolment_text
from transcriber.domain.live import percentile

DATA = Path("~/.local/share/ragputin").expanduser()
PIPER_PYTHON = DATA / "tts-venv/bin/python"
VOICES = DATA / "voices"
BOT_VOICE = "es_ES-davefx-medium"
REFERENCES = {
    "sharvard": ("es_ES-sharvard-medium", "Lucía"),
    "mls9972": ("es_ES-mls_9972-low", "Marta"),
}
"""Reference id → (Piper voice, name read in the enrolment text)."""
ENGINES = {
    "chatterbox": DATA / "clone-chatterbox-venv/bin/python",
    "openvoice": DATA / "clone-openvoice-venv/bin/python",
}
WORKER = Path(__file__).resolve().parents[2] / "lab/runners/voice_clone.py"
WHISPER = DATA / "lab-models/faster-whisper-large-v3-turbo-int8"
SENTENCES = (
    "Claro, ahora mismo reviso el estado de tu pedido.",
    "Tu factura de este mes ya está disponible en el área de cliente.",
    "Para cambiar la contraseña, entra en ajustes y pulsa en seguridad.",
    "Lo siento, ese dato no aparece en la documentación que tengo.",
    "El técnico pasará mañana por la tarde, entre las cuatro y las seis.",
    "Si el router sigue sin conexión, apágalo y vuelve a encenderlo.",
    "Perfecto, he anotado tu teléfono para que te llamen esta semana.",
    "La devolución tarda unos cinco días laborables en llegar a tu cuenta.",
    "¿Quieres que te envíe el resumen por correo electrónico?",
    "Gracias por tu paciencia, enseguida te paso con un compañero.",
)
"""Ten synthetic bot replies, 8 to 13 words, numbers written as words."""

Floats = npt.NDArray[Any]
"""An array of samples or a vector."""


def piper(voice: str, text: str, out: Path, speaker: int = 0) -> Path:
    """Render ``text`` with Piper (its own venv) into ``out``."""
    command = [str(PIPER_PYTHON), "-m", "piper", "-m",
               str(VOICES / f"{voice}.onnx"), "-f", str(out)]
    if speaker:
        command += ["-s", str(speaker)]
    subprocess.run(command, input=text, text=True, check=True,
                   capture_output=True)
    return out


def read_wav(path: Path) -> Tuple[Floats, int]:
    """Return a 16-bit mono WAV file's samples in [-1, 1] and its rate."""
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        pcm = reader.readframes(reader.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, rate


def at_16k(path: Path) -> Floats:
    """Return the file's samples at 16 kHz."""
    samples, rate = read_wav(path)
    if rate == 16000:
        return samples
    divisor = np.gcd(rate, 16000)
    resampled: Floats = resample_poly(samples, 16000 // divisor,
                                      rate // divisor)
    return resampled.astype(np.float32)


class GpuPeak:
    """Poll ``nvidia-smi`` for the peak memory of one process."""

    def __init__(self, pid: int) -> None:
        """Start polling ``pid`` every 0.2 s."""
        self.pid = pid
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        """Keep the largest value seen."""
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(self.pid):
                    self.peak = max(self.peak, int(parts[1]))
            self._stop.wait(0.2)

    def stop(self) -> int:
        """Stop polling; return the peak MiB."""
        self._stop.set()
        self._thread.join()
        return self.peak


def run_engine(run: str, references: Dict[str, Path],
               work: Path) -> Dict[str, Any]:
    """Run one engine's worker (``engine@device``) over every reference."""
    engine, device = run.split("@")
    command = [str(ENGINES[engine]), str(WORKER), "--engine", engine,
               "--device", device]
    if engine == "openvoice":
        command += ["--piper-python", str(PIPER_PYTHON), "--piper-model",
                    str(VOICES / f"{BOT_VOICE}.onnx")]
    log = (work / f"{run}.stderr").open("w")
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=log,
                               text=True, bufsize=1, cwd="/tmp")
    peak = GpuPeak(process.pid)
    assert process.stdin is not None and process.stdout is not None
    ready = json.loads(process.stdout.readline())
    cold_ms = (time.perf_counter() - started) * 1000.0

    def ask(text: str, reference: Path,
            out: Path) -> Tuple[float, Dict[str, Any]]:
        assert process.stdin is not None and process.stdout is not None
        request = {"id": out.stem, "text": text, "reference": str(reference),
                   "out": str(out), "language": "es"}
        begin = time.perf_counter()
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        answer = json.loads(process.stdout.readline())
        return (time.perf_counter() - begin) * 1000.0, answer

    result: Dict[str, Any] = {"ready": ready, "cold_ms": round(cold_ms, 1),
                              "references": {}}
    for ref_id, reference in references.items():
        outdir = work / run / ref_id
        outdir.mkdir(parents=True, exist_ok=True)
        prep_ms, _ = ask("Hola, buenos días.", reference,
                         outdir / "warmup.wav")
        rows = []
        for n, sentence in enumerate(SENTENCES):
            wall, answer = ask(sentence, reference, outdir / f"s{n:02d}.wav")
            rows.append({"n": n, "wall_ms": round(wall, 1),
                         "synth_ms": answer.get("synth_ms"),
                         "error": answer.get("error"),
                         "path": answer.get("path")})
        result["references"][ref_id] = {
            "reference_prep_ms": round(prep_ms, 1), "sentences": rows}
    process.stdin.close()
    process.wait(timeout=60)
    result["gpu_peak_mib"] = peak.stop()
    log.close()
    return result


def stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Return n, p50, p90 and max (one decimal)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 1)

    return {"n": len(values), "p50": r(percentile(values, 0.5)),
            "p90": r(percentile(values, 0.9)),
            "max": r(max(values) if values else None)}


class Scorer:
    """WavLM similarity and Whisper WER (both on the GPU)."""

    def __init__(self) -> None:
        """Load both models."""
        from faster_whisper import WhisperModel

        from transcriber.adapters.outbound.wavlm_voice_encoder import (
            WavLMVoiceEncoder,
        )
        self.encoder = WavLMVoiceEncoder(device="cuda")
        self.whisper = WhisperModel(str(WHISPER), device="cuda",
                                    compute_type="int8_float16")

    def vector(self, samples: Floats) -> Floats:
        """Return the unit x-vector of a whole 16 kHz clip."""
        audio = Audio(samples=samples, sample_rate=16000)
        row: Floats = self.encoder.embed(audio, [(0.0, audio.duration)])[0]
        return row

    def wer(self, path: Path, text: str) -> Optional[float]:
        """Return Whisper's word error rate on ``path`` against ``text``."""
        segments, _ = self.whisper.transcribe(str(path), language="es",
                                              beam_size=5)
        heard = " ".join(s.text for s in segments)
        return word_error(normalise_text(text), normalise_text(heard))


def mean(values: Sequence[float]) -> Optional[float]:
    """Return the mean (four decimals), ``None`` for no values."""
    return round(float(np.mean(values)), 4) if values else None


def score_set(scorer: Scorer, paths: Sequence[Path], ref_vec: Floats,
              true_vecs: Sequence[Floats]) -> Dict[str, Any]:
    """Similarity to the reference and to the true voice, and WER."""
    to_ref, to_true, wers = [], [], []
    for n, path in enumerate(paths):
        vec = scorer.vector(at_16k(path))
        to_ref.append(float(vec @ ref_vec))
        to_true.append(float(vec @ true_vecs[n]))
        wer = scorer.wer(path, SENTENCES[n])
        if wer is not None:
            wers.append(wer)
    return {"similarity_to_reference": mean(to_ref),
            "similarity_to_reference_min": round(min(to_ref), 4),
            "similarity_to_true_voice": mean(to_true),
            "wer_mean": mean(wers),
            "wer_pooled_sentences_with_errors": sum(w > 0 for w in wers)}


def main() -> None:
    """Render, run the engines, score, write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--runs", nargs="+",
        default=["chatterbox@cuda", "openvoice@cuda", "openvoice@cpu"],
        help="engine@device to run",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/live/"
                                     "voice_cloning_results.json"))
    args = parser.parse_args()
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    text = enrolment_text("es")
    references: Dict[str, Path] = {}
    truth: Dict[str, List[Path]] = {}
    for ref_id, (voice, name) in REFERENCES.items():
        references[ref_id] = piper(voice, text.text(name),
                                   work / f"reference_{ref_id}.wav")
        truth[ref_id] = [piper(voice, s, work / f"true_{ref_id}_{n:02d}.wav")
                         for n, s in enumerate(SENTENCES)]
    bot = [piper(BOT_VOICE, s, work / f"bot_{n:02d}.wav")
           for n, s in enumerate(SENTENCES)]
    runs = {run: run_engine(run, references, work) for run in args.runs}

    scorer = Scorer()
    results: Dict[str, Any] = {
        "ticket": "#207", "synthetic": True,
        "sentences": list(SENTENCES),
        "references": {r: {"piper_voice": v, "enrolment_text": text.version,
                           "seconds": round(len(at_16k(references[r]))
                                            / 16000, 1)}
                       for r, (v, _) in REFERENCES.items()},
        "similarity_model": scorer.encoder.model_name,
        "wer_model": "faster-whisper large-v3-turbo int8_float16, beam 5",
        "baselines": {}, "engines": {},
    }
    for ref_id, reference in references.items():
        samples = at_16k(reference)
        ref_vec = scorer.vector(samples)
        half = len(samples) // 2
        halves = float(scorer.vector(samples[:half])
                       @ scorer.vector(samples[half:]))
        true_vecs = [scorer.vector(at_16k(p)) for p in truth[ref_id]]
        results["baselines"][ref_id] = {
            "reference_halves_similarity": round(halves, 4),
            "true_voice": score_set(scorer, truth[ref_id], ref_vec,
                                    true_vecs),
            "bot_voice_davefx": score_set(scorer, bot, ref_vec, true_vecs),
        }
        for engine, run in runs.items():
            ref_run = run["references"][ref_id]
            paths = [Path(r["path"]) for r in ref_run["sentences"]
                     if r["path"]]
            entry = results["engines"].setdefault(engine, {
                "cold_load_ms": run["cold_ms"],
                "sample_rate": run["ready"].get("sample_rate"),
                "gpu_peak_mib": run["gpu_peak_mib"], "references": {}})
            errors = [r for r in ref_run["sentences"] if r["error"]]
            entry["references"][ref_id] = {
                "reference_prep_ms": ref_run["reference_prep_ms"],
                "first_audio_ms": stats([r["wall_ms"]
                                         for r in ref_run["sentences"]
                                         if not r["error"]]),
                "errors": len(errors),
                **(score_set(scorer, paths, ref_vec, true_vecs)
                   if len(paths) == len(SENTENCES) else {}),
            }
    for engine, run in runs.items():
        walls = [r["wall_ms"] for ref in run["references"].values()
                 for r in ref["sentences"] if not r["error"]]
        results["engines"][engine]["first_audio_ms_all"] = stats(walls)
        audio_ms = []
        for ref in run["references"].values():
            for r in ref["sentences"]:
                if r["path"]:
                    samples, rate = read_wav(Path(r["path"]))
                    audio_ms.append(len(samples) * 1000.0 / rate)
        results["engines"][engine]["audio_ms_per_sentence"] = stats(audio_ms)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False)
                        + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
