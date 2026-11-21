"""Which local Spanish TTS sounds most natural on the RTX 3070 Ti (#214).

The voice bot speaks with Piper ``es_ES-davefx-medium`` today, which the
owner finds robotic. This measures newer open models that fit the 8 GB
GPU beside the live recogniser, each the way the bot would run it: one
worker process per engine in the engine's own virtual environment (this
same file run with ``--worker ENGINE``), fed one sentence at a time, the
model warm. Per engine:

* **first audio**: time from receiving a sentence until the first audio
  chunk exists (streaming engines), or the whole sentence (the others);
  measured inside the worker, so pipe and file costs are left out;
* **RTF**: synthesis time over audio duration (below 1 is faster than
  real time);
* **cold load** (process start to ``ready``) and **peak GPU memory** of
  the worker process (``nvidia-smi``, polled every 0.2 s);
* **WER** of faster-whisper large-v3-turbo (int8, beam 5) against the
  text: intelligibility, not naturalness;
* **UTMOS** (``utmos22_strong``, SpeechMOS v1.2.0, the VoiceMOS 2022
  winner; predicts a 1-5 listener MOS): a naturalness *predictor* trained
  on English, read only as a rough ranking, never instead of the owner's
  ears;
* **similarity** to the reference voice for the cloning engines (WavLM
  x-vectors, as ``voice_cloning.py``).

Sentences: 5 interview sentences (the listening samples) and the 10 bot
replies of ``voice_cloning.py`` (more latency points). The cloning
engines clone one *human* Spanish voice: CML-TTS (CC BY 4.0, LibriVox),
Spanish test row 217, speaker 6306, 12.5 s, fetched with ``--fetch-
reference`` (its accent was not checked).

Engines (``ENGINE@device[:voice]``):

``piper``     Piper davefx, sentence-level chunks (today's bot voice)
``openvoice`` Piper davefx + OpenVoice v2 converter (today's clone,
              ``lab/runners/voice_clone.py``)
``kokoro``    Kokoro-82M v1.0 (Apache-2.0), voices ``ef_dora``/``em_alex``
``pocket``    Kyutai Pocket TTS ``spanish`` (CC-BY-4.0), voice ``lola``
``qwen3``     Qwen3-TTS-12Hz-0.6B-Base (Apache-2.0) through
              faster-qwen3-tts (CUDA graphs), streaming, 4 codec steps
              (333 ms of audio) per chunk, cloning the reference,
              repetition penalty 1.3 (the default 1.05 babbles on)
``ming``      Ming-3.0-tts-0.5B (MIT), cloning the reference; streamed
              per latent patch through the audio VAE's incremental decode

Usage (take the GPU lease first; never the shared ``.venv``'s ``uv run``)::

    L=~/.local/share/ragputin/tts-lab
    $L/score-venv/bin/python experiments/live/natural_tts.py \\
        --fetch-reference $L/reference_cml217.wav
    HF_HUB_OFFLINE=1 TORCH_HOME=$L/torch-hub PYTHONPATH=. \\
        $L/score-venv/bin/python experiments/live/natural_tts.py \\
        --work DIR [--runs piper@cpu kokoro@cuda:ef_dora ...]

``score-venv`` is the shared ``.venv`` (through a ``.pth``) plus a
Python-only torchaudio for UTMOS.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = Path("~/.local/share/ragputin").expanduser()
LAB = DATA / "tts-lab"
VENVS = {
    "piper": DATA / "tts-venv/bin/python",
    "openvoice": DATA / "clone-openvoice-venv/bin/python",
    "kokoro": LAB / "kokoro-venv/bin/python",
    "pocket": LAB / "pocket-venv/bin/python",
    "qwen3": LAB / "qwen3-venv/bin/python",
    "ming": LAB / "ming-venv/bin/python",
}
MING_SOURCE = LAB / "src/Ming-3.0-tts"
PIPER_MODEL = DATA / "voices/es_ES-davefx-medium.onnx"
WHISPER = DATA / "lab-models/faster-whisper-large-v3-turbo-int8"
REFERENCE_ROW = 217
REFERENCE_TEXT = (
    "Los niños, vestidos de blanco y rosa, saltaban y corrían tras sus"
    " juguetes, o formaban alegres corros girando como ruedas de colores"
)
"""CML-TTS Spanish test row 217 (speaker 6306) and its transcript."""
INTERVIEW = (
    "Cuéntame un poco sobre tu experiencia con bases de datos.",
    "Vale, entiendo. ¿Y cómo resolviste ese problema?",
    "Tómate tu tiempo, no hay prisa.",
    "Interesante. ¿Podrías darme un ejemplo concreto?",
    "Perfecto, pasamos al siguiente tema.",
)
"""The five listening sentences (issue #214)."""
BOT_REPLIES = (
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
"""The ten bot replies of ``voice_cloning.py``."""
SENTENCES = INTERVIEW + BOT_REPLIES
CLONERS = ("openvoice", "qwen3", "ming")


# --------------------------------------------------------------- worker ---

def write_wav(path: Path, audio: Any, rate: int) -> float:
    """Write float audio in [-1, 1] as 16-bit mono; return its seconds."""
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1),
                      -1.0, 1.0)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((samples * 32767.0).astype("<i2").tobytes())
    return len(samples) / float(rate)


class Engine:
    """One TTS model: ``stream(text)`` yields float audio chunks."""

    rate = 24000

    def stream(self, text: str) -> Iterator[Any]:
        """Yield the audio of ``text`` chunk by chunk."""
        raise NotImplementedError


class PiperEngine(Engine):
    """Piper (its own venv), one chunk per sentence of the text."""

    def __init__(self, device: str, voice: str) -> None:
        """Load the ONNX voice."""
        from piper import PiperVoice  # type: ignore[import-not-found]
        self.voice = PiperVoice.load(str(PIPER_MODEL),
                                     use_cuda=device == "cuda")
        self.rate = self.voice.config.sample_rate

    def stream(self, text: str) -> Iterator[Any]:
        """Yield one chunk per sentence."""
        for chunk in self.voice.synthesize(text):
            yield chunk.audio_float_array


class KokoroEngine(Engine):
    """Kokoro-82M, Spanish G2P (espeak-ng), one chunk per text segment."""

    def __init__(self, device: str, voice: str) -> None:
        """Load the pipeline and the voice."""
        from kokoro import KPipeline  # type: ignore[import-not-found]
        self.pipeline = KPipeline(lang_code="e", repo_id="hexgrad/Kokoro-82M",
                                  device=device)
        self.voice = voice or "ef_dora"

    def stream(self, text: str) -> Iterator[Any]:
        """Yield one chunk per segment the pipeline splits."""
        for _, _, audio in self.pipeline(text, voice=self.voice):
            yield audio.detach().cpu().numpy()


class PocketEngine(Engine):
    """Kyutai Pocket TTS, Spanish model, a catalogue voice, streaming."""

    def __init__(self, device: str, voice: str) -> None:
        """Load the Spanish model and the voice state."""
        from pocket_tts import TTSModel  # type: ignore[import-not-found]
        self.model = TTSModel.load_model(language="spanish")
        if device == "cuda":
            self.model.to("cuda")
        self.state = self.model.get_state_for_audio_prompt(voice or "lola")
        self.rate = self.model.sample_rate

    def stream(self, text: str) -> Iterator[Any]:
        """Yield the model's streamed chunks."""
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield chunk.detach().float().cpu().numpy()


class Qwen3Engine(Engine):
    """Qwen3-TTS 0.6B Base through faster-qwen3-tts, cloning, streaming."""

    def __init__(self, device: str, voice: str, reference: str) -> None:
        """Load the model, capture the CUDA graphs."""
        from faster_qwen3_tts import (  # type: ignore[import-not-found]
            FasterQwen3TTS,
        )
        self.model = FasterQwen3TTS.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", device=device)
        self.reference = reference

    def stream(self, text: str) -> Iterator[Any]:
        """Yield 4-step (333 ms) chunks; cap the tokens to the text."""
        words = len(text.split())
        cap = int(12 * (2.0 + 0.6 * words) * 1.5)
        # The default repetition penalty (1.05) let 5 of 16 test sentences
        # run on to the cap (babble after the text); 1.3 stopped all 16.
        for audio, rate, _ in self.model.generate_voice_clone_streaming(
                text=text, language="Spanish", ref_audio=self.reference,
                ref_text=REFERENCE_TEXT, chunk_size=4, max_new_tokens=cap,
                repetition_penalty=1.3):
            self.rate = rate
            yield audio


class MingEngine(Engine):
    """Ming-3.0-tts-0.5B, cloning, streamed patch by patch."""

    def __init__(self, device: str, voice: str, reference: str) -> None:
        """Load the model from its source checkout."""
        import soundfile
        import torch
        import torchaudio  # type: ignore[import-not-found]
        sys.path.insert(0, str(MING_SOURCE))

        def load(path: Any, *args: Any, **kwargs: Any) -> Any:
            data, rate = soundfile.read(str(path), dtype="float32",
                                        always_2d=True)
            return torch.from_numpy(data.T.copy()), rate

        # torchaudio >= 2.9 loads files through torchcodec, not installed.
        torchaudio.load = load
        from ming_tts import MingTTS  # type: ignore[import-not-found]
        self.tts = MingTTS("inclusionAI/Ming-3.0-tts-0.5B",
                           device=f"{device}:0" if device == "cuda"
                           else device, text_normalization=False)
        self.rate = self.tts.sample_rate
        self.reference = reference
        import tts_tasks  # type: ignore[import-not-found]
        self.request = tts_tasks.build_task_generation_kwargs(
            "MULTILINGUAL", text="x", language_id="Spanish",
            instruction=None, controls=None, instruction_schema=None,
            prompt_wav_path=reference, prompt_text=REFERENCE_TEXT,
            spk_emb_path=None, use_spk_emb=True, use_zero_spk_emb=False,
            cfg=2.0, sigma=0.25, temperature=0.0)
        self.latent, self.spk = self.tts.prepare_prompt(reference, True)

    def stream(self, text: str) -> Iterator[Any]:
        """Yield audio patch by patch: ``sample`` + streaming VAE decode.

        ``MingTTS.synthesize`` decodes the whole utterance at once; the
        model's ``sample`` already yields one latent patch per step and
        the audio VAE decodes incrementally (``use_cache``), which is
        what a voice bot needs. The text goes in one piece (sentences
        are short; the bot sends one at a time).
        """
        import torch
        request = self.request
        model = self.tts.model
        past, state = None, (None, None, None)
        with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                    dtype=torch.bfloat16):
            for latent, last in model.sample(
                    prompt=request["prompt"], text=text,
                    spk_emb=[e.clone() for e in self.spk],
                    instruction=request["instruction"],
                    prompt_waveform=None, prompt_text=REFERENCE_TEXT,
                    max_decode_steps=1000, cfg=request["cfg"],
                    sigma=request["sigma"],
                    temperature=request["temperature"],
                    use_zero_spk_emb=False, prompt_latent=self.latent):
                audio, state, past = model.audio.decode(
                    latent, past_key_values=past, use_cache=True,
                    stream_state=state, last_chunk=bool(last))
                if audio.numel():
                    yield audio.float().cpu().reshape(-1).numpy()


def make_engine(name: str, device: str, voice: str,
                reference: str) -> Engine:
    """Return the engine ``name``."""
    if name == "piper":
        return PiperEngine(device, voice)
    if name == "kokoro":
        return KokoroEngine(device, voice)
    if name == "pocket":
        return PocketEngine(device, voice)
    if name == "qwen3":
        return Qwen3Engine(device, voice, reference)
    if name == "ming":
        return MingEngine(device, voice, reference)
    raise SystemExit(f"unknown engine {name}")


def worker(name: str, device: str, voice: str, reference: str) -> None:
    """Line protocol: ``{"id", "text", "out"}`` in, timings out."""
    protocol = sys.stdout
    sys.stdout = sys.stderr  # engines print progress: keep it off the wire

    def emit(document: Dict[str, Any]) -> None:
        protocol.write(json.dumps(document) + "\n")
        protocol.flush()

    engine = make_engine(name, device, voice, reference)
    emit({"ready": True, "engine": name, "sample_rate": engine.rate})
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        first: Optional[float] = None
        chunks = []
        try:
            for chunk in engine.stream(request["text"]):
                if first is None:
                    first = time.perf_counter() - started
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            total = time.perf_counter() - started
            seconds = write_wav(Path(request["out"]), np.concatenate(chunks),
                                engine.rate)
            emit({"id": request["id"], "path": request["out"],
                  "first_ms": round((first or total) * 1000.0, 1),
                  "total_ms": round(total * 1000.0, 1),
                  "chunks": len(chunks), "audio_s": round(seconds, 3)})
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            emit({"id": request["id"], "error": repr(error)})


# ---------------------------------------------------------- orchestrator ---

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
                capture_output=True, text=True).stdout
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


def run_engine(run: str, reference: Path, work: Path) -> Dict[str, Any]:
    """Run one ``engine@device[:voice]`` over every sentence."""
    engine, rest = run.split("@")
    device, _, voice = rest.partition(":")
    outdir = work / run.replace("@", "_").replace(":", "_")
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if engine == "openvoice":
        command = [str(VENVS[engine]),
                   str(REPO / "lab/runners/voice_clone.py"),
                   "--engine", "openvoice", "--device", device,
                   "--piper-python", str(VENVS["piper"]),
                   "--piper-model", str(PIPER_MODEL)]
    else:
        command = [str(VENVS[engine]), str(HERE), "--worker", engine,
                   "--device", device, "--voice", voice,
                   "--reference", str(reference)]
        env.pop("PYTHONPATH", None)
        env["HF_HOME"] = str(LAB / "hf")  # the engines' weights
    log = (outdir / "worker.stderr").open("w")
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=log, text=True,
                               bufsize=1, cwd="/tmp", env=env)
    peak = GpuPeak(process.pid)
    assert process.stdin is not None and process.stdout is not None
    ready = json.loads(process.stdout.readline())
    cold_ms = (time.perf_counter() - started) * 1000.0

    def ask(text: str, out: Path) -> Dict[str, Any]:
        assert process.stdin is not None and process.stdout is not None
        request = {"id": out.stem, "text": text, "out": str(out),
                   "reference": str(reference), "language": "es"}
        begin = time.perf_counter()
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        answer: Dict[str, Any] = json.loads(process.stdout.readline())
        wall = (time.perf_counter() - begin) * 1000.0
        if engine == "openvoice" and "error" not in answer:
            answer["first_ms"] = answer["total_ms"] = answer.get("synth_ms")
            answer["chunks"] = 1
        answer["wall_ms"] = round(wall, 1)
        return answer

    warm = [ask("Hola, buenos días.", outdir / "warmup.wav"),
            ask("Muy bien, gracias por venir hoy.", outdir / "warmup2.wav")]
    rows = []
    for n, sentence in enumerate(SENTENCES):
        answer = ask(sentence, outdir / f"s{n:02d}.wav")
        answer["n"] = n
        rows.append(answer)
    process.stdin.close()
    process.wait(timeout=120)
    log.close()
    return {"engine": engine, "device": device, "voice": voice,
            "ready": ready, "cold_ms": round(cold_ms, 1),
            "warmup_first_ms": [w.get("first_ms") for w in warm],
            "gpu_peak_mib": peak.stop(), "sentences": rows}


def percentile(values: Sequence[float], share: float) -> Optional[float]:
    """Linear-interpolation percentile, ``None`` for no values."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), share * 100))


def stats(values: Sequence[float], digits: int = 1) -> Dict[str, Any]:
    """Return n, p50, p90 and max (``digits`` decimals)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, digits)
    return {"n": len(values), "p50": r(percentile(values, 0.5)),
            "p90": r(percentile(values, 0.9)),
            "max": r(max(values) if values else None)}


def read_16k(path: Path) -> Any:
    """Return a 16-bit mono WAV's samples at 16 kHz, float32."""
    from scipy.signal import resample_poly
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        pcm = reader.readframes(reader.getnframes())
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if rate != 16000:
        divisor = int(np.gcd(rate, 16000))
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    return samples.astype(np.float32)


class Scorer:
    """Whisper WER, UTMOS and WavLM similarity, all on the GPU."""

    def __init__(self) -> None:
        """Load the three models."""
        import torch
        from faster_whisper import WhisperModel

        from transcriber.adapters.outbound.wavlm_voice_encoder import (
            WavLMVoiceEncoder,
        )
        self.torch = torch
        self.whisper = WhisperModel(str(WHISPER), device="cuda",
                                    compute_type="int8_float16")
        self.utmos = torch.hub.load(  # type: ignore[no-untyped-call]
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
            trust_repo=True).cuda().eval()
        self.encoder = WavLMVoiceEncoder(device="cuda")

    def wer(self, path: Path, text: str) -> Optional[float]:
        """Word error rate of Whisper on ``path`` against ``text``."""
        from transcriber.domain.comparison import normalise_text, word_error
        segments, _ = self.whisper.transcribe(str(path), language="es",
                                              beam_size=5)
        heard = " ".join(s.text for s in segments)
        return word_error(normalise_text(text), normalise_text(heard))

    def mos(self, samples: Any) -> float:
        """UTMOS of 16 kHz samples."""
        wave_ = self.torch.from_numpy(samples)[None].cuda()
        with self.torch.no_grad():
            return float(self.utmos(wave_, 16000).item())

    def vector(self, samples: Any) -> Any:
        """Unit WavLM x-vector of a whole clip."""
        from transcriber.application.ports import Audio
        audio = Audio(samples=samples, sample_rate=16000)
        return self.encoder.embed(audio, [(0.0, audio.duration)])[0]


def mean(values: Sequence[float]) -> Optional[float]:
    """Mean with four decimals, ``None`` for no values."""
    return round(float(np.mean(values)), 4) if values else None


def summarise(run: Dict[str, Any], scorer: Scorer,
              ref_vec: Any) -> Dict[str, Any]:
    """Latency, RTF, WER, UTMOS and similarity of one run."""
    ok = [r for r in run["sentences"] if "error" not in r]
    wers, moses, sims, rtfs = [], [], [], []
    for row in ok:
        path = Path(row["path"])
        samples = read_16k(path)
        if not row.get("audio_s"):  # the OpenVoice worker does not say
            row["audio_s"] = round(len(samples) / 16000.0, 3)
        row["wer"] = scorer.wer(path, SENTENCES[row["n"]])
        row["utmos"] = round(scorer.mos(samples), 3)
        if row["wer"] is not None:
            wers.append(row["wer"])
        moses.append(row["utmos"])
        if run["engine"] in CLONERS:
            sims.append(float(scorer.vector(samples) @ ref_vec))
        if row.get("audio_s"):
            rtfs.append(row["total_ms"] / 1000.0 / row["audio_s"])
    interview = [r for r in ok if r["n"] < len(INTERVIEW)]
    return {
        "first_audio_ms": stats([r["first_ms"] for r in ok]),
        "sentence_ms": stats([r["total_ms"] for r in ok]),
        "rtf": stats(rtfs, 3),
        "rtf_mean": round(float(np.mean(rtfs)), 3) if rtfs else None,
        "errors": len(run["sentences"]) - len(ok),
        "wer_mean": mean(wers),
        "sentences_with_word_errors": sum(w > 0 for w in wers),
        "utmos_mean": mean(moses),
        "utmos_interview_mean": mean([r["utmos"] for r in interview]),
        "similarity_to_reference": mean(sims),
        "audio_s_mean": mean([r["audio_s"] for r in ok if r.get("audio_s")]),
    }


def disk_mb(paths: Sequence[Path]) -> int:
    """Total size of the paths, MB (``du``)."""
    total = 0
    for path in paths:
        if path.exists():
            out = subprocess.run(["du", "-sm", str(path)],
                                 capture_output=True, text=True).stdout
            total += int(out.split()[0])
    return total


def fetch_reference(out: Path) -> None:
    """Download CML-TTS Spanish test row 217 (CC BY 4.0) to ``out``."""
    import urllib.request
    url = ("https://datasets-server.huggingface.co/rows?dataset=ylacombe/"
           f"cml-tts&config=spanish&split=test&offset={REFERENCE_ROW}"
           "&length=1")
    with urllib.request.urlopen(url) as answer:
        row = json.load(answer)["rows"][0]["row"]
    assert row["text"].strip() == REFERENCE_TEXT, row["text"]
    urllib.request.urlretrieve(row["audio"][0]["src"], str(out))
    print(f"wrote {out} (speaker {row['speaker_id']},"
          f" {row['duration']:.1f} s)")


def main() -> None:
    """Worker mode, or run the engines, score and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voice", default="")
    parser.add_argument("--reference", type=Path,
                        default=LAB / "reference_cml217.wav")
    parser.add_argument("--fetch-reference", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--runs", nargs="+", default=[
        "piper@cpu", "openvoice@cuda", "kokoro@cuda:ef_dora",
        "kokoro@cuda:em_alex", "pocket@cpu:lola", "pocket@cuda:lola",
        "qwen3@cuda", "ming@cuda"])
    parser.add_argument("--samples", type=Path,
                        default=HERE.parent / "tts_samples")
    parser.add_argument("--out", type=Path,
                        default=HERE.parent / "natural_tts_results.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.device, args.voice, str(args.reference))
        return
    if args.fetch_reference:
        fetch_reference(args.fetch_reference)
        return
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw_runs.json"
    runs = json.loads(raw.read_text()) if raw.exists() else {}
    runs = {name: run for name, run in runs.items() if name in args.runs}
    for run in args.runs:
        if run in runs:
            print(f"reusing {run} from {raw}", file=sys.stderr, flush=True)
            continue
        print(f"running {run}", file=sys.stderr, flush=True)
        runs[run] = run_engine(run, args.reference, work)
        raw.write_text(json.dumps(runs, indent=1))

    scorer = Scorer()
    ref_samples = read_16k(args.reference)
    ref_vec = scorer.vector(ref_samples)
    previous: Dict[str, Any] = {}
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
    results: Dict[str, Any] = {
        "ticket": "#214", "date": time.strftime("%Y-%m-%d"),
        "gpu": "RTX 3070 Ti 8 GB (shared with the running services)",
        "sentences": {"interview": list(INTERVIEW),
                      "bot_replies": list(BOT_REPLIES)},
        "reference": {"source": "CML-TTS es test row 217, speaker 6306,"
                                " CC BY 4.0", "text": REFERENCE_TEXT,
                      "utmos": round(scorer.mos(ref_samples), 3)},
        "wer_model": "faster-whisper large-v3-turbo int8_float16, beam 5",
        "naturalness_model": "UTMOS22 strong (SpeechMOS v1.2.0)",
        "similarity_model": "microsoft/wavlm-base-plus-sv",
        "runs": dict(previous.get("runs", {})),
    }
    args.samples.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        summary = summarise(run, scorer, ref_vec)
        results["runs"][name] = {
            "engine": run["engine"], "device": run["device"],
            "voice": run["voice"] or None,
            "sample_rate": run["ready"].get("sample_rate"),
            "cold_load_ms": run["cold_ms"],
            "warmup_first_ms": run["warmup_first_ms"],
            "gpu_peak_mib": run["gpu_peak_mib"], **summary,
            "per_sentence": [
                {k: r.get(k) for k in ("n", "first_ms", "total_ms",
                                       "chunks", "audio_s", "wer",
                                       "utmos", "error")}
                for r in run["sentences"]],
        }
        label = name.replace("@", "_").replace(":", "_")
        for row in run["sentences"]:
            if row["n"] < len(INTERVIEW) and "error" not in row:
                shutil.copy(row["path"],
                            args.samples / f"{label}_{row['n'] + 1}.wav")
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False)
                        + "\n", encoding="utf-8")
    for name, run in results["runs"].items():
        print(name, run["first_audio_ms"]["p50"], run["rtf_mean"],
              run["gpu_peak_mib"], run["wer_mean"], run["utmos_mean"],
              run["similarity_to_reference"])


if __name__ == "__main__":
    main()
