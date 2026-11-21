"""Voice-cloning worker of the voice bot (#207), one engine per process.

Run by the engine's own virtual environment (never imported by the
services; the engines are not installed in the shared ``.venv``)::

    voice_clone.py --engine chatterbox|openvoice --device cuda|cpu
                   [--models-dir DIR]
                   [--piper-python P --piper-model M.onnx]

Engines:

``chatterbox``
    Chatterbox multilingual (Resemble AI, MIT), ``language_id="es"``:
    the text is spoken directly in the reference's voice. Weights in
    ``DIR`` (default ``~/.local/share/ragputin/clone-models/
    chatterbox``). Its output carries Resemble's inaudible Perth
    watermark, as the library applies it.
``openvoice``
    OpenVoice v2 (MyShell, MIT) tone-colour converter over Piper: Piper
    (``--piper-python``, ``--piper-model``; its own GPL-3.0 process,
    never imported) speaks the text, the converter moves the timbre to
    the reference's. Weights in ``DIR/converter`` (default
    ``~/.local/share/ragputin/clone-models/openvoice``).

Protocol: one JSON object per line on the standard input, ``{"id",
"text", "reference": "/abs/ref.wav", "out": "/abs/out.wav",
"language": "es"}``; one JSON object per line on the standard output:
first ``{"ready": true, "engine", "sample_rate"}`` once the models are
loaded, then for each request ``{"id", "path", "sample_rate",
"synth_ms"}`` (the 16-bit mono WAV is written at ``out``) or ``{"id",
"error"}``. The reference's speaker conditioning is computed once per
reference path and kept. Logs go to the standard error only; the text
is never logged.
"""

import argparse
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, IO, Optional

import numpy as np

# The runners' directory holds ``transformers.py``: it must not shadow
# the real package the engines import.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]

MODELS = Path("~/.local/share/ragputin/clone-models").expanduser()
SOURCE_TEXT = (
    "Hola, soy el asistente de voz. Leo este texto con calma para que se"
    " oiga bien el timbre de mi voz. El reloj de la plaza marca las doce"
    " y cuarto, y un perro marrón cruza la calle. Uno, dos, tres, cuatro,"
    " cinco, seis, siete, ocho, nueve y diez. Gracias por escuchar."
)
"""What Piper reads once so OpenVoice knows the source voice."""
WROTE = re.compile(r"Wrote (.+\.wav)\s*$")


def log(message: str) -> None:
    """Write one line to the standard error."""
    print(f"voice_clone: {message}", file=sys.stderr, flush=True)


def write_wav(path: Path, audio: Any, rate: int) -> None:
    """Write float audio in [-1, 1] as a 16-bit mono WAV file."""
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1),
                      -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(pcm)


class Chatterbox:
    """Chatterbox multilingual, conditionals cached per reference."""

    def __init__(self, models: Path, device: str) -> None:
        """Load the model and speak once with its built-in voice (warm)."""
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_local(models, device)
        self.sample_rate = int(self.model.sr)
        self._conds: Dict[str, Any] = {}
        self.model.generate("Hola, ¿qué tal?", language_id="es")

    def speak(self, text: str, reference: str, language: str) -> Any:
        """Return the float audio of ``text`` in the reference's voice."""
        conds = self._conds.get(reference)
        if conds is None:
            self.model.prepare_conditionals(reference)
            conds = self._conds[reference] = self.model.conds
        self.model.conds = conds
        wav = self.model.generate(text, language_id=language)
        return wav.squeeze(0).cpu().numpy()


class _Piper:
    """A running ``python -m piper`` reading one line per utterance."""

    def __init__(self, python: str, model: str, outdir: Path) -> None:
        """Start Piper writing WAV files to ``outdir``."""
        self.process = subprocess.Popen(
            [python, "-m", "piper", "-m", model, "-d", str(outdir)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Copy Piper's standard error lines to the queue."""
        stream: Optional[IO[str]] = self.process.stderr
        assert stream is not None
        for line in stream:
            self.lines.put(line)
        self.lines.put("")

    def say(self, text: str, timeout: float = 30.0) -> Path:
        """Synthesise one line; return the WAV file Piper wrote."""
        stdin = self.process.stdin
        assert stdin is not None
        stdin.write(text.replace("\n", " ") + "\n")
        stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line == "":
                raise RuntimeError("piper stopped")
            match = WROTE.search(line)
            if match:
                return Path(match.group(1).strip())
        raise RuntimeError("piper did not answer in time")


class OpenVoice:
    """Piper, then OpenVoice v2's tone-colour converter."""

    def __init__(self, models: Path, device: str, piper_python: str,
                 piper_model: str) -> None:
        """Load the converter, start Piper and learn Piper's voice."""
        from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

        converter = models / "converter"
        # The constructor imports wavmark (not installed) to watermark;
        # build it without, as ``enable_watermark=False`` would mean.
        self.converter = ToneColorConverter.__new__(ToneColorConverter)
        OpenVoiceBaseClass.__init__(
            self.converter, str(converter / "config.json"), device=device,
        )
        self.converter.watermark_model = None
        self.converter.version = getattr(self.converter.hps, "_version_",
                                         "v1")
        self.converter.load_ckpt(str(converter / "checkpoint.pth"))
        self.sample_rate = int(self.converter.hps.data.sampling_rate)
        self._tmp = tempfile.TemporaryDirectory(prefix="voice-clone-")
        self.piper = _Piper(piper_python, piper_model, Path(self._tmp.name))
        source = self.piper.say(SOURCE_TEXT)
        self.source_se = self.converter.extract_se([str(source)])
        source.unlink()
        self._targets: Dict[str, Any] = {}

    def speak(self, text: str, reference: str, language: str) -> Any:
        """Return the float audio of ``text`` in the reference's voice."""
        target = self._targets.get(reference)
        if target is None:
            target = self._targets[reference] = self.converter.extract_se(
                [reference]
            )
        base = self.piper.say(text)
        try:
            return self.converter.convert(
                str(base), self.source_se, target, output_path=None,
            )
        finally:
            base.unlink()


PROTOCOL = sys.stdout
"""The real standard output: the libraries' prints go to stderr."""


def emit(document: Dict[str, Any]) -> None:
    """Write one JSON line on the standard output."""
    PROTOCOL.write(json.dumps(document) + "\n")
    PROTOCOL.flush()


def main() -> int:
    """Load the engine, then answer requests until the input ends."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engine", choices=("chatterbox", "openvoice"),
                        required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--piper-python")
    parser.add_argument("--piper-model")
    args = parser.parse_args()
    sys.stdout = sys.stderr
    models = (args.models_dir or MODELS / args.engine).expanduser()
    started = time.perf_counter()
    engine: Any
    if args.engine == "chatterbox":
        engine = Chatterbox(models, args.device)
    else:
        if not (args.piper_python and args.piper_model):
            parser.error("openvoice needs --piper-python and --piper-model")
        engine = OpenVoice(models, args.device, args.piper_python,
                           args.piper_model)
    log(f"{args.engine} loaded in {time.perf_counter() - started:.1f} s")
    emit({"ready": True, "engine": args.engine,
          "sample_rate": engine.sample_rate})
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            begin = time.perf_counter()
            audio = engine.speak(str(request["text"]),
                                 str(request["reference"]),
                                 str(request.get("language") or "es"))
            out = Path(request["out"])
            write_wav(out, audio, engine.sample_rate)
            emit({"id": request_id, "path": str(out),
                  "sample_rate": engine.sample_rate,
                  "synth_ms": round((time.perf_counter() - begin) * 1000, 1)})
        except Exception as exc:  # one bad request must not stop the worker
            log(f"request failed: {type(exc).__name__}")
            emit({"id": request_id,
                  "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
