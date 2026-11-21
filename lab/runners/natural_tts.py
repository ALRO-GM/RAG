"""Natural-voice worker of the voice bot (#214), one engine per process.

Run by the engine's own virtual environment (never imported by the
services; the engines are not installed in the shared ``.venv``;
install steps in ``docs/live/NATURAL_TTS.md`` §6)::

    natural_tts.py --engine pocket|qwen3 --device cpu|cuda [--voice lola]

Engines (the settings the study measured, ``experiments/live/
natural_tts.py``):

``pocket``
    Kyutai Pocket TTS (CC-BY-4.0), ``spanish`` model, a catalogue voice
    (``lola``), streamed as the model generates it.
``qwen3``
    Qwen3-TTS-12Hz-0.6B-Base (Apache-2.0) through ``faster-qwen3-tts``
    (CUDA graphs), cloning the request's reference clip, streamed 4
    codec steps (333 ms of audio) at a time, **repetition penalty 1.3**
    (the default 1.05 ran on past the text in 5 of 16 study sentences)
    and a token cap proportional to the words. With the reference's
    transcript (``reference_text``) it clones in-context, as measured;
    without one it uses the speaker embedding only (``xvec_only``).

Protocol: one JSON object per line on the standard input, either a
request ``{"id", "text", "language", "reference"?, "reference_text"?}``
or ``{"cancel": id}``; one JSON object per line on the standard output:
first ``{"ready": true, "engine", "sample_rate"}`` once the model is
loaded, then for each request any number of ``{"id", "pcm": <base64
16-bit mono>, "sample_rate"}`` as the audio is generated, and last
``{"id", "done": true, "first_ms", "total_ms", "audio_ms",
"cancelled"}`` or ``{"id", "error"}``. A cancelled request stops at the
next chunk. Logs go to the standard error only; the text is never
logged.
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

import numpy as np

# The runners' directory holds ``transformers.py``: it must not shadow
# the real package the engines import.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]

QWEN3_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_LANGUAGES = {"es": "Spanish", "en": "English"}
QWEN3_PENALTY = 1.3
"""Repetition penalty: 1.05 (the default) babbled on in the study."""
QWEN3_CHUNK = 4
"""Codec steps per streamed chunk (12 Hz: 333 ms of audio)."""


def log(message: str) -> None:
    """Write one line to the standard error."""
    print(f"natural_tts: {message}", file=sys.stderr, flush=True)


def pcm16(audio: Any) -> bytes:
    """Return float audio in [-1, 1] as 16-bit little-endian PCM."""
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1),
                      -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def token_cap(text: str) -> int:
    """The most codec tokens a sentence may take (12 per second).

    Two seconds plus 0.6 s per word, half as much again: the study's cap.
    """
    return int(12 * (2.0 + 0.6 * len(text.split())) * 1.5)


class Pocket:
    """Kyutai Pocket TTS, Spanish model, one catalogue voice."""

    def __init__(self, device: str, voice: str) -> None:
        """Load the model and the voice, and speak once (warm)."""
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model(language="spanish")
        if device == "cuda":
            self.model.to("cuda")
        self.state = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = int(self.model.sample_rate)
        for _ in self.stream("Hola.", {}):
            pass

    def stream(self, text: str, request: Dict[str, Any]) -> Iterator[Any]:
        """Yield the model's streamed chunks of ``text``."""
        for chunk in self.model.generate_audio_stream(self.state, text):
            yield chunk.detach().float().cpu().numpy()


class Qwen3:
    """Qwen3-TTS 0.6B Base cloning a reference voice, streamed."""

    def __init__(self, device: str, warm_reference: Optional[str],
                 warm_text: str) -> None:
        """Load the model and capture its CUDA graphs (first sentence)."""
        from faster_qwen3_tts import FasterQwen3TTS

        self.model = FasterQwen3TTS.from_pretrained(QWEN3_MODEL,
                                                    device=device)
        self.sample_rate = 24000
        if warm_reference:
            for _ in self.stream("Hola.", {"reference": warm_reference,
                                           "reference_text": warm_text}):
                pass

    def stream(self, text: str, request: Dict[str, Any]) -> Iterator[Any]:
        """Yield 333 ms chunks of ``text`` in the reference's voice."""
        reference = str(request.get("reference") or "")
        if not Path(reference).is_file():
            raise ValueError("no reference clip")
        reference_text = str(request.get("reference_text") or "")
        language = QWEN3_LANGUAGES.get(
            str(request.get("language") or "es"), "Spanish")
        for audio, rate, _ in self.model.generate_voice_clone_streaming(
                text=text, language=language, ref_audio=reference,
                ref_text=reference_text, xvec_only=not reference_text,
                chunk_size=QWEN3_CHUNK, max_new_tokens=token_cap(text),
                repetition_penalty=QWEN3_PENALTY):
            self.sample_rate = int(rate)
            yield audio


PROTOCOL = sys.stdout
"""The real standard output: the libraries' prints go to stderr."""
_WRITE = threading.Lock()


def emit(document: Dict[str, Any]) -> None:
    """Write one JSON line on the standard output."""
    with _WRITE:
        PROTOCOL.write(json.dumps(document) + "\n")
        PROTOCOL.flush()


def read_input(requests: "queue.Queue[Optional[Dict[str, Any]]]",
               cancelled: Set[str]) -> None:
    """Read the standard input: requests to the queue, cancels to the set."""
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        if "cancel" in message:
            cancelled.add(str(message["cancel"]))
        else:
            requests.put(message)
    requests.put(None)


def answer(engine: Any, request: Dict[str, Any], cancelled: Set[str]) -> None:
    """Stream one request's audio, then say it is done."""
    ident = str(request.get("id"))
    begin = time.perf_counter()
    first: Optional[float] = None
    samples = 0
    stopped = False
    for chunk in engine.stream(" ".join(str(request["text"]).split()),
                               request):
        if ident in cancelled:
            stopped = True
            break
        data = pcm16(chunk)
        if not data:
            continue
        if first is None:
            first = time.perf_counter() - begin
        samples += len(data) // 2
        emit({"id": ident, "sample_rate": engine.sample_rate,
              "pcm": base64.b64encode(data).decode("ascii")})
    cancelled.discard(ident)
    total = time.perf_counter() - begin
    emit({"id": ident, "done": True, "cancelled": stopped,
          "first_ms": round((first if first is not None else total) * 1000,
                            1),
          "total_ms": round(total * 1000, 1),
          "audio_ms": round(samples * 1000.0 / engine.sample_rate, 1)})


def main() -> int:
    """Load the engine, then answer requests until the input ends."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--engine", choices=("pocket", "qwen3"),
                        required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--voice", default="lola",
                        help="Pocket TTS catalogue voice")
    parser.add_argument("--warm-reference", default=None,
                        help="Qwen3: clip to speak once with at start")
    parser.add_argument("--warm-reference-text", default="")
    args = parser.parse_args()
    sys.stdout = sys.stderr
    started = time.perf_counter()
    engine: Any
    if args.engine == "pocket":
        engine = Pocket(args.device, args.voice)
    else:
        engine = Qwen3(args.device, args.warm_reference,
                       args.warm_reference_text)
    log(f"{args.engine} on {args.device} loaded in"
        f" {time.perf_counter() - started:.1f} s")
    emit({"ready": True, "engine": args.engine,
          "sample_rate": engine.sample_rate})
    requests: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    cancelled: Set[str] = set()
    threading.Thread(target=read_input, args=(requests, cancelled),
                     daemon=True).start()
    while True:
        request = requests.get()
        if request is None:
            return 0
        try:
            answer(engine, request, cancelled)
        except Exception as exc:  # one bad request must not stop the worker
            log(f"request failed: {type(exc).__name__}")
            emit({"id": request.get("id"),
                  "error": f"{type(exc).__name__}: {str(exc)[:120]}"})


if __name__ == "__main__":
    sys.exit(main())
