"""Natural voices in processes of their own (``lab/runners/natural_tts.py``).

Pocket TTS and Qwen3-TTS need PyTorch and libraries the orchestrator
must not import, so each runs in its own virtual environment (installed
under ``~/.local/share/ragputin/natural-tts/``, ``docs/live/
NATURAL_TTS.md`` §6) as ``python lab/runners/natural_tts.py --engine
<engine> --device <device>``. The process keeps the model loaded and
speaks a line protocol (its docstring): one JSON request per line, the
sentence's audio streamed back as base64 PCM lines, then a ``done``
line; ``{"cancel": id}`` stops a sentence. One process per engine,
device and voice, started on first use and again if it dies; one
sentence at a time goes through it. The text is never logged.

:class:`WorkerEngines` hands them out: an engine on the CPU is shared
and stays loaded; one on the GPU is started for its caller (a session,
a sample) and stopped when given back, so its memory is free again.
"""

import base64
import binascii
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import (
    IO, AbstractSet, Any, Dict, FrozenSet, Iterator, List, Optional, Tuple,
)

from orchestrator.application.ports import (
    CancelToken,
    SpeechStream,
    TtsUnavailable,
)
from orchestrator.domain.voices import EngineVoice

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "lab" / "runners" / "natural_tts.py"
DEFAULT_DIR = Path("~/.local/share/ragputin/natural-tts")
"""Where the engines' venvs (``pocket-venv``, ``qwen3-venv``), their
weights (``hf``) and Qwen3's default reference voice (``voices``) are."""
REFERENCE = "voices/cml-tts-es-6306.wav"
REFERENCE_TEXT = (
    "Los niños, vestidos de blanco y rosa, saltaban y corrían tras sus"
    " juguetes, o formaban alegres corros girando como ruedas de colores"
)
"""Qwen3's default voice: CML-TTS (CC BY 4.0) Spanish test row 217,
speaker 6306, and its transcript (the study's reference)."""
DRAIN_SECONDS = 5.0
"""How long a cancelled sentence may take to stop before the worker is."""


class WorkerSpeech:
    """The :class:`~orchestrator.application.ports.SpeechStream` of a worker.

    Args:
        command: The command that starts the worker.
        env: Its environment.
        load_timeout: Seconds to wait for the model to load.
        timeout: Seconds to wait for each piece of a sentence.
        stop_timeout: Seconds a stopped worker may take to exit before
            it is killed (it is this adapter's own child).
    """

    def __init__(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        load_timeout: float = 180.0,
        timeout: float = 30.0,
        stop_timeout: float = 3.0,
    ):
        """Keep the settings; no process is started yet."""
        self.command = command
        self.stop_timeout = stop_timeout
        self.env = env
        self.load_timeout = load_timeout
        self.timeout = timeout
        self._process: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._ready = False
        self._count = 0

    @property
    def loaded(self) -> bool:
        """Whether the worker runs with its model loaded."""
        process = self._process
        return self._ready and process is not None and process.poll() is None

    def warm(self, voice: EngineVoice) -> None:
        """Start the worker and say one word with ``voice``."""
        for _ in self.stream("Hola.", voice, "es", CancelToken()):
            pass

    def stream(
        self, text: str, voice: EngineVoice, language: str,
        cancel: CancelToken,
    ) -> Iterator[Tuple[bytes, int]]:
        """Yield the sentence's PCM pieces as the worker sends them."""
        with self._lock:
            self._ensure()
            self._count += 1
            ident = f"s{self._count}"
            request: Dict[str, Any] = {
                "id": ident, "text": " ".join(text.split()),
                "language": language,
            }
            if voice.reference:
                request["reference"] = voice.reference
                request["reference_text"] = voice.reference_text
            self._send(request)
            done = False
            try:
                while True:
                    message = self._read(ident, self.timeout, cancel)
                    if message is None:  # cancelled: stop it there
                        break
                    if "error" in message:
                        done = True
                        raise TtsUnavailable(
                            f"{voice.engine}: {str(message['error'])[:120]}"
                        )
                    if message.get("done") is True:
                        done = True
                        return
                    piece = _piece(message)
                    if piece is not None:
                        yield piece
            finally:
                if not done:
                    self._cancel(ident)

    def close(self) -> None:
        """Stop the worker."""
        with self._lock:
            self._stop()

    def _ensure(self) -> None:
        """Start the worker if it is not running, and wait until ready."""
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        self._lines = queue.Queue()
        try:
            self._process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1, env=self.env,
            )
        except OSError as exc:
            raise TtsUnavailable(f"cannot run the voice: {exc}") from exc
        threading.Thread(target=_pump, args=(self._process.stdout,
                                             self._lines),
                         daemon=True).start()
        self._read(None, self.load_timeout, CancelToken())
        self._ready = True

    def _send(self, message: Dict[str, Any]) -> None:
        """Write one line to the worker."""
        process = self._process
        if process is None or process.stdin is None:
            raise TtsUnavailable("the voice is not running")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False)
                                + "\n")
            process.stdin.flush()
        except OSError as exc:
            self._stop()
            raise TtsUnavailable("the voice stopped") from exc

    def _cancel(self, ident: str) -> None:
        """Stop sentence ``ident`` and wait for its end (or stop all)."""
        try:
            self._send({"cancel": ident})
            deadline = time.monotonic() + DRAIN_SECONDS
            while True:
                message = self._read(ident, deadline - time.monotonic(),
                                     CancelToken())
                if message is not None and (message.get("done") is True
                                            or "error" in message):
                    return
        except TtsUnavailable:
            self._stop()

    def _read(
        self, ident: Optional[str], timeout: float, cancel: CancelToken,
    ) -> Optional[Dict[str, Any]]:
        """Wait for the next line about ``ident`` (``None``: ready line).

        Returns ``None`` once ``cancel`` is set.

        Raises:
            TtsUnavailable: Nothing in time, or the worker stopped.
        """
        deadline = time.monotonic() + timeout
        while not cancel.cancelled:
            left = deadline - time.monotonic()
            if left <= 0:
                self._stop()
                raise TtsUnavailable("the voice did not answer in time")
            try:
                line = self._lines.get(timeout=min(left, 0.05))
            except queue.Empty:
                continue
            if line == "":
                self._stop()
                raise TtsUnavailable("the voice stopped")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if ident is None and message.get("ready") is True:
                return message
            if ident is not None and message.get("id") == ident:
                return message
        return None

    def _stop(self) -> None:
        """Stop the worker, if any (its own process only)."""
        process, self._process = self._process, None
        self._ready = False
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _pump(stream: Optional[IO[str]], lines: "queue.Queue[str]") -> None:
    """Copy the worker's standard output to ``lines``."""
    if stream is None:  # pragma: no cover - always a pipe
        return
    for line in stream:
        lines.put(line)
    lines.put("")


def _piece(message: Dict[str, Any]) -> Optional[Tuple[bytes, int]]:
    """Return a pcm line's audio and rate, or ``None`` if it is not one."""
    rate = message.get("sample_rate")
    data = message.get("pcm")
    if not isinstance(rate, int) or rate <= 0 or not isinstance(data, str):
        return None
    try:
        pcm = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    return (pcm, rate) if pcm else None


class WorkerEngines:
    """The :class:`~orchestrator.application.ports.SpeechEngines` on disk.

    Args:
        home: The install folder (:data:`DEFAULT_DIR`).
        python: The venv of each engine (default ``<home>/<engine>-venv``).
        worker: The worker script.
        load_timeout: Seconds to wait for a model to load.
    """

    def __init__(
        self,
        home: Path = DEFAULT_DIR,
        python: Optional[Dict[str, Path]] = None,
        worker: Path = WORKER,
        load_timeout: float = 180.0,
    ):
        """Find the installed engines; nothing is started."""
        self.home = home.expanduser()
        self.python = {
            engine: (python or {}).get(
                engine, self.home / f"{engine}-venv" / "bin" / "python"
            ).expanduser()
            for engine in ("pocket", "qwen3")
        }
        self.worker = worker
        self.load_timeout = load_timeout
        self._shared: Dict[Tuple[str, str, str], WorkerSpeech] = {}
        self._lock = threading.Lock()

    def installed(self) -> AbstractSet[str]:
        """The engines whose venv exists."""
        found: FrozenSet[str] = frozenset(
            engine for engine, python in self.python.items()
            if python.exists()
        )
        return found

    def reference(self) -> Optional[Tuple[str, str]]:
        """Qwen3's default voice (clip, transcript), if it is there."""
        clip = self.home / REFERENCE
        return (str(clip), REFERENCE_TEXT) if clip.is_file() else None

    def get(self, voice: EngineVoice) -> SpeechStream:
        """A shared engine on the CPU; a new one on the GPU."""
        if voice.device == "cuda":
            return self._new(voice)
        key = (voice.engine, voice.device, voice.voice)
        with self._lock:
            engine = self._shared.get(key)
            if engine is None:
                engine = self._shared[key] = self._new(voice)
            return engine

    def release(self, engine: SpeechStream) -> None:
        """Stop a GPU engine; a shared CPU one keeps running."""
        if isinstance(engine, WorkerSpeech) \
                and engine not in self._shared.values():
            engine.close()

    def close(self) -> None:
        """Stop the shared engines."""
        with self._lock:
            for engine in self._shared.values():
                engine.close()
            self._shared.clear()

    def _new(self, voice: EngineVoice) -> WorkerSpeech:
        """Return a worker for ``voice`` (not started)."""
        command = [str(self.python[voice.engine]), str(self.worker),
                   "--engine", voice.engine, "--device", voice.device]
        if voice.engine == "pocket":
            command += ["--voice", voice.voice or "lola"]
        elif voice.reference:
            command += ["--warm-reference", voice.reference,
                        "--warm-reference-text", voice.reference_text]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(HF_HOME=str(self.home / "hf"), HF_HUB_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false")
        return WorkerSpeech(command, env=env, load_timeout=self.load_timeout)
