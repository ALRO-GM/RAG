"""Voice cloning in a process of its own (``lab/runners/voice_clone.py``).

The cloning model needs PyTorch and libraries the orchestrator must not
import, so it runs in its own virtual environment under
``~/.local/share/ragputin/`` (like Piper), as ``python
lab/runners/voice_clone.py --engine <engine> --device cuda``. The
process keeps the model loaded and speaks a line protocol: one JSON
request per line on its standard input (``id``, ``text``,
``reference``, ``out``, ``language``), one JSON answer per line on its
standard output (first ``{"ready": true, ...}`` once loaded, then
``{"id", "path", "sample_rate", "synth_ms"}`` or ``{"id", "error"}``).
The WAV it writes is read, its file deleted, and its frames returned.
The text is never logged. One process, started on first use and again
if it dies; one sentence at a time goes through it.
"""

import json
import queue
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple

from orchestrator.application.ports import CancelToken, TtsUnavailable

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "lab" / "runners" / "voice_clone.py"


class SubprocessCloner:
    """The :class:`~orchestrator.application.ports.VoiceCloner` of a worker.

    Args:
        python: The Python of the cloning model's own venv.
        engine: The model (``chatterbox`` or ``openvoice``).
        device: ``cuda`` or ``cpu``.
        extra: More arguments for the worker.
        load_timeout: Seconds to wait for the model to load.
        timeout: Seconds to wait for one sentence.
        command: The whole command (replaces the default; tests).
    """

    def __init__(
        self,
        python: Path,
        engine: str,
        device: str = "cuda",
        extra: Optional[List[str]] = None,
        load_timeout: float = 180.0,
        timeout: float = 30.0,
        command: Optional[List[str]] = None,
    ):
        """Keep the settings; no process is started yet."""
        self.command = command or [
            str(python.expanduser()), str(WORKER), "--engine", engine,
            "--device", device, *(extra or []),
        ]
        self.load_timeout = load_timeout
        self.timeout = timeout
        self._tmp = tempfile.TemporaryDirectory(prefix="clone-")
        self.workdir = Path(self._tmp.name)
        self._process: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._count = 0

    def warm(self, reference: str, language: str) -> None:
        """Start the worker and speak one word with ``reference``."""
        self.speak("Hola.", reference, language, CancelToken())

    def speak(
        self, text: str, reference: str, language: str, cancel: CancelToken
    ) -> Tuple[bytes, int]:
        """Say one sentence like ``reference``; return PCM and its rate."""
        if not Path(reference).is_file():
            raise TtsUnavailable("no reference clip")
        with self._lock:
            self._ensure()
            self._count += 1
            ident = f"c{self._count}"
            out = self.workdir / f"{ident}.wav"
            self._send({"id": ident, "text": " ".join(text.split()),
                        "reference": reference, "out": str(out),
                        "language": language})
            answer = self._read(ident, self.timeout)
        if "error" in answer:
            raise TtsUnavailable(f"cloner: {str(answer['error'])[:120]}")
        try:
            with wave.open(str(out), "rb") as reader:
                rate = reader.getframerate()
                frames = reader.readframes(reader.getnframes())
            out.unlink()
        except (OSError, wave.Error, EOFError) as exc:
            raise TtsUnavailable("the cloner's audio is missing") from exc
        return frames, rate

    def close(self) -> None:
        """Stop the worker and remove its directory."""
        with self._lock:
            self._stop()
        self._tmp.cleanup()

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
                bufsize=1,
            )
        except OSError as exc:
            raise TtsUnavailable(f"cannot run the cloner: {exc}") from exc
        stream = self._process.stdout
        threading.Thread(target=self._pump, args=(stream, self._lines),
                         daemon=True).start()
        self._read(None, self.load_timeout)

    @staticmethod
    def _pump(stream: Optional[IO[str]], lines: "queue.Queue[str]") -> None:
        """Copy the worker's standard output to ``lines``."""
        if stream is None:  # pragma: no cover - always a pipe
            return
        for line in stream:
            lines.put(line)
        lines.put("")

    def _send(self, message: Dict[str, Any]) -> None:
        """Write one request line."""
        process = self._process
        if process is None or process.stdin is None:
            raise TtsUnavailable("the cloner is not running")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False)
                                + "\n")
            process.stdin.flush()
        except OSError as exc:
            self._stop()
            raise TtsUnavailable("the cloner stopped") from exc

    def _read(self, ident: Optional[str], timeout: float) -> Dict[str, Any]:
        """Wait for the answer to ``ident`` (``None``: the ready line)."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self._stop()
                raise TtsUnavailable("the cloner did not answer in time")
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line == "":
                self._stop()
                raise TtsUnavailable("the cloner stopped")
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

    def _stop(self) -> None:
        """Stop the worker, if any."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
