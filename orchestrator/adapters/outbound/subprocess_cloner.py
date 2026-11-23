"""Voice cloning in a process of its own (``lab/runners/voice_clone.py``).

The cloning model needs PyTorch and libraries the orchestrator must not
import, so it runs in its own virtual environment under
``~/.local/share/ragputin/`` (like Piper), as ``python
lab/runners/voice_clone.py --engine <engine> --device cuda``. The
process keeps the model loaded and speaks a line protocol: one JSON
request per line on its standard input (``id``, ``text``,
``reference``, ``out``, ``language`` and, when set, the Piper controls
``length_scale``, ``noise_scale``, ``noise_w``, ``sentence_silence``
that the engine's Piper base voice speaks with), one JSON answer per
line on its standard output (first ``{"ready": true, ...}`` once loaded, then
``{"id", "path", "sample_rate", "synth_ms"}`` or ``{"id", "error"}``).
The WAV it writes is read, its file deleted, and its frames returned.
The text is never logged. One process, started on first use and again
if it dies; one sentence at a time goes through it.
"""

import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import IO, Any, Dict, List, Mapping, Optional, Tuple

from orchestrator.application.ports import CancelToken, TtsUnavailable

GRACE_S = 2.0
"""Seconds a ready worker gets to exit on its own after its input closes."""
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
        self.device = device
        self.command = command or [
            str(python.expanduser()), str(WORKER), "--engine", engine,
            "--device", device, *(extra or []),
        ]
        self.load_timeout = load_timeout
        self.timeout = timeout
        self._tmp = tempfile.TemporaryDirectory(prefix="clone-")
        self.workdir = Path(self._tmp.name)
        self.worker_tmp = self.workdir / "worker-tmp"
        """The worker's ``TMPDIR``: what it leaves is removed here."""
        self._process: Optional[subprocess.Popen[str]] = None
        self._ready = False
        self._swap = threading.Lock()
        """Guards ``_process`` and ``_aborts`` (held only for a swap)."""
        self._aborts = 0
        self._generation = 0
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._count = 0

    @property
    def gpu(self) -> bool:
        """Whether the worker runs on the GPU (it then needs the lease)."""
        return self.device != "cpu"

    @property
    def loaded(self) -> bool:
        """Whether the worker process is running (its memory is taken)."""
        with self._swap:
            process = self._process
        return process is not None and process.poll() is None

    @property
    def generation(self) -> int:
        """The current start: it grows each time a worker is started."""
        with self._swap:
            return self._generation

    def warm(
        self, reference: str, language: str,
        controls: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Start the worker and speak one word with ``reference``."""
        self.speak("Hola.", reference, language, CancelToken(), controls)

    def unload(self, generation: Optional[int] = None) -> None:
        """Stop the worker now, even while it loads or speaks.

        With ``generation`` only that start is stopped: when a newer
        worker has begun to start since the decision was taken, nothing
        happens (the check and the abort share one lock hold).

        It does not wait for the lock a sentence or the loading holds
        (only for the tiny swap lock). A worker that is ready gets its
        input closed and up to :data:`GRACE_S` seconds to exit and clean
        up its temporary directory; one that is still loading is killed
        at once, and its directory is removed from here. A load that has
        not started its process yet is aborted when it does. The thread
        waiting on the process fails at once (``TtsUnavailable``).
        """
        with self._swap:
            if generation is not None and generation != self._generation:
                return
            self._aborts += 1
            process = self._process
            graceful = self._ready
        if process is None:
            return
        self._terminate(process, graceful)

    def _terminate(
        self, process: "subprocess.Popen[str]", graceful: bool
    ) -> None:
        """Stop ``process`` (EOF first when ``graceful``), clean its tmp."""
        if graceful:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            self._kill_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        else:  # exited: whatever it started (Piper) goes with it
            self._kill_group(process)
        shutil.rmtree(self.worker_tmp, ignore_errors=True)

    @staticmethod
    def _kill_group(process: "subprocess.Popen[str]") -> None:
        """Kill the worker's whole process group (Piper children too).

        The worker is started in a session of its own, so its pid is its
        group id and nothing outside the worker is hit.
        """
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:  # pragma: no cover - already gone
                pass

    def release(self) -> None:
        """The only holder is done: a GPU worker is stopped."""
        if self.gpu:
            self.unload()

    def speak(
        self, text: str, reference: str, language: str, cancel: CancelToken,
        controls: Optional[Mapping[str, float]] = None,
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
                        "language": language, **(controls or {})})
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
        with self._swap:
            process = self._process
        if process is not None and process.poll() is None:
            return
        self._stop()
        self._lines = queue.Queue()
        self.worker_tmp.mkdir(exist_ok=True)
        with self._swap:
            self._generation += 1  # a stale unload(old) is now a no-op
            aborts = self._aborts
        env = {**os.environ, "TMPDIR": str(self.worker_tmp)}
        try:
            started = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                bufsize=1, env=env, start_new_session=True,
            )
        except OSError as exc:
            raise TtsUnavailable(f"cannot run the cloner: {exc}") from exc
        with self._swap:
            self._process = started
            self._ready = False
            cancelled = self._aborts != aborts
        if cancelled:  # unload() came while the process was being started
            self._terminate(started, False)
        threading.Thread(target=self._pump, args=(started.stdout,
                                                  self._lines),
                         daemon=True).start()
        self._read(None, self.load_timeout)
        with self._swap:
            if self._process is started:
                self._ready = True

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
        with self._swap:
            process = self._process
        if process is None or process.stdin is None:
            raise TtsUnavailable("the cloner is not running")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False)
                                + "\n")
            process.stdin.flush()
        except (OSError, ValueError) as exc:  # ValueError: input closed
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
        with self._swap:
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
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self._kill_group(process)  # also Piper children left behind
        process.wait()
        shutil.rmtree(self.worker_tmp, ignore_errors=True)
