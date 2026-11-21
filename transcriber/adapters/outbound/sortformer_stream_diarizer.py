"""Streaming diarizer adapter: Sortformer v2 in a worker process (LV-6).

NVIDIA's streaming Sortformer needs NeMo, which does not install in the
service's environment; it runs in the model lab's ``nemo`` environment
(``make lab-env FAMILY=nemo``) as a long-lived worker,
``lab/runners/sortformer_stream.py``, that speaks JSON lines on its
standard input and output (the protocol is in that script).

:class:`SortformerStreamDiarizer` starts the worker for a session
(:meth:`~SortformerStreamDiarizer.start`) and stops it at its end
(:meth:`~SortformerStreamDiarizer.close`), so its GPU memory is held
only while a mono session runs. Audio goes out through a writer
thread, so :meth:`~SortformerStreamDiarizer.feed` never waits for the
worker (it reads nothing while its model loads); a reader thread
collects the frames as they come back. The process starts from an
argument list (no shell), in its own process group, with a reduced
environment; its error stream goes to ``log``.
"""

import base64
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from transcriber.adapters.outbound.subprocess_lab_tool import (
    DEFAULT_PATH,
    PASSED_VARIABLES,
)
from transcriber.domain.errors import DiarizerError

LOG = logging.getLogger(__name__)

MODEL_NAME = "sortformer-streaming-v2"
"""The lab catalogue's id of the model."""

FRAME_SECONDS = 0.08
SLOTS = 4
STOP_SECONDS = 5.0
"""How long the worker may take to exit before it is killed."""


def worker_command(
    interpreter: Path, script: Path, device: str
) -> List[str]:
    """Return the command line of the Sortformer worker."""
    return [str(interpreter), str(script), "--device", device]


class SortformerStreamDiarizer:
    """The ``StreamDiarizer`` port over the Sortformer worker."""

    def __init__(
        self,
        command: Sequence[str],
        environ: Mapping[str, str],
        log: Optional[Path] = None,
        model_name: str = MODEL_NAME,
        frame_seconds: float = FRAME_SECONDS,
        slots: int = SLOTS,
    ) -> None:
        """Keep how to start the worker; nothing starts yet.

        ``command[0]`` is the interpreter and ``command[1]`` the
        script; both must exist when the session starts.
        """
        self._command = list(command)
        self._environ = environ
        self._log = log
        self._model_name = model_name
        self._frame = frame_seconds
        self._slots = slots
        self._process: Optional["subprocess.Popen[str]"] = None
        self._outbox: "queue.Queue[Optional[str]]" = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._changed = threading.Condition()
        self._rows: List[List[float]] = []
        self._given = 0
        self._error: Optional[str] = None
        self._closing = False
        self.ready: Optional[Dict[str, Any]] = None

    @property
    def model_name(self) -> str:
        """Return the model's id in the lab catalogue."""
        return self._model_name

    @property
    def frame_seconds(self) -> float:
        """Return the length of one frame (80 ms)."""
        return self._frame

    @property
    def slots(self) -> int:
        """Return how many speakers the model tells apart (4)."""
        return self._slots

    def environment(self) -> Dict[str, str]:
        """Return the worker's environment (a few variables only)."""
        env = {"PATH": DEFAULT_PATH, "PYTHONUNBUFFERED": "1"}
        for name in PASSED_VARIABLES:
            if name in self._environ:
                env[name] = self._environ[name]
        env.setdefault("LANG", "C.UTF-8")
        return env

    def start(self) -> None:
        """Start the worker for a new stream (it loads in the background).

        Raises :class:`DiarizerError` when its interpreter or script is
        missing or it cannot be started.
        """
        self.close()
        for path in self._command[:2]:
            if not Path(path).exists():
                raise DiarizerError(
                    f"the streaming diarizer is not installed ({path}"
                    " missing; make lab-env FAMILY=nemo)"
                )
        with self._changed:
            self._rows, self._given, self._error = [], 0, None
            self._closing, self.ready = False, None
        errors: IO[str]
        if self._log is not None:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            errors = open(self._log, "w", encoding="utf-8")
        else:
            errors = open(os.devnull, "w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self._command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=errors,
                env=self.environment(), text=True, encoding="utf-8",
                start_new_session=True,
            )
        except OSError as exc:
            raise DiarizerError(
                f"the streaming diarizer did not start: {exc}"
            ) from exc
        finally:
            errors.close()
        self._outbox = queue.Queue()
        self._threads = [
            threading.Thread(
                target=self._write, args=(self._process, self._outbox),
                name="sortformer-writer", daemon=True,
            ),
            threading.Thread(
                target=self._read, args=(self._process,),
                name="sortformer-reader", daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        self._send({"type": "start"})

    def _send(self, message: Mapping[str, Any]) -> None:
        """Queue one request line for the worker."""
        self._outbox.put(json.dumps(message, separators=(",", ":")))

    @staticmethod
    def _write(
        process: "subprocess.Popen[str]", outbox: "queue.Queue[Optional[str]]"
    ) -> None:
        """Write the queued lines to the worker (writer thread)."""
        stdin = process.stdin
        assert stdin is not None
        while True:
            line = outbox.get()
            if line is None:
                break
            try:
                stdin.write(line + "\n")
                stdin.flush()
            except (OSError, ValueError):
                break
        try:
            stdin.close()
        except (OSError, ValueError):
            pass

    def _read(self, process: "subprocess.Popen[str]") -> None:
        """Collect the worker's answers (reader thread)."""
        stdout = process.stdout
        assert stdout is not None
        for line in stdout:
            try:
                answer = json.loads(line)
            except ValueError:
                self._failed(f"unreadable answer {line[:80]!r}")
                continue
            self._take(answer)
        code = process.wait()
        with self._changed:
            closing = self._closing
        if not closing:
            self._failed(f"the diarizer exited (code {code})")

    def _take(self, answer: Mapping[str, Any]) -> None:
        """Handle one answer of the worker."""
        kind = answer.get("type")
        if kind == "ready":
            with self._changed:
                self.ready = dict(answer)
                self._changed.notify_all()
        elif kind == "frames":
            with self._changed:
                if int(answer.get("first", -1)) != len(self._rows):
                    self._error = self._error or (
                        f"frame {answer.get('first')} came out of order"
                    )
                else:
                    self._rows.extend(
                        [float(p) for p in row]
                        for row in answer.get("probs", [])
                    )
                self._changed.notify_all()
        elif kind == "error":
            self._failed(str(answer.get("message", "unknown error")))

    def _failed(self, message: str) -> None:
        """Record the first failure and wake whoever waits."""
        with self._changed:
            if self._error is None:
                self._error = message
                LOG.warning("Streaming diarizer: %s", message)
            self._changed.notify_all()

    def feed(self, samples: npt.NDArray[np.float32]) -> None:
        """Queue audio (mono, 16 kHz, in [-1, 1]) for the worker."""
        if self._process is None:
            raise DiarizerError("the streaming diarizer is not started")
        pcm = np.clip(samples, -1.0, 32767 / 32768) * 32768.0
        data = pcm.astype("<i2").tobytes()
        self._send({
            "type": "audio", "pcm": base64.b64encode(data).decode("ascii"),
        })

    def frames(
        self, until: float = 0.0, timeout: float = 0.0
    ) -> Tuple[int, List[List[float]]]:
        """Return the frames that came since the last call (see the port)."""
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._changed:
            while (
                self._error is None
                and len(self._rows) * self._frame < until
            ):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._changed.wait(left)
            if self._error is not None:
                raise DiarizerError(self._error)
            first = self._given
            rows = self._rows[first:]
            self._given = len(self._rows)
            return first, rows

    def finish(self) -> None:
        """Ask the worker for the frames of the last audio."""
        if self._process is not None:
            self._send({"type": "flush"})

    def close(self) -> None:
        """Stop the worker (asking first, then killing its group)."""
        process = self._process
        if process is None:
            return
        with self._changed:
            self._closing = True
        self._send({"type": "stop"})
        self._outbox.put(None)
        try:
            process.wait(STOP_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        for thread in self._threads:
            thread.join(STOP_SECONDS)
        self._process = None
        self._threads = []
