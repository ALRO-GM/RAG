"""Tunnel adapter: an SSH reverse forward run by the system ``ssh`` client.

``transcriber serve`` listens on 127.0.0.1; the server's Caddy reaches
it through a unix socket that this forward creates on the server
(``docs/transcriber/ARCHITECTURE.md`` §5). :class:`SshTunnel` keeps the
forward open from a supervisor thread:

1. a clean-up connection ``ssh <options> <target> true``: the server's
   forced command for the tunnel key removes a stale socket (a socket
   left by a dropped connection would make the forward fail). Its
   failure is reported and the forward is tried anyway;
2. the forward ``ssh -N -R <socket>:127.0.0.1:<port> <options>
   <target>``, with ``ExitOnForwardFailure`` so a refused forward ends
   the process instead of leaving a useless connection;
3. when the process ends, back to 1 after a delay that doubles with
   every failure up to a cap (a connection that lasted long enough
   resets it).

Arguments are passed as a list, never through a shell. The process
runner is injected: :func:`start_process` runs the real client, unit
tests pass fakes.
"""

import logging
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Protocol, Sequence

from transcriber.application.ports import TunnelState, TunnelStatus
from transcriber.config import TunnelSettings

LOG = logging.getLogger(__name__)

STDERR_LINES = 20
"""Lines of the client's error output kept to explain an exit."""

STOP_TIMEOUT = 5.0
"""Seconds a stopped child gets to end before it is killed."""


class Process(Protocol):
    """A started child process, as the supervisor uses it."""

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """Wait up to ``timeout`` seconds; return the exit code or None."""
        ...

    def terminate(self) -> None:
        """Ask the process to end; kill it if it does not."""
        ...

    def output(self) -> str:
        """Return the last lines it wrote to its error output."""
        ...


ProcessRunner = Callable[[Sequence[str]], Process]
"""Start a process from its argument list."""


class _PopenProcess:
    """A :class:`Process` over :class:`subprocess.Popen`.

    The error output is drained by a thread so the child never blocks
    on a full pipe; the last :data:`STDERR_LINES` lines are kept.
    """

    def __init__(self, args: Sequence[str]) -> None:
        """Start ``args`` in its own session (Ctrl-C does not reach it)."""
        self._popen = subprocess.Popen(
            list(args), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._lines: Deque[str] = deque(maxlen=STDERR_LINES)
        self._drain = threading.Thread(
            target=self._read, name="ssh-stderr", daemon=True
        )
        self._drain.start()

    def _read(self) -> None:
        """Keep the last lines of the error output until it closes."""
        stream = self._popen.stderr
        if stream is None:  # pragma: no cover - always a pipe here
            return
        for raw in stream:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._lines.append(line)
        stream.close()

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """Wait up to ``timeout`` seconds; return the exit code or None."""
        try:
            code = self._popen.wait(timeout)
        except subprocess.TimeoutExpired:
            return None
        self._drain.join(1.0)
        return code

    def terminate(self) -> None:
        """Send SIGTERM, then SIGKILL after :data:`STOP_TIMEOUT` seconds."""
        if self._popen.poll() is not None:
            return
        self._popen.terminate()
        try:
            self._popen.wait(STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._popen.kill()
            self._popen.wait()

    def output(self) -> str:
        """Return the kept lines of the error output."""
        return "\n".join(self._lines)


def start_process(args: Sequence[str]) -> Process:
    """Start ``args`` as a child process (the real runner)."""
    return _PopenProcess(args)


def ssh_options(settings: TunnelSettings) -> List[str]:
    """Return the options both connections use (key, strict host check)."""
    options = [
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-i", str(settings.identity_file.expanduser()),
        "-o", "StrictHostKeyChecking=yes",
    ]
    if settings.known_hosts is not None:
        options += [
            "-o", f"UserKnownHostsFile={settings.known_hosts.expanduser()}",
        ]
    return options


def cleanup_command(settings: TunnelSettings) -> List[str]:
    """Return the clean-up connection: ``ssh <options> <target> true``."""
    return [settings.ssh, *ssh_options(settings), settings.target, "true"]


def forward_command(settings: TunnelSettings, port: int) -> List[str]:
    """Return the forward of the remote socket to ``127.0.0.1:port``."""
    return [
        settings.ssh, "-N",
        "-R", f"{settings.remote_socket}:127.0.0.1:{port}",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        *ssh_options(settings),
        settings.target,
    ]


def backoff(settings: TunnelSettings, failures: int) -> float:
    """Return the delay before reconnecting after ``failures`` failures."""
    delay = settings.backoff_initial * settings.backoff_factor ** max(
        failures - 1, 0
    )
    return float(min(delay, settings.backoff_max))


def _exit_reason(code: Optional[int], output: str) -> str:
    """Explain how a child ended: its code and last error line."""
    last = output.strip().splitlines()[-1] if output.strip() else ""
    reason = f"ssh exited with code {code}"
    return f"{reason}: {last}" if last else reason


class SshTunnel:
    """Keep the reverse forward to the server open (a :class:`Tunnel`).

    :meth:`start` returns at once; a daemon thread runs the clean-up,
    the forward and the reconnections. :meth:`stop` ends the child and
    the thread. ``clock`` (seconds) times how long a connection lasted;
    ``pause(seconds)`` waits before a reconnection and tells whether the
    tunnel was stopped meanwhile (tests pass one that records delays).
    """

    def __init__(
        self,
        settings: TunnelSettings,
        port: int,
        runner: ProcessRunner = start_process,
        clock: Callable[[], float] = time.monotonic,
        pause: Optional[Callable[[float], bool]] = None,
    ) -> None:
        """Forward the server's socket to ``127.0.0.1:port``."""
        self._settings = settings
        self._port = port
        self._runner = runner
        self._clock = clock
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._pause = pause or self._stopping.wait
        self._child: Optional[Process] = None
        self._thread: Optional[threading.Thread] = None
        self._status = TunnelStatus()
        self.connections = 0
        """Forward processes started so far (for the CLI and tests)."""

    def start(self) -> None:
        """Start supervising the forward (once)."""
        with self._lock:
            if self._thread is not None:
                return
            self._set("connecting", None)
            self._thread = threading.Thread(
                target=self._supervise, name="ssh-tunnel", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """End the forward and stop reconnecting; wait for the thread."""
        with self._lock:
            self._stopping.set()
            child = self._child
            thread = self._thread
        if child is not None:
            child.terminate()
        if thread is not None:
            thread.join(self._settings.cleanup_timeout + STOP_TIMEOUT * 2)
        with self._lock:
            self._set("stopped", None)

    def status(self) -> TunnelStatus:
        """Return where the tunnel is now."""
        with self._lock:
            return self._status

    def _set(self, state: TunnelState, reason: Optional[str]) -> None:
        """Record and log a new status (hold ``_lock``)."""
        status = TunnelStatus(state=state, reason=reason)
        if status != self._status:
            LOG.info("Tunnel %s", status.describe())
        self._status = status

    def _spawn(self, args: Sequence[str]) -> Optional[Process]:
        """Start ``args`` as the current child, unless stopping."""
        with self._lock:
            if self._stopping.is_set():
                return None
            try:
                self._child = self._runner(args)
            except OSError as exc:
                LOG.warning("Cannot start %s: %s", args[0], exc)
                self._child = None
                raise
            return self._child

    def _clean_up(self) -> bool:
        """Run the clean-up connection; tell whether to go on."""
        try:
            child = self._spawn(cleanup_command(self._settings))
        except OSError:
            return not self._stopping.is_set()
        if child is None:
            return False
        code = child.wait(self._settings.cleanup_timeout)
        if code is None:
            child.terminate()
            LOG.warning("Tunnel clean-up timed out; forwarding anyway")
        elif code != 0:
            LOG.warning(
                "Tunnel clean-up failed (%s); forwarding anyway",
                _exit_reason(code, child.output()),
            )
        return not self._stopping.is_set()

    def _forward(self) -> Optional[str]:
        """Run one forward until it ends; return why it ended.

        ``None`` when it ended because the tunnel was stopped.
        """
        try:
            child = self._spawn(forward_command(self._settings, self._port))
        except OSError as exc:
            return f"cannot start {self._settings.ssh}: {exc}"
        if child is None:
            return None
        self.connections += 1
        code = child.wait(self._settings.connect_grace)
        if code is None:
            with self._lock:
                if not self._stopping.is_set():
                    self._set("connected", None)
            code = child.wait(None)
        if self._stopping.is_set():
            return None
        return _exit_reason(code, child.output())

    def _supervise(self) -> None:
        """Clean up, forward, and reconnect with back-off until stopped."""
        failures = 0
        while not self._stopping.is_set():
            if not self._clean_up():
                break
            began = self._clock()
            reason = self._forward()
            if reason is None:
                break
            lasted = self._clock() - began
            failures = 1 if lasted >= self._settings.stable_after else (
                failures + 1
            )
            delay = backoff(self._settings, failures)
            with self._lock:
                self._set("reconnecting", reason)
            LOG.warning("Tunnel down (%s); retrying in %.0f s", reason, delay)
            if self._pause(delay) or self._stopping.is_set():
                break
