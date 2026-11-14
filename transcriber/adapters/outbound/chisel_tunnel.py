"""Tunnel adapter: a chisel client that reaches the server over HTTPS.

``transcriber serve`` listens on 127.0.0.1; the server's Caddy reaches
it through a reverse port that this client opens in the server's
``tunnel`` container (``docs/transcriber/ARCHITECTURE.md`` §5,
``docs/DEPLOYMENT.md``). :class:`ChiselTunnel` keeps the client running
from a supervisor thread:

1. at start, a client left running by a ``serve`` that crashed (its pid
   is in ``<data_dir>/tunnel.pid``) is ended, after checking that the
   pid really is that chisel client;
2. the client ``chisel client --keepalive <n>s --max-retry-count 0
   <server> R:0.0.0.0:<remote port>:127.0.0.1:<port>`` runs with the
   credential in its ``AUTH`` environment variable (never on the command
   line, never logged). It exits when its connection drops;
3. it is reported ``connected`` once it writes chisel's ``Connected``
   line and the injected probe (``/api/health`` through the server)
   succeeds;
4. when it ends, it is started again after a delay that doubles with
   every failure up to a cap (a connection that lasted long enough
   resets it).

Arguments are passed as a list, never through a shell, and every value
in them comes from validated settings. The process runner, the probe
and the process inspection are injected: unit tests pass fakes.
"""

import logging
import os
import re
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from transcriber.application.ports import TunnelState, TunnelStatus
from transcriber.config import CREDENTIAL_PATTERN, TunnelSettings

LOG = logging.getLogger(__name__)

STDERR_LINES = 20
"""Lines of the client's error output kept to explain an exit."""

STOP_TIMEOUT = 5.0
"""Seconds a stopped child gets to end before it is killed."""

POLL_SECONDS = 0.2
"""How often a starting client is checked for chisel's ``Connected``."""

CONNECTED_MARK = "client: Connected"
"""What chisel writes once the server accepted it and its remotes."""

PID_FILE = "tunnel.pid"

PASSED_ENVIRONMENT = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY",
    "no_proxy",
)
"""Variables of this process the client also gets (besides ``AUTH``)."""

_TIMESTAMP = re.compile(r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d(?:\.\d+)? ")


class TunnelSetupError(Exception):
    """The client or its credential is missing or unsafe."""


class Process(Protocol):
    """A started child process, as the supervisor uses it."""

    pid: int

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        """Wait up to ``timeout`` seconds; return the exit code or None."""
        ...

    def terminate(self) -> None:
        """Ask the process to end; kill it if it does not."""
        ...

    def output(self) -> str:
        """Return the last lines it wrote to its error output."""
        ...


LineHandler = Callable[[str], None]
ProcessRunner = Callable[[Sequence[str], Mapping[str, str], LineHandler],
                         Process]
"""Start a process from its arguments and environment; every line of its
error output is passed to the handler."""

Probe = Callable[[], Optional[str]]
"""Check the tunnel end to end: ``None`` when it works, else why not."""

CommandLine = Callable[[int], Optional[List[str]]]
"""The arguments of a running process, or None if there is none."""


class _PopenProcess:
    """A :class:`Process` over :class:`subprocess.Popen`.

    The error output is drained by a thread so the child never blocks
    on a full pipe; every line goes to the handler and the last
    :data:`STDERR_LINES` lines are kept.
    """

    def __init__(
        self,
        args: Sequence[str],
        env: Mapping[str, str],
        on_line: LineHandler,
    ) -> None:
        """Start ``args`` in its own session (Ctrl-C does not reach it)."""
        self._popen = subprocess.Popen(
            list(args), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True, env=dict(env),
        )
        self.pid = self._popen.pid
        self._on_line = on_line
        self._lines: Deque[str] = deque(maxlen=STDERR_LINES)
        self._drain = threading.Thread(
            target=self._read, name="chisel-stderr", daemon=True
        )
        self._drain.start()

    def _read(self) -> None:
        """Pass on and keep the error output until it closes."""
        stream = self._popen.stderr
        if stream is None:  # pragma: no cover - always a pipe here
            return
        for raw in stream:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._lines.append(line)
                self._on_line(line)
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


def start_process(
    args: Sequence[str], env: Mapping[str, str], on_line: LineHandler
) -> Process:
    """Start ``args`` as a child process (the real runner)."""
    return _PopenProcess(args, env, on_line)


def chisel_path(settings: TunnelSettings) -> Path:
    """Return the client executable, with ``~`` expanded."""
    return settings.chisel.expanduser()


def client_command(settings: TunnelSettings, port: int) -> List[str]:
    """Return the client's arguments for the local API on ``port``.

    Raises :class:`ValueError` for a port outside 1-65535.
    """
    if not 1 <= port <= 65535:
        raise ValueError(f"not a port: {port}")
    return [
        str(chisel_path(settings)), "client",
        "--keepalive", f"{settings.keepalive}s",
        "--max-retry-count", "0",
        settings.url(),
        f"R:0.0.0.0:{settings.remote_port}:127.0.0.1:{port}",
    ]


def read_credential(path: Path) -> str:
    """Return the ``user:password`` kept in ``path``.

    Raises :class:`TunnelSetupError` when the file is missing, is not a
    regular file of this user, may be read or written by others, or
    does not hold a credential. The message never contains the file's
    content.
    """
    path = path.expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise TunnelSetupError(
            f"{path} does not exist: run `make tunnel-secret`."
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise TunnelSetupError(f"{path} is not a regular file.")
    if info.st_uid != os.getuid():
        raise TunnelSetupError(f"{path} belongs to another user.")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise TunnelSetupError(
            f"{path} may be read by others (mode {mode:o}): run"
            f" `chmod 600 {path}`."
        )
    credential = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(CREDENTIAL_PATTERN, credential):
        raise TunnelSetupError(
            f"{path} does not hold a user:password credential."
        )
    return credential


def check_client(settings: TunnelSettings) -> None:
    """Raise :class:`TunnelSetupError` if the client cannot be run.

    The executable must exist and be executable, and the credential
    must be readable (:func:`read_credential`).
    """
    executable = chisel_path(settings)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise TunnelSetupError(
            f"There is no chisel client at {executable}: run"
            " `make tunnel-client`."
        )
    read_credential(settings.credential_file)


def client_environment(
    credential: str, base: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Return the client's environment: a few variables and ``AUTH``."""
    source = os.environ if base is None else base
    env = {key: source[key] for key in PASSED_ENVIRONMENT if key in source}
    env["AUTH"] = credential
    return env


def http_probe(url: str, timeout: float) -> Probe:
    """Return a probe that expects ``200`` from ``GET url``."""

    def probe() -> Optional[str]:
        """Ask ``url``; return why it failed, or None."""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as answer:
                status = int(answer.status)
        except urllib.error.HTTPError as exc:
            status = exc.code
        except (OSError, ValueError) as exc:
            return f"{url}: {exc}"
        return None if status == 200 else f"{url} answered {status}"

    return probe


def command_line(pid: int) -> Optional[List[str]]:
    """Return the arguments of process ``pid``, or None if there is none.

    Reads ``/proc`` where there is one, else asks ``ps``.
    """
    proc = Path(f"/proc/{pid}/cmdline")
    if Path("/proc/self").exists():
        try:
            raw = proc.read_bytes()
        except OSError:
            return None
        parts = raw.decode("utf-8", "replace").split("\0")
        return [part for part in parts if part] or None
    done = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        capture_output=True, text=True, check=False,
    )
    text = done.stdout.strip()
    return text.split() if done.returncode == 0 and text else None


def is_our_client(arguments: Optional[List[str]], executable: Path) -> bool:
    """Tell whether ``arguments`` are those of a client we started."""
    return (
        arguments is not None and len(arguments) >= 2
        and arguments[0] == str(executable) and arguments[1] == "client"
    )


def end_leftover(
    path: Path,
    executable: Path,
    describe: CommandLine = command_line,
    send: Callable[[int, int], None] = os.kill,
    pause: Callable[[float], None] = time.sleep,
) -> Optional[int]:
    """End the client a crashed ``serve`` left running; return its pid.

    ``path`` is the pid file. The process is only signalled when its
    arguments show it is ``executable client`` (a pid may have been
    reused by anything else); it gets SIGTERM, then SIGKILL after
    :data:`STOP_TIMEOUT` seconds. The pid file is removed. Returns None
    when there was nothing of ours to end.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    path.unlink(missing_ok=True)
    if not text.isdigit() or int(text) <= 1:
        return None
    pid = int(text)
    if not is_our_client(describe(pid), executable):
        return None
    LOG.warning("Ending the tunnel client left by an earlier run (%d)", pid)
    try:
        send(pid, signal.SIGTERM)
    except ProcessLookupError:
        return pid
    waited = 0.0
    while waited < STOP_TIMEOUT:
        if not is_our_client(describe(pid), executable):
            return pid
        pause(0.1)
        waited += 0.1
    try:
        send(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return pid


def backoff(settings: TunnelSettings, failures: int) -> float:
    """Return the delay before reconnecting after ``failures`` failures."""
    delay = settings.backoff_initial * settings.backoff_factor ** max(
        failures - 1, 0
    )
    return float(min(delay, settings.backoff_max))


def exit_reason(code: Optional[int], output: str) -> str:
    """Explain how a client ended: its code and most telling line."""
    lines = [_TIMESTAMP.sub("", line) for line in output.strip().splitlines()]
    telling = [line for line in lines
               if "error" in line.lower() or "denied" in line]
    last = (telling or lines or [""])[-1]
    reason = f"chisel exited with code {code}"
    return f"{reason}: {last}" if last else reason


class ChiselTunnel:
    """Keep the reverse tunnel to the server open (a :class:`Tunnel`).

    :meth:`start` returns at once; a daemon thread ends a leftover
    client, runs the client and restarts it. :meth:`stop` ends the
    child and the thread. ``data_dir`` holds the pid file. ``clock``
    (seconds) times how long a connection lasted; ``pause(seconds)``
    waits before a reconnection and tells whether the tunnel was
    stopped meanwhile (tests pass one that records delays). ``probe``
    checks the tunnel end to end before it is reported ``connected``
    (None: chisel's ``Connected`` line is enough). ``environ`` is the
    environment the client's variables are taken from; ``describe`` and
    ``signal_process`` find and end a leftover client (tests pass fakes).
    """

    def __init__(
        self,
        settings: TunnelSettings,
        port: int,
        data_dir: Path,
        runner: ProcessRunner = start_process,
        clock: Callable[[], float] = time.monotonic,
        pause: Optional[Callable[[float], bool]] = None,
        probe: Optional[Probe] = None,
        describe: CommandLine = command_line,
        environ: Optional[Mapping[str, str]] = None,
        signal_process: Callable[[int, int], None] = os.kill,
    ) -> None:
        """Open ``R:0.0.0.0:<remote_port>`` to ``127.0.0.1:port``."""
        self._settings = settings
        self._command = client_command(settings, port)
        self._pid_file = data_dir / PID_FILE
        self._runner = runner
        self._clock = clock
        self._probe = probe
        self._describe = describe
        self._signal = signal_process
        self._environ = environ
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._pause = pause or self._stopping.wait
        self._child: Optional[Process] = None
        self._thread: Optional[threading.Thread] = None
        self._status = TunnelStatus()
        self.connections = 0
        """Clients started so far (for the CLI and tests)."""

    def start(self) -> None:
        """Start supervising the client (once)."""
        with self._lock:
            if self._thread is not None:
                return
            self._set("connecting", None)
            self._thread = threading.Thread(
                target=self._supervise, name="chisel-tunnel", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """End the client and stop reconnecting; wait for the thread."""
        with self._lock:
            self._stopping.set()
            child = self._child
            thread = self._thread
        if child is not None:
            child.terminate()
        if thread is not None:
            thread.join(STOP_TIMEOUT * 3)
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

    def _spawn(self, on_line: LineHandler) -> Optional[Process]:
        """Start the client as the current child, unless stopping.

        Raises :class:`TunnelSetupError` for an unusable credential and
        :class:`OSError` when the executable cannot be started.
        """
        credential = read_credential(self._settings.credential_file)
        env = client_environment(credential, self._environ)
        with self._lock:
            if self._stopping.is_set():
                return None
            self._child = self._runner(self._command, env, on_line)
            self._write_pid(self._child.pid)
            return self._child

    def _write_pid(self, pid: int) -> None:
        """Record the child's pid (best effort; mode 600)."""
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._pid_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"{pid}\n")
        except OSError as exc:
            LOG.warning("Cannot write %s: %s", self._pid_file, exc)

    def _connect(self) -> Optional[str]:
        """Run one client until it ends; return why it ended.

        ``None`` when it ended because the tunnel was stopped.
        """
        seen = threading.Event()

        def on_line(line: str) -> None:
            """Notice chisel's ``Connected`` line."""
            if CONNECTED_MARK in line:
                seen.set()

        try:
            child = self._spawn(on_line)
        except TunnelSetupError as exc:
            return str(exc)
        except OSError as exc:
            return f"cannot start {self._command[0]}: {exc}"
        if child is None:
            return None
        self.connections += 1
        working = False
        next_probe = 0.0
        while True:
            code = child.wait(None if working else POLL_SECONDS)
            if code is not None:
                break
            if not seen.is_set() or time.monotonic() < next_probe:
                continue
            reason = self._probe() if self._probe is not None else None
            next_probe = time.monotonic() + self._settings.probe_interval
            with self._lock:
                if self._stopping.is_set():
                    continue
                if reason is None:
                    self._set("connected", None)
                    working = True
                else:
                    self._set("connecting", f"chisel connected; {reason}")
        self._pid_file.unlink(missing_ok=True)
        if self._stopping.is_set():
            return None
        return exit_reason(code, child.output())

    def _supervise(self) -> None:
        """End a leftover client, then connect and reconnect until stopped."""
        try:
            end_leftover(
                self._pid_file, chisel_path(self._settings), self._describe,
                self._signal,
            )
        except OSError as exc:
            LOG.warning("Cannot check %s: %s", self._pid_file, exc)
        failures = 0
        while not self._stopping.is_set():
            began = self._clock()
            reason = self._connect()
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
        self._pid_file.unlink(missing_ok=True)
