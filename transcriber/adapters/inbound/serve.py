"""``transcriber serve``: the API on 127.0.0.1, the tunnel, the join link.

The command (``docs/transcriber/ARCHITECTURE.md`` §5):

1. takes an exclusive lock on ``<data_dir>/serve.lock``
   (:func:`hold_serve_lock`): one service per data directory, so a
   second one never runs recovery under the first one's jobs;
2. binds and listens on the API's port on 127.0.0.1
   (:func:`bind_port`) before anything changes on disk: a taken port,
   even one another ``serve`` on another data directory has only just
   reserved, stops it here too;
3. makes the access code (or reads the kept one with ``--keep-code``);
4. builds the container with recovery (:func:`start_container`): this
   process runs the jobs;
5. runs the API with uvicorn on the bound socket;
6. once the API listens, starts the tunnel (unless ``--no-tunnel``) and
   prints the join link and its QR code;
7. prints a line for every change of a job or of the tunnel until
   Ctrl-C (or SIGTERM), then stops the tunnel and the server.

The access code is written to the terminal and nowhere else: it is
never passed to a logger, and the API never logs URLs with it (the code
travels in a header and in the link's fragment).
"""

import fcntl
import signal
import socket
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import IO, Callable, Dict, List, Optional, TextIO, Tuple

import uvicorn
from fastapi import FastAPI

from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.join import LOCAL_HOST, join_link, qr_text
from transcriber.application.ports import Tunnel
from transcriber.bootstrap import TranscriberContainer
from transcriber.config import TranscriberSettings
from transcriber.domain.models import SessionState

WEB_DIR = Path(__file__).resolve().parent / "web"
"""The static pages, served at ``/`` with ``--dev-pages``."""

POLL_SECONDS = 1.0
"""How often job and tunnel changes are looked for."""

STOP_JOBS_TIMEOUT = 5.0
"""Seconds a running job gets to finish when the service stops.

A job still running then is abandoned; the next start fails it as
interrupted (:class:`~transcriber.application.recover.RecoverSessions`).
"""

ACTIVE = (SessionState.QUEUED, SessionState.RUNNING)

LOCK_FILE = "serve.lock"


def hold_serve_lock(data_dir: Path) -> Optional[IO[str]]:
    """Take the exclusive lock of ``data_dir``'s service, or return None.

    The lock (``flock`` on ``<data_dir>/serve.lock``) lasts as long as
    the returned file stays open, that is until the process ends. None
    means another ``transcriber serve`` holds it.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = open(data_dir / LOCK_FILE, "a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def bind_port(port: int) -> socket.socket:
    """Return a socket listening on ``127.0.0.1:port`` for the API.

    The socket already listens when this returns: with ``SO_REUSEADDR``
    two sockets that are only bound may share a port on Linux, so a
    second ``serve`` (on another data directory) would get past this
    and run recovery before uvicorn failed. A listening socket keeps
    the port to itself; connections that arrive before uvicorn serves
    it wait in the kernel's queue. uvicorn takes the socket as it is
    (``loop.create_server(sock=...)`` listens again with its backlog).
    Raises :class:`OSError` when the port is taken.
    """
    bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        bound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bound.bind((LOCAL_HOST, port))
        bound.listen()
    except OSError:
        bound.close()
        raise
    return bound


Snapshot = Tuple[str, Optional[str], int, Optional[str], Optional[str]]
"""State, stage, tenths done, last note and run id of a session."""


def _time() -> str:
    """Return the wall-clock time for a progress line."""
    return datetime.now().strftime("%H:%M:%S")


class ProgressLines:
    """Turn changes of the jobs and of the tunnel into printed lines.

    :meth:`poll` compares every session with the last poll and writes
    one line per change: a new session, a new state or run, a new stage
    or 10 % more of a running job. The first poll only reports the
    sessions that are queued or running. A tunnel status that changed
    gets its own line.
    """

    def __init__(
        self,
        container: TranscriberContainer,
        tunnel: Optional[Tunnel],
        write: Callable[[str], None],
        now: Callable[[], str] = _time,
    ) -> None:
        """Watch ``container``'s jobs and ``tunnel``; print with ``write``."""
        self._container = container
        self._tunnel = tunnel
        self._write = write
        self._now = now
        self._seen: Optional[Dict[str, Snapshot]] = None
        self._tunnel_line: Optional[str] = None

    def poll(self) -> None:
        """Write a line for every change since the last poll."""
        if self._tunnel is not None:
            line = self._tunnel.status().describe()
            if line != self._tunnel_line:
                self._tunnel_line = line
                self._write(f"{self._now()}  tunnel  {line}")
        seen = self._seen or {}
        current: Dict[str, Snapshot] = {}
        for session in self._container.list_sessions().sessions:
            previous = seen.get(session.id)
            active = session.state in ACTIVE
            snapshot: Snapshot
            if (
                previous is not None and not active
                and previous[0] == session.state.value
                and previous[4] == session.run_id
            ):
                snapshot = previous  # nothing moved: no need to ask
            else:
                status = self._container.transcription_status(session.id)
                snapshot = (
                    session.state.value, status.stage,
                    int(status.fraction * 10), status.message,
                    session.run_id,
                )
            current[session.id] = snapshot
            first = self._seen is None
            if snapshot != previous and (active or not first):
                self._write(self._line(session.id, snapshot, session.failure))
        self._seen = current

    def _line(
        self, session_id: str, snapshot: Snapshot, failure: Optional[str]
    ) -> str:
        """Return the progress line of one session."""
        state, stage, tenths, message, _ = snapshot
        parts = [self._now(), session_id, state]
        if state in (SessionState.QUEUED.value, SessionState.RUNNING.value):
            if stage:
                parts.append(f"{stage} {tenths * 10}%")
        elif state == SessionState.FAILED.value and failure:
            parts.append(failure)
        if message:
            parts.append(f"({message})")
        return "  ".join(parts)


def server_config(app: FastAPI, port: int) -> uvicorn.Config:
    """Return how uvicorn runs the API: on 127.0.0.1, quiet.

    ``proxy_headers`` is off: uvicorn would otherwise replace the
    client address with the last ``X-Forwarded-For`` hop (the server's
    proxy, the same for everyone), and the throttling of wrong codes
    reads its own header instead (``api.client_ip_header``).
    """
    return uvicorn.Config(
        app, host=LOCAL_HOST, port=port, log_level="warning",
        access_log=False, lifespan="off", server_header=False,
        proxy_headers=False,
    )


@dataclass(frozen=True)
class ServeOptions:
    """What ``transcriber serve`` was asked for."""

    port: int
    tunnel: bool = True
    dev_pages: bool = False


class _Interrupted(Exception):
    """SIGTERM arrived: stop like Ctrl-C."""


def _on_sigterm(signum: int, frame: Optional[FrameType]) -> None:
    """Turn SIGTERM into a clean stop."""
    raise _Interrupted()


def serve(
    container: TranscriberContainer,
    code: str,
    options: ServeOptions,
    tunnel: Optional[Tunnel],
    out: TextIO,
    bound: Optional[socket.socket] = None,
) -> int:
    """Run the service until interrupted; return the exit code.

    ``container`` must come from ``start_container``; ``tunnel`` is
    started once the API listens (``None`` with ``--no-tunnel``).
    ``bound`` is the socket :func:`bind_port` returned (without it
    uvicorn binds the port itself). The exit code is 0 after Ctrl-C or
    SIGTERM, 1 if the API could not start (the port is taken).
    """
    settings = container.settings

    def write(line: str) -> None:
        """Print one line to the terminal at once."""
        out.write(line + "\n")
        out.flush()

    app = create_app(container, code)
    server = uvicorn.Server(server_config(app, options.port))
    progress = ProgressLines(container, tunnel, write)
    stopping = threading.Event()

    def watch() -> None:
        """Wait for the API, start the tunnel, print, then poll."""
        while not server.started:
            if stopping.wait(0.05):
                return
        if tunnel is not None:
            tunnel.start()
        progress.poll()  # the sessions as they are now: no lines for them
        link = join_link(
            code, settings.tunnel.public_host, options.port, tunnel is not None
        )
        write(f"Transcriber listening on http://{LOCAL_HOST}:{options.port}")
        write("Open this link on the phone (it holds the access code):")
        write(link)
        write(qr_text(link))
        write("Ctrl-C to stop.")
        while not stopping.wait(POLL_SECONDS):
            try:
                progress.poll()
            except Exception as exc:  # keep watching; never stop serving
                write(f"{_time()}  progress unavailable: {exc}")

    watcher = threading.Thread(target=watch, name="serve-watch", daemon=True)
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _on_sigterm) if main else None
    started = False
    try:
        watcher.start()
        sockets: Optional[List[socket.socket]] = (
            None if bound is None else [bound]
        )
        server.run(sockets=sockets)
    except (KeyboardInterrupt, _Interrupted):
        pass
    except SystemExit:
        write(f"The API could not listen on port {options.port}.")
    finally:
        started = server.started
        stopping.set()
        watcher.join(POLL_SECONDS * 2)
        if tunnel is not None:
            tunnel.stop()
        container.runner.close(STOP_JOBS_TIMEOUT)
        if bound is not None:
            bound.close()
        if main:
            signal.signal(signal.SIGTERM, previous)
    write("Stopped.")
    return 0 if started else 1


def dev_settings(settings: TranscriberSettings) -> TranscriberSettings:
    """Return ``settings`` serving the pages of :data:`WEB_DIR` at ``/``."""
    return settings.model_copy(update={
        "api": settings.api.model_copy(update={"static_dir": WEB_DIR}),
    })
