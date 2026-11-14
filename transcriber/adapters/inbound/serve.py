"""``transcriber serve``: the API on 127.0.0.1, the tunnel, the join link.

The command (``docs/transcriber/ARCHITECTURE.md`` §5):

1. makes the access code (or reads the kept one with ``--keep-code``);
2. builds the container with recovery (:func:`start_container`): this
   process runs the jobs;
3. runs the API with uvicorn on 127.0.0.1 only;
4. once the API listens, starts the tunnel (unless ``--no-tunnel``) and
   prints the join link and its QR code;
5. prints a line for every change of a job or of the tunnel until
   Ctrl-C (or SIGTERM), then stops the tunnel and the server.

The access code is written to the terminal and nowhere else: it is
never passed to a logger, and the API never logs URLs with it (the code
travels in a header and in the link's fragment).
"""

import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Callable, Dict, Optional, TextIO, Tuple

import uvicorn

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
) -> int:
    """Run the service until interrupted; return the exit code.

    ``container`` must come from ``start_container``; ``tunnel`` is
    started once the API listens (``None`` with ``--no-tunnel``). The
    exit code is 0 after Ctrl-C or SIGTERM, 1 if the API could not
    start (the port is taken).
    """
    settings = container.settings

    def write(line: str) -> None:
        """Print one line to the terminal at once."""
        out.write(line + "\n")
        out.flush()

    app = create_app(container, code)
    server = uvicorn.Server(uvicorn.Config(
        app, host=LOCAL_HOST, port=options.port, log_level="warning",
        access_log=False, lifespan="off", server_header=False,
    ))
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
        server.run()
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
        if main:
            signal.signal(signal.SIGTERM, previous)
    write("Stopped.")
    return 0 if started else 1


def dev_settings(settings: TranscriberSettings) -> TranscriberSettings:
    """Return ``settings`` serving the pages of :data:`WEB_DIR` at ``/``."""
    return settings.model_copy(update={
        "api": settings.api.model_copy(update={"static_dir": WEB_DIR}),
    })
