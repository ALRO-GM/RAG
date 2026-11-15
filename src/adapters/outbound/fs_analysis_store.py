"""Analysis store adapter: each session's analysis in its own directory.

``<directory>/<session id>/run.json`` is the run's record (every step
with its status, time, who ran it and counts); the step results sit
next to it (``sentences.json`` … ``summary.json``, ``scorecard.md``,
``report-<n>.html``). Files are written atomically (temporary file in
the same directory, flushed, renamed), so a reader polling the run
never sees half a file. Session ids are checked against the
contract's pattern and file names against a plain-name pattern before
any path is built, so neither can reach outside the directory.
"""

import fcntl
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
    RagError,
)
from src.domain.skills.analysis_run import AnalysisRun
from src.domain.skills.transcript_contract import is_session_id

RUN_FILE = "run.json"
"""The run's record inside a session's directory."""
LOCK_FILE = "analysis.lock"
"""Held (``flock``) by the process analysing the session; holds its pid."""

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}\.(json|md|html)$")


class FileSystemAnalysisStore:
    """Keep each session's analysis under ``directory/<session id>/``."""

    def __init__(self, directory: str) -> None:
        """Store under ``directory`` (created on the first write)."""
        self.directory = Path(directory)

    def path_for(self, session_id: str, name: str) -> Path:
        """Return the file ``name`` of ``session_id``; refuse unsafe ones."""
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        if not _NAME.fullmatch(name):
            raise InvalidInputError(f"Invalid result name {name!r}.")
        return self.directory / session_id / name

    def load(self, session_id: str) -> Optional[AnalysisRun]:
        """Return the stored run of ``session_id``, if any."""
        try:
            text = self.read(session_id, RUN_FILE)
        except NotFoundError:
            return None
        try:
            return AnalysisRun.model_validate_json(text)
        except ValidationError as exc:
            raise CorruptedDataError(
                f"Unreadable analysis record of {session_id!r}: "
                f"{exc.error_count()} problem(s)."
            )

    def save(self, run: AnalysisRun) -> None:
        """Store ``run`` as the session's record."""
        self.write(run.session_id, RUN_FILE, run.model_dump_json(indent=1))

    def write(self, session_id: str, name: str, text: str) -> str:
        """Write one result file atomically; return its path."""
        target = self.path_for(session_id, name)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(text.encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise RagError(f"Cannot write {target}: {exc}")
        return str(target)

    def read(self, session_id: str, name: str) -> str:
        """Return one result file of ``session_id``."""
        target = self.path_for(session_id, name)
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFoundError(f"No {name} for session {session_id!r}.")
        except (OSError, UnicodeDecodeError) as exc:
            raise RagError(f"Cannot read {target}: {exc}")


class FileAnalysisLock:
    """One analysis per session, across processes: ``flock`` on a file.

    ``<directory>/<session id>/analysis.lock`` is locked for the whole
    run and holds the worker's process id. The operating system drops
    the lock when that process ends, however it ends, so a killed run
    never blocks the next one and :meth:`holder` tells a live run from
    one left ``running`` by a dead process.
    """

    def __init__(self, directory: str, pid: Optional[int] = None) -> None:
        """Lock under ``directory``; ``pid`` defaults to this process."""
        self.directory = Path(directory)
        self.pid = pid if pid is not None else os.getpid()
        self._held: Dict[str, int] = {}

    def _path(self, session_id: str) -> Path:
        """Return the lock file of ``session_id``; refuse unsafe ids."""
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        return self.directory / session_id / LOCK_FILE

    def acquire(self, session_id: str) -> Optional[int]:
        """Take the session; return this pid, or ``None`` if taken."""
        path = self._path(session_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise RagError(f"Cannot open {path}: {exc}")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(handle)
            return None
        os.ftruncate(handle, 0)
        os.write(handle, str(self.pid).encode("ascii"))
        self._held[session_id] = handle
        return self.pid

    def release(self, session_id: str) -> None:
        """Give the session back (nothing if it was not taken here)."""
        handle = self._held.pop(session_id, None)
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)

    def holder(self, session_id: str) -> Optional[int]:
        """Return the pid of the process holding the session, if any."""
        path = self._path(session_id)
        try:
            handle = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RagError(f"Cannot open {path}: {exc}")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                text = os.read(handle, 32).decode("ascii", "replace")
                return int(text) if text.strip().isdigit() else -1
            fcntl.flock(handle, fcntl.LOCK_UN)
            return None
        finally:
            os.close(handle)
