"""Analysis store adapter: each session's analysis in its own directory.

``<directory>/<session id>/run.json`` is the run's record (every step
with its status, time, who ran it and counts); the step results sit
next to it (``sentences.json`` … ``summary.json``, ``scorecard.md``,
``report-<n>.html``). Files are written atomically (temporary file in
the same directory, flushed, renamed), so a reader polling the run
never sees half a file. Session ids are checked against the
contract's pattern and file names against a plain-name pattern before
any path is built, so neither can reach outside the directory.
``<directory>/.starts.json`` keeps the analyses started through the API
(:class:`JsonStartLog`), so the start limits survive a server restart.
"""

import fcntl
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
STARTS_FILE = ".starts.json"
"""The API's analysis starts, at the top of the analysis directory.

Its leading dot keeps it out of the session ids' pattern, so no session
directory can take its name (#99 review)."""
SHARED_WAIT = 0.05
"""Seconds an exclusive lock waits once when only a reader holds it."""

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}\.(json|md|html)$")


def _exclusive(handle: int) -> bool:
    """Lock ``handle`` exclusively; tell whether it was taken.

    :meth:`FileAnalysisLock.holder` takes a shared lock for an instant
    to see whether the file is held; a reader alone never makes the
    session busy: when a shared lock can be had (no exclusive holder),
    the exclusive one is tried once more after :data:`SHARED_WAIT`.
    """
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        pass
    try:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    fcntl.flock(handle, fcntl.LOCK_UN)
    time.sleep(SHARED_WAIT)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


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

    def clear(self, session_id: str) -> None:
        """Delete the run's record and results (the lock and log stay)."""
        folder = self.path_for(session_id, RUN_FILE).parent
        if not folder.is_dir():
            return
        try:
            for item in folder.iterdir():
                if item.is_file() and _NAME.fullmatch(item.name):
                    item.unlink()
        except OSError as exc:
            raise RagError(f"Cannot clear {folder}: {exc}")

    def erase(self, session_id: str) -> None:
        """Delete the session's whole analysis directory."""
        folder = self.path_for(session_id, RUN_FILE).parent
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RagError(f"Cannot delete {folder}: {exc}")

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
        if not _exclusive(handle):
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

    def held(self) -> List[str]:
        """Return the sessions some process analyses now (any process)."""
        try:
            folders = sorted(self.directory.iterdir())
        except OSError:
            return []
        return [
            f.name for f in folders
            if is_session_id(f.name) and (f / LOCK_FILE).is_file()
            and self.holder(f.name) is not None
        ]


class JsonStartLog:
    """The API's analysis starts in one small JSON file.

    Written atomically; an unreadable file counts as no start (the
    limits bound spending, they are not a record to defend).
    """

    def __init__(self, directory: str) -> None:
        """Keep the log in ``directory/.starts.json``."""
        self.path = Path(directory) / STARTS_FILE

    def load(self) -> List[Tuple[float, str]]:
        """Return the starts kept (empty when none or unreadable)."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [(float(t), str(s)) for t, s in data]
        except (OSError, ValueError, TypeError):
            return []

    def save(self, starts: Sequence[Tuple[float, str]]) -> None:
        """Replace the starts kept with ``starts``."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                prefix=f".{STARTS_FILE}.", suffix=".tmp",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump([[t, s] for t, s in starts], stream)
                os.replace(temporary, self.path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise RagError(f"Cannot write {self.path}: {exc}")


class FileSystemSessionOutputs:
    """The per-session files of the other ``skills`` commands.

    ``skills assess`` writes ``<output_dir>/<id>.assessment.json`` and
    ``.scorecard.md`` (read by ``skills profile``), ``skills tag``
    ``<id>.tags.json``, ``skills report`` ``<id>.<person>.report.*``
    and ``skills acts`` ``<transcripts_dir>/<id>.acts.json``.
    :meth:`erase` deletes exactly those names (the people's names are
    read from the assessment), never a pattern that could match
    another session.
    """

    def __init__(self, output_dir: str, transcripts_dir: str) -> None:
        """Look under both directories."""
        self.output_dir = Path(output_dir)
        self.transcripts_dir = Path(transcripts_dir)

    def erase(self, session_id: str) -> None:
        """Delete the session's files; nothing when there are none."""
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        out = self.output_dir
        names = [f"{session_id}.{s}" for s in (
            "assessment.json", "scorecard.md", "tags.json",
        )]
        try:
            data = json.loads(
                (out / f"{session_id}.assessment.json").read_text("utf-8")
            )
            people = [str(c.get("participant", ""))
                      for c in data.get("scorecards", [])]
        except (OSError, ValueError, AttributeError):
            people = []
        for person in people:
            if person and "/" not in person and person not in (".", ".."):
                names += [f"{session_id}.{person}.report.html",
                          f"{session_id}.{person}.report.md"]
        targets = [out / n for n in names]
        targets.append(self.transcripts_dir / f"{session_id}.acts.json")
        for target in targets:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RagError(f"Cannot delete {target}: {exc}")
