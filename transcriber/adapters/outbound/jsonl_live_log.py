"""Live log adapter: one JSON line per ``live.event.v1`` event (LV-7).

Implements :class:`~transcriber.application.ports.LiveLog` under
``<data_dir>/live/<session>/``: ``events.jsonl`` gets every event as it
is emitted (flushed at once, so a crash keeps what was said until
then) and ``transcript.json`` the session's ``transcript.v1`` at the
end; ``attribution.jsonl`` the speaker attribution's numbers of a
two-party session (#249: ids, times, slots, cosines, decisions; no
text); ``session.json`` the session's small record (#264: the
start request, speakers, names, model and state, so a restart can
rebuild the transcript from the events). No audio is ever written
here: live mode keeps text only (``docs/platform/ARCHITECTURE.md``
§8). Files are private to the user (mode 600 in a 700 directory).
"""

import json
import os
import threading
from pathlib import Path
from typing import IO, Any, Dict, List, Mapping, Optional, Sequence

LIVE_DIR = "live"
EVENTS_FILE = "events.jsonl"
TRANSCRIPT_FILE = "transcript.json"
ATTRIBUTION_FILE = "attribution.jsonl"
SESSION_FILE = "session.json"
HANDED_FILE = "handed"
"""Empty file: the session's hand-over is done (#264)."""


class JsonlLiveLog:
    """The live logs of a data directory."""

    def __init__(self, data_dir: Path) -> None:
        """Keep the root; nothing is created until the first event."""
        self.root = data_dir / LIVE_DIR
        self._open: Dict[str, IO[str]] = {}
        self._lock = threading.Lock()

    def directory(self, session_id: str) -> Path:
        """Return the directory of ``session_id``'s log."""
        return self.root / session_id

    def _folder(self, session_id: str) -> Path:
        """Create the session's directory (private) and return it."""
        folder = self.directory(session_id)
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        return folder

    def append(self, session_id: str, event: Mapping[str, Any]) -> None:
        """Write ``event`` as one line and flush it."""
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            handle = self._open.get(session_id)
            if handle is None:
                path = self._folder(session_id) / EVENTS_FILE
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
                )
                handle = os.fdopen(descriptor, "a", encoding="utf-8")
                self._open[session_id] = handle
            handle.write(line + "\n")
            handle.flush()

    def save_transcript(
        self, session_id: str, document: Mapping[str, Any]
    ) -> None:
        """Write the transcript atomically (temporary file, rename)."""
        folder = self._folder(session_id)
        temporary = folder / f".{TRANSCRIPT_FILE}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=1)
        os.replace(temporary, folder / TRANSCRIPT_FILE)

    def save_attribution(
        self, session_id: str, records: Sequence[Mapping[str, Any]]
    ) -> None:
        """Write the attribution's records, one per line, atomically."""
        folder = self._folder(session_id)
        temporary = folder / f".{ATTRIBUTION_FILE}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, folder / ATTRIBUTION_FILE)

    def save_session(
        self, session_id: str, document: Mapping[str, Any]
    ) -> None:
        """Write the session's record atomically (temporary file, rename)."""
        folder = self._folder(session_id)
        temporary = folder / f".{SESSION_FILE}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=1)
        os.replace(temporary, folder / SESSION_FILE)

    def pending(self) -> List[str]:
        """Return the sessions with a record and no ``handed`` mark.

        Only names and file existence are looked at: the work does not
        grow with what was handed over long ago.
        """
        if not self.root.is_dir():
            return []
        return sorted(
            folder.name for folder in self.root.iterdir()
            if folder.is_dir() and (folder / SESSION_FILE).is_file()
            and not (folder / HANDED_FILE).exists()
        )

    def mark_handed(self, session_id: str) -> None:
        """Leave the ``handed`` mark (private, empty)."""
        path = self._folder(session_id) / HANDED_FILE
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))

    def has_transcript(self, session_id: str) -> bool:
        """Tell whether ``transcript.json`` is there (not reading it)."""
        return (self.directory(session_id) / TRANSCRIPT_FILE).is_file()

    def read_transcript(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the kept transcript, or ``None`` without one."""
        path = self.directory(session_id) / TRANSCRIPT_FILE
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{path} is not a JSON object")
        return document

    def read_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the logged events; a line cut by a crash is skipped."""
        path = self.directory(session_id) / EVENTS_FILE
        events: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events

    def read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the session's record, or ``None`` without one."""
        path = self.directory(session_id) / SESSION_FILE
        if not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{path} is not a JSON object")
        return document

    def close(self, session_id: str) -> None:
        """Close the session's events file, if open."""
        with self._lock:
            handle = self._open.pop(session_id, None)
        if handle is not None:
            handle.close()
