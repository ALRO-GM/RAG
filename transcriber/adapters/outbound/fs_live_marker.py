"""Live marker adapter: ``$XDG_RUNTIME_DIR/ragputin/live-active.json`` (#264).

While a live session is on (starting, running, or waiting for its audio
source to come back after a lost connection, #263) the transcriber keeps
this small file: its pid, when it was written and, per session, the id,
the state and the source's state. The services script reads it (with
the GPU lease, which a session waiting for its source may not hold) and
refuses to stop or restart the services under a running interview. No
words, no audio. Written atomically; removed when no session is on; a
file whose pid is gone is stale (the process died).
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DIRECTORY = "ragputin"
MARKER_FILE = "live-active.json"
MARKER_SCHEMA = "ragputin.live-active"


def marker_path(environ: Mapping[str, str]) -> Path:
    """Return the marker's path under ``XDG_RUNTIME_DIR``.

    Without it (no login session) a per-user directory under the
    system's temporary directory is used, as for the GPU lease.
    """
    runtime = environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) / DIRECTORY if runtime else Path(
        tempfile.gettempdir()) / f"{DIRECTORY}-{os.getuid()}"
    return base / MARKER_FILE


class FileLiveMarker:
    """The marker file of this process's live sessions."""

    def __init__(self, path: Path) -> None:
        """Keep the path; nothing is written until a session starts."""
        self.path = path

    def mark(self, sessions: Sequence[Mapping[str, Any]]) -> None:
        """Say which live sessions are on; none removes the file."""
        if not sessions:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        document = {
            "schema": MARKER_SCHEMA, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sessions": [dict(s) for s in sessions],
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream)
        os.replace(temporary, self.path)
