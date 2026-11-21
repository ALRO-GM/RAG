"""Read the live mode's GPU lease: ``$XDG_RUNTIME_DIR/gpu-lease/lease.json``.

The transcriber holds the lease while a live session runs
(``docs/platform/CONTRACTS.md`` §7). Ingestion, a batch job, only reads
it: the GPU is free when the file is absent, unreadable, expired or its
holder's process is gone (read from ``/proc``; no signal is sent). The
two services share the file format, not code: neither imports the other.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional

LEASE_FILE = "lease.json"
DIRECTORY = "gpu-lease"


def lease_directory(environ: Mapping[str, str]) -> Path:
    """Return the lease directory (under ``XDG_RUNTIME_DIR`` when set)."""
    runtime = environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / DIRECTORY
    return Path(tempfile.gettempdir()) / f"{DIRECTORY}-{os.getuid()}"


def process_alive(pid: int) -> bool:
    """Tell whether process ``pid`` exists."""
    return pid > 0 and Path("/proc", str(pid)).exists()


def utc_now() -> datetime:
    """Return the current time, timezone-aware."""
    return datetime.now(timezone.utc)


class FileGpuGate:
    """Tell whether the live mode's lease leaves the GPU free.

    Args:
        directory: Folder holding ``lease.json``.
        now: Clock, for tests.
        alive: Whether a pid is running, for tests.
    """

    def __init__(
        self,
        directory: Path,
        now: Callable[[], datetime] = utc_now,
        alive: Callable[[int], bool] = process_alive,
    ) -> None:
        """Keep the directory and the clocks."""
        self.path = directory / LEASE_FILE
        self.now = now
        self.alive = alive

    def free(self) -> bool:
        """Return ``True`` unless a valid lease is held right now."""
        return not self._held()

    def _held(self) -> bool:
        """Tell whether a lease file is valid: unexpired, holder alive."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            expires = _parse(data["expires_at"])
            pid = int(data["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if expires is None:
            return False
        return expires > self.now() and self.alive(pid)


def _parse(value: object) -> Optional[datetime]:
    """Parse an ISO timestamp with offset (``Z`` allowed); ``None`` if not."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None
