"""GPU lease adapter: ``$XDG_RUNTIME_DIR/gpu-lease/lease.json``.

Implements :class:`~transcriber.application.ports.GpuLease` over files.
The lease is written atomically (a temporary file in the same
directory, then ``os.replace``), renewed by a heartbeat thread every
5 s and deleted on release. A crashed holder stops renewing, so its
lease expires within 15 s; a lease also stops counting as soon as the
holder's pid is gone. Liveness is read from ``/proc``: no signal is
ever sent to another process.
"""

import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional

from pydantic import ValidationError

from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.application.ports import Clock
from transcriber.domain.errors import StateError
from transcriber.domain.gpu_lease import (
    HEARTBEAT_SECONDS, LIVE_HOLDER, AckFile, LeaseFile, Priority, new_lease,
)

LOG = logging.getLogger(__name__)

LEASE_FILE = "lease.json"
ACK_FILE = "lease.ack.json"
DIRECTORY = "gpu-lease"


def lease_directory(environ: Mapping[str, str]) -> Path:
    """Return the lease directory under ``XDG_RUNTIME_DIR``.

    Without it (no login session) a per-user directory under the
    system's temporary directory is used.
    """
    runtime = environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / DIRECTORY
    return Path(tempfile.gettempdir()) / f"{DIRECTORY}-{os.getuid()}"


def pid_alive(pid: int) -> bool:
    """Tell whether process ``pid`` exists (read from ``/proc``)."""
    return pid > 0 and Path("/proc", str(pid)).exists()


class FileSystemGpuLease:
    """The lease of one holder, kept in a directory shared by processes."""

    def __init__(
        self,
        directory: Path,
        holder: str = LIVE_HOLDER,
        priority: Priority = "live",
        clock: Optional[Clock] = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        alive: Callable[[int], bool] = pid_alive,
        pid: Optional[int] = None,
    ) -> None:
        """Set up; no file is touched until :meth:`acquire`."""
        self._dir = directory
        self._holder = holder
        self._priority: Priority = priority
        self._clock: Clock = clock or SystemClock()
        self._period = heartbeat_seconds
        self._alive = alive
        self._pid = os.getpid() if pid is None else pid
        self._mine: Optional[LeaseFile] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def path(self) -> Path:
        """Return the path of ``lease.json``."""
        return self._dir / LEASE_FILE

    def acquire(
        self, need_mib: int, starts_at: Optional[datetime] = None
    ) -> LeaseFile:
        """Write the lease and start the heartbeat.

        Raises :class:`StateError` if another process holds a valid
        lease (this process's own earlier one is simply replaced).
        """
        with self._lock:
            held = self.current()
            if held is not None and held.pid != self._pid:
                raise StateError(
                    f"The GPU lease is held by {held.holder}"
                    f" (pid {held.pid}) until {held.expires_at.isoformat()}."
                )
            self._stop_heartbeat()
            self._dir.mkdir(parents=True, exist_ok=True)
            self._remove(self._dir / ACK_FILE)
            lease = new_lease(
                self._holder, self._priority, self._pid,
                self._clock.now(), need_mib, starts_at,
            )
            self._write(lease)
            self._mine = lease
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._beat, args=(self._stop,),
                name="gpu-lease-heartbeat", daemon=True,
            )
            self._thread.start()
            return lease

    def heartbeat(self) -> None:
        """Push the expiry of this holder's lease forward once."""
        with self._lock:
            if self._mine is None:
                return
            self._mine = self._mine.renewed(self._clock.now())
            self._write(self._mine)

    def release(self) -> None:
        """Stop the heartbeat and delete the lease file if it is ours."""
        with self._lock:
            self._stop_heartbeat()
            if self._mine is not None:
                stored = self._read()
                if stored is None or stored.pid == self._pid:
                    self._remove(self.path)
                self._mine = None

    def clear(self) -> bool:
        """Delete the lease file whoever holds it; tell if one existed.

        For an operator (``transcriber gpu-lease release``); the holder,
        if alive, is not signalled and will write the file again at its
        next heartbeat.
        """
        with self._lock:
            existed = self.path.exists()
            self._remove(self.path)
            return existed

    def current(self) -> Optional[LeaseFile]:
        """Return the lease if it is valid: not expired, holder alive."""
        lease = self._read()
        if lease is None:
            return None
        if lease.is_valid(self._clock.now(), self._alive(lease.pid)):
            return lease
        return None

    def ack(self) -> Optional[AckFile]:
        """Return the ack file's content if it is well formed."""
        try:
            text = (self._dir / ACK_FILE).read_text(encoding="utf-8")
            return AckFile.model_validate_json(text)
        except (OSError, ValidationError):
            return None

    def freed(self) -> bool:
        """Tell whether the other side acknowledged the current lease."""
        lease = self.current()
        ack = self.ack()
        return lease is not None and ack is not None and ack.answers(lease)

    def _beat(self, stop: threading.Event) -> None:
        """Renew the lease every period until ``stop`` is set."""
        while not stop.wait(self._period):
            try:
                with self._lock:
                    if stop.is_set() or self._mine is None:
                        return
                    self._mine = self._mine.renewed(self._clock.now())
                    self._write(self._mine)
            except OSError:
                LOG.exception("The GPU lease could not be renewed")

    def _stop_heartbeat(self) -> None:
        """Tell the heartbeat thread to end (hold ``_lock``).

        The thread checks the flag under the lock before every write,
        so nothing is written after this returns.
        """
        self._stop.set()
        self._thread = None

    def _read(self) -> Optional[LeaseFile]:
        """Read ``lease.json``; ``None`` if absent or not a valid lease."""
        try:
            return LeaseFile.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            return None

    def _write(self, lease: LeaseFile) -> None:
        """Write the lease atomically: temporary file, then rename."""
        handle, name = tempfile.mkstemp(
            dir=self._dir, prefix=".lease-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(lease.model_dump_json(by_alias=True))
            os.replace(name, self.path)
        except BaseException:
            self._remove(Path(name))
            raise

    @staticmethod
    def _remove(path: Path) -> None:
        """Delete ``path`` if it exists."""
        try:
            path.unlink()
        except FileNotFoundError:
            pass
