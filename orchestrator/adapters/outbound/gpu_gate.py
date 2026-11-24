"""The GPU gate of the natural voices: ``gpu-lease.v1`` and free memory.

The transcriber holds the live lease (``$XDG_RUNTIME_DIR/gpu-lease/
lease.json``, ``docs/platform/ARCHITECTURE.md`` §7) while a live
session runs on the GPU; the image worker stays off the card while it
is valid. A voice-bot session is such a session, so a natural voice on
the GPU runs **under that lease**: :meth:`LeaseGpuGate.check` allows
the GPU only while a valid ``live`` lease is held and, before a model
loads, ``nvidia-smi`` says enough memory is free. Nothing here takes
the lease during a session: the file holds one holder, and taking it
would push the transcriber's own session to the CPU.

Only «Escuchar ejemplo», outside any session, :meth:`takes
<LeaseGpuGate.take>` the lease when nobody holds it (holder
``ragputin-voicebot``, renewed every 5 s, deleted by :meth:`drop`), so
the image worker frees the card for those seconds. The holder's
liveness is read from ``/proc``; no process is ever signalled. The two
services share the file format, not code.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

LEASE_FILE = "lease.json"
DIRECTORY = "gpu-lease"
HOLDER = "ragputin-voicebot"
TTL_SECONDS = 15.0
HEARTBEAT_SECONDS = 5.0
WAIT_SECONDS = 20.0
"""How long :meth:`LeaseGpuGate.take` waits for the memory to be freed
(the image worker is asked to answer within 20 s)."""


def lease_directory(environ: Mapping[str, str]) -> Path:
    """Return the lease directory (under ``XDG_RUNTIME_DIR`` when set)."""
    runtime = environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / DIRECTORY
    return Path(tempfile.gettempdir()) / f"{DIRECTORY}-{os.getuid()}"


def process_alive(pid: int) -> bool:
    """Tell whether process ``pid`` exists (read from ``/proc``)."""
    return pid > 0 and Path("/proc", str(pid)).exists()


def utc_now() -> datetime:
    """Return the current time, timezone-aware."""
    return datetime.now(timezone.utc)


def nvidia_free_mib() -> Optional[int]:
    """Free memory of the first GPU (``nvidia-smi``), MiB; ``None``: none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class LeaseGpuGate:
    """The :class:`~orchestrator.application.ports.GpuGate` of the lease.

    Args:
        directory: Folder of ``lease.json``.
        free_mib: Free GPU memory, MiB (``None``: no GPU).
        now: Clock.
        alive: Whether a pid runs.
        sleep: Waits.
        wait_seconds: How long :meth:`take` waits for memory.
    """

    def __init__(
        self,
        directory: Path,
        free_mib: Callable[[], Optional[int]] = nvidia_free_mib,
        now: Callable[[], datetime] = utc_now,
        alive: Callable[[int], bool] = process_alive,
        sleep: Callable[[float], None] = time.sleep,
        wait_seconds: float = WAIT_SECONDS,
    ):
        """Keep the folder and the probes; nothing is written."""
        self.path = directory / LEASE_FILE
        self.free_mib = free_mib
        self.now = now
        self.alive = alive
        self.sleep = sleep
        self.wait_seconds = wait_seconds
        self._mine: Optional[Dict[str, Any]] = None
        self._users = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def check(self, need_mib: int) -> Optional[str]:
        """Why the GPU may not be used now (``None``: it may)."""
        lease = self._valid()
        if lease is None:
            return "no live GPU lease"
        if lease.get("priority") != "live":
            return f"the GPU lease is held by {str(lease.get('holder'))[:40]}"
        return self._memory(need_mib)

    def take(self, need_mib: int) -> Optional[str]:
        """Hold the lease if nobody does, wait for the memory; why not.

        Every call that returns is paired with one :meth:`drop` (one
        that raises counts for nothing): the lease this gate
        wrote is deleted by the last one, so two samples at once (a
        natural voice and a clone) never delete each other's lease.
        """
        with self._lock:
            self._users += 1
            try:
                lease = self._valid()
                if lease is None:
                    self._acquire(need_mib)
                elif lease.get("priority") != "live":
                    return (f"the GPU lease is held by "
                            f"{str(lease.get('holder'))[:40]}")
            except BaseException:
                # A take that raised (the lease file could not be
                # written) is not paired with a drop: undo its count.
                self._users -= 1
                raise
        waited = 0.0
        while True:
            short = self._memory(need_mib)
            if short is None or waited >= self.wait_seconds:
                return short
            self.sleep(0.5)
            waited += 0.5

    def take_alone(self, need_mib: int) -> Optional[str]:
        """:meth:`take`, but refused while anybody holds the lease, a
        live session included: checked and taken in one step, under the
        same lock (a demo render, #254)."""
        with self._lock:
            self._users += 1
            try:
                lease = self._valid()
                if lease is not None:
                    return (f"the GPU lease is held by "
                            f"{str(lease.get('holder'))[:40]}")
                self._acquire(need_mib)
            except BaseException:
                self._users -= 1
                raise
        waited = 0.0
        while True:
            short = self._memory(need_mib)
            if short is None or waited >= self.wait_seconds:
                return short
            self.sleep(0.5)
            waited += 0.5

    def drop(self) -> None:
        """Give back one :meth:`take`; the last one deletes the lease.

        Only a lease this gate wrote (and that is still ours by pid) is
        deleted; the live session's is never touched.
        """
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users > 0:
                return
            self._stop.set()
            mine, self._mine = self._mine, None
            if mine is None:
                return
            stored = self._read()
            if stored is not None and stored.get("pid") == mine["pid"]:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass

    def _memory(self, need_mib: int) -> Optional[str]:
        """Why ``need_mib`` cannot be had (``None``: it is free)."""
        if need_mib <= 0:
            return None
        free = self.free_mib()
        if free is None:
            return "no GPU found"
        if free < need_mib:
            return f"{free} MiB free on the GPU, {need_mib} needed"
        return None

    def _read(self) -> Optional[Dict[str, Any]]:
        """The lease file as JSON, or ``None``."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _valid(self) -> Optional[Dict[str, Any]]:
        """The lease if it counts: unexpired, holder alive."""
        data = self._read()
        if data is None:
            return None
        try:
            expires = datetime.fromisoformat(
                str(data["expires_at"]).replace("Z", "+00:00"))
            pid = int(data["pid"])
        except (KeyError, TypeError, ValueError):
            return None
        if expires.tzinfo is None:
            return None
        return data if expires > self.now() and self.alive(pid) else None

    def _acquire(self, need_mib: int) -> None:
        """Write this process's lease and keep it alive (hold the lock)."""
        now = self.now()
        self._mine = {
            "schema": "gpu-lease.v1", "holder": HOLDER, "priority": "live",
            "pid": os.getpid(), "acquired_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=TTL_SECONDS)).isoformat(),
            "starts_at": None, "need_mib": max(0, int(need_mib)),
        }
        self._write(self._mine)
        self._stop = threading.Event()
        threading.Thread(target=self._beat, args=(self._stop,),
                         daemon=True).start()

    def _beat(self, stop: threading.Event) -> None:
        """Push the expiry forward every few seconds until dropped."""
        while not stop.wait(HEARTBEAT_SECONDS):
            with self._lock:
                if stop.is_set() or self._mine is None:
                    return
                self._mine["expires_at"] = (
                    self.now() + timedelta(seconds=TTL_SECONDS)
                ).isoformat()
                try:
                    self._write(self._mine)
                except OSError:
                    return

    def _write(self, lease: Dict[str, Any]) -> None:
        """Write the lease atomically: temporary file, then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=self.path.parent,
                                        prefix=".lease-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(lease, stream)
            os.replace(name, self.path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
