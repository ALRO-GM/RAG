"""Session locks for one process: a re-entrant lock per session id.

The HTTP requests and the job worker of the transcriber run as threads
of one process, so a :class:`threading.RLock` per session is enough to
make each use case's read-then-write atomic. A lock exists only while
some thread holds or waits for it, so ids seen once do not accumulate.
It does not protect against a second process using the same data
directory.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Tuple


class ThreadSessionLocks:
    """Per-session mutual exclusion between the threads of a process."""

    def __init__(self) -> None:
        """Start with no lock; they are made on first use."""
        self._guard = threading.Lock()
        # session id -> (lock, number of threads holding or waiting)
        self._locks: Dict[str, Tuple[threading.RLock, int]] = {}

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        """Hold the lock of ``session_id`` for the ``with`` block."""
        with self._guard:
            lock, users = self._locks.get(
                session_id, (threading.RLock(), 0)
            )
            self._locks[session_id] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._guard:
                lock, users = self._locks[session_id]
                if users == 1:
                    del self._locks[session_id]
                else:
                    self._locks[session_id] = (lock, users - 1)

    def active(self) -> int:
        """Return how many sessions have a lock held or awaited."""
        with self._guard:
            return len(self._locks)
