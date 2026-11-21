"""Brute-force protection of the password login.

A password such as a PIN is weak, so the login is throttled three ways,
all counted only for *wrong* passwords:

* per client: after ``per_client_limit`` failures within ``window``
  seconds the client is refused for ``base_cooldown`` seconds, doubled
  for every further lock (up to ``max_cooldown``);
* globally: after ``global_limit`` failures within ``window`` seconds
  (all clients together) every login is refused for ``global_cooldown``
  seconds, so a botnet gains nothing from many addresses;
* while refused, the password is not even checked.

The clock is injected. Thread-safe. Nothing here sees a password.
"""

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Tuple

__all__ = ["LoginThrottle"]

MAX_CLIENTS = 10000


class LoginThrottle:
    """Decide whether a login attempt may be checked, and count failures."""

    def __init__(
        self,
        per_client_limit: int = 5,
        window: float = 900.0,
        base_cooldown: float = 60.0,
        max_cooldown: float = 3600.0,
        global_limit: int = 30,
        global_window: float = 3600.0,
        global_cooldown: float = 900.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set the limits (seconds) and the clock."""
        self._limit = per_client_limit
        self._window = window
        self._base = base_cooldown
        self._max = max_cooldown
        self._global_limit = global_limit
        self._global_window = global_window
        self._global_cooldown = global_cooldown
        self._now = now
        self._failures: Dict[str, Deque[float]] = {}
        self._locks: Dict[str, Tuple[float, int]] = {}
        self._all: Deque[float] = deque()
        self._global_until = 0.0
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> Tuple[float, bool]:
        """Return ``(seconds to wait, whether the lock is global)``.

        Zero seconds means the attempt may be checked.
        """
        with self._lock:
            now = self._now()
            if self._global_until > now:
                return self._global_until - now, True
            lock = self._locks.get(key)
            if lock is not None and lock[0] > now:
                return lock[0] - now, False
            return 0.0, False

    def failed(self, key: str) -> bool:
        """Count a wrong password of ``key``.

        Returns ``True`` when this failure started the global lock.
        """
        with self._lock:
            now = self._now()
            failures = self._failures.setdefault(key, deque())
            failures.append(now)
            while failures and failures[0] <= now - self._window:
                failures.popleft()
            if len(failures) >= self._limit:
                level = self._locks.get(key, (0.0, 0))[1]
                cooldown = min(self._base * 2 ** level, self._max)
                self._locks[key] = (now + cooldown, level + 1)
                del self._failures[key]
            self._all.append(now)
            while self._all and self._all[0] <= now - self._global_window:
                self._all.popleft()
            started = False
            if len(self._all) >= self._global_limit:
                started = self._global_until <= now
                self._global_until = now + self._global_cooldown
                self._all.clear()
            self._trim(now)
            return started

    def succeeded(self, key: str) -> None:
        """Forget the failures and locks of ``key`` after a right password."""
        with self._lock:
            self._failures.pop(key, None)
            self._locks.pop(key, None)

    def _trim(self, now: float) -> None:
        """Keep the tables bounded (hold ``_lock``)."""
        if len(self._failures) + len(self._locks) <= MAX_CLIENTS:
            return
        for key in [k for k, v in self._locks.items() if v[0] <= now]:
            del self._locks[key]
        for key in [
            k for k, v in self._failures.items()
            if not v or v[-1] <= now - self._window
        ]:
            del self._failures[key]
