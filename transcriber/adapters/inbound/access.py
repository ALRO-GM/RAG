"""Access to the HTTP API: the access code and the throttling of failures.

The API is reached from the internet through the server's proxy
(``docs/transcriber/ARCHITECTURE.md`` §5), so every ``/api`` call but
the health check carries the access code in the ``X-Access-Code``
header. The code is compared in constant time and never read from the
path or the query string (which end up in logs).

Wrong or missing codes are counted per client by :class:`FailureLimiter`:
after ``limit`` failures within ``window`` seconds the client is refused
for ``cooldown`` seconds. Behind the tunnel every request arrives from
the local end of the SSH forward, so the client is the left-most
``X-Forwarded-For`` address when the direct peer is loopback (or a unix
socket, which has no address); otherwise it is the peer
(:func:`client_key`).

A browser ``<audio>`` element cannot send a header, so the audio route
also accepts a playback cookie (:data:`PLAYBACK_COOKIE`): a random
token made when the API starts (:func:`new_playback_token`), never the
access code, handed out by a request that carries the code.
"""

import hmac
import ipaddress
import secrets
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Mapping, Optional

ACCESS_HEADER = "X-Access-Code"
FORWARDED_HEADER = "X-Forwarded-For"
MIN_CODE_LENGTH = 16
"""Shortest access code accepted (``transcriber serve`` makes longer)."""

MAX_CLIENTS = 10000
"""Clients tracked at most; the oldest are forgotten beyond it."""

UNKNOWN_CLIENT = "unknown"

PLAYBACK_COOKIE = "transcriber_playback"
"""Name of the cookie that lets the audio player in."""

PLAYBACK_COOKIE_PATH = "/api"
PLAYBACK_TOKEN_BYTES = 32


def new_playback_token() -> str:
    """Return a fresh random playback token (URL-safe, 43 characters)."""
    return secrets.token_urlsafe(PLAYBACK_TOKEN_BYTES)


def check_code(expected: str, given: Optional[str]) -> bool:
    """Tell whether ``given`` is the access code, in constant time."""
    if given is None:
        return False
    return hmac.compare_digest(
        expected.encode("utf-8"), given.encode("utf-8", "surrogateescape")
    )


def _is_loopback(host: str) -> bool:
    """Tell whether ``host`` is a loopback address."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def client_key(peer: Optional[str], headers: Mapping[str, str]) -> str:
    """Return the key the failures of this request are counted under.

    ``peer`` is the direct peer's address (``None`` for a unix socket).
    From loopback or a unix socket (the tunnel) the key is the
    left-most ``X-Forwarded-For`` value, the address the proxy saw;
    otherwise, or without that header, it is the peer.
    """
    if peer is None or _is_loopback(peer):
        forwarded = headers.get(FORWARDED_HEADER, "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    return peer or UNKNOWN_CLIENT


class FailureLimiter:
    """Refuse a client for a while after too many failed codes.

    Not tied to HTTP: it counts failures under string keys with the
    clock ``now`` (seconds, monotonic by default). Thread-safe.
    """

    def __init__(
        self,
        limit: int,
        window: float,
        cooldown: float,
        now: Callable[[], float] = time.monotonic,
        max_clients: int = MAX_CLIENTS,
    ) -> None:
        """Allow ``limit`` failures per ``window``; block for ``cooldown``."""
        self.limit = limit
        self.window = window
        self.cooldown = cooldown
        self._now = now
        self._max_clients = max_clients
        self._failures: Dict[str, Deque[float]] = {}
        self._blocked: Dict[str, float] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """Return how many seconds ``key`` is still blocked (0 if not)."""
        with self._lock:
            until = self._blocked.get(key)
            if until is None:
                return 0.0
            left = until - self._now()
            if left <= 0:
                del self._blocked[key]
                return 0.0
            return left

    def failed(self, key: str) -> None:
        """Count one failure of ``key``; block it at the limit."""
        with self._lock:
            now = self._now()
            failures = self._failures.setdefault(key, deque())
            failures.append(now)
            while failures and failures[0] <= now - self.window:
                failures.popleft()
            if len(failures) >= self.limit:
                self._blocked[key] = now + self.cooldown
                del self._failures[key]
            self._forget_old(now)

    def succeeded(self, key: str) -> None:
        """Forget the failures of ``key`` after a right code."""
        with self._lock:
            self._failures.pop(key, None)

    def _forget_old(self, now: float) -> None:
        """Keep the tables bounded (hold ``_lock``)."""
        if len(self._failures) + len(self._blocked) <= self._max_clients:
            return
        for key in [k for k, v in self._blocked.items() if v <= now]:
            del self._blocked[key]
        for key in [
            k for k, v in self._failures.items()
            if not v or v[-1] <= now - self.window
        ]:
            del self._failures[key]
        while len(self._failures) > self._max_clients // 2:
            del self._failures[next(iter(self._failures))]
        while len(self._blocked) > self._max_clients // 2:
            del self._blocked[next(iter(self._blocked))]
