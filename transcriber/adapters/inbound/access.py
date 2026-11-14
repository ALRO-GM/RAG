"""Access to the HTTP API: the access code and the throttling of failures.

The API is reached from the internet through the server's proxy
(``docs/transcriber/ARCHITECTURE.md`` §5), so every ``/api`` call but
the health check carries the access code in the ``X-Access-Code``
header. The code is compared in constant time and never read from the
path or the query string (which end up in logs).

Wrong or missing codes are counted per client by :class:`FailureLimiter`:
after ``limit`` failures within ``window`` seconds the client is refused
for ``cooldown`` seconds. A right code is never refused, blocked or not:
the code is checked first and only a wrong or missing one meets the
block (so an attacker cannot lock the owner out). Behind the tunnel
every request arrives from the local chisel client (loopback), so the
client of a request whose direct peer is loopback (or a unix socket,
which has no address) is the dedicated header the server's proxy
writes (``X-Client-IP`` by default, overwriting any value the caller
sent); otherwise it is the peer (:func:`client_key`).
``X-Forwarded-For`` is never read: a caller can append to it, and
uvicorn must run with ``proxy_headers=False`` so that it does not
replace the peer with that header either.

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
CLIENT_HEADER = "X-Client-IP"
"""Header the server's proxy sets to the caller's address (default)."""

MAX_KEY_LENGTH = 64
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


def client_key(
    peer: Optional[str],
    headers: Mapping[str, str],
    header: str = CLIENT_HEADER,
) -> str:
    """Return the key the failures of this request are counted under.

    ``peer`` is the direct peer's address (``None`` for a unix socket).
    From loopback or a unix socket (the tunnel) the key is the value of
    ``header``, the address the server's proxy saw; otherwise, or
    without that header, it is the peer.
    """
    if peer is None or _is_loopback(peer):
        given = headers.get(header, "").strip()
        if given:
            return given[:MAX_KEY_LENGTH]
    return peer or UNKNOWN_CLIENT


class FailureLimiter:
    """Refuse a client for a while after too many failed codes.

    Not tied to HTTP: it counts failures under string keys with the
    clock ``now`` (seconds, monotonic by default). Thread-safe. A right
    code does not reset the count: the pages poll, so an owner sharing
    an address with an attacker would otherwise keep the attacker from
    ever being blocked; failures simply age out of the window.

    At most ``max_clients`` keys are tracked. When the table is full,
    keys that are not blocked are forgotten first (oldest first); a
    blocked key is dropped only when blocked keys alone fill the table.
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
            self._forget_old(now, key)

    def _forget_old(self, now: float, keep: str) -> None:
        """Keep the tables bounded (hold ``_lock``).

        Expired blocks and failures that left the window go first, then
        the oldest unblocked keys, and only then the blocked keys that
        end soonest. ``keep``, the key just counted, is never dropped.
        """
        if self._size() <= self._max_clients:
            return
        for key in [k for k, v in self._blocked.items() if v <= now]:
            del self._blocked[key]
        for key in [
            k for k, v in self._failures.items()
            if not v or v[-1] <= now - self.window
        ]:
            del self._failures[key]
        unblocked = [k for k in self._failures if k != keep]
        unblocked.sort(key=lambda k: self._failures[k][-1])
        for key in unblocked[:max(self._size() - self._max_clients, 0)]:
            del self._failures[key]
        blocked = sorted(
            (k for k in self._blocked if k != keep),
            key=self._blocked.__getitem__,
        )
        for key in blocked[:max(self._size() - self._max_clients, 0)]:
            del self._blocked[key]

    def _size(self) -> int:
        """Return how many keys are tracked (hold ``_lock``)."""
        return len(self._failures) + len(self._blocked)
