"""«Compartir pantalla» (#278): the rules of a shared presentation.

One device *shares* (it is the only source of truth: its page runs the
application) and up to :data:`MAX_VIEWERS` devices *join* it with the
session's name and a 4-digit password. Nothing here does I/O: these are
the limits, the checks of what a person types, the password's hash and
the rate limiter every connection is held to. The relay that uses them
is :mod:`transcriber.application.screen_share`; the protocol is in
``docs/web/SCREEN_SHARE.md``.

The password is weak by design (10 000 values), so it is never kept in
clear (PBKDF2 with a random salt per session), every wrong attempt is
counted per client and per name (also for names that do not exist, so
an answer never tells whether a name is in use), and a session that
received :data:`MAX_WRONG_PINS` wrong passwords is ended: whoever tries
can guess right with a chance of at most 20 in 10 000 per session.
"""

import hashlib
import hmac
import re
import unicodedata
from dataclasses import dataclass

from transcriber.domain.errors import InvalidInputError

PIN_DIGITS = 4
"""Length of the password: digits only."""

PIN_HASH_ITERATIONS = 50_000
"""PBKDF2-SHA256 rounds of the password's hash (about 8 ms here)."""

NAME_MAX = 40
"""Longest name of a shared session, in characters."""

LABEL_MAX = 40
"""Longest name of a device as the others see it."""

MAX_VIEWERS = 4
"""Devices that can join one shared session (the sharing one apart)."""

MAX_SESSIONS = 8
"""Shared sessions open at the same time on one server."""

RESUME_SECONDS = 600.0
"""A device whose link dropped keeps its place (no password) this long.

Ten minutes: the tunnel's hourly reset (a few seconds), a Wi-Fi that
comes back, and the presenter's device that sleeps or switches app for a
while. When the sharing device stays away longer the shared session ends.
"""

IDLE_SECONDS = 3600.0
"""A shared session whose sharing device sent nothing for this long ends.

Its page pings every 15 s while it is connected, so a static slide never
ends the session; an abandoned one (the device gone) ends after it.
"""

MAX_LIFETIME_SECONDS = 12 * 3600.0
"""No shared session lasts longer than this, active or not."""

MAX_WRONG_PINS = 20
"""Wrong passwords a session takes before it is ended."""

HOST_MESSAGE_BYTES = 4 * 1024 * 1024
"""Largest message of the sharing device (a full page with its sounds)."""

VIEWER_MESSAGE_BYTES = 16 * 1024
"""Largest message of a joined device (one action or a pointer move)."""

HOST_RATE = 60.0
"""Messages per second the sharing device may send on average."""

HOST_BURST = 120.0
"""Messages the sharing device may send at once."""

VIEWER_RATE = 40.0
"""Messages per second a joined device may send on average."""

VIEWER_BURST = 80.0
"""Messages a joined device may send at once."""

OUTBOX_BYTES = 8 * 1024 * 1024
"""Bytes waiting for one slow device before its page is sent again whole."""

_NAME_ALLOWED = re.compile(r"^[\w .,'()&+-]+$", re.UNICODE)
_PIN = re.compile(r"^[0-9]{%d}$" % PIN_DIGITS)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def normalise_name(text: object) -> str:
    """Return the key of a session's name: trimmed, one space, casefolded.

    «Sala  Uno» and «sala uno» are the same session. Raises
    :class:`InvalidInputError` for an empty, too long or odd name.
    """
    if not isinstance(text, str):
        raise InvalidInputError("the name is text")
    name = " ".join(unicodedata.normalize("NFC", text).split())
    if not name or len(name) > NAME_MAX or not _NAME_ALLOWED.match(name):
        raise InvalidInputError(
            f"a name of 1 to {NAME_MAX} letters, digits or spaces"
        )
    return name.casefold()


def check_pin(text: object) -> str:
    """Return the password if it is exactly four digits; raise otherwise."""
    if not isinstance(text, str) or not _PIN.match(text):
        raise InvalidInputError(f"the password is {PIN_DIGITS} digits")
    return text


def clean_label(text: object, fallback: str) -> str:
    """Return a device's name as typed, made safe and short."""
    if not isinstance(text, str):
        return fallback
    label = " ".join(_CONTROL.sub(" ", text).split())[:LABEL_MAX]
    return label or fallback


def hash_pin(
    pin: str, salt: bytes, iterations: int = PIN_HASH_ITERATIONS
) -> bytes:
    """Return the password's PBKDF2-SHA256 hash with ``salt``."""
    return hashlib.pbkdf2_hmac(
        "sha256", pin.encode("ascii"), salt, iterations
    )


def pin_matches(
    pin: str, salt: bytes, digest: bytes,
    iterations: int = PIN_HASH_ITERATIONS,
) -> bool:
    """Tell, in constant time, whether ``pin`` gives ``digest``."""
    return hmac.compare_digest(hash_pin(pin, salt, iterations), digest)


@dataclass
class TokenBucket:
    """A rate limit: ``rate`` per second on average, ``burst`` at once.

    The clock is given on every call, so the bucket holds no time source.
    """

    rate: float
    burst: float
    tokens: float = -1.0
    at: float = 0.0

    def take(self, now: float) -> bool:
        """Spend one token at ``now``; tell whether there was one."""
        if self.tokens < 0:
            self.tokens, self.at = self.burst, now
        self.tokens = min(
            self.burst, self.tokens + (now - self.at) * self.rate
        )
        self.at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True
