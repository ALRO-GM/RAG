"""Barge-in and echo: when the user's voice stops the bot (VB-3).

Plain data and functions: no I/O, time is passed in. The bot is
*speaking* while the audio it sent is still being played: the page
plays the chunks back to back, so the end of playback is estimated
here from their durations (:class:`Playback`). While it speaks:

- the page's own voice detector stops playback at once and tells the
  orchestrator (barge-in ``vad``): the fastest path;
- a user ``partial`` or ``final`` of at least ``min_words`` words also
  stops it (barge-in ``speech``), so barge-in works without the page's
  detector, a little later (the transcriber's first partial);
- unless what was heard is the bot's own voice coming back through the
  microphone (:func:`is_echo`): most of its words are words the bot
  just said. Such text never starts a turn.

The design assumes a headset (``docs/live/VOICEBOT.md``): with
loudspeakers the echo check catches the bot's words that are
transcribed, but the page's voice detector would hear the loudspeaker,
so the page only uses it when the person says they wear a headset.
"""

import re
import unicodedata
from typing import Iterable, List, Optional, Set

from orchestrator.domain.events import LiveEvent

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> List[str]:
    """Lower-case words of ``text`` without accents."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return _WORD.findall(plain)


def is_echo(heard: str, said: Iterable[str], share: float = 0.6) -> bool:
    """Whether ``heard`` is mostly the bot's own recent words.

    True when at least ``share`` of the heard words (and at least two)
    appear in what the bot ``said``.
    """
    words = _words(heard)
    if len(words) < 2:
        return False
    spoken: Set[str] = set()
    for sentence in said:
        spoken.update(_words(sentence))
    if not spoken:
        return False
    found = sum(1 for w in words if w in spoken)
    return found / len(words) >= share


class Playback:
    """When the audio sent to the page ends, estimated.

    Chunks are played one after the other from the moment the first
    one arrives; :meth:`add` pushes the end by each chunk's duration.

    Args:
        margin_s: Extra time after the estimated end still counted as
            speaking (decoding, the page's output buffer).
    """

    def __init__(self, margin_s: float = 0.3):
        """Start silent."""
        self.margin_s = margin_s
        self.until = 0.0
        self.turn: Optional[int] = None

    def add(self, turn: int, duration_ms: float, now: float) -> None:
        """Count one chunk of ``turn`` sent at ``now``."""
        start = max(now, self.until)
        self.until = start + duration_ms / 1000.0
        self.turn = turn

    def speaking(self, now: float) -> bool:
        """Whether the page is (probably) still playing."""
        return self.recent(now, self.margin_s)

    def recent(self, now: float, window_s: float) -> bool:
        """Whether audio played less than ``window_s`` seconds ago."""
        return self.until > 0 and now < self.until + window_s

    def stop(self) -> None:
        """The page stopped playing (barge-in)."""
        self.until = 0.0


class BargeInPolicy:
    """Decide whether a user event interrupts the speaking bot.

    Args:
        user_roles: Roles whose speech interrupts (the user's).
        min_words: Words a partial needs ("eh" alone does not).
    """

    def __init__(self, user_roles: Iterable[str], min_words: int = 2):
        """Keep the rules."""
        self.user_roles = set(user_roles)
        self.min_words = min_words

    def decide(
        self, event: LiveEvent, role: str, speaking: bool,
        said: Iterable[str],
    ) -> Optional[str]:
        """Return ``"speech"`` when ``event`` barges in, else ``None``."""
        if not speaking or event.type not in ("partial", "final"):
            return None
        u = event.utterance
        if u is None or role not in self.user_roles:
            return None
        if len(u.text.split()) < self.min_words:
            return None
        if is_echo(u.text, said):
            return None
        return "speech"
