"""When the AI interviewer may speak: turn-taking (VB-10, #213).

Plain data and functions: no I/O, time is passed in (monotonic
seconds). The support voice bot answers each final of the user; an
interview needs more patience. :class:`TurnTaking` follows one
interview and, at every :meth:`TurnTaking.tick`, says what the bot may
do now:

- **think time:** after the bot finished speaking a question, nothing
  happens for ``think_s`` seconds;
- **end of the answer:** the candidate's finals are accumulated into
  one answer, and the answer ends only after ``turn_silence_s`` of
  silence after the last speech (any partial, final or voice activity
  restarts the wait). An answer that :func:`looks_unfinished` (it ends
  in «y», «pero», «porque», «o sea», «eh», a comma…) waits up to
  ``max_wait_s`` instead;
- **re-ask:** when the candidate says nothing at all for ``max_wait_s``
  after a question, the bot re-asks it once in other words; a second
  silence as long ends the turn with an empty answer;
- **stuck hint:** when the candidate said something and then stays
  silent for ``stuck_s`` (an unfinished answer), the bot gives one
  brief hint on how to go on, at most once per slot;
- **backchannel:** during a long answer, at a natural pause (at least
  ``backchannel_pause_s``, shorter than the end of the answer) and at
  least ``backchannel_gap_s`` after the previous one, a very short
  «ajá» / «vale» / «entiendo» / «mm-hm» (``backchannel``: ``voice``
  spoken and shown, ``visual`` shown only, ``off``).

The bot never decides while it is busy (preparing or speaking a turn),
and once it decided a turn it waits for that turn: new speech of the
candidate is the bot's cue to drop the turn (:meth:`TurnTaking.heard`
says so) and keep listening.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

Backchannel = Literal["voice", "visual", "off"]
Decision = Literal["wait", "end_turn", "reask", "hint", "backchannel"]
BotTurn = Literal["question", "reask", "hint"]

BACKCHANNELS: Dict[str, Tuple[str, ...]] = {
    "es": ("Ajá.", "Vale.", "Entiendo.", "Mm-hm."),
    "en": ("Uh-huh.", "Okay.", "I see.", "Mm-hm."),
}
"""The short tokens of active listening, per language."""

_UNFINISHED_WORDS = frozenset({
    # Spanish conjunctions, connectors, fillers, articles, prepositions;
    # accents kept, so «qué», «sí», «cómo» and «más» are not here.
    "y", "e", "o", "u", "ni", "pero", "porque", "que", "aunque", "sino",
    "entonces", "pues", "como", "cuando", "si", "mientras", "donde",
    "eh", "em", "emm", "mm", "mmm", "este", "bueno", "además",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al",
    "del", "de", "en", "con", "para", "por", "a", "sin", "sobre", "mi",
    "mis", "su", "sus", "tu", "tus", "muy",
    # English.
    "and", "but", "because", "so", "or", "um", "uh", "er", "like",
    "the", "an", "to", "of", "with", "for", "in", "on", "my", "then",
})
"""Last words after which an answer is clearly not over (lower case)."""
_UNFINISHED_PAIRS = frozenset({
    ("o", "sea"), ("es", "decir"), ("por", "ejemplo"), ("lo", "que"),
    ("you", "know"), ("i", "mean"),
})
_UNFINISHED_ENDS = (",", ";", ":", "-", "…", "...")
_WORD = re.compile(r"\w+", re.UNICODE)
_EPS = 1e-6
"""Times closer than a microsecond are equal (sums of float ticks)."""


def _words(text: str) -> List[str]:
    """Lower-case words of ``text`` (accents kept, NFC)."""
    return _WORD.findall(unicodedata.normalize("NFC", text.lower()))


def looks_unfinished(text: str) -> bool:
    """Whether ``text`` is clearly an answer cut in the middle.

    A cheap local check, no model: it ends in a comma, a colon, a dash
    or an ellipsis, or its last word is a conjunction, a connector, a
    filler, an article or a preposition («y», «pero», «porque»,
    «o sea», «eh», «de»…). A question mark or a full stop after a
    content word is a finished answer.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(_UNFINISHED_ENDS):
        return True
    words = _words(stripped)
    if not words:
        return False
    if len(words) >= 2 and (words[-2], words[-1]) in _UNFINISHED_PAIRS:
        return True
    return words[-1] in _UNFINISHED_WORDS


@dataclass(frozen=True)
class TurnConfig:
    """The interviewer's patience, in seconds (``settings.v1``).

    Args:
        turn_silence_s: Silence after the last speech that ends a
            finished-looking answer.
        think_s: Silence kept after a question before any reaction.
        max_wait_s: Longest silence: before re-asking a question with
            no answer, and before ending an unfinished-looking answer.
        stuck_s: Silence after part of an answer before a hint.
        backchannel: ``voice``, ``visual`` or ``off``.
        backchannel_pause_s: Pause in an answer long enough for one.
        backchannel_gap_s: Least time between two (and from the start
            of the answer to the first).
    """

    turn_silence_s: float = 3.5
    think_s: float = 6.0
    max_wait_s: float = 10.0
    stuck_s: float = 7.0
    backchannel: Backchannel = "voice"
    backchannel_pause_s: float = 1.2
    backchannel_gap_s: float = 8.0


class TurnTaking:
    """Decide, tick by tick, whether the interviewer speaks.

    The bot tells it what happens (:meth:`bot_spoke`, :meth:`heard`,
    :meth:`activity`) and asks :meth:`tick`; nothing here reads a
    clock.

    Args:
        config: The thresholds.
    """

    def __init__(self, config: TurnConfig):
        """Nothing asked yet: no decision before the bot first spoke."""
        self.config = config
        self.asked_at: Optional[float] = None
        self.heard_at: Optional[float] = None
        self.answer_started: Optional[float] = None
        self.parts: Dict[str, str] = {}
        self.reasked = False
        self.hinted: Set[str] = set()
        self.decided = False
        self.last_backchannel: Optional[float] = None
        self.backchannel_at: Optional[float] = None

    # ---- what happens --------------------------------------------------

    def bot_spoke(self, ended_at: float, kind: BotTurn) -> None:
        """The bot finished speaking at ``ended_at`` (estimated).

        A ``question`` starts a new answer; a ``reask`` too, but it is
        the one re-ask of that question; a ``hint`` keeps the answer
        heard so far and waits for the rest.
        """
        self.asked_at = ended_at
        self.heard_at = None
        self.answer_started = None
        self.decided = False
        self.last_backchannel = None
        self.backchannel_at = None
        if kind != "hint":
            self.parts = {}
        if kind == "question":
            self.reasked = False

    def heard(self, at: float, utterance: str, text: str,
              final: bool) -> bool:
        """Take a partial or final of the candidate heard at ``at``.

        Returns whether it is speech (some words): the bot must then
        drop a turn it is preparing and keep listening.
        """
        speech = bool(_words(text))
        if final:
            if text.strip():
                self.parts[utterance] = text.strip()
            else:
                self.parts.pop(utterance, None)
        if not speech:
            return False
        self.heard_at = at
        if self.answer_started is None:
            self.answer_started = at
        self.decided = False
        return True

    def activity(self, at: float) -> None:
        """Voice activity (the page's detector): the wait restarts."""
        self.heard_at = at
        self.decided = False

    def dropped(self) -> None:
        """The turn decided was dropped: decide again when due."""
        self.decided = False

    # ---- what the bot may do -------------------------------------------

    @property
    def answer(self) -> str:
        """The candidate's finals since the question, in order."""
        return " ".join(self.parts.values())

    @property
    def last_utterance(self) -> Optional[str]:
        """The id of the answer's last final, if any."""
        return next(reversed(self.parts), None) if self.parts else None

    def tick(self, now: float, busy: bool,
             slot: Optional[str] = None) -> Decision:
        """Say what the bot does at ``now``.

        Args:
            now: Monotonic seconds.
            busy: Whether a turn is being prepared or spoken.
            slot: The slot being asked (hints are one per slot).
        """
        if busy or self.decided or self.asked_at is None:
            return "wait"
        cfg = self.config
        if now - self.asked_at + _EPS < cfg.think_s:
            return "wait"
        last = max(self.asked_at, self.heard_at or self.asked_at)
        silence = now - last + _EPS
        longest = max(cfg.max_wait_s, cfg.turn_silence_s)
        if self.answer_started is None and not self.parts:
            if silence < longest:
                return "wait"
            if not self.reasked:
                self.reasked = True
                return self._decide("reask")
            return self._decide("end_turn")
        unfinished = looks_unfinished(self.answer)
        if (unfinished and silence >= cfg.stuck_s and silence < longest
                and slot is not None and slot not in self.hinted):
            self.hinted.add(slot)
            return self._decide("hint")
        if silence >= (longest if unfinished else cfg.turn_silence_s):
            return self._decide("end_turn")
        if self._backchannel_due(now, silence):
            self.last_backchannel = now
            self.backchannel_at = self.heard_at
            return "backchannel"
        return "wait"

    def _decide(self, decision: Decision) -> Decision:
        """Remember that a turn was decided, and return it."""
        self.decided = True
        return decision

    def _backchannel_due(self, now: float, silence: float) -> bool:
        """Whether a backchannel fits this pause of the answer."""
        cfg = self.config
        if cfg.backchannel == "off" or self.answer_started is None:
            return False
        if not cfg.backchannel_pause_s <= silence < cfg.turn_silence_s:
            return False
        if self.heard_at is None or self.backchannel_at == self.heard_at:
            return False
        since = self.last_backchannel or self.answer_started
        return now - since + _EPS >= cfg.backchannel_gap_s
