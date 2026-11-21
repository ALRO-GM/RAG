"""The trigger rules: which events ask the reasoner for a suggestion.

Pure and driven by the stream time of the events, so a replay at any
speed decides exactly as a live session does. A trigger fires when

- the event is the configured kind (``final`` or ``turn_end``),
- the speaker's role is one of the configured roles,
- the sentence (or turn) has at least ``min_words`` words, and
- at least ``debounce_ms`` of stream time passed since the last
  trigger that fired.

Whether a call is still in flight (cancel the stale one, or wait) is the
use case's decision, not this policy's.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from orchestrator.domain.config import Trigger as TriggerConfig
from orchestrator.domain.events import LiveEvent


@dataclass(frozen=True)
class Firing:
    """A trigger that fired."""

    utterance_id: str
    speaker: str
    role: str
    text: str
    stream_time: float


@dataclass(frozen=True)
class Skip:
    """An event that did not fire, with the rule that stopped it."""

    reason: str


def words(text: str) -> int:
    """Count the words of ``text``."""
    return len(text.split())


class TriggerPolicy:
    """Decide, event by event, whether to ask the reasoner.

    Args:
        config: The product's trigger section.
        roles: Speaker id → role of the conversation.
    """

    def __init__(self, config: TriggerConfig, roles: Dict[str, str]):
        """Keep the rules; no event seen yet."""
        self.config = config
        self.roles = dict(roles)
        self._last_fired: Optional[float] = None
        self._turn_speaker: Optional[str] = None
        self._turn_text: str = ""
        self._turn_last_id: str = ""

    def decide(self, event: LiveEvent) -> Optional["Firing | Skip"]:
        """Return a :class:`Firing`, a :class:`Skip`, or ``None``.

        ``None`` means the event is not one the policy looks at.
        """
        if event.type == "final" and event.utterance is not None:
            utterance = event.utterance
            self._track(utterance.speaker, utterance.text, utterance.id)
            if self.config.on != "final":
                return None
            return self._judge(
                event, utterance.id, utterance.speaker, utterance.text
            )
        if event.type == "turn_end" and self.config.on == "turn_end":
            speaker, text = self._turn_speaker, self._turn_text
            self._turn_speaker, self._turn_text = None, ""
            if speaker is None:
                return Skip("no_turn")
            return self._judge(event, self._turn_last_id, speaker, text)
        return None

    def _track(self, speaker: str, text: str, utterance_id: str) -> None:
        """Accumulate the text of the current turn."""
        if speaker != self._turn_speaker:
            self._turn_speaker, self._turn_text = speaker, ""
        self._turn_text = f"{self._turn_text} {text}".strip()
        self._turn_last_id = utterance_id

    def _judge(
        self, event: LiveEvent, utterance_id: str, speaker: str, text: str
    ) -> "Firing | Skip":
        role = self.roles.get(speaker, "unknown")
        if role not in self.config.speaker_roles:
            return Skip("role")
        if words(text) < self.config.min_words:
            return Skip("min_words")
        last = self._last_fired
        if (
            last is not None and self.config.debounce_ms
            and (event.stream_time - last) * 1000.0 < self.config.debounce_ms
        ):
            return Skip("debounce")
        self._last_fired = event.stream_time
        return Firing(utterance_id, speaker, role, text, event.stream_time)
