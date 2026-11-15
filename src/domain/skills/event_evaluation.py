"""Compare derived conversation events with a hand-written expectation.

An expected event names its type and one sentence: the question, the
proposal, the petición or compromiso, or one sentence of the decision.
A derived event matches it when the types agree, that sentence is among
the event's supporting sentences (its key preferred) and, for an action
item with an expected ``owner``, the owners agree. Each derived event
matches at most one expected event. Precision and recall per type
follow; ``None`` when there is nothing to divide by.
"""

from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.conversation_events import (
    EVENT_TYPES,
    ConversationEvent,
    EventType,
    ReplyLink,
)
from src.domain.skills.conversation_summary import rate


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedEvent(_Frozen):
    """One event a reader expects in a dialogue."""

    type: EventType
    sentence: str = Field(min_length=1)
    owner: Optional[str] = None
    note: str = ""


class ExpectedEvents(_Frozen):
    """Expected events per dialogue, and who wrote them and how."""

    note: str
    dialogues: Dict[str, List[ExpectedEvent]]


class TypeScore(_Frozen):
    """Matches of one event type."""

    type: str
    expected: int
    derived: int
    matched: int
    precision: Optional[float]
    recall: Optional[float]


class LinkScore(_Frozen):
    """How inferred links compare with known ones.

    ``known`` links were removed and inferred again: ``recovered`` got
    the same antecedent back, ``missed`` got none; ``spurious`` are
    inferred links on sentences that had no known link.
    """

    known: int
    recovered: int
    wrong: int
    missed: int
    spurious: int
    accuracy: Optional[float]


def _matches(
    expected: ExpectedEvent, event: ConversationEvent
) -> bool:
    """Return whether ``event`` can count for ``expected``."""
    if event.type != expected.type or expected.sentence not in event.sentences:
        return False
    return expected.owner is None or expected.owner == event.owner


def match(
    expected: List[ExpectedEvent], derived: List[ConversationEvent]
) -> List[Tuple[ExpectedEvent, ConversationEvent]]:
    """Pair expected and derived events one to one (keys first)."""
    pairs: List[Tuple[ExpectedEvent, ConversationEvent]] = []
    free = list(derived)
    pending = list(expected)
    for exact in (True, False):
        for item in list(pending):
            for event in free:
                if _matches(item, event) and (
                    not exact or event.key == item.sentence
                ):
                    pairs.append((item, event))
                    free.remove(event)
                    pending.remove(item)
                    break
    return pairs


def score_types(
    expected: Mapping[str, List[ExpectedEvent]],
    derived: Mapping[str, List[ConversationEvent]],
) -> List[TypeScore]:
    """Return precision and recall per event type over the dialogues in
    ``expected`` (derived events of other dialogues are ignored)."""
    totals = {kind: [0, 0, 0] for kind in EVENT_TYPES}
    for dialogue, wanted in expected.items():
        found = derived.get(dialogue, [])
        for item in wanted:
            totals[item.type][0] += 1
        for event in found:
            totals[event.type][1] += 1
        for item, _ in match(wanted, found):
            totals[item.type][2] += 1
    return [
        TypeScore(
            type=kind, expected=e, derived=d, matched=m,
            precision=rate(m, d), recall=rate(m, e),
        )
        for kind, (e, d, m) in totals.items()
    ]


def score_links(
    known: Mapping[str, str], inferred: List[ReplyLink]
) -> LinkScore:
    """Compare ``inferred`` links with the ``known`` ``{source: target}``.
    """
    got = {link.source: link.target for link in inferred}
    recovered = sum(1 for s, t in known.items() if got.get(s) == t)
    missed = sum(1 for s in known if s not in got)
    return LinkScore(
        known=len(known), recovered=recovered,
        wrong=len(known) - recovered - missed, missed=missed,
        spurious=sum(1 for s in got if s not in known),
        accuracy=rate(recovered, len(known)),
    )
