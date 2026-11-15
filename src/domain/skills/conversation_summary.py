"""Per-person and per-session counts of the conversation events.

Counts and rates only, never ratings: how many questions someone asked
and how many got an answer, what happened to their proposals, which
commitments they took, how many turns their replies took. A rate is
``None`` when its denominator is zero, so "no proposals" is never
shown as "0 % accepted".
"""

from statistics import median
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.domain.skills.act_sequence import ActSession
from src.domain.skills.conversation_events import (
    EVENT_TYPES,
    ConversationEvent,
    ReplyLink,
)


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def rate(part: int, whole: int) -> Optional[float]:
    """Return ``part / whole`` rounded to 3 places, or ``None`` for 0."""
    return round(part / whole, 3) if whole else None


class PersonEvents(_Frozen):
    """What the events say about one speaker (counts and rates).

    ``answers_given`` counts other people's questions the speaker
    answered; ``median_reply_turns`` is the median number of turns
    between someone else's first-pair part and the speaker's reply to it.
    """

    speaker: str
    questions_asked: int
    questions_answered: int
    questions_unanswered: int
    answer_rate: Optional[float]
    answers_given: int
    proposals_made: int
    proposals_accepted: int
    proposals_rejected: int
    proposals_contested: int
    proposals_ignored: int
    acceptance_rate: Optional[float]
    ignored_rate: Optional[float]
    acceptances_given: int
    rejections_given: int
    commitments: int
    decisions_involved: int
    median_reply_turns: Optional[float]


class SessionEvents(_Frozen):
    """Totals of a session and the per-person counts.

    ``counts`` has every event type (zeros included); ``ignored_by``
    is whose proposals were ignored and how many times.
    """

    session_id: str
    sentences: int
    links_given: int
    links_inferred: int
    counts: Dict[str, int]
    median_reply_turns: Optional[float]
    ignored_by: Dict[str, int]
    per_person: List[PersonEvents]

    def of(self, speaker: str) -> PersonEvents:
        """Return the counts of ``speaker``."""
        for item in self.per_person:
            if item.speaker == speaker:
                return item
        raise KeyError(speaker)


def _median(values: List[int]) -> Optional[float]:
    """Return the median of ``values``, or ``None`` when empty."""
    return float(median(values)) if values else None


def summarise(
    session: ActSession,
    links: List[ReplyLink],
    events: List[ConversationEvent],
) -> SessionEvents:
    """Return the session totals and each speaker's counts."""
    speaker_of = {s.id: s.speaker for s in session.sentences}
    sentence = {s.id: s for s in session.sentences}
    first_parts = {
        e.key for e in events
        if e.type.startswith(("question", "proposal"))
        or (e.type == "action_item" and e.owner != e.by)
    }
    latencies: Dict[str, List[int]] = {}
    for link in links:
        if link.target in first_parts:
            latencies.setdefault(speaker_of[link.source], []).append(
                link.turns)

    def count(kind: str, person: str, field: str = "by") -> int:
        """Count the events of ``kind`` where ``person`` is ``field``."""
        return sum(
            1 for e in events
            if e.type == kind and getattr(e, field) == person
        )

    def replies(person: str, label: str) -> int:
        """Count ``person``'s linked replies carrying ``label``."""
        return sum(
            1 for link in links
            if speaker_of[link.source] == person
            and sentence[link.source].weight(label) > 0
        )

    people: List[PersonEvents] = []
    for person in session.speakers():
        answered = count("question_answered", person)
        unanswered = count("question_unanswered", person)
        outcomes = {
            kind: count(f"proposal_{kind}", person)
            for kind in ("accepted", "rejected", "contested", "ignored")
        }
        made = sum(outcomes.values())
        people.append(PersonEvents(
            speaker=person,
            questions_asked=answered + unanswered,
            questions_answered=answered, questions_unanswered=unanswered,
            answer_rate=rate(answered, answered + unanswered),
            answers_given=sum(
                1 for e in events
                if e.type == "question_answered" and person in e.responders
            ),
            proposals_made=made,
            proposals_accepted=outcomes["accepted"],
            proposals_rejected=outcomes["rejected"],
            proposals_contested=outcomes["contested"],
            proposals_ignored=outcomes["ignored"],
            acceptance_rate=rate(outcomes["accepted"], made),
            ignored_rate=rate(outcomes["ignored"], made),
            acceptances_given=replies(person, "aceptacion"),
            rejections_given=replies(person, "rechazo"),
            commitments=count("action_item", person, "owner"),
            decisions_involved=sum(
                1 for e in events
                if e.type == "decision" and person in e.people
            ),
            median_reply_turns=_median(latencies.get(person, [])),
        ))
    counts: Dict[str, int] = {kind: 0 for kind in EVENT_TYPES}
    for event in events:
        counts[event.type] += 1
    return SessionEvents(
        session_id=session.id, sentences=len(session.sentences),
        links_given=sum(1 for k in links if not k.inferred),
        links_inferred=sum(1 for k in links if k.inferred),
        counts=counts,
        median_reply_turns=_median(
            [t for values in latencies.values() for t in values]),
        ignored_by={
            p.speaker: p.proposals_ignored for p in people
            if p.proposals_ignored
        },
        per_person=people,
    )
