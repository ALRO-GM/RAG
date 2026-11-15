"""Use case: the explanatory profile of each person (step i, #113).

:func:`person_facts` gathers what the analysis already has for one
person (participation, dialogue acts, events they took part in, firm
levels, provisional estimates, observations and quotes with times) and
gives every quote and event an id; :class:`ProfileSynthesizer` asks the
judge (Claude Code through ``claude -p``; the local model behind it)
for one JSON profile and checks it against those facts
(:func:`~src.domain.skills.synthesis.check_reply`). Replies are cached
like the other judge calls, so a resumed run or an unchanged job
description costs no call.
"""

from typing import Any, Dict, List, Optional

from src.application.caching import cache_key
from src.application.ports import Cache
from src.application.skills.ports import ChatModel
from src.domain.skills.assessment import (
    CompetencyAssessment,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.synthesis import (
    MAX_EVENTS,
    Basis,
    Citable,
    CompetencyLine,
    PersonFacts,
    PersonSynthesis,
    check_reply,
    skipped,
    system_prompt,
    user_prompt,
)

MAX_TOKENS = 1500
"""Reply budget of one profile (the local model needs one)."""

EVENT_FIGURES = (
    "questions_asked", "questions_answered", "questions_unanswered",
    "answers_given", "proposals_made", "proposals_accepted",
    "proposals_rejected", "proposals_ignored", "acceptances_given",
    "commitments", "decisions_involved",
)
"""Per-person event figures listed in the prompt, when present."""


def _rating(item: CompetencyAssessment) -> Optional[int]:
    """Return the firm level or the provisional estimate, if any."""
    if item.rating is not None:
        return item.rating
    return item.estimate.rating if item.estimate is not None else None


def _quotes(
    card: ParticipantScorecard, first: int
) -> List[Citable]:
    """Return the person's quotes as citable items ``Q<first>``…."""
    found: List[Citable] = []
    for item in card.assessments:
        rating = _rating(item)
        kind: Basis
        if item.status == "rated":
            kind = "firm"
            entries = [
                (e.turn, e.quote, e.indicator_id, e.polarity, e.start)
                for e in item.evidence
            ]
        else:
            kind = (
                "provisional" if item.status == "provisional"
                else "observation"
            )
            entries = [
                (o.turn, o.quote, o.indicator_id, o.polarity, o.start)
                for o in item.observed()
            ]
        for turn, quote, indicator, polarity, start in entries:
            found.append(Citable(
                id=f"Q{first + len(found)}", kind=kind, text=quote,
                competency_id=item.competency_id, indicator_id=indicator,
                polarity=polarity, turn=turn, start=start, rating=rating,
            ))
    return found


def _events(events: Optional[Any], participant: str) -> List[Citable]:
    """Return the conversation events ``participant`` took part in."""
    items = events.get("events", []) if isinstance(events, dict) else []
    found: List[Citable] = []
    for event in items:
        if not isinstance(event, dict):
            continue
        if participant not in (event.get("people") or []):
            continue
        what = event.get("what") or event.get("text") or ""
        by = event.get("by") or ""
        found.append(Citable(
            id=f"E{len(found) + 1}", kind="event",
            text=f"{by}: {what}" if by else str(what),
            indicator_id=str(event.get("type") or "event"),
            turn=event.get("turn"), start=event.get("start"),
        ))
        if len(found) >= MAX_EVENTS:
            break
    return found


def _event_counts(
    events: Optional[Any], participant: str
) -> Dict[str, float]:
    """Return the person's event figures from the events' summary."""
    summary = events.get("summary", {}) if isinstance(events, dict) else {}
    for row in summary.get("per_person", []) if isinstance(
        summary, dict
    ) else []:
        if isinstance(row, dict) and row.get("speaker") == participant:
            return {
                k: float(row[k]) for k in EVENT_FIGURES
                if isinstance(row.get(k), (int, float)) and row[k]
            }
    return {}


def _acts(acts: Optional[Any], participant: str) -> Dict[str, int]:
    """Return the person's sentences counted by first act label."""
    items = acts.get("utterances", []) if isinstance(acts, dict) else []
    counts: Dict[str, int] = {}
    for utterance in items:
        if not isinstance(utterance, dict):
            continue
        if utterance.get("speaker") != participant:
            continue
        labels = utterance.get("labels") or []
        first = labels[0] if labels else None
        label = first.get("label") if isinstance(first, dict) else first
        if label:
            counts[str(label)] = counts.get(str(label), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def person_facts(
    assessment: SessionAssessment,
    catalogue: RubricCatalogue,
    participant: str,
    acts: Optional[Any] = None,
    events: Optional[Any] = None,
) -> PersonFacts:
    """Return what the synthesis of ``participant`` may use.

    ``acts`` and ``events`` are the stored results of steps d and e
    (``None`` when those steps did not run).
    """
    card = assessment.scorecard(participant)
    if card is None:
        raise KeyError(participant)
    lines = []
    for item in card.assessments:
        competency = catalogue.competency(item.competency_id)
        lines.append(CompetencyLine(
            competency_id=item.competency_id, name=competency.name,
            status=item.status, rating=_rating(item),
            confidence=item.estimate.confidence if item.estimate else "",
        ))
    quotes = _quotes(card, 1)
    speech = card.interaction
    return PersonFacts(
        participant=participant, language=assessment.language,
        maximum=catalogue.scale.maximum, words=speech.words,
        turns=speech.turns, word_share=speech.word_share,
        questions=speech.questions, acts=_acts(acts, participant),
        event_counts=_event_counts(events, participant),
        competencies=lines,
        citables=quotes + _events(events, participant),
        others=[
            c.participant for c in assessment.scorecards
            if c.participant != participant
        ],
    )


class ProfileSynthesizer:
    """Write the checked profile of one person with a judge.

    Args:
        judge: The model asked (Claude Code, its local fallback, or the
            local model alone), counted in the run's call budget.
        cache: Where replies are kept (``None``: always ask).
        max_tokens: Reply budget handed to the model.
    """

    def __init__(
        self,
        judge: ChatModel,
        cache: Optional[Cache] = None,
        max_tokens: int = MAX_TOKENS,
    ):
        """Wire the judge and the cache."""
        self.judge = judge
        self.cache = cache
        self.max_tokens = max_tokens

    def write(self, facts: PersonFacts, job: str = "") -> PersonSynthesis:
        """Return the profile of the person ``facts`` describes.

        A person with nothing quoted is skipped without a call
        (``no_evidence``): there is nothing to explain.
        """
        if not facts.quotes():
            return skipped(facts, "no_evidence")
        system = system_prompt(facts.language)
        user = user_prompt(facts, job)
        key = cache_key("synthesis", self.judge.model_name, system, user)
        reply = self.cache.get(key) if self.cache else None
        model = self.judge.model_name
        if reply is None:
            reply = self.judge.complete(system, user, self.max_tokens)
            model = self.judge.model_name
            if self.cache:
                self.cache.set(
                    cache_key("synthesis", model, system, user), reply
                )
        return check_reply(reply, facts, model)
