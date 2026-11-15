"""Assessment results: evidence-backed competency ratings.

A competency has one of three results (#103): a firm ``rated`` level,
backed by the minimum evidence; a ``provisional`` estimate when there
is some real evidence but not enough for a level, labelled with its
confidence and the reasons it is not firm; or ``insufficient_evidence``
when nothing at all was seen. Provisional estimates are hypotheses for
the interviewer to check, never assessments.
"""

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.domain.skills.interaction import SpeakerInteraction, TeamInteraction
from src.domain.skills.rubric import Polarity

Status = Literal["rated", "provisional", "insufficient_evidence"]
Confidence = Literal["low", "medium"]
ObservationSource = Literal["below_minimum", "weak_match", "confirmation"]

FORMAT_VERSION = 2
"""Version of the assessment JSON. 2 added the provisional tier and the
observations (#103); files without it are version 1 and still read."""


class Evidence(BaseModel):
    """A verbatim quote of the participant backing one indicator.

    ``similarity`` is the cosine between the turn and the indicator
    according to the embedding model: a second opinion, independent of
    the judge, on whether the quote shows that behaviour.
    """

    turn: int
    quote: str
    indicator_id: str
    polarity: Polarity
    similarity: Optional[float] = None
    start: Optional[float] = None


class Observation(BaseModel):
    """A real sentence of the participant that hints at an indicator.

    It did not count towards a firm level, but a reviewer can check it
    in the transcript (``turn``, ``start`` seconds, ``sentence_id``).
    ``source`` says where it comes from: ``below_minimum``, a verified
    quote of a competency that did not reach the minimum evidence;
    ``weak_match``, a verified quote that the embedding model found too
    far from its indicator; ``confirmation``, a sentence the judge of
    step f confirmed as showing the indicator.
    """

    turn: int
    quote: str
    indicator_id: str
    polarity: Polarity
    source: ObservationSource
    start: Optional[float] = None
    sentence_id: Optional[str] = None
    similarity: Optional[float] = None


class Estimate(BaseModel):
    """A provisional level: a hypothesis to review, not an assessment.

    ``reasons`` are the codes of why it is not firm (see
    :mod:`src.domain.skills.provisional`); ``quotes`` counts the
    distinct turns of verified judge quotes, ``confirmations`` the
    sentences confirmed in step f, ``needed`` the turns a firm level
    needs and ``words`` how much the participant said.
    ``judge_rating`` is what the judge proposed, before the bounds.
    """

    rating: int
    confidence: Confidence
    reasons: List[str] = Field(default_factory=list)
    quotes: int = 0
    confirmations: int = 0
    needed: int = 0
    words: int = 0
    judge_rating: Optional[int] = None


class CompetencyAssessment(BaseModel):
    """The rating of one competency for one participant.

    ``rating`` is ``None`` when the evidence does not reach the minimum
    number of independent incidents. ``candidate_turns`` are the turns
    shown to the judge; ``proposed`` and ``rejected`` count the
    evidence items the judge returned and the ones discarded because
    the quote could not be verified in the transcript; ``unsupported``
    counts verified quotes dropped because the embedding model found
    them too far from the indicator they were filed under.

    ``estimate`` is set only when the status is ``provisional``, and
    ``observations`` (only when there is no firm level) are the other
    real sentences a reviewer should check: verified quotes below the
    similarity threshold and the indicators confirmed in step f.
    """

    competency_id: str
    status: Status
    rating: Optional[int] = None
    evidence: List[Evidence] = Field(default_factory=list)
    rationale: str = ""
    candidate_turns: List[int] = Field(default_factory=list)
    proposed: int = 0
    rejected: int = 0
    unsupported: int = 0
    flags: List[str] = Field(default_factory=list)
    estimate: Optional[Estimate] = None
    observations: List[Observation] = Field(default_factory=list)

    def observed(self) -> List[Observation]:
        """Return what a reviewer should check, in speaking order.

        Nothing for a firm level (its evidence is shown with it);
        otherwise the verified quotes that fell short of the minimum
        and the other observations.
        """
        if self.status == "rated":
            return []
        quotes = [
            Observation(
                turn=e.turn, quote=e.quote, indicator_id=e.indicator_id,
                polarity=e.polarity, source="below_minimum", start=e.start,
                similarity=e.similarity,
            )
            for e in self.evidence
        ]
        return sorted(quotes + self.observations, key=lambda o: o.turn)


class ParticipantScorecard(BaseModel):
    """Every assessed competency of one participant.

    ``person_id`` is the id the transcriber enrolled the participant
    under, when the session came with one: only scorecards that carry
    it can be joined into a profile across sessions.
    """

    participant: str
    person_id: Optional[str] = None
    interaction: SpeakerInteraction
    assessments: List[CompetencyAssessment]

    def assessment(self, competency_id: str) -> Optional[CompetencyAssessment]:
        """Return the assessment of a competency, if present."""
        for item in self.assessments:
            if item.competency_id == competency_id:
                return item
        return None

    def tiers(self) -> Tuple[int, int, int]:
        """Return how many firm levels, provisional ones and observations."""
        return (
            sum(1 for a in self.assessments if a.status == "rated"),
            sum(1 for a in self.assessments if a.status == "provisional"),
            sum(len(a.observed()) for a in self.assessments),
        )


class SessionAssessment(BaseModel):
    """The scorecards of a session and how they were produced.

    ``started_at`` (ISO 8601) and ``conversation_type`` come from the
    transcript when it has them; a profile across sessions uses them to
    order the sessions and to count the kinds of situation seen.
    """

    session_id: str
    language: str
    catalogue: str
    catalogue_version: str
    judge_model: str
    embedding_model: str
    min_incidents: int
    rules: Dict[str, float] = Field(default_factory=dict)
    team: TeamInteraction
    scorecards: List[ParticipantScorecard]
    seconds: float = 0.0
    started_at: Optional[str] = None
    conversation_type: Optional[str] = None
    format_version: int = 1

    def scorecard(self, participant: str) -> Optional[ParticipantScorecard]:
        """Return the scorecard of a participant, if present."""
        for card in self.scorecards:
            if card.participant == participant:
                return card
        return None

    def scorecard_of_person(
        self, person_id: str
    ) -> Optional[ParticipantScorecard]:
        """Return the scorecard of an enrolled person, if present."""
        for card in self.scorecards:
            if card.person_id == person_id:
                return card
        return None

    def ratings(self) -> Dict[str, Dict[str, Optional[int]]]:
        """Return participant → competency id → firm rating."""
        return {
            card.participant: {
                a.competency_id: a.rating for a in card.assessments
            }
            for card in self.scorecards
        }
