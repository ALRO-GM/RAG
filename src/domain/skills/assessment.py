"""Assessment results: evidence-backed competency ratings."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.domain.skills.interaction import SpeakerInteraction, TeamInteraction
from src.domain.skills.rubric import Polarity

Status = Literal["rated", "insufficient_evidence"]


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


class CompetencyAssessment(BaseModel):
    """The rating of one competency for one participant.

    ``rating`` is ``None`` when the evidence does not reach the minimum
    number of independent incidents. ``candidate_turns`` are the turns
    shown to the judge; ``proposed`` and ``rejected`` count the
    evidence items the judge returned and the ones discarded because
    the quote could not be verified in the transcript; ``unsupported``
    counts verified quotes dropped because the embedding model found
    them too far from the indicator they were filed under.
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
        """Return participant → competency id → rating."""
        return {
            card.participant: {
                a.competency_id: a.rating for a in card.assessments
            }
            for card in self.scorecards
        }
