"""Rubric catalogue: competencies, behavioural indicators and anchors."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.domain.errors import InvalidInputError

Polarity = Literal["positive", "negative"]


class Indicator(BaseModel):
    """An observable behaviour that counts for or against a competency."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    polarity: Polarity = "positive"
    examples: List[str] = Field(default_factory=list)

    def query(self) -> str:
        """Return the text matched against transcript turns."""
        return " ".join([self.text, *self.examples])


class Competency(BaseModel):
    """A behavioural competency with its indicators and rating anchors.

    ``anchors`` describe what a rating means in behavioural terms;
    ``sources`` name the published frameworks the competency comes
    from; ``limits`` states what cannot be inferred from a transcript.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    indicators: List[Indicator] = Field(min_length=1)
    anchors: Dict[int, str] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    limits: str = ""

    def indicator(self, indicator_id: str) -> Optional[Indicator]:
        """Return the indicator with this id, if it belongs here."""
        for indicator in self.indicators:
            if indicator.id == indicator_id:
                return indicator
        return None


class Scale(BaseModel):
    """The ordinal rating scale shared by every competency."""

    minimum: int = 1
    maximum: int = 5
    labels: Dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> "Scale":
        """Reject an empty or inverted scale."""
        if self.maximum <= self.minimum:
            raise ValueError("scale maximum must be above its minimum")
        return self

    def points(self) -> List[int]:
        """Return every point of the scale, lowest first."""
        return list(range(self.minimum, self.maximum + 1))

    def midpoint(self) -> float:
        """Return the centre of the scale."""
        return (self.minimum + self.maximum) / 2


class Exclusion(BaseModel):
    """A competency deliberately not assessed, with the reason."""

    name: str
    reason: str


class RubricCatalogue(BaseModel):
    """A versioned set of competencies in one language."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    language: str = Field(min_length=2)
    scale: Scale = Field(default_factory=Scale)
    competencies: List[Competency] = Field(min_length=1)
    excluded: List[Exclusion] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "RubricCatalogue":
        """Check unique ids, anchors on the scale, a positive indicator."""
        points = set(self.scale.points())
        competency_ids: set[str] = set()
        indicator_ids: set[str] = set()
        for competency in self.competencies:
            if competency.id in competency_ids:
                raise ValueError(f"duplicate competency id {competency.id}")
            competency_ids.add(competency.id)
            if not set(competency.anchors) <= points:
                raise ValueError(
                    f"{competency.id}: anchors outside the scale"
                )
            if all(i.polarity != "positive" for i in competency.indicators):
                raise ValueError(
                    f"{competency.id}: needs a positive indicator"
                )
            for indicator in competency.indicators:
                if indicator.id in indicator_ids:
                    raise ValueError(
                        f"duplicate indicator id {indicator.id}"
                    )
                indicator_ids.add(indicator.id)
        return self

    def competency(self, competency_id: str) -> Competency:
        """Return a competency by id."""
        for competency in self.competencies:
            if competency.id == competency_id:
                return competency
        raise InvalidInputError(
            f"Unknown competency {competency_id!r} in catalogue "
            f"{self.name}; choose one of "
            f"{', '.join(c.id for c in self.competencies)}."
        )

    def select(self, ids: Optional[List[str]]) -> List[Competency]:
        """Return the competencies named by ``ids`` (all when empty)."""
        if not ids:
            return list(self.competencies)
        return [self.competency(i) for i in ids]

    def structure(self) -> Dict[str, List[str]]:
        """Return competency id → indicator ids, to compare languages."""
        return {
            c.id: [f"{i.id}:{i.polarity}" for i in c.indicators]
            for c in self.competencies
        }
