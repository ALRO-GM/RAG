"""Metadata filters of ``knowledge.query.v1`` (CONTRACTS.md §2)."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.models import ChunkMetadata
from src.domain.skills.dialogue_acts import CATEGORIES

MAX_FILTER_VALUES = 64
"""Most values one filter may list."""


class KnowledgeFilters(BaseModel):
    """Which chunks a query may return, by their metadata.

    A chunk is kept when, for every filter given, its value is one of
    the listed values; for ``tags`` (a list) when at least one of its
    tags is listed. A filter left out, ``null`` or empty does not
    restrict anything, so a client may always send the four keys.
    """

    model_config = ConfigDict(extra="forbid")

    intent: List[str] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )
    category: List[str] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )
    script_phase: List[str] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )
    tags: List[str] = Field(
        default_factory=list, max_length=MAX_FILTER_VALUES
    )

    @field_validator("category")
    @classmethod
    def _known_categories(cls, values: List[str]) -> List[str]:
        """Refuse categories that are not dialogue-act categories."""
        unknown = [v for v in values if v not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown category {unknown[0]!r}")
        return values

    def is_empty(self) -> bool:
        """Tell whether no filter restricts anything."""
        return not (
            self.intent or self.category or self.script_phase or self.tags
        )

    def matches(self, metadata: ChunkMetadata) -> bool:
        """Tell whether a chunk with ``metadata`` passes every filter."""
        for allowed, value in (
            (self.intent, metadata.intent),
            (self.category, metadata.category),
            (self.script_phase, metadata.script_phase),
        ):
            if allowed and value not in allowed:
                return False
        if self.tags and not set(self.tags) & set(metadata.tags):
            return False
        return True
