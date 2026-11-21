"""Metadata filters of ``knowledge.query.v1`` (CONTRACTS.md §2)."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import ChunkMetadata

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

    intent: Optional[List[str]] = Field(
        default=None, max_length=MAX_FILTER_VALUES
    )
    category: Optional[List[str]] = Field(
        default=None, max_length=MAX_FILTER_VALUES
    )
    script_phase: Optional[List[str]] = Field(
        default=None, max_length=MAX_FILTER_VALUES
    )
    tags: Optional[List[str]] = Field(
        default=None, max_length=MAX_FILTER_VALUES
    )

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
