"""Mandatory script items of a knowledge domain (LV-16).

Some things must be said in every conversation of a product: the legal
data-protection notice of an online contract, the conditions of the
contract, telling a candidate what their data is used for. A domain's
``manifest.yaml`` lists them under ``mandatory``, versioned: each item
has an id, the text shown to the employee, the script phase it belongs
to and ``must_say`` examples of a sentence that says it. An item counts
as said when one of the employee's final sentences is close enough to
one of its examples (cosine similarity of the knowledge embedding
model, at or above the list's ``threshold``, measured in
``docs/live/MANDATORY_ITEMS.md``). Plain data and rules: no I/O.
"""

import re
from typing import Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

ITEM_ID_PATTERN = r"[a-z0-9][a-z0-9_.-]{0,47}"
"""What an item id may look like (it travels in URLs)."""
MAX_ITEMS = 30
MAX_EXAMPLES = 12
MAX_TEXT = 300
DEFAULT_THRESHOLD = 0.6
"""Fallback similarity threshold; each list states its measured one."""
DEFAULT_REMIND_SECONDS = 45


class MandatoryItem(BaseModel):
    """One thing the employee must say."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=rf"^{ITEM_ID_PATTERN}$")
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    phase: Optional[str] = Field(default=None, max_length=64)
    must_say: List[str] = Field(min_length=1, max_length=MAX_EXAMPLES)

    @model_validator(mode="after")
    def _examples(self) -> "MandatoryItem":
        """Refuse empty or too long examples."""
        for example in self.must_say:
            if not example.strip() or len(example) > MAX_TEXT:
                raise ValueError(
                    f"must_say examples of {self.id!r} must have 1 to "
                    f"{MAX_TEXT} characters"
                )
        return self


class MandatoryList(BaseModel):
    """The versioned list of a domain's mandatory items.

    ``version`` grows whenever an item or example changes (a session
    records which version it was checked against); ``remind_seconds``
    is how often the live page brings back an item still unsaid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0.0, lt=1.0)
    remind_seconds: int = Field(default=DEFAULT_REMIND_SECONDS, ge=10,
                                le=600)
    items: List[MandatoryItem] = Field(min_length=1, max_length=MAX_ITEMS)

    @model_validator(mode="after")
    def _unique(self) -> "MandatoryList":
        """Refuse two items with one id."""
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("mandatory item ids must be unique")
        return self

    def examples(self) -> List[str]:
        """Return every example, item after item."""
        return [e for item in self.items for e in item.must_say]

    def to_json(self) -> Dict[str, object]:
        """Return the list as JSON."""
        return self.model_dump(mode="json")


class ItemScore(BaseModel):
    """How close a sentence came to one item: best cosine, and verdict."""

    model_config = ConfigDict(frozen=True)

    id: str
    score: float
    said: bool


def item_scores(
    listed: MandatoryList, similarities: Sequence[float]
) -> List[ItemScore]:
    """Return each item's best similarity and whether it was said.

    ``similarities`` has one cosine per example, in the order of
    :meth:`MandatoryList.examples`. A sentence says at most one item:
    the closest one, when it reaches the threshold (items of one domain
    share words, e.g. the contract's conditions and accepting them, and
    the runner-up was the source of most false hits measured).
    """
    if len(similarities) != len(listed.examples()):
        raise ValueError("one similarity per example is needed")
    best: List[float] = []
    at = 0
    for item in listed.items:
        scores = similarities[at:at + len(item.must_say)]
        at += len(item.must_say)
        best.append(round(float(max(scores)), 4))
    top = max(best)
    return [
        ItemScore(id=item.id, score=score,
                  said=score >= listed.threshold and score == top)
        for item, score in zip(listed.items, best)
    ]


class CloseExample(BaseModel):
    """An example of one item that comes close to an example of another.

    A sentence like ``example`` could count as saying ``other`` instead
    of ``item`` (a sentence says at most one item: the closest), so the
    owner is warned; it is a heuristic, not a measurement (#294).
    """

    model_config = ConfigDict(frozen=True)

    item: str
    example: str
    other: str
    other_example: str
    score: float


def close_examples(
    listed: MandatoryList, similarities: Sequence[Sequence[float]],
    threshold: Optional[float] = None,
) -> List[CloseExample]:
    """Return the examples that reach an example of another item.

    ``similarities`` is the square matrix of cosines between the
    examples, in the order of :meth:`MandatoryList.examples`. An
    example is close to another item when its best cosine to that
    item's examples reaches ``threshold`` (default: the list's); only
    its closest other item is named, once per example.
    """
    owners = [item for item in listed.items for _ in item.must_say]
    examples = listed.examples()
    if len(similarities) != len(examples) or any(
        len(row) != len(examples) for row in similarities
    ):
        raise ValueError("one row and column per example is needed")
    limit = listed.threshold if threshold is None else threshold
    found: List[CloseExample] = []
    for i, row in enumerate(similarities):
        best: Optional[int] = None
        for j, score in enumerate(row):
            if owners[j].id == owners[i].id:
                continue
            if best is None or score > row[best]:
                best = j
        if best is not None and row[best] >= limit:
            found.append(CloseExample(
                item=owners[i].id, example=examples[i],
                other=owners[best].id, other_example=examples[best],
                score=round(float(row[best]), 4),
            ))
    return found


def check_item_id(value: str) -> bool:
    """Tell whether ``value`` is a valid item id."""
    return isinstance(value, str) and bool(
        re.fullmatch(ITEM_ID_PATTERN, value)
    )


def describe(domain_id: str, listed: Optional[MandatoryList]) -> Mapping[
    str, object
]:
    """Return what ``GET .../mandatory`` answers for a domain."""
    if listed is None:
        return {"domain": domain_id, "version": 0, "items": []}
    return {"domain": domain_id, **listed.to_json()}
