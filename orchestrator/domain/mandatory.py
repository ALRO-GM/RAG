"""The mandatory items of a live session and what became of each (LV-16).

The product's knowledge domain lists what the employee must say in
every conversation (``mandatory`` of the domain manifest, served by the
RAG). During a session each item is ``pending`` until the employee says
it (``said``: a final sentence of theirs matched it, with the utterance
and the similarity) or ticks it by hand after confirming (``ticked``:
logged with the moment, never silent). Nothing dismisses an item
otherwise. At the end the hand-off lists what was said, ticked and
left unsaid. Plain data: no I/O.
"""

from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, Field

ItemState = Literal["pending", "said", "ticked"]
EMPLOYEE_ROLES = ("agent", "interviewer")
"""Roles whose sentences can say a mandatory item."""


class ChecklistItem(BaseModel):
    """One mandatory item in a session."""

    id: str
    text: str
    phase: Optional[str] = None
    state: ItemState = "pending"
    utterance: Optional[str] = None
    score: Optional[float] = None
    at: Optional[float] = None


class Checklist(BaseModel):
    """The session's mandatory items, as the page and the hand-off see them.

    ``remind_seconds`` is how often the page brings an unsaid item
    back; ``ticks`` logs every manual tick (item, stream time).
    """

    domain: str
    version: int = 0
    threshold: Optional[float] = None
    remind_seconds: int = 45
    items: List[ChecklistItem] = Field(default_factory=list)
    ticks: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_list(cls, listed: Mapping[str, Any]) -> "Checklist":
        """Build the checklist from the RAG's list of a domain."""
        items = [
            ChecklistItem(id=str(i["id"]), text=str(i["text"]),
                          phase=i.get("phase"))
            for i in listed.get("items") or []
            if isinstance(i, dict) and i.get("id") and i.get("text")
        ]
        return cls(
            domain=str(listed.get("domain", "")),
            version=int(listed.get("version") or 0),
            threshold=listed.get("threshold"),
            remind_seconds=int(listed.get("remind_seconds") or 45),
            items=items,
        )

    def find(self, item_id: str) -> Optional[ChecklistItem]:
        """Return the item ``item_id``, if listed."""
        return next((i for i in self.items if i.id == item_id), None)

    def pending(self) -> List[str]:
        """Return the ids of the items not said nor ticked yet."""
        return [i.id for i in self.items if i.state == "pending"]

    def mark_said(
        self, item_id: str, utterance: str, score: float, at: float
    ) -> bool:
        """Mark a pending item said by ``utterance``; tell if it changed."""
        item = self.find(item_id)
        if item is None or item.state != "pending":
            return False
        item.state, item.utterance = "said", utterance
        item.score, item.at = score, at
        return True

    def tick(self, item_id: str, at: float) -> bool:
        """Tick a pending item by hand (logged); tell if it changed."""
        item = self.find(item_id)
        if item is None or item.state != "pending":
            return False
        item.state, item.at = "ticked", at
        self.ticks.append({"item": item_id, "at": at})
        return True

    def summary(self) -> Dict[str, Any]:
        """Return the hand-off's account: said, ticked and missing."""
        def ids(state: ItemState) -> List[str]:
            return [i.id for i in self.items if i.state == state]

        return {
            "domain": self.domain, "version": self.version,
            "said": ids("said"), "ticked": ids("ticked"),
            "missing": [{"id": i.id, "text": i.text}
                        for i in self.items if i.state == "pending"],
            "ticks": list(self.ticks),
        }


def said_items(check: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return ``{"id", "score"}`` of the items a check found said."""
    return [
        {"id": str(i["id"]), "score": float(i.get("score") or 0.0)}
        for i in check.get("items") or []
        if isinstance(i, dict) and i.get("said") is True and i.get("id")
    ]
