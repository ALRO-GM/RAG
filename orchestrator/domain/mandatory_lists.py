"""The owner's lists of «Lo que hay que decir» (#294).

Each knowledge domain has its hand-written list of mandatory items
(``mandatory`` of its manifest: the **general list** of a product,
read-only here). The owner can also define lists from the web, for the
kind of call being made: a call-centre campaign or commercial offer
(``support``) or a job offer (``interview``). A list has a name and
items; an item has the text shown on the live page, the moment of the
conversation it belongs to (``opening``, ``during``, ``closing``) and
one or more example sentences of how it is said (``must_say``: the
check compares what the employee says with them). A list either
**adds** its items to the general list (the default, so the legal
notices of the general list are not lost by accident) or **replaces**
it. Every save makes a new ``version``; a session records the id, name,
version and mode of the list it used (:func:`list_ref`), so a later
edit never changes what an old session says. Plain data and rules: no
I/O.
"""

import json
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

ListKind = Literal["support", "interview"]
ListMode = Literal["add", "replace"]
Moment = Literal["opening", "during", "closing"]
KINDS = ("support", "interview")
MOMENTS = ("opening", "during", "closing")
LIST_ID = r"[a-z0-9][a-z0-9-]{0,63}"
"""What the id of an owner's list may look like (it travels in URLs)."""
ITEM_ID = r"[a-z0-9][a-z0-9_.-]{0,47}"
"""What an item id may look like (the live page's tick uses it)."""
GENERAL_PREFIX = "general."
"""The id of a general list is this prefix and the kind."""
MAX_ITEMS = 20
MAX_EXAMPLES = 6
MAX_TEXT = 300
MAX_NAME = 80
MAX_LISTS = 200
"""Lists kept at most (a bound for the store, not a measurement)."""
SHORT_WORDS = 5
"""An example with fewer words than this gets a warning: few words
match badly (a heuristic, not measured)."""
DEFAULT_THRESHOLD = 0.6
DEFAULT_REMIND = 45
RAG_MAX_ITEMS = 30
"""Items the RAG checks in one list (``src/domain/knowledge/mandatory.py``)."""
RAG_MAX_EXAMPLES = 12
"""Examples per item the RAG takes."""
RAG_MAX_BYTES = 64 * 1024
"""Largest body the RAG reads for one check (its query limit)."""
SENTENCE_RESERVE = 12 * 1024
"""Room kept in that body for the sentence (2000 characters, escaped)."""
PHASE_MOMENTS = {
    "opening": "opening", "closing": "closing",
}
"""A general list's phase as a moment; any other phase is ``during``."""


class OwnerItem(BaseModel):
    """One thing the employee must say, in an owner's list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=rf"^{ITEM_ID}$")
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    phase: Optional[Moment] = None
    must_say: List[str] = Field(min_length=1, max_length=MAX_EXAMPLES)

    @model_validator(mode="after")
    def _examples(self) -> "OwnerItem":
        """Refuse a blank text and blank, long or repeated examples."""
        if not self.text.strip():
            raise ValueError("an item needs a text")
        seen = set()
        for example in self.must_say:
            if not example.strip() or len(example) > MAX_TEXT:
                raise ValueError(
                    f"examples must have 1 to {MAX_TEXT} characters")
            key = " ".join(example.lower().split())
            if key in seen:
                raise ValueError("an item repeats an example")
            seen.add(key)
        return self


class OwnerList(BaseModel):
    """A list of «Lo que hay que decir» defined by the owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=rf"^{LIST_ID}$")
    name: str = Field(min_length=1, max_length=MAX_NAME)
    kind: ListKind
    mode: ListMode = "add"
    version: int = Field(default=1, ge=1)
    items: List[OwnerItem] = Field(min_length=1, max_length=MAX_ITEMS)
    updated_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _unique(self) -> "OwnerList":
        """Refuse a blank name and two items with one id."""
        if not self.name.strip():
            raise ValueError("a list needs a name")
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique")
        return self

    def to_json(self) -> Dict[str, Any]:
        """Return the list as JSON."""
        return self.model_dump(mode="json")


def kind_of_roles(roles: Sequence[str]) -> Optional[str]:
    """The kind of list a conversation with these roles uses.

    ``support`` when an agent speaks, ``interview`` when an interviewer
    does; ``None`` otherwise (a voice bot's user).
    """
    if "agent" in roles:
        return "support"
    if "interviewer" in roles:
        return "interview"
    return None


def general_id(kind: str) -> str:
    """Return the id of the general list of ``kind``."""
    return f"{GENERAL_PREFIX}{kind}"


def is_general(list_id: str) -> bool:
    """Tell whether ``list_id`` names a general (read-only) list."""
    return list_id in (general_id(k) for k in KINDS)


def moment(phase: Optional[str]) -> Optional[str]:
    """Return a general list's phase as a moment of an owner's list."""
    if not phase:
        return None
    return PHASE_MOMENTS.get(phase, "during")


def slug(text: str, limit: int = 40) -> str:
    """Return a lower-case id made of ``text``'s words."""
    plain = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore").decode("ascii").lower()
    made = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")[:limit].strip("-")
    return made or "item"


def unique_id(base: str, taken: Sequence[str]) -> str:
    """Return ``base``, or ``base-2``, ``base-3``… not in ``taken``."""
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def with_item_ids(
    items: Sequence[Mapping[str, Any]], keep: Sequence[str] = (),
    reserved: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Return the editor's items with an id each (the server's ids).

    An item keeps the id it brings when that id is in ``keep`` (the
    ids of the list being edited, or of the list it was copied from);
    any other item gets one made from its text, unique in the list and
    never one of ``reserved`` (the general list's ids: a new item must
    never take a general item's place by its title, #294 review).
    """
    used: List[str] = []
    for raw in items:
        given = raw.get("id")
        if isinstance(given, str) and given in keep and given not in used:
            used.append(given)
    taken: List[str] = [*reserved, *keep, *used]
    out: List[Dict[str, Any]] = []
    kept: List[str] = []
    for raw in items:
        item = dict(raw)
        given = item.get("id")
        if isinstance(given, str) and given in keep and given not in kept:
            chosen = given
            kept.append(given)
        else:
            chosen = unique_id(slug(str(item.get("text") or "")), taken)
        taken.append(chosen)
        item["id"] = chosen
        out.append(item)
    return out


def general_view(kind: str, listed: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a domain's list as the read-only general list of ``kind``.

    ``listed`` is the RAG's list (``GET .../mandatory``): its phases
    are shown as moments; it keeps its domain and version.
    """
    items = [
        {"id": str(i.get("id")), "text": str(i.get("text") or ""),
         "phase": moment(i.get("phase")),
         "must_say": [str(e) for e in i.get("must_say") or []]}
        for i in listed.get("items") or [] if isinstance(i, dict)
    ]
    return {
        "id": general_id(kind), "name": None, "kind": kind,
        "general": True, "mode": "replace",
        "domain": listed.get("domain"),
        "version": int(listed.get("version") or 0), "items": items,
    }


def effective(general: Mapping[str, Any], owned: OwnerList) -> Dict[str, Any]:
    """Return the list a session checks: the general list and ``owned``.

    ``general`` is the RAG's list of the product's domain (its
    threshold and reminder are kept: they were measured for the check,
    not for one list). With ``add`` the owner's items follow the
    general ones, and an item with a general item's id takes its place
    (a copy of the general list never doubles it); with ``replace``
    only the owner's items are listed. The result has the shape of a
    domain's list, so the RAG checks it the same way.
    """
    base = [dict(i) for i in general.get("items") or []
            if isinstance(i, dict)] if owned.mode == "add" else []
    mine = {item.id: item for item in owned.items}
    items: List[Dict[str, Any]] = []
    for item in base:
        replaced = mine.pop(str(item.get("id")), None)
        items.append(_item(replaced) if replaced is not None else {
            "id": item.get("id"), "text": item.get("text"),
            "phase": item.get("phase"),
            "must_say": list(item.get("must_say") or []),
        })
    items.extend(_item(i) for i in owned.items if i.id in mine)
    threshold = general.get("threshold")
    return {
        "domain": general.get("domain"), "version": owned.version,
        "threshold": threshold if isinstance(threshold, (int, float))
        and not isinstance(threshold, bool) else DEFAULT_THRESHOLD,
        "remind_seconds": int(general.get("remind_seconds")
                              or DEFAULT_REMIND),
        "items": items,
    }


def too_big(listed: Mapping[str, Any]) -> Optional[str]:
    """Tell why the RAG could not check ``listed`` (``None``: it can).

    More than :data:`RAG_MAX_ITEMS` items or :data:`RAG_MAX_EXAMPLES`
    examples in one, or a body that with a sentence would pass the
    RAG's :data:`RAG_MAX_BYTES`. The reason is a stable code.
    """
    items = [i for i in listed.get("items") or [] if isinstance(i, dict)]
    if len(items) > RAG_MAX_ITEMS:
        return "too_many_items"
    if any(len(i.get("must_say") or []) > RAG_MAX_EXAMPLES for i in items):
        return "too_many_examples"
    size = len(json.dumps({"list": dict(listed)}).encode("utf-8"))
    if size + SENTENCE_RESERVE > RAG_MAX_BYTES:
        return "too_long"
    return None


def _item(item: OwnerItem) -> Dict[str, Any]:
    """Return an owner's item in the shape of a domain list's item."""
    return {"id": item.id, "text": item.text, "phase": item.phase,
            "must_say": list(item.must_say)}


def list_ref(owned: Optional[OwnerList], general: Mapping[str, Any],
             kind: str) -> Dict[str, Any]:
    """Return what a session records of the list it used.

    The owner's list's id, name, version and mode; without one, the
    general list (its domain and version).
    """
    if owned is None:
        return {"id": general_id(kind), "name": None, "general": True,
                "domain": general.get("domain"),
                "version": int(general.get("version") or 0)}
    return {"id": owned.id, "name": owned.name, "general": False,
            "version": owned.version, "mode": owned.mode,
            "domain": general.get("domain")}


def short_examples(owned: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return the examples with fewer than :data:`SHORT_WORDS` words.

    A heuristic warning for the editor (few words match badly); never a
    reason to refuse the list.
    """
    found: List[Dict[str, Any]] = []
    for item in owned.get("items") or []:
        if not isinstance(item, dict):
            continue
        for example in item.get("must_say") or []:
            words = len(re.findall(r"\w+", str(example)))
            if 0 < words < SHORT_WORDS:
                found.append({"item": item.get("id"), "example": example,
                              "words": words})
    return found


def summary(owned: OwnerList) -> Dict[str, Any]:
    """Return a list as the list of lists shows it (no examples)."""
    return {
        "id": owned.id, "name": owned.name, "kind": owned.kind,
        "mode": owned.mode, "version": owned.version, "general": False,
        "items": len(owned.items),
        "updated_at": owned.updated_at.isoformat()
        if owned.updated_at else None,
    }
