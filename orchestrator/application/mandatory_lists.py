"""Use case: the owner's lists of «Lo que hay que decir» (#294).

The owner defines from the web which things must be said in the kind
of call being made: a list per call-centre campaign or commercial offer
(``support``) and per job offer (``interview``). This use case lists,
reads, creates (also as a copy), edits, deletes and reviews them, keeps
which list each job offer uses, and tells a session which list it uses
(:meth:`MandatoryLists.for_session`). The general list of each kind is
the product's knowledge domain's (``mandatory`` of its manifest): shown
read-only, it can be copied as a starting point.

Every save is a new version (an edit names the version it was based
on; an older one is refused, as with the interview plans). The review
warns, never refuses: an example close to an example of another item
(the RAG's embedding check, or the word check without the RAG) and an
example of few words. Both are heuristics, not measurements.
"""

import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional

from orchestrator.application.ports import (
    Document,
    MandatoryListStore,
    MandatoryPort,
    MandatoryUnavailable,
)
from orchestrator.domain.mandatory_lists import (
    KINDS,
    MAX_EXAMPLES,
    MAX_ITEMS,
    MAX_LISTS,
    MAX_NAME,
    MAX_TEXT,
    MOMENTS,
    SHORT_WORDS,
    OwnerList,
    effective,
    general_view,
    is_general,
    list_ref,
    short_examples,
    slug,
    summary,
    too_big,
    unique_id,
    with_item_ids,
)


class ListError(Exception):
    """A refused request about a list, with a stable ``code``."""

    def __init__(self, code: str, message: str):
        """Keep the code (``not_found``, ``invalid_request``…)."""
        super().__init__(message)
        self.code = code


class SessionList:
    """The list a session uses, as the tracker needs it.

    ``owned`` is the owner's list (``None``: the general list only);
    :meth:`effective` and :meth:`ref` combine it with the general list
    read when the session starts.
    """

    def __init__(self, kind: str, owned: Optional[OwnerList],
                 lost: Optional[Dict[str, Any]] = None):
        """Keep the kind and the owner's list (a frozen snapshot).

        ``lost``: the list a recorded session used (``{"id",
        "version"}``) when that version could no longer be read, so the
        general list stands in for it and the record says so.
        """
        self.kind = kind
        self.owned = owned
        self.lost = lost

    def effective(self, general: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the list the session checks."""
        if self.owned is None:
            return dict(general)
        return effective(general, self.owned)

    def ref(self, general: Mapping[str, Any]) -> Dict[str, Any]:
        """Return what the session records of its list."""
        ref = list_ref(self.owned, general, self.kind)
        if self.lost is not None:
            ref["instead_of"] = dict(self.lost)
        return ref


class MandatoryLists:
    """The owner's lists, the general ones and the offers' choice.

    Args:
        store: Where the owner's lists are kept.
        port: The mandatory items of the knowledge domains (the general
            lists and the review's similarity check).
        domains: The knowledge domain of each kind (``support``,
            ``interview``): the general list.
        now: The clock of ``updated_at``.
        plan_exists: Tells whether a job offer exists.
    """

    def __init__(
        self,
        store: MandatoryListStore,
        port: MandatoryPort,
        domains: Mapping[str, str],
        now: Callable[[], datetime],
        plan_exists: Callable[[str], bool] = lambda plan_id: True,
    ):
        """Keep the ports."""
        self.store = store
        self.port = port
        self.domains = dict(domains)
        self.now = now
        self.plan_exists = plan_exists
        self._lock = threading.Lock()
        """Saves, creates and deletes one at a time (the routes run on
        a thread pool): read, check and write never interleave."""

    # ---- queries -------------------------------------------------------

    def overview(self) -> Document:
        """Every list (the general ones first) and the offers' choice."""
        lists: List[Document] = []
        for kind in KINDS:
            if kind not in self.domains:
                continue
            general = self._general_view(kind)
            lists.append({**{k: v for k, v in general.items()
                             if k != "items"},
                          "items": len(general["items"])})
        lists.extend(summary(o) for o in self.store.list())
        return {"lists": lists, "offers": self.store.offers(),
                "moments": list(MOMENTS), "limits": limits()}

    def get(self, list_id: str) -> Document:
        """One list in full (a general list is read-only)."""
        if is_general(list_id):
            kind = list_id.split(".", 1)[1]
            if kind not in self.domains:
                raise ListError("not_found", f"no list {list_id!r}")
            return self._general_view(kind)
        owned = self._load(list_id)
        used = sorted(p for p, used in self.store.offers().items()
                      if used == owned.id)
        return {**owned.to_json(), "general": False, "offers": used}

    def general(self, kind: str) -> Document:
        """The RAG's list of ``kind``'s domain (no items when none)."""
        domain = self.domains.get(kind)
        if domain is None:
            raise ListError("not_found", f"no general list for {kind!r}")
        try:
            return self.port.items(domain)
        except MandatoryUnavailable as exc:
            raise ListError("unavailable",
                            "the general list cannot be read now") from exc

    # ---- commands ------------------------------------------------------

    def create(self, body: Mapping[str, Any]) -> Document:
        """Make a list: ``{"name", "kind", "mode"?, "items"}``.

        With ``{"from": id, "name"}`` it is a copy of that list (a copy
        of a general list adds to it, the default: the items it keeps
        have the general ids, so they are not listed twice; a copy of
        an owner's list keeps its mode). The general list is read to
        reserve its ids and check the combined size: when it cannot be
        read nothing is saved (``unavailable``).
        """
        with self._lock:
            return self._create(body)

    def _create(self, body: Mapping[str, Any]) -> Document:
        if len(self.store.list()) >= MAX_LISTS:
            raise ListError("conflict", f"at most {MAX_LISTS} lists")
        source = body.get("from")
        data = dict(body)
        if source is not None:
            if not isinstance(source, str):
                raise ListError("invalid_request", "from must be a list id")
            copied = self.get(source)
            data = {
                "name": body.get("name") or copied.get("name") or "",
                "kind": copied["kind"],
                "mode": "add" if copied.get("general")
                else copied.get("mode", "add"),
                "items": copied["items"],
            }
            keep = [str(i.get("id")) for i in copied["items"]]
        else:
            keep = []
        kind = data.get("kind")
        if kind not in KINDS:
            raise ListError("invalid_request",
                            f"kind must be one of {KINDS}")
        taken = [o.id for o in self.store.list()]
        list_id = unique_id(slug(str(data.get("name") or ""), 48), taken)
        general = self.general(str(kind))
        owned = self._valid(data, list_id, str(kind), 1, keep,
                            _ids(general))
        self._fits(general, owned)
        self.store.save(owned)
        return self._saved(owned)

    def update(self, list_id: str, body: Mapping[str, Any]) -> Document:
        """Save an edit: ``{"version", "name", "mode", "items"}``.

        ``version`` is the one the edit was based on (an older one is
        refused: someone saved first); the kind never changes, and an
        edit without ``mode`` keeps the list's.
        """
        with self._lock:
            current = self._load(list_id)
            if body.get("version") != current.version:
                raise ListError(
                    "conflict",
                    f"the list is at version {current.version}; reload it")
            general = self.general(current.kind)
            owned = self._valid({"mode": current.mode, **body}, current.id,
                                current.kind, current.version + 1,
                                [i.id for i in current.items],
                                _ids(general))
            self._fits(general, owned)
            self.store.save(owned)
        return self._saved(owned)

    def delete(self, list_id: str) -> Document:
        """Delete an owner's list (its offers go back to the general)."""
        if is_general(list_id):
            raise ListError("invalid_request",
                            "the general list cannot be deleted")
        with self._lock:
            self._load(list_id)
            self.store.delete(list_id)
        return {"id": list_id, "deleted": True}

    def review(self, body: Mapping[str, Any]) -> Document:
        """Warnings for a list being edited (saved or not).

        ``short``: examples of fewer than :data:`SHORT_WORDS` words;
        ``close``: examples that come close to an example of another
        item of the list the session would check (with ``add``, the
        general items too). ``checked`` is false when the similarity
        check could not run.
        """
        kind = body.get("kind")
        if kind not in KINDS:
            raise ListError("invalid_request",
                            f"kind must be one of {KINDS}")
        named = {**body, "name": body.get("name") or "-"}
        try:
            reserved = _ids(self.general(str(kind)))
        except ListError:
            reserved = []
        owned = self._valid(named, "review", str(kind), 1,
                            [str(i.get("id")) for i in
                             body.get("items") or [] if isinstance(i, dict)],
                            reserved)
        return self._warnings(owned)

    # ---- the job offers --------------------------------------------------

    def offer(self, plan_id: str) -> Document:
        """The list a job offer uses (``list`` ``None``: the general)."""
        self._check_plan(plan_id)
        used = self.store.offer_list(plan_id)
        if used is not None:
            try:
                owned = self.store.load(used)
            except KeyError:
                used, owned = None, None
        name = owned.name if used is not None and owned else None
        return {"plan": plan_id, "list": used, "name": name}

    def assign(self, plan_id: str, list_id: Optional[str]) -> Document:
        """Make a job offer use an interview list (``None``: general)."""
        self._check_plan(plan_id)
        with self._lock:
            if list_id is not None and not is_general(list_id):
                owned = self._load(list_id)
                if owned.kind != "interview":
                    raise ListError("invalid_request",
                                    "a job offer takes an interview list")
            else:
                list_id = None
            try:
                self.store.assign(plan_id, list_id)
            except KeyError as exc:
                raise ListError("not_found",
                                f"no offer {plan_id!r}") from exc
        return self.offer(plan_id)

    # ---- sessions --------------------------------------------------------

    def for_session(
        self, kind: Optional[str], list_id: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> Optional[SessionList]:
        """Return the list a new session uses (``None``: no kind).

        An interview with a job offer uses the offer's list; otherwise
        the list chosen at the start (``list_id``), which must be of the
        session's kind; without one, the general list. The list is read
        now: an edit made while the session runs applies to the next.
        """
        if kind not in KINDS:
            if list_id is not None and not is_general(list_id):
                raise ListError("invalid_request",
                                "this product takes no list")
            return None
        if kind == "interview" and plan_id is not None:
            used = self.store.offer_list(plan_id)
            if used is not None:
                try:
                    owned = self.store.load(used)
                except KeyError:
                    return SessionList(kind, None)
                return self._checked(SessionList(kind, owned))
        if list_id is None or is_general(list_id):
            return SessionList(kind, None)
        owned = self._load(list_id)
        if owned.kind != kind:
            raise ListError("invalid_request",
                            f"the list {list_id!r} is not for this product")
        return self._checked(SessionList(kind, owned))

    def _checked(self, chosen: SessionList) -> SessionList:
        """Refuse a list the RAG could not check (read again now)."""
        if chosen.owned is not None:
            try:
                general = self.general(chosen.kind)
            except ListError:
                return chosen  # the tracker says so once if it stays down
            self._fits(general, chosen.owned)
        return chosen

    def for_record(self, kind: Optional[str],
                   ref: Optional[Mapping[str, Any]]) -> Optional[SessionList]:
        """Return the list a recorded session used (a recovered round).

        ``ref`` is the hand-off's ``mandatory.list``: the owner's list
        is read at the version recorded; when that version is gone the
        general list stands in and the record says so (``instead_of``).
        """
        if kind not in KINDS:
            return None
        if not isinstance(ref, Mapping) or ref.get("general") is not False:
            return SessionList(str(kind), None)
        list_id, version = ref.get("id"), ref.get("version")
        try:
            if not isinstance(list_id, str) or not isinstance(version, int):
                raise KeyError(list_id)
            return SessionList(str(kind), self.store.load(list_id, version))
        except KeyError:
            return SessionList(str(kind), None,
                               lost={"id": list_id, "version": version})

    # ---- helpers ---------------------------------------------------------

    def _general_view(self, kind: str) -> Document:
        """The general list of ``kind`` (``unavailable`` when unread)."""
        domain = self.domains[kind]
        try:
            listed = self.port.items(domain)
        except MandatoryUnavailable:
            return {**general_view(kind, {"domain": domain}),
                    "unavailable": True}
        return general_view(kind, listed)

    def _load(self, list_id: str) -> OwnerList:
        try:
            return self.store.load(list_id)
        except KeyError as exc:
            raise ListError("not_found", f"no list {list_id!r}") from exc

    def _check_plan(self, plan_id: str) -> None:
        if not self.plan_exists(plan_id):
            raise ListError("not_found", f"no offer {plan_id!r}")

    def _fits(self, general: Mapping[str, Any], owned: OwnerList) -> None:
        """Refuse a list the RAG could not check with the general one."""
        reason = too_big(effective(general, owned))
        if reason is not None:
            raise ListError("list_too_big", TOO_BIG[reason])

    def _valid(self, body: Mapping[str, Any], list_id: str, kind: str,
               version: int, keep: List[str],
               reserved: List[str]) -> OwnerList:
        """Return the edit as a list, or refuse it with a short reason."""
        items = body.get("items")
        if not isinstance(items, list) or not all(
                isinstance(i, dict) for i in items):
            raise ListError("invalid_request", "items must be a list")
        name = body.get("name")
        try:
            return OwnerList.model_validate({
                "id": list_id,
                "name": " ".join(name.split()) if isinstance(name, str)
                else name,
                "kind": kind, "mode": body.get("mode", "add"),
                "version": version,
                "items": [_clean(i) for i in
                          with_item_ids(items, keep, reserved)],
                "updated_at": self.now(),
            })
        except ValueError as exc:
            raise ListError("invalid_request", _reason(exc)) from exc

    def _saved(self, owned: OwnerList) -> Document:
        """The stored list and its warnings."""
        return {**owned.to_json(), "general": False,
                "warnings": self._warnings(owned)}

    def _warnings(self, owned: OwnerList) -> Document:
        """Short examples and examples close to another item's."""
        listed = owned.to_json()
        found: Document = {"short": short_examples(listed), "close": [],
                           "checked": False, "short_words": SHORT_WORDS}
        try:
            general = self.general(owned.kind)
            checked = effective(general, owned)
            if checked["items"]:
                answer = self.port.review(checked)
                found["close"] = [c for c in answer.get("close") or []
                                  if isinstance(c, dict)]
                found["threshold"] = answer.get("threshold")
            found["checked"] = True
        except (ListError, MandatoryUnavailable):
            pass
        return found


TOO_BIG = {
    "too_many_items": "together with the general list it has more than "
                      "30 points; remove some or make it replace the "
                      "general list",
    "too_many_examples": "a point has more than 12 ways of saying it",
    "too_long": "together with the general list its texts are too long "
                "to check; shorten them or remove some points",
}
"""Plain reasons for a list the RAG could not check (#294 review)."""


def _ids(listed: Mapping[str, Any]) -> List[str]:
    """The item ids of a domain's list."""
    return [str(i.get("id")) for i in listed.get("items") or []
            if isinstance(i, dict) and i.get("id")]


def limits() -> Document:
    """The bounds the editor shows and checks."""
    return {"items": MAX_ITEMS, "examples": MAX_EXAMPLES, "text": MAX_TEXT,
            "name": MAX_NAME, "lists": MAX_LISTS}


def _clean(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an editor's item with its texts on one line each."""
    def line(value: Any) -> Any:
        return " ".join(value.split()) if isinstance(value, str) else value

    examples = item.get("must_say")
    return {
        "id": item.get("id"), "text": line(item.get("text")),
        "phase": item.get("phase") or None,
        "must_say": [line(e) for e in examples if not (
            isinstance(e, str) and not e.strip())]
        if isinstance(examples, list) else examples,
    }


def _reason(exc: ValueError) -> str:
    """A short reason for a refused list (no values echoed back)."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts = []
        for error in errors()[:3]:
            where = ".".join(str(p) for p in error.get("loc", ()))
            parts.append(f"{where}: {error.get('msg', '')}".strip(": "))
        return "; ".join(parts)[:300] or "invalid list"
    return str(exc)[:300]
