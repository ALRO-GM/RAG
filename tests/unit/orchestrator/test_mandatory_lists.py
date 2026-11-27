"""#294: the owner's lists of «Lo que hay que decir»: the rules, the use
case over fakes, the tracker with a session's list and the sessions."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.jsonl_sink import MemoryHandoff
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.mandatory import MandatoryTracker
from orchestrator.application.mandatory_lists import (
    ListError,
    MandatoryLists,
    SessionList,
    limits,
)
from orchestrator.application.ports import Document, MandatoryUnavailable
from orchestrator.application.sessions import (
    LiveSessions,
    SessionError,
    list_kind,
)
from orchestrator.domain.mandatory_lists import (
    MAX_ITEMS,
    OwnerList,
    effective,
    general_view,
    is_general,
    list_ref,
    moment,
    short_examples,
    slug,
    unique_id,
    with_item_ids,
)
from tests.unit.orchestrator.fixtures import REPO, product
from tests.unit.orchestrator.test_employee_ux import final
from tests.unit.orchestrator.test_sessions import (
    Clock,
    FakeEvents,
    FakeTranscriber,
    InlineRunner,
    Transcripts,
)
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner

NOW = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)
GENERAL: Document = {
    "domain": "support.demo", "version": 1, "threshold": 0.575,
    "remind_seconds": 45,
    "items": [
        {"id": "aviso-datos", "text": "Aviso de datos", "phase": "opening",
         "must_say": ["le informo de que sus datos se tratan según la ley"]},
        {"id": "desistimiento", "text": "14 días", "phase": "resolution",
         "must_say": ["tiene catorce días para desistir del contrato"]},
    ],
}
ITEMS: List[Dict[str, Any]] = [
    {"text": "Precio de la tarifa Verano", "phase": "during",
     "must_say": ["la tarifa Verano cuesta veinte euros al mes",
                  "son veinte euros al mes con la tarifa Verano"]},
    {"text": "Sin permanencia", "phase": "closing",
     "must_say": ["no tiene permanencia"]},
]


def owner(mode: str = "add", **extra: Any) -> OwnerList:
    """An owner's support list."""
    return OwnerList.model_validate({
        "id": "verano", "name": "Campaña Verano", "kind": "support",
        "mode": mode, "version": 3,
        "items": with_item_ids(ITEMS), **extra})


# ---- rules ------------------------------------------------------------------

def test_ids_are_made_from_the_text_and_kept_when_known() -> None:
    made_id = slug("¿Aviso de protección de datos?")
    assert made_id == "aviso-de-proteccion-de-datos"
    assert slug("¡¡!!") == "item"
    assert unique_id("a", ["a", "a-2"]) == "a-3"
    made = with_item_ids([{"id": "keep", "text": "x"},
                          {"id": "keep", "text": "Otra cosa"},
                          {"id": "nope", "text": "Otra cosa"}], ["keep"])
    assert [i["id"] for i in made] == ["keep", "otra-cosa", "otra-cosa-2"]


def test_a_list_is_bounded_and_validated() -> None:
    good = owner()
    assert [i.id for i in good.items] == ["precio-de-la-tarifa-verano",
                                          "sin-permanencia"]
    bad: List[Dict[str, Any]] = [
        {"name": " "},
        {"items": []},
        {"items": [{"id": "a", "text": "a", "must_say": []}]},
        {"items": [{"id": "a", "text": "a", "must_say": ["x", " X "]}]},
        {"items": [{"id": "a", "text": " ", "must_say": ["x"]}]},
        {"items": [{"id": "a", "text": "a", "must_say": ["x" * 301]}]},
        {"items": [{"id": "a", "text": "a", "phase": "resolution",
                    "must_say": ["x"]}]},
        {"items": [{"id": "a", "text": "a", "must_say": ["x"]}] * 2},
        {"items": [{"id": f"i{n}", "text": "a", "must_say": ["x"]}
                   for n in range(MAX_ITEMS + 1)]},
        {"id": "Bad Id"},
    ]
    for change in bad:
        with pytest.raises(ValueError):
            OwnerList.model_validate({**good.to_json(), **change})


def test_adding_keeps_the_general_items_first() -> None:
    listed = effective(GENERAL, owner())
    assert [i["id"] for i in listed["items"]] == [
        "aviso-datos", "desistimiento", "precio-de-la-tarifa-verano",
        "sin-permanencia"]
    assert listed["threshold"] == 0.575 and listed["version"] == 3
    assert listed["remind_seconds"] == 45
    assert listed["domain"] == "support.demo"


def test_replacing_lists_only_the_owner_s_items() -> None:
    listed = effective(GENERAL, owner("replace"))
    assert [i["id"] for i in listed["items"]] == [
        "precio-de-la-tarifa-verano", "sin-permanencia"]


def test_an_item_with_a_general_id_takes_its_place() -> None:
    copy = owner(items=[{"id": "aviso-datos", "text": "Aviso propio",
                         "must_say": ["sus datos son suyos y nuestros"]}])
    listed = effective(GENERAL, copy)
    texts = [i["text"] for i in listed["items"]]
    assert texts == ["Aviso propio", "14 días"]
    bare = effective({"items": "nope"}, copy)
    assert bare["threshold"] == 0.6 and bare["remind_seconds"] == 45


def test_what_a_session_records_of_its_list() -> None:
    assert list_ref(None, GENERAL, "support") == {
        "id": "general.support", "name": None, "general": True,
        "domain": "support.demo", "version": 1}
    assert list_ref(owner(), GENERAL, "support") == {
        "id": "verano", "name": "Campaña Verano", "general": False,
        "version": 3, "mode": "add", "domain": "support.demo"}
    assert is_general("general.interview") and not is_general("verano")


def test_the_general_list_shows_moments() -> None:
    view = general_view("support", GENERAL)
    assert [i["phase"] for i in view["items"]] == ["opening", "during"]
    assert view["id"] == "general.support" and view["general"] is True
    assert moment(None) is None and moment("closing") == "closing"


def test_short_examples_are_warned() -> None:
    found = short_examples(owner().to_json())
    assert found == [{"item": "sin-permanencia",
                      "example": "no tiene permanencia", "words": 3}]
    assert short_examples({"items": ["junk"]}) == []


# ---- use case ---------------------------------------------------------------

class MemoryLists:
    """The lists in memory (every version kept)."""

    def __init__(self) -> None:
        self.versions: Dict[str, Dict[int, OwnerList]] = {}
        self.offered: Dict[str, str] = {}
        self.deleted: set[str] = set()

    def list(self) -> List[OwnerList]:
        return [v[max(v)] for k, v in self.versions.items()
                if k not in self.deleted]

    def load(self, list_id: str, version: Optional[int] = None) -> OwnerList:
        if version is None and list_id in self.deleted:
            raise KeyError(list_id)
        found = self.versions[list_id]
        return found[max(found) if version is None else version]

    def save(self, owned: OwnerList) -> None:
        self.versions.setdefault(owned.id, {})[owned.version] = owned

    def taken(self) -> List[str]:
        return list(self.versions)

    def delete(self, list_id: str) -> bool:
        self.offered = {p: x for p, x in self.offered.items()
                        if x != list_id}
        existed = list_id in self.versions and list_id not in self.deleted
        self.deleted.add(list_id)
        return existed

    def offer_list(self, plan_id: str) -> Optional[str]:
        return self.offered.get(plan_id)

    def assign(self, plan_id: str, list_id: Optional[str]) -> None:
        if plan_id == "bad id":
            raise KeyError(plan_id)
        if list_id is None:
            self.offered.pop(plan_id, None)
        else:
            self.offered[plan_id] = list_id

    def offers(self) -> Dict[str, str]:
        return dict(self.offered)


class Port:
    """The general lists and a review that names one close pair."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.reviewed: List[Document] = []

    def items(self, domain: str) -> Document:
        if self.fail:
            raise MandatoryUnavailable("down")
        if domain == "support.demo":
            return dict(GENERAL)
        return {"domain": domain, "version": 2, "items": [
            {"id": "uso-datos", "text": "Uso de datos", "phase": "opening",
             "must_say": ["sus datos solo se usan para este proceso"]}]}

    def check(self, domain: str, text: str) -> Document:
        return {"items": []}

    def check_list(self, listed: Document, text: str) -> Document:
        return {"items": []}

    def review(self, listed: Document) -> Document:
        if self.fail:
            raise MandatoryUnavailable("down")
        self.reviewed.append(listed)
        return {"threshold": 0.575, "close": [
            {"item": "a", "example": "x", "other": "b",
             "other_example": "y", "score": 0.7}, "junk"]}


def use_case(port: Optional[Port] = None) -> MandatoryLists:
    return MandatoryLists(
        MemoryLists(), port or Port(),
        {"support": "support.demo", "interview": "interview.generic"},
        lambda: NOW, plan_exists=lambda plan: plan != "missing")


def test_create_edit_and_delete_a_list() -> None:
    lists = use_case()
    made = lists.create({"name": "  Campaña   Verano ", "kind": "support",
                         "items": ITEMS})
    assert made["id"] == "campana-verano" and made["version"] == 1
    assert made["name"] == "Campaña Verano" and made["mode"] == "add"
    assert made["warnings"]["checked"] is True
    assert made["warnings"]["close"][0]["other"] == "b"
    assert made["warnings"]["short"][0]["item"] == "sin-permanencia"
    again = lists.create({"name": "Campaña Verano", "kind": "support",
                          "items": ITEMS})
    assert again["id"] == "campana-verano-2"
    edit = {**made, "mode": "replace",
            "items": [*made["items"], {"text": "Nuevo punto",
                                       "must_say": ["una frase nueva"]}]}
    saved = lists.update("campana-verano", edit)
    assert saved["version"] == 2 and saved["mode"] == "replace"
    assert [i["id"] for i in saved["items"]][:2] == [
        i["id"] for i in made["items"]]  # the ids stay
    with pytest.raises(ListError) as stale:
        lists.update("campana-verano", edit)  # based on version 1
    assert stale.value.code == "conflict"
    with pytest.raises(ListError) as invalid:
        lists.update("campana-verano", {**saved, "items": []})
    assert invalid.value.code == "invalid_request"
    assert lists.delete("campana-verano") == {"id": "campana-verano",
                                              "deleted": True}
    with pytest.raises(ListError):
        lists.delete("campana-verano")
    with pytest.raises(ListError):
        lists.delete("general.support")
    with pytest.raises(ListError):
        lists.get("nope")
    with pytest.raises(ListError):
        lists.get("general.other")


@pytest.mark.parametrize("body", [
    {"name": "x", "kind": "other", "items": ITEMS},
    {"name": "x", "kind": "support", "items": "nope"},
    {"name": "x", "kind": "support", "items": ["nope"]},
    {"name": "", "kind": "support", "items": ITEMS},
    {"from": 3, "name": "x"},
])
def test_bad_lists_are_refused(body: Dict[str, Any]) -> None:
    with pytest.raises(ListError) as refused:
        use_case().create(body)
    assert refused.value.code == "invalid_request"


def test_a_copy_of_the_general_list_adds_to_it_without_doubling() -> None:
    """#294 review: the ticket's default; the ids keep it from doubling."""
    lists = use_case()
    copy = lists.create({"from": "general.support", "name": "Mi copia"})
    assert copy["mode"] == "add" and copy["kind"] == "support"
    assert [i["id"] for i in copy["items"]] == ["aviso-datos",
                                                "desistimiento"]
    assert [i["phase"] for i in copy["items"]] == ["opening", "during"]
    owned = lists.store.load(copy["id"])
    assert len(effective(GENERAL, owned)["items"]) == 2
    second = lists.create({"from": copy["id"], "name": "Otra copia"})
    assert second["mode"] == "add" and second["id"] == "otra-copia"
    mine = lists.create({"name": "Propia", "kind": "support",
                         "mode": "replace", "items": ITEMS})
    third = lists.create({"from": mine["id"], "name": "Tercera"})
    assert third["mode"] == "replace"


def test_the_overview_lists_the_general_lists_first() -> None:
    lists = use_case()
    lists.create({"name": "Verano", "kind": "support", "items": ITEMS})
    seen = lists.overview()
    assert [x["id"] for x in seen["lists"]] == [
        "general.support", "general.interview", "verano"]
    assert seen["lists"][0]["items"] == 2
    assert seen["moments"] == ["opening", "during", "closing"]
    assert seen["limits"] == limits()
    down = use_case(Port(fail=True))
    assert down.overview()["lists"][0]["unavailable"] is True
    with pytest.raises(ListError):
        down.general("support")
    with pytest.raises(ListError):
        down.general("other")


def test_warnings_without_the_rag_keep_the_short_ones() -> None:
    lists = use_case(Port(fail=True))
    found = lists.review({"kind": "support", "items": ITEMS})
    assert found["checked"] is False and found["close"] == []
    assert len(found["short"]) == 1
    with pytest.raises(ListError):
        lists.review({"kind": "nope", "items": ITEMS})


def test_the_review_checks_what_the_session_would_check() -> None:
    port = Port()
    lists = use_case(port)
    lists.review({"kind": "support", "mode": "add", "items": ITEMS})
    assert [i["id"] for i in port.reviewed[-1]["items"]][:2] == [
        "aviso-datos", "desistimiento"]
    lists.review({"kind": "support", "mode": "replace", "items": ITEMS})
    assert len(port.reviewed[-1]["items"]) == 2


def test_a_job_offer_uses_an_interview_list() -> None:
    lists = use_case()
    mine = lists.create({"name": "Panadería", "kind": "interview",
                         "items": ITEMS})
    support = lists.create({"name": "Verano", "kind": "support",
                            "items": ITEMS})
    assert lists.offer("plan-1") == {"plan": "plan-1", "list": None,
                                     "name": None}
    assert lists.assign("plan-1", mine["id"])["name"] == "Panadería"
    assert lists.get(mine["id"])["offers"] == ["plan-1"]
    with pytest.raises(ListError):
        lists.assign("plan-1", support["id"])
    with pytest.raises(ListError):
        lists.assign("missing", mine["id"])
    with pytest.raises(ListError):
        lists.assign("bad id", None)
    assert lists.assign("plan-1", "general.interview")["list"] is None
    lists.assign("plan-1", mine["id"])
    lists.delete(mine["id"])
    assert lists.offer("plan-1")["list"] is None


def test_the_list_of_a_session() -> None:
    lists = use_case()
    store = lists.store
    mine = lists.create({"name": "Panadería", "kind": "interview",
                         "items": ITEMS})
    verano = lists.create({"name": "Verano", "kind": "support",
                           "items": ITEMS})
    lists.assign("plan-1", mine["id"])
    chosen = lists.for_session("interview", None, "plan-1")
    assert chosen is not None and chosen.owned is not None
    assert chosen.owned.id == mine["id"]
    plain = lists.for_session("interview", None, "plan-2")
    assert plain is not None and plain.owned is None
    support = lists.for_session("support", verano["id"])
    assert support is not None and support.owned is not None
    assert lists.for_session("support", "general.support") is not None
    assert lists.for_session(None) is None
    for kind, chosen_id in (("interview", verano["id"]),
                            (None, verano["id"]), ("support", "nope")):
        with pytest.raises(ListError):
            lists.for_session(kind, chosen_id)
    # An offer whose list file went missing falls back to the general.
    assert isinstance(store, MemoryLists)
    store.deleted.add(mine["id"])
    gone = lists.for_session("interview", None, "plan-1")
    assert gone is not None and gone.owned is None
    assert lists.offer("plan-1")["list"] is None


def test_the_number_of_lists_is_bounded(monkeypatch: pytest.MonkeyPatch
                                        ) -> None:
    import orchestrator.application.mandatory_lists as module
    monkeypatch.setattr(module, "MAX_LISTS", 1)
    lists = use_case()
    lists.create({"name": "Uno", "kind": "support", "items": ITEMS})
    with pytest.raises(ListError) as full:
        lists.create({"name": "Dos", "kind": "support", "items": ITEMS})
    assert full.value.code == "conflict"


# ---- the tracker with a session's list --------------------------------------

class Checks:
    """Says the owner's item for «veinte euros»; records what was sent."""

    def __init__(self) -> None:
        self.sent: List[Document] = []
        self.domain: List[str] = []

    def items(self, domain: str) -> Document:
        return dict(GENERAL)

    def check(self, domain: str, text: str) -> Document:
        self.domain.append(text)
        return {"items": []}

    def check_list(self, listed: Document, text: str) -> Document:
        self.sent.append(listed)
        said = "veinte euros" in text
        return {"items": [{"id": "precio-de-la-tarifa-verano",
                           "score": 0.8, "said": said}]}

    def review(self, listed: Document) -> Document:
        return {"close": []}


def test_the_tracker_checks_the_session_s_list() -> None:
    port = Checks()
    published: List[Document] = []
    track = MandatoryTracker(
        port, "support.demo", {"S1": "agent"}, published.append,
        InlineExecutor(), chosen=SessionList("support", owner()))
    track.start()
    view = track.view()
    assert [i["id"] for i in view["items"]] == [
        "aviso-datos", "desistimiento", "precio-de-la-tarifa-verano",
        "sin-permanencia"]
    assert view["list"]["id"] == "verano" and view["version"] == 3
    track.on_event(final("t1.s0", "S1", "Son veinte euros al mes."))
    assert port.domain == [] and len(port.sent) == 1
    assert port.sent[0]["items"][2]["id"] == "precio-de-la-tarifa-verano"
    track._warm()
    assert len(port.sent) == 2
    summary = track.summary()
    assert summary is not None and summary["said"] == [
        "precio-de-la-tarifa-verano"]
    assert summary["list"] == {"id": "verano", "name": "Campaña Verano",
                               "general": False, "version": 3,
                               "mode": "add", "domain": "support.demo"}


def test_the_general_list_is_checked_by_domain_and_recorded() -> None:
    port = Checks()
    track = MandatoryTracker(
        port, "support.demo", {"S1": "agent"}, lambda m: None,
        InlineExecutor(), chosen=SessionList("support", None))
    track.start()
    track.on_event(final("t1.s0", "S1", "Buenos días."))
    assert port.domain == ["Buenos días."] and port.sent == []
    summary = track.summary()
    assert summary is not None
    assert summary["list"]["id"] == "general.support"


# ---- sessions ---------------------------------------------------------------

def sessions(lists: Optional[MandatoryLists], handoffs: MemoryHandoff
             ) -> LiveSessions:
    return LiveSessions(
        {"live_assist.support": product(),
         "live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor, FakeTranscriber(), FakeEvents([]),
        mandatory=FileMandatory(REPO / "knowledge" / "domains"),
        handoff=lambda sid, name: handoffs, lists=lists,
    )


def test_a_live_call_uses_the_list_chosen_at_the_start() -> None:
    lists = MandatoryLists(
        MemoryLists(), FileMandatory(REPO / "knowledge" / "domains"),
        {"support": "support.demo"}, lambda: NOW)
    made = lists.create({"name": "Verano", "kind": "support",
                         "items": ITEMS})
    handoffs = MemoryHandoff()
    live = sessions(lists, handoffs)
    view = live.start_live("live_assist.support", "stereo", "l-1",
                           mandatory_list=made["id"])
    texts = [i["text"] for i in view["mandatory"]["items"]]
    assert texts[-2:] == ["Precio de la tarifa Verano", "Sin permanencia"]
    assert len(texts) == 6  # the four general items and the two added
    # An edit while the session ran applies to the next session only.
    lists.update(made["id"], {**made, "items": ITEMS[:1]})
    record = handoffs.records[-1]["mandatory"]
    assert record["list"]["id"] == "verano"
    assert record["list"]["version"] == 1
    for bad in ("Bad id!", "nope"):
        with pytest.raises(SessionError):
            live.start_live("live_assist.support", "stereo", "l-2",
                            mandatory_list=bad)
    with pytest.raises(SessionError):
        live.start_live("live_assist.interview", "stereo", "l-3",
                        mandatory_list=made["id"])


def test_without_lists_the_general_list_is_recorded() -> None:
    handoffs = MemoryHandoff()
    live = sessions(None, handoffs)
    live.start_live("live_assist.support", "stereo", "l-1")
    assert handoffs.records[-1]["mandatory"]["list"]["id"] == \
        "general.support"
    with pytest.raises(SessionError) as refused:
        live.start_live("live_assist.support", "stereo", "l-2",
                        mandatory_list="verano")
    assert refused.value.code == "unavailable"


def test_the_kind_of_list_of_a_product() -> None:
    assert list_kind(product()) == "support"
    assert list_kind(product("live_assist.interview")) == "interview"
    assert list_kind(product("voicebot")) is None


def test_the_stored_report_of_a_round_names_its_list() -> None:
    from orchestrator.application.live_judgement import (
        LiveJudge,
        replayed_tracker,
    )
    from tests.unit.orchestrator.test_interview_attribution import (
        the_plan,
        wiring,
    )
    from tests.unit.orchestrator.test_interview_coverage import (
        SESSION,
        Asks,
        stored,
        transcript as round_transcript,
    )
    from tests.unit.orchestrator.test_interview_usecases import Store
    store = Store()
    plan = the_plan()
    store.save(plan)
    judge = LiveJudge(plan, wiring(store, Asks()), SESSION, lambda m: None)
    judge.mandatory_list = list_ref(owner(), GENERAL, "interview")
    judge.judge(replayed_tracker(round_transcript(), plan))
    kept = stored(store)  # valid against interview.sheet.v1
    assert kept["mandatory_list"]["id"] == "verano"
    assert kept["mandatory_list"]["version"] == 3


# ---- #294 review -----------------------------------------------------------

def test_a_new_item_never_takes_a_general_item_s_place() -> None:
    lists = use_case()
    made = lists.create({"name": "Verano", "kind": "support", "items": [
        {"text": "Desistimiento", "must_say": ["catorce días para pensarlo"]},
        {"id": "aviso-datos", "text": "Aviso de datos",
         "must_say": ["sus datos se tratan según la ley vigente"]}]})
    ids = [i["id"] for i in made["items"]]
    assert ids == ["desistimiento-2", "aviso-de-datos"]  # not the general
    listed = effective(GENERAL, lists.store.load(made["id"]))
    assert [i["id"] for i in listed["items"]] == [
        "aviso-datos", "desistimiento", "desistimiento-2", "aviso-de-datos"]
    assert listed["items"][1]["text"] == "14 días"  # the legal item stays
    # A copy of the general list keeps its ids; a new item added to it
    # titled like one of them still gets its own.
    copy = lists.create({"from": "general.support", "name": "Copia"})
    edit = {**copy, "items": [*copy["items"], {
        "text": "Aviso datos", "must_say": ["otro aviso de datos más"]}]}
    saved = lists.update(copy["id"], edit)
    assert [i["id"] for i in saved["items"]] == [
        "aviso-datos", "desistimiento", "aviso-datos-2"]


def test_an_edit_without_mode_keeps_the_list_s_mode() -> None:
    lists = use_case()
    made = lists.create({"name": "Fibra", "kind": "support",
                         "mode": "replace", "items": ITEMS})
    body = {k: v for k, v in made.items() if k != "mode"}
    assert lists.update(made["id"], body)["mode"] == "replace"


def test_a_save_needs_the_general_list() -> None:
    lists = use_case(Port(fail=True))
    with pytest.raises(ListError) as down:
        lists.create({"name": "x", "kind": "support", "items": ITEMS})
    assert down.value.code == "unavailable"


def big_items(n: int, words: int = 8) -> List[Dict[str, Any]]:
    return [{"text": f"Punto {i}", "must_say": [
        " ".join(["palabra"] * words) + f" {i} {j}" for j in range(6)]}
        for i in range(n)]


def test_a_list_too_big_for_the_check_is_refused_at_save() -> None:
    lists = use_case()
    # 2 general items + 20 = 22: fine; 29 general + 2 would not be.
    lists.create({"name": "Veinte", "kind": "support",
                  "items": big_items(20, 3)})
    many = dict(GENERAL, items=[
        {"id": f"g{i}", "text": "g", "must_say": ["x"]} for i in range(29)])
    port = Port()
    port.items = lambda domain: many  # type: ignore[method-assign]
    crowded = use_case(port)
    with pytest.raises(ListError) as refused:
        crowded.create({"name": "Dos", "kind": "support",
                        "items": big_items(2, 3)})
    assert "30 points" in str(refused.value)
    assert refused.value.code == "list_too_big"
    from orchestrator.adapters.inbound.http_server import STATUS
    assert STATUS["list_too_big"] == 422
    replace = crowded.create({"name": "Dos", "kind": "support",
                              "mode": "replace", "items": big_items(2, 3)})
    assert replace["mode"] == "replace"
    wordy = dict(GENERAL, items=[
        {"id": f"g{i}", "text": "g", "must_say": [
            "x" * 280 + str(j) for j in range(12)]} for i in range(25)])
    port_long = Port()
    port_long.items = lambda domain: wordy  # type: ignore[method-assign]
    with pytest.raises(ListError) as long:
        use_case(port_long).create({"name": "Larga", "kind": "support",
                                    "items": big_items(2, 3)})
    assert "too long" in str(long.value)


def test_a_session_list_is_checked_again_when_the_general_grew() -> None:
    port = Port()
    lists = use_case(port)
    made = lists.create({"name": "Dos", "kind": "support",
                         "items": big_items(2, 3)})
    port.items = lambda domain: dict(  # type: ignore[method-assign]
        GENERAL, items=[{"id": f"g{i}", "text": "g", "must_say": ["x"]}
                        for i in range(29)])
    with pytest.raises(ListError):
        lists.for_session("support", made["id"])
    del port.items
    port.fail = True  # the general list cannot be read: no check now
    assert lists.for_session("support", made["id"]) is not None


class SlowLists(MemoryLists):
    """Saves that take a while, so two requests overlap."""

    def save(self, owned: OwnerList) -> None:
        import time
        time.sleep(0.05)
        super().save(owned)


def test_two_saves_of_one_version_give_one_conflict() -> None:
    import threading
    lists = MandatoryLists(
        SlowLists(), Port(), {"support": "support.demo"}, lambda: NOW)
    made = lists.create({"name": "Verano", "kind": "support",
                         "items": ITEMS})
    answers: List[str] = []

    def save(name: str) -> None:
        try:
            lists.update(made["id"], {**made, "name": name})
            answers.append("ok")
        except ListError as exc:
            answers.append(exc.code)

    threads = [threading.Thread(target=save, args=(n,)) for n in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(answers) == ["conflict", "ok"]
    store = lists.store
    assert isinstance(store, MemoryLists)
    assert sorted(store.versions[made["id"]]) == [1, 2]

    made_ids: List[str] = []

    def create() -> None:
        made_ids.append(lists.create({"name": "Igual", "kind": "support",
                                      "items": ITEMS})["id"])

    threads = [threading.Thread(target=create) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(made_ids) == ["igual", "igual-2"]


def test_a_recorded_list_is_read_at_its_version() -> None:
    lists = use_case()
    made = lists.create({"name": "Verano", "kind": "support",
                         "items": ITEMS})
    lists.update(made["id"], {**made, "items": ITEMS[:1]})
    ref = {"id": made["id"], "general": False, "version": 1}
    again = lists.for_record("support", ref)
    assert again is not None and again.owned is not None
    assert again.owned.version == 1 and len(again.owned.items) == 2
    gone = lists.for_record("support", {**ref, "version": 9})
    assert gone is not None and gone.owned is None
    assert gone.ref(GENERAL)["instead_of"] == {"id": made["id"],
                                               "version": 9}
    general = lists.for_record("support", None)
    assert general is not None and general.owned is None
    assert general.ref(GENERAL).get("instead_of") is None
    assert lists.for_record(None, ref) is None


def test_a_deleted_id_is_never_given_to_another_list() -> None:
    """#294 review 2: «Verano» deleted and made again is another list; a
    recovered round of the old session still reads the old one."""
    lists = use_case()
    old = lists.create({"name": "Verano", "kind": "support",
                        "items": ITEMS})
    lists.delete(old["id"])
    assert lists.overview()["lists"][-1]["id"] != old["id"]
    new = lists.create({"name": "Verano", "kind": "support",
                        "items": ITEMS[:1]})
    assert new["id"] == "verano-2" and old["id"] == "verano"
    again = lists.for_record("support", {"id": old["id"], "general": False,
                                         "version": 1})
    assert again is not None and again.owned is not None
    assert again.owned.id == "verano" and len(again.owned.items) == 2
    with pytest.raises(ListError):
        lists.get(old["id"])  # gone from the page
