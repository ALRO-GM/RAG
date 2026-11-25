"""The offer page's figure, order and table (#271).

Synthetic data only: invented offers, people and answers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from orchestrator.application.candidates import CandidateBook
from orchestrator.application.interview import InterviewWiring
from orchestrator.domain.candidate import Round, new_profile
from orchestrator.domain.interview import (
    InterviewPlan,
    SheetSlot,
    Slot,
    suitability,
)
from orchestrator.domain.offer_board import (
    LABEL_VALUE,
    interview_view,
    live_value,
    ordered,
    point_result,
)
from orchestrator.adapters.outbound.fakes import FakeReasoner
from tests.unit.orchestrator.test_candidate import Core, MemoryCandidates
from tests.unit.orchestrator.test_interview_usecases import Store, planner

NOW = datetime(2026, 11, 25, 12, 0, 0)


def offer(revision: int = 0, extra: Optional[List[Slot]] = None
          ) -> InterviewPlan:
    """An invented offer: two required points and one desirable."""
    slots = [
        Slot(id="s1", label="Tipos de café", level="minimo", weight=4,
             questions=["¿Qué cafés conoces?"]),
        Slot(id="s2", label="Atención en barra", level="minimo", weight=4,
             questions=["¿Cómo atiendes una barra llena?"]),
        Slot(id="s3", label="Inglés", level="deseable", weight=2,
             questions=["¿Qué nivel de inglés tienes?"]),
        *(extra or []),
    ]
    return InterviewPlan(id="cafe", title="Barista", slots=slots,
                         revision=revision)


def judged(label: str) -> Dict[str, Any]:
    """A judgement of a point (invented quote and reason)."""
    return {"label": label, "reason": "Razón inventada.",
            "quote": "frase inventada", "utterance": "u1",
            "basis": "general", "cites": []}


def slot(sid: str, level: str = "minimo", weight: int = 4,
         **fields: Any) -> Dict[str, Any]:
    """A stored sheet slot."""
    return {"id": sid, "label": sid, "level": level, "weight": weight,
            **fields}


def live(session: str, slots: List[Any], **fields: Any
         ) -> Dict[str, Any]:
    """A stored live report."""
    return {"schema": "interview.sheet.v1", "kind": "live",
            "session_id": session, "plan_id": "cafe", "title": "Barista",
            "plan_revision": 0, "slots": slots, "suitability": None,
            "attribution": "after_session", **fields}


# ---- one point -------------------------------------------------------------

def test_a_live_point_is_valued_from_its_judgement_and_concepts() -> None:
    concepts = [{"id": "p1", "text": "espresso", "status": "treated"},
                {"id": "p2", "text": "filtro", "status": "pending"}]
    half = SheetSlot.model_validate(slot(
        "s1", status="partial", judgement=judged("solid"), points=concepts))
    assert live_value(half, True) == 75.0  # 100 × (0.5 + 0.5 × 1/2)
    basic = SheetSlot.model_validate(slot(
        "s1", status="filled", judgement=judged("correct_basic")))
    assert live_value(basic, True) == 70.0
    wrong_concept = SheetSlot.model_validate(slot(
        "s1", status="filled", judgement=judged("solid"), points=[
            {"id": "p1", "text": "a", "status": "treated",
             "judgement": judged("wrong")},
            {"id": "p2", "text": "b", "status": "treated"}]))
    assert live_value(wrong_concept, True) == 50.0
    from_concepts = SheetSlot.model_validate(slot(
        "s1", status="filled", points=[
            {"id": "p1", "text": "a", "status": "treated",
             "judgement": judged("solid")},
            {"id": "p2", "text": "b", "status": "treated",
             "judgement": judged("correct_basic")}]))
    assert live_value(from_concepts, True) == 85.0
    answered = SheetSlot.model_validate(slot("s1", status="filled"))
    assert live_value(answered, True) == 40.0
    empty = SheetSlot.model_validate(slot("s3", status="empty"))
    assert live_value(empty, True) == 10.0
    assert live_value(empty, False) is None


def test_the_values_keep_the_order_the_ticket_asks_for() -> None:
    """solid > basic > unclear/to verify > not treated > off topic/wrong."""
    v = LABEL_VALUE
    assert v["solid"] > v["correct_basic"] > v["to_verify"] >= v["unclear"]
    assert v["unclear"] > v["not_treated"] > v["off_topic"] >= v["wrong"]


def test_a_point_says_its_state_and_evidence() -> None:
    result = point_result("s1", SheetSlot.model_validate(slot(
        "s1", status="partial", judgement=judged("unclear"))), "live", True)
    assert (result.state, result.label, result.quote, result.reason) == (
        "unclear", "unclear", "frase inventada", "Razón inventada.")
    assert point_result("s9", None, "live", True).state == "not_asked"
    unknown = point_result("s1", SheetSlot.model_validate(slot(
        "s1", status="unknown")), "live", True)
    assert (unknown.state, unknown.value) == ("does_not_know", 10.0)
    majority = point_result("s1", SheetSlot.model_validate(slot(
        "s1", status="filled", points=[
            {"id": "p1", "text": "a", "status": "treated",
             "judgement": judged("wrong")},
            {"id": "p2", "text": "b", "status": "treated",
             "judgement": judged("solid")}])), "live", True)
    assert majority.state == "wrong"  # a tie goes to the lower value
    ai = point_result("s1", SheetSlot.model_validate(slot(
        "s1", status="filled", score=40, scored_by="status")), "ai", True)
    assert (ai.value, ai.basis, ai.score) == (40.0, "status", 40)


# ---- one interview ---------------------------------------------------------

def ai_sheet(session: str = "vb-1", **fields: Any) -> Dict[str, Any]:
    """A stored AI interview sheet with its suitability."""
    slots = [slot("s1", status="filled", score=80, scored_by="model"),
             slot("s2", status="partial", score=60, scored_by="model"),
             slot("s3", "deseable", 2, status="empty")]
    stored = suitability([SheetSlot.model_validate(s) for s in slots])
    return {"session_id": session, "kind": "ai", "plan_revision": 0,
            "slots": slots, "suitability": stored, "candidate": "Ana",
            "finished_at": "2026-11-20T10:00:00", **fields}


def test_an_ai_sheet_keeps_its_own_suitability() -> None:
    view = interview_view(offer(), ai_sheet())
    assert view.figure == view.stored_suitability == 70.0
    assert view.figure_basis == "scores" and not view.less_reliable
    assert view.missing_required == [] and not view.plan_changed
    failed = interview_view(offer(), ai_sheet(report_error="timeout"))
    assert failed.figure_basis == "status"


def test_a_live_report_gets_a_figure_from_its_labels() -> None:
    view = interview_view(offer(), live("r1", [
        slot("s1", status="filled", judgement=judged("solid")),
        slot("s2", status="filled", judgement=judged("off_topic")),
    ], to_check=[{"quote": "x", "why": "vague"}], candidate="Luis"))
    # s1 100, s2 5 (off topic), s3 never reached: desirable, left out.
    assert view.figure == round((4 * 100 + 4 * 5) / 8, 1)
    assert view.missing_required == ["s2"]
    assert [p.state for p in view.points] == [
        "solid", "off_topic", "not_treated"]
    assert view.to_check == 1 and view.figure_basis == "labels"


def test_word_rule_reports_are_marked_less_reliable() -> None:
    rules = interview_view(offer(), live("r1", [], attribution="rules"))
    older = interview_view(offer(), live("r2", [], attribution=None))
    model = interview_view(offer(), live("r3", []))
    assert rules.less_reliable and older.less_reliable
    assert not model.less_reliable


def test_an_edited_plan_compares_on_the_points_that_still_exist() -> None:
    added = Slot(id="s4", label="Latte art", level="minimo", weight=2,
                 questions=["¿Haces latte art?"])
    sheet = ai_sheet()
    sheet["slots"].append(slot("s9", status="filled", score=0))  # removed
    view = interview_view(offer(revision=1, extra=[added]), sheet)
    assert view.plan_changed and view.not_asked == ["s4"]
    assert view.figure == 70.0  # s4 left out, s9 dropped
    assert view.missing_required == [] and view.unasked_required == ["s4"]
    assert not view.covers_required  # no evidence on s4
    assert [p.id for p in view.points] == ["s1", "s2", "s3", "s4"]


def test_saving_the_plan_never_changes_a_live_figure_by_itself() -> None:
    """Review of #271: a live report keeps only the slots it judged."""
    reached = live("ra", [slot("s1", status="filled",
                               judgement=judged("solid"))])
    both = live("rb", [
        slot("s1", status="filled", judgement=judged("solid")),
        slot("s2", status="filled", judgement=judged("correct_basic"))])
    before = [interview_view(offer(), r) for r in (reached, both)]
    after = [interview_view(offer(revision=1), r) for r in (reached, both)]
    assert [v.figure for v in before] == [v.figure for v in after] == [
        55.0, 85.0]
    assert [p.state for p in after[0].points] == [
        "solid", "not_treated", "not_treated"]
    assert [v.session_id for v in ordered(after)] == ["rb", "ra"]
    # With the revision's slot ids kept, a point really added later is
    # «no preguntado»; a point of that revision never reached is not.
    added = Slot(id="s4", label="Latte art", level="minimo", weight=2,
                 questions=["¿Haces latte art?"])
    listed = {**reached, "plan_slots": ["s1", "s2", "s3"]}
    view = interview_view(offer(revision=2, extra=[added]), listed)
    assert [p.state for p in view.points] == [
        "solid", "not_treated", "not_treated", "not_asked"]
    assert view.figure == 55.0
    assert view.missing_required == ["s2"]
    assert view.unasked_required == ["s4"]
    # An older live report without the list: never «no preguntado».
    old = interview_view(offer(revision=2, extra=[added]), reached)
    assert old.not_asked == [] and old.points[3].state == "not_treated"


def test_an_unreadable_slot_is_left_out_and_counts_nothing() -> None:
    view = interview_view(offer(), live("r1", [
        {"id": "s1", "status": "nonsense"}, "x",
        slot("s2", status="filled", judgement=judged("solid"))]))
    assert view.points[0].state == "not_treated"
    assert view.figure == round((4 * 10 + 4 * 100) / 8, 1)


def test_accuracy_counts_add_up_over_the_points() -> None:
    view = interview_view(offer(), live("r1", [
        slot("s1", status="filled", judgement=judged("solid"),
             accuracy={"solid": 2, "not_treated": 1}),
        slot("s2", status="filled", judgement=judged("wrong")),
    ]))
    assert view.accuracy == {"solid": 2, "wrong": 1, "not_treated": 1}


# ---- the order -------------------------------------------------------------

def test_a_person_missing_a_required_point_never_passes_an_equal_one(
) -> None:
    plan = InterviewPlan(id="cafe", title="Barista", slots=[
        Slot(id="s1", label="Tipos de café", level="minimo", weight=3,
             questions=["¿Qué cafés conoces?"]),
        Slot(id="s2", label="Inglés", level="deseable", weight=3,
             questions=["¿Qué nivel de inglés tienes?"])])
    covers = interview_view(plan, live("r1", [
        slot("s1", weight=3, status="filled", judgement=judged("unclear")),
        slot("s2", "deseable", 3, status="filled",
             judgement=judged("correct_basic"))],
        candidate="Bea", finished_at="2026-11-01T10:00:00"))
    misses = interview_view(plan, live("r2", [
        slot("s1", weight=3, status="skipped"),
        slot("s2", "deseable", 3, status="filled",
             judgement=judged("solid"))],
        candidate="Ciro", finished_at="2026-11-24T10:00:00"))
    assert covers.figure == misses.figure == 55.0
    assert [v.session_id for v in ordered([misses, covers])] == ["r1", "r2"]
    later = interview_view(plan, live("r3", [
        slot("s1", weight=3, status="skipped"),
        slot("s2", "deseable", 3, status="filled",
             judgement=judged("solid")),
    ], plan_revision=0, candidate="Dani"))
    best = interview_view(plan, ai_sheet("vb-9", slots=[
        slot("s1", weight=3, status="filled", score=90),
        slot("s2", "deseable", 3, status="filled", score=90)]))
    nothing = interview_view(plan, {"session_id": "odd", "kind": "ai",
                                    "slots": [], "plan_revision": 0})
    assert nothing.figure == 0.0  # a required point without a score: 0
    order = ordered([nothing, later, covers, best, misses])
    assert [v.session_id for v in order] == ["vb-9", "r1", "r2", "r3",
                                             "odd"]


def test_without_a_figure_a_row_goes_last_and_ties_go_by_date_then_name(
) -> None:
    plan = InterviewPlan(id="x", title="X", slots=[
        Slot(id="s1", label="Algo", level="deseable",
             questions=["¿Algo?"]),
        Slot(id="s2", label="Otro", level="minimo", questions=["¿Otro?"])])
    blank = interview_view(plan, {"session_id": "a", "plan_revision": 3,
                                  "plan_slots": ["s9"]})
    assert blank.figure is None and blank.plan_changed
    old = interview_view(plan, live("b", [], candidate="Zoe",
                                    finished_at="2026-11-01T09:00:00"))
    new = interview_view(plan, live("c", [], candidate="Abel",
                                    finished_at="2026-11-02T09:00:00"))
    same = interview_view(plan, live("d", [], candidate="Ángela",
                                     finished_at="2026-11-02T09:00:00"))
    odd = interview_view(plan, live("e", [], finished_at="mañana"))
    assert [v.session_id for v in ordered([blank, old, same, new, odd])
            ] == ["c", "d", "b", "e", "a"]


# ---- the table -------------------------------------------------------------

def test_the_board_joins_a_persons_interviews_and_orders_the_rows() -> None:
    store = Store()
    store.save(offer())
    profiles = MemoryCandidates()
    ana = new_profile("cafe", "Ana Ruiz", [], NOW)
    ana = ana.model_copy(update={"rounds": [
        Round(session_id="vb-old", at="2026-11-01T10:00:00", kind="bot")]})
    profiles.save(ana)
    store.save_result("cafe", ai_sheet("vb-old", candidate="Ana Ruiz",
                                       finished_at="2026-11-01T10:00:00"))
    store.save_result("cafe", live("r-ana", [
        slot("s1", status="filled", judgement=judged("correct_basic")),
        slot("s2", status="filled", judgement=judged("correct_basic"))],
        candidate="Ana R.", candidate_id=ana.id,
        finished_at="2026-11-24T10:00:00", attribution="rules"))
    store.save_result("cafe", ai_sheet("vb-2", candidate="Luis Gil"))
    store.save_result("cafe", {"session_id": "vb-3", "kind": "ai"})
    plans = planner(store=store)
    book = CandidateBook(profiles, Core(), lambda p: p in store.plans, 30,
                         now=lambda: NOW, plans=store)
    InterviewWiring(plans, store, FakeReasoner(), candidates=book)
    board = plans.board("cafe")
    assert board["schema"] == "interview.board.v1"
    assert board["required"] == ["s1", "s2"] and board["desirable"] == ["s3"]
    assert board["interviews"] == 4 and board["truncated"] is False
    people = board["people"]
    # The file's current name, not the one stored with the report.
    assert [p["name"] for p in people] == ["Ana Ruiz", "Luis Gil", None]
    assert [p["position"] for p in people] == [1, 2, 3]
    first = people[0]
    assert first["person"] == f"file:{ana.id}"
    assert first["latest"]["session_id"] == "r-ana"
    assert first["latest"]["figure"] == 70.0
    assert first["latest"]["less_reliable"] is True
    assert first["latest"]["owned"] is True
    assert [o["session_id"] for o in first["others"]] == ["vb-old"]
    assert people[1]["latest"]["owned"] is False
    assert people[1]["person"] == "session:vb-2"
    assert people[2]["latest"]["figure"] == 0.0
    row = plans.list()[0]
    assert (row["people"], row["interviews"]) == (3, 4)


def test_the_board_is_bounded_and_the_list_counts_each_kind(
        monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    store.save(offer())
    for i in range(4):
        store.save_result("cafe", live(f"r{i}", [],
                                       finished_at=f"2026-11-0{i + 1}T10:00"))
    store.save_result("cafe", ai_sheet("vb-1"))
    plans = planner(store=store)
    monkeypatch.setattr("orchestrator.application.interview.MAX_REPORTS", 3)
    board = plans.board("cafe")
    assert board["interviews"] == 3 and board["truncated"] is True
    row = plans.list()[0]
    assert (row["minimums"], row["desirables"]) == (2, 1)
    assert (row["interviews"], row["interviews_ai"],
            row["interviews_live"]) == (5, 1, 4)
    assert row["last_interview"] == "2026-11-20T10:00:00"


def test_without_a_book_only_a_named_file_joins_rows() -> None:
    store = Store()
    store.save(offer())
    store.save_result("cafe", live("r1", [], candidate_id="eva"))
    store.save_result("cafe", live("r2", [], candidate_id="eva"))
    store.save_result("cafe", live("r3", []))
    people = planner(store=store).board("cafe")["people"]
    assert sorted(p["person"] for p in people) == ["file:eva",
                                                   "session:r3"]
