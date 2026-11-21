"""The interview plan and sheet (``interview.plan.v1``, #208)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jsonschema
import pytest

from orchestrator.domain.interview import (
    COMPETENCIES,
    InterviewPlan,
    PlannerInfo,
    PlanSource,
    Sheet,
    SheetSlot,
    StaleRevision,
    counts,
    draft_slots,
    edited_plan,
    fallback_plan,
    is_protected,
    plan_from_draft,
    suitability,
)

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 21, 10, 0, 0)


def draft(slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"title": "Dependiente/a", "role_type": "comercio",
            "summary": "Tienda de deportes.", "slots": slots}


def item(label: str, level: str = "minimo", **more: Any) -> Dict[str, Any]:
    return {"label": label, "level": level,
            "questions": [f"¿Qué experiencia tienes en {label}?"],
            "evidence": "Dónde y cuánto tiempo.", **more}


def make(slots: List[Dict[str, Any]]) -> InterviewPlan:
    return plan_from_draft(draft(slots), "plan-1", PlanSource(), 20, "es",
                           NOW, PlannerInfo(model="fake"))


def test_the_rubric_competencies_are_the_ones_of_the_rubric_file() -> None:
    rubric = json.loads((ROOT / "knowledge" / "rubrics" / "core.es.json")
                        .read_text("utf-8"))
    assert [(c["id"], c["name"]) for c in rubric["competencies"]] == list(
        COMPETENCIES)


def test_a_draft_becomes_a_plan_with_levels_weights_and_competencies(
) -> None:
    plan = make([
        item("atención en tienda", "Imprescindible", weight=5),
        item("turnos de sábado", "deseable", weight=9),
        item("Comunicación", "opcional", competency="comm", weight=0),
        item("algo raro", "???"),
    ])
    assert [s.id for s in plan.slots] == ["s1", "s2", "s3", "s4"]
    assert [s.level for s in plan.slots] == [
        "minimo", "deseable", "opcional", "deseable"]
    assert [s.weight for s in plan.slots] == [5, 5, 1, 3]
    assert plan.slots[2].kind == "competency"
    assert plan.slots[2].competency == "COMM"
    assert plan.slots[2].phase == "behavioural"
    assert plan.role_type == "comercio" and plan.title == "Dependiente/a"


def test_protected_slots_and_questions_are_dropped() -> None:
    slots, dropped = draft_slots([
        item("Edad y estado civil"),
        {"label": "Experiencia", "questions": [
            "¿Tienes hijos?", "¿Cuántos años has trabajado en caja?"]},
        "not a slot",
        {"label": "Sin preguntas", "questions": []},
    ])
    assert dropped == 3
    assert [s.questions for s in slots] == [
        ["¿Cuántos años has trabajado en caja?"]]
    assert slots[0].level == "minimo"  # one minimo at least
    assert is_protected("¿Estás embarazada?")
    assert not is_protected("¿Qué turnos puedes hacer?")


def test_a_draft_without_usable_slots_is_refused() -> None:
    with pytest.raises(ValueError):
        make([item("Religión que practicas")])


def test_the_fallback_plan_is_valid_and_has_minimums() -> None:
    plan = fallback_plan("plan-2", "Mozo de almacén", PlanSource(), 200,
                         "es", NOW, "timeout", "almacen")
    assert plan.minutes == 90 and plan.role_type == "almacen"
    assert plan.planner.status == "fallback"
    assert plan.planner.error == "timeout"
    assert any(s.level == "minimo" for s in plan.slots)
    assert plan.ordered()[0].level == "minimo"


def test_an_edit_needs_the_current_revision_and_one_minimo() -> None:
    plan = make([item("caja"), item("reposición", "deseable")])
    document = plan.to_contract()
    document["slots"][1]["level"] = "minimo"
    document["slots"][1]["weight"] = 2
    document["minutes"] = 30
    document["id"] = "plan-other"
    edited = edited_plan(plan, document, NOW)
    assert edited.id == "plan-1" and edited.revision == 1
    assert edited.minutes == 30 and edited.slots[1].weight == 2
    with pytest.raises(StaleRevision):
        edited_plan(edited, document, NOW)
    bad = edited.to_contract()
    for slot in bad["slots"]:
        slot["level"] = "opcional"
    with pytest.raises(ValueError):
        edited_plan(edited, bad, NOW)
    bad = edited.to_contract()
    bad["slots"][0]["questions"] = ["¿Qué religión tienes?"]
    with pytest.raises(ValueError):
        edited_plan(edited, bad, NOW)


def test_suitability_weighs_scores_and_counts_missing_minimums() -> None:
    slots = [
        SheetSlot(id="s1", label="a", level="minimo", weight=4, score=100,
                  status="filled"),
        SheetSlot(id="s2", label="b", level="minimo", weight=1,
                  status="empty"),
        SheetSlot(id="s3", label="c", level="opcional", weight=5,
                  status="empty"),
        SheetSlot(id="s4", label="d", level="deseable", weight=5, score=40,
                  status="unknown", above=False),
    ]
    assert suitability(slots) == pytest.approx((400 + 0 + 200) / 10)
    assert counts(slots) == {"meets": 1, "gaps": 2, "above": 0,
                             "unknown": 1, "skipped": 0}
    assert suitability([]) is None


@pytest.mark.parametrize("name,model", [
    ("interview.plan.v1", InterviewPlan), ("interview.sheet.v1", Sheet),
])
def test_contract_examples_validate(name: str, model: Any) -> None:
    schema = json.loads((ROOT / "contracts" / f"{name}.schema.json")
                        .read_text("utf-8"))
    example = json.loads((ROOT / "contracts" / "examples" / f"{name}.json")
                         .read_text("utf-8"))
    jsonschema.validate(example, schema)
    assert model.model_validate(example).to_contract()["schema"] == name
