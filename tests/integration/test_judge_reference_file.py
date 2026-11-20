"""The committed indicator reference is consistent with the rubric (#85)."""

import json
from pathlib import Path

from src.domain.skills.judge_reference import ReferenceSet
from src.domain.skills.rubric import RubricCatalogue

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/skills/judge_indicators/reference.json"
RUBRIC = ROOT / "knowledge/rubrics/core.es.json"


def load() -> ReferenceSet:
    """Read the reference file."""
    return ReferenceSet.model_validate(
        json.loads(REFERENCE.read_text(encoding="utf-8"))
    )


def test_reference_is_balanced_and_covers_negative_indicators() -> None:
    """Half the items show an indicator, half only look like one."""
    items = load().items
    true = [i for i in items if i.expect]
    wrong = [i for i in items if not i.expect]
    assert len(items) == 120 and len(true) == len(wrong) == 60
    assert len({i.id for i in items}) == 120
    assert sum(1 for i in true if ".N" in str(i.expect)) >= 15
    assert all(i.kind == "wrong" and i.resembles for i in wrong)


def test_every_indicator_belongs_to_an_offered_competency() -> None:
    """The intended indicator can be named from what is offered."""
    rubric = RubricCatalogue.model_validate(
        json.loads(RUBRIC.read_text(encoding="utf-8"))
    )
    for item in load().items:
        offered = [rubric.competency(c) for c in item.offered]
        if item.expect:
            assert any(c.indicator(item.expect) for c in offered), item.id
        if item.resembles:
            assert any(c.indicator(item.resembles) for c in offered), item.id
