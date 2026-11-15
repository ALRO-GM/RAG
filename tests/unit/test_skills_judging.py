"""Judges: quota status, label comparison, dialogue-act prompt."""

from datetime import datetime, timedelta, timezone

import pytest

from src.domain.runtime import ModelOption, largest_fitting, model_memory_gb
from src.domain.skills.act_prompting import (
    ACT_IDS,
    act_system_prompt,
    act_user_prompt,
    parse_act,
)
from src.domain.skills.judging import JudgeState, compare_labels, quota_status

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


def test_quota_is_unknown_until_claude_is_called() -> None:
    assert quota_status(JudgeState(), NOW) == "unknown"
    assert quota_status(JudgeState(quota_ok_at=NOW), NOW) == "available"


def test_a_recent_refusal_means_exhausted_until_an_answer_or_time() -> None:
    refused = NOW - timedelta(hours=1)
    state = JudgeState(quota_exhausted_at=refused)
    assert quota_status(state, NOW) == "exhausted"
    assert quota_status(state, NOW + timedelta(hours=5)) == "available"
    answered = state.model_copy(update={"quota_ok_at": NOW})
    assert quota_status(answered, NOW) == "available"
    older = state.model_copy(update={"quota_ok_at": refused - timedelta(1)})
    assert quota_status(older, NOW) == "exhausted"


def test_identical_labels_agree_fully() -> None:
    result = compare_labels(["a", "b", "a"], ["a", "b", "a"])
    assert (result.items, result.agreement, result.kappa) == (3, 1.0, 1.0)
    assert result.per_category == {"a": 1.0, "b": 1.0}
    assert result.confusion == {"a": {"a": 2}, "b": {"b": 1}}
    assert result.accuracy_a is None


def test_kappa_corrects_for_chance_and_counts_confusions() -> None:
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    result = compare_labels(a, b)
    assert result.agreement == 0.5
    assert result.kappa == 0.0
    assert result.confusion == {"x": {"x": 1, "y": 1}, "y": {"x": 1, "y": 1}}


def test_accuracy_against_a_reference_admits_several_labels() -> None:
    result = compare_labels(
        ["x", "y", "z", "x"], ["x", "x", "x", "x"],
        [["x"], ["y", "x"], ["y"], None],
    )
    assert result.with_reference == 3
    assert result.accuracy_a == round(2 / 3, 4)
    assert result.accuracy_b == round(2 / 3, 4)
    assert result.per_category_accuracy["y"] == {
        "items": 2.0, "a": 0.5, "b": 0.5,
    }


def test_comparison_needs_aligned_items() -> None:
    with pytest.raises(ValueError):
        compare_labels(["a"], [])
    with pytest.raises(ValueError):
        compare_labels(["a"], ["a"], [])
    empty = compare_labels([], [])
    assert (empty.agreement, empty.kappa) == (None, None)
    assert compare_labels(["a"], ["a"], [None]).with_reference == 0


def test_act_prompt_lists_the_13_categories_and_reads_one() -> None:
    system = act_system_prompt()
    assert all(f"- {act}:" in system for act in ACT_IDS)
    assert len(ACT_IDS) == 13
    user = act_user_prompt([("Ana", "¿Vienes?")], "Luis", "Sí.")
    assert "Ana: ¿Vienes?" in user and "dicha por Luis:\nSí." in user
    assert "(inicio)" in act_user_prompt([], "Luis", "Hola.")
    assert parse_act('{"act": "respuesta"}') == "respuesta"
    assert parse_act('Vale: {"act": " Opinion "} fin') == "opinion"
    assert parse_act("escucha") == "escucha"
    assert parse_act('{"act": "saludo"}') is None
    assert parse_act("{no json}") is None
    assert parse_act('["respuesta"]') is None


def test_quantised_models_need_less_gpu_memory() -> None:
    full = ModelOption(name="m", params_billion=8.0)
    four = ModelOption(name="m4", params_billion=8.0, bits=4)
    assert model_memory_gb(four, "cuda") == pytest.approx(8 * 0.5 * 1.3)
    assert model_memory_gb(four, "cpu") == model_memory_gb(full, "cpu")
    ladder = [ModelOption(name=n, params_billion=b, bits=4)
              for n, b in (("s", 1.7), ("m", 4.0), ("l", 8.2))]
    assert largest_fitting(ladder, "cuda", 3.0) == ladder[1]
    assert largest_fitting(ladder, "cuda", 6.0) == ladder[2]
    assert largest_fitting(ladder, "cuda", 1.0) is None
