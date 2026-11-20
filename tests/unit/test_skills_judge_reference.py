"""Scores of a judge on the indicator reference (#85)."""

from typing import List, Optional

from src.domain.skills.judge_reference import (
    Order,
    Outcome,
    ReferenceItem,
    agreement,
    position_bias,
    score,
)

NEGATIVES = {"L3"}


def item(item_id: str, expect: Optional[str]) -> ReferenceItem:
    """A reference item showing ``expect`` (or nothing)."""
    return ReferenceItem(
        id=item_id, offered=["listening"], speaker="A", text="hola",
        expect=expect, kind="wrong" if expect is None else "true",
    )


def outcome(item_id: str, named: Optional[str], order: Order = "forward",
            parsed: bool = True, first: str = "L1") -> Outcome:
    """An outcome with the indicator listed first set to ``first``."""
    return Outcome(item_id=item_id, order=order, named=named,
                   parsed=parsed, first_listed=first)


ITEMS = [item("a", "L1"), item("b", "L3"), item("c", None), item("d", None)]


def test_score_counts_precision_recall_and_false_confirmations() -> None:
    """Confirmations are right only when they name the intended one."""
    outcomes: List[Outcome] = [
        outcome("a", "L1"),   # right
        outcome("b", "L2"),   # a different indicator than intended
        outcome("c", "L1"),   # confirmed what is not there
        outcome("d", None),
    ]
    figures = score(ITEMS, outcomes, NEGATIVES)
    assert figures["confirmed"] == 3
    assert figures["correct"] == 1
    assert figures["precision"] == 0.333
    assert figures["recall"] == 0.5
    assert figures["false_confirm_rate"] == 0.5
    assert figures["wrong_indicator_on_true"] == 1
    assert figures["negative_recall"] == 0.0
    assert figures["first_listed_rate"] == 0.667


def test_score_negative_recall_and_unparsed() -> None:
    """Unreadable replies are counted apart and change no rate."""
    outcomes = [
        outcome("a", None, parsed=False),
        outcome("b", "L3"),
        outcome("c", None),
        outcome("d", None),
    ]
    figures = score(ITEMS, outcomes, NEGATIVES)
    assert figures["unparsed"] == 1
    assert figures["negative_recall"] == 1.0
    assert figures["recall"] == 1.0
    assert figures["false_confirm_rate"] == 0.0
    assert figures["precision"] == 1.0


def test_score_without_confirmations_has_no_precision() -> None:
    """Nothing confirmed leaves precision undefined, not zero."""
    figures = score(ITEMS, [outcome("a", None)], NEGATIVES)
    assert figures["precision"] is None
    assert figures["first_listed_rate"] is None
    assert figures["false_confirm_rate"] is None


def test_position_bias_counts_answers_that_flip() -> None:
    """An answer that changes with the order is a flip."""
    forward = [outcome("a", "L1"), outcome("b", "L3"),
               outcome("c", None, parsed=False)]
    reverse = [outcome("a", "L1", "reversed"), outcome("b", "L2", "reversed"),
               outcome("c", None, "reversed")]
    figures = position_bias(forward, reverse)
    assert figures == {"compared": 2, "flips": 1, "flip_rate": 0.5}
    assert position_bias([], [])["flip_rate"] is None


def test_agreement_between_two_judges() -> None:
    """Same answer counts indicator identity, kappa the yes or no."""
    a = [outcome("a", "L1"), outcome("b", "L3"), outcome("c", None),
         outcome("d", "L1")]
    b = [outcome("a", "L1"), outcome("b", "L1"), outcome("c", None),
         outcome("d", None)]
    figures = agreement(a, b)
    assert figures["compared"] == 4
    assert figures["same_answer"] == 0.5
    assert figures["kappa"] == 0.5
    assert agreement(a, [])["kappa"] is None


def test_agreement_kappa_is_undefined_for_a_judge_that_never_varies() -> None:
    """A judge confirming everything leaves kappa meaningless."""
    always = [outcome("a", "L1"), outcome("b", "L1"), outcome("c", "L1")]
    varied = [outcome("a", "L1"), outcome("b", None), outcome("c", "L1")]
    figures = agreement(always, varied)
    assert figures["same_answer"] == 0.667
    assert figures["kappa"] is None
