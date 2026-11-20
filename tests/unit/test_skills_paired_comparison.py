"""Unit tests for the paired comparison of two taggers."""

import pytest

from src.domain.skills.paired_comparison import (
    mcnemar_exact,
    paired_loss_gap,
)


def test_mcnemar_counts_and_exact_p() -> None:
    """Eight wins against two losses give the known exact p-value."""
    cand = [True] * 8 + [False] * 2 + [True] * 5
    base = [False] * 8 + [True] * 2 + [True] * 5
    got = mcnemar_exact(cand, base)
    assert (got.only_candidate, got.only_baseline) == (8, 2)
    assert got.p_value == pytest.approx(0.109375)


def test_mcnemar_no_discordant_pairs() -> None:
    """Identical hits cannot be told apart."""
    assert mcnemar_exact([True, False], [True, False]).p_value == 1.0


def test_mcnemar_rejects_different_lengths() -> None:
    """The sentences must be the same."""
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_loss_gap_sign_and_interval() -> None:
    """A candidate that is always better has a positive interval."""
    mean, low, high = paired_loss_gap([0.1] * 20, [0.5] * 20)
    assert mean == pytest.approx(0.4)
    assert low > 0 and high >= low


def test_loss_gap_rejects_empty() -> None:
    """Nothing to compare is an error."""
    with pytest.raises(ValueError):
        paired_loss_gap([], [])
