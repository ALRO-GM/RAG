"""Agreement statistics against published reference values."""

from typing import List, Optional

import pytest

from src.domain.skills.agreement import (
    Rating,
    cohen_kappa,
    icc_two_way,
    interpret_alpha,
    krippendorff_alpha,
    mean_absolute_error,
    paired,
    percent_agreement,
    set_scores,
)

N = None
KRIPPENDORFF: List[List[Rating]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
]
"""Krippendorff's 4 observers × 12 units example (rows are raters)."""
FIVE = [1.0, 2.0, 3.0, 4.0, 5.0]
SHROUT_FLEISS = [
    [9.0, 6, 8, 7, 10, 6], [2.0, 1, 4, 1, 5, 2],
    [5.0, 3, 6, 2, 6, 4], [8.0, 2, 8, 6, 9, 7],
]
"""Shrout & Fleiss (1979): 4 judges (rows) rating 6 targets."""


# ------------------------------------------------------- pairwise basics


def test_paired_keeps_units_rated_by_both() -> None:
    """Units with a missing rating on either side are dropped."""
    assert paired([1, None, 3, 4], [1, 2, None, 5]) == ([1.0, 4.0],
                                                        [1.0, 5.0])
    assert paired([], []) == ([], [])
    assert paired([None], [2]) == ([], [])


def test_paired_stops_at_the_shorter_sequence() -> None:
    """Sequences of different length are compared on their overlap."""
    assert paired([1, 2, 3], [1]) == ([1.0], [1.0])


def test_percent_agreement_exact_and_within_tolerance() -> None:
    """Exact agreement and agreement within one point."""
    a: List[Rating] = [1, 2, 3, 5, None]
    b: List[Rating] = [1, 3, 5, 5, 4]
    assert percent_agreement(a, b) == pytest.approx(0.5)
    assert percent_agreement(a, b, tolerance=1.0) == pytest.approx(0.75)
    assert percent_agreement(a, b, tolerance=2.0) == pytest.approx(1.0)


def test_percent_agreement_and_mae_without_co_rated_units() -> None:
    """Nothing in common gives ``None``, not zero."""
    assert percent_agreement([1, None], [None, 2]) is None
    assert mean_absolute_error([1, None], [None, 2]) is None
    assert percent_agreement([], []) is None


def test_mean_absolute_error() -> None:
    """The mean distance over co-rated units."""
    assert mean_absolute_error([1, 2, 5, None], [1, 4, 1, 3]) == (
        pytest.approx(2.0)
    )
    assert mean_absolute_error([3, 3], [3, 3]) == 0.0


# ---------------------------------------------------------- Cohen kappa


def test_cohen_kappa_textbook_two_by_two() -> None:
    """Observed 0.70, expected 0.50 → kappa 0.40."""
    a = [1.0] * 25 + [0.0] * 25
    b = [1.0] * 20 + [0.0] * 5 + [1.0] * 10 + [0.0] * 15
    assert cohen_kappa(a, b, [0, 1], "none") == pytest.approx(0.4)


def test_cohen_kappa_perfect_agreement_and_disagreement() -> None:
    """Identical raters score 1; systematically opposite ones score -1."""
    a: List[Rating] = [1, 2, 1, 2, 3, 3]
    assert cohen_kappa(a, a, [1, 2, 3], "none") == pytest.approx(1.0)
    assert cohen_kappa(a, a, [1, 2, 3], "linear") == pytest.approx(1.0)
    assert cohen_kappa(a, a, [1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa([1, 2, 1, 2], [2, 1, 2, 1], [1, 2], "none") == (
        pytest.approx(-1.0)
    )


def test_cohen_kappa_weights_order_near_misses() -> None:
    """One-point misses cost less with linear and quadratic weights."""
    a: List[Rating] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    b: List[Rating] = [2, 2, 3, 4, 4, 1, 3, 3, 5, 5]
    nominal = cohen_kappa(a, b, FIVE, "none")
    linear = cohen_kappa(a, b, FIVE, "linear")
    quadratic = cohen_kappa(a, b, FIVE, "quadratic")
    assert nominal is not None and linear is not None
    assert quadratic is not None
    assert nominal < linear < quadratic < 1.0
    assert cohen_kappa(a, b, FIVE) == quadratic


def test_cohen_kappa_quadratic_reference_value() -> None:
    """A hand-computed 3-category weighted kappa."""
    a: List[Rating] = [1, 2, 3, 1]
    b: List[Rating] = [1, 2, 3, 2]
    assert cohen_kappa(a, b, [1, 2, 3], "linear") == pytest.approx(5 / 7)
    assert cohen_kappa(a, b, [1, 2, 3], "quadratic") == pytest.approx(0.8)


def test_cohen_kappa_is_symmetric() -> None:
    """Swapping the raters does not change kappa."""
    a: List[Rating] = [1, 2, 3, 3, 5, None, 2]
    b: List[Rating] = [1, 3, 3, 4, 4, 2, None]
    assert cohen_kappa(a, b, FIVE) == pytest.approx(
        cohen_kappa(b, a, FIVE) or 0.0
    )


def test_cohen_kappa_without_co_rated_units() -> None:
    """No unit in common gives ``None``."""
    assert cohen_kappa([], [], FIVE) is None
    assert cohen_kappa([1, None], [None, 1], FIVE) is None


def test_cohen_kappa_single_category() -> None:
    """One shared category is full agreement; two constants are chance."""
    assert cohen_kappa([2, 2, 2], [2, 2, 2], FIVE) == 1.0
    assert cohen_kappa([2, 2, 2], [2, 2, 2], FIVE, "none") == 1.0
    assert cohen_kappa([1, 1], [2, 2], [1, 2], "none") == pytest.approx(0.0)


def test_cohen_kappa_ignores_units_one_rater_skipped() -> None:
    """Missing ratings do not count as disagreements."""
    a: List[Rating] = [1, 2, None, 2, 1]
    b: List[Rating] = [1, 2, 5, 2, 1]
    assert cohen_kappa(a, b, FIVE) == pytest.approx(1.0)


# ------------------------------------------------- Krippendorff's alpha


@pytest.mark.parametrize("level, expected", [
    ("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849),
])
def test_krippendorff_alpha_reference_example(
    level: str, expected: float
) -> None:
    """Krippendorff's worked example at three measurement levels."""
    alpha = krippendorff_alpha(KRIPPENDORFF, FIVE, level)
    assert alpha == pytest.approx(expected, abs=0.0005)


def test_krippendorff_alpha_defaults_to_ordinal() -> None:
    """The default level is the one for rating scales."""
    assert krippendorff_alpha(KRIPPENDORFF, FIVE) == krippendorff_alpha(
        KRIPPENDORFF, FIVE, "ordinal"
    )


def test_krippendorff_alpha_does_not_depend_on_rater_order() -> None:
    """Shuffling the raters leaves alpha unchanged."""
    assert krippendorff_alpha(KRIPPENDORFF[::-1], FIVE) == pytest.approx(
        krippendorff_alpha(KRIPPENDORFF, FIVE) or 0.0
    )


def test_krippendorff_alpha_perfect_agreement() -> None:
    """Identical rows with some variety give 1."""
    rows: List[List[Rating]] = [[1, 2, 3, 5], [1, 2, 3, 5], [1, 2, 3, 5]]
    for level in ("nominal", "ordinal", "interval"):
        assert krippendorff_alpha(rows, FIVE, level) == pytest.approx(1.0)


def test_krippendorff_alpha_single_category_is_one() -> None:
    """Every rating equal: no variation to disagree on, by convention 1."""
    assert krippendorff_alpha([[3, 3, 3], [3, 3, 3]], FIVE) == 1.0


def test_krippendorff_alpha_systematic_disagreement_is_negative() -> None:
    """Raters who always swap the two ends score below zero."""
    rows: List[List[Rating]] = [[1, 5, 1, 5], [5, 1, 5, 1]]
    for level in ("nominal", "ordinal", "interval"):
        alpha = krippendorff_alpha(rows, FIVE, level)
        assert alpha is not None and alpha < -0.5


def test_krippendorff_alpha_no_unit_rated_twice() -> None:
    """Without a unit rated by two raters alpha is not computable."""
    empty: List[List[Rating]] = []
    assert krippendorff_alpha(empty, FIVE) is None
    assert krippendorff_alpha([[]], FIVE) is None
    assert krippendorff_alpha([[1, 2, 3]], FIVE) is None
    assert krippendorff_alpha([[1, None], [None, 2]], FIVE) is None
    assert krippendorff_alpha([[None, None], [None, None]], FIVE) is None


def test_krippendorff_alpha_ignores_units_rated_once() -> None:
    """A unit with one rating adds nothing."""
    base: List[List[Rating]] = [[1, 2, 3, 3], [1, 2, 3, 4]]
    more: List[List[Rating]] = [[1, 2, 3, 3, 5], [1, 2, 3, 4, None]]
    assert krippendorff_alpha(more, FIVE) == pytest.approx(
        krippendorff_alpha(base, FIVE) or 0.0
    )


def test_krippendorff_ordinal_penalises_far_misses_more() -> None:
    """A two-point miss lowers ordinal alpha more than a one-point one."""
    near: List[List[Rating]] = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 4]]
    far: List[List[Rating]] = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 3]]
    near_alpha = krippendorff_alpha(near, FIVE, "ordinal")
    far_alpha = krippendorff_alpha(far, FIVE, "ordinal")
    assert near_alpha is not None and far_alpha is not None
    assert far_alpha < near_alpha < 1.0
    assert krippendorff_alpha(near, FIVE, "nominal") == pytest.approx(
        krippendorff_alpha(far, FIVE, "nominal") or 0.0
    )


def test_statistics_expect_ratings_among_the_categories() -> None:
    """Off-category values are a caller error (use cases check first)."""
    with pytest.raises(KeyError):
        krippendorff_alpha([[1, 9], [1, 2]], [1.0, 2.0, 3.0])
    with pytest.raises(KeyError):
        cohen_kappa([1, 9], [1, 2], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ ICC


def test_icc_shrout_fleiss_reference() -> None:
    """ICC(2,1) = 0.290 and ICC(2,k) = 0.620 on the classic table."""
    result = icc_two_way(SHROUT_FLEISS)
    assert result is not None
    single, average = result
    assert single == pytest.approx(0.290, abs=0.0005)
    assert average == pytest.approx(0.620, abs=0.0005)


def test_icc_perfect_agreement() -> None:
    """Identical raters on varied units give 1 and 1."""
    result = icc_two_way([[1.0, 3, 5, 2], [1.0, 3, 5, 2], [1.0, 3, 5, 2]])
    assert result == pytest.approx((1.0, 1.0))


def test_icc_penalises_a_constant_offset() -> None:
    """Absolute agreement: a rater always one point higher lowers ICC."""
    result = icc_two_way([[1.0, 2, 3, 4], [2.0, 3, 4, 5]])
    assert result is not None
    assert 0.0 < result[0] < 1.0
    assert result[0] < result[1] < 1.0


@pytest.mark.parametrize("ratings", [
    [], [[1.0, 2.0, 3.0]], [[1.0], [2.0]], [[], []],
    [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
])
def test_icc_not_computable(ratings: List[List[float]]) -> None:
    """One rater, one unit, no data or no variance give ``None``."""
    assert icc_two_way(ratings) is None


def test_icc_opposite_raters_is_negative() -> None:
    """Raters who rank the units in reverse do worse than chance."""
    result = icc_two_way([[1.0, 2, 3, 4, 5], [5.0, 4, 3, 2, 1]])
    assert result is not None and result[0] < 0


# ----------------------------------------------------------- set scores


def test_set_scores_partial_overlap() -> None:
    """Precision 2/3, recall 2/4 and their harmonic mean."""
    scores = set_scores({1, 2, 9}, {1, 2, 3, 4})
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(4 / 7)


def test_set_scores_edge_cases() -> None:
    """Two empty sets agree; one empty side or no overlap scores 0."""
    perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    nothing = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert set_scores(set(), set()) == perfect
    assert set_scores({1, 2}, {1, 2}) == perfect
    assert set_scores(set(), {1}) == nothing
    assert set_scores({1}, set()) == nothing
    assert set_scores({1}, {2}) == nothing


@pytest.mark.parametrize("alpha, band", [
    (None, "not computable"), (1.0, "reliable"), (0.80, "reliable"),
    (0.7999, "tentative"), (0.667, "tentative"), (0.6669, "unreliable"),
    (0.0, "unreliable"), (-0.5, "unreliable"),
])
def test_interpret_alpha_bands(alpha: Optional[float], band: str) -> None:
    """Krippendorff's 0.80 and 0.667 thresholds name the bands."""
    assert interpret_alpha(alpha) == band
