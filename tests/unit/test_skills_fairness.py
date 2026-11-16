"""Group tables of a fairness audit, on simulated data (#86)."""

import math
from typing import List

import pytest

from src.domain.skills.fairness import (
    AuditPolicy,
    Measurement,
    Outcome,
    above_median,
    mean_audit,
    rate_audit,
    ratio_interval,
    swap_changes,
    wilson_interval,
)


def people(group: str, selected: int, n: int) -> List[Outcome]:
    """Return ``n`` people of ``group``, the first ``selected`` selected."""
    return [Outcome(unit=f"{group}{i}", group=group, selected=i < selected)
            for i in range(n)]


def test_wilson_interval_matches_the_textbook_value() -> None:
    """5 of 10 at 95 %: (0.2366, 0.7634) (Wilson, 1927)."""
    low, high = wilson_interval(5, 10)
    assert (round(low, 4), round(high, 4)) == (0.2366, 0.7634)


def test_wilson_interval_at_the_edges_stays_in_bounds() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert 0 < wilson_interval(0, 10)[1] < 0.3


@pytest.mark.parametrize("successes,n", [(1, 0), (-1, 5), (6, 5)])
def test_wilson_interval_refuses_impossible_counts(
    successes: int, n: int
) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


def test_ratio_interval_is_the_katz_log_interval() -> None:
    """0.30 / 0.50: exp(ln 0.6 ± 1.96 √(1/30 − 1/100 + 1/50 − 1/100))."""
    low, high = ratio_interval(30, 100, 50, 100) or (0.0, 0.0)
    spread = 1.959964 * math.sqrt(1 / 30 - 1 / 100 + 1 / 50 - 1 / 100)
    assert low == pytest.approx(0.6 * math.exp(-spread), rel=1e-5)
    assert high == pytest.approx(0.6 * math.exp(spread), rel=1e-5)


def test_ratio_interval_with_an_empty_cell_is_finite() -> None:
    interval = ratio_interval(0, 10, 5, 10)
    assert interval is not None and 0 < interval[0] < interval[1] < 2
    full = ratio_interval(10, 10, 10, 10)
    assert full is not None and full[0] < 1 < full[1]
    assert ratio_interval(3, 10, 0, 10) is None
    with pytest.raises(ValueError):
        ratio_interval(1, 0, 1, 1)
    with pytest.raises(ValueError):
        ratio_interval(4, 3, 1, 1)


def test_four_fifths_flags_the_simulated_adverse_impact() -> None:
    """A: 50 of 100, B: 30 of 100 → impact ratio 0.6 < 0.8."""
    table = rate_audit(people("A", 50, 100) + people("B", 30, 100), "sel")
    a, b = table.rows
    assert table.reference_group == "A" and table.units == 200
    assert (a.status, a.ratio, a.ratio_low) == ("reference", 1.0, None)
    assert (b.status, b.selected, b.n) == ("flagged", 30, 100)
    assert b.ratio == pytest.approx(0.6)
    assert b.ratio_high is not None and b.ratio_high > 0.8
    assert b.certain is False
    assert table.flagged() == ["B"]


def test_a_clear_gap_is_certain_and_parity_is_ok() -> None:
    clear = rate_audit(people("A", 80, 200) + people("B", 20, 200), "sel")
    assert clear.rows[1].status == "flagged" and clear.rows[1].certain
    even = rate_audit(people("A", 40, 100) + people("B", 38, 100), "sel")
    assert even.rows[1].status == "ok" and even.flagged() == []


def test_small_groups_are_described_but_get_no_ratio() -> None:
    """Under 5 people, or under 2 %, a group is only described."""
    table = rate_audit(
        people("A", 50, 100) + people("B", 45, 100) + people("C", 1, 3)
        + people("D", 0, 4), "sel",
        AuditPolicy(min_units=5),
    )
    rows = {r.group: r for r in table.rows}
    assert rows["C"].status == "described_only" and rows["C"].ratio is None
    assert rows["C"].value == pytest.approx(1 / 3)
    assert rows["D"].low == 0.0 and (rows["D"].high or 0) > 0
    tiny = rate_audit(people("A", 50, 300) + people("B", 5, 7), "sel",
                      AuditPolicy(min_units=1, min_share=0.02))
    assert tiny.reference_group == "B"
    tinier = rate_audit(people("A", 50, 300) + people("B", 5, 5), "sel",
                        AuditPolicy(min_units=1, min_share=0.02))
    assert tinier.rows[1].status == "described_only"
    assert tinier.reference_group == "A"


def test_lower_is_better_takes_the_lowest_rate_as_reference() -> None:
    """Misattribution rates: 10 % against 20 % is a ratio of 2."""
    table = rate_audit(
        people("F", 20, 100) + people("M", 10, 100), "wrong",
        AuditPolicy(direction="lower_is_better", threshold=1.25),
    )
    assert table.reference_group == "M"
    assert table.rows[0].ratio == pytest.approx(2.0)
    assert table.rows[0].status == "flagged"


def test_a_reference_rate_of_zero_gives_no_ratio() -> None:
    table = rate_audit(people("A", 0, 10) + people("B", 0, 10), "sel")
    assert all(r.ratio is None for r in table.rows)
    assert table.rows[0].status == "reference"


def test_no_eligible_group_means_no_reference() -> None:
    table = rate_audit(people("A", 1, 2) + people("B", 1, 3), "sel")
    assert table.reference_group is None
    assert {r.status for r in table.rows} == {"described_only"}


def test_rate_audit_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        rate_audit([], "sel")
    with pytest.raises(ValueError):
        rate_audit(people("A", 1, 2) + people("A", 1, 1), "sel")


def test_mean_audit_counts_a_person_once_and_flags_the_gap() -> None:
    """F errors around 0.10, M around 0.20; F0 measured three times."""
    measurements = [
        Measurement(unit=f"F{i}", group="F", value=0.08 + 0.01 * i)
        for i in range(5)
    ] + [
        Measurement(unit=f"M{i}", group="M", value=0.18 + 0.01 * i)
        for i in range(5)
    ] + [Measurement(unit="F0", group="F", value=0.08)] * 2
    table = mean_audit(measurements, "speaker_error")
    f, m = table.rows
    assert table.units == 10 and (f.n, m.n) == (5, 5)
    assert f.value == pytest.approx(0.10) and m.value == pytest.approx(0.20)
    assert f.low is not None and f.high is not None
    assert f.low <= f.value <= f.high
    assert table.reference_group == "F"
    assert m.ratio == pytest.approx(2.0) and m.status == "flagged"
    assert m.ratio_low is not None and m.ratio_low > 1.25 and m.certain
    again = mean_audit(measurements, "speaker_error")
    assert again == table


def test_mean_audit_higher_is_better_and_zero_reference() -> None:
    up = mean_audit(
        [Measurement(unit=f"{g}{i}", group=g, value=v + 0.01 * i)
         for g, v in (("A", 4.0), ("B", 3.5)) for i in range(5)],
        "score", AuditPolicy(direction="higher_is_better"),
    )
    assert up.reference_group == "A" and up.policy.threshold == 0.80
    assert up.rows[1].ratio == pytest.approx(3.52 / 4.02)
    assert up.rows[1].status == "ok" and up.rows[1].certain
    zero = mean_audit(
        [Measurement(unit=f"{g}{i}", group=g, value=0.0)
         for g in "AB" for i in range(5)], "error",
    )
    assert all(r.ratio is None for r in zero.rows)
    assert zero.rows[1].status == "ok" and zero.zero_reference() == []


def test_mean_audit_a_zero_reference_is_never_within_the_threshold() -> None:
    """A errs 0.0 for all, B 0.3: no ratio, but B is not 'ok' either."""
    table = mean_audit(
        [Measurement(unit=f"{g}{i}", group=g, value=v)
         for g, v in (("A", 0.0), ("B", 0.3)) for i in range(5)], "error",
    )
    a, b = table.rows
    assert table.reference_group == "A" and a.status == "reference"
    assert (b.status, b.ratio) == ("zero_reference", None)
    assert b.difference == pytest.approx(0.3)
    assert table.zero_reference() == ["B"] and table.flagged() == []


def test_rate_audit_a_zero_reference_is_never_within_the_threshold() -> None:
    """Misattribution A 0/10, B 5/10, lower is better: B has no ratio."""
    table = rate_audit(people("A", 0, 10) + people("B", 5, 10), "wrong",
                       AuditPolicy(direction="lower_is_better"))
    assert table.policy.threshold == 1.25
    b = table.rows[1]
    assert (b.status, b.ratio) == ("zero_reference", None)
    assert b.difference == 0.5
    assert table.zero_reference() == ["B"]


def test_mean_audit_gives_no_interval_without_two_distinct_values() -> None:
    """One person (or identical values): no interval, nor for the ratio."""
    table = mean_audit(
        [Measurement(unit="a0", group="A", value=0.1)]
        + [Measurement(unit=f"b{i}", group="B", value=0.2 + 0.01 * i)
           for i in range(5)],
        "error", AuditPolicy(direction="lower_is_better", min_units=1),
    )
    a, b = table.rows
    assert (a.low, a.high) == (None, None)
    assert b.low is not None and b.high is not None and b.low < b.high
    assert b.ratio == pytest.approx(2.2) and b.ratio_low is None
    assert b.certain is None


def test_policy_threshold_follows_the_direction() -> None:
    assert AuditPolicy().threshold == 0.80
    assert AuditPolicy(direction="lower_is_better").threshold == 1.25
    assert AuditPolicy(direction="lower_is_better",
                       threshold=1.5).threshold == 1.5
    with pytest.raises(ValueError):
        AuditPolicy(direction="lower_is_better", threshold=0.8)
    with pytest.raises(ValueError):
        AuditPolicy(direction="higher_is_better", threshold=1.25)


def test_mean_audit_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        mean_audit([], "e")
    with pytest.raises(ValueError):
        mean_audit([Measurement(unit="x", group="A", value=1.0),
                    Measurement(unit="x", group="B", value=1.0)], "e")
    with pytest.raises(ValueError):
        mean_audit([Measurement(unit="x", group="A", value=math.nan)], "e")


def test_above_median_is_the_ll144_scoring_rate() -> None:
    """Median of 1..6 is 3.5: 4, 5 and 6 are above it."""
    scores = [Measurement(unit=f"p{v}", group="AB"[v % 2], value=float(v))
              for v in range(1, 7)]
    outcomes = above_median(scores)
    assert [o.selected for o in outcomes] == [False, False, False,
                                              True, True, True]
    assert above_median([]) == []


def test_swap_changes_counts_changes_against_the_reference_variant() -> None:
    labels = {
        ("b1", "f"): "pregunta", ("b1", "m"): "pregunta",
        ("b1", "c"): "opinion",
        ("b2", "f"): "rechazo", ("b2", "m"): None, ("b2", "c"): "rechazo",
        ("b3", "m"): "otro",
    }
    c, m = swap_changes(labels, "f")
    assert (c.variant, c.pairs, c.changed, c.bases) == ("c", 2, 1, ["b1"])
    assert (m.variant, m.pairs, m.changed, m.bases) == ("m", 2, 1, ["b2"])
    assert m.rate == 0.5 and m.low < 0.5 < m.high
    assert swap_changes({("b1", "m"): "x"}, "f") == []
