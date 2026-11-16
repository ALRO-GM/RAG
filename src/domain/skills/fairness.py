"""Group comparisons of a fairness audit (#86, ``FAIRNESS.md`` §4.3).

An audit asks whether the system's outcomes or errors fall more on one
group of people than on another. Two kinds of table are computed here,
both with one row per group (n, rate or mean, confidence interval, ratio
to the reference group and the ratio's interval):

- **rate** (``rate_audit``): each person is selected or not (above a
  cut, assessed rather than abstained, misattributed...). The rate per
  group has a Wilson score interval; the ratio to the group with the
  best rate (NYC LL144's *impact ratio*; the EEOC four-fifths rule
  flags a ratio under 0.80) has a Katz log interval. For scores without
  a cut, ``above_median`` turns them into LL144's *scoring rate* (share
  of the group above the median of the whole sample).
- **mean** (``mean_audit``): each person has a number (a speaker error,
  a word error rate...). Repeated measurements of one person are
  averaged first, so a person counts once; the group mean and the ratio
  to the best group have percentile bootstrap intervals over people.

``direction`` says which way is better: for a selection rate, higher;
for an error, lower. The reference group is the best eligible one, and
a group is flagged when its ratio is worse than ``threshold`` (0.80 for
a rate where higher is better, 1.25 for an error where lower is better;
the 1.25 is ours and provisional). Groups with fewer than
``min_units`` people or less than ``min_share`` of them are described
(n, rate, interval) but get no ratio and are never the reference
(LL144 lets the auditor leave out groups under 2 % if they are still
reported). Pure computation: no I/O, the bootstrap has a fixed seed.
"""

import math
from statistics import NormalDist
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["higher_is_better", "lower_is_better"]
AuditKind = Literal["rate", "mean"]
RowStatus = Literal["reference", "ok", "flagged", "described_only"]

FOUR_FIFTHS = 0.80
"""EEOC 29 CFR 1607.4(D): a selection rate under 4/5 of the highest is
evidence of adverse impact (an alarm, not a finding)."""

ERROR_RATIO = 1.25
"""Ours, provisional (``FAIRNESS.md`` T1/D1): a group's error over the
best group's error above this is flagged."""

MIN_SHARE = 0.02
"""NYC DCWP § 5-301: groups under 2 % of the data may be left out of
the ratios, provided their n and rate are still published."""


class Outcome(BaseModel):
    """One person's yes/no outcome and the group they declared."""

    model_config = ConfigDict(frozen=True)

    unit: str
    group: str
    selected: bool


class Measurement(BaseModel):
    """One measurement of one person (a person may have several)."""

    model_config = ConfigDict(frozen=True)

    unit: str
    group: str
    value: float


class AuditPolicy(BaseModel):
    """Rules of an audit table.

    ``threshold`` is the ratio past which a group is flagged (see the
    module); ``min_units`` and ``min_share`` decide which groups are
    only described; ``resamples`` and ``seed`` drive the bootstrap of
    mean tables.
    """

    model_config = ConfigDict(frozen=True)

    direction: Direction = "higher_is_better"
    threshold: float = Field(default=FOUR_FIFTHS, gt=0)
    min_units: int = Field(default=5, ge=1)
    min_share: float = Field(default=MIN_SHARE, ge=0, lt=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    resamples: int = Field(default=2000, ge=100)
    seed: int = 0


class AuditRow(BaseModel):
    """One group of an audit table.

    ``n`` people (``share`` of all people in the table); ``value`` is
    the rate or the mean, with ``low``/``high`` its interval;
    ``selected`` is the count behind a rate (``None`` for a mean).
    ``ratio`` is ``value`` over the reference group's, with its
    interval (none for the reference itself, whose ratio is 1);
    ``None`` for groups only described or when the reference value is
    0. ``certain`` tells whether the ratio's whole interval is
    on one side of the threshold.
    """

    model_config = ConfigDict(frozen=True)

    group: str
    n: int
    share: float
    selected: Optional[int] = None
    value: float
    low: float
    high: float
    ratio: Optional[float] = None
    ratio_low: Optional[float] = None
    ratio_high: Optional[float] = None
    status: RowStatus
    certain: Optional[bool] = None


class AuditTable(BaseModel):
    """A filled audit table: what was measured, the rules and the rows."""

    model_config = ConfigDict(frozen=True)

    metric: str
    kind: AuditKind
    policy: AuditPolicy
    units: int
    reference_group: Optional[str]
    rows: List[AuditRow]

    def flagged(self) -> List[str]:
        """Return the groups whose ratio is past the threshold."""
        return [r.group for r in self.rows if r.status == "flagged"]


def _z(confidence: float) -> float:
    """Return the two-sided normal quantile for ``confidence``."""
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Return the Wilson score interval of ``successes`` out of ``n``.

    It stays inside [0, 1] and behaves on small n and on rates of 0 or 1,
    where the normal approximation does not.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = _z(confidence)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def ratio_interval(
    a: int, n1: int, b: int, n2: int, confidence: float = 0.95
) -> Optional[Tuple[float, float]]:
    """Return the Katz log interval of the ratio (a/n1) / (b/n2).

    When a cell of the 2×2 table is empty (nobody or everybody selected
    in a group) 0.5 is added to every cell (Haldane–Anscombe), so the
    interval stays finite and not empty (it is then centred on the
    corrected ratio; the table's point ratio is not corrected).
    ``None`` when the reference rate ``b/n2`` is 0.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both groups need people")
    if not (0 <= a <= n1 and 0 <= b <= n2):
        raise ValueError("counts must be between 0 and n")
    if b == 0:
        return None
    x, m, y, k = float(a), float(n1), float(b), float(n2)
    if 0 in (a, n1 - a, b, n2 - b):
        x, m, y, k = a + 0.5, n1 + 1.0, b + 0.5, n2 + 1.0
    centre = math.log((x / m) / (y / k))
    spread = math.sqrt(1 / x - 1 / m + 1 / y - 1 / k)
    z = _z(confidence)
    return math.exp(centre - z * spread), math.exp(centre + z * spread)


def above_median(measurements: Sequence[Measurement]) -> List[Outcome]:
    """Return LL144's scoring outcome: is each person above the median?

    The median is over all people of the sample (each person's
    measurements averaged first); "above" is strictly greater.
    """
    people = _per_unit(measurements)
    if not people:
        return []
    median = float(np.median([v for _, v in people.values()]))
    return [
        Outcome(unit=unit, group=group, selected=value > median)
        for unit, (group, value) in sorted(people.items())
    ]


def _per_unit(
    measurements: Sequence[Measurement],
) -> Dict[str, Tuple[str, float]]:
    """Average each person's measurements; a person has one group."""
    groups: Dict[str, str] = {}
    values: Dict[str, List[float]] = {}
    for m in measurements:
        if not math.isfinite(m.value):
            raise ValueError(f"measurement of {m.unit!r} is not finite")
        if groups.setdefault(m.unit, m.group) != m.group:
            raise ValueError(f"{m.unit!r} is in two groups")
        values.setdefault(m.unit, []).append(m.value)
    return {
        unit: (groups[unit], sum(v) / len(v)) for unit, v in values.items()
    }


def _better(left: float, right: float, direction: Direction) -> bool:
    """Tell whether ``left`` is a better value than ``right``."""
    return left > right if direction == "higher_is_better" else left < right


def _past(ratio: float, policy: AuditPolicy) -> bool:
    """Tell whether a ratio is past the threshold (worse than allowed)."""
    if policy.direction == "higher_is_better":
        return ratio < policy.threshold
    return ratio > policy.threshold


def _certain(
    interval: Optional[Tuple[float, float]], policy: AuditPolicy
) -> Optional[bool]:
    """Tell whether the whole interval lies on one side of the threshold."""
    if interval is None:
        return None
    low, high = interval
    return not low <= policy.threshold <= high


def _eligible(n: int, total: int, policy: AuditPolicy) -> bool:
    """Tell whether a group is large enough to get a ratio."""
    return n >= policy.min_units and n / total >= policy.min_share


def _reference(
    values: Dict[str, float], eligible: Sequence[str], direction: Direction
) -> Optional[str]:
    """Return the eligible group with the best value (ties: by name)."""
    best: Optional[str] = None
    for group in sorted(eligible):
        if best is None or _better(values[group], values[best], direction):
            best = group
    return best


def _status(
    group: str, reference: Optional[str], ratio: Optional[float],
    eligible: bool, policy: AuditPolicy,
) -> RowStatus:
    """Return a row's status."""
    if not eligible:
        return "described_only"
    if group == reference:
        return "reference"
    if ratio is not None and _past(ratio, policy):
        return "flagged"
    return "ok"


def rate_audit(
    outcomes: Sequence[Outcome], metric: str,
    policy: AuditPolicy = AuditPolicy(),
) -> AuditTable:
    """Return the rate table of ``outcomes`` (one outcome per person)."""
    if not outcomes:
        raise ValueError("no outcomes to audit")
    seen: Dict[str, Outcome] = {}
    for outcome in outcomes:
        if outcome.unit in seen:
            raise ValueError(f"{outcome.unit!r} has two outcomes")
        seen[outcome.unit] = outcome
    total = len(outcomes)
    counts: Dict[str, List[int]] = {}
    for outcome in outcomes:
        cell = counts.setdefault(outcome.group, [0, 0])
        cell[0] += int(outcome.selected)
        cell[1] += 1
    rates = {g: s / n for g, (s, n) in counts.items()}
    eligible = [g for g, (_, n) in counts.items()
                if _eligible(n, total, policy)]
    reference = _reference(rates, eligible, policy.direction)
    rows: List[AuditRow] = []
    for group in sorted(counts):
        selected, n = counts[group]
        low, high = wilson_interval(selected, n, policy.confidence)
        ratio: Optional[float] = None
        interval: Optional[Tuple[float, float]] = None
        ok = group in eligible
        if ok and reference is not None and rates[reference] > 0:
            ref_selected, ref_n = counts[reference]
            ratio = rates[group] / rates[reference]
            if group != reference:
                interval = ratio_interval(
                    selected, n, ref_selected, ref_n, policy.confidence
                )
        rows.append(AuditRow(
            group=group, n=n, share=n / total, selected=selected,
            value=rates[group], low=low, high=high, ratio=ratio,
            ratio_low=None if interval is None else interval[0],
            ratio_high=None if interval is None else interval[1],
            status=_status(group, reference, ratio, ok, policy),
            certain=_certain(interval, policy),
        ))
    return AuditTable(metric=metric, kind="rate", policy=policy,
                      units=total, reference_group=reference, rows=rows)


def _percentiles(
    values: Sequence[float], confidence: float
) -> Tuple[float, float]:
    """Return the two-sided percentile interval of bootstrap values."""
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(np.asarray(values), [tail, 100 - tail])
    return float(low), float(high)


def mean_audit(
    measurements: Sequence[Measurement], metric: str,
    policy: AuditPolicy = AuditPolicy(
        direction="lower_is_better", threshold=ERROR_RATIO
    ),
) -> AuditTable:
    """Return the mean table of ``measurements`` (people may repeat).

    Each person's measurements are averaged; the group mean is the mean
    over its people. The bootstrap resamples people within each group
    (``policy.resamples`` times, seed ``policy.seed``); the ratio's
    interval comes from the same resamples of both groups.
    """
    people = _per_unit(measurements)
    if not people:
        raise ValueError("no measurements to audit")
    total = len(people)
    by_group: Dict[str, List[float]] = {}
    for unit in sorted(people):
        group, value = people[unit]
        by_group.setdefault(group, []).append(value)
    means = {g: float(np.mean(v)) for g, v in by_group.items()}
    eligible = [g for g, v in by_group.items()
                if _eligible(len(v), total, policy)]
    reference = _reference(means, eligible, policy.direction)
    rng = np.random.default_rng(policy.seed)
    draws: Dict[str, npt.NDArray[np.float64]] = {}
    for group in sorted(by_group):
        values = np.asarray(by_group[group], dtype=np.float64)
        picks = rng.integers(0, len(values),
                             size=(policy.resamples, len(values)))
        draws[group] = values[picks].mean(axis=1)
    rows: List[AuditRow] = []
    for group in sorted(by_group):
        low, high = _percentiles(list(draws[group]), policy.confidence)
        ratio: Optional[float] = None
        interval: Optional[Tuple[float, float]] = None
        ok = group in eligible
        if ok and reference is not None and means[reference] > 0:
            ratio = means[group] / means[reference]
            base = draws[reference]
            kept = base > 0
            ratios = draws[group][kept] / base[kept]
            if group != reference and ratios.size:
                interval = _percentiles(list(ratios), policy.confidence)
        rows.append(AuditRow(
            group=group, n=len(by_group[group]),
            share=len(by_group[group]) / total, value=means[group],
            low=low, high=high, ratio=ratio,
            ratio_low=None if interval is None else interval[0],
            ratio_high=None if interval is None else interval[1],
            status=_status(group, reference, ratio, ok, policy),
            certain=_certain(interval, policy),
        ))
    return AuditTable(metric=metric, kind="mean", policy=policy,
                      units=total, reference_group=reference, rows=rows)
