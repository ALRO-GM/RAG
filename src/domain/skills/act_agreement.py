"""Agreement and accuracy for weighted dialogue-act labels.

A sentence carries 1 to 3 weighted labels (owner decision 2 on #57,
:class:`~src.domain.skills.dialogue_acts.WeightedLabel`). The measures
chosen for them (``docs/skills/CONVERSATION.md`` §6.3) are:

- **Between annotators.** Cohen's κ on the primary label, overall and
  per category (one category against the rest), each with a percentile
  bootstrap 95 % interval over sentences: the rule of §5.3.1 b is
  unchanged. For the weights: **mean overlap** (Σ of the smaller weight
  of each category; 1 means the same labels with the same weights) and
  **Krippendorff's α** with the distance ``1 − overlap`` between the two
  label distributions, which corrects that overlap for chance and is the
  nominal α when every sentence has one label.
- **Tagger against the reference.** Top-1 and top-3 accuracy (against
  the primary label), log-loss (cross-entropy against the reference
  weights) and expected calibration error of the top-1 confidence.

A tagger's labels use the same shape: its top 3 calibrated
probabilities in whole %, what is left kept as an explicit ``rest``
(not renormalised, which would inflate them), and a label whose
probability does not reach its category's threshold marked doubtful.
Everything here is a pure function.
"""

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.agreement import (
    cohen_kappa,
    krippendorff_alpha_distance,
)
from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    MAX_LABELS,
    WeightedLabel,
)

Labels = Sequence[WeightedLabel]
Annotation = Optional[Labels]
"""One sentence's labels as one rater gave them (``None``: not rated)."""

Distribution = Tuple[Tuple[str, int], ...]
"""Labels as a hashable value: ``(category, weight)`` sorted by category."""

Probabilities = Mapping[str, float]
"""A tagger's output for one sentence: category → probability (0–1)."""

_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


class KappaEstimate(BaseModel):
    """A κ with its percentile bootstrap 95 % interval.

    ``kappa`` is ``None`` when it cannot be computed (no co-rated
    sentence, or both raters never or always used the category); ``low``
    and ``high`` are ``None`` when no resample could be computed.
    ``count_a`` and ``count_b`` are the sentences each rater gave the
    category (for the overall κ: the co-rated sentences).
    """

    model_config = ConfigDict(frozen=True)

    kappa: Optional[float]
    low: Optional[float]
    high: Optional[float]
    sentences: int = Field(ge=0)
    count_a: int = Field(ge=0)
    count_b: int = Field(ge=0)


class TaggerLabels(BaseModel):
    """A tagger's labels for one sentence and the share left over (%)."""

    model_config = ConfigDict(frozen=True)

    labels: List[WeightedLabel]
    rest: int = Field(ge=0, le=100)


def primary(labels: Labels) -> str:
    """Return the primary (first) label of a sentence."""
    return labels[0].label


def as_distribution(labels: Labels) -> Distribution:
    """Return ``labels`` as a hashable distribution (order-free)."""
    return tuple(sorted((lab.label, lab.weight) for lab in labels))


def overlap(a: Labels, b: Labels) -> float:
    """Return the share (0–1) of function two label lists agree on.

    Σ over categories of the smaller of the two weights, / 100. One
    minus it is the total variation distance between the two
    distributions. "compromiso 70, aceptacion 30" against "compromiso
    100" overlaps 0.7.
    """
    weights = {lab.label: lab.weight for lab in b}
    return sum(min(lab.weight, weights.get(lab.label, 0))
               for lab in a) / 100.0


def distribution_distance(a: Distribution, b: Distribution) -> float:
    """Return ``1 − overlap`` between two distributions (0 when equal)."""
    weights = dict(b)
    return 1.0 - sum(min(w, weights.get(c, 0)) for c, w in a) / 100.0


def _co_rated(
    a: Sequence[Annotation], b: Sequence[Annotation]
) -> List[Tuple[Labels, Labels]]:
    """Return the sentences both raters labelled."""
    if len(a) != len(b):
        raise ValueError("both raters must cover the same sentences")
    return [(x, y) for x, y in zip(a, b) if x and y]


def mean_overlap(
    a: Sequence[Annotation], b: Sequence[Annotation]
) -> Optional[float]:
    """Return the mean :func:`overlap` over co-rated sentences."""
    pairs = _co_rated(a, b)
    if not pairs:
        return None
    return sum(overlap(x, y) for x, y in pairs) / len(pairs)


def weighted_alpha(ratings: Sequence[Sequence[Annotation]]) -> Optional[float]:
    """Return Krippendorff's α over label distributions.

    ``ratings`` has one row per rater and one column per sentence;
    distance is :func:`distribution_distance`. With one label per
    sentence it equals the nominal α.
    """
    rows = [[as_distribution(x) if x else None for x in row]
            for row in ratings]
    return krippendorff_alpha_distance(rows, distribution_distance)


def _kappa_draws(
    x: npt.NDArray[np.int64], y: npt.NDArray[np.int64], k: int,
    resamples: int, seed: int,
) -> npt.NDArray[np.float64]:
    """Return nominal κ of ``resamples`` bootstrap resamples (finite)."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, x.size, size=(resamples, x.size))
    xs, ys = x[picks], y[picks]
    observed = (xs == ys).mean(axis=1)
    expected = np.zeros(resamples)
    for c in range(k):
        expected += (xs == c).mean(axis=1) * (ys == c).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws: npt.NDArray[np.float64] = (
            (observed - expected) / (1.0 - expected))
    kept: npt.NDArray[np.float64] = draws[expected < 1.0]
    return kept


def _estimate(
    x: List[int], y: List[int], k: int, resamples: int, seed: int,
    counts: Tuple[int, int],
) -> KappaEstimate:
    """Return the nominal κ of index lists ``x``/``y`` with its interval."""
    degenerate = len(set(x) | set(y)) < 2
    kappa = None if not x or degenerate else cohen_kappa(
        [float(i) for i in x], [float(i) for i in y],
        [float(i) for i in range(k)], weights="none")
    low = high = None
    if x and resamples > 0:
        draws = _kappa_draws(np.asarray(x), np.asarray(y), k,
                             resamples, seed)
        if draws.size:
            lo, hi = np.percentile(draws, [2.5, 97.5])
            low, high = float(lo), float(hi)
    return KappaEstimate(kappa=kappa, low=low, high=high,
                         sentences=len(x), count_a=counts[0],
                         count_b=counts[1])


def primary_kappa(
    a: Sequence[Annotation], b: Sequence[Annotation],
    resamples: int = 2000, seed: int = 0,
) -> KappaEstimate:
    """Return Cohen's κ on the primary label with a 95 % interval.

    Only sentences both raters labelled count; the interval resamples
    those sentences with replacement (percentile bootstrap).
    """
    pairs = _co_rated(a, b)
    x = [_INDEX[primary(p)] for p, _ in pairs]
    y = [_INDEX[primary(q)] for _, q in pairs]
    return _estimate(x, y, len(CATEGORIES), resamples, seed,
                     (len(pairs), len(pairs)))


def category_kappas(
    a: Sequence[Annotation], b: Sequence[Annotation],
    resamples: int = 2000, seed: int = 0,
) -> Dict[str, KappaEstimate]:
    """Return κ per category (that category against the rest).

    Computed on the primary label of the co-rated sentences, each with
    its percentile bootstrap 95 % interval, for the keep / provisional /
    drop rule of ``CONVERSATION.md`` §5.3.1 b.
    """
    pairs = _co_rated(a, b)
    result: Dict[str, KappaEstimate] = {}
    for category in CATEGORIES:
        x = [int(primary(p) == category) for p, _ in pairs]
        y = [int(primary(q) == category) for _, q in pairs]
        result[category] = _estimate(x, y, 2, resamples, seed,
                                     (sum(x), sum(y)))
    return result


# --- The tagger ----------------------------------------------------------


def full_distribution(probabilities: Probabilities) -> Dict[str, float]:
    """Return a probability for each of the 13 categories.

    Categories the tagger left out share what is left to 1 equally (a
    top-3 output). Raises :class:`ValueError` on an unknown category, a
    probability outside 0–1, a total above 1, or a total below 1 with
    no category left out.
    """
    for category, p in probabilities.items():
        if category not in _INDEX:
            raise ValueError(f"unknown category {category!r}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability of {category} is {p}")
    rest = 1.0 - sum(probabilities.values())
    if rest < -1e-9:
        raise ValueError("probabilities add up to more than 1")
    missing = [c for c in CATEGORIES if c not in probabilities]
    if not missing and rest > 1e-9:
        raise ValueError("probabilities of all categories add up to "
                         "less than 1")
    share = max(rest, 0.0) / len(missing) if missing else 0.0
    return {c: float(probabilities.get(c, share)) for c in CATEGORIES}


def ranked(probabilities: Probabilities) -> List[Tuple[str, float]]:
    """Return categories from most to least probable (ties: list order)."""
    full = full_distribution(probabilities)
    return sorted(full.items(), key=lambda kv: (-kv[1], _INDEX[kv[0]]))


def _check_lengths(gold: Sequence[Labels],
                   predicted: Sequence[Probabilities]) -> None:
    """Refuse empty or unequal inputs."""
    if not gold or len(gold) != len(predicted):
        raise ValueError("need one prediction per reference sentence")


def top_k_accuracy(
    gold: Sequence[Labels], predicted: Sequence[Probabilities], k: int = 1
) -> float:
    """Return the share of sentences whose primary label is in the top k.
    """
    _check_lengths(gold, predicted)
    if k < 1:
        raise ValueError("k must be >= 1")
    hits = sum(
        primary(labels) in [c for c, _ in ranked(p)[:k]]
        for labels, p in zip(gold, predicted))
    return hits / len(gold)


def log_loss(
    gold: Sequence[Labels], predicted: Sequence[Probabilities],
    eps: float = 1e-15,
) -> float:
    """Return the mean cross-entropy (natural log) against the reference.

    The reference of a sentence is its weights / 100 (one label at 100
    gives the usual log-loss, −ln p of that label); probabilities are
    clipped to ``eps`` so a confident miss costs a large finite amount.
    """
    _check_lengths(gold, predicted)
    total = 0.0
    for labels, p in zip(gold, predicted):
        full = full_distribution(p)
        total -= sum(lab.weight / 100.0 * math.log(max(full[lab.label], eps))
                     for lab in labels)
    return total / len(gold)


def expected_calibration_error(
    gold: Sequence[Labels], predicted: Sequence[Probabilities],
    bins: int = 10,
) -> float:
    """Return the expected calibration error of the top-1 confidence.

    Sentences go into ``bins`` equal-width bins of their top
    probability ((0, .1], (.1, .2], …; 0 falls in the first); ECE is
    Σ |bin| / n · |accuracy − mean confidence| (Guo et al., 2017).
    """
    _check_lengths(gold, predicted)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    sums: Dict[int, List[float]] = {}
    for labels, p in zip(gold, predicted):
        top, confidence = ranked(p)[0]
        index = int(np.searchsorted(edges, confidence, side="left"))
        cell = sums.setdefault(index, [0.0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += float(top == primary(labels))
        cell[2] += confidence
    return sum(abs(hits - conf) for _, hits, conf in sums.values()) / len(
        gold)


def base_rates(gold: Sequence[Labels]) -> Dict[str, float]:
    """Return each category's share of the reference, weights included.

    The mean over sentences of weight / 100: with one label per sentence
    it is the share of sentences with that label (``CONVERSATION.md``
    §5.3.1 c).
    """
    if not gold:
        raise ValueError("no reference sentences")
    rates = {c: 0.0 for c in CATEGORIES}
    for labels in gold:
        for lab in labels:
            rates[lab.label] += lab.weight / 100.0
    return {c: r / len(gold) for c, r in rates.items()}


def doubt_thresholds(
    rates: Mapping[str, float], margin: float
) -> Dict[str, float]:
    """Return the least probability per category that is not doubtful.

    The base rate plus ``margin`` (in probability, e.g. 0.1). The margin
    is set on the gold set when it exists; there is no default.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    return {c: rates.get(c, 0.0) + margin for c in CATEGORIES}


def _hamilton(values: Sequence[float]) -> List[int]:
    """Round shares of 1 to whole % that add up to exactly 100."""
    raw = [v * 100.0 for v in values]
    floors = [math.floor(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (floors[i] - raw[i], i))
    for i in order[:100 - sum(floors)]:
        floors[i] += 1
    return floors


def tagger_labels(
    probabilities: Probabilities, k: int = MAX_LABELS,
    thresholds: Optional[Mapping[str, float]] = None,
) -> TaggerLabels:
    """Return a tagger's top ``k`` labels in whole % and the rest.

    Not renormalised: the weights stay the calibrated probabilities,
    rounded so that labels and rest add up to 100 (largest remainder);
    a label that rounds to 0 is left out. With ``thresholds`` (see
    :func:`doubt_thresholds`), a label below its category's threshold is
    marked doubtful.
    """
    if not 1 <= k <= MAX_LABELS:
        raise ValueError(f"k must be 1 to {MAX_LABELS}")
    top = ranked(probabilities)[:k]
    weights = _hamilton([p for _, p in top]
                        + [max(0.0, 1.0 - sum(p for _, p in top))])
    labels = [
        WeightedLabel(
            label=c, weight=w,
            doubtful=thresholds is not None and p < thresholds[c])
        for (c, p), w in zip(top, weights) if w > 0
    ]
    return TaggerLabels(labels=labels, rest=weights[-1])
