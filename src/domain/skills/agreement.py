"""Agreement statistics between raters (human or automatic).

Used twice: between human annotators, to know whether the reference
data is reliable enough to be a reference, and between the system and
the human consensus, to know how good the system is.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

Rating = Optional[float]

ALPHA_RELIABLE = 0.80
ALPHA_TENTATIVE = 0.667


def paired(
    a: Sequence[Rating], b: Sequence[Rating]
) -> Tuple[List[float], List[float]]:
    """Return the units that both raters rated."""
    pairs = [
        (float(x), float(y)) for x, y in zip(a, b)
        if x is not None and y is not None
    ]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def percent_agreement(
    a: Sequence[Rating], b: Sequence[Rating], tolerance: float = 0.0
) -> Optional[float]:
    """Return the share of co-rated units within ``tolerance`` points."""
    x, y = paired(a, b)
    if not x:
        return None
    return sum(abs(i - j) <= tolerance for i, j in zip(x, y)) / len(x)


def mean_absolute_error(
    a: Sequence[Rating], b: Sequence[Rating]
) -> Optional[float]:
    """Return the mean absolute difference over co-rated units."""
    x, y = paired(a, b)
    if not x:
        return None
    return sum(abs(i - j) for i, j in zip(x, y)) / len(x)


def cohen_kappa(
    a: Sequence[Rating],
    b: Sequence[Rating],
    categories: Sequence[float],
    weights: str = "quadratic",
) -> Optional[float]:
    """Return Cohen's kappa of two raters over the co-rated units.

    ``weights`` is ``"none"`` (nominal), ``"linear"`` or
    ``"quadratic"`` (ordinal scales: a one-point disagreement costs
    less than a three-point one). Returns ``None`` without co-rated
    units and 1.0 when both raters used a single identical category.
    """
    x, y = paired(a, b)
    if not x:
        return None
    index = {float(c): i for i, c in enumerate(categories)}
    k = len(index)
    observed = np.zeros((k, k), dtype=np.float64)
    for i, j in zip(x, y):
        observed[index[i], index[j]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    if weights == "none":
        cost = (grid > 0).astype(np.float64)
    elif weights == "linear":
        cost = grid / max(k - 1, 1)
    else:
        cost = (grid / max(k - 1, 1)) ** 2
    disagreement = float((cost * expected).sum())
    if disagreement == 0.0:
        return 1.0
    return 1.0 - float((cost * observed).sum()) / disagreement


def _distance(
    level: str, totals: npt.NDArray[np.float64], c: int, k: int
) -> float:
    """Return Krippendorff's squared distance between two categories."""
    if c == k:
        return 0.0
    if level == "nominal":
        return 1.0
    low, high = min(c, k), max(c, k)
    if level == "ordinal":
        between = float(totals[low:high + 1].sum())
        ends = float(totals[low] + totals[high])
        return (between - ends / 2.0) ** 2
    return float(high - low) ** 2


def krippendorff_alpha(
    ratings: Sequence[Sequence[Rating]],
    categories: Sequence[float],
    level: str = "ordinal",
) -> Optional[float]:
    """Return Krippendorff's alpha for any number of raters.

    ``ratings`` is one row per rater and one column per unit;
    ``None`` marks a unit the rater did not rate, which alpha handles
    natively. ``level`` is ``nominal``, ``ordinal`` or ``interval``
    (for ``interval`` the categories must be equally spaced). Returns
    ``None`` when no unit has two ratings and 1.0 when every rating is
    the same value.
    """
    index = {float(c): i for i, c in enumerate(categories)}
    k = len(index)
    coincidence = np.zeros((k, k), dtype=np.float64)
    n_units = len(ratings[0]) if ratings else 0
    for unit in range(n_units):
        cells = [row[unit] for row in ratings]
        values = [index[float(v)] for v in cells if v is not None]
        m = len(values)
        if m < 2:
            continue
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j:
                    coincidence[a, b] += 1.0 / (m - 1)
    totals = coincidence.sum(axis=1)
    n = float(totals.sum())
    if n <= 1:
        return None
    observed = 0.0
    expected = 0.0
    for c in range(k):
        for j in range(k):
            delta = _distance(level, totals, c, j)
            observed += coincidence[c, j] * delta
            expected += totals[c] * totals[j] * delta
    if expected == 0.0:
        return 1.0
    return 1.0 - (n - 1) * observed / expected


def icc_two_way(
    ratings: Sequence[Sequence[float]],
) -> Optional[Tuple[float, float]]:
    """Return ICC(2,1) and ICC(2,k): two-way random, absolute agreement.

    ``ratings`` is one row per rater and one column per unit, with no
    missing value. ICC(2,1) is the reliability of a single rater,
    ICC(2,k) that of the mean of the ``k`` raters. Returns ``None``
    with fewer than two raters or two units, or without variance.
    """
    data = np.asarray(ratings, dtype=np.float64).T
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None
    n, k = data.shape
    grand = data.mean()
    rows = data.mean(axis=1)
    cols = data.mean(axis=0)
    ms_rows = k * float(((rows - grand) ** 2).sum()) / (n - 1)
    ms_cols = n * float(((cols - grand) ** 2).sum()) / (k - 1)
    residual = data - rows[:, None] - cols[None, :] + grand
    ms_error = float((residual ** 2).sum()) / ((n - 1) * (k - 1))
    single_den = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    average_den = ms_rows + (ms_cols - ms_error) / n
    if single_den == 0.0 or average_den == 0.0:
        return None
    return (
        (ms_rows - ms_error) / single_den,
        (ms_rows - ms_error) / average_den,
    )


def set_scores(predicted: Set[int], gold: Set[int]) -> Dict[str, float]:
    """Return precision, recall and F1 of a predicted set.

    Two empty sets agree perfectly; an empty prediction against a
    non-empty reference (or the reverse) scores 0.
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    hits = len(predicted & gold)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    total = precision + recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / total if total else 0.0,
    }


def interpret_alpha(alpha: Optional[float]) -> str:
    """Name the reliability band of an alpha (Krippendorff's bounds)."""
    if alpha is None:
        return "not computable"
    if alpha >= ALPHA_RELIABLE:
        return "reliable"
    if alpha >= ALPHA_TENTATIVE:
        return "tentative"
    return "unreliable"
