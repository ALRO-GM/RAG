"""Experiment models and statistics (pure, no I/O).

A *trial* is one configuration of the pipeline evaluated on datasets
with reference spans. Its result keeps the rank obtained by every
question, so any two trials can be compared question by question.
"""

import hashlib
import itertools
import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.domain.evaluation import IOU_THRESHOLD, iou
from src.domain.models import MinimalSource

K_VALUES = (1, 3, 5, 10)
Ranks = List[List[int]]
"""Per question, the 1-based rank of each reference (0 = not found)."""
Scores = Dict[str, List[float]]
"""Per dataset, one score per question."""


class TrialSpec(BaseModel):
    """A named set of parameter overrides."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class Suite(BaseModel):
    """A family of trials around a common base configuration.

    - ``one_at_a_time``: each factor value is tried alone (ablation or
      sweep of one constant).
    - ``grid``: every combination of the factor values (interactions).
    - ``list``: the explicit ``trials``.
    """

    name: str
    description: str = ""
    base: Dict[str, Any] = Field(default_factory=dict)
    strategy: Literal["one_at_a_time", "grid", "list"] = "one_at_a_time"
    factors: Dict[str, List[Any]] = Field(default_factory=dict)
    trials: List[TrialSpec] = Field(default_factory=list)

    def expand(self) -> List[TrialSpec]:
        """Return the baseline followed by every distinct trial."""
        specs = [TrialSpec(name="baseline", params=dict(self.base))]
        if self.strategy == "one_at_a_time":
            for factor, values in self.factors.items():
                for value in values:
                    specs.append(TrialSpec(
                        name=f"{factor}={value}",
                        params={**self.base, factor: value},
                    ))
        elif self.strategy == "grid":
            names = list(self.factors)
            for combo in itertools.product(*self.factors.values()):
                specs.append(TrialSpec(
                    name=",".join(f"{n}={v}" for n, v in zip(names, combo)),
                    params={**self.base, **dict(zip(names, combo))},
                ))
        else:
            for trial in self.trials:
                specs.append(TrialSpec(
                    name=trial.name, params={**self.base, **trial.params}
                ))
        seen: Dict[str, TrialSpec] = {}
        for spec in specs:
            seen.setdefault(params_id(spec.params), spec)
        return list(seen.values())


class TrialResult(BaseModel):
    """Everything measured for one trial."""

    trial_id: str
    suite: str
    name: str
    params: Dict[str, Any]
    baseline_id: str
    metrics: Dict[str, Dict[str, float]]
    ranks: Dict[str, Ranks]
    search_seconds_per_100: float
    index_seconds: float
    index_cold: bool
    n_chunks: int
    created: str = ""

    def mean_metric(self, name: str) -> float:
        """Return a metric averaged over the datasets."""
        values = [m[name] for m in self.metrics.values()]
        return sum(values) / len(values) if values else 0.0


def params_id(params: Mapping[str, Any], salt: str = "") -> str:
    """Return a short stable identifier of a parameter set."""
    payload = json.dumps([salt, dict(params)], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def source_ranks(
    expected: Sequence[MinimalSource], retrieved: Sequence[MinimalSource]
) -> List[int]:
    """Return, per reference, the rank of the first matching result."""
    ranks: List[int] = []
    for reference in expected:
        rank = 0
        for position, source in enumerate(retrieved, start=1):
            if iou(reference, source) >= IOU_THRESHOLD:
                rank = position
                break
        ranks.append(rank)
    return ranks


def recall_scores(ranks: Ranks, k: int) -> List[float]:
    """Return the recall@k of every question."""
    return [
        sum(1 for r in q if 0 < r <= k) / len(q) if q else 0.0
        for q in ranks
    ]


def reciprocal_ranks(ranks: Ranks) -> List[float]:
    """Return the mean reciprocal rank of every question."""
    return [
        sum(1.0 / r for r in q if r > 0) / len(q) if q else 0.0
        for q in ranks
    ]


def rank_metrics(ranks: Ranks) -> Dict[str, float]:
    """Return recall@1/3/5/10 and MRR of a dataset."""
    n = max(len(ranks), 1)
    metrics = {
        f"recall@{k}": sum(recall_scores(ranks, k)) / n for k in K_VALUES
    }
    metrics["mrr"] = sum(reciprocal_ranks(ranks)) / n
    return metrics


def objective_scores(
    ranks: Mapping[str, Ranks], k: int = 5, mrr_weight: float = 0.1
) -> Scores:
    """Return the per-question objective: recall@k + mrr_weight·RR.

    The small MRR term breaks ties between configurations with the
    same recall@k in favour of the one ranking the answer higher.
    """
    return {
        name: [
            r + mrr_weight * m
            for r, m in zip(recall_scores(rk, k), reciprocal_ranks(rk))
        ]
        for name, rk in ranks.items()
    }


def macro_mean(
    scores: Mapping[str, Sequence[float]],
    subset: Optional[Mapping[str, Sequence[int]]] = None,
) -> float:
    """Average the datasets' means, optionally on a subset of questions."""
    means: List[float] = []
    for name, values in scores.items():
        chosen = (
            list(values) if subset is None
            else [values[i] for i in subset.get(name, [])]
        )
        if chosen:
            means.append(sum(chosen) / len(chosen))
    return sum(means) / len(means) if means else 0.0


def paired_bootstrap(
    candidate: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    samples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Compare two trials on the same questions.

    Returns ``(delta, low, high, p_gain)``: the difference of macro
    means, its 95% bootstrap interval (questions resampled inside each
    dataset) and the share of resamples where the candidate wins.
    """
    names = [n for n in candidate if n in baseline]
    diffs = [
        np.asarray(candidate[n], dtype=float)
        - np.asarray(baseline[n], dtype=float)
        for n in names
    ]
    diffs = [d for d in diffs if d.size]
    if not diffs:
        return 0.0, 0.0, 0.0, 0.0
    delta = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(seed)
    draws = np.mean([
        d[rng.integers(0, d.size, size=(samples, d.size))].mean(axis=1)
        for d in diffs
    ], axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    return delta, float(low), float(high), float((draws > 0).mean())


def pareto_front(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Return the indices of the non-dominated (cost, value) points.

    A point is kept when no other point is at least as cheap and at
    least as good, and strictly better on one axis. The result is
    sorted by increasing cost.
    """
    order = sorted(
        range(len(points)), key=lambda i: (points[i][0], -points[i][1])
    )
    front: List[int] = []
    best = float("-inf")
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def kfold_indices(n: int, folds: int, seed: int = 0) -> List[List[int]]:
    """Split ``range(n)`` into ``folds`` shuffled test sets."""
    order = np.random.default_rng(seed).permutation(n)
    return [
        sorted(int(i) for i in order[f::folds]) for f in range(folds)
    ]


def synergy(base: float, a: float, b: float, both: float) -> float:
    """Return the interaction ``gain(A+B) − gain(A) − gain(B)``."""
    return (both - base) - (a - base) - (b - base)
