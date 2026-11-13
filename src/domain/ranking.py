"""Ranking primitives: top-k selection and score fusion."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Ranked = List[Tuple[int, float]]

FUSIONS = ("linear", "rrf")


def top_k(scores: FloatArray, k: int) -> List[int]:
    """Return the indices of the ``k`` highest scores, best first."""
    k = min(k, int(scores.shape[0]))
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in part[np.argsort(-scores[part], kind="stable")]]


def linear_scores(
    lexical: FloatArray, dense: FloatArray, weight: float
) -> FloatArray:
    """Return ``lexical / max(lexical) + weight * dense`` for every chunk.

    Scaling BM25 to [0, 1] per query puts it on the range of a cosine
    similarity, so ``weight`` is a meaningful balance between the two.
    """
    top = float(lexical.max()) if lexical.size else 0.0
    fused: FloatArray = (
        (lexical / top if top > 0 else lexical) + weight * dense
    ).astype(np.float32)
    return fused


def ranked(scores: FloatArray, k: int) -> Ranked:
    """Return the ``k`` best (position, score) pairs of a score vector."""
    return [(i, float(scores[i])) for i in top_k(scores, k)]


def linear_fusion(
    lexical: FloatArray, dense: FloatArray, weight: float, k: int
) -> Ranked:
    """Return the top ``k`` of :func:`linear_scores`."""
    return ranked(linear_scores(lexical, dense, weight), k)


def apply_priors(
    scores: FloatArray,
    multipliers: FloatArray,
    file_ids: npt.NDArray[np.int64],
    file_boost: float,
) -> FloatArray:
    """Scale scores by per-chunk priors, then share file-level evidence.

    ``file_boost`` adds to every chunk that share of the best score
    found in its file. All chunks of a file move together, so the
    order inside a file is kept while strong files gain rank.
    """
    adjusted: FloatArray = (scores * multipliers).astype(np.float32)
    if file_boost > 0 and adjusted.size:
        best = np.full(int(file_ids.max()) + 1, -np.inf, dtype=np.float32)
        np.maximum.at(best, file_ids, adjusted)
        adjusted = adjusted + np.float32(file_boost) * best[file_ids]
    return adjusted


def rrf_fusion(
    rankings: Sequence[Tuple[float, Sequence[int]]],
    k: int,
    rrf_k: int = 60,
) -> Ranked:
    """Reciprocal Rank Fusion: ``sum_r w_r / (rrf_k + rank_r(d))``."""
    fused: Dict[int, float] = {}
    for weight, ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + weight / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _min_max(values: Sequence[float]) -> List[float]:
    """Scale ``values`` to [0, 1]; a constant list maps to zeros."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def rerank_fusion(
    first: Sequence[float], second: Sequence[float], weight: float
) -> Ranked:
    """Blend two scorings of the same candidates, best first.

    Both lists are min-max normalised per query, then combined as
    ``(1 - weight) * first + weight * second``. Ties keep the order of
    ``first`` (the candidates are expected in first-stage order).
    """
    if len(first) != len(second):
        raise ValueError("Both scorings must cover the same candidates.")
    fused = [
        (1.0 - weight) * a + weight * b
        for a, b in zip(_min_max(first), _min_max(second))
    ]
    return sorted(enumerate(fused), key=lambda kv: (-kv[1], kv[0]))
