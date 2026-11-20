"""Paired comparison of two taggers scored on the same sentences.

Pure functions, no I/O. Top-1 hits are compared with the exact McNemar
test (only the sentences where exactly one tagger is right carry
information) and per-sentence log-losses with a seeded paired bootstrap.
"""

from dataclasses import dataclass
from math import comb
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PairedTop1:
    """Result of the exact McNemar test on top-1 hits.

    Attributes:
        only_candidate: Sentences only the candidate got right.
        only_baseline: Sentences only the baseline got right.
        p_value: Two-sided exact p-value under "equally likely".
    """

    only_candidate: int
    only_baseline: int
    p_value: float


def mcnemar_exact(candidate: Sequence[bool],
                  baseline: Sequence[bool]) -> PairedTop1:
    """Compare two sets of top-1 hits with the exact McNemar test.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(candidate) != len(baseline):
        raise ValueError("Both taggers must score the same sentences.")
    b = sum(1 for c, o in zip(candidate, baseline) if c and not o)
    c = sum(1 for c_, o in zip(candidate, baseline) if o and not c_)
    n = b + c
    if n == 0:
        return PairedTop1(0, 0, 1.0)
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return PairedTop1(b, c, min(1.0, 2 * tail))


def paired_loss_gap(candidate: Sequence[float], baseline: Sequence[float],
                    samples: int = 2000, seed: int = 0
                    ) -> Tuple[float, float, float]:
    """Return ``(mean gap, low, high)`` of ``baseline - candidate`` losses.

    A positive gap means the candidate has the lower log-loss. The
    interval is a 95 % percentile bootstrap over sentences.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("Need the same, non-empty sentences for both.")
    gap = np.asarray(baseline, dtype=float) - np.asarray(candidate,
                                                         dtype=float)
    rng = np.random.default_rng(seed)
    draws = gap[rng.integers(0, gap.size, size=(samples, gap.size))].mean(
        axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(gap.mean()), float(low), float(high)
