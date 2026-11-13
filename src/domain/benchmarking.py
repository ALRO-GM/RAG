"""Metrics of public retrieval benchmarks (document-level relevance)."""

import math
from typing import Dict, List, Mapping, Sequence


def unique_documents(ranked: Sequence[str], skip: str = "") -> List[str]:
    """Return document ids in rank order, each once.

    Several chunks of one document may be retrieved: the document
    keeps the rank of its best chunk. ``skip`` drops the document that
    is the query itself (BEIR ignores identical ids).
    """
    seen: Dict[str, None] = {}
    for doc_id in ranked:
        if doc_id != skip:
            seen.setdefault(doc_id, None)
    return list(seen)


def ndcg_at_k(
    ranked: Sequence[str], relevance: Mapping[str, int], k: int
) -> float:
    """Return nDCG@k as ``trec_eval`` computes it.

    Gain is the relevance grade, the discount ``log2(rank + 1)``, and
    the ideal ranking sorts the judged documents by grade.
    """
    gains = [relevance.get(doc_id, 0) for doc_id in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)
    best = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / best if best > 0 else 0.0


def recall_at_k(
    ranked: Sequence[str], relevance: Mapping[str, int], k: int
) -> float:
    """Return the share of relevant documents found in the top ``k``."""
    relevant = {d for d, g in relevance.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def hit_at_k(
    ranked: Sequence[str], relevance: Mapping[str, int], k: int
) -> float:
    """Return 1.0 when a relevant document is in the top ``k``."""
    return 1.0 if any(relevance.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def mean(values: Sequence[float]) -> float:
    """Return the mean, 0.0 for no value."""
    return sum(values) / len(values) if values else 0.0
