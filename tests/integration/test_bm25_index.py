"""BM25 index adapter against a reference implementation."""

import math
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pytest

from src.adapters.outbound.bm25_index import BM25Builder, BM25Index

DOCS: List[Dict[str, int]] = [
    {"lora": 2, "adapter": 1},
    {"kv": 1, "cache": 3, "block": 1},
    {"lora": 1, "cache": 1},
    {},
]


def reference_bm25(
    docs: Sequence[Mapping[str, int]], query: Sequence[str],
    k1: float, b: float,
) -> List[float]:
    """Okapi BM25 written naively, as in the textbook."""
    n = len(docs)
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / n
    out = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for term in query:
            tf = doc.get(term, 0)
            if not tf:
                continue
            df = sum(1 for d in docs if term in d)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * length / avg)
            )
        out.append(score)
    return out


@pytest.mark.parametrize("query", [
    ["lora"], ["cache"], ["lora", "cache"], ["cache", "cache"], ["kv"],
])
@pytest.mark.parametrize("k1, b", [(1.2, 0.75), (2.0, 0.0)])
def test_matches_reference(query: List[str], k1: float, b: float) -> None:
    """Scores equal the textbook formula (repeated terms add up)."""
    index = BM25Builder(k1, b).build(DOCS)
    np.testing.assert_allclose(
        index.scores(query), reference_bm25(DOCS, query, k1, b), rtol=1e-5
    )


def test_unknown_terms_score_zero() -> None:
    """Out-of-vocabulary queries give an all-zero float32 vector."""
    scores = BM25Index.build(DOCS).scores(["nothing", "here"])
    assert scores.dtype == np.float32
    assert scores.shape == (4,)
    assert not scores.any()
    assert not BM25Index.build(DOCS).scores([]).any()


def test_ranking_order() -> None:
    """The document with most occurrences of a rare term ranks first."""
    scores = BM25Index.build(DOCS).scores(["lora"])
    assert int(np.argmax(scores)) == 0
    assert scores[1] == 0 and scores[3] == 0


def test_empty_collection() -> None:
    """An index without documents scores nothing."""
    index = BM25Index.build([])
    assert index.scores(["x"]).shape == (0,)


def test_state_round_trip() -> None:
    """The persisted state rebuilds an identical index."""
    index = BM25Index.build(DOCS)
    weights, vocabulary = index.state()
    clone = BM25Index(weights, vocabulary)
    np.testing.assert_array_equal(
        clone.scores(["lora", "cache"]), index.scores(["lora", "cache"])
    )


def test_builder_parameters() -> None:
    """The builder forwards its parameters."""
    index = BM25Builder(k1=1.5, b=0.3).build(DOCS)
    assert (index.k1, index.b) == (1.5, 0.3)
