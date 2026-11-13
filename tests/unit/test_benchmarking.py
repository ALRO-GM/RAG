"""Public benchmark metrics and the benchmark use case."""

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pytest

from src.application.benchmarking import BenchmarkData, BenchmarkService
from src.domain.benchmarking import (
    hit_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    unique_documents,
)
from src.domain.errors import InvalidInputError
from src.domain.models import IndexReport, ScoredChunk
from tests.fakes import make_chunk


def test_unique_documents_keeps_best_rank_and_skips_the_query() -> None:
    """A document appears once, at the rank of its first chunk."""
    assert unique_documents(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert unique_documents(["q", "a", "q"], skip="q") == ["a"]
    assert unique_documents([]) == []


def test_ndcg_matches_hand_computation() -> None:
    """Graded gains, log2 discount, ideal ordering of judged documents."""
    relevance = {"a": 2, "b": 1, "c": 0}
    perfect = ndcg_at_k(["a", "b", "x"], relevance, 10)
    assert perfect == pytest.approx(1.0)
    swapped = ndcg_at_k(["b", "a"], relevance, 10)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert swapped == pytest.approx(expected)
    assert ndcg_at_k(["x", "y"], relevance, 10) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0
    assert ndcg_at_k(["x", "a"], relevance, 1) == 0.0


def test_recall_and_hit() -> None:
    """Recall is the share of relevant documents; hit is any of them."""
    relevance = {"a": 1, "b": 2, "c": 0}
    assert recall_at_k(["a", "x"], relevance, 10) == 0.5
    assert recall_at_k(["x", "a", "b"], relevance, 1) == 0.0
    assert recall_at_k(["a"], {"c": 0}, 10) == 0.0
    assert hit_at_k(["x", "b"], relevance, 2) == 1.0
    assert hit_at_k(["x", "b"], relevance, 1) == 0.0
    assert hit_at_k(["c"], relevance, 5) == 0.0
    assert mean([]) == 0.0 and mean([1.0, 0.0]) == 0.5


class StubPipeline:
    """A pipeline returning scripted chunk paths per query."""

    def __init__(self, answers: Mapping[str, Sequence[str]]):
        """Map each query text to the files it retrieves."""
        self.answers = answers

    def index(self) -> Tuple[IndexReport, bool]:
        """Report a tiny cold index."""
        return IndexReport(
            files_total=3, files_reindexed=3, files_reused=0,
            files_removed=0, chunks=4, seconds=1.25, incremental=False,
        ), True

    def search(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the scripted chunks of every query."""
        return [
            [
                ScoredChunk(chunk=make_chunk(path, 0, "text"), score=1.0)
                for path in self.answers.get(query, [])
            ]
            for query in queries
        ]


class StubFactory:
    """A factory recording the parameters it was asked for."""

    def __init__(self, pipeline: StubPipeline):
        """Always return ``pipeline``."""
        self.pipeline = pipeline
        self.params: List[Dict[str, Any]] = []

    def create(self, params: Mapping[str, Any]) -> StubPipeline:
        """Record ``params`` and return the pipeline."""
        self.params.append(dict(params))
        return self.pipeline


class StubSource:
    """A source with one tiny dataset."""

    def load(self, name: str) -> BenchmarkData:
        """Return three documents and two judged queries."""
        return BenchmarkData(
            name=name, raw_dir="corpus/files",
            doc_ids={"d0": "A", "d1": "B", "d2": "q1"},
            queries={"q1": "first", "q2": "second"},
            qrels={"q1": {"A": 1}, "q2": {"B": 2, "A": 1}},
        )


def test_benchmark_scores_documents_from_chunks() -> None:
    """Chunks map to documents; the query's own document is ignored."""
    pipeline = StubPipeline({
        "first": ["corpus/files/d2.txt", "corpus/files/d0.txt"],
        "second": ["corpus/files/d0.txt", "corpus/files/d0.txt",
                   "corpus/files/d1.txt"],
    })
    factory = StubFactory(pipeline)
    service = BenchmarkService(StubSource(), factory)
    results = service.run("tiny", {"base": {}, "lexical": {"mode": "bm25"}})
    assert [r.config for r in results] == ["base", "lexical"]
    assert factory.params == [
        {"raw_dir": "corpus/files"},
        {"mode": "bm25", "raw_dir": "corpus/files"},
    ]
    first = results[0]
    assert (first.documents, first.queries, first.chunks) == (3, 2, 4)
    second_query = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert first.ndcg10 == pytest.approx((1.0 + second_query) / 2, abs=1e-4)
    assert first.recall10 == 1.0 and first.recall100 == 1.0
    assert first.hit == {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}
    assert first.index_seconds == 1.2 and first.index_cold


def test_benchmark_limits_queries_and_validates_input() -> None:
    """``max_queries`` truncates; empty configurations are rejected."""
    service = BenchmarkService(StubSource(), StubFactory(StubPipeline({})))
    limited = service.run("tiny", {"base": {}}, max_queries=1)
    assert limited[0].queries == 1 and limited[0].ndcg10 == 0.0
    with pytest.raises(InvalidInputError):
        service.run("tiny", {})
