"""Use case: run the retrieval pipeline on a public benchmark."""

import time
from pathlib import PurePath
from typing import Any, Dict, List, Mapping, Protocol, Sequence

from pydantic import BaseModel

from src.application.ports import PipelineFactory
from src.domain.benchmarking import (
    hit_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    unique_documents,
)
from src.domain.errors import InvalidInputError


class BenchmarkData(BaseModel):
    """A benchmark ready to run: files to index, queries, judgements.

    ``doc_ids`` maps the stem of each corpus file to the benchmark's
    document id.
    """

    name: str
    raw_dir: str
    doc_ids: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]


class BenchmarkSource(Protocol):
    """Provide benchmark datasets."""

    def load(self, name: str) -> BenchmarkData:
        """Return the dataset ``name``."""
        ...


class BenchmarkResult(BaseModel):
    """Scores of one configuration on one dataset."""

    dataset: str
    config: str
    params: Dict[str, Any]
    suite: str = ""
    documents: int
    chunks: int
    queries: int
    ndcg10: float
    recall10: float
    recall100: float
    hit: Dict[int, float]
    index_seconds: float
    index_cold: bool
    search_seconds: float


class BenchmarkService:
    """Index a benchmark corpus and score the pipeline on its queries.

    The benchmark judges documents while the pipeline returns chunks,
    so a document takes the rank of its best chunk. ``depth`` chunks
    are retrieved per query to have at least 100 distinct documents
    for recall@100 in almost every case.
    """

    def __init__(
        self,
        source: BenchmarkSource,
        factory: PipelineFactory,
        depth: int = 150,
    ):
        """Wire the dataset source and the pipeline factory."""
        self.source = source
        self.factory = factory
        self.depth = depth

    def run(
        self,
        dataset: str,
        configs: Mapping[str, Mapping[str, Any]],
        max_queries: int = 0,
    ) -> List[BenchmarkResult]:
        """Run every configuration on ``dataset``."""
        if not configs:
            raise InvalidInputError("No configuration to benchmark.")
        data = self.source.load(dataset)
        query_ids = list(data.queries)
        if max_queries > 0:
            query_ids = query_ids[:max_queries]
        if not query_ids:
            raise InvalidInputError(f"{dataset} has no judged test query.")
        texts = [data.queries[q] for q in query_ids]
        results: List[BenchmarkResult] = []
        for name, params in configs.items():
            pipeline = self.factory.create(
                {**params, "raw_dir": data.raw_dir}
            )
            report, cold = pipeline.index()
            pipeline.search(texts[:1], self.depth)
            start = time.perf_counter()
            found = pipeline.search(texts, self.depth)
            seconds = time.perf_counter() - start
            ranked = [
                unique_documents(
                    [
                        data.doc_ids.get(
                            PurePath(hit.chunk.file_path).stem, ""
                        )
                        for hit in hits
                    ],
                    skip=query_id,
                )
                for query_id, hits in zip(query_ids, found)
            ]
            results.append(self._score(
                data, name, dict(params), query_ids, ranked,
                report.chunks, report.seconds, cold, seconds,
            ))
        return results

    def _score(
        self,
        data: BenchmarkData,
        config: str,
        params: Dict[str, Any],
        query_ids: Sequence[str],
        ranked: Sequence[Sequence[str]],
        chunks: int,
        index_seconds: float,
        cold: bool,
        search_seconds: float,
    ) -> BenchmarkResult:
        """Average the metrics over the queries."""
        judged = [data.qrels[q] for q in query_ids]
        pairs = list(zip(ranked, judged))
        return BenchmarkResult(
            dataset=data.name,
            config=config,
            params=params,
            documents=len(data.doc_ids),
            chunks=chunks,
            queries=len(query_ids),
            ndcg10=round(mean([ndcg_at_k(r, j, 10) for r, j in pairs]), 4),
            recall10=round(mean([recall_at_k(r, j, 10) for r, j in pairs]), 4),
            recall100=round(
                mean([recall_at_k(r, j, 100) for r, j in pairs]), 4
            ),
            hit={
                k: round(mean([hit_at_k(r, j, k) for r, j in pairs]), 4)
                for k in (1, 3, 5, 10)
            },
            index_seconds=round(index_seconds, 1),
            index_cold=cold,
            search_seconds=round(search_seconds, 2),
        )
