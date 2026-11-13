"""Use case: retrieve the top-k chunks for one or many queries."""

import json
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from src.application.caching import cache_key
from src.application.ports import (
    Analyzer,
    Cache,
    Embedder,
    IndexRepository,
    IndexSnapshot,
    NoProgress,
    Progress,
    Reranker,
)
from src.application.retrieval import (
    MODES,
    HybridRetriever,
    LexicalRetriever,
    PriorAdjuster,
    RerankingRetriever,
    Retriever,
    SemanticRetriever,
    validate_rerank,
)
from src.domain.errors import InvalidInputError
from src.domain.models import ScoredChunk
from src.domain.priors import RouterModel
from src.domain.validation import validate_k

AUTO = "auto"


class SearchService:
    """Select a retriever, run queries in batches and cache results.

    The index and the retriever are loaded lazily and kept, so a
    long-running process pays the cold start once. Cached results are
    keyed by the index fingerprint, so re-indexing invalidates them.
    """

    def __init__(
        self,
        repository: IndexRepository,
        analyzer: Analyzer,
        embedder: Embedder,
        cache: Optional[Cache] = None,
        mode: str = AUTO,
        semantic_weight: float = 0.4,
        fusion: str = "linear",
        progress: Optional[Progress] = None,
        batch_size: int = 64,
        reranker: Optional[Reranker] = None,
        rerank_depth: int = 10,
        rerank_weight: float = 0.3,
        router: Optional[RouterModel] = None,
        route_strength: float = 0.0,
        test_weight: float = 1.0,
        file_boost: float = 0.0,
        query_prefix: str = "",
        identifier_weight: Optional[float] = None,
        identifier_saturation: float = 0.15,
    ):
        """Configure the service; nothing heavy is loaded yet."""
        if mode not in MODES + (AUTO,):
            raise InvalidInputError(
                f"Unknown mode {mode!r}; use one of "
                f"{', '.join(MODES + (AUTO,))}."
            )
        self.repository = repository
        self.analyzer = analyzer
        self.embedder = embedder
        self.cache = cache
        self.requested_mode = mode
        self.semantic_weight = semantic_weight
        self.fusion = fusion
        self.progress = progress or NoProgress()
        self.batch_size = batch_size
        validate_rerank(rerank_depth, rerank_weight)
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.rerank_weight = rerank_weight
        self.router = router
        self.route_strength = route_strength if router else 0.0
        self.test_weight = test_weight
        self.file_boost = file_boost
        self.query_prefix = query_prefix
        self.identifier_weight = identifier_weight
        self.identifier_saturation = identifier_saturation
        self._snapshot: Optional[IndexSnapshot] = None
        self._retriever: Optional[Retriever] = None
        self._lock = threading.RLock()

    @property
    def snapshot(self) -> IndexSnapshot:
        """Return the loaded index."""
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self.repository.load()
            return self._snapshot

    @property
    def mode(self) -> str:
        """Return the effective retrieval mode."""
        if self.requested_mode != AUTO:
            return self.requested_mode
        return "hybrid" if self.snapshot.embeddings is not None else "bm25"

    @property
    def retriever(self) -> Retriever:
        """Return the retriever implementing the effective mode."""
        with self._lock:
            if self._retriever is None:
                self._retriever = self._build_retriever()
            return self._retriever

    def _build_retriever(self) -> Retriever:
        """Instantiate the retriever, wrapped by the reranker if any."""
        base = self._first_stage()
        if self.reranker is None:
            return base
        return RerankingRetriever(
            base, self.reranker, self.rerank_depth, self.rerank_weight
        )

    def _first_stage(self) -> Retriever:
        """Instantiate the retriever for the effective mode."""
        snapshot = self.snapshot
        priors = PriorAdjuster(
            snapshot, self.router, self.route_strength,
            self.test_weight, self.file_boost,
        )
        adjuster = priors if priors.active else None
        if self.mode == "bm25":
            return LexicalRetriever(snapshot, self.analyzer, adjuster)
        if self.mode == "semantic":
            return SemanticRetriever(
                snapshot, self.embedder, adjuster, self.query_prefix
            )
        return HybridRetriever(
            LexicalRetriever(snapshot, self.analyzer),
            SemanticRetriever(
                snapshot, self.embedder, query_prefix=self.query_prefix
            ),
            self.semantic_weight, self.fusion, adjuster=adjuster,
            identifier_weight=self.identifier_weight,
            identifier_saturation=self.identifier_saturation,
        )

    def _key(self, query: str, k: int) -> str:
        """Return the cache key of a search."""
        return cache_key(
            "search", self.snapshot.manifest.fingerprint, query, k,
            self.mode, self.semantic_weight, self.fusion,
            self._rerank_signature(),
            [self.route_strength, self.test_weight, self.file_boost,
             self.query_prefix,
             self.router.model_dump() if self.router else None,
             *([self.identifier_weight, self.identifier_saturation]
               if self.identifier_weight is not None else [])],
        )

    def _rerank_signature(self) -> Optional[List[object]]:
        """Describe the reranking stage for cache keys (None if off)."""
        if self.reranker is None:
            return None
        return [
            self.reranker.model_name, self.rerank_depth, self.rerank_weight
        ]

    def search(self, query: str, k: int) -> List[ScoredChunk]:
        """Return the top-``k`` chunks for one query."""
        return self.search_many([query], k)[0]

    def search_many(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the top-``k`` chunks per query; blank queries get []."""
        validate_k(k)
        with self._lock:
            results: Dict[int, List[ScoredChunk]] = {}
            misses: List[Tuple[int, str]] = []
            for pos, query in enumerate(queries):
                if not query.strip():
                    results[pos] = []
                    continue
                cached = self.cache.get(self._key(query, k)) \
                    if self.cache else None
                if cached is None:
                    misses.append((pos, query))
                else:
                    results[pos] = self._decode(cached)
            batches = [
                misses[i:i + self.batch_size]
                for i in range(0, len(misses), self.batch_size)
            ]
            for batch in self.progress.track(
                batches, len(batches), "Searching"
            ):
                found = self.retriever.search_batch([q for _, q in batch], k)
                for (pos, query), hits in zip(batch, found):
                    results[pos] = hits
                    if self.cache:
                        self.cache.set(self._key(query, k), self._encode(hits))
            return [results[i] for i in range(len(queries))]

    def _encode(self, hits: List[ScoredChunk]) -> str:
        """Serialise hits as (index position, score) pairs."""
        snapshot = self.snapshot
        return json.dumps(
            [[snapshot.position_of(h.chunk), h.score] for h in hits]
        )

    def _decode(self, payload: str) -> List[ScoredChunk]:
        """Rebuild hits from cached (position, score) pairs."""
        chunks = self.snapshot.chunks
        return [
            ScoredChunk(chunk=chunks[int(i)], score=float(s))
            for i, s in json.loads(payload)
        ]
