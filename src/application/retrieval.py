"""Retrievers: lexical, semantic and hybrid ranking over a snapshot."""

from typing import List, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from src.application.ports import (
    Analyzer,
    Embedder,
    FloatArray,
    IndexSnapshot,
    Reranker,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.models import ScoredChunk
from src.domain.priors import RouterModel, is_test_path
from src.domain.text_analysis import adaptive_weight, identifier_share
from src.domain.ranking import (
    FUSIONS,
    Ranked,
    apply_priors,
    NORMALISED_FUSIONS,
    linear_scores,
    normalised_fusion,
    ranked,
    rerank_fusion,
    rrf_fusion,
    top_k,
)

MODES = ("bm25", "semantic", "hybrid")


class Retriever(Protocol):
    """Rank indexed chunks for queries."""

    def search_batch(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the ``k`` best chunks for each query."""
        ...


class ScoreAdjuster(Protocol):
    """Modify the score of every chunk for a query before ranking."""

    def adjust(self, query: str, scores: FloatArray) -> FloatArray:
        """Return the adjusted scores (same shape)."""
        ...


class PriorAdjuster:
    """Apply query routing, a test-file prior and file aggregation.

    - ``route_strength`` scales code chunks by ``1 + s·(2p − 1)`` and
      text chunks by ``1 − s·(2p − 1)``, where ``p`` is the router's
      probability that the answer is in code.
    - ``test_weight`` multiplies the score of chunks in test files.
    - ``file_boost`` adds a share of the best score of the file.
    """

    def __init__(
        self,
        snapshot: IndexSnapshot,
        router: Optional[RouterModel] = None,
        route_strength: float = 0.0,
        test_weight: float = 1.0,
        file_boost: float = 0.0,
    ):
        """Precompute the per-chunk features of ``snapshot``."""
        if route_strength < 0 or test_weight < 0 or file_boost < 0:
            raise InvalidInputError(
                "route_strength, test_weight and file_boost must be >= 0."
            )
        self.router = router
        self.route_strength = route_strength if router else 0.0
        self.file_boost = file_boost
        chunks = snapshot.chunks
        self.is_code = np.asarray(
            [c.kind == "python" for c in chunks], dtype=bool
        )
        self.static: FloatArray = np.ones(len(chunks), dtype=np.float32)
        if test_weight != 1.0:
            tests = [is_test_path(c.file_path) for c in chunks]
            self.static[np.asarray(tests, dtype=bool)] = test_weight
        ids: dict[str, int] = {}
        self.file_ids: npt.NDArray[np.int64] = np.asarray(
            [ids.setdefault(c.file_path, len(ids)) for c in chunks],
            dtype=np.int64,
        )

    @property
    def active(self) -> bool:
        """Tell whether the adjuster changes anything."""
        return bool(
            self.route_strength or self.file_boost
            or (self.static != 1.0).any()
        )

    def adjust(self, query: str, scores: FloatArray) -> FloatArray:
        """Return the scores after priors and file aggregation."""
        multipliers: FloatArray = self.static
        if self.route_strength and self.router is not None:
            shift = self.route_strength * (
                2.0 * self.router.p_code(query) - 1.0
            )
            multipliers = multipliers * np.where(
                self.is_code, max(1.0 + shift, 0.0), max(1.0 - shift, 0.0)
            ).astype(np.float32)
        return apply_priors(
            scores, multipliers, self.file_ids, self.file_boost
        )


def _hits(snapshot: IndexSnapshot, ranked: Ranked) -> List[ScoredChunk]:
    """Turn (position, score) pairs into scored chunks."""
    return [
        ScoredChunk(chunk=snapshot.chunks[i], score=s) for i, s in ranked
    ]


class LexicalRetriever:
    """Rank chunks with the lexical index (BM25)."""

    def __init__(
        self,
        snapshot: IndexSnapshot,
        analyzer: Analyzer,
        adjuster: Optional[ScoreAdjuster] = None,
    ):
        """Search ``snapshot`` with queries analysed by ``analyzer``."""
        self.snapshot = snapshot
        self.analyzer = analyzer
        self.adjuster = adjuster

    def scores(self, query: str) -> FloatArray:
        """Return the lexical score of every chunk."""
        return self.snapshot.lexical.scores(self.analyzer(query))

    def search_batch(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the ``k`` best chunks with a positive score per query."""
        results: List[List[ScoredChunk]] = []
        for query in queries:
            scores = self.scores(query)
            if self.adjuster is not None:
                scores = self.adjuster.adjust(query, scores)
            best = [
                (i, float(scores[i])) for i in top_k(scores, k)
                if scores[i] > 0
            ]
            results.append(_hits(self.snapshot, best))
        return results


class SemanticRetriever:
    """Rank chunks by cosine similarity of their embeddings."""

    def __init__(
        self,
        snapshot: IndexSnapshot,
        embedder: Embedder,
        adjuster: Optional[ScoreAdjuster] = None,
        query_prefix: str = "",
    ):
        """Search the embedding matrix of ``snapshot``."""
        if snapshot.embeddings is None:
            raise NotFoundError(
                "This index has no embeddings. Re-run 'index' with "
                "--semantic=True or use --mode=bm25."
            )
        self.snapshot = snapshot
        self.matrix: FloatArray = snapshot.embeddings
        self.embedder = embedder
        self.adjuster = adjuster
        self.query_prefix = query_prefix

    def scores_batch(self, queries: Sequence[str]) -> FloatArray:
        """Return a (queries x chunks) cosine-similarity matrix."""
        vectors = self.embedder.encode(
            [self.query_prefix + q for q in queries]
        )
        return vectors @ self.matrix.T

    def search_batch(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the ``k`` most similar chunks per query."""
        results: List[List[ScoredChunk]] = []
        for query, row in zip(queries, self.scores_batch(queries)):
            if self.adjuster is not None:
                row = self.adjuster.adjust(query, row)
            results.append(_hits(self.snapshot, ranked(row, k)))
        return results


class HybridRetriever:
    """Fuse lexical and semantic rankings into one list.

    ``linear`` fusion (default) adds max-normalised BM25 and a weighted
    cosine; ``rrf`` applies Reciprocal Rank Fusion to the top ``depth``
    results of each ranking; ``convex`` and ``dbsf`` normalise both
    scores over those candidates before adding them (see
    :func:`~src.domain.ranking.normalised_fusion`).
    """

    def __init__(
        self,
        lexical: LexicalRetriever,
        semantic: SemanticRetriever,
        semantic_weight: float = 0.4,
        fusion: str = "linear",
        depth: int = 100,
        adjuster: Optional[ScoreAdjuster] = None,
        identifier_weight: Optional[float] = None,
        identifier_saturation: float = 0.15,
    ):
        """Combine two retrievers over the same snapshot.

        With ``identifier_weight`` the semantic weight depends on the
        query: ``semantic_weight`` for prose, sliding to
        ``identifier_weight`` as code identifiers make up
        ``identifier_saturation`` of its words (see
        :func:`~src.domain.text_analysis.adaptive_weight`).
        """
        if fusion not in FUSIONS:
            raise InvalidInputError(
                f"Unknown fusion {fusion!r}; use one of {', '.join(FUSIONS)}."
            )
        self.lexical = lexical
        self.semantic = semantic
        self.semantic_weight = semantic_weight
        self.fusion = fusion
        self.depth = depth
        self.adjuster = adjuster
        self.identifier_weight = identifier_weight
        self.identifier_saturation = identifier_saturation

    def weight_for(self, query: str) -> float:
        """Return the semantic weight used for ``query``."""
        if self.identifier_weight is None:
            return self.semantic_weight
        return adaptive_weight(
            self.semantic_weight, self.identifier_weight,
            identifier_share(query), self.identifier_saturation,
        )

    def _adjust(self, query: str, scores: FloatArray) -> FloatArray:
        """Apply the adjuster, if any."""
        if self.adjuster is None:
            return scores
        return self.adjuster.adjust(query, scores)

    def search_batch(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the ``k`` best fused chunks per query."""
        dense = self.semantic.scores_batch(queries)
        results: List[List[ScoredChunk]] = []
        for query, dense_row in zip(queries, dense):
            lexical_row = self.lexical.scores(query)
            weight = self.weight_for(query)
            if self.fusion == "linear":
                fused = self._adjust(query, linear_scores(
                    lexical_row, dense_row, weight
                ))
                best = ranked(fused, k)
            elif self.fusion in NORMALISED_FUSIONS:
                best = normalised_fusion(
                    self._adjust(query, lexical_row),
                    self._adjust(query, dense_row),
                    weight, max(self.depth, k), k,
                    self.fusion,
                )
            else:
                depth = max(self.depth, k)
                lexical_row = self._adjust(query, lexical_row)
                dense_row = self._adjust(query, dense_row)
                lexical_rank = [
                    i for i in top_k(lexical_row, depth) if lexical_row[i] > 0
                ]
                best = rrf_fusion(
                    [(1.0, lexical_rank),
                     (weight, top_k(dense_row, depth))],
                    k,
                )
            results.append(_hits(self.lexical.snapshot, best))
        return results


def validate_rerank(depth: int, weight: float) -> None:
    """Check the reranking depth and weight."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise InvalidInputError(
            f"Rerank depth must be a positive integer (got {depth!r})."
        )
    if not 0.0 <= weight <= 1.0:
        raise InvalidInputError(
            f"Rerank weight must be between 0 and 1 (got {weight!r})."
        )


class RerankingRetriever:
    """Re-order the head of another retriever's ranking.

    The ``depth`` best first-stage candidates are scored by the reranker
    (e.g. a cross-encoder reading query and passage together). Its score
    is blended with the first-stage score, ``weight`` being the share of
    the reranker. Candidates beyond ``depth`` keep their order after
    the reranked head, with a score of 0.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        depth: int = 10,
        weight: float = 0.3,
    ):
        """Wrap ``base`` with a second-stage ``reranker``."""
        validate_rerank(depth, weight)
        self.base = base
        self.reranker = reranker
        self.depth = depth
        self.weight = weight

    def search_batch(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return the ``k`` best chunks per query after reranking."""
        candidates = self.base.search_batch(queries, max(k, self.depth))
        heads = [hits[:self.depth] for hits in candidates]
        pairs = [
            (query, hit.chunk.search_text())
            for query, head in zip(queries, heads) if len(head) > 1
            for hit in head
        ]
        scores = self.reranker.score(pairs) if pairs else []
        results: List[List[ScoredChunk]] = []
        offset = 0
        for head, hits in zip(heads, candidates):
            if len(head) > 1:
                second = scores[offset:offset + len(head)]
                offset += len(head)
                ranked = rerank_fusion([h.score for h in head], second,
                                       self.weight)
                head = [
                    ScoredChunk(chunk=head[i].chunk, score=s)
                    for i, s in ranked
                ]
            tail = [
                ScoredChunk(chunk=h.chunk, score=0.0)
                for h in hits[self.depth:]
            ]
            results.append((head + tail)[:k])
        return results
