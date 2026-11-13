"""SearchService: mode resolution, validation, caching and batching."""

from typing import Any, Dict, List, Optional

import pytest

from src.application.retrieval import (
    HybridRetriever,
    LexicalRetriever,
    PriorAdjuster,
    RerankingRetriever,
    SemanticRetriever,
)
from src.application.search import SearchService
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.models import Chunk
from src.domain.priors import FEATURE_NAMES, RouterModel
from src.domain.text_analysis import tokenize
from tests.fakes import (
    FakeEmbedder,
    FakeLexicalIndex,
    FakeReranker,
    InMemoryCache,
    InMemoryIndexRepository,
    RecordingProgress,
    make_chunk,
    make_snapshot,
)


def service(
    chunks: List[Chunk],
    mode: str = "auto",
    semantic: bool = False,
    cache: Optional[InMemoryCache] = None,
    batch_size: int = 64,
    fusion: str = "linear",
    reranker: Optional[FakeReranker] = None,
    rerank_weight: float = 0.3,
) -> SearchService:
    """Return a SearchService over an in-memory index of ``chunks``."""
    embedder = FakeEmbedder()
    repo = InMemoryIndexRepository()
    repo.snapshot = make_snapshot(chunks, embedder if semantic else None)
    return SearchService(
        repository=repo,
        analyzer=tokenize,
        embedder=embedder,
        cache=cache,
        mode=mode,
        fusion=fusion,
        progress=RecordingProgress(),
        batch_size=batch_size,
        reranker=reranker,
        rerank_weight=rerank_weight,
    )


def lexical_calls(svc: SearchService) -> int:
    """Return how many times the fake lexical index was queried."""
    lexical = svc.snapshot.lexical
    assert isinstance(lexical, FakeLexicalIndex)
    return lexical.calls


@pytest.mark.parametrize("semantic, mode, retriever", [
    (False, "bm25", LexicalRetriever),
    (True, "hybrid", HybridRetriever),
])
def test_auto_mode_follows_index(
    chunks: List[Chunk], semantic: bool, mode: str, retriever: type
) -> None:
    """``auto`` is hybrid with embeddings, bm25 without."""
    svc = service(chunks, semantic=semantic)
    assert svc.mode == mode
    assert isinstance(svc.retriever, retriever)


@pytest.mark.parametrize("mode, retriever", [
    ("bm25", LexicalRetriever),
    ("semantic", SemanticRetriever),
    ("hybrid", HybridRetriever),
])
def test_explicit_modes(
    chunks: List[Chunk], mode: str, retriever: type
) -> None:
    """An explicit mode selects the matching retriever."""
    svc = service(chunks, mode=mode, semantic=True)
    assert svc.mode == mode
    assert isinstance(svc.retriever, retriever)
    assert svc.search("quantization", 2)


def test_invalid_mode(chunks: List[Chunk]) -> None:
    """An unknown mode is rejected at construction time."""
    with pytest.raises(InvalidInputError, match="Unknown mode"):
        service(chunks, mode="fuzzy")


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_semantic_mode_without_embeddings(
    chunks: List[Chunk], mode: str
) -> None:
    """Asking for dense retrieval on a lexical index is a clear error."""
    svc = service(chunks, mode=mode)
    with pytest.raises(NotFoundError):
        svc.search("lora", 3)


def test_invalid_fusion_surfaces_on_first_search(
    chunks: List[Chunk]
) -> None:
    """An unknown fusion fails when the hybrid retriever is built."""
    svc = service(chunks, semantic=True, fusion="nope")
    with pytest.raises(InvalidInputError, match="Unknown fusion"):
        svc.search("lora", 3)


def test_missing_index() -> None:
    """Searching without an index raises NotFoundError."""
    svc = SearchService(InMemoryIndexRepository(), tokenize, FakeEmbedder())
    with pytest.raises(NotFoundError):
        svc.search("lora", 3)


def test_index_loaded_once(chunks: List[Chunk]) -> None:
    """The snapshot and retriever are loaded lazily and kept."""
    svc = service(chunks)
    assert isinstance(svc.repository, InMemoryIndexRepository)
    assert svc.repository.loads == 0
    svc.search("lora", 2)
    svc.search("cache", 2)
    assert svc.repository.loads == 1
    assert svc.retriever is svc.retriever


@pytest.mark.parametrize("k", [0, -1, True, 2.5])
def test_k_validation(chunks: List[Chunk], k: object) -> None:
    """Non-positive or non-integer k is rejected."""
    svc = service(chunks)
    with pytest.raises(InvalidInputError):
        svc.search_many(["lora"], k)  # type: ignore[arg-type]


def test_blank_queries_get_empty_results(chunks: List[Chunk]) -> None:
    """Blank queries are answered with [] without touching the index."""
    svc = service(chunks)
    results = svc.search_many(["", "   ", "quantization"], 2)
    assert results[0] == [] and results[1] == []
    assert results[2][0].chunk == chunks[2]
    assert lexical_calls(svc) == 1


def test_search_returns_at_most_k(chunks: List[Chunk]) -> None:
    """At most k hits are returned."""
    svc = service(chunks)
    assert len(svc.search("a.py b.md", 1)) == 1
    assert len(svc.search("a.py b.md", 10)) <= len(chunks)


def test_cache_hit_avoids_retriever(chunks: List[Chunk]) -> None:
    """A repeated query is served from the cache with the same hits."""
    cache = InMemoryCache()
    svc = service(chunks, cache=cache)
    first = svc.search("kv cache", 3)
    assert cache.sets == 1
    assert lexical_calls(svc) == 1
    second = svc.search("kv cache", 3)
    assert lexical_calls(svc) == 1
    assert cache.hits == 1
    assert second == first


def test_cache_key_depends_on_k_and_fingerprint(chunks: List[Chunk]) -> None:
    """Different k or a re-indexed corpus miss the cache."""
    cache = InMemoryCache()
    svc = service(chunks, cache=cache)
    svc.search("kv cache", 3)
    svc.search("kv cache", 2)
    assert cache.hits == 0
    other = service(chunks, cache=cache)
    other.snapshot.manifest.fingerprint = "another"
    other.search("kv cache", 3)
    assert cache.hits == 0
    assert len(cache.data) == 3


def test_cache_shared_between_services(chunks: List[Chunk]) -> None:
    """A new service over the same index reuses cached results."""
    cache = InMemoryCache()
    first = service(chunks, cache=cache).search("lora adapter", 2)
    svc = service(chunks, cache=cache)
    assert svc.search("lora adapter", 2) == first
    assert lexical_calls(svc) == 0


def test_batching(chunks: List[Chunk]) -> None:
    """Queries are sent to the retriever in batches of batch_size."""
    svc = service(chunks, batch_size=2)
    queries = ["lora", "cache", "fp8", "draft", "memory"]
    results = svc.search_many(queries, 1)
    assert len(results) == 5
    assert isinstance(svc.progress, RecordingProgress)
    assert svc.progress.loops == [("Searching", 3)]
    assert results == [svc.search(q, 1) for q in queries]


def test_batching_with_partial_cache(chunks: List[Chunk]) -> None:
    """Cached and uncached queries are merged back in input order."""
    cache = InMemoryCache()
    svc = service(chunks, cache=cache)
    svc.search("cache", 2)
    results = svc.search_many(["lora", "cache", "fp8"], 2)
    assert cache.hits == 1
    assert results[1] == svc.search("cache", 2)
    assert results[0][0].chunk == chunks[0]
    assert results[2][0].chunk == chunks[2]


def test_reranker_wraps_first_stage(chunks: List[Chunk]) -> None:
    """With a reranker the first-stage retriever is wrapped."""
    svc = service(chunks, mode="bm25", reranker=FakeReranker("FP8"),
                  rerank_weight=1.0)
    retriever = svc.retriever
    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever.base, LexicalRetriever)
    top = svc.search("lora adapter fp8 quantization", 2)
    assert top[0].chunk.text.startswith("FP8")


def test_reranker_changes_cache_key(chunks: List[Chunk]) -> None:
    """Reranked and plain results never share a cache entry."""
    cache = InMemoryCache()
    plain = service(chunks, mode="bm25", cache=cache)
    reranked = service(chunks, mode="bm25", cache=cache,
                       reranker=FakeReranker("FP8"))
    assert plain._key("lora", 3) != reranked._key("lora", 3)
    other = service(chunks, mode="bm25", cache=cache,
                    reranker=FakeReranker("FP8"), rerank_weight=0.9)
    assert reranked._key("lora", 3) != other._key("lora", 3)


def test_invalid_rerank_weight(chunks: List[Chunk]) -> None:
    """An out-of-range weight is rejected when the service is built."""
    with pytest.raises(InvalidInputError):
        service(chunks, reranker=FakeReranker(), rerank_weight=2.0)


# -------------------------------------------------------------- priors


def router(bias: float) -> RouterModel:
    """Return a router answering code (bias > 0) or docs (bias < 0)."""
    return RouterModel(weights=[0.0] * len(FEATURE_NAMES), bias=bias)


def tuned(
    chunks: List[Chunk],
    mode: str = "bm25",
    cache: Optional[InMemoryCache] = None,
    reranker: Optional[FakeReranker] = None,
    **priors: Any,
) -> SearchService:
    """Return a SearchService over a semantic index with prior settings."""
    embedder = FakeEmbedder()
    repo = InMemoryIndexRepository()
    repo.snapshot = make_snapshot(chunks, embedder)
    return SearchService(
        repository=repo, analyzer=tokenize, embedder=embedder, cache=cache,
        mode=mode, reranker=reranker, **priors,
    )


@pytest.mark.parametrize("mode", ["bm25", "semantic", "hybrid"])
def test_no_adjuster_when_priors_are_neutral(
    chunks: List[Chunk], mode: str
) -> None:
    """Default prior settings leave the retrievers without adjuster."""
    retriever = tuned(chunks, mode).retriever
    assert isinstance(
        retriever, (LexicalRetriever, SemanticRetriever, HybridRetriever)
    )
    assert retriever.adjuster is None


def test_route_strength_without_router_is_neutral(chunks: List[Chunk]) -> None:
    """Routing needs a trained router; alone the strength does nothing."""
    svc = tuned(chunks, route_strength=0.7)
    assert svc.route_strength == 0.0
    retriever = svc.retriever
    assert isinstance(retriever, LexicalRetriever)
    assert retriever.adjuster is None
    assert svc._key("lora", 3) == tuned(chunks)._key("lora", 3)


def test_prior_settings_reach_the_adjuster(chunks: List[Chunk]) -> None:
    """Router, strengths and weights configure one PriorAdjuster."""
    model = router(20.0)
    svc = tuned(chunks, router=model, route_strength=0.3, test_weight=0.5,
                file_boost=0.2)
    retriever = svc.retriever
    assert isinstance(retriever, LexicalRetriever)
    adjuster = retriever.adjuster
    assert isinstance(adjuster, PriorAdjuster)
    assert adjuster.router is model
    assert (adjuster.route_strength, adjuster.file_boost) == (0.3, 0.2)
    assert adjuster.active is True


def test_file_boost_changes_the_results(chunks: List[Chunk]) -> None:
    """With file aggregation the sibling chunk of a match is returned."""
    assert [h.chunk for h in tuned(chunks).search("KVCacheManager", 5)] == [
        chunks[1],
    ]
    boosted = tuned(chunks, file_boost=0.5).search("KVCacheManager", 5)
    assert [h.chunk for h in boosted] == [chunks[1], chunks[0]]


@pytest.mark.parametrize("bias, first", [(20.0, 0), (-20.0, 2)])
def test_routing_changes_the_results(
    chunks: List[Chunk], bias: float, first: int
) -> None:
    """The router decides which side wins an ambiguous query."""
    svc = tuned(chunks, router=router(bias), route_strength=0.8)
    assert svc.search("adapter quantization", 4)[0].chunk == chunks[first]


def test_test_weight_changes_the_results(chunks: List[Chunk]) -> None:
    """Chunks of test files are demoted by test_weight."""
    test_chunk = make_chunk("tests/test_b.py", 0, "fp8 quantization fp8",
                            "def test_fp8_quantization", "python")
    corpus = [*chunks, test_chunk]
    query = "fp8 quantization"
    assert tuned(corpus).search(query, 1)[0].chunk == test_chunk
    assert tuned(corpus, test_weight=0.1).search(query, 1)[0].chunk == (
        chunks[2]
    )


def test_semantic_mode_gets_adjuster_and_query_prefix(
    chunks: List[Chunk]
) -> None:
    """Semantic search embeds prefixed queries and adjusts the cosines."""
    svc = tuned(chunks, "semantic", file_boost=0.1, query_prefix="query: ")
    retriever = svc.retriever
    assert isinstance(retriever, SemanticRetriever)
    assert isinstance(retriever.adjuster, PriorAdjuster)
    svc.search("lora", 2)
    assert isinstance(svc.embedder, FakeEmbedder)
    assert svc.embedder.encoded[-1] == "query: lora"


@pytest.mark.parametrize("fusion", ["linear", "rrf"])
def test_hybrid_mode_adjusts_once_at_the_fusion(
    chunks: List[Chunk], fusion: str
) -> None:
    """The hybrid retriever owns the adjuster; its parts stay plain."""
    svc = tuned(chunks, "hybrid", fusion=fusion, file_boost=0.1,
                query_prefix="query: ")
    retriever = svc.retriever
    assert isinstance(retriever, HybridRetriever)
    assert isinstance(retriever.adjuster, PriorAdjuster)
    assert retriever.lexical.adjuster is None
    assert retriever.semantic.adjuster is None
    assert retriever.semantic.query_prefix == "query: "
    assert svc.search("lora adapter", 2)


def test_reranker_wraps_the_adjusted_first_stage(chunks: List[Chunk]) -> None:
    """Priors apply before reranking."""
    svc = tuned(chunks, reranker=FakeReranker("FP8"), file_boost=0.3)
    retriever = svc.retriever
    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever.base, LexicalRetriever)
    assert isinstance(retriever.base.adjuster, PriorAdjuster)


@pytest.mark.parametrize("priors", [
    {"file_boost": 0.2},
    {"test_weight": 0.5},
    {"query_prefix": "query: "},
    {"router": router(1.0), "route_strength": 0.3},
])
def test_prior_settings_change_the_cache_key(
    chunks: List[Chunk], priors: Dict[str, Any]
) -> None:
    """Results cached with other prior settings are never reused."""
    cache = InMemoryCache()
    tuned(chunks, cache=cache).search("lora adapter", 3)
    changed = tuned(chunks, cache=cache, **priors)
    changed.search("lora adapter", 3)
    assert cache.hits == 0
    assert cache.sets == 2
    tuned(chunks, cache=cache, **priors).search("lora adapter", 3)
    assert cache.hits == 1


def test_cache_key_depends_on_the_router_model(chunks: List[Chunk]) -> None:
    """A retrained router or another strength invalidates cached results."""
    keys = {
        tuned(chunks, router=model, route_strength=strength)._key("q", 3)
        for model in (router(1.0), router(2.0))
        for strength in (0.3, 0.6)
    }
    assert len(keys) == 4
    assert tuned(chunks, router=router(1.0))._key("q", 3) not in keys


def test_negative_prior_is_rejected_on_first_search(
    chunks: List[Chunk]
) -> None:
    """Invalid prior values surface as a user error when searching."""
    svc = tuned(chunks, file_boost=-0.5)
    with pytest.raises(InvalidInputError, match="must be >= 0"):
        svc.search("lora", 2)
