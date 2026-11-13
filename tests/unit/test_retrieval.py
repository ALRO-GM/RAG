"""Lexical, semantic and hybrid retrievers over an in-memory snapshot."""

from typing import List, Sequence

import numpy as np
import pytest

from src.application.ports import FloatArray
from src.application.retrieval import (
    HybridRetriever,
    LexicalRetriever,
    PriorAdjuster,
    RerankingRetriever,
    SemanticRetriever,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.models import Chunk
from src.domain.priors import FEATURE_NAMES, RouterModel
from src.domain.text_analysis import tokenize
from tests.fakes import (
    FakeEmbedder,
    FakeReranker,
    make_chunk,
    make_snapshot,
)


def test_lexical_ranks_matching_chunk_first(chunks: List[Chunk]) -> None:
    """The chunk sharing the query terms ranks first."""
    retriever = LexicalRetriever(make_snapshot(chunks), tokenize)
    [hits] = retriever.search_batch(["quantization memory"], 3)
    assert hits[0].chunk == chunks[2]
    assert all(h.score > 0 for h in hits)


def test_lexical_drops_zero_scores(chunks: List[Chunk]) -> None:
    """Chunks without any query term are not returned."""
    retriever = LexicalRetriever(make_snapshot(chunks), tokenize)
    assert retriever.search_batch(["zzzunknown"], 3) == [[]]
    [hits] = retriever.search_batch(["KVCacheManager"], 10)
    assert [h.chunk for h in hits] == [chunks[1]]


def test_lexical_batch_and_k(chunks: List[Chunk]) -> None:
    """One result list per query, each at most k long, scores sorted."""
    retriever = LexicalRetriever(make_snapshot(chunks), tokenize)
    results = retriever.search_batch(["a.py lora", "b.md"], 1)
    assert len(results) == 2
    assert all(len(r) == 1 for r in results)
    [hits] = retriever.search_batch(["b.md model"], 10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_semantic_requires_embeddings(chunks: List[Chunk]) -> None:
    """A lexical-only snapshot cannot be searched semantically."""
    with pytest.raises(NotFoundError, match="no embeddings"):
        SemanticRetriever(make_snapshot(chunks), FakeEmbedder())


def test_semantic_ranks_by_cosine(chunks: List[Chunk]) -> None:
    """The most similar embedding ranks first; all k are returned."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    retriever = SemanticRetriever(snapshot, embedder)
    calls = embedder.calls
    results = retriever.search_batch(
        [chunks[3].search_text(), chunks[0].search_text()], 4
    )
    assert embedder.calls == calls + 1, "queries are encoded in one batch"
    assert results[0][0].chunk == chunks[3]
    assert results[1][0].chunk == chunks[0]
    assert results[0][0].score == pytest.approx(1.0)
    assert len(results[0]) == 4


@pytest.mark.parametrize("fusion", ["linear", "rrf", "convex", "dbsf"])
def test_hybrid_fusions(chunks: List[Chunk], fusion: str) -> None:
    """Every fusion ranks the lexically and semantically best chunk first."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    hybrid = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder),
        semantic_weight=0.5,
        fusion=fusion,
    )
    results = hybrid.search_batch(
        ["speculative decoding draft model", "FP8 quantization"], 2
    )
    assert [r[0].chunk for r in results] == [chunks[3], chunks[2]]
    assert all(len(r) == 2 for r in results)


def test_hybrid_rrf_ignores_zero_lexical_scores(chunks: List[Chunk]) -> None:
    """With RRF, chunks without lexical match get only the dense term."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    hybrid = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder),
        semantic_weight=1.0, fusion="rrf",
    )
    [hits] = hybrid.search_batch(["KVCacheManager"], 4)
    assert hits[0].chunk == chunks[1]
    assert hits[0].score == pytest.approx(2 / 61)
    assert all(h.score <= 1 / 62 for h in hits[1:])


def test_hybrid_rejects_unknown_fusion(chunks: List[Chunk]) -> None:
    """An unknown fusion strategy is a user error."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    with pytest.raises(InvalidInputError, match="Unknown fusion"):
        HybridRetriever(
            LexicalRetriever(snapshot, tokenize),
            SemanticRetriever(snapshot, embedder),
            fusion="max",
        )


def lexical(chunks: List[Chunk]) -> LexicalRetriever:
    """Return a lexical retriever over ``chunks``."""
    return LexicalRetriever(make_snapshot(chunks), tokenize)


def test_reranker_promotes_preferred_chunk(chunks: List[Chunk]) -> None:
    """A strong reranker score moves a lower candidate to the top."""
    base = lexical(chunks)
    query = "lora adapter fp8 quantization"
    assert base.search_batch([query], 2)[0][0].chunk.file_path == "a.py"
    reranked = RerankingRetriever(base, FakeReranker("FP8"), weight=1.0)
    top = reranked.search_batch([query], 2)[0]
    assert top[0].chunk.text.startswith("FP8")
    assert len(top) == 2


def test_reranker_weight_zero_keeps_first_stage(chunks: List[Chunk]) -> None:
    """With weight 0 the first-stage order is preserved."""
    base = lexical(chunks)
    query = "lora adapter fp8 quantization"
    expected = [h.chunk for h in base.search_batch([query], 3)[0]]
    reranked = RerankingRetriever(base, FakeReranker("FP8"), weight=0.0)
    assert [h.chunk for h in reranked.search_batch([query], 3)[0]] == expected


def test_reranker_scores_only_head_in_one_batch(chunks: List[Chunk]) -> None:
    """Only ``depth`` candidates per query are scored, in a single call."""
    fake = FakeReranker("FP8")
    reranked = RerankingRetriever(
        SemanticRetriever(make_snapshot(chunks, FakeEmbedder()),
                          FakeEmbedder()),
        fake, depth=2, weight=1.0,
    )
    results = reranked.search_batch(["lora", "cache blocks"], 4)
    assert len(fake.calls) == 1
    assert len(fake.calls[0]) == 4
    assert [len(r) for r in results] == [4, 4]
    assert all(h.score == 0.0 for r in results for h in r[2:])


def test_reranker_skips_single_candidate(chunks: List[Chunk]) -> None:
    """Nothing to reorder with fewer than two candidates."""
    fake = FakeReranker("FP8")
    reranked = RerankingRetriever(lexical(chunks), fake)
    assert len(reranked.search_batch(["speculative"], 5)[0]) == 1
    assert reranked.search_batch(["zzzz"], 5) == [[]]
    assert fake.calls == []


@pytest.mark.parametrize("depth, weight", [
    (0, 0.3), (-1, 0.3), (True, 0.3), (2.5, 0.3), (10, -0.1), (10, 1.5),
])
def test_reranker_rejects_bad_parameters(
    chunks: List[Chunk], depth: object, weight: float
) -> None:
    """Depth must be a positive integer and weight within [0, 1]."""
    with pytest.raises(InvalidInputError):
        RerankingRetriever(
            lexical(chunks), FakeReranker(),
            depth, weight,  # type: ignore[arg-type]
        )


# -------------------------------------------------------------- priors

TEST_CHUNK = make_chunk(
    "tests/test_a.py", 0,
    "def test_load_lora_adapter(): load_lora_adapter('lora adapter')",
    "def test_load_lora_adapter", "python",
)


def router(bias: float) -> RouterModel:
    """Return a router answering code (bias > 0) or docs (bias < 0)."""
    return RouterModel(weights=[0.0] * len(FEATURE_NAMES), bias=bias)


CODE, DOCS = router(20.0), router(-20.0)


def ones(n: int) -> FloatArray:
    """Return a vector of ``n`` ones."""
    return np.ones(n, dtype=np.float32)


class ScriptedAdjuster:
    """Multiply the scores by fixed factors and record the queries."""

    def __init__(self, factors: Sequence[float]):
        """Scale chunk ``i`` by ``factors[i]``."""
        self.factors = np.asarray(factors, dtype=np.float32)
        self.queries: List[str] = []

    def adjust(self, query: str, scores: FloatArray) -> FloatArray:
        """Return the scaled scores."""
        self.queries.append(query)
        scaled: FloatArray = scores * self.factors
        return scaled


def test_prior_adjuster_neutral_settings_change_nothing(
    chunks: List[Chunk]
) -> None:
    """With the default values the adjuster is inactive and transparent."""
    adjuster = PriorAdjuster(make_snapshot(chunks))
    assert adjuster.active is False
    scores = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    assert adjuster.adjust("any", scores).tolist() == scores.tolist()


@pytest.mark.parametrize("route_strength, test_weight, file_boost", [
    (-0.1, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, -0.5),
])
def test_prior_adjuster_rejects_negative_values(
    chunks: List[Chunk], route_strength: float, test_weight: float,
    file_boost: float,
) -> None:
    """Negative strengths and weights are user errors."""
    with pytest.raises(InvalidInputError, match="must be >= 0"):
        PriorAdjuster(
            make_snapshot(chunks), CODE, route_strength, test_weight,
            file_boost,
        )


def test_prior_adjuster_route_strength_needs_a_router(
    chunks: List[Chunk]
) -> None:
    """Without a trained router the routing strength is ignored."""
    adjuster = PriorAdjuster(make_snapshot(chunks), None, route_strength=0.5)
    assert adjuster.route_strength == 0.0
    assert adjuster.active is False
    assert adjuster.adjust("any", ones(4)).tolist() == [1.0] * 4


@pytest.mark.parametrize("model, expected", [
    (CODE, [1.5, 1.5, 0.5, 0.5]),
    (DOCS, [0.5, 0.5, 1.5, 1.5]),
    (router(0.0), [1.0, 1.0, 1.0, 1.0]),
])
def test_prior_adjuster_routing_scales_chunks_by_kind(
    chunks: List[Chunk], model: RouterModel, expected: List[float]
) -> None:
    """Python chunks gain what text chunks lose, following the router."""
    adjuster = PriorAdjuster(make_snapshot(chunks), model, route_strength=0.5)
    assert adjuster.active is True
    assert adjuster.adjust("any", ones(4)).tolist() == pytest.approx(expected)


def test_prior_adjuster_routing_never_goes_negative(
    chunks: List[Chunk]
) -> None:
    """A strength above 1 clamps the losing side at zero."""
    adjuster = PriorAdjuster(make_snapshot(chunks), CODE, route_strength=3.0)
    assert adjuster.adjust("any", ones(4)).tolist() == pytest.approx(
        [4.0, 4.0, 0.0, 0.0]
    )


def test_prior_adjuster_test_weight_scales_test_files(
    chunks: List[Chunk]
) -> None:
    """Only the chunks of test files are multiplied by test_weight."""
    snapshot = make_snapshot([*chunks, TEST_CHUNK])
    adjuster = PriorAdjuster(snapshot, test_weight=0.25)
    assert adjuster.active is True
    assert adjuster.adjust("any", ones(5)).tolist() == [
        1.0, 1.0, 1.0, 1.0, 0.25,
    ]
    assert PriorAdjuster(make_snapshot(chunks), test_weight=0.25).active \
        is False, "no test file in the index: nothing to change"


def test_prior_adjuster_file_boost_shares_the_file_best(
    chunks: List[Chunk]
) -> None:
    """Chunks of one file all gain a share of its best score."""
    adjuster = PriorAdjuster(make_snapshot(chunks), file_boost=0.5)
    assert adjuster.active is True
    scores = np.asarray([1.0, 0.0, 0.2, 0.6], dtype=np.float32)
    assert adjuster.adjust("any", scores).tolist() == pytest.approx(
        [1.5, 0.5, 0.5, 0.9]
    )


def test_prior_adjuster_combines_routing_tests_and_files(
    chunks: List[Chunk]
) -> None:
    """Multipliers apply first, then the file-level share."""
    snapshot = make_snapshot([*chunks, TEST_CHUNK])
    adjuster = PriorAdjuster(snapshot, CODE, 0.5, 0.5, 1.0)
    assert adjuster.adjust("any", ones(5)).tolist() == pytest.approx(
        [3.0, 3.0, 1.0, 1.0, 1.5]
    )


def test_prior_adjuster_on_an_empty_index() -> None:
    """An index without chunks is handled without error."""
    adjuster = PriorAdjuster(make_snapshot([]), CODE, 0.5, 0.5, 0.5)
    assert adjuster.adjust("any", ones(0)).shape == (0,)


def test_lexical_test_weight_demotes_test_files(chunks: List[Chunk]) -> None:
    """A test file full of the query terms no longer outranks the code."""
    snapshot = make_snapshot([*chunks, TEST_CHUNK])
    query = "load lora adapter"
    [plain] = LexicalRetriever(snapshot, tokenize).search_batch([query], 2)
    assert plain[0].chunk == TEST_CHUNK
    adjuster = PriorAdjuster(snapshot, test_weight=0.2)
    [hits] = LexicalRetriever(snapshot, tokenize, adjuster).search_batch(
        [query], 2
    )
    assert [h.chunk for h in hits] == [chunks[0], TEST_CHUNK]
    assert hits[1].score == pytest.approx(0.2 * plain[0].score)


@pytest.mark.parametrize("model, first", [(CODE, 0), (DOCS, 2)])
def test_lexical_routing_decides_between_code_and_docs(
    chunks: List[Chunk], model: RouterModel, first: int
) -> None:
    """The routed side wins a query matching both a code and a doc chunk."""
    snapshot = make_snapshot(chunks)
    adjuster = PriorAdjuster(snapshot, model, route_strength=0.8)
    [hits] = LexicalRetriever(snapshot, tokenize, adjuster).search_batch(
        ["adapter quantization"], 4
    )
    assert hits[0].chunk == chunks[first]
    assert {h.chunk.file_path for h in hits} == {"a.py", "b.md"}


def test_lexical_file_boost_returns_siblings_of_a_match(
    chunks: List[Chunk]
) -> None:
    """Chunks of the matching file follow it; other files stay out."""
    snapshot = make_snapshot(chunks)
    adjuster = PriorAdjuster(snapshot, file_boost=0.5)
    retriever = LexicalRetriever(snapshot, tokenize, adjuster)
    [hits] = retriever.search_batch(["KVCacheManager"], 10)
    assert [h.chunk for h in hits] == [chunks[1], chunks[0]]
    assert hits[0].score == pytest.approx(3 * hits[1].score)
    assert retriever.search_batch(["zzzunknown"], 10) == [[]]


def test_lexical_adjuster_receives_each_query(chunks: List[Chunk]) -> None:
    """The adjuster is called once per query; zeroed chunks are dropped."""
    adjuster = ScriptedAdjuster([0.0, 1.0, 1.0, 1.0])
    retriever = LexicalRetriever(make_snapshot(chunks), tokenize, adjuster)
    results = retriever.search_batch(["lora adapter", "a.py"], 4)
    assert adjuster.queries == ["lora adapter", "a.py"]
    assert results[0] == []
    assert [h.chunk for h in results[1]] == [chunks[1]]


def test_semantic_adjuster_reorders_similarities(chunks: List[Chunk]) -> None:
    """Adjusted cosines are ranked; the adjuster sees the raw query."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    adjuster = ScriptedAdjuster([1.0, 1.0, 1.0, 0.0])
    retriever = SemanticRetriever(snapshot, embedder, adjuster, "query: ")
    query = chunks[3].search_text()
    [hits] = retriever.search_batch([query], 4)
    assert hits[-1].chunk == chunks[3]
    assert hits[-1].score == 0.0
    assert adjuster.queries == [query]


def test_semantic_query_prefix_is_only_given_to_the_embedder(
    chunks: List[Chunk]
) -> None:
    """Queries are embedded with the prefix; chunks were not."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    retriever = SemanticRetriever(snapshot, embedder, query_prefix="query: ")
    retriever.search_batch(["lora", "cache"], 1)
    assert embedder.encoded[-2:] == ["query: lora", "query: cache"]
    plain = SemanticRetriever(snapshot, embedder)
    assert plain.query_prefix == ""
    plain.search_batch(["lora"], 1)
    assert embedder.encoded[-1] == "lora"


def test_semantic_query_prefix_changes_the_scores(chunks: List[Chunk]) -> None:
    """The prefix words are part of the query vector."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    query = chunks[2].search_text()
    [plain] = SemanticRetriever(snapshot, embedder).search_batch([query], 1)
    [prefixed] = SemanticRetriever(
        snapshot, embedder, query_prefix="speculative draft "
    ).search_batch([query], 1)
    assert plain[0].score == pytest.approx(1.0)
    assert prefixed[0].score < 0.99


@pytest.mark.parametrize("fusion, calls", [("linear", 1), ("rrf", 2)])
def test_hybrid_adjuster_applies_to_both_fusions(
    chunks: List[Chunk], fusion: str, calls: int
) -> None:
    """Linear adjusts the fused scores; RRF adjusts each ranking."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    query = "speculative decoding draft model adapter"

    def hybrid(adjuster: ScriptedAdjuster) -> HybridRetriever:
        """Return a hybrid retriever using ``adjuster``."""
        return HybridRetriever(
            LexicalRetriever(snapshot, tokenize),
            SemanticRetriever(snapshot, embedder),
            semantic_weight=0.5, fusion=fusion, adjuster=adjuster,
        )

    neutral = ScriptedAdjuster([1.0, 1.0, 1.0, 1.0])
    [plain] = hybrid(neutral).search_batch([query], 4)
    assert plain[0].chunk == chunks[3]
    assert neutral.queries == [query] * calls
    [hits] = hybrid(ScriptedAdjuster([1.0, 1.0, 1.0, 0.0])).search_batch(
        [query], 4
    )
    assert hits[0].chunk == chunks[0]
    assert hits[0].chunk != plain[0].chunk


@pytest.mark.parametrize("fusion", ["linear", "rrf"])
def test_hybrid_with_prior_adjuster_routes_to_code(
    chunks: List[Chunk], fusion: str
) -> None:
    """A code-routed query ranks the Python chunk above the best doc."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    hybrid = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder),
        semantic_weight=0.5, fusion=fusion,
        adjuster=PriorAdjuster(snapshot, CODE, route_strength=2.0),
    )
    [hits] = hybrid.search_batch(
        ["speculative decoding draft model adapter"], 2
    )
    assert hits[0].chunk == chunks[0]


def test_hybrid_without_adjuster_keeps_plain_fusion(
    chunks: List[Chunk]
) -> None:
    """The default hybrid retriever has no adjuster."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    hybrid = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder),
    )
    assert hybrid.adjuster is None
    neutral = HybridRetriever(
        hybrid.lexical, hybrid.semantic,
        adjuster=ScriptedAdjuster([1.0, 1.0, 1.0, 1.0]),
    )
    query = ["FP8 quantization"]
    assert neutral.search_batch(query, 4) == hybrid.search_batch(query, 4)


def test_hybrid_weight_adapts_to_identifiers(chunks: List[Chunk]) -> None:
    """Prose keeps the semantic weight; identifiers slide towards lexical."""
    embedder = FakeEmbedder()
    snapshot = make_snapshot(chunks, embedder)
    hybrid = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder),
        semantic_weight=4.0, fusion="convex",
        identifier_weight=1.0, identifier_saturation=0.25,
    )
    assert hybrid.weight_for("how does quantization work") == 4.0
    assert hybrid.weight_for("where is load_lora_adapter defined") == 1.0
    assert hybrid.weight_for(
        "tell me in detail where load_lora_adapter is"
    ) == pytest.approx(4.0 - 3.0 * (1 / 7) / 0.25)
    fixed = HybridRetriever(
        LexicalRetriever(snapshot, tokenize),
        SemanticRetriever(snapshot, embedder), semantic_weight=4.0,
    )
    assert fixed.weight_for("where is load_lora_adapter defined") == 4.0
    results = hybrid.search_batch(["load_lora_adapter", "FP8 quantization"], 1)
    assert [r[0].chunk for r in results] == [chunks[0], chunks[2]]
