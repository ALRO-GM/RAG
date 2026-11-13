"""Composition root wired with real adapters, lexical only."""

import json
from pathlib import Path
from typing import Mapping

import pytest

from src.adapters.outbound.caching_embedder import CachingEmbedder
from src.adapters.outbound.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from src.adapters.outbound.json_router_repository import (
    JsonRouterRepository,
)
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound.parallel_processor import (
    ParallelDocumentProcessor,
)
from src.adapters.outbound.sqlite_cache import SqliteCache
from src.application.retrieval import (
    LexicalRetriever,
    PriorAdjuster,
    RerankingRetriever,
)
from src.bootstrap import Container, LabPipelineFactory
from src.config import Settings
from src.domain.chunking import TextChunker
from src.domain.errors import CorruptedDataError, InvalidInputError
from src.domain.experiments import Suite
from src.domain.priors import FEATURE_NAMES, RouterModel
from src.domain.prompting import NO_ANSWER
from tests.samples import MD_PATH, PY_PATH, TXT_PATH, write_json


def settings(**overrides: object) -> Settings:
    """Return lexical-only settings for the tiny corpus."""
    values: dict[str, object] = {
        "raw_dir": "data/raw", "index_dir": "idx", "semantic": False,
        "mode": "bm25", "progress": False, "workers": 1,
        "max_chunk_size": 300,
        # Neutral retrieval settings: these tests must not depend on
        # the tuned defaults of src/config.py.
        "route_strength": 0.0, "test_weight": 1.0, "file_boost": 0.0,
        "path_weight": 1, "context_weight": 1, "table_context": False,
        "bm25_b": 0.75,
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_container_index_search_and_datasets(project: Path) -> None:
    """Index, search and run a dataset through the wired services."""
    app = Container(settings())
    report = app.indexing.run(app.settings.max_chunk_size)
    assert report.files_total == 3
    assert app.indexing.embedder is None
    hits = app.search.search("load_lora_adapter", 2)
    assert hits[0].chunk.file_path == PY_PATH
    assert isinstance(app.search.retriever, LexicalRetriever)
    assert isinstance(app.cache, SqliteCache)
    assert (project / "idx" / "query_cache.sqlite").is_file()
    dataset = write_json(project / "q.json", {"rag_questions": [
        {"question_id": "a", "question": "FP8 quantization",
         "answer": "x", "sources": [{
             "file_path": MD_PATH, "first_character_index": 0,
             "last_character_index": 60}]},
    ]})
    run = app.datasets.search_dataset(dataset, 3, "out")
    assert run.mode == "bm25"
    saved = json.loads(Path(run.output_path).read_text())
    assert saved["search_results"][0]["retrieved_sources"][0][
        "file_path"] == MD_PATH
    evaluation, warnings = app.datasets.evaluate(run.output_path, dataset, 3)
    assert warnings == []
    assert evaluation.n_questions == 1


def test_container_is_lazy(project: Path) -> None:
    """Services are cached; models are not loaded by lexical commands."""
    app = Container(settings())
    assert app.search is app.search
    assert isinstance(app.embedder, SentenceTransformerEmbedder)
    assert app.embedder._model is None
    assert app.generator._model is None
    assert app.answers.answer_from("question?", []) == NO_ANSWER
    assert app.generator._model is None


def test_container_settings_propagate(project: Path) -> None:
    """Settings reach the adapters they configure."""
    app = Container(settings(embedding_model="m", embedding_max_tokens=64,
                             llm="tiny", context_tokens=512,
                             max_new_tokens=8, use_cache=False))
    assert isinstance(app.embedder, SentenceTransformerEmbedder)
    assert app.embedder.model_name == "m"
    assert app.embedder.max_seq_length == 64
    assert app.generator.model_name == "tiny"
    assert app.generator.context_tokens == 512
    assert app.generator.max_new_tokens == 8
    assert isinstance(app.cache, SqliteCache)
    assert app.cache.enabled is False
    assert not (project / "idx").exists()


def test_container_rerank_wiring(project: Path) -> None:
    """--rerank wraps the retriever; the model loads only when scoring."""
    app = Container(settings(rerank=True, rerank_model="tiny-ce",
                             rerank_depth=4, rerank_weight=0.5,
                             rerank_max_tokens=64))
    app.indexing.run(300)
    retriever = app.search.retriever
    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever.base, LexicalRetriever)
    assert (retriever.depth, retriever.weight) == (4, 0.5)
    assert isinstance(app.reranker, CrossEncoderReranker)
    assert app.reranker.model_name == "tiny-ce"
    assert app.reranker.max_length == 64
    assert app.reranker.score([]) == []
    assert app.reranker._model is None


def test_container_rerank_off_by_default(project: Path) -> None:
    """Without --rerank no reranker is wired in."""
    app = Container(settings())
    assert app.search.reranker is None


def test_container_invalid_mode(project: Path) -> None:
    """An unknown mode is reported when the search service is built."""
    with pytest.raises(InvalidInputError):
        Container(settings(mode="nope")).search


def test_settings_validation() -> None:
    """Out-of-range settings are rejected by pydantic."""
    with pytest.raises(ValueError):
        Settings(context_tokens=1)
    with pytest.raises(ValueError):
        Settings(rerank_weight=1.5)
    with pytest.raises(ValueError):
        Settings(rerank_depth=0)
    with pytest.raises(ValueError):
        Settings(embedding_max_tokens=2)


# ------------------------------------------------- priors and routing

ROUTER = RouterModel(weights=[0.0] * len(FEATURE_NAMES), bias=20.0)


def test_route_strength_without_router_file_is_no_routing(
    project: Path
) -> None:
    """Routing is silently off until a router has been trained."""
    app = Container(settings(route_strength=0.5))
    app.indexing.run(300)
    assert app.router is None
    assert app.search.route_strength == 0.0
    retriever = app.search.retriever
    assert isinstance(retriever, LexicalRetriever)
    assert retriever.adjuster is None
    assert app.search.search("load_lora_adapter", 2)


def test_route_strength_with_saved_router_activates_priors(
    project: Path
) -> None:
    """A saved router is loaded and drives an active PriorAdjuster."""
    JsonRouterRepository("models/router.json").save(ROUTER)
    app = Container(settings(route_strength=0.5,
                             router_path="models/router.json"))
    app.indexing.run(300)
    assert app.router == ROUTER
    retriever = app.search.retriever
    assert isinstance(retriever, LexicalRetriever)
    adjuster = retriever.adjuster
    assert isinstance(adjuster, PriorAdjuster)
    assert adjuster.active is True
    assert adjuster.route_strength == 0.5
    hits = app.search.search("lora adapter endpoint", 5)
    assert hits[0].chunk.file_path == PY_PATH


def test_router_file_is_not_read_when_routing_is_off(project: Path) -> None:
    """With route_strength 0 even a broken router file is harmless."""
    write_json(project / "models/router.json", {"weights": [1.0]})
    off = Container(settings(router_path="models/router.json"))
    assert off.router is None
    on = Container(settings(route_strength=0.3,
                            router_path="models/router.json"))
    with pytest.raises(CorruptedDataError, match="train_router"):
        on.router


def test_prior_settings_propagate_to_search(project: Path) -> None:
    """test_weight, file_boost and query_prefix reach the search service."""
    app = Container(settings(test_weight=0.4, file_boost=0.25,
                             query_prefix="query: "))
    app.indexing.run(300)
    search = app.search
    assert (search.test_weight, search.file_boost, search.query_prefix) == (
        0.4, 0.25, "query: ",
    )
    retriever = search.retriever
    assert isinstance(retriever, LexicalRetriever)
    assert isinstance(retriever.adjuster, PriorAdjuster)
    assert retriever.adjuster.file_boost == 0.25
    plain = Container(settings()).search.search("KVCacheManager", 10)
    boosted = search.search("KVCacheManager", 10)
    assert len(boosted) > len(plain)
    assert {h.chunk.file_path for h in boosted} == {PY_PATH}


def test_index_settings_propagate_to_indexing(project: Path) -> None:
    """Field weights, BM25 constants and table context reach indexing."""
    app = Container(settings(
        path_weight=3, context_weight=2, table_context=True, bm25_k1=0.9,
        bm25_b=0.5, embed_style="body", embedding_max_tokens=64,
    ))
    indexing = app.indexing
    assert indexing.embed_style == "body"
    assert indexing.extra_params == {
        "bm25_k1": 0.9, "bm25_b": 0.5, "path_weight": 3,
        "context_weight": 2, "table_context": True,
        "embedding_max_tokens": 64,
        "bigrams": False,
    }
    processor = indexing.processor
    assert isinstance(processor, ParallelDocumentProcessor)
    assert (processor.path_weight, processor.context_weight) == (3, 2)
    chunker = processor.registry_factory().for_path("a.md", 100, None)
    assert isinstance(chunker, TextChunker)
    assert chunker.table_context is True


@pytest.mark.parametrize("changed", [
    {"path_weight": 2}, {"context_weight": 0}, {"table_context": True},
    {"bm25_k1": 0.9}, {"bm25_b": 0.3},
])
def test_changed_index_setting_rebuilds_the_index(
    project: Path, changed: dict[str, object]
) -> None:
    """An index built with other index settings is not reused."""
    first = Container(settings())
    first.indexing.run(300)
    before = first.index_repository.load().manifest.fingerprint
    assert Container(settings()).indexing.run(300).files_reused == 3
    other = Container(settings(**changed))
    report = other.indexing.run(300)
    assert (report.files_reindexed, report.incremental) == (3, False)
    assert other.index_repository.load().manifest.fingerprint != before


def test_index_embedder_is_cached_only_with_a_cache_dir(
    project: Path
) -> None:
    """The caching wrapper is used for indexing when a directory is set."""
    plain = Container(settings())
    assert plain.index_embedder is plain.embedder
    cached = Container(settings(embedding_cache_dir="cache/emb",
                                embedding_model="m"))
    wrapper = cached.index_embedder
    assert isinstance(wrapper, CachingEmbedder)
    assert wrapper.inner is cached.embedder
    assert wrapper.model_name == "m"
    assert wrapper.directory.parent == Path("cache/emb")
    assert cached.search.embedder is cached.embedder, "queries are not cached"
    assert cached.indexing.embedder is None, "lexical-only index"
    assert not (project / "cache").exists()


def test_semantic_indexing_uses_the_index_embedder(project: Path) -> None:
    """With semantic=True the indexing service gets the cached embedder."""
    app = Container(settings(semantic=True, embedding_cache_dir="cache/emb"))
    assert app.indexing.embedder is app.index_embedder
    assert isinstance(app.indexing.embedder, CachingEmbedder)
    assert isinstance(app.embedder, SentenceTransformerEmbedder)
    assert app.embedder._model is None


@pytest.mark.parametrize("overrides", [
    {"route_strength": -0.1}, {"test_weight": -1.0}, {"file_boost": -0.5},
    {"path_weight": -1}, {"context_weight": -1},
])
def test_new_settings_validation(overrides: dict[str, object]) -> None:
    """Negative priors and field weights are rejected by pydantic."""
    with pytest.raises(ValueError):
        settings(**overrides)


# ------------------------------------------------------------- the lab


def lab_settings() -> Settings:
    """Return lexical settings with the lab under tmp directories."""
    return settings(lab_dir="lab", lab_work_dir="work")


def test_lab_factory_rejects_unknown_parameters(project: Path) -> None:
    """A parameter that is not a setting is named in the error."""
    factory = LabPipelineFactory(lab_settings())
    with pytest.raises(InvalidInputError,
                       match=r"Unknown parameter\(s\): nope, zzz"):
        factory.create({"zzz": 1, "file_boost": 0.1, "nope": 2})


@pytest.mark.parametrize("params", [
    {"rerank_depth": 0}, {"file_boost": -1.0}, {"max_chunk_size": "big"},
    {"rerank_weight": 2.0},
])
def test_lab_factory_rejects_invalid_values(
    project: Path, params: dict[str, object]
) -> None:
    """A value refused by the settings is a user error."""
    with pytest.raises(InvalidInputError, match="Invalid parameters"):
        LabPipelineFactory(lab_settings()).create(params)


def test_lab_factory_index_dir_follows_index_parameters(
    project: Path
) -> None:
    """Search-only changes share an index; index changes get their own."""
    factory = LabPipelineFactory(lab_settings())

    def index_dir(params: Mapping[str, object]) -> str:
        """Return the index directory of the pipeline for ``params``."""
        return factory.create(params).container.settings.index_dir

    base = index_dir({})
    assert base.startswith("work/indexes/")
    for search_only in ({"file_boost": 0.3}, {"test_weight": 0.5},
                        {"route_strength": 0.2}, {"mode": "bm25"},
                        {"query_prefix": "q: "}, {"rerank": True}):
        assert index_dir(search_only) == base
    index_dirs = {base} | {
        index_dir(p) for p in (
            {"path_weight": 2}, {"context_weight": 2}, {"bm25_k1": 0.9},
            {"table_context": True}, {"max_chunk_size": 500},
            {"target_chunk_size": 200}, {"embed_style": "body"},
        )
    }
    assert len(index_dirs) == 8
    assert not (project / "work").exists(), "nothing is built yet"


def test_lab_factory_pipelines_are_isolated_from_user_state(
    project: Path
) -> None:
    """Lab pipelines use no query cache and share the embedding cache."""
    factory = LabPipelineFactory(lab_settings())
    pipeline = factory.create({"file_boost": 0.3})
    s = pipeline.container.settings
    assert (s.use_cache, s.progress) == (False, False)
    assert s.embedding_cache_dir == "work/embeddings"
    assert (s.file_boost, s.semantic, s.mode) == (0.3, False, "bm25")
    assert factory.create({"file_boost": 0.3}).container is (
        pipeline.container
    )
    assert factory.create({"file_boost": 0.4}).container is not (
        pipeline.container
    )


def test_lab_pipeline_indexes_once_and_searches(project: Path) -> None:
    """The first build is cold; the next one reuses the index."""
    factory = LabPipelineFactory(lab_settings())
    pipeline = factory.create({})
    report, cold = pipeline.index()
    assert (report.files_total, cold) == (3, True)
    index_dir = Path(pipeline.container.settings.index_dir)
    assert (index_dir / "manifest.json").is_file()
    assert not (index_dir / "query_cache.sqlite").exists()
    again, cold_again = factory.create({"file_boost": 0.2}).index()
    assert (again.files_reused, cold_again) == (3, False)
    results = pipeline.search(["load_lora_adapter", "FP8 quantization"], 3)
    assert [r[0].chunk.file_path for r in results] == [PY_PATH, MD_PATH]


def lab_dataset(project: Path) -> str:
    """Write a dataset with one question per corpus file."""
    def question(text: str, path: str, last: int) -> dict[str, object]:
        """Return an answered question as JSON-ready data."""
        return {"question_id": text, "question": text, "answer": "x",
                "sources": [{"file_path": path, "first_character_index": 0,
                             "last_character_index": last}]}

    return write_json(project / "q.json", {"rag_questions": [
        question("How to load a LoRA adapter?", PY_PATH, 200),
        question("What does speculative decoding use?", TXT_PATH, 58),
        question("FP8 quantization memory", MD_PATH, 60),
        {"question_id": "open", "question": "unanswered"},
    ]})


def test_lab_end_to_end_lexical_suite(project: Path) -> None:
    """A lexical suite runs, is stored, resumed and analysed."""
    app = Container(lab_settings())
    suite = Suite(name="tiny", base={"semantic": False, "mode": "bm25"},
                  factors={"file_boost": [0.5], "path_weight": [2]})
    paths = {"tiny": lab_dataset(project)}
    results = app.experiments.run_suite(suite, paths)
    assert [r.name for r in results] == [
        "baseline", "file_boost=0.5", "path_weight=2",
    ]
    assert [r.index_cold for r in results] == [True, False, True]
    for result in results:
        assert result.baseline_id == results[0].trial_id
        assert result.metrics["tiny"]["recall@5"] == 1.0
        assert len(result.ranks["tiny"]) == 3
        assert result.n_chunks > 0
    assert len(list((project / "work/indexes").iterdir())) == 2
    lines = (project / "lab/results/trials.jsonl").read_text().splitlines()
    assert len(lines) == 3
    fresh = Container(lab_settings())
    assert fresh.experiments.run_suite(suite, paths) == results
    assert fresh.trial_store.all() == results
    report = fresh.analysis.analyze()
    assert report.n_trials == 3
    assert {r.verdict for r in report.rows} == {"baseline", "neutral"}
    assert {p.name for p in report.profiles} == {
        "precision", "balanced", "speed",
    }
    assert not (project / "experiments").exists()


def test_lab_services_share_the_experiment_runner(project: Path) -> None:
    """Tuner, analysis and experiments use the same store."""
    app = Container(lab_settings())
    assert app.tuner.experiments is app.experiments
    assert app.experiments.store is app.trial_store
    assert app.analysis.store is app.trial_store
    assert app.trial_store.path == Path("lab/results/trials.jsonl")
    assert app.experiments.repeats == 3


def test_router_trainer_saves_where_the_container_loads(
    project: Path
) -> None:
    """A router trained through the container is used by a later search."""
    def question(text: str, path: str) -> dict[str, object]:
        """Return an answered question as JSON-ready data."""
        return {"question_id": text, "question": text, "answer": "x",
                "sources": [{"file_path": path, "first_character_index": 0,
                             "last_character_index": 10}]}

    dataset = write_json(project / "router.json", {"rag_questions": [
        question(f"What is the default value of option_{i} in Cls{i}A?",
                 PY_PATH) for i in range(6)
    ] + [
        question(f"How do you install part {i} with Docker?", MD_PATH)
        for i in range(6)
    ]})
    app = Container(settings(router_path="models/r.json"))
    report = app.router_trainer.train([dataset], folds=3)
    assert report.path == "models/r.json"
    assert report.n_questions == 12
    assert report.train_accuracy == 1.0
    routed = Container(settings(router_path="models/r.json",
                                route_strength=0.5))
    assert routed.router is not None
    assert routed.router.p_code("What is the default of foo_bar?") > 0.5


def test_lab_cold_disables_every_reuse(project: Path) -> None:
    """A cold trial rebuilds its own index without the vector cache."""
    factory = LabPipelineFactory(lab_settings())
    warm = factory.create({})
    cold = factory.create({"lab_cold": True})
    warm_settings = warm.container.settings
    cold_settings = cold.container.settings
    assert cold_settings.index_dir == warm_settings.index_dir + "-cold"
    assert cold_settings.embedding_cache_dir is None
    assert warm_settings.embedding_cache_dir is not None
    first, was_cold = cold.index()
    again, still_cold = cold.index()
    assert was_cold and still_cold
    assert again.files_reindexed == first.files_total
    assert warm.index()[1] is True
    assert warm.index()[1] is False
