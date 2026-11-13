"""Composition root: the only place that knows concrete adapters.

Swapping an implementation (another LLM, another vector store, another
corpus type) means changing a line here, not the use cases.
"""

from functools import cached_property, partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.adapters.outbound.beir_repository import BeirRepository
from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.caching_embedder import CachingEmbedder
from src.adapters.outbound.coir_repository import (
    BenchmarkCatalogue,
    CoirRepository,
)
from src.adapters.outbound.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from src.adapters.outbound.fs_corpus import FileSystemCorpus
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_source_reader import FileSystemSourceReader
from src.adapters.outbound.json_dataset_repository import (
    JsonDatasetRepository,
)
from src.adapters.outbound.json_router_repository import (
    JsonRouterRepository,
)
from src.adapters.outbound.jsonl_trial_store import JsonlTrialStore
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound.parallel_processor import (
    ParallelDocumentProcessor,
)
from src.adapters.outbound.sqlite_cache import SqliteCache
from src.adapters.outbound.subprocess_benchmark import SubprocessBenchmark
from src.adapters.outbound.system_probe import SystemProbe
from src.adapters.outbound.torch_runtime import (
    load_measured,
    precision_tag,
)
from src.adapters.outbound.tqdm_progress import TqdmProgress
from src.adapters.outbound.transformers_generator import (
    TransformersGenerator,
)
from src.application.analysis import AnalysisService
from src.application.answering import AnswerService
from src.application.benchmarking import BenchmarkService
from src.application.datasets import DatasetService
from src.application.experiments import ExperimentService
from src.application.indexing import IndexingService
from src.application.ports import Analyzer, Cache, Embedder, Reranker
from src.application.runtime_tuning import RuntimeTuner
from src.application.search import SearchService
from src.application.tuning import RouterTrainer, Tuner
from src.config import INDEX_FIELDS, Settings
from src.domain.chunking import default_registry
from src.domain.errors import InvalidInputError, short_reason
from src.domain.experiments import params_id
from src.domain.models import IndexReport, ScoredChunk
from src.domain.priors import RouterModel
from src.domain.runtime import HardwareProfile, RuntimePlan, plan_runtime
from src.domain.text_analysis import tokenize


def resolve_runtime(
    device: str,
    threads: Optional[int],
    cores: str,
    plan_path: str,
    probe: Optional[SystemProbe] = None,
) -> Tuple[HardwareProfile, RuntimePlan]:
    """Probe this machine and decide device, CPUs and threads.

    A plan measured earlier on the same hardware (``system
    --benchmark``) is preferred to the rule of thumb.
    """
    profile = (probe or SystemProbe()).profile()
    plan = plan_runtime(
        profile, device, threads, cores, load_measured(plan_path, profile)
    )
    return profile, plan


class Container:
    """Build and hold the services for a given :class:`Settings`.

    Every dependency is created lazily on first access, so a command
    only pays for what it uses (e.g. ``search --mode bm25`` never loads
    PyTorch models).
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ):
        """Wire services according to ``settings``.

        ``embedder`` and ``reranker`` let a caller share already loaded
        models between containers (the experiment lab does).
        """
        self.settings = settings
        self._shared_embedder = embedder
        self._shared_reranker = reranker

    @cached_property
    def hardware(self) -> Tuple[HardwareProfile, RuntimePlan]:
        """This machine and the plan chosen for it."""
        s = self.settings
        return resolve_runtime(
            s.device, s.torch_threads, s.cpu_cores, s.runtime_plan_path
        )

    @property
    def runtime(self) -> RuntimePlan:
        """Device, CPUs and threads the models run with."""
        return self.hardware[1]

    @cached_property
    def runtime_tuner(self) -> RuntimeTuner:
        """Benchmark of the candidate runtime plans."""
        s = self.settings
        return RuntimeTuner(
            SystemProbe(), SubprocessBenchmark(s.embedding_model, s.llm)
        )

    @cached_property
    def benchmark_factory(self) -> "LabPipelineFactory":
        """Pipelines of the public benchmarks, on their own directories."""
        s = self.settings
        return LabPipelineFactory(s.model_copy(update={
            "lab_work_dir": f"{s.benchmark_dir}/work",
        }))

    @cached_property
    def benchmarks(self) -> BenchmarkService:
        """Public benchmark runner, on its own index directories."""
        s = self.settings
        return BenchmarkService(
            BenchmarkCatalogue(
                BeirRepository(f"{s.benchmark_dir}/beir"),
                CoirRepository(f"{s.benchmark_dir}/coir"),
            ),
            self.benchmark_factory,
        )

    @cached_property
    def analyzer(self) -> Analyzer:
        """Lexical analyzer shared by indexing and search."""
        if self.settings.bigrams:
            return partial(tokenize, bigrams=True)
        return tokenize

    @cached_property
    def progress(self) -> TqdmProgress:
        """Progress reporter."""
        return TqdmProgress(self.settings.progress)

    @cached_property
    def index_repository(self) -> FileSystemIndexRepository:
        """Index persistence."""
        return FileSystemIndexRepository(self.settings.index_dir)

    @cached_property
    def embedder(self) -> Embedder:
        """Dense embedding model (loaded on first encode)."""
        if self._shared_embedder is not None:
            return self._shared_embedder
        s = self.settings
        return SentenceTransformerEmbedder(
            s.embedding_model, max_seq_length=s.embedding_max_tokens,
            plan=self.runtime,
        )

    @cached_property
    def index_embedder(self) -> Embedder:
        """Embedder used for indexing, with a vector cache if enabled.

        Queries never go through the cache, so search timings stay
        representative.
        """
        s = self.settings
        if s.embedding_cache_dir is None:
            return self.embedder
        return CachingEmbedder(
            self.embedder, s.embedding_cache_dir,
            f"{s.embedding_model}|{s.embedding_max_tokens}"
            f"{precision_tag(self.runtime.device)}",
        )

    @cached_property
    def reranker(self) -> Reranker:
        """Cross-encoder reranker (loaded on first score)."""
        if self._shared_reranker is not None:
            return self._shared_reranker
        return CrossEncoderReranker(
            self.settings.rerank_model,
            max_length=self.settings.rerank_max_tokens,
            plan=self.runtime,
        )

    @cached_property
    def router(self) -> Optional[RouterModel]:
        """Query router model, when routing is enabled and trained."""
        if self.settings.route_strength <= 0:
            return None
        return JsonRouterRepository(self.settings.router_path).load()

    @cached_property
    def generator(self) -> TransformersGenerator:
        """Answer generator (loaded on first use)."""
        s = self.settings
        return TransformersGenerator(
            s.llm, s.context_tokens, s.max_new_tokens, self.runtime
        )

    @cached_property
    def cache(self) -> Cache:
        """Persistent query/answer cache."""
        return SqliteCache(self.settings.index_dir, self.settings.use_cache)

    @cached_property
    def indexing(self) -> IndexingService:
        """Indexing use case."""
        s = self.settings
        return IndexingService(
            corpus=FileSystemCorpus(s.raw_dir, s.extensions),
            processor=ParallelDocumentProcessor(
                partial(default_registry, table_context=s.table_context),
                self.analyzer, s.workers, s.path_weight, s.context_weight,
            ),
            lexical_builder=BM25Builder(s.bm25_k1, s.bm25_b),
            repository=self.index_repository,
            embedder=self.index_embedder if s.semantic else None,
            progress=self.progress,
            embed_style=s.embed_style,
            passage_prefix=s.passage_prefix,
            extra_params={
                "bm25_k1": s.bm25_k1, "bm25_b": s.bm25_b,
                "path_weight": s.path_weight,
                "context_weight": s.context_weight,
                "table_context": s.table_context,
                "bigrams": s.bigrams,
                "embedding_max_tokens": s.embedding_max_tokens,
            },
        )

    @cached_property
    def search(self) -> SearchService:
        """Search use case."""
        s = self.settings
        return SearchService(
            repository=self.index_repository,
            analyzer=self.analyzer,
            embedder=self.embedder,
            cache=self.cache,
            mode=s.mode,
            semantic_weight=s.semantic_weight,
            fusion=s.fusion,
            progress=self.progress,
            reranker=self.reranker if s.rerank else None,
            rerank_depth=s.rerank_depth,
            rerank_weight=s.rerank_weight,
            router=self.router,
            route_strength=s.route_strength,
            test_weight=s.test_weight,
            file_boost=s.file_boost,
            query_prefix=s.query_prefix,
            identifier_weight=s.identifier_weight,
            identifier_saturation=s.identifier_saturation,
        )

    @cached_property
    def answers(self) -> AnswerService:
        """Question-answering use case."""
        return AnswerService(self.search, self.generator, self.cache)

    @cached_property
    def datasets(self) -> DatasetService:
        """Dataset batch use cases."""
        return DatasetService(
            JsonDatasetRepository(),
            self.search,
            self.answers,
            FileSystemSourceReader(),
            self.progress,
        )

    @cached_property
    def trial_store(self) -> JsonlTrialStore:
        """Experiment results."""
        return JsonlTrialStore(
            f"{self.settings.lab_dir}/results/trials.jsonl"
        )

    @cached_property
    def experiments(self) -> ExperimentService:
        """Experiment runner."""
        return ExperimentService(
            LabPipelineFactory(self.settings),
            JsonDatasetRepository(),
            self.trial_store,
            self.progress,
            repeats=3,
        )

    @cached_property
    def tuner(self) -> Tuner:
        """Cross-validated parameter tuner."""
        return Tuner(self.experiments)

    @cached_property
    def analysis(self) -> AnalysisService:
        """Statistics and recommendations over the stored trials."""
        return AnalysisService(self.trial_store)

    @cached_property
    def router_trainer(self) -> RouterTrainer:
        """Query router training."""
        return RouterTrainer(
            JsonDatasetRepository(),
            JsonRouterRepository(self.settings.router_path),
        )


class LabPipeline:
    """A :class:`Container` seen as an experiment pipeline."""

    def __init__(self, container: Container):
        """Wrap ``container``."""
        self.container = container

    def index(self) -> Tuple[IndexReport, bool]:
        """Index the corpus; cold when nothing came from a cache."""
        embedder = self.container.index_embedder
        before = getattr(embedder, "hits", 0)
        s = self.container.settings
        report = self.container.indexing.run(
            s.max_chunk_size, s.target_chunk_size, force=s.lab_cold
        )
        cached = getattr(embedder, "hits", 0) - before
        return report, report.files_reused == 0 and cached == 0

    def search(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Search without the query cache."""
        return self.container.search.search_many(queries, k)


class LabPipelineFactory:
    """Build one pipeline per parameter set for the experiment lab.

    Each distinct set of index parameters gets its own index directory.
    Loaded models and an embedding cache are shared between pipelines,
    so a trial that only changes search parameters costs a few seconds.
    The parameter ``lab_cold`` disables every reuse for that trial, to
    measure the real indexing time.
    """

    def __init__(self, base: Settings):
        """Derive pipelines from the ``base`` settings."""
        self.base = base
        self._embedders: Dict[str, Embedder] = {}
        self._rerankers: Dict[str, Reranker] = {}
        self._containers: Dict[str, Container] = {}
        self._plan: Optional[RuntimePlan] = None

    def create(self, params: Mapping[str, Any]) -> LabPipeline:
        """Return the pipeline configured by ``params``."""
        unknown = sorted(set(params) - set(Settings.model_fields))
        if unknown:
            raise InvalidInputError(
                f"Unknown parameter(s): {', '.join(unknown)}."
            )
        values = {**self.base.model_dump(), **params}
        index_id = params_id({
            f: values[f] for f in INDEX_FIELDS
            # A field added later with an empty value keeps old ids valid.
            if not (f == "passage_prefix" and not values[f])
        })
        work = self.base.lab_work_dir
        cold = bool(values.get("lab_cold"))
        values.update(
            index_dir=f"{work}/indexes/{index_id}{'-cold' if cold else ''}",
            embedding_cache_dir=None if cold else f"{work}/embeddings",
            use_cache=False,
            progress=False,
        )
        try:
            settings = Settings.model_validate(values)
        except ValueError as exc:
            raise InvalidInputError(
                f"Invalid parameters: {short_reason(exc)}"
            )
        key = params_id(settings.model_dump())
        if key not in self._containers:
            self._containers[key] = Container(
                settings, self._embedder(settings), self._reranker(settings)
            )
        return LabPipeline(self._containers[key])

    def forget(self) -> None:
        """Drop the cached pipelines (their index files may be deleted).

        Loaded models stay shared: only containers, which hold index
        snapshots, are released.
        """
        self._containers.clear()

    def _embedder(self, s: Settings) -> Embedder:
        """Return the shared embedder of a model configuration."""
        key = f"{s.embedding_model}|{s.embedding_max_tokens}"
        if key not in self._embedders:
            self._embedders[key] = SentenceTransformerEmbedder(
                s.embedding_model, max_seq_length=s.embedding_max_tokens,
                plan=self._runtime(),
            )
        return self._embedders[key]

    def _runtime(self) -> RuntimePlan:
        """Return the runtime plan shared by every pipeline."""
        if self._plan is None:
            b = self.base
            self._plan = resolve_runtime(
                b.device, b.torch_threads, b.cpu_cores, b.runtime_plan_path
            )[1]
        return self._plan

    def _reranker(self, s: Settings) -> Reranker:
        """Return the shared reranker of a model configuration."""
        key = f"{s.rerank_model}|{s.rerank_max_tokens}"
        if key not in self._rerankers:
            self._rerankers[key] = CrossEncoderReranker(
                s.rerank_model, max_length=s.rerank_max_tokens,
                plan=self._runtime(),
            )
        return self._rerankers[key]
