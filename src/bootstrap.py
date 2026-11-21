"""Composition root: the only place that knows concrete adapters.

Swapping an implementation (another LLM, another vector store, another
corpus type) means changing a line here, not the use cases.
"""

import os
import shutil
from datetime import datetime, timezone
from functools import cached_property, partial
from pathlib import Path
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
from src.adapters.outbound.fs_corpus import FileSystemCorpus, MultiRootCorpus
from src.adapters.outbound.docling_pages import DoclingPages, docling_available
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.adapters.outbound.gpu_lease_file import FileGpuGate, lease_directory
from src.adapters.outbound.fs_document_store import FileSystemDocumentStore
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
    InMemoryIndexCopyRepository,
    index_state,
)
from src.adapters.outbound.fs_source_reader import FileSystemSourceReader
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.adapters.outbound.json_dataset_repository import (
    JsonDatasetRepository,
)
from src.adapters.outbound.knowledge_processor import KnowledgeProcessor
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
from src.adapters.outbound.thread_job_queue import ThreadJobQueue
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
from src.application.knowledge.ports import IndexState
from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.ports import Analyzer, Cache, Embedder, Reranker
from src.application.runtime_tuning import RuntimeTuner
from src.application.search import SearchService
from src.application.skills.ingest import (
    ForgetIngestedTranscript,
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.tuning import RouterTrainer, Tuner
from src.config import INDEX_FIELDS, KnowledgeSettings, Settings
from src.domain.chunking import default_registry
from src.domain.errors import InvalidInputError, short_reason
from src.domain.experiments import params_id
from src.domain.knowledge.domains import DomainManifest, check_domain_id
from src.domain.knowledge.sources import SOURCE_EXTENSIONS, knowledge_terms
from src.domain.models import IndexReport, ScoredChunk
from src.domain.priors import RouterModel
from src.domain.runtime import HardwareProfile, RuntimePlan, plan_runtime
from src.domain.text_analysis import tokenize


def utc_now() -> str:
    """Return the current UTC time as ISO 8601 text (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        knowledge: Optional[KnowledgeSettings] = None,
    ):
        """Wire services according to ``settings``.

        ``embedder`` and ``reranker`` let a caller share already loaded
        models between containers (the experiment lab does).
        ``knowledge`` configures the knowledge domains.
        """
        self.settings = settings
        self.knowledge_settings = knowledge or KnowledgeSettings()
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
    def transcript_store(self) -> FileSystemTranscriptStore:
        """``transcript.v1`` documents received from the transcriber."""
        return FileSystemTranscriptStore(self.settings.transcripts_dir)

    @cached_property
    def ingest_transcript(self) -> IngestTranscript:
        """Receive a transcript from the transcriber."""
        return IngestTranscript(self.transcript_store)

    @cached_property
    def show_transcript(self) -> ShowIngestedTranscript:
        """Give a received transcript back."""
        return ShowIngestedTranscript(self.transcript_store)

    @cached_property
    def forget_transcript(self) -> ForgetIngestedTranscript:
        """Delete a received transcript (the transcriber deleted it)."""
        return ForgetIngestedTranscript(self.transcript_store)

    def transcripts_token(self, environ: Mapping[str, str]) -> str:
        """Return the ingest token held by ``environ`` ('' if none)."""
        return environ.get(self.settings.transcripts_token_env, "").strip()

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

    def _knowledge_model(self) -> Embedder:
        """Return a new (lazily loaded) knowledge embedding model."""
        k = self.knowledge_settings
        return SentenceTransformerEmbedder(
            k.embedding_model, max_seq_length=k.embedding_max_tokens,
            plan=self.runtime,
        )

    @cached_property
    def knowledge_embedder(self) -> Embedder:
        """Embedding model of knowledge queries (loaded on first use)."""
        return self._knowledge_model()

    @cached_property
    def knowledge_index_embedder(self) -> Embedder:
        """A second copy for indexing, used by the background worker.

        A tokenizer used by two threads at once can fail ("Already
        borrowed"), and sharing one lock would make queries wait for a
        whole document's embedding; two copies avoid both.
        """
        return self._knowledge_model()

    @cached_property
    def knowledge_catalogue(self) -> FileSystemDomainCatalogue:
        """The domains' manifests."""
        k = self.knowledge_settings
        return FileSystemDomainCatalogue(k.domains_dir, k.data_dir)

    @cached_property
    def knowledge_indexes(self) -> "KnowledgeIndexFactory":
        """Per-domain indexing and search services."""
        return KnowledgeIndexFactory(
            self.knowledge_settings, self.settings, self.knowledge_embedder,
            self.knowledge_index_embedder,
        )

    @cached_property
    def knowledge_registry(self) -> DomainIndexRegistry:
        """Registry of the knowledge-domain indexes (PL-6)."""
        return DomainIndexRegistry(
            self.knowledge_catalogue, self.knowledge_indexes,
            self.knowledge_settings.max_chunk_size,
            extra_documents=lambda d: (
                self.knowledge_library.pending_documents(d)
            ),
        )

    @cached_property
    def document_extractors(self) -> DocumentExtractors:
        """Upload readers; with layout and OCR when group ``ocr`` is there."""
        k = self.knowledge_settings
        layout = None
        if k.ocr and docling_available():
            layout = DoclingPages(
                FileGpuGate(lease_directory(os.environ))
            )
        return DocumentExtractors(
            layout, k.layout_text_pages, k.layout_batch_pages
        )

    @cached_property
    def knowledge_library(self) -> KnowledgeLibrary:
        """Domains and their uploaded documents (one background worker)."""
        k = self.knowledge_settings
        return KnowledgeLibrary(
            self.knowledge_registry,
            FileSystemDocumentStore(k.data_dir),
            self.document_extractors,
            ThreadJobQueue(),
            k.max_upload_bytes,
            utc_now,
        )

    @cached_property
    def knowledge_query(self) -> KnowledgeQueryService:
        """Answer ``knowledge.query.v1`` (PL-8)."""
        return KnowledgeQueryService(self.knowledge_registry)


class KnowledgeIndexFactory:
    """Build each knowledge domain's indexing and search services.

    A domain ``d`` reads ``<domains_dir>/d`` (authored files) and
    ``<data_dir>/d/extracted`` (text of its uploads) and keeps its index
    in ``<index_dir>/d``. Retrieval uses the fusion settings of
    ``base`` without the 42-specific priors (routing, test files, file
    boost) and without the query cache.
    """

    def __init__(
        self,
        settings: KnowledgeSettings,
        base: Settings,
        embedder: Embedder,
        index_embedder: Optional[Embedder] = None,
    ) -> None:
        """Keep the settings and the embedders.

        ``embedder`` encodes queries; ``index_embedder`` (default: the
        same) encodes chunks. Both must be the same model.
        """
        self.settings = settings
        self.base = base
        self.embedder = embedder
        self.index_embedder = index_embedder or embedder

    def index_dir(self, domain_id: str) -> Path:
        """Return the folder of a domain's index."""
        return Path(self.settings.index_dir) / check_domain_id(domain_id)

    def roots(self, domain_id: str) -> List[str]:
        """Return the folders holding a domain's sources."""
        k = self.settings
        check_domain_id(domain_id)
        return [
            f"{k.domains_dir}/{domain_id}",
            f"{k.data_dir}/{domain_id}/extracted",
        ]

    def indexer(self, manifest: DomainManifest) -> IndexingService:
        """Return the indexing service of a domain."""
        k, b = self.settings, self.base
        return IndexingService(
            corpus=MultiRootCorpus(self.roots(manifest.id), SOURCE_EXTENSIONS),
            processor=KnowledgeProcessor(
                manifest, k.path_weight, k.context_weight
            ),
            lexical_builder=BM25Builder(b.bm25_k1, b.bm25_b),
            repository=FileSystemIndexRepository(
                str(self.index_dir(manifest.id))
            ),
            embedder=self.index_embedder,
            embed_style=k.embed_style,
            extra_params={
                "kind": "knowledge-domain",
                "bm25_k1": b.bm25_k1, "bm25_b": b.bm25_b,
                "path_weight": k.path_weight,
                "context_weight": k.context_weight,
                "embedding_max_tokens": k.embedding_max_tokens,
            },
        )

    def searcher(self, manifest: DomainManifest, mode: str) -> SearchService:
        """Return a search service of a domain in ``mode``."""
        b = self.base
        return SearchService(
            repository=InMemoryIndexCopyRepository(
                str(self.index_dir(manifest.id))
            ),
            analyzer=knowledge_terms,
            embedder=self.embedder,
            mode=mode,
            semantic_weight=b.semantic_weight,
            fusion=b.fusion,
            identifier_weight=b.identifier_weight,
            identifier_saturation=b.identifier_saturation,
        )

    def state(self, domain_id: str) -> Optional[IndexState]:
        """Return the state of a domain's index, if built."""
        found = index_state(self.index_dir(domain_id))
        if found is None:
            return None
        manifest, built_at = found
        return IndexState(
            manifest.fingerprint, manifest.n_chunks, manifest.n_files,
            built_at,
        )

    def sources(self, domain_id: str) -> List[str]:
        """Return a domain's indexable files."""
        return MultiRootCorpus(
            self.roots(domain_id), SOURCE_EXTENSIONS
        ).list_files()

    def drop(self, domain_id: str) -> None:
        """Delete a domain's index folder."""
        shutil.rmtree(self.index_dir(domain_id), ignore_errors=True)


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
