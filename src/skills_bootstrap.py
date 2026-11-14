"""Composition root of the soft-skills analyzer.

It reuses the retrieval building blocks of the RAG (BM25, embedder,
cache, experiment lab) behind the same ports, and adds the adapters
that are specific to transcripts and rubrics.
"""

from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.adapters.outbound.anthropic_chat_model import AnthropicChatModel
from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.fs_transcript_repository import (
    FileSystemTranscriptRepository,
)
from src.adapters.outbound.json_catalogue_repository import (
    JsonCatalogueRepository,
)
from src.adapters.outbound.json_dataset_repository import (
    JsonDatasetRepository,
)
from src.adapters.outbound.jsonl_trial_store import JsonlTrialStore
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound.sqlite_cache import SqliteCache
from src.adapters.outbound.tqdm_progress import TqdmProgress
from src.adapters.outbound.transformers_chat_model import (
    TransformersChatModel,
)
from src.application.analysis import AnalysisService
from src.application.experiments import ExperimentService
from src.application.ports import Cache, Embedder
from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.lab import SkillsLabPipeline, load_rubric
from src.application.skills.ports import ChatModel
from src.application.skills.tagging import SentenceTagger, TagConfirmer
from src.application.tuning import Tuner
from src.bootstrap import resolve_runtime
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError, short_reason
from src.domain.experiments import params_id
from src.domain.runtime import (
    HardwareProfile,
    ModelChoice,
    ModelOption,
    RuntimePlan,
    choose_model,
)
from src.domain.skills.rubric import RubricCatalogue


class SkillsContainer:
    """Build and hold the soft-skills services for given settings.

    Dependencies are created lazily, so listing a catalogue or
    measuring annotator agreement never loads a model.
    """

    def __init__(
        self,
        settings: SkillsSettings,
        embedder: Optional[Embedder] = None,
        judge: Optional[ChatModel] = None,
    ):
        """Wire services; ``embedder`` and ``judge`` can be shared."""
        self.settings = settings
        self._shared_embedder = embedder
        self._shared_judge = judge

    @cached_property
    def progress(self) -> TqdmProgress:
        """Progress reporter."""
        return TqdmProgress(self.settings.progress)

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
    def model_choice(self) -> ModelChoice:
        """The local judge model picked for this machine."""
        s = self.settings
        profile, plan = self.hardware
        return choose_model(
            profile, plan,
            [ModelOption(name=n, params_billion=b) for n, b in s.model_ladder],
            s.min_tokens_per_second,
        )

    def local_model(self) -> str:
        """Return the local judge model: the given one, or the choice."""
        s = self.settings
        return self.model_choice.model if s.llm == "auto" else s.llm

    @cached_property
    def transcripts(self) -> FileSystemTranscriptRepository:
        """Transcript files."""
        return FileSystemTranscriptRepository(self.settings.transcripts_dir)

    @cached_property
    def catalogues(self) -> JsonCatalogueRepository:
        """Rubric catalogues."""
        return JsonCatalogueRepository(self.settings.rubric_dir)

    def rubric(self, language: str) -> RubricCatalogue:
        """Return the catalogue for a transcript in ``language``."""
        s = self.settings
        wanted = language if s.rubric_language == "auto" else s.rubric_language
        return load_rubric(self.catalogues, s.catalogue, wanted)

    @cached_property
    def embedder(self) -> Embedder:
        """Multilingual embedding model (loaded on first encode)."""
        if self._shared_embedder is not None:
            return self._shared_embedder
        s = self.settings
        return SentenceTransformerEmbedder(
            s.embedding_model, max_seq_length=s.embedding_max_tokens,
            plan=self.runtime,
        )

    @cached_property
    def retriever(self) -> EvidenceRetriever:
        """Evidence retrieval."""
        s = self.settings
        return EvidenceRetriever(
            BM25Builder(s.bm25_k1, s.bm25_b),
            self.embedder if s.semantic else None,
            semantic_weight=s.semantic_weight,
            min_words=s.min_words,
            query_prefix=s.query_prefix,
            passage_prefix=s.passage_prefix,
        )

    @cached_property
    def tagger(self) -> SentenceTagger:
        """Candidate rubric indicators per sentence."""
        s = self.settings
        return SentenceTagger(
            self.retriever, top=s.tags_per_sentence,
            min_similarity=s.min_similarity, per_competency=s.candidates,
        )

    @cached_property
    def confirmer(self) -> TagConfirmer:
        """Classifier that confirms or rejects candidate tags."""
        return TagConfirmer(self.judge, self.cache, self.progress)

    @cached_property
    def judge(self) -> ChatModel:
        """Language model that quotes and rates."""
        if self._shared_judge is not None:
            return self._shared_judge
        s = self.settings
        if s.judge == "api":
            return AnthropicChatModel(s.api_model, s.api_effort)
        return TransformersChatModel(
            self.local_model(), plan=self.runtime
        )

    @cached_property
    def cache(self) -> Cache:
        """Persistent cache of the judge's replies."""
        s = self.settings
        if s.use_cache:
            try:
                Path(s.cache_dir).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        return SqliteCache(s.cache_dir, s.use_cache)

    @cached_property
    def assessment(self) -> AssessmentService:
        """Scorecard use case."""
        s = self.settings
        return AssessmentService(
            self.retriever, self.judge, self.cache, self.progress,
            candidates=s.candidates, min_incidents=s.min_incidents,
            max_tokens=s.max_new_tokens, min_similarity=s.min_similarity,
            min_episodes=s.min_episodes, episode_gap=s.episode_gap,
            min_words=s.min_participant_words,
        )

    @cached_property
    def trial_store(self) -> JsonlTrialStore:
        """Experiment results of the skills lab."""
        return JsonlTrialStore(
            f"{self.settings.lab_dir}/results/trials.jsonl"
        )

    @cached_property
    def experiments(self) -> ExperimentService:
        """Experiment runner over evidence retrieval."""
        return ExperimentService(
            SkillsLabFactory(self.settings, self._shared_embedder),
            JsonDatasetRepository(),
            self.trial_store,
            self.progress,
            repeats=1,
        )

    @cached_property
    def tuner(self) -> Tuner:
        """Cross-validated parameter tuner."""
        return Tuner(self.experiments)

    @cached_property
    def analysis(self) -> AnalysisService:
        """Statistics and recommendations over the stored trials."""
        return AnalysisService(self.trial_store)


class SkillsLabFactory:
    """Build one evidence-retrieval pipeline per parameter set.

    Embedding models are shared between pipelines, so a trial that
    only changes a weight costs no model load.
    """

    def __init__(
        self, base: SkillsSettings, embedder: Optional[Embedder] = None
    ):
        """Derive pipelines from ``base``; ``embedder`` serves its model."""
        self.base = base
        self._embedders: Dict[str, Embedder] = {}
        if embedder is not None:
            key = f"{base.embedding_model}|{base.embedding_max_tokens}"
            self._embedders[key] = embedder
        self._pipelines: Dict[str, SkillsLabPipeline] = {}

    def create(self, params: Mapping[str, Any]) -> SkillsLabPipeline:
        """Return the pipeline configured by ``params``."""
        unknown = sorted(set(params) - set(SkillsSettings.model_fields))
        if unknown:
            raise InvalidInputError(
                f"Unknown parameter(s): {', '.join(unknown)}."
            )
        values = {**self.base.model_dump(), **params, "progress": False}
        try:
            settings = SkillsSettings.model_validate(values)
        except ValueError as exc:
            raise InvalidInputError(
                f"Invalid parameters: {short_reason(exc)}"
            )
        key = params_id(settings.model_dump())
        if key not in self._pipelines:
            model = (
                f"{settings.embedding_model}|{settings.embedding_max_tokens}"
            )
            if model not in self._embedders:
                self._embedders[model] = SentenceTransformerEmbedder(
                    settings.embedding_model,
                    max_seq_length=settings.embedding_max_tokens,
                    plan=resolve_runtime(
                        settings.device, settings.torch_threads,
                        settings.cpu_cores, settings.runtime_plan_path,
                    )[1],
                )
            container = SkillsContainer(settings, self._embedders[model])
            self._pipelines[key] = SkillsLabPipeline(
                container.retriever, container.transcripts,
                container.catalogues, settings.catalogue,
                settings.rubric_language,
            )
        return self._pipelines[key]
