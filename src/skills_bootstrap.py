"""Composition root of the soft-skills analyzer.

It reuses the retrieval building blocks of the RAG (BM25, embedder,
cache, experiment lab) behind the same ports, and adds the adapters
that are specific to transcripts and rubrics.
"""

from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.adapters.outbound.anthropic_chat_model import AnthropicChatModel
from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.adapters.outbound.fs_assessment_store import (
    FileSystemAssessmentStore,
)
from src.adapters.outbound.fs_transcript_repository import (
    FileSystemTranscriptRepository,
)
from src.adapters.outbound.json_catalogue_repository import (
    JsonCatalogueRepository,
)
from src.adapters.outbound.json_dataset_repository import (
    JsonDatasetRepository,
    read_model,
)
from src.adapters.outbound.json_judge_state import JsonJudgeStateStore
from src.adapters.outbound.jsonl_trial_store import JsonlTrialStore
from src.adapters.outbound.local_judge import LocalJudge
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound.sqlite_cache import SqliteCache
from src.adapters.outbound.subprocess_analysis import (
    SubprocessAnalysisLauncher,
)
from src.adapters.outbound.tqdm_progress import TqdmProgress
from src.application.analysis import AnalysisService
from src.application.experiments import ExperimentService
from src.application.ports import Cache, Embedder
from src.application.skills.analysis_steps import (
    RuleEventExtractor,
    TaggerActStep,
)
from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.act_tagger import (
    CombinedTagger,
    RuleTagger,
    TaggerCard,
)
from src.application.skills.judges import (
    ActClassifier,
    FallbackJudge,
    JudgeSelection,
)
from src.application.skills.lab import SkillsLabPipeline, load_rubric
from src.application.skills.ports import (
    ActTagger,
    ChatModel,
    DialogueActTagger,
    EventExtractor,
    TranscriptStore,
)
from src.application.skills.profile import ProfileService
from src.application.skills.session_analysis import (
    AnalysisRequests,
    AnalysisResults,
    JudgeSetup,
    SessionAnalysis,
)
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
from src.domain.skills.judging import JudgeMode
from src.domain.skills.rubric import RubricCatalogue


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ACT_TAGGERS = ("rules", "encoder", "combined")
"""Dialogue-act taggers ``skills acts --tagger`` can build."""
"""The checkout this code runs from (relative data paths start here)."""


def judge_state_file(path: str, root: Path = REPOSITORY_ROOT) -> Path:
    """Return the judge state file: ``path`` itself, or under ``root``.

    The RAG server and the analysis commands may start in different
    working directories; both must read and write the same file.
    """
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


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

    def profiles(self, directory: str = "") -> ProfileService:
        """Profiles across the assessments kept in ``directory``.

        The default directory is ``output_dir``, where ``assess``
        writes them.
        """
        s = self.settings
        return ProfileService(
            FileSystemAssessmentStore(directory or s.output_dir),
            s.profile_min_situations,
        )

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
    def second_confirmer(self) -> TagConfirmer:
        """The local judge's confirmer, compared with the first (both)."""
        return TagConfirmer(self.local_judge, self.cache, self.progress)

    def act_classifier(self, judge: ChatModel) -> ActClassifier:
        """Return a dialogue-act classifier over ``judge``."""
        return ActClassifier(judge, self.cache, self.progress)

    def act_card(self) -> TaggerCard:
        """Return the dialogue-act tagger's measured settings."""
        return read_model(self.settings.act_card, TaggerCard)

    def act_tagger(self, kind: str = "default") -> Tuple[ActTagger, str]:
        """Return the dialogue-act tagger asked for and a note.

        ``kind`` is ``default`` (the card's choice), ``rules``,
        ``encoder`` or ``combined``. Without the fine-tuned weights
        (git-ignored, trained on the workstation) the rules answer and
        the note says so.
        """
        card = self.act_card()
        kind = card.name.split("+")[0] if kind == "default" else kind
        if kind not in ACT_TAGGERS:
            raise InvalidInputError(
                f"Unknown tagger {kind!r}; choose one of "
                f"default, {', '.join(ACT_TAGGERS)}.")
        rules = RuleTagger(card.rules)
        if kind == "rules":
            return rules, ""
        from src.adapters.outbound.encoder_act_tagger import (
            CARD,
            EncoderActTagger,
        )

        if not (Path(card.encoder) / CARD).is_file():
            return rules, (
                f"No fine-tuned weights in {card.encoder or '(none)'}: "
                f"the rules answered instead of the {kind} tagger.")
        device = "cuda" if self.runtime.device == "cuda" else "cpu"
        encoder = EncoderActTagger(card.encoder, device,
                                   temperature=card.temperature)
        if kind == "encoder":
            return encoder, ""
        return CombinedTagger({"rules": rules, "encoder": encoder},
                              card.routes, card.fallback), ""

    @cached_property
    def judge_selection(self) -> JudgeSelection:
        """The judge chosen from the web and Claude's quota record."""
        return JudgeSelection(
            JsonJudgeStateStore(str(judge_state_file(
                self.settings.judge_state_path
            ))),
            lambda: datetime.now(timezone.utc),
        )

    def judge_mode(self) -> JudgeMode:
        """Return the judge of this job: the setting, or the web's choice.

        ``api`` is reported as ``claude_code`` here: it is a single
        hosted judge without the local one behind it.
        """
        s = self.settings
        if s.judge == "selected":
            return self.judge_selection.mode()
        if s.judge == "api":
            return "claude_code"
        return s.judge

    @cached_property
    def claude_judge(self) -> ClaudeCodeJudge:
        """Claude through the ``claude`` command line (subscription)."""
        s = self.settings
        return ClaudeCodeJudge(
            s.claude_model, s.claude_executable, s.claude_timeout,
            s.claude_effort,
        )

    @cached_property
    def local_judge(self) -> ChatModel:
        """The local model, picked by the memory free when it is needed."""
        if self._shared_judge is not None:
            return self._shared_judge
        s = self.settings
        return LocalJudge(
            [
                ModelOption(name=n, params_billion=b, bits=s.local_bits)
                for n, b in s.model_ladder
            ],
            self.runtime,
            cpu_model=self.local_model(),
            fixed=None if s.llm == "auto" else s.llm,
            policy=s.local_policy,
            preference=s.local_preference,
            wait_seconds=s.local_wait_seconds,
        )

    @cached_property
    def judge(self) -> ChatModel:
        """Language model that quotes, rates and confirms tags.

        Claude Code (also the first judge of ``both``) has the local
        model behind it for when its quota runs out.
        """
        if self._shared_judge is not None:
            return self._shared_judge
        s = self.settings
        if s.judge == "api":
            return AnthropicChatModel(s.api_model, s.api_effort)
        if self.judge_mode() == "local":
            return self.local_judge
        return FallbackJudge(
            self.claude_judge, self.local_judge, self.judge_selection
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

    @property
    def acts_tagger(self) -> Optional[DialogueActTagger]:
        """Step d of ``skills analyze``: the tagger of #82 (``skills acts``).

        The card's choice (the fine-tuned encoder, the rules when its
        weights are missing); ``None`` when the card itself is missing.
        """
        if not Path(self.settings.act_card).is_file():
            return None
        return TaggerActStep(lambda: self.act_tagger("default"), self.act_card)

    @property
    def event_extractor(self) -> EventExtractor:
        """Step e of ``skills analyze``: the rules of #83 over the acts."""
        return RuleEventExtractor(self.settings.event_rules())

    def judge_setup(self) -> JudgeSetup:
        """Return the judges of the analysis (built on first use)."""
        mode = self.judge_mode()
        return JudgeSetup(
            mode=mode, first=self.confirmer, assessment=self.assessment,
            second=self.second_confirmer if mode == "both" else None,
        )

    def analysis_requests(
        self, transcripts: TranscriptStore, transcripts_dir: str
    ) -> AnalysisRequests:
        """Return the API's analysis use case over the ingested sessions.

        Analyses run as ``skills analyze`` children reading
        ``transcripts_dir`` (where ``transcripts`` keeps them).
        """
        s = self.settings
        return AnalysisRequests(
            AnalysisResults(FileSystemAnalysisStore(s.analysis_dir)),
            transcripts,
            SubprocessAnalysisLauncher(
                REPOSITORY_ROOT, transcripts_dir, s.analysis_dir
            ),
        )

    @cached_property
    def session_analysis(self) -> SessionAnalysis:
        """The whole flow on one session (``skills analyze``)."""
        s = self.settings
        catalogues = self.catalogues
        return SessionAnalysis(
            self.transcripts,
            FileSystemAnalysisStore(s.analysis_dir),
            self.rubric,
            catalogues.load,
            self.tagger,
            self.judge_setup,
            lambda: datetime.now(timezone.utc),
            acts=self.acts_tagger,
            events=self.event_extractor,
            max_confirmations=s.max_confirmations,
            embedding_model=s.embedding_model if s.semantic else "",
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
