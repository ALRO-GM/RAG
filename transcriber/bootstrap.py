"""Composition root: builds the transcriber from its settings.

:func:`build_container` wires the outbound adapters (file-system
repositories under ``settings.data_dir``, the PyAV decoder, the models
chosen by ``settings.runtime``, the job runner, the session locks, the
publisher) into every use case. Building it does no I/O and
loads no model: directories are created on the first write and models
on the first job, and the job runner unloads the models after every
job and gives the GPU memory back (:class:`TorchGpuMemory`); a stage
that runs out of GPU memory frees them and is tried once more. The
publisher is the directory one or, with ``publisher: "http"``, the one
that posts to the RAG (:func:`build_publisher`); that one also deletes
the RAG's copy when a session is deleted.
:func:`start_container` is for the process that runs the jobs
(the service): it builds the container and then fails the sessions a
previous process left queued or running (:class:`RecoverSessions`).
:func:`build_tunnel` and :func:`access_code` give ``transcriber serve``
its tunnel (a chisel client) and its access code.
"""

import atexit
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from transcriber.adapters.outbound.av_audio_decoder import (
    AvAudioDecoder, read_channels,
)
from transcriber.adapters.outbound.chisel_tunnel import (
    ChiselTunnel,
    TunnelSetupError,
    check_client,
    http_probe,
)
from transcriber.adapters.outbound.fs_access_code import (
    code_file,
    kept_access_code,
    new_access_code,
)
from transcriber.adapters.outbound.fake_models import (
    FakeAsrStream,
    FakeOverlapDetector,
    FakeSpeechRecogniser,
    FakeVad,
    FakeVoiceEncoder,
)
from transcriber.adapters.outbound.gpu_free_memory import NvidiaSmiFreeMemory
from transcriber.adapters.outbound.jsonl_live_log import JsonlLiveLog
from transcriber.adapters.outbound.jsonschema_event_check import (
    build_event_check,
)
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.fs_transcript_publisher import (
    DirectoryTranscriptPublisher,
)
from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)
from transcriber.adapters.outbound.http_analysis_judge import (
    HttpAnalysisJudge,
)
from transcriber.adapters.outbound.http_knowledge import HttpKnowledge
from transcriber.adapters.outbound.http_session_analysis import (
    HttpSessionAnalysis,
)
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)
from transcriber.adapters.outbound.hf_reference_corpus import (
    CORPORA_DIR,
    HuggingFaceCorpus,
)
from transcriber.adapters.outbound.lab_runner_diarizer import (
    LabRunnerDiarizer,
)
from transcriber.adapters.outbound.fs_lab_repository import (
    FileSystemLabRepository,
)
from transcriber.adapters.outbound.subprocess_lab_tool import (
    FileSystemLabEnvironment,
    JsonCatalogue,
    OsProcessTable,
    SubprocessLabTool,
)
from transcriber.adapters.outbound.system_clock import (
    SystemClock, SystemTimer,
)
from transcriber.adapters.outbound.local_name_model import LocalNameModel
from transcriber.adapters.outbound.fs_gpu_lease import (
    FileSystemGpuLease, lease_directory,
)
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.adapters.outbound.torch_gpu_memory import TorchGpuMemory
from transcriber.adapters.outbound.thread_session_locks import (
    ThreadSessionLocks,
)
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
)
from transcriber.application.browse import (
    ListSessions,
    ListSpeakers,
    SessionAudio,
    ShowSession,
    ShowTranscript,
)
from transcriber.application.consent import ShowConsentNotice
from transcriber.application.enrol import (
    EnrolSpeaker,
    ForgetSpeaker,
    ShowEnrolmentText,
)
from transcriber.application.lab import (
    CancelLabRun,
    LabAudio,
    LabLabels,
    LabResultsView,
    LabRunStatus,
    LabScores,
    LabSessionData,
    ListLabRuns,
    ResumeLabRun,
    RunLab,
    ShowCatalogue,
    StartLabRun,
)
from transcriber.application.lab_builtin import InProcessTools
from transcriber.application.lab_ports import LabRepository
from transcriber.application.lab_progress import LabOverview, ReferenceTurns
from transcriber.application.lab_reference import (
    CheckReferenceSpeech,
    ImportReference,
    ReferenceSummary,
)
from transcriber.application.ports import (
    AudioDecoder,
    Clock,
    GpuLease,
    Diarizer,
    OverlapDetector,
    SessionAnalysisPort,
    SessionLocks,
    SessionRepository,
    SpeechRecogniser,
    TranscriptPublisher,
    Tunnel,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.application.analysis import (
    AnalysisJudgeSetting,
    SessionAnalysisAccess,
)
from transcriber.application.knowledge import KnowledgeAccess
from transcriber.application.live import (
    LiveOptions, LiveTranscription,
)
from transcriber.application.live_session import LivePorts, LiveSessions
from transcriber.application.publish import PublishTranscript
from transcriber.application.record import (
    AppendAudio,
    CloseSession,
    DeleteSession,
    DescribeSession,
    ImportRecording,
    OpenSession,
)
from transcriber.application.recover import RecoverSessions
from transcriber.application.review import (
    ReassignUtterance,
    RenameSpeaker,
    UnnameSpeaker,
)
from transcriber.application.speaker_names import (
    FindSpeakerNames,
    QueueNameSearch,
    ShowSpeakerNames,
    UndoAutoName,
)
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
    TranscriptionStatus,
)
from transcriber import __version__
from transcriber.config import REPOSITORY, TranscriberSettings
from transcriber.domain.comparison import ComparisonParameters
from transcriber.domain.diarization import BUILTIN_RULE, DiarizerRule
from transcriber.domain.lab import BUILTIN_DIARIZER

LOG = logging.getLogger(__name__)


class _Releasable(Protocol):
    """A model adapter that can unload its model."""

    def release(self) -> None:
        """Unload the model."""
        ...


class ReleasableRecogniser(SpeechRecogniser, _Releasable, Protocol):
    """A speech recogniser whose model can be unloaded."""


class ReleasableEncoder(VoiceEncoder, _Releasable, Protocol):
    """A voice encoder whose model can be unloaded."""


@dataclass(frozen=True)
class LabUseCases:
    """The model lab's use cases (``docs/transcriber/LAB.md``)."""

    repository: LabRepository
    catalogue: ShowCatalogue
    start: StartLabRun
    status: LabRunStatus
    runs: ListLabRuns
    cancel: CancelLabRun
    resume: ResumeLabRun
    results: LabResultsView
    labels: LabLabels
    scores: LabScores
    audio: LabAudio
    import_reference: ImportReference
    check_reference: CheckReferenceSpeech
    reference_summary: ReferenceSummary
    overview: LabOverview
    reference_turns: ReferenceTurns


@dataclass(frozen=True)
class TranscriberContainer:
    """Everything the inbound adapters need, built once."""

    settings: TranscriberSettings
    sessions: SessionRepository
    voiceprints: VoiceprintRepository
    decoder: AudioDecoder
    recogniser: ReleasableRecogniser
    encoder: ReleasableEncoder
    runner: ThreadJobRunner
    publisher: TranscriptPublisher
    clock: Clock
    locks: SessionLocks
    open_session: OpenSession
    append_audio: AppendAudio
    close_session: CloseSession
    import_recording: ImportRecording
    describe_session: DescribeSession
    delete_session: DeleteSession
    enrol_speaker: EnrolSpeaker
    forget_speaker: ForgetSpeaker
    enrolment_text: ShowEnrolmentText
    consent_notice: ShowConsentNotice
    transcribe: TranscribeSession
    queue_transcription: QueueTranscription
    transcription_status: TranscriptionStatus
    publish: PublishTranscript
    rename_speaker: RenameSpeaker
    unname_speaker: UnnameSpeaker
    reassign_utterance: ReassignUtterance
    find_names: FindSpeakerNames
    queue_names: QueueNameSearch
    show_names: ShowSpeakerNames
    undo_auto_name: UndoAutoName
    recover_sessions: RecoverSessions
    list_sessions: ListSessions
    list_speakers: ListSpeakers
    show_session: ShowSession
    session_audio: SessionAudio
    show_transcript: ShowTranscript
    overlaps: Optional[OverlapDetector] = None
    lab: Optional[LabUseCases] = None
    analysis_judge: AnalysisJudgeSetting = AnalysisJudgeSetting(None)
    diarizer: Optional[LabRunnerDiarizer] = None
    diarizer_rule: DiarizerRule = BUILTIN_RULE
    session_analysis: SessionAnalysisAccess = SessionAnalysisAccess(None)
    knowledge: KnowledgeAccess = KnowledgeAccess(None)
    name_model: Optional[LocalNameModel] = None
    live: Optional[LiveSessions] = None

    def release_models(self) -> None:
        """Unload the speech, voice, overlap and name models (after a
        job)."""
        self.recogniser.release()
        self.encoder.release()
        if self.overlaps is not None:
            self.overlaps.release()
        if self.name_model is not None:
            self.name_model.release()

    def stop_runners(self) -> None:
        """Kill a lab diarizer's runner in progress; start no other one.

        The service calls it when it stops (and at exit): a runner lives
        in its own process group and would outlive the service.
        """
        if self.diarizer is not None:
            self.diarizer.close()


def build_overlap_detector(
    settings: TranscriberSettings,
) -> Optional[OverlapDetector]:
    """Return the overlap detector ``settings.overlap_detector`` picks.

    ``off`` (the default) gives none: overlaps are flagged from the
    voice windows only. ``fake`` is the deterministic stand-in (tests,
    demonstrations). No real detector ships yet: the candidate measured
    in ``docs/transcriber/OVERLAP.md`` needs a runtime the project's
    environment does not have (§3 there).
    """
    if settings.overlap_detector == "fake":
        return FakeOverlapDetector()
    return None


DIARIZE_DIR = "diarize"
"""Directory of ``data_dir`` holding a diarizer's audio while it runs."""


def build_diarizer(
    settings: TranscriberSettings,
    environ: Mapping[str, str],
    before_run: Callable[[], None] = lambda: None,
) -> Tuple[Optional[LabRunnerDiarizer], DiarizerRule]:
    """Return the lab diarizer and the rule ``settings.diarizer`` gives.

    With ``runtime: "fake"``, or a rule that only picks ``builtin``,
    there is no diarizer and the built-in grouping is used. Nothing is
    checked or loaded here (:func:`diarizer_problems` checks).
    ``before_run`` is called before each runner starts (the service
    unloads its recogniser there to free GPU memory).
    """
    rule = settings.diarizer
    if settings.runtime == "fake" or all(
        model == BUILTIN_DIARIZER for model in rule.models()
    ):
        return None, BUILTIN_RULE
    lab = settings.lab
    environment = FileSystemLabEnvironment(
        lab.envs_dir, lab.runners_dir, environ
    )
    tool = SubprocessLabTool(environment, environ, lab.step_timeout)
    return LabRunnerDiarizer(
        tool, environment, JsonCatalogue(lab.catalogue),
        settings.data_dir / DIARIZE_DIR, before_run,
    ), rule


def diarizer_problems(
    diarizer: Optional[Diarizer], rule: DiarizerRule
) -> Dict[str, str]:
    """Return, per model of ``rule`` that cannot run, why not."""
    if diarizer is None:
        return {}
    problems: Dict[str, str] = {}
    for model in rule.models():
        if model == BUILTIN_DIARIZER:
            continue
        problem = diarizer.problem(model)
        if problem is not None:
            problems[model] = problem
    return problems


def build_models(
    settings: TranscriberSettings,
) -> Tuple[ReleasableRecogniser, ReleasableEncoder]:
    """Return the recogniser and voice encoder ``settings.runtime`` picks.

    Neither loads its model here.
    """
    if settings.runtime == "fake":
        return FakeSpeechRecogniser(), FakeVoiceEncoder()
    return (
        WhisperRecogniser(settings.recogniser, device=settings.device),
        WavLMVoiceEncoder(settings.voice_model, device=settings.device),
    )


def build_lab(
    settings: TranscriberSettings,
    sessions: SessionRepository,
    voiceprints: VoiceprintRepository,
    decoder: AudioDecoder,
    recogniser: ReleasableRecogniser,
    encoder: ReleasableEncoder,
    runner: ThreadJobRunner,
    clock: Clock,
    environ: Mapping[str, str],
    locks: SessionLocks,
) -> LabUseCases:
    """Return the lab's use cases over the service's own adapters.

    The project's recogniser is reused for its own model name; another
    Whisper model of the catalogue gets an adapter of its own, made on
    first use and unloaded after every step (with the service's models).
    ``locks`` are the sessions' locks: a start or resume holds them
    against a concurrent deletion of the session.
    """
    lab = settings.lab
    repository = FileSystemLabRepository(settings.data_dir)
    environment = FileSystemLabEnvironment(
        lab.envs_dir, lab.runners_dir, environ
    )
    catalogue = JsonCatalogue(lab.catalogue)
    processes = OsProcessTable()
    extra: Dict[str, ReleasableRecogniser] = {}

    def recogniser_for(name: str) -> SpeechRecogniser:
        """Return the recogniser of model ``name`` (``""``: the service's)."""
        if not name or name == recogniser.model_name or (
            settings.runtime == "fake"
        ):
            return recogniser
        if name not in extra:
            extra[name] = WhisperRecogniser(name, device=settings.device)
        return extra[name]

    def release() -> None:
        """Unload every model the lab used in this process."""
        recogniser.release()
        encoder.release()
        for model in extra.values():
            model.release()

    comparison = ComparisonParameters(
        frame=lab.frame, collar=lab.collar,
        change_tolerance=lab.change_tolerance, cut_merge=lab.cut_merge,
        word_time_tolerance=lab.word_time_tolerance,
    )
    internal = InProcessTools(
        recogniser_for, encoder, voiceprints, settings.transcription,
        release,
    )
    external = SubprocessLabTool(environment, environ, lab.step_timeout)
    execute = RunLab(
        sessions, voiceprints, decoder, repository, catalogue, external,
        internal, encoder, clock, processes, settings.transcription,
        comparison, release,
    )
    status = LabRunStatus(sessions, repository, runner, processes)
    start = StartLabRun(
        sessions, repository, catalogue, runner, execute, clock, processes,
        decoder, locks,
    )
    return LabUseCases(
        repository=repository,
        catalogue=ShowCatalogue(catalogue, environment),
        start=start,
        status=status,
        runs=ListLabRuns(sessions, repository, status),
        cancel=CancelLabRun(repository, status),
        # a lock per run (keyed by its job key), apart from the sessions'
        resume=ResumeLabRun(
            repository, start, status, processes, ThreadSessionLocks()
        ),
        results=LabResultsView(status, repository),
        labels=LabLabels(sessions, voiceprints, repository, decoder, clock),
        scores=LabScores(status, repository, comparison),
        audio=LabAudio(status, repository),
        import_reference=ImportReference(
            sessions, repository,
            HuggingFaceCorpus(settings.data_dir / CORPORA_DIR), decoder,
            ImportRecording(sessions, locks, recogniser.supported_languages),
            locks, clock,
        ),
        check_reference=CheckReferenceSpeech(
            sessions, repository, decoder, locks
        ),
        reference_summary=ReferenceSummary(sessions, repository),
        overview=LabOverview(sessions, repository, status),
        reference_turns=ReferenceTurns(sessions, repository),
    )


def build_publisher(
    settings: TranscriberSettings,
    environ: Optional[Mapping[str, str]] = None,
) -> TranscriptPublisher:
    """Return the publisher ``settings.publisher`` picks.

    The HTTP publisher reads its token from ``environ`` (the process
    environment if omitted) when it publishes, not here.
    """
    http = settings.publish_http
    if settings.publisher == "http" and http.url is not None:
        return HttpTranscriptPublisher(
            http.url, http.token_env,
            os.environ if environ is None else environ,
            timeout=http.timeout, retries=http.retries,
            backoff_initial=http.backoff_initial,
            backoff_factor=http.backoff_factor,
            token_file=http.token_file,
            retract_timeout=http.retract_timeout,
        )
    return DirectoryTranscriptPublisher(settings.published())


def build_gpu_lease(environ: Mapping[str, str]) -> GpuLease:
    """Return the file lease of the live mode (``gpu-lease.v1``)."""
    return FileSystemGpuLease(lease_directory(environ))


read_live_audio = read_channels
"""Reads a sound file for ``live-file`` (one stream per channel or one)."""


def build_live(options: LiveOptions) -> LiveTranscription:
    """Return the live transcription: Silero VAD, faster-whisper stream.

    Both adapters load their model on first use (``live`` dependency
    group); nothing here touches the GPU yet.
    """
    from transcriber.adapters.outbound.faster_whisper_stream import (
        FasterWhisperStream,
    )
    from transcriber.adapters.outbound.silero_vad import SileroVad

    asr = FasterWhisperStream(
        model=options.model, device=options.device,
        compute_type=options.compute_type, language=options.language,
    )
    return LiveTranscription(
        SileroVad, asr, SystemClock(), SystemTimer(), options.settings
    )


CONTRACTS_DIR = REPOSITORY / "contracts"
"""The platform's contracts (``live.event.v1`` checks the live events)."""


def build_fake_live(options: LiveOptions) -> LiveTranscription:
    """Return the live transcription over the fake VAD and recogniser."""
    return LiveTranscription(
        FakeVad, FakeAsrStream(options.language), SystemClock(),
        SystemTimer(), options.settings,
    )


def build_live_sessions(
    settings: TranscriberSettings,
    environ: Mapping[str, str],
    publisher: TranscriptPublisher,
    analysis: Optional[SessionAnalysisPort],
) -> LiveSessions:
    """Return the live sessions of ``/api/live`` (LV-2 to LV-11).

    Real runtime: Silero VAD and faster-whisper (``live`` dependency
    group, loaded when a session starts), the GPU lease of
    ``gpu-lease.v1`` and the free memory from ``nvidia-smi``; on a
    machine whose ``device`` is ``cpu`` the CPU model runs and no lease
    is taken. Fake runtime: the loudness stand-ins, on the "CPU".
    """
    check, why = build_event_check(CONTRACTS_DIR)
    if check is None:
        LOG.warning("Live events are not contract-checked: %s", why)
    fake = settings.runtime == "fake"
    ports = LivePorts(
        build=build_fake_live if fake else build_live,
        clock=SystemClock(), timer=SystemTimer(),
        lease=None if fake else build_gpu_lease(environ),
        memory=None if fake else NvidiaSmiFreeMemory(),
        log=JsonlLiveLog(settings.data_dir),
        check=check, publisher=publisher, analysis=analysis,
    )
    device = "cpu" if fake or settings.device == "cpu" else "cuda"
    return LiveSessions(ports, settings.live, device, __version__)


def build_container(
    settings: Optional[TranscriberSettings] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> TranscriberContainer:
    """Return the container for ``settings`` (the defaults if omitted).

    ``environ`` holds the HTTP publisher's token (the process
    environment if omitted).
    """
    settings = settings or TranscriberSettings()
    recogniser, encoder = build_models(settings)
    overlaps = build_overlap_detector(settings)
    environment = os.environ if environ is None else environ
    diarizer, rule = build_diarizer(
        settings, environment, recogniser.release
    )

    name_model = build_name_model(settings)

    def release_speech_models() -> None:
        """Unload the speech, voice and overlap models."""
        recogniser.release()
        encoder.release()
        if overlaps is not None:
            overlaps.release()

    def release_models() -> None:
        """Unload the speech, voice, overlap and name models."""
        release_speech_models()
        if name_model is not None:
            name_model.release()

    # The runner frees the GPU after every job: the models are unloaded,
    # collected and PyTorch's cache is given back (#117).
    gpu = TorchGpuMemory(release_models)

    sessions = FileSystemSessionRepository(settings.data_dir)
    voiceprints = FileSystemVoiceprintRepository(
        settings.data_dir, encoder.model_name
    )
    decoder = AvAudioDecoder()
    runner = ThreadJobRunner(
        after_job=gpu.free,
        lease=FileSystemGpuLease(
            lease_directory(environment), holder="transcriber-batch",
            priority="batch",
        ),
    )
    publisher = build_publisher(settings, environ)
    clock = SystemClock()
    locks = ThreadSessionLocks()
    parameters = settings.transcription
    transcribe = TranscribeSession(
        sessions, voiceprints, decoder, recogniser, encoder, locks,
        parameters, overlaps, diarizer, rule,
        gpu if settings.runtime != "fake" else None,
    )
    status = TranscriptionStatus(sessions, runner)
    rename = RenameSpeaker(sessions, voiceprints, locks)
    unname = UnnameSpeaker(sessions, voiceprints, locks)
    find_names = FindSpeakerNames(
        sessions, sessions, locks, rename, unname, clock, name_model,
        settings.names.search, free_models=release_speech_models,
        gpu=gpu if settings.runtime != "fake" else None,
    )
    lab = build_lab(
        settings, sessions, voiceprints, decoder, recogniser, encoder,
        runner, clock, environment, locks,
    )
    return TranscriberContainer(
        settings=settings,
        sessions=sessions,
        voiceprints=voiceprints,
        decoder=decoder,
        recogniser=recogniser,
        encoder=encoder,
        runner=runner,
        publisher=publisher,
        clock=clock,
        locks=locks,
        open_session=OpenSession(
            sessions, clock, locks, recogniser.supported_languages
        ),
        append_audio=AppendAudio(sessions, locks),
        close_session=CloseSession(sessions, locks),
        import_recording=ImportRecording(
            sessions, locks, recogniser.supported_languages
        ),
        describe_session=DescribeSession(sessions, locks),
        delete_session=DeleteSession(
            sessions, locks, LabSessionData(lab.repository, lab.status),
            publisher if isinstance(publisher, HttpTranscriptPublisher)
            else None,
        ),
        enrol_speaker=EnrolSpeaker(
            decoder, recogniser, encoder, voiceprints, clock, runner,
            parameters, settings.enrolment,
        ),
        forget_speaker=ForgetSpeaker(voiceprints),
        enrolment_text=ShowEnrolmentText(settings.api.consent_version),
        consent_notice=ShowConsentNotice(settings.api.consent_version),
        transcribe=transcribe,
        queue_transcription=QueueTranscription(
            sessions, voiceprints, runner, transcribe, locks,
            after=(
                find_names.after_transcription if settings.names.enabled
                else None
            ),
        ),
        transcription_status=status,
        publish=PublishTranscript(sessions, publisher, locks, runner),
        rename_speaker=rename,
        unname_speaker=unname,
        find_names=find_names,
        queue_names=QueueNameSearch(sessions, runner, find_names, locks),
        show_names=ShowSpeakerNames(sessions, sessions, runner),
        undo_auto_name=UndoAutoName(sessions, sessions, locks, unname),
        reassign_utterance=ReassignUtterance(sessions, voiceprints, locks),
        recover_sessions=RecoverSessions(sessions, locks),
        list_sessions=ListSessions(sessions),
        list_speakers=ListSpeakers(voiceprints),
        show_session=ShowSession(sessions, status),
        session_audio=SessionAudio(sessions),
        show_transcript=ShowTranscript(sessions),
        overlaps=overlaps,
        lab=lab,
        analysis_judge=AnalysisJudgeSetting(
            HttpAnalysisJudge(publisher.base_url, publisher.bearer_token)
            if isinstance(publisher, HttpTranscriptPublisher) else None
        ),
        diarizer=diarizer,
        diarizer_rule=rule,
        session_analysis=SessionAnalysisAccess(
            HttpSessionAnalysis(publisher.base_url, publisher.bearer_token)
            if isinstance(publisher, HttpTranscriptPublisher) else None
        ),
        knowledge=KnowledgeAccess(
            HttpKnowledge(publisher.base_url, publisher.bearer_token)
            if isinstance(publisher, HttpTranscriptPublisher) else None
        ),
        name_model=name_model,
        live=build_live_sessions(
            settings, environment, publisher,
            HttpSessionAnalysis(publisher.base_url, publisher.bearer_token)
            if isinstance(publisher, HttpTranscriptPublisher) else None,
        ),
    )


def build_name_model(
    settings: TranscriberSettings,
) -> Optional[LocalNameModel]:
    """Return the local model for names (``None`` with fake models).

    Nothing is loaded here; the model loads when a search reads its
    first candidate and is freed after the job.
    """
    if settings.runtime == "fake":
        return None
    names = settings.names
    return LocalNameModel(
        names.models, device=settings.device,
        wait_seconds=names.wait_seconds, poll_seconds=names.poll_seconds,
    )


def start_container(
    settings: Optional[TranscriberSettings] = None,
) -> TranscriberContainer:
    """Build the container of the job-running process and recover.

    Every session a previous process left ``queued`` or ``running`` is
    failed as ``interrupted`` (its job was lost), so it can be queued
    again or deleted. A session whose record cannot be read is left
    alone and logged. Only the one process that runs the jobs for
    ``settings.data_dir`` (the service) may call it: a session failed
    here is assumed to have no live job.
    """
    container = build_container(settings)
    if container.diarizer is not None:
        swept = container.diarizer.sweep()
        if swept:
            LOG.warning(
                "Deleted %d diarizer work folder(s) left by a stopped"
                " service.", swept,
            )
        atexit.register(container.stop_runners)
    problems = diarizer_problems(
        container.diarizer, container.diarizer_rule
    )
    for model, problem in problems.items():
        LOG.warning(
            "Diarizer %s cannot run (%s): sessions that would use it get"
            " the built-in grouping, which over-counts speakers"
            " (docs/transcriber/LAB.md §13.8).", model, problem,
        )
    recovery = container.recover_sessions()
    for session_id in recovery.damaged:
        LOG.warning("Session %s cannot be read; left as it is", session_id)
    return container


def build_tunnel(settings: TranscriberSettings, port: int) -> Tunnel:
    """Return the tunnel to the server for the API on ``port``.

    It is reported ``connected`` only once ``/api/health`` answers
    through the server.
    """
    tunnel = settings.tunnel
    return ChiselTunnel(
        tunnel, port, settings.data_dir,
        probe=http_probe(tunnel.health_url(), tunnel.probe_timeout),
    )


def tunnel_problem(settings: TranscriberSettings) -> Optional[str]:
    """Return why the tunnel client cannot run, or None if it can.

    The chisel executable must be there and the credential file safe
    (:func:`~transcriber.adapters.outbound.chisel_tunnel.check_client`).
    """
    try:
        check_client(settings.tunnel)
    except TunnelSetupError as exc:
        return str(exc)
    return None


def access_code(
    settings: TranscriberSettings, keep: bool, renew: bool = False
) -> str:
    """Return a fresh access code, or with ``keep`` the kept one.

    The kept code lives in ``<data_dir>/access_code`` (mode 600) and is
    made on first use; see
    :func:`~transcriber.adapters.outbound.fs_access_code.kept_access_code`
    for when it is refused. ``renew`` deletes the kept code first, so a
    new one is made and kept.
    """
    if keep:
        path = code_file(settings.data_dir)
        if renew:
            path.unlink(missing_ok=True)
        return kept_access_code(path)
    return new_access_code()
