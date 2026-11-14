"""Composition root: builds the transcriber from its settings.

:func:`build_container` wires the outbound adapters (file-system
repositories under ``settings.data_dir``, the PyAV decoder, the models
chosen by ``settings.runtime``, the job runner, the session locks, the
publisher) into every use case. Building it does no I/O and
loads no model: directories are created on the first write and models
on the first job, and the job runner unloads the models after every
job. The publisher is the directory one or, with ``publisher:
"http"``, the one that posts to the RAG (:func:`build_publisher`).
:func:`start_container` is for the process that runs the jobs
(the service): it builds the container and then fails the sessions a
previous process left queued or running (:class:`RecoverSessions`).
:func:`build_tunnel` and :func:`access_code` give ``transcriber serve``
its tunnel (a chisel client) and its access code.
"""

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

from transcriber.adapters.outbound.av_audio_decoder import AvAudioDecoder
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
    FakeOverlapDetector,
    FakeSpeechRecogniser,
    FakeVoiceEncoder,
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
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)
from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
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
from transcriber.application.ports import (
    AudioDecoder,
    Clock,
    OverlapDetector,
    SessionLocks,
    SessionRepository,
    SpeechRecogniser,
    TranscriptPublisher,
    Tunnel,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.application.publish import PublishTranscript
from transcriber.application.record import (
    AppendAudio,
    CloseSession,
    DeleteSession,
    ImportRecording,
    OpenSession,
)
from transcriber.application.recover import RecoverSessions
from transcriber.application.review import RenameSpeaker, ReassignUtterance
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
    TranscriptionStatus,
)
from transcriber.config import TranscriberSettings

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
    reassign_utterance: ReassignUtterance
    recover_sessions: RecoverSessions
    list_sessions: ListSessions
    list_speakers: ListSpeakers
    show_session: ShowSession
    session_audio: SessionAudio
    show_transcript: ShowTranscript
    overlaps: Optional[OverlapDetector] = None

    def release_models(self) -> None:
        """Unload the speech, voice and overlap models (after a job)."""
        self.recogniser.release()
        self.encoder.release()
        if self.overlaps is not None:
            self.overlaps.release()


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
        )
    return DirectoryTranscriptPublisher(settings.published())


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

    def release_models() -> None:
        """Unload the models; the runner calls it after every job."""
        recogniser.release()
        encoder.release()
        if overlaps is not None:
            overlaps.release()

    sessions = FileSystemSessionRepository(settings.data_dir)
    voiceprints = FileSystemVoiceprintRepository(
        settings.data_dir, encoder.model_name
    )
    decoder = AvAudioDecoder()
    runner = ThreadJobRunner(after_job=release_models)
    publisher = build_publisher(settings, environ)
    clock = SystemClock()
    locks = ThreadSessionLocks()
    parameters = settings.transcription
    transcribe = TranscribeSession(
        sessions, voiceprints, decoder, recogniser, encoder, locks,
        parameters, overlaps,
    )
    status = TranscriptionStatus(sessions, runner)
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
        delete_session=DeleteSession(sessions, locks),
        enrol_speaker=EnrolSpeaker(
            decoder, recogniser, encoder, voiceprints, clock, runner,
            parameters, settings.enrolment,
        ),
        forget_speaker=ForgetSpeaker(voiceprints),
        enrolment_text=ShowEnrolmentText(settings.api.consent_version),
        consent_notice=ShowConsentNotice(settings.api.consent_version),
        transcribe=transcribe,
        queue_transcription=QueueTranscription(
            sessions, voiceprints, runner, transcribe, locks
        ),
        transcription_status=status,
        publish=PublishTranscript(sessions, publisher),
        rename_speaker=RenameSpeaker(sessions, voiceprints, locks),
        reassign_utterance=ReassignUtterance(sessions, voiceprints, locks),
        recover_sessions=RecoverSessions(sessions, locks),
        list_sessions=ListSessions(sessions),
        list_speakers=ListSpeakers(voiceprints),
        show_session=ShowSession(sessions, status),
        session_audio=SessionAudio(sessions),
        show_transcript=ShowTranscript(sessions),
        overlaps=overlaps,
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
