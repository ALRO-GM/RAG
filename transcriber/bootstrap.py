"""Composition root: builds the transcriber from its settings.

:func:`build_container` wires the outbound adapters (file-system
repositories under ``settings.data_dir``, the PyAV decoder, the models
chosen by ``settings.runtime``, the job runner, the directory
publisher) into every use case. Building it does no I/O and loads no
model: directories are created on the first write and models on the
first job, and the job runner unloads the models after every job.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from transcriber.adapters.outbound.av_audio_decoder import AvAudioDecoder
from transcriber.adapters.outbound.fake_models import (
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
from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
)
from transcriber.application.enrol import EnrolSpeaker, ForgetSpeaker
from transcriber.application.ports import (
    AudioDecoder,
    Clock,
    SessionRepository,
    SpeechRecogniser,
    TranscriptPublisher,
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
from transcriber.application.review import RenameSpeaker, ReassignUtterance
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
    TranscriptionStatus,
)
from transcriber.config import TranscriberSettings


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
    open_session: OpenSession
    append_audio: AppendAudio
    close_session: CloseSession
    import_recording: ImportRecording
    delete_session: DeleteSession
    enrol_speaker: EnrolSpeaker
    forget_speaker: ForgetSpeaker
    transcribe: TranscribeSession
    queue_transcription: QueueTranscription
    transcription_status: TranscriptionStatus
    publish: PublishTranscript
    rename_speaker: RenameSpeaker
    reassign_utterance: ReassignUtterance

    def release_models(self) -> None:
        """Unload the speech and voice models (after a job)."""
        self.recogniser.release()
        self.encoder.release()


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


def build_container(
    settings: Optional[TranscriberSettings] = None,
) -> TranscriberContainer:
    """Return the container for ``settings`` (the defaults if omitted)."""
    settings = settings or TranscriberSettings()
    recogniser, encoder = build_models(settings)

    def release_models() -> None:
        """Unload both models; the runner calls it after every job."""
        recogniser.release()
        encoder.release()

    sessions = FileSystemSessionRepository(settings.data_dir)
    voiceprints = FileSystemVoiceprintRepository(
        settings.data_dir, encoder.model_name
    )
    decoder = AvAudioDecoder()
    runner = ThreadJobRunner(after_job=release_models)
    publisher = DirectoryTranscriptPublisher(settings.published())
    clock = SystemClock()
    parameters = settings.transcription
    transcribe = TranscribeSession(
        sessions, voiceprints, decoder, recogniser, encoder, parameters
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
        open_session=OpenSession(sessions, clock),
        append_audio=AppendAudio(sessions),
        close_session=CloseSession(sessions),
        import_recording=ImportRecording(sessions),
        delete_session=DeleteSession(sessions),
        enrol_speaker=EnrolSpeaker(
            decoder, recogniser, encoder, voiceprints, clock, parameters,
            settings.enrolment,
        ),
        forget_speaker=ForgetSpeaker(voiceprints),
        transcribe=transcribe,
        queue_transcription=QueueTranscription(
            sessions, voiceprints, runner, transcribe
        ),
        transcription_status=TranscriptionStatus(sessions, runner),
        publish=PublishTranscript(sessions, publisher),
        rename_speaker=RenameSpeaker(sessions, voiceprints),
        reassign_utterance=ReassignUtterance(sessions, voiceprints),
    )
