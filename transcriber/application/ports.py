"""Ports of the transcriber's use cases.

Each port is a :class:`~typing.Protocol` an outbound adapter implements
(``docs/transcriber/ARCHITECTURE.md`` §3). The use cases depend only on
these and on the domain; the fakes in ``tests/unit/transcriber/fakes.py``
implement them in memory.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import (
    AbstractSet,
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.gpu_lease import AckFile, LeaseFile
from transcriber.domain.diarization import Diarization
from transcriber.domain.live import Hypothesis
from transcriber.domain.models import (
    ConsentRecord,
    Session,
    SessionState,
    Transcript,
    Voiceprint,
    Word,
)
from transcriber.domain.speaker_names import SpeakerNames
from transcriber.domain.voices import FloatArray

PCM_FORMAT = "pcm_s16le_16000_mono"
"""Format of audio recorded from the phone: raw 16-bit PCM, 16 kHz, mono."""

PCM_SAMPLE_WIDTH = 2
"""Bytes per sample of :data:`PCM_FORMAT`; a chunk holds whole samples."""


@dataclass(frozen=True)
class RecordedAudio:
    """The stored bytes of a recording and what they are.

    ``format`` is :data:`PCM_FORMAT` for audio uploaded in chunks, or
    the file extension (``wav``, ``m4a``...) of an imported recording.
    """

    data: bytes
    format: str


@dataclass(frozen=True)
class Audio:
    """A decoded recording: mono samples in [-1, 1] at ``sample_rate``."""

    samples: npt.NDArray[np.float32]
    sample_rate: int

    @property
    def duration(self) -> float:
        """Length of the recording in seconds."""
        return float(self.samples.shape[0]) / self.sample_rate

    def cut(self, start: float, end: float) -> npt.NDArray[np.float32]:
        """Return the samples between two times (clamped to the audio)."""
        first = max(int(start * self.sample_rate), 0)
        last = min(int(end * self.sample_rate), int(self.samples.shape[0]))
        return self.samples[first:max(last, first)]


class PlaybackAudio(Protocol):
    """A session's audio as a WAV file (16-bit mono), read by byte ranges.

    Holds an open file until :meth:`close`; reading never loads more
    than the bytes asked for.
    """

    @property
    def size(self) -> int:
        """Return the size of the whole WAV file in bytes."""
        ...

    def read(self, offset: int, length: int) -> bytes:
        """Return up to ``length`` bytes from ``offset`` (fewer at the end)."""
        ...

    def close(self) -> None:
        """Release the file."""
        ...


class RecognisedSpeech(BaseModel):
    """What a speech recogniser heard: timed words and the language."""

    model_config = ConfigDict(frozen=True)

    words: List[Word]
    language: Optional[str] = None


class LanguageSpan(BaseModel):
    """The language the recogniser detected in a stretch of audio."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    language: str = Field(min_length=2)


class WordScores(BaseModel):
    """The recogniser's own probability of each word it was given.

    ``words`` are the words given, in the same order, each with its
    ``probability`` (``None`` where the model could not score it).
    ``languages`` are the stretches whose language the recogniser
    detected on the way (one per scoring window; empty when it does not
    detect).
    """

    model_config = ConfigDict(frozen=True)

    words: List[Word]
    languages: List[LanguageSpan] = Field(default_factory=list)


class JobStatus(BaseModel):
    """Where the job of a session is: its state, stage and progress.

    ``fraction`` is the share of the whole job done, from 0 to 1.
    ``message`` is the last note the job gave (:meth:`Progress.note`),
    for the person waiting.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    state: SessionState
    stage: Optional[str] = None
    fraction: float = Field(default=0.0, ge=0, le=1)
    message: Optional[str] = None


class AudioDecoder(Protocol):
    """Turn stored recordings into samples."""

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return ``recording`` as mono samples."""
        ...


class SpeechRecogniser(Protocol):
    """Speech-to-text with word timestamps."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the recognition model."""
        ...

    @property
    def supported_languages(self) -> AbstractSet[str]:
        """Return the base language codes the recogniser knows."""
        ...

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Return the words spoken in ``audio`` with their times.

        ``language`` is a base code of :attr:`supported_languages`, or
        ``None`` to detect it. The words carry no probability yet.
        """
        ...

    def score_words(
        self, audio: Audio, words: Sequence[Word],
        language: Optional[str] = None,
    ) -> WordScores:
        """Return how probable the model finds each of ``words``.

        Called once the words are final (seams resolved, words over
        silence dropped), so each is scored against the audio it was
        heard in.
        """
        ...


class OverlapDetector(Protocol):
    """Find where two or more people speak at once.

    Used only to flag sentences (``overlap``); the voices are still
    found by the voice encoder and the domain's grouping.
    """

    @property
    def model_name(self) -> str:
        """Return the identifier of the detection model."""
        ...

    def overlaps(self, audio: Audio) -> List[Tuple[float, float]]:
        """Return the ``(start, end)`` seconds with two or more voices."""
        ...

    def release(self) -> None:
        """Unload the model, if one is loaded."""
        ...


class Diarizer(Protocol):
    """Separate the voices of a recording with a diarization model.

    An alternative to the built-in grouping of voice windows (TR-18):
    the models are those of the lab's catalogue, named by their id.
    The model is loaded for the call and freed after it.
    """

    def problem(self, model: str) -> Optional[str]:
        """Return why ``model`` cannot run here, or ``None`` if it can."""
        ...

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Diarization:
        """Return who speaks when in ``audio`` according to ``model``.

        ``num_speakers`` is passed to models that take it. Raises an
        exception (any) when the model fails. ``cancelled`` is polled
        while the model runs: when it turns true the model is stopped
        and a ``LabStepError`` with code ``cancelled`` is raised.
        """
        ...


class VoiceEncoder(Protocol):
    """Map windows of audio to voice embeddings.

    It only embeds; grouping the vectors into voices is domain logic.
    """

    @property
    def model_name(self) -> str:
        """Return the identifier of the voice model."""
        ...

    def embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """Return one embedding row per window (seconds) of ``audio``.

        With no windows the result has zero rows.
        """
        ...


class SessionRepository(Protocol):
    """Sessions, their audio, their words and their transcripts."""

    def get(self, session_id: str) -> Optional[Session]:
        """Return the session, or ``None`` if it does not exist."""
        ...

    def ids(self) -> List[str]:
        """Return the id of every stored session."""
        ...

    def create(self, session: Session) -> None:
        """Store a new session; it must not exist yet."""
        ...

    def save(self, session: Session) -> None:
        """Replace the stored session with ``session``."""
        ...

    def delete(self, session_id: str) -> None:
        """Remove the session with its audio, words and transcript."""
        ...

    def append_chunk(self, session_id: str, number: int, data: bytes) -> None:
        """Store chunk ``number`` (from 0) of :data:`PCM_FORMAT` audio.

        Only the next chunk, or the last stored chunk again, is
        accepted: storing the last number twice keeps it once (with the
        last bytes given). A ``number`` past the next one, or below the
        last stored one, is an :class:`InvalidInputError` (it would
        leave a gap or drop the chunks after it).
        """
        ...

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Store a whole recording (an imported file)."""
        ...

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the stored audio of the session."""
        ...

    def playback(self, session_id: str) -> PlaybackAudio:
        """Return the session's audio as WAV for the player.

        The rendering is made once, kept beside the audio until the
        audio changes, and removed with the session; while the session
        is ``recording`` it is the PCM received so far behind a WAV
        header. :class:`NotFoundError` without any audio.
        """
        ...

    def save_words(self, session_id: str, words: Sequence[Word]) -> None:
        """Store the attributed words of the session (for the player)."""
        ...

    def words(self, session_id: str) -> List[Word]:
        """Return the stored words of the session (empty if none)."""
        ...

    def save_transcript(self, transcript: Transcript) -> None:
        """Store the transcript of ``transcript.session``."""
        ...

    def transcript(self, session_id: str) -> Optional[Transcript]:
        """Return the transcript of the session, if there is one."""
        ...


class VoiceprintRepository(Protocol):
    """Enrolled voiceprints and the consent that goes with each."""

    def save(self, voiceprint: Voiceprint, consent: ConsentRecord) -> None:
        """Store a voiceprint and its consent, replacing earlier ones."""
        ...

    def get(self, speaker_id: str) -> Optional[Voiceprint]:
        """Return the voiceprint of ``speaker_id``, if enrolled."""
        ...

    def consent(self, speaker_id: str) -> Optional[ConsentRecord]:
        """Return the consent record of ``speaker_id``, if enrolled."""
        ...

    def all(self) -> List[Voiceprint]:
        """Return every enrolled voiceprint."""
        ...

    def delete(self, speaker_id: str) -> bool:
        """Remove voiceprint and consent; tell whether they existed."""
        ...


class TranscriptPublisher(Protocol):
    """Hand a finished transcript over to the RAG."""

    def publish(self, document: Mapping[str, Any]) -> None:
        """Deliver one ``transcript.v1`` document."""
        ...


class PublishedTranscripts(Protocol):
    """The copies of transcripts the RAG received from this transcriber."""

    def retract(self, session_id: str) -> None:
        """Delete the RAG's copy of ``session_id`` (none is not an error).

        Raises :class:`~transcriber.domain.errors.PublishError` when the
        RAG refuses or cannot be reached.
        """
        ...


class AnalysisJudge(Protocol):
    """The judge setting of the RAG's soft-skills analysis.

    Answers are the RAG's own JSON objects (``mode``, ``modes``,
    ``claude_quota`` and when the quota was last seen). Raises
    :class:`~transcriber.domain.errors.AnalysisUnavailableError` when
    the RAG cannot be reached or refuses.
    """

    def status(self) -> Dict[str, Any]:
        """Return the judge in use and Claude's quota status."""
        ...

    def choose(self, mode: str) -> Dict[str, Any]:
        """Set the judge of the next analyses; return the new status."""
        ...


class KnowledgePort(Protocol):
    """The RAG's knowledge routes (``/knowledge/...``).

    ``call`` sends one request below ``/knowledge/`` and returns the
    RAG's JSON answer (``None`` for an empty one). The RAG's refusals
    become domain errors carrying its message:
    :class:`~transcriber.domain.errors.InvalidInputError` (400/422),
    :class:`~transcriber.domain.errors.NotFoundError` (404),
    :class:`~transcriber.domain.errors.StateError` (409),
    :class:`~transcriber.domain.errors.TooLargeError` (413),
    :class:`~transcriber.domain.errors.UnsupportedMediaError` (415) and
    :class:`~transcriber.domain.errors.RagUnavailableError` otherwise.
    """

    def call(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> Any:
        """Send ``method path`` with ``body``; return the JSON answer."""
        ...


class SessionAnalysisPort(Protocol):
    """The RAG's whole analysis of a published session (#88).

    Answers are the RAG's own JSON (the run's steps; a step's result)
    or the report's HTML. Raises
    :class:`~transcriber.domain.errors.NotFoundError` for an unknown
    analysis, step or report,
    :class:`~transcriber.domain.errors.StateError` when the analysis is
    already running and
    :class:`~transcriber.domain.errors.AnalysisUnavailableError` when
    the RAG cannot be reached or refuses.
    """

    def start(
        self,
        session_id: str,
        restart: bool,
        synthetic: bool,
        steps: Sequence[str] = (),
        job: Optional[str] = None,
        claude_synthesis: bool = False,
        claude_models: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Start the analysis; return at once.

        ``steps`` (not empty) runs only those steps again; ``job`` and
        ``claude_synthesis`` are the job sought and the leave to use
        Claude for the profiles, sent only when ``job`` is given;
        ``claude_models`` the checked Claude model per judged step
        (sent only when not empty).
        """
        ...

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return every step's record and whether it runs now."""
        ...

    def result(self, session_id: str, step: str) -> Any:
        """Return one step's result."""
        ...

    def cancel(self, session_id: str) -> None:
        """Stop the running analysis."""
        ...

    def report(self, session_id: str, position: int) -> str:
        """Return the HTML report of the ``position``-th person."""
        ...


class Progress(Protocol):
    """Receive the stages of a long job as it runs."""

    def report(self, stage: str, fraction: float) -> None:
        """``stage`` started; ``fraction`` of the whole job is done."""
        ...

    def note(self, message: str) -> None:
        """Tell the person waiting something about the job."""
        ...


Job = Callable[[Progress], None]
"""Work submitted to a :class:`JobRunner`; it reports to ``Progress``."""

T = TypeVar("T")


class JobRunner(Protocol):
    """Run long jobs (one at a time on the one GPU) and report on them.

    Every use of the models goes through the runner: transcriptions as
    queued jobs (:meth:`submit`), enrolments as jobs the caller waits
    for (:meth:`run`). So two jobs never share the GPU, and the models
    are released after each, whatever kind it was.
    """

    def submit(self, session_id: str, job: Job) -> None:
        """Queue ``job`` for ``session_id``."""
        ...

    def run(self, work: Callable[[], T]) -> T:
        """Queue ``work`` like any job, wait for it, return its result.

        An exception raised by ``work`` is raised to the caller. The
        call waits for the jobs queued before it.
        """
        ...

    def status(self, session_id: str) -> Optional[JobStatus]:
        """Return the status of the session's job, if one was submitted."""
        ...


class GpuLease(Protocol):
    """The cooperative GPU lease (``gpu-lease.v1``, PL-12).

    The live mode acquires it and keeps it alive; the other side (the
    image worker, the transcriber's own batch jobs) reads it and stays
    off the card while it is valid. A holder that dies stops renewing
    it and it expires by itself.
    """

    def acquire(
        self, need_mib: int, starts_at: Optional[datetime] = None
    ) -> LeaseFile:
        """Take the lease, renewing it every 5 s until :meth:`release`.

        Raises :class:`StateError` if another process holds a valid one.
        """
        ...

    def release(self) -> None:
        """Stop renewing and delete this holder's lease file."""
        ...

    def current(self) -> Optional[LeaseFile]:
        """Return the valid lease (not expired, holder alive), if any."""
        ...

    def ack(self) -> Optional[AckFile]:
        """Return the other side's ack of the current lease, if any."""
        ...

    def freed(self) -> bool:
        """Tell whether the other side acknowledged the current lease."""
        ...

    def clear(self) -> bool:
        """Delete the lease whoever holds it (operator); tell if one was."""
        ...


class SessionLocks(Protocol):
    """Mutual exclusion per session, across requests and the job.

    Every use case that reads a session and then writes it does both
    while holding the session's lock, so two requests (or a request
    and the job) never act on the same stale copy.
    """

    def hold(self, session_id: str) -> ContextManager[None]:
        """Return a context that holds the lock of ``session_id``.

        The lock is re-entrant for the thread holding it.
        """
        ...


class SessionLabData(Protocol):
    """What the model lab stored about a session, seen when deleting it.

    The lab keeps a copy of the recording, denoised audio, transcripts,
    speaker mappings and the owner's labels per session
    (``docs/transcriber/LAB.md``); they go with the session.
    """

    def active_runs(self, session_id: str) -> List[str]:
        """Return the ids of the session's queued or running lab runs.

        A run whose process is gone (``interrupted``) is not active.
        """
        ...

    def delete(self, session_id: str) -> None:
        """Remove everything the lab stored for ``session_id``.

        Nothing stored is not an error.
        """
        ...


class Clock(Protocol):
    """The current time."""

    def now(self) -> datetime:
        """Return the current, timezone-aware time."""
        ...


TunnelState = Literal["stopped", "connecting", "connected", "reconnecting"]


class TunnelStatus(BaseModel):
    """Where the tunnel to the server is, and why it is not up.

    ``reason`` says why the last connection ended (while
    ``reconnecting``) or why a connection is not working yet (while
    ``connecting``).
    """

    model_config = ConfigDict(frozen=True)

    state: TunnelState = "stopped"
    reason: Optional[str] = None

    def describe(self) -> str:
        """Return the status as one line: ``reconnecting (why)``."""
        if self.reason and self.state in ("connecting", "reconnecting"):
            return f"{self.state} ({self.reason})"
        return self.state


class Tunnel(Protocol):
    """The connection that makes the local API reachable from outside."""

    def start(self) -> None:
        """Open the tunnel and keep it open until :meth:`stop`."""
        ...

    def stop(self) -> None:
        """Close the tunnel and stop reopening it."""
        ...

    def status(self) -> TunnelStatus:
        """Return where the tunnel is now."""
        ...


class ChatModel(Protocol):
    """A local chat model that reads candidate sentences for names.

    It runs on this machine only (the text is real people's speech);
    loading it may wait for GPU memory and raises
    :class:`~transcriber.domain.errors.ModelLoadError` when it cannot
    run. :meth:`release` frees it.
    """

    @property
    def model_name(self) -> str:
        """Return the identifier of the model that answers."""
        ...

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply to ``user`` under ``system``."""
        ...

    def release(self) -> None:
        """Free the model's memory."""
        ...


class SpeakerNameStore(Protocol):
    """Where the outcome of the last search for names is kept.

    One record per session, removed with the session.
    """

    def save_names(self, names: SpeakerNames) -> None:
        """Store ``names`` for its session, replacing the previous."""
        ...

    def names(self, session_id: str) -> Optional[SpeakerNames]:
        """Return the stored record of ``session_id``, if any."""
        ...


class GpuMemory(Protocol):
    """The GPU memory the jobs share with other processes (#117).

    The use cases ask it whether a failure was the GPU running out of
    memory, free every model of the service before trying a stage
    again, and describe the memory for a debug log or a message. It
    never stops another process.
    """

    def out_of_memory(self, error: BaseException) -> bool:
        """Tell whether ``error`` is the GPU running out of memory."""
        ...

    def free(self) -> None:
        """Unload every model of the service and give the memory back."""
        ...

    def usage(self) -> str:
        """Return this process's GPU memory in one short line (cheap)."""
        ...

    def holders(self) -> str:
        """Return who holds the GPU memory now, for a message."""
        ...


class Vad(Protocol):
    """Voice activity of one live stream, frame by frame (LV-3).

    Stateful: one instance per stream. Each frame holds
    :data:`~transcriber.domain.live.FRAME_SAMPLES` mono samples at
    16 kHz (32 ms); frames come in order.
    """

    def probability(self, frame: npt.NDArray[np.float32]) -> float:
        """Return the probability (0-1) that ``frame`` holds speech."""
        ...

    def reset(self) -> None:
        """Forget the stream (a new one starts)."""
        ...


class AsrStream(Protocol):
    """The streaming recogniser: decodes the open segment again and again.

    One warm model shared by the streams of a session. A partial decode
    may skip word times; a final one returns them (LV-4).
    """

    def warm_up(self) -> None:
        """Load the model and run it once, so the first segment is fast."""
        ...

    def decode(
        self, samples: npt.NDArray[np.float32], final: bool
    ) -> Hypothesis:
        """Return what is said in ``samples`` (mono, 16 kHz)."""
        ...


class Timer(Protocol):
    """Monotonic time and waiting, to pace a stream in real time."""

    def monotonic(self) -> float:
        """Return seconds from an arbitrary origin; never goes back."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""
        ...
