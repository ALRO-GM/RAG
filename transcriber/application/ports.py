"""Ports of the transcriber's use cases.

Each port is a :class:`~typing.Protocol` an outbound adapter implements
(``docs/transcriber/ARCHITECTURE.md`` §3). The use cases depend only on
these and on the domain; the fakes in ``tests/unit/transcriber/fakes.py``
implement them in memory.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    ContextManager,
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

from transcriber.domain.models import (
    ConsentRecord,
    Session,
    SessionState,
    Transcript,
    Voiceprint,
    Word,
)
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

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Return the words spoken in ``audio`` with their times."""
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


class Clock(Protocol):
    """The current time."""

    def now(self) -> datetime:
        """Return the current, timezone-aware time."""
        ...


TunnelState = Literal["stopped", "connecting", "connected", "reconnecting"]


class TunnelStatus(BaseModel):
    """Where the tunnel to the server is, and why it is not up.

    ``reason`` says why the last connection ended (while
    ``reconnecting``) or what the last clean-up reported.
    """

    model_config = ConfigDict(frozen=True)

    state: TunnelState = "stopped"
    reason: Optional[str] = None

    def describe(self) -> str:
        """Return the status as one line: ``reconnecting (why)``."""
        if self.reason and self.state == "reconnecting":
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
