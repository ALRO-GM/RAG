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
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
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


class RecognisedSpeech(BaseModel):
    """What a speech recogniser heard: timed words and the language."""

    model_config = ConfigDict(frozen=True)

    words: List[Word]
    language: Optional[str] = None


class JobStatus(BaseModel):
    """Where the job of a session is: its state, stage and progress.

    ``fraction`` is the share of the whole job done, from 0 to 1.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    state: SessionState
    stage: Optional[str] = None
    fraction: float = Field(default=0.0, ge=0, le=1)


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
        """Append chunk ``number`` of :data:`PCM_FORMAT` audio."""
        ...

    def store_recording(
        self, session_id: str, recording: RecordedAudio
    ) -> None:
        """Store a whole recording (an imported file)."""
        ...

    def recording(self, session_id: str) -> RecordedAudio:
        """Return the stored audio of the session."""
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


Job = Callable[[Progress], None]
"""Work submitted to a :class:`JobRunner`; it reports to ``Progress``."""


class JobRunner(Protocol):
    """Run long jobs (one at a time on the one GPU) and report on them."""

    def submit(self, session_id: str, job: Job) -> None:
        """Queue ``job`` for ``session_id``."""
        ...

    def status(self, session_id: str) -> Optional[JobStatus]:
        """Return the status of the session's job, if one was submitted."""
        ...


class Clock(Protocol):
    """The current time."""

    def now(self) -> datetime:
        """Return the current, timezone-aware time."""
        ...
