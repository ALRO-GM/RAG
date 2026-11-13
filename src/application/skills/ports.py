"""Ports of the soft-skills use cases."""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.speech import SpeakerSpan
from src.domain.skills.transcript import Transcript, Word


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

    words: List[Word]
    language: Optional[str] = None


class AudioReader(Protocol):
    """Decode audio files."""

    def read(self, path: str) -> Audio:
        """Return the recording at ``path`` as mono samples."""
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


class Diarizer(Protocol):
    """Tell who spoke when."""

    def diarize(
        self,
        audio: Audio,
        regions: Sequence[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSpan]:
        """Return speaker spans over the speech ``regions`` of ``audio``.

        ``num_speakers`` is the number of people when it is known.
        """
        ...


class ChatModel(Protocol):
    """A language model answering one system + user message."""

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        ...

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply."""
        ...


class TranscriptRepository(Protocol):
    """Read diarized transcripts."""

    def read(self, path: str) -> Transcript:
        """Return the transcript stored at ``path``."""
        ...

    def find(self, session_id: str) -> Transcript:
        """Return the transcript of a session by its identifier."""
        ...

    def sessions(self) -> List[str]:
        """Return the identifiers of the stored sessions."""
        ...


class TranscriptWriter(Protocol):
    """Store diarized transcripts."""

    def save(self, transcript: Transcript, path: str = "") -> str:
        """Store ``transcript`` and return where it was written."""
        ...


class CatalogueRepository(Protocol):
    """Read rubric catalogues."""

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` written in ``language``."""
        ...

    def available(self) -> List[Tuple[str, str]]:
        """Return the (name, language) pairs that can be loaded."""
        ...
