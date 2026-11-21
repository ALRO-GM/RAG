"""Live transcription: endpointing.

The live mode (``docs/platform/ARCHITECTURE.md``, stage 2) hears audio
as it arrives, 32 ms at a time. Everything here is plain arithmetic on
probabilities and times; the model is behind a port
(:class:`~transcriber.application.ports.Vad`).

:class:`Endpointer` reads the voice activity detector's speech
probability of each frame and says when speech starts and when it has
ended: after ``hang_ms`` of silence (300-700 ms), or when a segment
reaches ``max_segment_seconds`` (the recogniser reads at most that
much at once).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAMPLE_RATE = 16000
"""Sample rate of the live audio (Hz)."""

FRAME_SAMPLES = 512
"""Samples per voice activity frame: 32 ms at 16 kHz (Silero's size)."""

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

MIN_HANG_MS = 300
MAX_HANG_MS = 700
"""Range of the silence hang the latency budget allows (§6)."""


class EndpointSettings(BaseModel):
    """How the endpointer decides that speech started and ended.

    A segment starts after ``min_speech_ms`` of frames at or above
    ``threshold`` and includes ``pre_roll_ms`` of audio before them (the
    first syllable is often below the threshold). While speaking, a
    frame at or above ``release`` counts as voiced (hysteresis: a lower
    bar to stay than to start). The segment ends ``hang_ms`` after the
    last voiced frame, or is cut at ``max_segment_seconds``.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(default=0.5, gt=0, lt=1)
    release: float = Field(default=0.35, gt=0, lt=1)
    hang_ms: int = Field(default=500, ge=MIN_HANG_MS, le=MAX_HANG_MS)
    min_speech_ms: int = Field(default=96, ge=0)
    pre_roll_ms: int = Field(default=192, ge=0)
    max_segment_seconds: float = Field(default=15.0, gt=1)

    @model_validator(mode="after")
    def _release_below_threshold(self) -> "EndpointSettings":
        """Keep the release level at or below the start threshold."""
        if self.release > self.threshold:
            raise ValueError("release must not exceed threshold")
        return self


@dataclass(frozen=True)
class SpeechStart:
    """Speech started; the segment's audio begins at ``start`` seconds."""

    start: float


@dataclass(frozen=True)
class SpeechEnd:
    """A segment ended.

    ``start`` and ``end`` bound its audio (``end`` is the end of the
    last voiced frame); ``endpoint_at`` is the stream time the end was
    decided (``end`` plus the hang); ``forced`` is true when the segment
    was cut at its maximum length while speech went on.
    """

    start: float
    end: float
    endpoint_at: float
    forced: bool = False


SpeechEvent = Union[SpeechStart, SpeechEnd]


class Endpointer:
    """Speech start and end from one probability per 32 ms frame.

    Frame ``n`` covers ``[n * frame_seconds, (n + 1) * frame_seconds)``
    of the stream. Feed frames in order with :meth:`push`; call
    :meth:`flush` when the stream ends.
    """

    def __init__(
        self,
        settings: EndpointSettings = EndpointSettings(),
        frame_seconds: float = FRAME_SECONDS,
    ) -> None:
        """Start idle at stream time zero."""
        self._s = settings
        self._frame = frame_seconds
        self._frames = 0
        self._run = 0
        self._run_start = 0.0
        self._start: Optional[float] = None
        self._voiced_end = 0.0
        self._silent = 0

    @property
    def speaking(self) -> bool:
        """Tell whether a segment is open."""
        return self._start is not None

    @property
    def segment_start(self) -> Optional[float]:
        """Return where the open segment's audio begins, if one is open."""
        return self._start

    @property
    def time(self) -> float:
        """Return the stream time at the end of the last frame pushed."""
        return self._frames * self._frame

    def _frames_for(self, ms: int) -> int:
        """Return how many whole frames last at least ``ms``."""
        return max(int(-(-ms // (self._frame * 1000))), 0)

    def push(self, probability: float) -> List[SpeechEvent]:
        """Read the next frame's speech probability; return what happened."""
        begin = self.time
        self._frames += 1
        end = self.time
        if self._start is None:
            return self._idle(probability, begin, end)
        events: List[SpeechEvent] = []
        if probability >= self._s.release:
            self._voiced_end = end
            self._silent = 0
        else:
            self._silent += 1
            if self._silent >= self._frames_for(self._s.hang_ms):
                events.append(SpeechEnd(self._start, self._voiced_end, end))
                self._close()
                return events
        if end - self._start >= self._s.max_segment_seconds:
            events.append(SpeechEnd(self._start, end, end, forced=True))
            self._start = end
            self._voiced_end = end
            events.append(SpeechStart(end))
        return events

    def _idle(
        self, probability: float, begin: float, end: float
    ) -> List[SpeechEvent]:
        """Handle a frame while no segment is open."""
        if probability < self._s.threshold:
            self._run = 0
            return []
        if self._run == 0:
            self._run_start = begin
        self._run += 1
        if self._run < max(self._frames_for(self._s.min_speech_ms), 1):
            return []
        self._start = max(self._run_start - self._s.pre_roll_ms / 1000, 0.0)
        self._voiced_end = end
        self._silent = 0
        self._run = 0
        return [SpeechStart(self._start)]

    def flush(self) -> Optional[SpeechEnd]:
        """End the open segment at the end of the stream, if any."""
        if self._start is None:
            return None
        ended = SpeechEnd(self._start, self._voiced_end, self.time)
        self._close()
        return ended

    def _close(self) -> None:
        """Go back to idle."""
        self._start = None
        self._silent = 0
        self._run = 0
