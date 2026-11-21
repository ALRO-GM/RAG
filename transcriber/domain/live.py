"""Live transcription: endpointing, partial agreement, sentence assembly.

The live mode (``docs/platform/ARCHITECTURE.md``, stage 2) hears audio
as it arrives, 32 ms at a time, and turns it into ``live.event.v1``
events. Everything here is plain arithmetic on probabilities, words and
times; the models are behind ports (:mod:`transcriber.application.ports`).

* :class:`Endpointer` reads the voice activity detector's speech
  probability of each frame and says when speech starts and when it has
  ended: after ``hang_ms`` of silence (300-700 ms), or when a segment
  reaches ``max_segment_seconds`` (the recogniser reads at most that
  much at once).
* :class:`LocalAgreement` keeps the words two consecutive hypotheses of
  the same growing segment agree on (LocalAgreement-2, as
  ``whisper_streaming``): partial text that does not flicker.
* :class:`SentenceAssembler` splits a segment's text into sentences and
  gives each the id ``t<turn>.s<index>`` of ``transcript.v1``. A partial
  and the final it becomes share the id, and ``revision`` grows with
  each update.
* :class:`LiveEventStream` numbers the events and writes them in the
  ``live.event.v1`` shape (``contracts/live.event.v1.schema.json``).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcriber.domain.models import Word
from transcriber.domain.sentences import split_sentences, utterance_id

SCHEMA = "live.event.v1"

SAMPLE_RATE = 16000
"""Sample rate of the live audio (Hz)."""

FRAME_SAMPLES = 512
"""Samples per voice activity frame: 32 ms at 16 kHz (Silero's size)."""

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

MIN_HANG_MS = 300
MAX_HANG_MS = 700
"""Range of the silence hang the latency budget allows (§6)."""

SourceKind = Literal["stereo_channels", "mono_diarized", "single_speaker"]
UtteranceEvent = Literal["partial", "final"]


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


_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def agreement_key(token: str) -> str:
    """Return ``token`` as compared by :class:`LocalAgreement`.

    Lower case, Unicode NFKC, punctuation dropped: "Hola," and "hola"
    agree (Whisper often moves a comma between two decodings).
    """
    plain = unicodedata.normalize("NFKC", token).lower()
    return _PUNCTUATION.sub("", plain)


class LocalAgreement:
    """LocalAgreement-2: the prefix two consecutive hypotheses share.

    Each hypothesis is the whole open segment decoded again (it grows
    as audio arrives). The committed words are the longest common
    prefix of the last two hypotheses; they only grow within a segment
    (a later, shorter agreement does not take words back).
    """

    def __init__(self) -> None:
        """Start with nothing heard."""
        self._previous: List[str] = []
        self._committed: List[str] = []

    @property
    def committed(self) -> List[str]:
        """Return the committed tokens."""
        return list(self._committed)

    def update(self, text: str) -> List[str]:
        """Read the next hypothesis; return the committed tokens."""
        tokens = text.split()
        shared = 0
        for old, new in zip(self._previous, tokens):
            if agreement_key(old) != agreement_key(new):
                break
            shared += 1
        if shared > len(self._committed):
            self._committed = tokens[:shared]
        self._previous = tokens
        return self.committed

    def reset(self) -> None:
        """Forget the segment (it ended)."""
        self._previous = []
        self._committed = []


class Hypothesis(BaseModel):
    """What the streaming recogniser heard in one segment.

    ``words`` carry times relative to the start of the audio decoded;
    they may be empty (partials are decoded without word times), and
    the text then stands alone. ``confidence`` is the mean word
    probability, when known.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    words: List[Word] = Field(default_factory=list)
    language: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class LiveUtterance(BaseModel):
    """One sentence of a live event (``live.event.v1`` ``utterance``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    channel: Optional[int] = Field(default=None, ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    revision: int = Field(ge=0)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    language: Optional[str] = None

    def to_contract(self) -> Dict[str, Any]:
        """Return the ``utterance`` object of a live event."""
        data = self.model_dump()
        data.update(
            start=round(self.start, 3), end=round(self.end, 3),
            speaker_certainty=None, speaker_provisional=False,
        )
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 4)
        return data


def sentence_spans(
    text: str, words: Sequence[Word], start: float, end: float
) -> List[Tuple[str, float, float]]:
    """Return the sentences of ``text`` with their absolute times.

    ``words`` (times relative to ``start``) give each sentence the time
    of its first and last word when they cover the text's whitespace
    tokens one to one; otherwise every sentence gets the whole span
    ``start``-``end``. Times stay inside the span.
    """
    sentences = split_sentences(text)
    tokens = [token for word in words for token in word.text.split()]
    owners = [n for n, word in enumerate(words) for _ in word.text.split()]
    aligned = tokens == " ".join(sentences).split()
    spans: List[Tuple[str, float, float]] = []
    position = 0
    for sentence in sentences:
        count = len(sentence.split())
        if aligned and count:
            first = words[owners[position]]
            last = words[owners[position + count - 1]]
            a = min(max(start + first.start, start), end)
            b = min(max(start + last.end, a), end)
            spans.append((sentence, a, b))
        else:
            spans.append((sentence, start, end))
        position += count
    return spans


@dataclass
class _Segment:
    """A speaker's open segment: its turn and its first sentence index."""

    turn: int
    base: int
    emitted: List[str] = field(default_factory=list)


class SentenceAssembler:
    """Gives the sentences of each segment their ids and revisions.

    The turn number grows whenever a segment opens for a speaker other
    than the one of the latest segment (the ``transcript.v1`` rule);
    a speaker's next segment in the same turn continues its sentence
    numbering. A segment opens at its first partial (or its final, when
    it had none), so the ids of a partial and its final match.
    """

    def __init__(self) -> None:
        """Start before the first turn."""
        self._turn = -1
        self._speaker: Optional[str] = None
        self._next_index = 0
        self._open: Dict[str, _Segment] = {}
        self._revisions: Dict[str, int] = {}

    def _segment(self, speaker: str) -> _Segment:
        """Return ``speaker``'s open segment, opening one if needed."""
        segment = self._open.get(speaker)
        if segment is not None:
            return segment
        if speaker != self._speaker:
            self._turn += 1
            self._speaker = speaker
            self._next_index = 0
        segment = _Segment(self._turn, self._next_index)
        self._open[speaker] = segment
        return segment

    def _revision(self, ident: str) -> int:
        """Return the next revision of ``ident`` and remember it."""
        revision = self._revisions.get(ident, -1) + 1
        self._revisions[ident] = revision
        return revision

    def _utterances(
        self, segment: _Segment, speaker: str, channel: Optional[int],
        spans: Sequence[Tuple[str, float, float]],
        hypothesis: Hypothesis, only_changed: bool,
    ) -> List[LiveUtterance]:
        """Build the utterances of ``spans`` (only changed ones if asked)."""
        out: List[LiveUtterance] = []
        for n, (text, start, end) in enumerate(spans):
            if only_changed and n < len(segment.emitted) \
                    and segment.emitted[n] == text:
                continue
            index = segment.base + n
            ident = utterance_id(segment.turn, index)
            out.append(LiveUtterance(
                id=ident, turn=segment.turn, index=index, speaker=speaker,
                channel=channel, start=start, end=end, text=text,
                revision=self._revision(ident),
                confidence=hypothesis.confidence,
                language=hypothesis.language,
            ))
        segment.emitted = [text for text, _, _ in spans]
        return out

    def partial(
        self, speaker: str, channel: Optional[int], start: float,
        end: float, hypothesis: Hypothesis,
    ) -> List[LiveUtterance]:
        """Return the partial sentences that changed since the last call."""
        spans = sentence_spans(hypothesis.text, [], start, end)
        if not spans:
            return []
        segment = self._segment(speaker)
        return self._utterances(
            segment, speaker, channel, spans, hypothesis, only_changed=True
        )

    def final(
        self, speaker: str, channel: Optional[int], start: float,
        end: float, hypothesis: Hypothesis,
    ) -> List[LiveUtterance]:
        """Return the final sentences of the segment and close it."""
        spans = sentence_spans(hypothesis.text, hypothesis.words, start, end)
        if not spans and speaker not in self._open:
            return []
        segment = self._segment(speaker)
        # Ids already shown by a partial are never given to another
        # sentence, even when the final has fewer sentences.
        used = max(len(spans), len(segment.emitted))
        out = self._utterances(
            segment, speaker, channel, spans, hypothesis, only_changed=False
        )
        del self._open[speaker]
        if segment.turn == self._turn:
            self._next_index = max(self._next_index, segment.base + used)
        return out


class Timings(BaseModel):
    """Where the time of one event went, in milliseconds (§6).

    ``endpoint_ms``: from the end of speech to the endpoint decision;
    ``asr_ms``: decoding; ``attribution_ms``: choosing the speaker;
    ``lag_ms``: from the end of speech (for a partial: the end of the
    audio decoded) to the event's emission.
    """

    model_config = ConfigDict(frozen=True)

    endpoint_ms: Optional[float] = Field(default=None, ge=0)
    asr_ms: Optional[float] = Field(default=None, ge=0)
    attribution_ms: Optional[float] = Field(default=None, ge=0)
    lag_ms: Optional[float] = Field(default=None, ge=0)

    def to_contract(self) -> Dict[str, float]:
        """Return the ``timings`` object (rounded to 0.1 ms, no nulls)."""
        return {
            key: round(value, 1)
            for key, value in self.model_dump().items() if value is not None
        }


@dataclass(frozen=True)
class LiveSpeaker:
    """A voice of the session: its id, channel and role."""

    id: str
    channel: Optional[int] = None
    role: str = "unknown"
    name: Optional[str] = None


def iso(at: datetime) -> str:
    """Return ``at`` as an ISO 8601 time in UTC with milliseconds."""
    text = at.isoformat(timespec="milliseconds")
    return text.replace("+00:00", "Z")


class LiveEventStream:
    """Numbers the events of one session and writes them as contracts."""

    def __init__(self, session_id: str) -> None:
        """Start at sequence number zero."""
        self.session_id = session_id
        self._seq = 0

    def _event(
        self, kind: str, at: datetime, stream_time: float
    ) -> Dict[str, Any]:
        """Return the common fields of the next event."""
        event: Dict[str, Any] = {
            "schema": SCHEMA, "session_id": self.session_id,
            "seq": self._seq, "type": kind, "at": iso(at),
            "stream_time": round(max(stream_time, 0.0), 3),
        }
        self._seq += 1
        return event

    def session_started(
        self, at: datetime, kind: SourceKind, channels: int,
        speakers: Sequence[LiveSpeaker],
    ) -> Dict[str, Any]:
        """Return the ``session_started`` event."""
        event = self._event("session_started", at, 0.0)
        event["source"] = {
            "kind": kind, "sample_rate": SAMPLE_RATE, "channels": channels,
        }
        event["speakers"] = [
            {"id": s.id, "name": s.name, "channel": s.channel,
             "enrolled": False, "role": s.role}
            for s in speakers
        ]
        return event

    def utterance(
        self, kind: UtteranceEvent, at: datetime, stream_time: float,
        utterance: LiveUtterance, timings: Timings,
    ) -> Dict[str, Any]:
        """Return a ``partial`` or ``final`` event."""
        event = self._event(kind, at, stream_time)
        event["utterance"] = utterance.to_contract()
        event["timings"] = timings.to_contract()
        return event

    def session_ended(
        self, at: datetime, stream_time: float
    ) -> Dict[str, Any]:
        """Return the ``session_ended`` event."""
        return self._event("session_ended", at, stream_time)


def percentile(values: Sequence[float], share: float) -> Optional[float]:
    """Return the ``share`` (0-1) percentile, linear interpolation.

    ``None`` when there are no values.
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(share, 0.0), 1.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
