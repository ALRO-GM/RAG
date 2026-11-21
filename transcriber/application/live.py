"""Use case: transcribe live audio into ``live.event.v1`` events (LV-3, LV-4).

:class:`LiveTranscription` reads one or two channels 32 ms at a time.
Each channel is a stream with its own voice activity detector,
:class:`~transcriber.domain.live.Endpointer` and
:class:`~transcriber.domain.live.LocalAgreement`; the recogniser is
shared (one warm model). While a segment is open it is decoded again
every ``partial_every_ms`` and the agreed words go out as ``partial``
events; at its endpoint the segment is decoded once more, with word
times, and its sentences go out as ``final`` events with their
``timings``.

In real time (``realtime=True``) a frame is read only once its audio
would have arrived, so a decode that takes long delays the frames
behind it exactly as it would with a microphone, and the timings are
wall-clock times: ``endpoint_ms`` from the end of speech to the
endpoint decision, ``asr_ms`` the final decode and ``lag_ms`` from the
end of speech to the event. Without real time the audio is read as
fast as possible and ``endpoint_ms`` is the hang in stream time.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from transcriber.application.ports import AsrStream, Clock, Timer, Vad
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    EndpointSettings,
    Endpointer,
    Hypothesis,
    LiveEventStream,
    LiveSpeaker,
    LiveUtterance,
    LocalAgreement,
    SentenceAssembler,
    SourceKind,
    SpeechEnd,
    SpeechStart,
    Timings,
    UtteranceEvent,
)

FloatArray = npt.NDArray[np.float32]
Emit = Callable[[Dict[str, object]], None]

MIN_PARTIAL_SECONDS = 0.5
"""A segment is not decoded for a partial before it holds this much."""


class LiveSettings(BaseModel):
    """How the live transcription runs."""

    model_config = ConfigDict(frozen=True)

    endpoint: EndpointSettings = EndpointSettings()
    partial_every_ms: int = Field(default=800, ge=100)
    realtime: bool = True


class LiveOptions(BaseModel):
    """What the live transcription is built with (models and settings).

    ``model``, ``device`` and ``compute_type`` choose the streaming
    recogniser (faster-whisper names); ``language`` ``None`` lets it
    detect the language of each segment.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8"
    language: Optional[str] = "es"
    settings: LiveSettings = LiveSettings()


@dataclass
class _Stream:
    """One channel being transcribed."""

    channel: int
    speaker: str
    samples: FloatArray
    vad: Vad
    endpointer: Endpointer
    agreement: LocalAgreement = field(default_factory=LocalAgreement)
    last_partial: float = 0.0


@dataclass
class LiveSummary:
    """What a run produced: the counts and each final's timings."""

    partials: int = 0
    finals: int = 0
    endpoint_ms: List[float] = field(default_factory=list)
    asr_ms: List[float] = field(default_factory=list)
    lag_ms: List[float] = field(default_factory=list)


class LiveTranscription:
    """Channels of audio in, ``live.event.v1`` events out."""

    def __init__(
        self,
        vad_factory: Callable[[], Vad],
        asr: AsrStream,
        clock: Clock,
        timer: Timer,
        settings: LiveSettings = LiveSettings(),
    ) -> None:
        """Keep the ports: a new VAD per stream, one shared recogniser."""
        self._vad_factory = vad_factory
        self._asr = asr
        self._clock = clock
        self._timer = timer
        self._s = settings

    def run(
        self, session_id: str, channels: Sequence[FloatArray], emit: Emit
    ) -> LiveSummary:
        """Transcribe ``channels`` (16 kHz mono each); emit every event.

        One channel is a single stream (``single_speaker``); two are
        two speakers, one per channel (``stereo_channels``), speaker
        ``S<channel + 1>``.
        """
        if not 1 <= len(channels) <= 2:
            raise InvalidInputError("live audio has one or two channels")
        self._asr.warm_up()
        streams = [
            _Stream(
                channel=n, speaker=f"S{n + 1}", samples=samples,
                vad=self._vad_factory(),
                endpointer=Endpointer(self._s.endpoint),
            )
            for n, samples in enumerate(channels)
        ]
        kind: SourceKind = (
            "stereo_channels" if len(streams) == 2 else "single_speaker"
        )
        stereo = len(streams) == 2
        events = LiveEventStream(session_id)
        emit(events.session_started(
            self._clock.now(), kind, len(streams),
            [LiveSpeaker(s.speaker, s.channel if stereo else None)
             for s in streams],
        ))
        run = _Run(
            self._asr, self._clock, self._timer, self._s, events, emit, stereo
        )
        frames = max(len(s.samples) for s in streams) // FRAME_SAMPLES
        for frame in range(frames):
            arrived = run.wait_for((frame + 1) * FRAME_SAMPLES / SAMPLE_RATE)
            for stream in streams:
                run.step(stream, frame, arrived)
        end = frames * FRAME_SAMPLES / SAMPLE_RATE
        for stream in streams:
            ended = stream.endpointer.flush()
            if ended is not None:
                run.final(stream, ended, run.wait_for(end))
        emit(events.session_ended(self._clock.now(), end))
        return run.summary


class _Run:
    """The state of one :meth:`LiveTranscription.run`."""

    def __init__(
        self, asr: AsrStream, clock: Clock, timer: Timer,
        settings: LiveSettings, events: LiveEventStream, emit: Emit,
        stereo: bool,
    ) -> None:
        """Start the run's clock."""
        self._asr = asr
        self._clock = clock
        self._timer = timer
        self._s = settings
        self._events = events
        self._emit = emit
        self._assembler = SentenceAssembler()
        self._stereo = stereo
        self._origin = timer.monotonic()
        self.summary = LiveSummary()

    def wall(self, stream_time: float) -> float:
        """Return when stream time ``stream_time`` arrives (monotonic)."""
        return self._origin + stream_time

    def wait_for(self, stream_time: float) -> float:
        """In real time, wait until ``stream_time`` has arrived.

        Returns the monotonic time the audio up to ``stream_time`` is
        available (now, when running behind or not in real time).
        """
        if self._s.realtime:
            self._timer.sleep(self.wall(stream_time) - self._timer.monotonic())
        return self._timer.monotonic()

    def step(self, stream: _Stream, frame: int, arrived: float) -> None:
        """Read ``frame`` of ``stream``; emit what it ends or updates."""
        first = frame * FRAME_SAMPLES
        chunk = stream.samples[first:first + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        probability = stream.vad.probability(chunk.astype(np.float32))
        for event in stream.endpointer.push(probability):
            if isinstance(event, SpeechEnd):
                self.final(stream, event, arrived)
            elif isinstance(event, SpeechStart):
                stream.agreement.reset()
                stream.last_partial = event.start
        self._maybe_partial(stream)

    def _audio(self, stream: _Stream, start: float, end: float) -> FloatArray:
        """Return ``stream``'s samples between two stream times."""
        first = int(round(start * SAMPLE_RATE))
        last = int(round(end * SAMPLE_RATE))
        return stream.samples[first:last]

    def _channel(self, stream: _Stream) -> Optional[int]:
        """Return the channel to report (stereo only)."""
        return stream.channel if self._stereo else None

    def _maybe_partial(self, stream: _Stream) -> None:
        """Decode the open segment for a partial when it is time."""
        start = stream.endpointer.segment_start
        now = stream.endpointer.time
        every = self._s.partial_every_ms / 1000
        if start is None or now - start < MIN_PARTIAL_SECONDS \
                or now - stream.last_partial < every:
            return
        stream.last_partial = now
        timer = self._timer
        began = timer.monotonic()
        heard = self._asr.decode(self._audio(stream, start, now), False)
        done = timer.monotonic()
        agreed = " ".join(stream.agreement.update(heard.text))
        hypothesis = Hypothesis(
            text=agreed, language=heard.language,
            confidence=heard.confidence,
        )
        utterances = self._assembler.partial(
            stream.speaker, self._channel(stream), start, now, hypothesis
        )
        timings = Timings(
            asr_ms=(done - began) * 1000,
            lag_ms=max(done - self.wall(now), 0.0) * 1000
            if self._s.realtime else None,
        )
        self._send("partial", utterances, now, timings)

    def final(self, stream: _Stream, ended: SpeechEnd, decided: float) -> None:
        """Decode a finished segment and emit its sentences."""
        timer = self._timer
        began = timer.monotonic()
        heard = self._asr.decode(
            self._audio(stream, ended.start, ended.end), True
        )
        done = timer.monotonic()
        stream.agreement.reset()
        picked = timer.monotonic()
        utterances = self._assembler.final(
            stream.speaker, self._channel(stream), ended.start, ended.end,
            heard,
        )
        attributed = timer.monotonic()
        asr_ms = (done - began) * 1000
        if self._s.realtime:
            spoken = self.wall(ended.end)
            endpoint_ms = max(decided - spoken, 0.0) * 1000
            lag_ms = max(attributed - spoken, 0.0) * 1000
        else:
            endpoint_ms = (ended.endpoint_at - ended.end) * 1000
            lag_ms = endpoint_ms + asr_ms
        timings = Timings(
            endpoint_ms=endpoint_ms, asr_ms=asr_ms,
            attribution_ms=(attributed - picked) * 1000, lag_ms=lag_ms,
        )
        if utterances:
            self.summary.endpoint_ms.append(endpoint_ms)
            self.summary.asr_ms.append(asr_ms)
            self.summary.lag_ms.append(lag_ms)
        self._send("final", utterances, ended.endpoint_at, timings)

    def _send(
        self, kind: UtteranceEvent, utterances: List[LiveUtterance],
        stream_time: float, timings: Timings,
    ) -> None:
        """Emit one event per utterance and count them."""
        for utterance in utterances:
            self._emit(self._events.utterance(
                kind, self._clock.now(), stream_time, utterance, timings
            ))
            if kind == "final":
                self.summary.finals += 1
            else:
                self.summary.partials += 1
