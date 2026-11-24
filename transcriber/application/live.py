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

Mono diarization (LV-6): an engine made with a
:class:`~transcriber.application.live_diarization.MonoAttributor`
hears several voices on its one channel (``mono_diarized``). Each
frame of audio also goes to the streaming diarizer; each final
sentence gets the speaker its frames say (``attribution_ms`` includes
the short wait for them) and, when a later frame or a voiceprint
changes it, a ``speaker_update`` event with the same id and a higher
revision.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from transcriber.application.live_diarization import MonoAttributor
from transcriber.application.ports import AsrStream, Clock, Timer, Vad
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live_session import Parties
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


class VoiceControl(Protocol):
    """What the page may ask of a two-party session's voices (#249).

    Thread-safe: the requests come from the API while the session's
    worker reads the audio; they take effect with its next frame.
    """

    def skip(self) -> None:
        """Skip the employee's voice check."""
        ...

    def correct(self, utterance: str, mine: bool) -> None:
        """Say the voice of final ``utterance`` is (not) the employee's."""
        ...


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
    detect the language of each segment. ``diarize`` adds the
    streaming diarizer of mono sessions (on ``device`` too) and
    ``enrolled`` limits the voiceprints that may name its voices
    (``None``: every enrolled person). ``parties`` makes the diarized
    session a two-party one with the employee's voice check (#249).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8"
    language: Optional[str] = "es"
    settings: LiveSettings = LiveSettings()
    diarize: bool = False
    enrolled: Optional[List[str]] = None
    parties: Optional[Parties] = None


KEEP_IDLE_SECONDS = 1.0
"""Audio kept before the current frame while no segment is open.

It covers the endpointer's pre-roll and its minimum speech run; older
audio of a pushed stream is dropped (:meth:`_Stream.trim`).
"""


@dataclass
class _Stream:
    """One channel being transcribed.

    ``samples`` holds the channel's audio from sample ``offset`` of the
    stream on: older audio no segment can need any more is dropped.
    """

    channel: int
    speaker: str
    samples: FloatArray
    vad: Vad
    endpointer: Endpointer
    agreement: LocalAgreement = field(default_factory=LocalAgreement)
    last_partial: float = 0.0
    offset: int = 0

    def available(self) -> int:
        """Return the number of samples received so far."""
        return self.offset + len(self.samples)

    def append(self, chunk: FloatArray) -> None:
        """Add ``chunk`` to the end of the audio."""
        self.samples = np.concatenate(
            [self.samples, chunk.astype(np.float32)]
        )

    def trim(self) -> None:
        """Drop the audio before what an open or next segment may need."""
        start = self.endpointer.segment_start
        keep = start if start is not None else (
            self.endpointer.time - KEEP_IDLE_SECONDS
        )
        drop = int(keep * SAMPLE_RATE) - self.offset
        if drop > 0:
            self.samples = self.samples[drop:]
            self.offset += drop


@dataclass
class LiveSummary:
    """What a run produced: the counts and each final's timings."""

    partials: int = 0
    finals: int = 0
    endpoint_ms: List[float] = field(default_factory=list)
    asr_ms: List[float] = field(default_factory=list)
    lag_ms: List[float] = field(default_factory=list)
    speaker_updates: int = 0
    attribution_ms: List[float] = field(default_factory=list)


class LiveTranscription:
    """Channels of audio in, ``live.event.v1`` events out."""

    def __init__(
        self,
        vad_factory: Callable[[], Vad],
        asr: AsrStream,
        clock: Clock,
        timer: Timer,
        settings: LiveSettings = LiveSettings(),
        attributor: Optional[Callable[
            [Sequence[LiveSpeaker]], MonoAttributor
        ]] = None,
    ) -> None:
        """Keep the ports: a new VAD per stream, one shared recogniser.

        ``attributor`` makes the speaker attribution of a mono session
        that asks for diarization (from the ids and roles given), when
        the transcription has one.
        """
        self._vad_factory = vad_factory
        self._asr = asr
        self._clock = clock
        self._timer = timer
        self._s = settings
        self._attributor = attributor

    @property
    def diarizes(self) -> bool:
        """Tell whether mono sessions can be diarized."""
        return self._attributor is not None

    def run(
        self, session_id: str, channels: Sequence[FloatArray], emit: Emit,
        diarize: bool = False,
    ) -> LiveSummary:
        """Transcribe ``channels`` (16 kHz mono each); emit every event.

        One channel is a single stream (``single_speaker``), or several
        voices told apart by the diarizer with ``diarize``
        (``mono_diarized``); two are two speakers, one per channel
        (``stereo_channels``), speaker ``S<channel + 1>``. With
        ``realtime`` the audio is read at the pace it would arrive
        from a microphone.
        """
        if not 1 <= len(channels) <= 2:
            raise InvalidInputError("live audio has one or two channels")
        engine = self.engine(
            session_id, default_speakers(len(channels)), emit,
            paced=self._s.realtime, diarize=diarize,
        )
        engine.start()
        longest = max(len(samples) for samples in channels)
        engine.push([
            np.pad(samples.astype(np.float32), (0, longest - len(samples)))
            for samples in channels
        ])
        return engine.finish()

    def engine(
        self, session_id: str, speakers: Sequence[LiveSpeaker], emit: Emit,
        paced: bool = False, diarize: bool = False,
    ) -> "LiveEngine":
        """Return a stream of ``speakers`` (one per channel) to push to.

        ``speakers[n]`` speaks on channel ``n``. ``paced`` waits for
        each frame's time before reading it (a file standing in for a
        microphone); a live source is not paced, its audio arrives in
        time by itself. ``diarize`` on one channel tells its voices
        apart; ``speakers`` then name the voices in the order they are
        first heard. Raises :class:`InvalidInputError` for diarization
        of two channels or without a diarizer.
        """
        mono: Optional[MonoAttributor] = None
        if diarize:
            if any(s.channel not in (None, 0) for s in speakers):
                raise InvalidInputError(
                    "only a one-channel session is diarized"
                )
            if self._attributor is None:
                raise InvalidInputError("no streaming diarizer is installed")
            mono = self._attributor(speakers)
            # The one stream is a key for its segments; the speakers of
            # its sentences come from the attribution.
            speakers = [LiveSpeaker(mono.speakers.label(0))]
        return LiveEngine(
            self._vad_factory, self._asr, self._clock, self._timer,
            self._s, session_id, speakers, emit, paced, mono,
        )


def default_speakers(channels: int) -> List[LiveSpeaker]:
    """Return speaker ``S<n + 1>`` on channel ``n`` (role unknown).

    The channel is named only with two channels (one channel is one
    stream, not a channel of a pair).
    """
    return [
        LiveSpeaker(f"S{n + 1}", n if channels == 2 else None)
        for n in range(channels)
    ]


class LiveEngine:
    """One live session's streams: audio pushed in, events out.

    :meth:`start` warms the recogniser and emits ``session_started``;
    :meth:`push` reads every whole 32 ms frame the channels hold;
    :meth:`finish` closes the open segments and emits
    ``session_ended``. Not thread-safe: one caller pushes.
    """

    def __init__(
        self,
        vad_factory: Callable[[], Vad],
        asr: AsrStream,
        clock: Clock,
        timer: Timer,
        settings: LiveSettings,
        session_id: str,
        speakers: Sequence[LiveSpeaker],
        emit: Emit,
        paced: bool,
        mono: Optional[MonoAttributor] = None,
    ) -> None:
        """Make one stream per speaker; nothing is emitted yet."""
        if not 1 <= len(speakers) <= 2:
            raise InvalidInputError("live audio has one or two channels")
        self._mono = mono
        self._asr = asr
        self._clock = clock
        self._timer = timer
        self._s = settings
        self._emit = emit
        self._paced = paced
        self.speakers = list(speakers)
        if mono is not None:
            self.speakers = mono.roster()
        self.streams = [
            _Stream(
                channel=n, speaker=speaker.id,
                samples=np.zeros(0, dtype=np.float32),
                vad=vad_factory(), endpointer=Endpointer(settings.endpoint),
            )
            for n, speaker in enumerate(speakers)
        ]
        self.events = LiveEventStream(session_id)
        self._run: Optional[_Run] = None
        self._frames = 0

    @property
    def kind(self) -> SourceKind:
        """Return the source kind of ``session_started``."""
        if self._mono is not None:
            return "mono_diarized"
        return (
            "stereo_channels" if len(self.streams) == 2 else "single_speaker"
        )

    @property
    def diarizer(self) -> Optional[str]:
        """Return the diarization model of a mono diarized session."""
        return self._mono.model_name if self._mono is not None else None

    @property
    def voice(self) -> Optional[VoiceControl]:
        """Return the voice requests of a two-party session (#249)."""
        mono = self._mono
        return mono if mono is not None and mono.voice_check else None

    def diagnosis(self) -> List[Dict[str, Any]]:
        """Return the attribution's records (two-party sessions, #249)."""
        return self._mono.diagnosis() if self._mono is not None else []

    @property
    def stream_time(self) -> float:
        """Return the stream time of the frames read so far."""
        return self._frames * FRAME_SAMPLES / SAMPLE_RATE

    @property
    def summary(self) -> LiveSummary:
        """Return what the session produced so far."""
        return self._run.summary if self._run is not None else LiveSummary()

    def start(self) -> None:
        """Warm the recogniser and emit ``session_started``.

        The diarizer, if any, starts first: it loads while the
        recogniser warms up.
        """
        if self._mono is not None:
            self._mono.start()
        try:
            self._asr.warm_up()
        except Exception:
            self.abort()
            raise
        self._emit(self.events.session_started(
            self._clock.now(), self.kind, len(self.streams), self.speakers,
        ))
        self._run = _Run(
            self._asr, self._clock, self._timer, self._s, self.events,
            self._emit, len(self.streams) == 2, self._paced, self._mono,
        )
        self._run.check_attribution()
        self._run.voice_news(0.0)

    def _started(self) -> "_Run":
        """Return the run, or refuse audio before :meth:`start`."""
        if self._run is None:
            raise InvalidInputError("the live session has not started")
        return self._run

    def push(
        self, chunks: Sequence[FloatArray], arrived: Optional[float] = None
    ) -> int:
        """Add one chunk per channel; read the whole frames they complete.

        Returns the number of frames read. Unpaced, the first audio
        pushed sets the origin of the wall-clock timings: its last
        sample arrived at ``arrived`` (monotonic; now if not given), so
        audio that queued while the model loaded counts as late.
        """
        run = self._started()
        if len(chunks) != len(self.streams):
            raise InvalidInputError(
                f"expected {len(self.streams)} channel(s), got {len(chunks)}"
            )
        if not self._paced and self._frames == 0 \
                and self.streams[0].available() == 0 and len(chunks[0]):
            run.anchor(len(chunks[0]) / SAMPLE_RATE, arrived)
        for stream, chunk in zip(self.streams, chunks):
            stream.append(chunk)
        read = 0
        while all(
            s.available() >= (self._frames + 1) * FRAME_SAMPLES
            for s in self.streams
        ):
            arrived = run.wait_for(
                (self._frames + 1) * FRAME_SAMPLES / SAMPLE_RATE
            )
            for stream in self.streams:
                run.step(stream, self._frames, arrived)
                stream.trim()
            self._frames += 1
            read += 1
        return read

    def finish(self) -> LiveSummary:
        """Close the open segments, emit ``session_ended``; return the
        summary."""
        run = self._started()
        end = self.stream_time
        for stream in self.streams:
            ended = stream.endpointer.flush()
            if ended is not None:
                run.final(stream, ended, run.wait_for(end))
        run.finish_attribution(end)
        self._emit(self.events.session_ended(self._clock.now(), end))
        return run.summary

    def abort(self) -> None:
        """Free what the session holds when it fails before its end."""
        if self._mono is not None:
            self._mono.close()


class _Run:
    """The state of one :meth:`LiveTranscription.run`."""

    def __init__(
        self, asr: AsrStream, clock: Clock, timer: Timer,
        settings: LiveSettings, events: LiveEventStream, emit: Emit,
        stereo: bool, paced: bool = True,
        mono: Optional[MonoAttributor] = None,
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
        self._paced = paced
        self._mono = mono
        self._warned = False
        self._origin = timer.monotonic()
        self.summary = LiveSummary()

    def check_attribution(self, stream_time: float = 0.0) -> None:
        """Warn once when the speaker attribution has stopped."""
        mono = self._mono
        if mono is None or mono.problem is None or self._warned:
            return
        self._warned = True
        self._emit(self._events.warning(
            self._clock.now(), stream_time, "other",
            "Speaker attribution stopped; sentences keep the last"
            f" speaker known ({mono.problem}).",
        ))

    def _updates(
        self, utterances: List[LiveUtterance], stream_time: float
    ) -> None:
        """Emit ``speaker_update`` events for revised finals."""
        for utterance in utterances:
            self._emit(self._events.utterance(
                "speaker_update", self._clock.now(), stream_time,
                utterance, Timings(),
            ))
            self.summary.speaker_updates += 1

    def finish_attribution(self, stream_time: float) -> None:
        """Read the diarizer's last frames and emit their revisions."""
        if self._mono is None:
            return
        self._updates(self._mono.finish(), stream_time)
        self.check_attribution(stream_time)
        self.voice_news(stream_time)

    def voice_news(self, stream_time: float) -> None:
        """Emit the voice check's news as ``voice_check`` events (#249)."""
        mono = self._mono
        if mono is None or mono.employee_id is None:
            return
        for state, seconds, needed in mono.notices():
            self._emit(self._events.voice_check(
                self._clock.now(), stream_time, state, mono.employee_id,
                seconds, needed,
            ))

    def anchor(self, seconds: float, at: Optional[float] = None) -> None:
        """Set the origin so that stream time ``seconds`` is ``at``.

        ``at`` is a monotonic time (now when ``None``).
        """
        now = self._timer.monotonic() if at is None else at
        self._origin = now - seconds

    def wall(self, stream_time: float) -> float:
        """Return when stream time ``stream_time`` arrives (monotonic)."""
        return self._origin + stream_time

    def wait_for(self, stream_time: float) -> float:
        """When paced, wait until ``stream_time`` has arrived.

        Returns the monotonic time the audio up to ``stream_time`` is
        available (now, when running behind or not paced).
        """
        if self._paced:
            self._timer.sleep(self.wall(stream_time) - self._timer.monotonic())
        return self._timer.monotonic()

    def step(self, stream: _Stream, frame: int, arrived: float) -> None:
        """Read ``frame`` of ``stream``; emit what it ends or updates."""
        first = frame * FRAME_SAMPLES - stream.offset
        chunk = stream.samples[first:first + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        probability = stream.vad.probability(chunk.astype(np.float32))
        if self._mono is not None:
            self._mono.feed(chunk.astype(np.float32))
        for event in stream.endpointer.push(probability):
            if isinstance(event, SpeechEnd):
                self.final(stream, event, arrived)
            elif isinstance(event, SpeechStart):
                stream.agreement.reset()
                stream.last_partial = event.start
        self._maybe_partial(stream)
        if self._mono is not None:
            now = stream.endpointer.time
            self._updates(self._mono.poll(), now)
            self.check_attribution(now)
            self.voice_news(now)

    def _audio(self, stream: _Stream, start: float, end: float) -> FloatArray:
        """Return ``stream``'s samples between two stream times."""
        first = max(int(round(start * SAMPLE_RATE)) - stream.offset, 0)
        last = max(int(round(end * SAMPLE_RATE)) - stream.offset, 0)
        return stream.samples[first:last]

    def _channel(self, stream: _Stream) -> Optional[int]:
        """Return the channel to report (stereo only)."""
        return stream.channel if self._stereo else None

    def _speaker(self, stream: _Stream, start: float, end: float) -> str:
        """Return who likely speaks in ``start``-``end`` of ``stream``."""
        if self._mono is None:
            return stream.speaker
        return self._mono.guess(start, end)

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
            self._speaker(stream, start, now), self._channel(stream), start,
            now, hypothesis, key=stream.speaker,
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
            self._speaker(stream, ended.start, ended.end),
            self._channel(stream), ended.start, ended.end, heard,
            key=stream.speaker,
        )
        if self._mono is not None:
            utterances = self._mono.attribute(utterances)
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
            self.summary.attribution_ms.append((attributed - picked) * 1000)
        self._send("final", utterances, ended.endpoint_at, timings)
        if self._mono is not None:
            self.voice_news(ended.endpoint_at)

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
