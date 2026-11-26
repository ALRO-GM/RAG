"""Use case: live sessions fed over a socket (LV-2).

:class:`LiveSessions` starts a live session from a
:class:`~transcriber.domain.live_session.LiveSessionRequest`, takes its
audio as raw PCM messages (:meth:`LiveSession.feed`) and stops it. Each
session runs in a worker thread of its own: the thread builds the
recogniser, reads the audio as it is queued and emits the
``live.event.v1`` events of :class:`~transcriber.application.live.
LiveEngine` into the session's :class:`EventHub`, from which any number
of readers follow them.

Backpressure: at most ``queue_seconds`` of audio wait for the
recogniser. :meth:`LiveSession.feed` blocks while the queue is full, so
the socket that calls it stops reading and the sender (the browser)
sees its buffer grow; once the queue holds ``behind_seconds`` the
session emits an ``asr_behind`` warning. A connected source is never
refused for a full queue: the browser keeps the audio (#263).

Resume (#263): the session counts the samples it took
(:attr:`LiveSession.received`); a source that comes back resends from
there. One source at a time: a newer socket takes the source over and
the older one's audio is refused (:class:`SourceReplacedError`), so
nothing is taken twice. Audio the client lost becomes silence
(:meth:`LiveSession.silence`, :meth:`LiveSession.align`) so the stream
time stays on the wall clock. A source that goes away without a stop
leaves the session waiting ``source_grace_seconds`` (instead of
``idle_seconds``) with a ``source_lost`` warning, and a ``source_back``
one with the time away when it returns. Start, end (with its reason),
source lost and back are logged on :data:`LIFECYCLE` without any
content.

Every event is checked against ``live.event.v1`` (:class:`EventCheck`)
and written to the session's live log (:class:`LiveLog`, text only)
before readers see it (LV-7).

GPU (LV-8, ``docs/platform/ARCHITECTURE.md`` §7): before loading the
recogniser on the GPU the session takes the GPU lease and waits for the
image worker's ack and for ``need_mib`` of free memory
(:func:`~transcriber.domain.live_session.gpu_choice`); if the memory
does not come it runs a small model on the CPU and emits a
``gpu_degraded`` warning. The lease is held until the session ends.

At the end (LV-11) the last revision of every final becomes a
``transcript.v1`` document, kept next to the log and, if the request
asks (``handoff``), published to the RAG like a batch session, after
which the RAG's post-call analysis may be started
(:class:`~transcriber.application.live_handover.HandOver`). Since #264
this also happens when the worker dies with an exception (the summary
says ``partial``), the publishing runs after the session has ended (a
stop never waits for the RAG), the session keeps a small record
(``session.json``: request, speakers, names, model, state) from its
start, and :meth:`LiveSessions.recover` rebuilds at startup the
sessions a stopped process left without a transcript.
"""

import logging
import queue
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from transcriber.application.live import (
    FloatArray,
    LiveEngine,
    LiveOptions,
    LiveSettings,
    LiveTranscription,
    VoiceControl,
)
from transcriber.application.live_handover import (
    Handed,
    HandOver,
    SessionFacts,
)
from transcriber.application.ports import (
    Clock,
    EventCheck,
    FreeGpuMemory,
    GpuLease,
    LiveLog,
    LiveMarker,
    SessionAnalysisPort,
    Timer,
    TranscriptPublisher,
)
from transcriber.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    SourceReplacedError,
    StateError,
)
from transcriber.domain.live import (
    SAMPLE_RATE,
    EndpointSettings,
    LiveSpeaker,
    renamed,
)
from transcriber.domain.live_session import (
    AudioBacklog,
    LiveServiceSettings,
    LiveSessionRequest,
    PaceWatch,
    align_samples,
    channel_speakers,
    decode_pcm,
    gap_room,
    gap_samples,
    gpu_choice,
    source_grace,
)

LOG = logging.getLogger(__name__)
LIFECYCLE_LOGGER = "transcriber.live.lifecycle"
LIFECYCLE = logging.getLogger(LIFECYCLE_LOGGER)
"""Where a live session's start, end, source lost / back and socket
closes are logged (#263): ids, reasons, codes and seconds, never words
or audio. ``transcriber serve`` prints it."""

UTTERANCE_ID = r"^t[0-9]+\.s[0-9]+$"
"""The pattern of a live utterance id (``live.event.v1``)."""

Event = Dict[str, Any]
Builder = Callable[[LiveOptions], LiveTranscription]
Arrival = Tuple[List[FloatArray], float]
"""Audio of every channel and the monotonic time it was received."""


@dataclass(frozen=True)
class SourceNotice:
    """The audio source went away (``lost``), came back (``back``) or
    has sent everything it held (``caught_up``).

    Queued with the audio, so the worker acts in stream order;
    ``seconds`` is how long the source was away (``back`` only).
    """

    kind: Literal["lost", "back", "caught_up"]
    seconds: float = 0.0


Queued = Union[Arrival, SourceNotice, None]
"""What the worker reads: audio, a source notice, or the end (None)."""

EndReason = Literal["stop", "idle", "source_lost", "failed", "restart"]

DEGRADED_MESSAGE = (
    "The GPU did not free {need} MiB within {wait:g} s ({why}); the"
    " session runs a smaller model on the CPU: slower and less accurate."
)


class LiveState(str, Enum):
    """Where a live session is."""

    STARTING = "starting"
    RUNNING = "running"
    ENDED = "ended"
    FAILED = "failed"
    RECOVERED = "recovered"
    """Rebuilt from its log at startup after a stop of the service."""


class EventHub:
    """The events of one session, kept in order for every reader.

    Thread-safe. :meth:`read` returns the events after a sequence number
    and waits for new ones; once the hub is closed (the session ended)
    a reader that has seen everything gets an empty list and ``True``.
    """

    def __init__(self) -> None:
        """Start empty and open."""
        self._events: List[Event] = []
        self._closed = False
        self._changed = threading.Condition()

    def publish(self, event: Event) -> None:
        """Add ``event`` (its ``seq`` is its position) and wake readers."""
        with self._changed:
            self._events.append(event)
            self._changed.notify_all()

    def close(self) -> None:
        """Say no more events will come."""
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    @property
    def closed(self) -> bool:
        """Tell whether the session's events are complete."""
        return self._closed

    def history(self) -> List[Event]:
        """Return every event so far."""
        with self._changed:
            return list(self._events)

    def read(self, start: int, timeout: float) -> Tuple[List[Event], bool]:
        """Return the events from position ``start`` on, and if closed.

        Waits up to ``timeout`` seconds when there are none yet.
        """
        with self._changed:
            if len(self._events) <= start and not self._closed:
                self._changed.wait(timeout)
            return list(self._events[start:]), self._closed


class LiveSession:
    """One live session: its audio queue, its worker and its events."""

    def __init__(
        self,
        request: LiveSessionRequest,
        settings: LiveServiceSettings,
        speakers: Sequence[LiveSpeaker],
        now: Callable[[], float],
    ) -> None:
        """Set the session up; its worker starts it.

        ``now`` is the monotonic clock that stamps the audio received
        (the same as the recogniser's timings).
        """
        self.request = request
        self.settings = settings
        self.speakers = list(speakers)
        self.hub = EventHub()
        self.state = LiveState.STARTING
        self.error: Optional[str] = None
        self.device: Optional[str] = None
        self.model: Optional[str] = None
        self.diarizer: Optional[str] = None
        self.transcript: Optional[Dict[str, Any]] = None
        self.voice: Optional[VoiceControl] = None
        self.rejected = 0
        self.received = 0
        """Samples (per channel) taken so far, silence included (#263)."""
        self.gap_seconds = 0.0
        """Seconds of silence put in for audio the source lost."""
        self.silences: List[Tuple[int, int]] = []
        """Where that silence lies in the stream, ``[start, end)`` in
        samples: the parts a recovery may fill (#265). Written to the
        record as it happens, so a crash keeps them."""
        self.started_at: Optional[str] = None
        """When the session started (wall clock, ISO; #265 bounds a
        recovery's offsets with it)."""
        self.on_silence: Callable[[], None] = lambda: None
        """Told when silence was put in (the record is written again)."""
        self.end_reason: Optional[EndReason] = None
        self.compute: Optional[str] = None
        self.handed_over = threading.Event()
        """Set once the transcript is kept and published (or not asked)."""
        self.record_extra: Dict[str, Any] = {}
        """What the end added to the record (kept on later writes)."""
        self.on_change: Callable[[], None] = lambda: None
        """Told when the source connects or goes (the live marker)."""
        self._audio: "queue.Queue[Queued]" = queue.Queue()
        self._now = now
        self._backlog = AudioBacklog(
            settings.queue_seconds, settings.behind_seconds
        )
        self._room = threading.Condition()
        self._stopping = False
        self._done = threading.Event()
        self._behind = False
        self._source: Optional[int] = None
        self._sources = 0
        self._lost_at: Optional[float] = None
        self._grace = settings.source_grace_seconds
        self._origin: Optional[float] = None
        self._ready = threading.Event()

    @property
    def id(self) -> str:
        """Return the session id."""
        return self.request.session_id

    @property
    def channels(self) -> int:
        """Return the number of audio channels."""
        return self.request.channels

    @property
    def active(self) -> bool:
        """Tell whether the session still takes audio."""
        return self.state in (LiveState.STARTING, LiveState.RUNNING) \
            and not self._stopping

    def feed(self, data: bytes, source: Optional[int] = None) -> float:
        """Queue one PCM message; return the seconds of audio waiting.

        Blocks while ``queue_seconds`` of audio already wait (the
        caller, a socket, then stops reading); it is never refused for
        that (#263). ``source`` is the claim of the socket that sends
        it (:meth:`claim_source`): once a newer socket took the source
        over this raises :class:`SourceReplacedError`, so no audio is
        taken twice. Raises :class:`StateError` once the session is
        stopping or over.
        """
        return self._take(decode_pcm(data, self.channels), source)

    def silence(
        self, samples: object, source: Optional[int] = None,
        bounded: bool = True,
    ) -> int:
        """Queue ``samples`` of silence for audio the source lost (#263).

        The stream time then stays on the wall clock. Queued one second
        at a time, with the same backpressure as :meth:`feed`. Returns
        the samples queued. ``bounded``: put in at most what keeps the
        stream within :data:`GAP_SLACK_SECONDS` of the wall clock
        (:func:`gap_room`), so a faulty client cannot push hours of it;
        the caller tells the client how much was taken.
        """
        count = gap_samples(samples)
        if bounded:
            with self._room:
                elapsed = (
                    None if self._origin is None
                    else self._now() - self._origin
                )
                count = min(count, gap_room(elapsed, self.received))
        left = count
        start = self.received
        while left > 0:
            piece = min(left, SAMPLE_RATE)
            self._take(
                [np.zeros(piece, dtype=np.float32)] * self.channels, source
            )
            left -= piece
        with self._room:
            self.gap_seconds += count / SAMPLE_RATE
            if count:
                self.silences.append((start, start + count))
        if count:
            self.on_silence()
        return count

    def align(self, held: object, source: Optional[int] = None) -> int:
        """Fill the time no audio came for with silence; return samples.

        A page that reattached after a reload (#263) does not know what
        it missed: it says how many samples it ``held`` to send, and the
        silence puts those right before the present. Nothing before
        the first audio.
        """
        with self._room:
            elapsed = (
                0.0 if self._origin is None else self._now() - self._origin
            )
            missing = 0 if self._origin is None else align_samples(
                elapsed, self.received, held
            )
        return self.silence(missing, source, bounded=False) if missing else 0

    def _take(
        self, chunks: Sequence[FloatArray], source: Optional[int]
    ) -> float:
        """Queue one piece of audio of every channel (see :meth:`feed`)."""
        samples = len(chunks[0])
        with self._room:
            self._room.wait_for(
                lambda: not self.active or not self._backlog.full()
                or self._replaced(source),
            )
            if self._replaced(source):
                raise SourceReplacedError(
                    f"another socket took live session {self.id}'s audio"
                    " over"
                )
            if not self.active:
                raise StateError(f"live session {self.id} is not running")
            self._backlog.add(samples)
            self._behind = self._behind or self._backlog.warn()
            waiting = self._backlog.seconds
            if samples:
                now = self._now()
                if self._origin is None:
                    self._origin = now - samples / SAMPLE_RATE
                self.received += samples
                # Under the lock: two sockets never interleave their audio.
                self._audio.put((list(chunks), now))
        return waiting

    def _replaced(self, source: Optional[int]) -> bool:
        """Tell whether ``source`` lost the source to a newer socket."""
        return source is not None and self._source != source

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the recogniser is loaded (or the session is over).

        A socket tells its client to start sending only then, so audio
        does not queue while the model loads. Returns whether it is.
        """
        return self._ready.wait(timeout)

    def mark_ready(self) -> None:
        """Say the recogniser is loaded and takes audio."""
        self.state = LiveState.RUNNING
        self._ready.set()

    def claim_source(self) -> Optional[int]:
        """Take the session's one audio source; ``None`` once it is over.

        Returns the claim (a number) the socket passes to
        :meth:`feed`. A newer socket takes the source over from an
        older one (#263: after a dropped connection the server may not
        have noticed the old socket died yet); the older one's audio is
        then refused. A source that comes back after one was lost is
        announced to the worker (``source_back``).
        """
        with self._room:
            if not self.active:
                return None
            self._sources += 1
            replaced = self._source is not None
            self._source = self._sources
            self._room.notify_all()
            away = None
            if self._lost_at is not None:
                away = max(self._now() - self._lost_at, 0.0)
                self._lost_at = None
                self._audio.put(SourceNotice("back", away))
        self.on_change()
        LIFECYCLE.info(
            "Live session %s: audio source %d connected"
            " (resume at %.1f s%s%s)",
            self.id, self._source, self.received / SAMPLE_RATE,
            ", replacing the previous one" if replaced else "",
            "" if away is None else f", back after {away:.1f} s",
        )
        return self._source

    def release_source(
        self, source: Optional[int] = None, close_code: Optional[int] = None,
    ) -> None:
        """The socket of ``source`` closed: let another one connect.

        Without a stop this is a lost source: the session waits
        ``source_grace_seconds`` for one to come back (#263), or
        ``source_gone_seconds`` when the socket closed with
        ``close_code`` 1001 (the page left). A replaced socket's release
        changes nothing. ``None`` releases whatever source is connected.
        """
        with self._room:
            if self._source is None or (
                source is not None and source != self._source
            ):
                return
            self._source = None
            self._room.notify_all()
            lost = self.active
            if lost:
                self._lost_at = self._now()
                self._grace = source_grace(
                    close_code, self.settings.source_grace_seconds,
                    self.settings.source_gone_seconds,
                )
                self._audio.put(SourceNotice("lost"))
        self.on_change()
        if lost:
            LIFECYCLE.info(
                "Live session %s: audio source lost at %.1f s (close %s);"
                " waiting up to %g s for it", self.id,
                self.received / SAMPLE_RATE, close_code, self._grace,
            )

    def stop_when_sent(self, cap: float) -> None:
        """Stop once the connected source has sent its audio (#263).

        A stop asked over HTTP while the page still sends what it held
        during an outage waits up to ``cap`` seconds for the page's own
        ``stop`` (or for its socket to close), so that audio is not
        refused; then it stops.
        """
        with self._room:
            self._room.wait_for(
                lambda: self._source is None or self._stopping, cap
            )
        self.stop()

    def stop(self, reason: EndReason = "stop") -> None:
        """Take no more audio; the worker ends the session."""
        with self._room:
            if self._stopping:
                return
            self._stopping = True
            if self.end_reason is None:
                self.end_reason = reason
            self._room.notify_all()
        self._audio.put(None)

    def caught_up(self) -> None:
        """The source has sent everything it held (#263 review).

        Until then a returned source's audio is a resend, never a sign
        that the audio is on time again.
        """
        self._audio.put(SourceNotice("caught_up"))

    @property
    def source_state(self) -> str:
        """``connected``, ``lost`` (waiting for it) or ``none``."""
        with self._room:
            if self._source is not None:
                return "connected"
            return "lost" if self._lost_at is not None else "none"

    def idle(self) -> bool:
        """Tell whether no audio waits for the worker."""
        return self._audio.empty()

    def wait_handed_over(self, timeout: Optional[float] = None) -> bool:
        """Wait until the transcript is kept and published (or not)."""
        return self.handed_over.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the session is over; tell whether it is."""
        return self._done.wait(timeout)

    def next_audio(self) -> Queued:
        """Return the next queued audio (and when it came), or ``None``.

        ``None`` once the session is stopped; a :class:`SourceNotice`
        when the source went away or came back.

        A session that gets no audio for ``idle_seconds`` stops by
        itself (a closed tab must not hold the GPU); one whose source
        went away without a stop waits ``source_grace_seconds`` from
        that moment instead (#263).
        """
        with self._room:
            lost_at = self._lost_at
            if lost_at is None:
                timeout = self.settings.idle_seconds
            else:
                timeout = max(self._grace - (self._now() - lost_at), 0.0)
        try:
            return self._audio.get(timeout=timeout)
        except queue.Empty:
            self.stop("idle" if lost_at is None else "source_lost")
            return None

    def read_audio(self, chunks: Sequence[FloatArray]) -> bool:
        """Count ``chunks`` as read; tell whether to warn of a backlog."""
        with self._room:
            self._backlog.take(len(chunks[0]))
            self._room.notify_all()
            behind, self._behind = self._behind, False
            return behind

    def finished(self, state: LiveState, error: Optional[str]) -> None:
        """Record the end of the session and wake everybody."""
        with self._room:
            self.state = state
            self.error = error
            self._stopping = True
            if state == LiveState.FAILED:
                self.end_reason = "failed"
            elif self.end_reason is None:
                self.end_reason = "stop"
            self._room.notify_all()
        self.hub.close()
        self._ready.set()
        self._done.set()
        LIFECYCLE.info(
            "Live session %s ended: %s (%s; %.1f s of audio, %.1f s of it"
            " silence for lost audio)", self.id, self.end_reason,
            state.value, self.received / SAMPLE_RATE, self.gap_seconds,
        )

    def info(self) -> Dict[str, Any]:
        """Return what the API shows of the session."""
        events = self.hub.history()
        return {
            "session_id": self.id,
            "state": self.state.value,
            "channels": self.channels,
            "device": self.device,
            "speakers": [
                {"id": s.id, "channel": s.channel, "role": s.role,
                 "name": s.name}
                for s in self.speakers
            ],
            "product": self.request.product,
            "model": self.model,
            "diarizer": self.diarizer,
            "events": len(events),
            "finals": sum(1 for e in events if e.get("type") == "final"),
            "rejected_events": self.rejected,
            "received_samples": self.received,
            "source": self.source_state,
            "end_reason": self.end_reason,
            "transcript": self.transcript,
            "error": self.error,
        }


@dataclass(frozen=True)
class LivePorts:
    """The ports a live session uses; the optional ones may be absent.

    ``build`` makes the live transcription for given options (the
    recogniser is loaded by its warm-up). Without ``lease`` and
    ``memory`` the GPU is taken as it is; without ``log`` nothing is
    kept; without ``check`` events are not checked; without
    ``publisher`` no transcript is handed over and without ``analysis``
    no post-call analysis is started. ``cpu_only``, read when a session
    starts, keeps it on the CPU model without asking for the GPU (the
    «Ajustes» page's ``live.asr: cpu``).
    """

    build: Builder
    clock: Clock
    timer: Timer
    lease: Optional[GpuLease] = None
    memory: Optional[FreeGpuMemory] = None
    log: Optional[LiveLog] = None
    check: Optional[EventCheck] = None
    publisher: Optional[TranscriptPublisher] = None
    analysis: Optional[SessionAnalysisPort] = None
    cpu_only: Optional[Callable[[], bool]] = None
    marker: Optional[LiveMarker] = None


class LiveSessions:
    """Start, feed and stop live sessions; keep the recent ones."""

    def __init__(
        self,
        ports: LivePorts,
        settings: LiveServiceSettings = LiveServiceSettings(),
        device: str = "cuda",
        version: str = "0",
    ) -> None:
        """Keep the ports and the settings.

        ``device`` is where the recogniser runs when nothing degrades
        it (``cpu`` skips the lease); ``version`` is the transcriber's,
        written in the transcripts.
        """
        self._p = ports
        self._s = settings
        self._device = device
        self._version = version
        self._handover = HandOver(
            ports.log, ports.publisher, ports.analysis, ports.clock, version,
        )
        self._sessions: Dict[str, LiveSession] = {}
        self._recovering: Set[str] = set()
        """Recoveries whose transcription holds the GPU now (#265)."""
        self._lock = threading.Lock()
        self._marking = threading.Lock()

    def start(self, request: LiveSessionRequest) -> LiveSession:
        """Start ``request``'s session in a worker thread; return it.

        Raises :class:`StateError` if a session of that id is known and
        :class:`BusyError` when ``max_sessions`` already run.
        """
        session = LiveSession(
            request, self._s, channel_speakers(request),
            self._p.timer.monotonic,
        )
        with self._lock:
            if request.session_id in self._sessions:
                raise StateError(
                    f"live session {request.session_id} already exists"
                )
            running = [s for s in self._sessions.values() if s.active]
            if len(running) >= self._s.max_sessions:
                raise BusyError(
                    "another live session is running; stop it first"
                )
            self._forget_old()
            self._sessions[request.session_id] = session
        worker = threading.Thread(
            target=self._work, args=(session,),
            name=f"live-{request.session_id}", daemon=True,
        )
        session.started_at = self._p.clock.now().isoformat()
        session.on_silence = lambda: self._record_now(session)
        worker.start()
        session.on_change = self._mark
        self._mark()
        LIFECYCLE.info(
            "Live session %s started (%d channel(s), %s)", session.id,
            session.channels, request.product or "no product",
        )
        return session

    def get(self, session_id: str) -> LiveSession:
        """Return the session, or raise :class:`NotFoundError`."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError(f"no live session {session_id}")
        return session

    def rename(self, session_id: str, speaker: str, name: str) -> LiveSession:
        """Name a speaker of a running session (its hand-off uses it).

        The page names the voices during the session (LV-16); the name
        goes into the ``transcript.v1`` speakers at the end.
        """
        session = self.get(session_id)
        if not session.active:
            raise StateError(f"live session {session_id} has ended")
        try:
            session.speakers = renamed(session.speakers, speaker, name)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        self._handover.record(self.facts(session), session.state.value,
                              transcript=session.transcript,
                              **session.record_extra)
        return session

    def source_connected(self) -> bool:
        """Tell whether a running session has its audio socket connected.

        The tunnel's supervisor does not restart the tunnel then (#264):
        that would cut the very socket that works.
        """
        with self._lock:
            sessions = list(self._sessions.values())
        return any(s.active and s.source_state == "connected"
                   for s in sessions)

    def _mark(self) -> None:
        """Announce the sessions that are on (#264); never raises."""
        marker = self._p.marker
        if marker is None:
            return
        with self._marking:  # one snapshot written at a time, in order
            with self._lock:
                sessions = list(self._sessions.values())
                recovering = sorted(self._recovering)
            self._write_marker(marker, sessions, recovering)

    def recovering(self, session_id: str, on: bool) -> None:
        """A recovery's transcription holds the GPU (or stopped, #265).

        While it does, the session is in the live marker (state
        ``recovering``), so the services guard refuses a restart: it
        is the one moment a restart costs real work.
        """
        with self._lock:
            changed = (session_id in self._recovering) != on
            if on:
                self._recovering.add(session_id)
            else:
                self._recovering.discard(session_id)
        if changed:
            self._mark()

    @staticmethod
    def _write_marker(
        marker: LiveMarker, sessions: Sequence[LiveSession],
        recovering: Sequence[str] = (),
    ) -> None:
        """Write the snapshot (under the marking lock: newest wins)."""
        try:
            # A session that ended stays until its transcript is handed
            # over: a restart then would lose the publish (#264).
            marker.mark([
                {"id": s.id,
                 "state": s.state.value if s.active else "handing_over",
                 "source": s.source_state}
                for s in sessions
                if s.active or not s.handed_over.is_set()
            ] + [{"id": i, "state": "recovering", "source": "none"}
                 for i in recovering])
        except OSError:
            LOG.exception("The live marker was not written")

    @staticmethod
    def facts(session: LiveSession) -> SessionFacts:
        """What ``session``'s transcript and record are made of.

        The record also keeps the start time and the silence put in so
        far (#265: what a recovery may fill, also after a crash).
        """
        return SessionFacts(
            session.id, session.request, list(session.speakers),
            session.channels, session.model, session.compute,
            session.diarizer,
            extra={"started_at": session.started_at,
                   "silences": [list(s) for s in session.silences]},
        )

    def _record_now(self, session: LiveSession) -> None:
        """Write the record again (silence was put in); never raises."""
        try:
            self._handover.record(
                self.facts(session), session.state.value,
                transcript=session.transcript, **session.record_extra)
        except Exception:  # noqa: BLE001 - the audio must go on
            LOG.exception("Live session %s: record not written", session.id)

    def recover(self) -> List[str]:
        """Rebuild the sessions a stopped process left unfinished (#264).

        Each one gets its transcript from its log, kept and (on a
        thread, so the start is not delayed) published as its start
        request asked; it is listed as ``recovered``. Returns their ids.
        """
        with self._lock:
            running = list(self._sessions)
        done = self._handover.recover(skip=running)
        adopted = [(h, self._adopt(h)) for h in done]
        if adopted:
            self._mark()  # handing over: the services guard waits

            deadline = self._p.timer.monotonic() + RECOVER_BUDGET_S

            def publish_all() -> None:
                for handed, session in adopted:
                    try:
                        if self._p.timer.monotonic() >= deadline:
                            # Left for the next start: the marker (and
                            # so the services guard) is not held longer.
                            LOG.warning(
                                "Live session %s: hand-over left for the"
                                " next start (time is up)",
                                handed.session_id)
                            continue
                        self._handover.finish(handed)
                    except Exception:  # noqa: BLE001 - go on with the rest
                        LOG.exception("Live session %s: not published",
                                      handed.session_id)
                    finally:
                        session.handed_over.set()
                        self._mark()

            threading.Thread(
                target=publish_all, name="live-recover", daemon=True,
            ).start()
        return [h.session_id for h in done]

    def _adopt(self, handed: Handed) -> LiveSession:
        """List a recovered session (ended, with its summary)."""
        facts = handed.facts
        request = (facts.request if facts is not None
                   else LiveSessionRequest(session_id=handed.session_id))
        session = LiveSession(request, self._s, [], self._p.timer.monotonic)
        session.transcript = handed.summary
        session.end_reason = "restart"
        session.finished(LiveState.RECOVERED, None)
        with self._lock:
            return self._sessions.setdefault(session.id, session)

    def voice(
        self, session_id: str, action: str, utterance: Optional[str] = None,
    ) -> LiveSession:
        """Skip the voice check or correct a sentence's voice (#249).

        ``action`` is ``skip``, or ``mine`` / ``not_mine`` with the id
        of a final ``utterance``. Applied by the session's worker with
        its next audio. Raises :class:`StateError` when the session has
        ended or is not a two-party one, :class:`InvalidInputError` for
        an unknown action or a missing utterance.
        """
        session = self.get(session_id)
        if not session.active:
            raise StateError(f"live session {session_id} has ended")
        control = session.voice
        if control is None:
            raise StateError(
                f"live session {session_id} has no voice check"
            )
        if action == "skip":
            control.skip()
        elif action in ("mine", "not_mine"):
            if not utterance or not re.fullmatch(UTTERANCE_ID, utterance):
                raise InvalidInputError("a correction names an utterance")
            control.correct(utterance, action == "mine")
        else:
            raise InvalidInputError(f"unknown voice action {action!r}")
        return session

    def stop(self, session_id: str, timeout: float = 60.0) -> LiveSession:
        """Stop the session and wait (up to ``timeout``) for its end.

        A connected source first gets ``stop_drain_seconds`` to send
        what it still holds (:meth:`LiveSession.stop_when_sent`).
        """
        session = self.get(session_id)
        session.stop_when_sent(min(self._s.stop_drain_seconds, timeout))
        session.wait(timeout)
        return session

    def busy(self, session_id: str) -> bool:
        """Tell whether ``session_id`` is still on here (#265).

        On, or ended and still handing its transcript over: a recovery
        of its audio waits until then.
        """
        with self._lock:
            session = self._sessions.get(session_id)
        return session is not None and (
            session.active or not session.handed_over.is_set())

    def list(self) -> List[LiveSession]:
        """Return the known sessions, oldest first."""
        with self._lock:
            return list(self._sessions.values())

    def _forget_old(self) -> None:
        """Drop the oldest ended sessions beyond ``kept_sessions``."""
        ended = [s for s in self._sessions.values() if not s.active]
        for session in ended[:max(len(ended) - self._s.kept_sessions, 0)]:
            del self._sessions[session.id]

    @staticmethod
    def options(
        request: LiveSessionRequest, device: str, model: str,
        compute_type: str,
    ) -> LiveOptions:
        """Return the options of the recogniser for ``request``."""
        return LiveOptions(
            model=model, device=device, compute_type=compute_type,
            language=request.language, diarize=request.diarize,
            enrolled=request.enrolled, parties=request.parties,
            settings=LiveSettings(
                endpoint=EndpointSettings(hang_ms=request.hang_ms),
                partial_every_ms=request.partial_every_ms,
                realtime=True,
            ),
        )

    def _emitter(self, session: LiveSession) -> Callable[[Event], None]:
        """Return the emit of ``session``: check, log, then publish."""
        def emit(event: Event) -> None:
            check = self._p.check
            problem = check.problem(event) if check is not None else None
            if problem is not None:
                session.rejected += 1
                LOG.error(
                    "Live event %s of %s breaks live.event.v1: %s",
                    event.get("seq"), session.id, problem,
                )
                return
            if self._p.log is not None:
                try:
                    self._p.log.append(session.id, event)
                except OSError:
                    LOG.exception("The live log of %s failed", session.id)
            session.hub.publish(event)

        return emit

    def _need(self, session: LiveSession) -> int:
        """Return the GPU memory ``session`` asks for (MiB)."""
        return self._s.need_mib + (
            self._s.diarize_need_mib if session.request.diarize else 0
        )

    def _place(self, session: LiveSession) -> Tuple[str, Optional[str]]:
        """Take the GPU (lease, ack, memory) or choose the CPU.

        Returns the device and, when degraded, why.
        """
        if self._device != "cuda":
            return self._device, None
        if self._p.cpu_only is not None and self._p.cpu_only():
            return "cpu", None
        lease, memory = self._p.lease, self._p.memory
        need = self._need(session)
        if lease is not None:
            try:
                lease.acquire(need)
            except StateError as exc:
                return "cpu", str(exc)
        if memory is None:
            return "cuda", None
        waited = 0.0
        while True:
            free = memory.free_mib()
            acked = lease.freed() if lease is not None else False
            choice = gpu_choice(
                waited, acked, free, need, self._s.ack_grace,
                self._s.ack_timeout,
            )
            if choice == "gpu":
                return "cuda", None
            if choice == "cpu":
                why = "no GPU found" if free is None else (
                    f"{free} MiB free"
                    + ("" if acked else ", no ack from the image worker")
                )
                return "cpu", why
            self._p.timer.sleep(self._s.poll_seconds)
            waited += self._s.poll_seconds

    def _work(self, session: LiveSession) -> None:
        """Run ``session`` to its end (the worker thread).

        Whatever ends it (the end of its audio, a stop, an exception),
        what was heard is handed over: kept at once, published once the
        session has ended (#264).
        """
        emit = self._emitter(session)
        state, error = LiveState.ENDED, None
        engine: Optional[LiveEngine] = None
        self._handover.record(self.facts(session), "starting")
        try:
            device, degraded = self._place(session)
            if device == "cuda":
                model, compute = self._s.model, self._s.compute_type
            else:
                model, compute = self._s.cpu_model, self._s.cpu_compute_type
            session.device, session.model = device, model
            session.compute = compute
            if degraded is not None and self._p.lease is not None:
                self._p.lease.release()
            live = self._p.build(
                self.options(session.request, device, model, compute)
            )
            engine = live.engine(
                session.id, session.speakers, emit,
                diarize=session.request.diarize,
            )
            session.speakers = list(engine.speakers)
            session.diarizer = engine.diarizer
            session.voice = engine.voice
            engine.start()
            if degraded is not None:
                emit(engine.events.warning(
                    self._p.clock.now(), 0.0, "gpu_degraded",
                    DEGRADED_MESSAGE.format(
                        need=self._need(session), wait=self._s.ack_timeout,
                        why=degraded,
                    ),
                ))
            session.mark_ready()
            self._mark()
            self._handover.record(self.facts(session), "running")
            pace = PaceWatch()
            resending = False
            while True:
                arrival = session.next_audio()
                if arrival is None:
                    break
                if isinstance(arrival, SourceNotice):
                    if arrival.kind == "caught_up":
                        resending = False
                        continue
                    emit(source_warning(engine, self._p.clock, arrival))
                    # Resent audio comes in a burst: no partials until
                    # it is read (#263); its finals still come. Its
                    # arrival times say nothing of when it was spoken.
                    resending = arrival.kind == "back"
                    engine.skip_partials = resending
                    pace.reset()
                    continue
                chunks, arrived = arrival
                engine.push(chunks, arrived)
                idle = session.idle()
                if idle:
                    engine.skip_partials = False
                on_pace = pace.observe(len(chunks[0]), arrived, idle)
                if on_pace and not resending and engine.realign(arrived):
                    LIFECYCLE.info(
                        "Live session %s: audio on time again at %.1f s;"
                        " timings re-anchored", session.id,
                        engine.stream_time,
                    )
                if session.read_audio(chunks):
                    emit(engine.events.warning(
                        self._p.clock.now(), engine.stream_time,
                        "asr_behind", "The recogniser runs behind the audio.",
                    ))
            engine.finish()
        except (InvalidInputError, StateError) as exc:
            state, error = LiveState.FAILED, str(exc)
            self._say_failed(session, engine, emit, error)
        except Exception as exc:  # the worker must always end the session
            LOG.exception("Live session %s failed", session.id)
            state, error = LiveState.FAILED, type(exc).__name__
            self._say_failed(session, engine, emit, error)
        finally:
            if engine is not None:
                engine.abort()
                self._keep_diagnosis(session, engine)
            if self._p.lease is not None:
                self._p.lease.release()
            if self._p.log is not None:
                self._p.log.close(session.id)
        handed = self._keep(session, failed=state == LiveState.FAILED)
        session.finished(state, error)
        self._mark()  # ended, still handing over
        session.record_extra = {"end_reason": session.end_reason,
                                "error": error,
                                "ended_at": self._p.clock.now().isoformat(),
                                # #265: what a recovery may fill.
                                "received_samples": session.received}
        self._handover.record(
            self.facts(session), state.value, transcript=session.transcript,
            **session.record_extra,
        )
        try:
            if handed is not None:
                # The record is written again with the publish's result;
                # until then the session stays in the live marker (#264).
                handed.state = state.value
                handed.extra = dict(session.record_extra)
                self._handover.finish(handed)
            elif self._p.log is not None:
                self._p.log.mark_handed(session.id)  # nothing to hand
        except Exception:  # noqa: BLE001 - logged; the session has ended
            LOG.exception("Live session %s: hand-over failed", session.id)
        finally:
            session.handed_over.set()
            self._mark()

    def _say_failed(
        self, session: LiveSession, engine: Optional[LiveEngine],
        emit: Callable[[Event], None], error: str,
    ) -> None:
        """Tell the event readers the session failed (no words, #264)."""
        if engine is None or not session.hub.history():
            return
        try:
            emit(engine.events.warning(
                self._p.clock.now(), engine.stream_time, "other",
                FAILED_MESSAGE.format(error=error),
            ))
        except Exception:  # noqa: BLE001 - the hand-over matters more
            LOG.exception("Live session %s: the failure was not told",
                          session.id)

    def _keep(self, session: LiveSession, failed: bool) -> Optional[Handed]:
        """Keep the transcript of what was heard (also after a failure).

        Nothing when no event was emitted (the recogniser never
        started). Never raises: a failure here is logged.
        """
        events = session.hub.history()
        if not events:
            return None
        try:
            handed = self._handover.keep(
                self.facts(session), events,
                mark="partial" if failed else None,
            )
        except Exception:  # noqa: BLE001 - the session must still end
            LOG.exception("Live session %s: no transcript", session.id)
            return None
        session.transcript = handed.summary
        return handed

    def _keep_diagnosis(
        self, session: LiveSession, engine: LiveEngine
    ) -> None:
        """Keep the attribution's numbers of a two-party session (#249).

        Ids, times, slots, cosines and decisions only: no text, no
        audio. Written next to the events, so a bad session can be
        explained afterwards.
        """
        records = engine.diagnosis()
        if not records:
            return
        checks = [r for r in records if r.get("event") == "check"]
        LOG.info(
            "Live session %s attribution: %d records, check %s",
            session.id, len(records),
            checks[-1].get("state") if checks else "none",
        )
        if self._p.log is None:
            return
        try:
            self._p.log.save_attribution(session.id, records)
        except OSError:
            LOG.exception("The attribution of %s was not kept", session.id)


RECOVER_BUDGET_S = 300.0
"""Longest time the hand-overs found at a start may take (and hold the
live marker); the rest are tried at the next start (#264)."""
FAILED_MESSAGE = (
    "The live session stopped on an error ({error}); the transcript of"
    " what was heard until then is kept."
)
SOURCE_LOST_MESSAGE = (
    "The audio source went away; the session waits for it to come back."
)
SOURCE_BACK_MESSAGE = (
    "The audio source is back after {seconds:.0f} s; its audio arrives"
    " late and is read as fast as possible."
)


def source_warning(
    engine: LiveEngine, clock: Clock, notice: SourceNotice
) -> Event:
    """Return the ``source_lost`` / ``source_back`` warning (#263)."""
    if notice.kind == "lost":
        return engine.events.warning(
            clock.now(), engine.stream_time, "source_lost",
            SOURCE_LOST_MESSAGE,
        )
    return engine.events.warning(
        clock.now(), engine.stream_time, "source_back",
        SOURCE_BACK_MESSAGE.format(seconds=notice.seconds), notice.seconds,
    )
