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
session emits an ``asr_behind`` warning.

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
which the RAG's post-call analysis may be started.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from transcriber.application.live import (
    FloatArray,
    LiveOptions,
    LiveSettings,
    LiveTranscription,
)
from transcriber.application.ports import (
    Clock,
    EventCheck,
    FreeGpuMemory,
    GpuLease,
    LiveLog,
    SessionAnalysisPort,
    Timer,
    TranscriptPublisher,
)
from transcriber.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    StateError,
    TranscriberError,
)
from transcriber.domain.live import EndpointSettings, LiveSpeaker
from transcriber.domain.live_session import (
    AudioBacklog,
    LiveServiceSettings,
    LiveSessionRequest,
    channel_speakers,
    decode_pcm,
    gpu_choice,
)
from transcriber.domain.live_transcript import live_transcript
from transcriber.domain.models import Pipeline

LOG = logging.getLogger(__name__)

Event = Dict[str, Any]
Builder = Callable[[LiveOptions], LiveTranscription]
Arrival = Tuple[List[FloatArray], float]
"""Audio of every channel and the monotonic time it was received."""

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
        self.transcript: Optional[Dict[str, Any]] = None
        self.rejected = 0
        self._audio: "queue.Queue[Optional[Arrival]]" = queue.Queue()
        self._now = now
        self._backlog = AudioBacklog(
            settings.queue_seconds, settings.behind_seconds
        )
        self._room = threading.Condition()
        self._stopping = False
        self._done = threading.Event()
        self._behind = False
        self._source = False

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

    def feed(self, data: bytes) -> float:
        """Queue one PCM message; return the seconds of audio waiting.

        Blocks while ``queue_seconds`` of audio already wait (the
        caller, a socket, then stops reading). Raises
        :class:`StateError` once the session is stopping or over and
        :class:`BusyError` when the queue stays full for
        ``feed_timeout`` seconds.
        """
        chunks = decode_pcm(data, self.channels)
        samples = len(chunks[0])
        with self._room:
            if not self._room.wait_for(
                lambda: not self.active or not self._backlog.full(),
                self.settings.feed_timeout,
            ):
                raise BusyError(
                    f"live session {self.id} is behind: its audio queue"
                    " stayed full"
                )
            if not self.active:
                raise StateError(f"live session {self.id} is not running")
            self._backlog.add(samples)
            self._behind = self._behind or self._backlog.warn()
            waiting = self._backlog.seconds
        if samples:
            self._audio.put((chunks, self._now()))
        return waiting

    def claim_source(self) -> bool:
        """Take the session's one audio source; False if taken or over."""
        with self._room:
            if self._source or not self.active:
                return False
            self._source = True
            return True

    def release_source(self) -> None:
        """Let another source connect (the socket closed)."""
        with self._room:
            self._source = False

    def stop(self) -> None:
        """Take no more audio; the worker ends the session."""
        with self._room:
            if self._stopping:
                return
            self._stopping = True
            self._room.notify_all()
        self._audio.put(None)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the session is over; tell whether it is."""
        return self._done.wait(timeout)

    def next_audio(self) -> Optional[Arrival]:
        """Return the next queued audio and when it came, or ``None``.

        ``None`` once the session is stopped.

        A session that gets no audio for ``idle_seconds`` stops by
        itself (a closed tab must not hold the GPU).
        """
        try:
            return self._audio.get(timeout=self.settings.idle_seconds)
        except queue.Empty:
            self.stop()
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
            self._room.notify_all()
        self.hub.close()
        self._done.set()

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
            "events": len(events),
            "finals": sum(1 for e in events if e.get("type") == "final"),
            "rejected_events": self.rejected,
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
    no post-call analysis is started.
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
        self._sessions: Dict[str, LiveSession] = {}
        self._lock = threading.Lock()

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
        worker.start()
        return session

    def get(self, session_id: str) -> LiveSession:
        """Return the session, or raise :class:`NotFoundError`."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError(f"no live session {session_id}")
        return session

    def stop(self, session_id: str, timeout: float = 60.0) -> LiveSession:
        """Stop the session and wait (up to ``timeout``) for its end."""
        session = self.get(session_id)
        session.stop()
        session.wait(timeout)
        return session

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
            language=request.language,
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

    def _place(self, session: LiveSession) -> Tuple[str, Optional[str]]:
        """Take the GPU (lease, ack, memory) or choose the CPU.

        Returns the device and, when degraded, why.
        """
        if self._device != "cuda":
            return self._device, None
        lease, memory = self._p.lease, self._p.memory
        need = self._s.need_mib
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
        """Run ``session`` to its end (the worker thread)."""
        emit = self._emitter(session)
        state, error = LiveState.ENDED, None
        try:
            device, degraded = self._place(session)
            if device == "cuda":
                model, compute = self._s.model, self._s.compute_type
            else:
                model, compute = self._s.cpu_model, self._s.cpu_compute_type
            session.device, session.model = device, model
            if degraded is not None and self._p.lease is not None:
                self._p.lease.release()
            live = self._p.build(
                self.options(session.request, device, model, compute)
            )
            engine = live.engine(session.id, session.speakers, emit)
            engine.start()
            if degraded is not None:
                emit(engine.events.warning(
                    self._p.clock.now(), 0.0, "gpu_degraded",
                    DEGRADED_MESSAGE.format(
                        need=self._s.need_mib, wait=self._s.ack_timeout,
                        why=degraded,
                    ),
                ))
            session.state = LiveState.RUNNING
            while True:
                arrival = session.next_audio()
                if arrival is None:
                    break
                chunks, arrived = arrival
                engine.push(chunks, arrived)
                if session.read_audio(chunks):
                    emit(engine.events.warning(
                        self._p.clock.now(), engine.stream_time,
                        "asr_behind", "The recogniser runs behind the audio.",
                    ))
            engine.finish()
            self._hand_over(session, model, compute)
        except (InvalidInputError, StateError) as exc:
            state, error = LiveState.FAILED, str(exc)
        except Exception as exc:  # the worker must always end the session
            LOG.exception("Live session %s failed", session.id)
            state, error = LiveState.FAILED, type(exc).__name__
        finally:
            if self._p.lease is not None:
                self._p.lease.release()
            if self._p.log is not None:
                self._p.log.close(session.id)
        session.finished(state, error)

    def _hand_over(
        self, session: LiveSession, model: str, compute: str
    ) -> None:
        """Assemble the transcript; keep it, publish it, analyse it (LV-11).

        A failure to publish or to start the analysis is recorded in
        the session's ``transcript`` summary, not raised: the live part
        went well.
        """
        events = session.hub.history()
        started = events[0]["at"] if events else None
        transcript = live_transcript(
            session.id, events, session.speakers,
            Pipeline(
                recogniser=f"faster-whisper {model} {compute}",
                voice_model="none", transcriber=self._version,
                diarizer="channels" if session.channels == 2 else None,
            ),
            language=session.request.language,
            started_at=_parse_time(started),
        )
        document = transcript.to_contract()
        summary: Dict[str, Any] = {
            "utterances": len(transcript.utterances),
            "published": False, "analysis": None, "error": None,
        }
        session.transcript = summary
        if self._p.log is not None:
            try:
                self._p.log.save_transcript(session.id, document)
            except OSError:
                LOG.exception("The transcript of %s was not kept", session.id)
        handoff = session.request.handoff
        if not handoff.transcript_v1 or not transcript.utterances:
            return
        if self._p.publisher is None:
            summary["error"] = "no RAG to publish to"
            return
        try:
            self._p.publisher.publish(document)
            summary["published"] = True
            if handoff.run_post_call:
                if self._p.analysis is None:
                    summary["error"] = "no RAG analysis configured"
                    return
                self._p.analysis.start(session.id, False, False)
                summary["analysis"] = "started"
        except TranscriberError as exc:
            summary["error"] = str(exc)


def _parse_time(text: Optional[str]) -> Optional[datetime]:
    """Return the ISO time ``text`` of an event as a datetime, if any."""
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
