"""Use case: run live-assist sessions for the live page.

A session is either **live** (the transcriber records it: this use case
starts it there with the product's speakers and roles, then follows its
``live.event.v1`` stream) or a **replay** of a recorded
``transcript.v1`` (a demo without a microphone). Both feed the same
:class:`~orchestrator.application.live_assist.LiveAssist` and the same
:class:`~orchestrator.application.feed.Feed`, so the page cannot tell
them apart except by ``mode``.

The feed carries, in order: ``{"type": "session"}`` (what the page
needs to draw: product, roles, banner, whether signals are shown, the
script phases), ``{"type": "event", "event": ...}`` for every event,
``{"type": "suggestion", "record": ...}`` for every reasoner call (with
its per-stage latency) and ``{"type": "status"}`` when the session
ends or fails. One session runs at a time (the transcriber runs one
live session at a time as well); ended ones are kept, newest first, up
to ``keep``.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from orchestrator.application.feed import Feed
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.ports import (
    CancelToken,
    Clock,
    Document,
    EventSource,
    Executor,
    KnowledgePort,
    LiveTranscriber,
    ReasonerPort,
    Runner,
    SuggestionSink,
    TranscriberUnavailable,
    TranscriptStore,
)
from orchestrator.application.replay import (
    Replay,
    default_roles,
    transcript_to_events,
)
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import SuggestionRecord

AUDIO_KINDS = ("stereo", "mic")
"""``stereo``: one channel per person (agent/interviewer on 0, the other
on 1, as the product says); ``mic``: the browser's microphone, one
channel heard as the person the product suggests about."""
MIN_SPEED, MAX_SPEED = 0.25, 20.0
SESSION_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$"


class SessionError(Exception):
    """A request the sessions cannot serve; ``code`` says which."""

    def __init__(self, code: str, message: str):
        """Keep the stable code and the message."""
        super().__init__(message)
        self.code = code


@dataclass
class _Session:
    """One running or ended session (internal)."""

    id: str
    product: ProductConfig
    mode: str
    roles: Dict[str, str]
    feed: Feed
    stop_token: CancelToken
    audio: Optional[str] = None
    transcript: Optional[str] = None
    state: str = "running"
    error: Optional[str] = None
    events: int = 0
    rejected: int = 0
    assist: Optional[LiveAssist] = None
    sinks: List[SuggestionSink] = field(default_factory=list)
    by_status: Dict[str, int] = field(default_factory=dict)

    def view(self) -> Document:
        """Return the session as JSON."""
        assist = self.assist
        return {
            "id": self.id, "product": self.product.product,
            "mode": self.mode, "audio": self.audio,
            "transcript": self.transcript, "state": self.state,
            "error": self.error, "events": self.events,
            "rejected_events": self.rejected,
            "calls": dict(self.by_status),
            "skipped": dict(assist.skips) if assist else {},
            "messages": self.feed.last, "page": page_info(
                self.product, self.roles
            ),
        }


class _Handler:
    """Sends a session's events to the feed and to the use case."""

    def __init__(self, session: _Session, assist: LiveAssist):
        """Keep the session."""
        self.session = session
        self.assist = assist

    def on_event(self, event: LiveEvent) -> None:
        """Show the event, then let the use case decide on it."""
        self.session.events += 1
        self.session.feed.publish(
            {"type": "event", "event": event.to_contract()}
        )
        self.assist.on_event(event)

    def finish(self) -> None:
        """Wait for the call in flight."""
        self.assist.finish()


class _FeedSink:
    """A suggestion sink that shows each record and logs it."""

    def __init__(self, session: _Session, log: Optional[SuggestionSink]):
        """Keep where the records go."""
        self.session = session
        self.log = log

    def write(self, record: SuggestionRecord) -> None:
        """Publish ``record`` and pass it to the log."""
        counts = self.session.by_status
        counts[record.status] = counts.get(record.status, 0) + 1
        self.session.feed.publish({
            "type": "suggestion", "record": record.model_dump(mode="json"),
        })
        if self.log is not None:
            self.log.write(record)


def page_info(config: ProductConfig, roles: Mapping[str, str]) -> Document:
    """What the live page needs to know about a product."""
    reasoner = config.reasoner
    trigger = config.trigger
    return {
        "product": config.product, "language": config.language,
        "banner": config.privacy.banner,
        "signals": bool(reasoner and reasoner.expressed_signals),
        "script_phases": list(config.ui.script_phases) if config.ui else [],
        "roles": dict(roles),
        "suggest_about": list(trigger.speaker_roles) if trigger else [],
        "channels": {
            str(c): {"speaker": ch.speaker, "role": ch.role}
            for c, ch in sorted(config.source.channels.items())
        },
    }


class LiveSessions:
    """Start, follow, stop and list live-assist sessions.

    Args:
        products: Live product configs by name.
        knowledge: The knowledge port.
        reasoner: The reasoner port.
        transcripts: Recorded conversations for replays.
        clock: Monotonic time (and waiting, for replays).
        runner: Runs a session in the background.
        executor: Makes the executor of one session's calls.
        transcriber: The transcriber's live sessions (``None``: replay
            only).
        events: The transcriber's event stream (``None``: replay only).
        log: Makes the log sink of a session (``session id``,
            ``product``), or ``None`` for no log.
        new_id: Makes an id for a live session started without one.
        keep: Ended sessions kept.
    """

    def __init__(
        self,
        products: Mapping[str, ProductConfig],
        knowledge: KnowledgePort,
        reasoner: ReasonerPort,
        transcripts: TranscriptStore,
        clock: Clock,
        runner: Runner,
        executor: Callable[[], Executor],
        transcriber: Optional[LiveTranscriber] = None,
        events: Optional[EventSource] = None,
        log: Optional[Callable[[str, str], SuggestionSink]] = None,
        new_id: Callable[[], str] = lambda: "live",
        keep: int = 20,
    ):
        """Wire the ports; no session yet."""
        self.products = {
            name: config for name, config in products.items()
            if config.is_live and config.ui is not None
            and config.ui.kind == "live_page"
        }
        self.knowledge = knowledge
        self.reasoner = reasoner
        self.transcripts = transcripts
        self.clock = clock
        self.runner = runner
        self.executor = executor
        self.transcriber = transcriber
        self.events = events
        self.log = log
        self.new_id = new_id
        self.keep = keep
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ---- queries -------------------------------------------------------

    def list_products(self) -> List[Document]:
        """The products the page can start, with what it shows."""
        return [
            {**page_info(c, self._channel_roles(c)),
             "live": self.transcriber is not None}
            for _, c in sorted(self.products.items())
        ]

    def list_transcripts(self) -> List[Document]:
        """Conversations a replay can use."""
        return self.transcripts.list()

    def list_sessions(self) -> List[Document]:
        """Every session kept, newest first."""
        with self._lock:
            return [s.view() for s in reversed(self._sessions.values())]

    def get(self, session_id: str) -> Document:
        """One session as JSON."""
        return self._find(session_id).view()

    def feed(self, session_id: str) -> Feed:
        """The feed of one session."""
        return self._find(session_id).feed

    # ---- commands ------------------------------------------------------

    def start_replay(
        self,
        session_id: str,
        product: str,
        transcript_id: str,
        speed: Optional[float] = 1.0,
    ) -> Document:
        """Replay a recorded conversation as if it were live."""
        config = self._product(product)
        _check_id(session_id)
        if speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
            raise SessionError(
                "invalid_request",
                f"speed must be between {MIN_SPEED} and {MAX_SPEED}",
            )
        try:
            transcript = self.transcripts.load(transcript_id)
        except KeyError as exc:
            raise SessionError(
                "not_found", f"no transcript {transcript_id!r}"
            ) from exc
        roles = default_roles(config, transcript)
        session = self._open(session_id, config, "replay", roles)
        session.transcript = transcript_id
        assist = self._assist(session, {
            "type": transcript.session.type, "goal": transcript.session.goal,
        })
        handler = _Handler(session, assist)
        events = list(transcript_to_events(transcript, roles))
        for event in events:
            # The replay's events carry the session id the page asked for.
            event.session_id = session_id

        def job() -> None:
            try:
                Replay(handler, self.clock, speed, session.stop_token).run(
                    iter(events)
                )
            finally:
                self._end(session)

        self.runner.start(job)
        return session.view()

    def start_live(
        self, product: str, audio: str, session_id: Optional[str] = None
    ) -> Document:
        """Start a live session on the transcriber and follow it."""
        config = self._product(product)
        if self.transcriber is None or self.events is None:
            raise SessionError(
                "unavailable", "no transcriber is linked for live sessions"
            )
        if audio not in AUDIO_KINDS:
            raise SessionError(
                "invalid_request", f"audio must be one of {AUDIO_KINDS}"
            )
        session_id = session_id or self.new_id()
        _check_id(session_id)
        speakers = live_speakers(config, audio)
        roles = {s["id"]: s["role"] for s in speakers}
        session = self._open(session_id, config, "live", roles)
        session.audio = audio
        try:
            started = self.transcriber.start(
                transcriber_body(config, session_id, speakers)
            )
        except TranscriberUnavailable as exc:
            session.error = str(exc)
            self._end(session, "failed")
            raise SessionError("transcriber", str(exc)) from exc
        assist = self._assist(session, {})
        session.feed.publish({"type": "transcriber", "session": started})
        handler = _Handler(session, assist)
        source = self.events

        def job() -> None:
            failed: Optional[str] = None
            try:
                for document in source.follow(session_id,
                                              session.stop_token):
                    try:
                        event = LiveEvent.model_validate(document)
                    except ValueError:
                        session.rejected += 1
                        continue
                    handler.on_event(event)
                    if event.type == "session_ended":
                        break
            except TranscriberUnavailable as exc:
                failed = str(exc)
            finally:
                handler.finish()
                session.error = failed
                self._end(session, "failed" if failed else "ended")

        self.runner.start(job)
        return session.view()

    def stop(self, session_id: str) -> Document:
        """Stop a session (a live one on the transcriber too)."""
        session = self._find(session_id)
        if session.state != "running":
            return session.view()
        if session.mode == "live" and self.transcriber is not None:
            try:
                self.transcriber.stop(session_id)
            except TranscriberUnavailable:
                pass
        session.stop_token.cancel()
        return session.view()

    # ---- internals -----------------------------------------------------

    def _product(self, name: str) -> ProductConfig:
        config = self.products.get(name)
        if config is None:
            raise SessionError("not_found", f"no live product {name!r}")
        return config

    def _find(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("not_found", f"no session {session_id!r}")
        return session

    @staticmethod
    def _channel_roles(config: ProductConfig) -> Dict[str, str]:
        return {ch.speaker: ch.role for ch in config.source.channels.values()}

    def _open(
        self,
        session_id: str,
        config: ProductConfig,
        mode: str,
        roles: Dict[str, str],
    ) -> _Session:
        """Register a new running session (one at a time)."""
        with self._lock:
            if session_id in self._sessions:
                raise SessionError(
                    "wrong_state", f"session {session_id!r} exists"
                )
            if any(s.state == "running" for s in self._sessions.values()):
                raise SessionError("busy", "a session is already running")
            session = _Session(
                id=session_id, product=config, mode=mode, roles=roles,
                feed=Feed(), stop_token=CancelToken(),
            )
            self._sessions[session_id] = session
            ended = [k for k, s in self._sessions.items()
                     if s.state != "running"]
            for old in ended[:max(0, len(ended) - self.keep)]:
                del self._sessions[old]
        session.feed.publish({"type": "session", "session": session.view()})
        return session

    def _assist(
        self, session: _Session, about: Dict[str, Optional[str]]
    ) -> LiveAssist:
        """Build the use case of one session."""
        log = (self.log(session.id, session.product.product)
               if self.log is not None else None)
        assist = LiveAssist(
            session.product, self.knowledge, self.reasoner,
            _FeedSink(session, log), self.clock, self.executor(),
            session.roles, about,
        )
        session.assist = assist
        return assist

    def _end(self, session: _Session, state: str = "ended") -> None:
        """Mark the session ended and close its feed."""
        with self._lock:
            session.state = state
        session.feed.publish({"type": "status", "session": session.view()})
        session.feed.close()


def _check_id(session_id: str) -> None:
    """Refuse an id the contract's ``session_id`` would not take."""
    if not re.fullmatch(SESSION_ID, session_id or ""):
        raise SessionError("invalid_request", "invalid session id")


def live_speakers(config: ProductConfig, audio: str) -> List[Document]:
    """The transcriber's speakers for ``audio``.

    ``stereo``: every channel of the product, as declared. ``mic``: one
    channel, heard as the first speaker whose role the product suggests
    about (the customer, the candidate), so a person talking at the
    microphone drives the suggestions. ``mic`` with a product whose
    attribution is ``sortformer_stream`` (LV-6): every declared speaker,
    all on channel 0, in channel order; the transcriber tells the
    voices of the mix apart and gives them these ids in the order they
    are first heard.
    """
    channels = sorted(config.source.channels.items())
    if audio == "stereo":
        return [{"channel": c, "id": ch.speaker, "role": ch.role}
                for c, ch in channels]
    if diarized(config) and channels:
        return [{"channel": 0, "id": ch.speaker, "role": ch.role}
                for _, ch in channels]
    wanted = config.trigger.speaker_roles if config.trigger else []
    chosen = next((ch for _, ch in channels if ch.role in wanted),
                  channels[-1][1] if channels else None)
    if chosen is None:
        return [{"channel": 0, "id": "S1", "role": "unknown"}]
    return [{"channel": 0, "id": chosen.speaker, "role": chosen.role}]


def diarized(config: ProductConfig) -> bool:
    """Tell whether the product asks for mono diarization (LV-6)."""
    return config.attribution.kind == "sortformer_stream"


def transcriber_body(
    config: ProductConfig, session_id: str, speakers: List[Document]
) -> Document:
    """The body of the transcriber's ``POST /api/live/sessions``.

    Speakers all on channel 0 under a ``sortformer_stream`` product
    make a diarized one-channel session.
    """
    channels = len({s["channel"] for s in speakers}) or 1
    body: Dict[str, Any] = {
        "session_id": session_id, "channels": channels,
        "speakers": speakers, "product": config.product,
        "language": config.language,
        "handoff": config.handoff.model_dump(),
    }
    if diarized(config) and channels == 1:
        body["diarize"] = True
    if config.vad is not None:
        body["hang_ms"] = config.vad.hang_ms
    if config.asr.partial_every_ms:
        body["partial_every_ms"] = config.asr.partial_every_ms
    return body
