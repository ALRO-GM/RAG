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

A session of the ``voicebot`` product (``ui.kind: speaker``) is fed to
a :class:`~orchestrator.application.voicebot.VoiceBot` instead: its
feed carries ``{"type": "bot"}`` (a turn started), ``{"type":
"speech"}`` (one sentence: text and base64 WAV), ``{"type":
"interrupt"}`` and ``{"type": "turn", "record": ...}`` (the turn's
record and latency), and :meth:`LiveSessions.barge_in` stops it. The
audio of a turn is dropped from the feed once a newer turn starts, so
the feed does not keep a session's audio.

LV-16 adds, for live-page sessions: the **employee** chosen at the
start (an enrolled person: in stereo they name the employee's channel,
on one microphone their voiceprint marks their voice among the others,
which are the other party); **names** given to speakers during the
session (``{"type": "rename"}``, sent to the transcriber for the
``transcript.v1`` hand-off); the **assistance mode** switched live
(``{"type": "mode"}``); the **mandatory items** of the product's
domain (``{"type": "mandatory"}``, :mod:`.mandatory`); and a
**hand-off record** written at the end (``{"type": "handoff"}``: names,
mode, what was said, ticked and left unsaid).
A session of ``voicebot.interview`` (#208) starts from an interview
plan (``plan``): its voice bot is directed by an
:class:`~orchestrator.application.interview.InterviewDirector`, speaks
first, and its feed also carries ``{"type": "coverage"}`` after every
turn, ``{"type": "interview"}`` (``closed``, ``reporting``) and, at the
end, ``{"type": "interview_result"}`` (the scored slot sheet and the
handoff). A plain ``voicebot`` session may name another knowledge
domain (``domain``) as its topic. An interview's bot waits for the
candidate (#213): the session ticks its turn-taking (``ticker``) and
the feed also carries ``{"type": "backchannel"}``.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union,
)

from orchestrator.application.avoid import AvoidTracker
from orchestrator.application.candidates import CoreTracker
from orchestrator.application.live_judgement import (
    LiveJudge,
    without_words,
)
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.demo import ScriptedAssist
from orchestrator.application.feed import Feed
from orchestrator.application.interview import (
    HeuristicFiller,
    InterviewDirector,
    InterviewWiring,
    PlanError,
    ReasonerFiller,
    SlotFiller,
)
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.mandatory import MandatoryTracker, TickRefused
from orchestrator.application.ports import (
    AvoidPort,
    CancelToken,
    Clock,
    DemoScripts,
    Document,
    EventSource,
    Executor,
    HandoffSink,
    KnowledgePort,
    LiveTranscriber,
    MandatoryPort,
    ReasonerPort,
    Runner,
    SettingsSource,
    SuggestionSink,
    GpuGate,
    TextToSpeech,
    TranscriberUnavailable,
    TranscriptStore,
    TtsUnavailable,
    TurnSink,
    VoiceCloner,
    VoiceProfiles,
)
from orchestrator.application.replay import (
    Replay,
    default_roles,
    transcript_to_events,
)
from orchestrator.application.natural_voice import (
    SAMPLE_BUDGET_MS,
    NaturalVoice,
    NaturalWiring,
    gpu_need,
)
from orchestrator.application.voice import (
    DEFAULT_BUDGET_MS,
    ClonedVoice,
    SharedCloner,
    render_sample,
    speak_sample,
    tts_controls,
)
from orchestrator.application.voicebot import VoiceBot
from orchestrator.domain.candidate import CandidateProfile
from orchestrator.domain.next_round import (
    focus_goal,
    reordered,
    topic_states,
    with_focus,
)
from orchestrator.domain.config import InterviewSection, ProductConfig, Tts
from orchestrator.domain.demo import with_partials
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.live_choices import (
    MODES,
    apply_live_choices,
    voice_choices,
)
from orchestrator.domain.mandatory import EMPLOYEE_ROLES
from orchestrator.domain.voices import (
    VoicePlan,
    VoiceProfile,
    plan_voice,
    voice_chain,
)
from orchestrator.domain.records import SuggestionRecord, TurnRecord
from orchestrator.domain.turn_taking import TurnConfig

AUDIO_KINDS = ("stereo", "mic")
"""``stereo``: one channel per person (agent/interviewer on 0, the other
on 1, as the product says); ``mic``: the browser's microphone, one
channel heard as the person the product suggests about."""
MIN_SPEED, MAX_SPEED = 0.25, 20.0
UI_KINDS = ("live_page", "speaker")
"""The products served here: the live page's and the voice bot's."""
SESSION_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$"
PERSON_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
"""An enrolled person's id or a speaker id (the contracts' pattern)."""
MAX_NAME = 60
"""Longest name a speaker can be given."""
DOMAIN_ID = r"^[a-z0-9][a-z0-9._-]{0,63}$"
CANDIDATE_CHARS = 80
TICK_S = 0.1
DEMO_STOP_S = 3.0
"""Longest wait for a playing demo to stop before a new one starts."""
STOP_POLL_S = 0.05
"""Seconds between two turn-taking ticks of an interview (#213)."""
LOG = logging.getLogger(__name__)
"""Where each «Escuchar ejemplo» is logged (no text, no audio; #253)."""


class _Assistant(Protocol):
    """What follows a session's events (live assist or the voice bot)."""

    skips: Dict[str, int]

    def on_event(self, event: LiveEvent) -> None:
        """Handle one event."""
        ...

    def finish(self) -> None:
        """Wait for the call in flight."""
        ...


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
    assist: Optional[_Assistant] = None
    sinks: List[SuggestionSink] = field(default_factory=list)
    by_status: Dict[str, int] = field(default_factory=dict)
    employee: Optional[Dict[str, str]] = None
    names: Dict[str, str] = field(default_factory=dict)
    renames: List[Document] = field(default_factory=list)
    tracker: Optional[MandatoryTracker] = None
    handoff: Optional[Document] = None
    avoid: Optional[AvoidTracker] = None
    demo: Optional[Document] = None
    core: Optional[CoreTracker] = None
    topics: Optional[TopicTracker] = None
    goal: Optional[str] = None
    judge: Optional[LiveJudge] = None
    accuracy: Optional[Document] = None
    """How accurate the round's answers were (#259), when judged."""

    @property
    def assist_mode(self) -> Optional[str]:
        """The assistance mode of a live-assist session."""
        if isinstance(self.assist, LiveAssist):
            return self.assist.mode or MODES[0]
        return None
    plan: Optional[str] = None
    candidate: Optional[str] = None

    def view(self) -> Document:
        """Return the session as JSON."""
        assist = self.assist
        tracker = self.tracker
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
            "employee": self.employee, "names": dict(self.names),
            "assist_mode": self.assist_mode,
            "mandatory": tracker.view() if tracker and tracker.active
            else None,
            "handoff": self.handoff,
            "plan": self.plan, "candidate": self.candidate,
            "candidate_id": self.core.profile.id if self.core else None,
            "goal": self.goal,
            "avoid": self.avoid.view() if self.avoid else None,
            "demo": self.demo, "accuracy": self.accuracy,
        }


class _Handler:
    """Sends a session's events to the feed and to the use cases."""

    def __init__(self, session: _Session, assist: _Assistant):
        """Keep the session."""
        self.session = session
        self.assist = assist

    def on_event(self, event: LiveEvent) -> None:
        """Show the event, then let the use cases decide on it."""
        self.session.events += 1
        self.session.feed.publish(
            {"type": "event", "event": event.to_contract()}
        )
        if self.session.topics is not None:
            # First, so the suggestion this answer asks for already
            # knows the concepts it treated (#258); its failure never
            # keeps the event from the other use cases.
            try:
                self.session.topics.on_event(event)
            except Exception:  # noqa: BLE001 - logged, the call goes on
                LOG.exception("topics: event %s not followed",
                              event.seq)
        self.assist.on_event(event)
        if self.session.tracker is not None:
            self.session.tracker.on_event(event)
        if self.session.avoid is not None:
            self.session.avoid.on_event(event)
        if self.session.core is not None:
            self.session.core.on_event(event)

    def finish(self) -> None:
        """Wait for the call and the checks in flight."""
        self.assist.finish()
        if self.session.tracker is not None:
            self.session.tracker.finish()
        if self.session.avoid is not None:
            self.session.avoid.finish()
        core = self.session.core
        if core is not None:
            if self.session.topics is not None:
                core.covered, core.revisit = self.session.topics.result()
            core.finish()


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


class _BotFeed:
    """The voice bot's output: the feed, the counts and the turn log."""

    def __init__(self, session: _Session, log: Optional[TurnSink]):
        """Keep where the messages go."""
        self.session = session
        self.log = log
        self._audio: Dict[int, List[int]] = {}
        self._backchannel: Optional[int] = None

    def publish(self, message: Document) -> None:
        """Publish ``message``; drop older turns' audio from the feed."""
        feed = self.session.feed
        if message.get("type") == "bot":
            turn = int(message.get("turn", 0))
            for old in [t for t in self._audio if t < turn]:
                feed.strip(self._audio.pop(old), "audio")
        if (message.get("type") == "backchannel"
                and self._backchannel is not None):
            feed.strip([self._backchannel], "audio")
        if (message.get("type") == "interview_result"
                and self.session.core is not None):
            self.session.core.note_sheet(dict(message.get("sheet") or {}))
        number = feed.publish(message)
        if message.get("type") == "speech" and message.get("audio"):
            self._audio.setdefault(int(message["turn"]), []).append(number)
        if message.get("type") == "backchannel" and message.get("audio"):
            self._backchannel = number

    def record(self, record: TurnRecord) -> None:
        """Count, publish and log one turn's record."""
        if self.session.core is not None and record.spoken:
            # The AI interviewer's share of the talk (#235); which core
            # question it asked comes from its director, not its words.
            self.session.core.add_talk("interviewer",
                                       record.audio_ms / 1000.0)
        counts = self.session.by_status
        counts[record.status] = counts.get(record.status, 0) + 1
        self.session.feed.publish({
            "type": "turn", "record": record.model_dump(mode="json"),
        })
        if self.log is not None:
            self.log.write(record)


def turn_config(section: InterviewSection) -> TurnConfig:
    """The interviewer's turn-taking thresholds from its section."""
    return TurnConfig(
        turn_silence_s=section.turn_silence_s, think_s=section.think_s,
        max_wait_s=section.max_wait_s, stuck_s=section.stuck_s,
        backchannel=section.backchannel,
    )


def page_info(config: ProductConfig, roles: Mapping[str, str]) -> Document:
    """What the live page needs to know about a product."""
    reasoner = config.reasoner
    trigger = config.trigger
    modes = bool(reasoner and reasoner.template
                 and reasoner.template.startswith("suggester.")
                 and reasoner.template.endswith("@2"))
    knowledge = config.knowledge
    return {
        "domain": knowledge.domains[0] if knowledge else None,
        "interview": config.interview is not None,
        "product": config.product, "language": config.language,
        "modes": list(MODES) if modes else [],
        "kind": config.ui.kind if config.ui else None,
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
        tts: Text to speech of the voice bot (``None``: no voice bot).
        turn_log: Makes the turn log of a voice-bot session (``session
            id``, ``product``), or ``None`` for no log.
        new_id: Makes an id for a live session started without one.
        keep: Ended sessions kept.
        settings: The pipeline settings saved on «Ajustes», read when a
            session starts (reasoner model, deadline, mono diarization,
            assistance mode; ``None``: the products as written).
        mandatory: The mandatory items of the knowledge domains
            (``None``: no checklist).
        handoff: Makes the hand-off sink of a session (``session id``,
            ``product``), or ``None`` for no record.
        profiles: The enrolled people's voice profiles (rate matching
            and cloning; ``None``: neither).
        cloner: The voice-cloning model (``None``: Piper only).
        clone_budget_ms: Longest wait for one cloned sentence before
            Piper says it.
        natural: The natural voices, Pocket TTS and Qwen3-TTS (#214;
            ``None``: Piper and the cloner only).
        interview: Plans, sheets and handoff of the AI interviewer
            (``None``: no ``voicebot.interview``).
        ticker: Runs the interviewer's turn-taking clock beside the
            session (#213): :meth:`VoiceBot.tick` every ``tick_s``
            seconds while the session runs (``None``: nobody ticks; the
            tests tick by hand).
        tick_s: Seconds between two ticks.
        avoid: The check of interview questions better avoided (LV-17;
            ``None``: no check).
        demos: The scripted demos (LV-17; ``None``: none).
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
        tts: Optional[TextToSpeech] = None,
        turn_log: Optional[Callable[[str, str], TurnSink]] = None,
        settings: Optional[SettingsSource] = None,
        mandatory: Optional[MandatoryPort] = None,
        handoff: Optional[Callable[[str, str], HandoffSink]] = None,
        profiles: Optional[VoiceProfiles] = None,
        cloner: Optional[VoiceCloner] = None,
        clone_budget_ms: float = DEFAULT_BUDGET_MS,
        natural: Optional[NaturalWiring] = None,
        clone_gate: Optional[GpuGate] = None,
        interview: Optional[InterviewWiring] = None,
        ticker: Optional[Runner] = None,
        tick_s: float = TICK_S,
        avoid: Optional[AvoidPort] = None,
        demos: Optional[DemoScripts] = None,
    ):
        """Wire the ports; no session yet."""
        self.products = {
            name: config for name, config in products.items()
            if config.is_live and config.ui is not None
            and config.ui.kind in UI_KINDS
            and (config.tts is None or tts is not None)
            and (config.interview is None or interview is not None)
        }
        self.interview = interview
        self.tts = tts
        self.turn_log = turn_log
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
        self.settings = settings
        self.mandatory = mandatory
        self.handoff_sink = handoff
        self.profiles = profiles
        self.cloner = cloner
        self.clone_budget_ms = clone_budget_ms
        self.natural = natural
        self.clone_gate = clone_gate
        self._shared: Dict[int, SharedCloner] = {}
        self.ticker = ticker
        self.tick_s = tick_s
        self.avoid = avoid
        self.demos = demos
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ---- queries -------------------------------------------------------

    def list_products(self) -> List[Document]:
        """The products the page can start, with what it shows."""
        return [
            {**page_info(c, self._channel_roles(c)),
             "live": self.transcriber is not None,
             # One microphone starts with the voice check (#249).
             "two_party": c.tts is None and two_parties(c) is not None}
            for _, c in sorted(self.products.items())
        ]

    def list_transcripts(self) -> List[Document]:
        """Conversations a replay can use."""
        return self.transcripts.list()

    def list_demos(self) -> List[Document]:
        """Scripted demos a replay can play (their product must exist)."""
        if self.demos is None:
            return []
        return [d for d in self.demos.list() if d["product"] in self.products]

    def demo_part(self, demo_id: str, part: str) -> Document:
        """Return the report or the voice example of a demo (#250).

        Args:
            demo_id: The demo.
            part: ``report`` or ``voicebot``.

        Raises:
            SessionError: ``not_found`` without that demo or that part.
        """
        try:
            if self.demos is None or part not in ("report", "voicebot"):
                raise KeyError(demo_id)
            script = self.demos.load(demo_id)
        except KeyError as exc:
            raise SessionError("not_found", f"no demo {demo_id!r}") from exc
        found = script.report if part == "report" else script.voicebot
        if found is None:
            raise SessionError("not_found", f"demo {demo_id!r} has no {part}")
        body: Document = found.model_dump(mode="json", exclude_none=True)
        if part == "report":
            # The voice example keeps its own title and English texts.
            body.update({"title": script.title, "en": dict(script.en or {})})
        body.update({"demo": True, "id": script.id, "note": script.note,
                     "sources": [s.model_dump(mode="json")
                                 for s in script.sources],
                     "mandatory_items": [
                         i.model_dump(mode="json", exclude={"said_by"})
                         for i in (script.mandatory.items
                                   if script.mandatory else [])],
                     "utterances": [
                         u.model_dump(mode="json", include={
                             "id", "speaker", "start", "end", "text"})
                         for u in script.transcript.utterances],
                     "speakers": [s.model_dump(mode="json")
                                  for s in script.transcript.speakers]})
        return body

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
        employee: Optional[Mapping[str, Any]] = None,
        demo: Optional[str] = None,
    ) -> Document:
        """Replay a recorded conversation as if it were live.

        ``employee`` (``{"id", "name"}``) names the transcript's speaker
        with the employee's role. ``demo`` plays a scripted demo instead
        (its own product and conversation; ``product`` and
        ``transcript_id`` are not used).
        """
        if demo is not None:
            return self._start_demo(session_id, demo, speed)
        config = self._product(product)
        _check_id(session_id)
        person = check_employee(employee)
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
        names: Dict[str, str] = {}
        speaker = next((s for s, r in roles.items() if r in EMPLOYEE_ROLES),
                       None)
        if person is not None and speaker is not None:
            person = {**person, "role": roles[speaker], "speaker": speaker}
            names[speaker] = person["name"]
        self._stop_demos()
        session = self._open(session_id, config, "replay", roles,
                             person if speaker else None, names)
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

    def _start_demo(
        self, session_id: str, demo_id: str, speed: Optional[float]
    ) -> Document:
        """Play a scripted demo through the replay (LV-17, #232).

        No knowledge, reasoner, mandatory or avoid call is made: the
        demo's prepared suggestions, hints and topics are published by
        :class:`ScriptedAssist`. The sentences type in (partials).
        """
        _check_id(session_id)
        if speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
            raise SessionError(
                "invalid_request",
                f"speed must be between {MIN_SPEED} and {MAX_SPEED}",
            )
        try:
            if self.demos is None:
                raise KeyError(demo_id)
            script = self.demos.load(demo_id)
        except KeyError as exc:
            raise SessionError("not_found", f"no demo {demo_id!r}") from exc
        config = self._product(script.product)
        with self._lock:
            taken = session_id in self._sessions
        if taken:
            raise SessionError(
                "wrong_state", f"session {session_id!r} exists"
            )
        self._stop_demos()
        roles = default_roles(config, script.transcript)
        names = {s.id: s.name for s in script.transcript.speakers
                 if s.name}
        session = self._open(
            session_id, config, "demo", roles, None, names,
            demo=script.summary(),
        )
        session.transcript = script.transcript.session.id
        assist = ScriptedAssist(
            script, session_id, roles, _FeedSink(session, None),
            session.feed.publish,
        )
        session.assist = assist
        assist.start()
        handler = _Handler(session, assist)
        events = list(transcript_to_events(script.transcript, roles))
        for event in events:
            event.session_id = session_id

        def job() -> None:
            try:
                Replay(handler, self.clock, speed, session.stop_token).run(
                    with_partials(events)
                )
            finally:
                self._end(session)

        self.runner.start(job)
        return session.view()

    def _stop_demos(self, wait_s: float = DEMO_STOP_S) -> None:
        """Stop a demo still playing, so a new session can start at once.

        A demo left open in another tab would otherwise hold the one
        session slot until it ends by itself (about a minute). Only
        demos are stopped: a real session is never stopped for anything.
        """
        with self._lock:
            playing = [s for s in self._sessions.values()
                       if s.state == "running" and s.mode == "demo"]
        for session in playing:
            session.stop_token.cancel()
        waited = 0.0
        while playing and waited < wait_s and any(
            s.state == "running" for s in playing
        ):
            self.clock.sleep(STOP_POLL_S)
            waited += STOP_POLL_S

    def start_live(
        self, product: str, audio: str, session_id: Optional[str] = None,
        employee: Optional[Mapping[str, Any]] = None,
        plan: Optional[str] = None, domain: Optional[str] = None,
        candidate: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> Document:
        """Start a live session on the transcriber and follow it.

        ``employee`` (``{"id", "name"}``, an enrolled person) names the
        employee's channel in stereo; on one microphone their voiceprint
        marks their voice (unless «Ajustes» turned the diarization off).
        ``plan`` is the interview plan of a ``voicebot.interview``
        session (required there); ``domain`` replaces the knowledge
        domain of a plain voice bot (its topic); ``candidate`` names
        the candidate on the interview's sheet. A live-page interview
        may name a plan too (the role, #235): with ``candidate_id`` (a
        candidate of that role) or ``candidate`` (a new one) the
        session keeps the candidate's answers to the core questions.
        """
        choices = self.settings.current() if self.settings else None
        config = self._topic(self._product(product, choices), domain)
        interviews = (config.interview is not None
                      or employee_role(config) == "interviewer")
        if (config.interview is not None and plan is None) or (
                plan is not None and not interviews):
            raise SessionError(
                "invalid_request",
                "an interview session needs a plan, and only it",
            )
        person = check_employee(employee)
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
        diarizing = (audio == "mic"
                     and (choices or {}).get("live.diarization") != "off")
        # Two people on one microphone (#249): a voice check, then every
        # sentence is the employee's or the other person's.
        parties = (two_parties(config, person)
                   if diarizing and config.tts is None else None)
        voiceprint = person is not None and diarizing and parties is None
        speakers = (party_speakers(config, parties, person)
                    if parties is not None
                    else live_speakers(config, audio, person, voiceprint))
        roles = {s["id"]: s["role"] for s in speakers}
        names = {s["id"]: s["name"] for s in speakers if s.get("name")}
        if parties is not None and person is not None:
            person = {**person, "role": employee_role(config),
                      "speaker": parties["employee"]}
        elif person is not None:
            role = employee_role(config)
            if voiceprint:
                roles[person["id"]] = role
                names[person["id"]] = person["name"]
                person = {**person, "role": role, "speaker": person["id"]}
            else:
                speaker = next((s["id"] for s in speakers
                                if s["role"] == role), None)
                person = ({**person, "role": role, "speaker": speaker}
                          if speaker else None)
        if plan is not None:
            self._check_plan(plan)
        profile = self._candidate(plan, candidate, candidate_id)
        made = profile if profile is not None and not candidate_id else None
        self._stop_demos()  # a demo never keeps a real session waiting
        try:
            session = self._open(session_id, config, "live", roles, person,
                                 names)
        except SessionError:
            self._forget_new(made)
            raise
        session.audio = audio
        session.plan = plan
        if candidate:
            session.candidate = " ".join(candidate.split())[:CANDIDATE_CHARS]
        if profile is not None and self.interview is not None \
                and self.interview.candidates is not None:
            session.candidate = profile.name
            session.core = CoreTracker(
                self.interview.candidates, profile, session_id,
                session.roles, session.feed.publish,
                "bot" if config.interview is not None else "live",
            )
            session.core.start()
            if config.interview is None and plan is not None:
                self._topics(session, plan, profile)
        # A voice bot's voice starts loading now, while the transcriber
        # starts its session (its GPU lease is what a GPU voice waits for).
        early = (self._assist(session, {})
                 if config.tts is not None and self.tts is not None
                 else None)
        try:
            started = self.transcriber.start(transcriber_body(
                config, session_id, speakers,
                diarize=True if voiceprint or parties else None,
                enrolled=([person["id"]] if person else [])
                if voiceprint or parties else None,
                parties=parties,
            ))
        except TranscriberUnavailable as exc:
            session.error = str(exc)
            self._end(session, "failed")
            # A candidate made for this session is not left behind empty.
            self._forget_new(made)
            raise SessionError("transcriber", str(exc)) from exc
        assist = early if early is not None else self._assist(
            session, self._about(session))
        self._follow_concepts(session, assist)
        session.feed.publish({"type": "transcriber", "session": started})
        handler = _Handler(session, assist)
        source = self.events
        if isinstance(assist, VoiceBot) and assist.director is not None:
            assist.open(session_id)
            if assist.turn_taking is not None:
                self._listen(session, assist)

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
                judging = (session.judge is not None
                           and session.topics is not None)
                self._end(session, "failed" if failed else "ended",
                          keep_feed=judging)
                if judging:
                    # After the end (#259 review): the session no longer
                    # counts as running while its round is judged.
                    self.runner.start(lambda: self._judge_round(session))

        self.runner.start(job)
        return session.view()

    def _judge_round(self, session: _Session) -> None:
        """Judge an ended live round's answers, then close its feed.

        The result goes to the feed (the page may still follow it) and
        into the hand-off record, whose file keeps only the labels and
        counts (:func:`~orchestrator.application.live_judgement.
        without_words`). A failure is logged and changes nothing else.
        """
        judge, topics = session.judge, session.topics
        try:
            if judge is not None and topics is not None:
                session.accuracy = judge.judge(topics)
                if session.accuracy is not None and session.handoff:
                    session.handoff = {**session.handoff,
                                       "accuracy": session.accuracy}
                    self._write_handoff(session)
        except Exception:  # noqa: BLE001 - logged, the session is over
            LOG.exception("accuracy: round %s not judged", session.id)
        finally:
            session.feed.publish({"type": "status",
                                  "session": session.view()})
            session.feed.close()

    def rejudge(self, session_id: str) -> Document:
        """Judge an ended live round again («Reintentar», #262).

        Only for a round with a role's plan whose judgement failed (or
        never came); it runs on the runner and the page reads the result
        in the session's view (``accuracy``).
        """
        session = self._find(session_id)
        if session.judge is None or session.topics is None:
            raise SessionError("wrong_state", "this round is not judged")
        if session.state == "running":
            raise SessionError("wrong_state", "the round is still running")
        state = (session.accuracy or {}).get("state")
        if state in ("reporting", "ready"):
            raise SessionError("wrong_state", f"the judgement is {state}")
        session.accuracy = {"state": "reporting"}
        self.runner.start(lambda: self._judge_again(session))
        return session.view()

    def _judge_again(self, session: _Session) -> None:
        """Run «Reintentar»: as at the end of the round, feed closed."""
        judge, topics = session.judge, session.topics
        try:
            if judge is not None and topics is not None:
                session.accuracy = judge.judge(topics)
                if session.accuracy is not None and session.handoff:
                    session.handoff = {**session.handoff,
                                       "accuracy": session.accuracy}
                    self._write_handoff(session)
        except Exception:  # noqa: BLE001 - logged, shown as failed
            LOG.exception("accuracy: round %s not judged again", session.id)
            session.accuracy = {"state": "failed", "saved": False}

    def barge_in(self, session_id: str, by: str = "vad") -> Document:
        """The user spoke over the voice bot: stop its turn and playback."""
        session = self._find(session_id)
        bot = session.assist
        if not isinstance(bot, VoiceBot):
            raise SessionError("wrong_state", "not a voice-bot session")
        stopped = session.state == "running" and bot.barge_in(by)
        return {"stopped": stopped}

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

    def product(self, name: str) -> ProductConfig:
        """A live product as the saved settings leave it now."""
        return self._product(name)

    def _product(
        self, name: str, choices: Optional[Mapping[str, str]] = None
    ) -> ProductConfig:
        config = self.products.get(name)
        if config is None:
            raise SessionError("not_found", f"no live product {name!r}")
        if choices is None and self.settings is not None:
            choices = self.settings.current()
        return apply_live_choices(config, choices)

    def _plan(self, tts: Tts) -> VoicePlan:
        """Decide the voice of ``tts`` from the people's profiles."""
        profiles = self.profiles
        return plan_voice(
            tts.voice, tts.length_scale,
            tts.clone_of if profiles is not None else None,
            tts.match_rate_of if profiles is not None else None,
            profiles.get if profiles is not None else (lambda _: None),
        )

    def _cloned(
        self, profile: VoiceProfile,
        controls: Optional[Dict[str, float]] = None,
        sample: bool = False,
        report: Optional[Callable[[str, str], None]] = None,
    ) -> ClonedVoice:
        """Return the cloned voice of ``profile`` (Piper behind it).

        A cloner on the GPU runs under the live GPU lease
        (``clone_gate``) and speaks with the session's Piper
        ``controls``.
        """
        assert self.cloner is not None and self.tts is not None
        with self._lock:
            shared = self._shared.get(id(self.cloner))
            if shared is None or shared.inner is not self.cloner:
                shared = self._shared[id(self.cloner)] = SharedCloner(
                    self.cloner)
        return ClonedVoice(shared.lend(), self.tts, profile,
                           SAMPLE_BUDGET_MS if sample else
                           self.clone_budget_ms, gate=self.clone_gate,
                           controls=controls, sleep=self.clock.sleep,
                           report=report)

    def _voice(
        self, config: ProductConfig,
        report: Optional[Callable[[str, str], None]] = None,
        sample: bool = False,
    ) -> Tuple[ProductConfig, TextToSpeech, VoicePlan, str]:
        """Return the voice bot's config, its voice, the plan and engine.

        The engine is the first that will try to speak (``piper``,
        ``openvoice`` or a natural voice's name); ``report`` is told
        when a natural voice's speaker changes. A sample waits for a
        GPU model to load instead of the session's budget.
        """
        assert config.tts is not None and self.tts is not None
        tts = config.tts
        plan = self._plan(tts)
        product = config.model_copy(update={"tts": tts.model_copy(
            update={"length_scale": plan.length_scale}
        )})
        natural = self.natural
        installed = (natural.engines.installed() if natural is not None
                     else frozenset())
        qwen3_clone = (plan.clone is not None and tts.clone_engine == "qwen3"
                       and "qwen3" in installed)
        if plan.clone is not None and not qwen3_clone:
            why = ("Qwen3-TTS not installed"
                   if tts.clone_engine == "qwen3" else "")
            if self.cloner is None:
                return product, self.tts, _with_reason(
                    plan, why, "no cloner installed", clone=False), "piper"
            voice = self._cloned(plan.clone,
                                 tts_controls(tts, plan.length_scale),
                                 sample, report)
            if not sample:
                self.runner.start(voice.warm)
            return product, voice, _with_reason(plan, why), "openvoice"
        if natural is None:
            why = "natural voices off" if tts.engine != "piper" else ""
            return product, self.tts, _with_reason(plan, why), "piper"
        chain, missing = voice_chain(
            tts.engine, tts.pocket_voice, tts.pocket_device,
            plan.clone if qwen3_clone else None, installed,
            natural.reference,
        )
        plan = _with_reason(plan, missing, clone=bool(chain) and qwen3_clone)
        if not chain:
            return product, self.tts, plan, "piper"
        speaker = NaturalVoice(
            chain, natural.engines, self.tts, natural.gate,
            SAMPLE_BUDGET_MS if sample else natural.budget_ms,
            language=config.language, report=report,
            sleep=self.clock.sleep, ready_wait_s=natural.ready_wait_s,
        )
        if not sample:
            speaker.warming()
            self.runner.start(speaker.warm)
        return product, speaker, plan, chain[0].name

    def sample(self, choices: Mapping[str, str]) -> Tuple[bytes, Document]:
        """Say the «Escuchar ejemplo» sentence with ``choices``; one WAV.

        ``choices`` are the voice bot's steps being edited (unsaved);
        anything not valid keeps the product's. A natural voice on the
        GPU takes the lease for the sample when nobody holds it. Every
        sample is logged (:func:`sample_log`): what spoke, why not the
        chosen voice, how long it took, or why it failed (#253).
        """
        started = time.perf_counter()
        kind = _voice_kind(choices)
        try:
            wav, used = self._sample(choices)
        except SessionError as exc:
            LOG.warning("%s", sample_log(kind, None, started, str(exc)))
            raise
        except Exception as exc:
            LOG.exception("%s", sample_log(kind, None, started,
                                           type(exc).__name__))
            raise
        LOG.info("%s", sample_log(kind, used, started))
        return wav, used

    def _sample(self, choices: Mapping[str, str]) -> Tuple[bytes, Document]:
        """Render the sample (see :meth:`sample`)."""
        config = next((c for c in self.products.values()
                       if c.tts is not None), None)
        if config is None or config.tts is None or self.tts is None:
            raise SessionError("unavailable", "no voice bot here")
        tts = config.tts.model_copy(update=voice_choices(choices))
        product = config.model_copy(update={"tts": tts})
        heard: List[str] = []
        _, speaker, plan, engine = self._voice(
            product, lambda _, why: heard.append(why), sample=True)
        try:
            if isinstance(speaker, ClonedVoice):
                return self._cloned_sample(tts, plan, speaker, config.language)
            if not isinstance(speaker, NaturalVoice):
                return speak_sample(tts, plan, self.tts, None,
                                    config.language)
            return self._natural_sample(tts, plan, speaker, heard,
                                        config.language)
        except TtsUnavailable as exc:
            raise SessionError("unavailable", str(exc)) from exc
        finally:
            if isinstance(speaker, (NaturalVoice, ClonedVoice)):
                speaker.close()

    def _cloned_sample(
        self, tts: Tts, plan: VoicePlan, speaker: ClonedVoice, language: str,
    ) -> Tuple[bytes, Document]:
        """Render the sample with the cloner (the lease taken if it needs
        the GPU and nobody holds it, like a natural voice's)."""
        assert self.tts is not None
        gate = self.clone_gate
        need = speaker.need_mib if gate is not None else 0
        # A model already loaded (a session holds it) needs no memory.
        ask = 0 if speaker.cloner.loaded else need
        refused: Optional[str] = None
        taken = False
        try:
            if gate is not None and need:
                refused = gate.take(ask)
                taken = True
            wav, used = speak_sample(
                tts, plan, self.tts,
                lambda profile: speaker, language)
        finally:
            if gate is not None and taken:
                gate.drop()
        if refused or speaker.refusal:
            used["reason"] = _join_reasons(
                used["reason"], refused, speaker.refusal)
        return wav, used

    def _natural_sample(
        self, tts: Tts, plan: VoicePlan, speaker: NaturalVoice,
        heard: List[str], language: str,
    ) -> Tuple[bytes, Document]:
        """Render the sample with a natural voice (lease taken if needed)."""
        gate = self.natural.gate if self.natural is not None else None
        need = gpu_need(speaker)
        refused: Optional[str] = None
        taken = False
        try:
            if gate is not None and need:
                refused = gate.take(need)
                taken = True
            wav = render_sample(tts, plan.length_scale, speaker, language)
        finally:
            if gate is not None and taken:
                gate.drop()
        engine = speaker.first_speaker or "piper"
        reason = _join_reasons(plan.reason, refused, *heard)
        return wav, {
            "voice": tts.voice, "length_scale": plan.length_scale,
            "clone": plan.clone is not None and engine.startswith("qwen3"),
            "reason": reason, "engine": engine,
        }

    def _topic(
        self, config: ProductConfig, domain: Optional[str]
    ) -> ProductConfig:
        """Return ``config`` with ``domain`` as its knowledge domain."""
        if not domain:
            return config
        if (config.ui is None or config.ui.kind != "speaker"
                or config.interview is not None
                or config.knowledge is None):
            raise SessionError("invalid_request",
                               "only a voice bot takes a topic domain")
        if not re.fullmatch(DOMAIN_ID, domain):
            raise SessionError("invalid_request", "invalid domain id")
        knowledge = config.knowledge.model_copy(update={"domains": [domain]})
        return config.model_copy(update={"knowledge": knowledge})

    def _candidate(
        self, plan: Optional[str], name: Optional[str],
        candidate_id: Optional[str],
    ) -> Optional[CandidateProfile]:
        """The candidate of a session for a role, made when new (#235).

        ``None`` without a plan, without a candidate or where candidate
        profiles are not kept.
        """
        wiring = self.interview
        book = wiring.candidates if wiring is not None else None
        if plan is None or book is None or not (candidate_id or name):
            return None
        try:
            if candidate_id:
                return book.profile(plan, candidate_id)
            made = book.create(plan, name)
            return book.profile(plan, str(made["id"]))
        except PlanError as exc:
            code = exc.code if exc.code in ("not_found", "invalid_request") \
                else "invalid_request"
            raise SessionError(code, str(exc)) from exc

    def _forget_new(self, made: Optional[CandidateProfile]) -> None:
        """Delete a candidate made for a session that did not start."""
        wiring = self.interview
        if made is not None and wiring is not None \
                and wiring.candidates is not None:
            wiring.candidates.forget_new(made.plan_id, made.id)

    def _topics(
        self, session: _Session, plan_id: str, profile: CandidateProfile
    ) -> None:
        """Show the role's topics in the next round's order (#235)."""
        wiring = self.interview
        if wiring is None:
            return
        try:
            plan = wiring.store.load(plan_id)
        except KeyError:
            return
        states = topic_states(plan, profile.rounds)
        session.topics = TopicTracker(plan, states, session.roles,
                                      session.feed.publish,
                                      session.product.language)
        session.topics.start()
        reasoner = session.product.reasoner
        session.judge = LiveJudge(
            plan, wiring, session.id, session.feed.publish,
            reasoner.model if reasoner else "haiku",
            candidate=profile.name, candidate_id=profile.id)
        goal = focus_goal(plan.title, states, len(profile.rounds),
                          session.product.language)
        session.goal = goal

    @staticmethod
    def _follow_concepts(session: _Session, assist: _Assistant) -> None:
        """Give the suggestions the concepts still to treat (#258).

        The concepts of the topic being asked that are still pending go
        into the goal of the suggester's next requests (its existing
        call: no extra model call). The request's session is replaced
        whole, never changed in place, as a call may be reading it.
        """
        topics = session.topics
        if topics is None or not isinstance(assist, LiveAssist):
            return
        base = session.goal

        def on_focus(line: Optional[str]) -> None:
            goal = with_focus(base, line)
            about = dict(assist.session)
            if goal:
                about.update({"type": "entrevista", "goal": goal})
            else:
                about.pop("goal", None)
            assist.session = about

        topics.on_focus = on_focus

    @staticmethod
    def _about(session: _Session) -> Dict[str, Optional[str]]:
        """The conversation's kind and goal for the suggestions.

        A later round of a candidate (#235) names what earlier rounds
        left, so the suggestions start from it.
        """
        if session.goal is None:
            return {}
        return {"type": "entrevista", "goal": session.goal}

    def _check_plan(self, plan_id: str) -> Document:
        """Check the interview plan exists and its position is open.

        A closed position (#262) starts no interview, AI or live.
        """
        if self.interview is None:
            raise SessionError("unavailable", "no interview plans here")
        try:
            found = self.interview.planner.get(plan_id)
        except PlanError as exc:
            raise SessionError("not_found", str(exc)) from exc
        if found.get("state") == "closed":
            raise SessionError("plan_closed",
                               f"the position {plan_id!r} is closed")
        return found

    def _director(
        self, session: _Session, output: "_BotFeed"
    ) -> Optional[InterviewDirector]:
        """The interview director of a ``voicebot.interview`` session."""
        section = session.product.interview
        wiring = self.interview
        if section is None or wiring is None or session.plan is None:
            return None
        plan = wiring.store.load(session.plan)
        if session.core is not None:
            # A later round starts from what earlier ones left (#235).
            plan = reordered(plan, topic_states(
                plan, session.core.profile.rounds))
        reasoner = session.product.reasoner
        model = reasoner.model if reasoner else "haiku"
        filler: SlotFiller = (
            ReasonerFiller(wiring.reasoner, section.filler_template, model,
                           session.product.language)
            if section.filler == "haiku" else HeuristicFiller()
        )
        return InterviewDirector(
            plan, session.id, section, filler, wiring.reasoner, output,
            self.clock, wiring.store, wiring.handoff, section.post_call,
            session.candidate, model,
            core_open=session.core.open if session.core else None,
            on_core=session.core.expect if session.core else None,
            knowledge=wiring.planner.knowledge,
            candidate_id=session.core.profile.id if session.core else None,
        )

    def _listen(self, session: _Session, bot: VoiceBot) -> None:
        """Warm the backchannels and tick the turn-taking (#213).

        The backchannels are synthesised once the opening turn is over,
        so they never take the voice from the opening line.
        """
        def backchannels() -> None:
            bot.executor.wait_idle()
            bot.warm_backchannels()

        self.runner.start(backchannels)
        ticker = self.ticker
        if ticker is None:
            return

        def loop() -> None:
            while (session.state == "running"
                   and not session.stop_token.cancelled):
                self.clock.sleep(self.tick_s)
                bot.tick()

        ticker.start(loop)

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
        employee: Optional[Dict[str, str]] = None,
        names: Optional[Dict[str, str]] = None,
        demo: Optional[Document] = None,
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
                feed=Feed(), stop_token=CancelToken(), employee=employee,
                names=dict(names or {}), demo=demo,
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
    ) -> Union[LiveAssist, VoiceBot]:
        """Build the use case of one session."""
        if session.product.tts is not None and self.tts is not None:
            turns = (self.turn_log(session.id, session.product.product)
                     if self.turn_log is not None else None)
            plan_clone = False

            def report(engine: str, why: str) -> None:
                session.feed.publish({
                    "type": "voice", "engine": engine,
                    "clone": plan_clone and (engine.startswith("qwen3")
                                             or engine == "openvoice"),
                    "length_scale": product.tts.length_scale
                    if product.tts is not None else 1.0,
                    "reason": why,
                })

            product, speaker, plan, engine = self._voice(session.product,
                                                         report)
            plan_clone = plan.clone is not None
            clone = plan_clone and speaker is not self.tts
            session.product = product
            session.feed.publish({
                "type": "voice", "engine": engine, "clone": clone,
                "length_scale": plan.length_scale, "reason": plan.reason,
            })
            if isinstance(speaker, NaturalVoice) and not speaker.ready:
                self._preparing(session, speaker)
            output = _BotFeed(session, turns)
            director = self._director(session, output)
            section = product.interview
            bot = VoiceBot(
                product, self.knowledge, self.reasoner, speaker,
                output, self.clock, self.executor(), session.roles,
                director, turn_config(section)
                if director is not None and section is not None else None,
            )
            session.assist = bot
            return bot
        log = (self.log(session.id, session.product.product)
               if self.log is not None else None)
        assist = LiveAssist(
            session.product, self.knowledge, self.reasoner,
            _FeedSink(session, log), self.clock, self.executor(),
            session.roles, about,
            screen=session.core.screen if session.core else None,
        )
        session.assist = assist
        knowledge = session.product.knowledge
        if self.mandatory is not None and knowledge is not None:
            tracker = MandatoryTracker(
                self.mandatory, knowledge.domains[0], session.roles,
                session.feed.publish, self.executor(),
            )
            session.tracker = tracker
            tracker.start()
        if self.avoid is not None and \
                employee_role(session.product) == "interviewer":
            session.avoid = AvoidTracker(
                self.avoid, session.roles, session.feed.publish,
                self.executor(), session.product.language,
            )
        return assist

    def _preparing(self, session: _Session, speaker: NaturalVoice) -> None:
        """Tell the page the voice is loading, then when it is ready.

        ``voice_state`` is ``warming`` now, then ``ready`` (every engine
        tried) or ``fallback`` (the bound passed: the fallbacks speak
        until it loads).
        """
        session.feed.publish({"type": "voice_state", "state": "warming"})

        def wait() -> None:
            ready = speaker.wait_ready()
            if session.state == "running":
                session.feed.publish({
                    "type": "voice_state",
                    "state": "ready" if ready else "fallback",
                })

        self.runner.start(wait)

    def _end(self, session: _Session, state: str = "ended",
             keep_feed: bool = False) -> None:
        """Mark the session ended, hand it off and close its feed.

        ``keep_feed``: leave the feed open (a round still being judged
        closes it when done).
        """
        if isinstance(session.assist, LiveAssist):
            self._hand_off(session)
        with self._lock:
            session.state = state
        if isinstance(session.assist, VoiceBot) and isinstance(
            session.assist.tts, (ClonedVoice, NaturalVoice)
        ):
            voice = session.assist.tts
            if isinstance(voice, ClonedVoice):
                session.feed.publish({
                    "type": "voice_summary", "engine": "openvoice",
                    "cloned": voice.cloned, "fell_back": voice.fell_back,
                })
            voice.close()
        session.feed.publish({"type": "status", "session": session.view()})
        if not keep_feed:
            session.feed.close()

    def _hand_off(self, session: _Session) -> None:
        """Publish and keep the session's hand-off record (LV-16)."""
        tracker = session.tracker
        record: Document = {
            "session_id": session.id, "product": session.product.product,
            "assist_mode": session.assist_mode,
            "employee": session.employee,
            "names": dict(session.names), "renames": list(session.renames),
            "mandatory": tracker.summary() if tracker else None,
        }
        session.handoff = record
        session.feed.publish({"type": "handoff", "handoff": record})
        self._write_handoff(session)

    def _write_handoff(self, session: _Session) -> None:
        """Keep the hand-off record in its file (judgement: labels only)."""
        record = session.handoff
        if self.handoff_sink is None or record is None:
            return
        kept = dict(record)
        if isinstance(kept.get("accuracy"), dict):
            kept["accuracy"] = without_words(kept["accuracy"])
        try:
            self.handoff_sink(session.id, session.product.product
                              ).write(kept)
        except OSError:
            session.feed.publish({
                "type": "notice", "code": "handoff_not_kept",
                "message": "the hand-off record was not written",
            })

    # ---- during a session (LV-16) ------------------------------------

    def set_mode(self, session_id: str, mode: str) -> Document:
        """Switch the assistance mode; the next call uses it."""
        session = self._running(session_id)
        if mode not in MODES:
            raise SessionError("invalid_request",
                               f"mode must be one of {MODES}")
        if not isinstance(session.assist, LiveAssist) \
                or not page_info(session.product, {})["modes"]:
            raise SessionError("wrong_state",
                               "this session has no assistance modes")
        session.assist.mode = mode
        session.feed.publish({"type": "mode", "mode": mode})
        return {"mode": mode}

    def rename(self, session_id: str, speaker: str, name: object) -> Document:
        """Give a speaker a name (live: also for the transcript)."""
        session = self._running(session_id)
        if not isinstance(speaker, str) or not re.fullmatch(
            PERSON_ID, speaker
        ):
            raise SessionError("invalid_request", "invalid speaker id")
        clean = check_name(name)
        session.names[speaker] = clean
        change: Document = {"speaker": speaker, "name": clean,
                            "at": round(self.clock.now(), 3)}
        session.renames.append(change)
        session.feed.publish({"type": "rename", **change})
        kept = None
        if session.mode == "live" and self.transcriber is not None:
            try:
                self.transcriber.rename(session_id, speaker, clean)
                kept = True
            except TranscriberUnavailable:
                kept = False
        return {"speaker": speaker, "name": clean, "transcript": kept}

    def tick(self, session_id: str, item: str, confirm: bool) -> Document:
        """Tick a mandatory item by hand (needs ``confirm``; logged)."""
        session = self._running(session_id)
        if session.tracker is None or not session.tracker.active:
            raise SessionError("not_found", "no mandatory items")
        try:
            return session.tracker.tick(item, confirm,
                                        round(self.clock.now(), 3))
        except TickRefused as exc:
            raise SessionError("invalid_request", str(exc)) from exc

    def _running(self, session_id: str) -> _Session:
        session = self._find(session_id)
        if session.state != "running":
            raise SessionError("wrong_state", "the session has ended")
        return session


def _with_reason(
    plan: VoicePlan, *more: str, clone: Optional[bool] = None
) -> VoicePlan:
    """Return ``plan`` with more reasons (and without its clone if not)."""
    reason = "; ".join(r for r in (plan.reason, *more) if r)
    return VoicePlan(plan.voice, plan.length_scale,
                     plan.clone if clone is not False else None, reason)


def _join_reasons(*reasons: Optional[str]) -> str:
    """Join the reasons that are set, each once, in order."""
    return "; ".join(dict.fromkeys(str(r) for r in reasons if r))


def _voice_kind(choices: Mapping[str, str]) -> str:
    """What kind of voice a sample asks for, without the person's id."""
    option = str(choices.get("voicebot.voice", "") or "")
    if option.startswith("clone:"):
        return f"clone ({choices.get('voicebot.clone_engine') or 'default'})"
    return option[:32] or "default"


def sample_log(
    kind: str, used: Optional[Document], started: float, error: str = "",
) -> str:
    """One log line about a sample: the voice asked for, what spoke, why.

    Never the text said, the audio or who is cloned (``kind`` comes from
    :func:`_voice_kind`); ``started`` is a ``time.perf_counter`` reading.
    """
    ms = round((time.perf_counter() - started) * 1000)
    if used is None:
        return f"voice sample failed: voice={kind} ms={ms} error={error[:160]}"
    reason = str(used.get("reason") or "")[:200]
    return (f"voice sample: voice={kind} engine={used.get('engine')} "
            f"clone={'yes' if used.get('clone') else 'no'} ms={ms}"
            + (f" reason={reason}" if reason else ""))


def _check_id(session_id: str) -> None:
    """Refuse an id the contract's ``session_id`` would not take."""
    if not re.fullmatch(SESSION_ID, session_id or ""):
        raise SessionError("invalid_request", "invalid session id")


def live_speakers(
    config: ProductConfig,
    audio: str,
    employee: Optional[Mapping[str, str]] = None,
    voiceprint: bool = False,
) -> List[Document]:
    """The transcriber's speakers for ``audio``.

    With an ``employee`` (LV-16): in ``stereo`` the channel of the
    employee's role takes their name; with ``voiceprint`` (``mic``)
    every declared speaker goes on channel 0 with the other party's
    role, and the employee's voiceprint names their voice (the
    transcriber's ``enrolled``), so whoever else talks is the other
    party.

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
        role = employee_role(config)
        return [
            {"channel": c, "id": ch.speaker, "role": ch.role,
             **({"name": employee["name"]}
                if employee and ch.role == role else {})}
            for c, ch in channels
        ]
    if voiceprint and channels:
        other = other_role(config)
        return [{"channel": 0, "id": ch.speaker, "role": other}
                for _, ch in channels]
    if diarized(config) and channels:
        return [{"channel": 0, "id": ch.speaker, "role": ch.role}
                for _, ch in channels]
    wanted = config.trigger.speaker_roles if config.trigger else []
    chosen = next((ch for _, ch in channels if ch.role in wanted),
                  channels[-1][1] if channels else None)
    if chosen is None:
        return [{"channel": 0, "id": "S1", "role": "unknown"}]
    return [{"channel": 0, "id": chosen.speaker, "role": chosen.role}]


def two_parties(
    config: ProductConfig, employee: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """The two parties of a one-microphone session (#249), if the product
    has them.

    A product with an employee's channel (agent, interviewer) and the
    other party's (customer, candidate) is two people: the employee's
    id is the chosen person's (their voiceprint, when enrolled, still
    helps) or that channel's speaker, the other party's is their
    channel's speaker. ``voice_check``: the session starts by
    listening to the employee. ``None`` for any other product.
    """
    channels = [ch for _, ch in sorted(config.source.channels.items())]
    mine = next((ch for ch in channels if ch.role in EMPLOYEE_ROLES), None)
    other = next((ch for ch in channels if ch.role == other_role(config)
                  and ch.role not in EMPLOYEE_ROLES), None)
    if mine is None or other is None:
        return None
    ident = employee["id"] if employee else mine.speaker
    if ident == other.speaker:
        return None
    return {"employee": ident, "other": other.speaker, "voice_check": True}


def party_speakers(
    config: ProductConfig, parties: Mapping[str, Any],
    employee: Optional[Mapping[str, str]] = None,
) -> List[Document]:
    """The transcriber's speakers of a two-party mono session (#249)."""
    mine: Document = {"channel": 0, "id": parties["employee"],
                      "role": employee_role(config)}
    if employee:
        mine["name"] = employee["name"]
    return [mine, {"channel": 0, "id": parties["other"],
                   "role": other_role(config)}]


def diarized(config: ProductConfig) -> bool:
    """Tell whether the product asks for mono diarization (LV-6)."""
    return config.attribution.kind == "sortformer_stream"


def transcriber_body(
    config: ProductConfig, session_id: str, speakers: List[Document],
    diarize: Optional[bool] = None,
    enrolled: Optional[List[str]] = None,
    parties: Optional[Mapping[str, Any]] = None,
) -> Document:
    """The body of the transcriber's ``POST /api/live/sessions``.

    Speakers all on channel 0 under a ``sortformer_stream`` product
    (or with ``diarize``) make a diarized one-channel session;
    ``enrolled`` limits the voiceprints that may name a voice;
    ``parties`` makes it a two-party session with a voice check (#249).
    """
    channels = len({s["channel"] for s in speakers}) or 1
    # The interviewer hands over its own transcript (both voices); the
    # transcriber hears only the candidate.
    handoff = (config.handoff.model_dump() if config.interview is None
               else {"transcript_v1": False, "run_post_call": False})
    body: Dict[str, Any] = {
        "session_id": session_id, "channels": channels,
        "speakers": speakers, "product": config.product,
        "language": config.language, "handoff": handoff,
    }
    if (diarized(config) if diarize is None else diarize) \
            and channels == 1:
        body["diarize"] = True
        if enrolled is not None:
            body["enrolled"] = list(enrolled)
        if parties is not None:
            body["parties"] = dict(parties)
    if config.vad is not None:
        body["hang_ms"] = config.vad.hang_ms
    if config.asr.partial_every_ms:
        body["partial_every_ms"] = config.asr.partial_every_ms
    return body


def employee_role(config: ProductConfig) -> str:
    """The employee's role in a product (agent, interviewer)."""
    roles = [ch.role for _, ch in sorted(config.source.channels.items())]
    return next((r for r in roles if r in EMPLOYEE_ROLES),
                EMPLOYEE_ROLES[0])


def other_role(config: ProductConfig) -> str:
    """The other party's role (customer, candidate)."""
    wanted = config.trigger.speaker_roles if config.trigger else []
    roles = [ch.role for _, ch in sorted(config.source.channels.items())]
    return next((r for r in roles if r in wanted),
                next((r for r in roles if r not in EMPLOYEE_ROLES),
                     "unknown"))


def check_name(name: object) -> str:
    """Return a speaker name trimmed to one line, or refuse it."""
    if not isinstance(name, str):
        raise SessionError("invalid_request", "name must be text")
    clean = " ".join(name.split())
    if not clean or len(clean) > MAX_NAME or any(
        ord(c) < 32 for c in clean
    ):
        raise SessionError(
            "invalid_request", f"a name has 1 to {MAX_NAME} characters"
        )
    return clean


def check_employee(
    employee: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, str]]:
    """Return ``{"id", "name"}`` of the chosen employee, or refuse it."""
    if employee is None:
        return None
    if not isinstance(employee, Mapping):
        raise SessionError("invalid_request", "employee must be an object")
    person = employee.get("id")
    if not isinstance(person, str) or not re.fullmatch(PERSON_ID, person):
        raise SessionError("invalid_request", "invalid employee id")
    name = employee.get("name") or person
    return {"id": person, "name": check_name(name)}
