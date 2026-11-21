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
domain (``domain``) as its topic.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union,
)

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
    CancelToken,
    Clock,
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
from orchestrator.application.voice import (
    DEFAULT_BUDGET_MS,
    ClonedVoice,
    speak_sample,
)
from orchestrator.application.voicebot import VoiceBot
from orchestrator.domain.config import ProductConfig, Tts
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.live_choices import (
    MODES,
    apply_live_choices,
    voice_choices,
)
from orchestrator.domain.mandatory import EMPLOYEE_ROLES
from orchestrator.domain.voices import VoicePlan, VoiceProfile, plan_voice
from orchestrator.domain.records import SuggestionRecord, TurnRecord

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
        self.assist.on_event(event)
        if self.session.tracker is not None:
            self.session.tracker.on_event(event)

    def finish(self) -> None:
        """Wait for the call and the checks in flight."""
        self.assist.finish()
        if self.session.tracker is not None:
            self.session.tracker.finish()


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

    def publish(self, message: Document) -> None:
        """Publish ``message``; drop older turns' audio from the feed."""
        feed = self.session.feed
        if message.get("type") == "bot":
            turn = int(message.get("turn", 0))
            for old in [t for t in self._audio if t < turn]:
                feed.strip(self._audio.pop(old), "audio")
        number = feed.publish(message)
        if message.get("type") == "speech" and message.get("audio"):
            self._audio.setdefault(int(message["turn"]), []).append(number)

    def record(self, record: TurnRecord) -> None:
        """Count, publish and log one turn's record."""
        counts = self.session.by_status
        counts[record.status] = counts.get(record.status, 0) + 1
        self.session.feed.publish({
            "type": "turn", "record": record.model_dump(mode="json"),
        })
        if self.log is not None:
            self.log.write(record)


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
        interview: Plans, sheets and handoff of the AI interviewer
            (``None``: no ``voicebot.interview``).
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
        interview: Optional[InterviewWiring] = None,
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
        employee: Optional[Mapping[str, Any]] = None,
    ) -> Document:
        """Replay a recorded conversation as if it were live.

        ``employee`` (``{"id", "name"}``) names the transcript's speaker
        with the employee's role.
        """
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

    def start_live(
        self, product: str, audio: str, session_id: Optional[str] = None,
        employee: Optional[Mapping[str, Any]] = None,
        plan: Optional[str] = None, domain: Optional[str] = None,
        candidate: Optional[str] = None,
    ) -> Document:
        """Start a live session on the transcriber and follow it.

        ``employee`` (``{"id", "name"}``, an enrolled person) names the
        employee's channel in stereo; on one microphone their voiceprint
        marks their voice (unless «Ajustes» turned the diarization off).
        ``plan`` is the interview plan of a ``voicebot.interview``
        session (required there); ``domain`` replaces the knowledge
        domain of a plain voice bot (its topic); ``candidate`` names
        the candidate on the interview's sheet.
        """
        choices = self.settings.current() if self.settings else None
        config = self._topic(self._product(product, choices), domain)
        if (plan is None) != (config.interview is None):
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
        voiceprint = (person is not None and audio == "mic"
                      and (choices or {}).get("live.diarization") != "off")
        speakers = live_speakers(config, audio, person, voiceprint)
        roles = {s["id"]: s["role"] for s in speakers}
        names = {s["id"]: s["name"] for s in speakers if s.get("name")}
        if person is not None:
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
        session = self._open(session_id, config, "live", roles, person,
                             names)
        session.audio = audio
        session.plan = plan
        if candidate:
            session.candidate = " ".join(candidate.split())[:CANDIDATE_CHARS]
        try:
            started = self.transcriber.start(transcriber_body(
                config, session_id, speakers,
                diarize=True if voiceprint else None,
                enrolled=[person["id"]] if voiceprint and person else None,
            ))
        except TranscriberUnavailable as exc:
            session.error = str(exc)
            self._end(session, "failed")
            raise SessionError("transcriber", str(exc)) from exc
        assist = self._assist(session, {})
        session.feed.publish({"type": "transcriber", "session": started})
        handler = _Handler(session, assist)
        source = self.events
        if isinstance(assist, VoiceBot) and assist.director is not None:
            assist.open(session_id)

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

    def _cloned(self, profile: VoiceProfile) -> ClonedVoice:
        """Return the cloned voice of ``profile`` (Piper behind it)."""
        assert self.cloner is not None and self.tts is not None
        return ClonedVoice(self.cloner, self.tts, profile,
                           self.clone_budget_ms)

    def _voice(self, config: ProductConfig) -> Tuple[
        ProductConfig, TextToSpeech, VoicePlan
    ]:
        """Return the voice bot's config, its voice and the plan."""
        assert config.tts is not None and self.tts is not None
        plan = self._plan(config.tts)
        product = config.model_copy(update={"tts": config.tts.model_copy(
            update={"length_scale": plan.length_scale}
        )})
        if plan.clone is None:
            return product, self.tts, plan
        if self.cloner is None:
            return product, self.tts, VoicePlan(
                plan.voice, plan.length_scale, None,
                "; ".join(r for r in (plan.reason, "no cloner installed")
                          if r),
            )
        voice = self._cloned(plan.clone)
        self.runner.start(voice.warm)
        return product, voice, plan

    def sample(self, choices: Mapping[str, str]) -> Tuple[bytes, Document]:
        """Say the «Escuchar ejemplo» sentence with ``choices``; one WAV.

        ``choices`` are the voice bot's steps being edited (unsaved);
        anything not valid keeps the product's.
        """
        config = next((c for c in self.products.values()
                       if c.tts is not None), None)
        if config is None or config.tts is None or self.tts is None:
            raise SessionError("unavailable", "no voice bot here")
        tts = config.tts.model_copy(update=voice_choices(choices))
        plan = self._plan(tts)
        try:
            return speak_sample(
                tts, plan, self.tts,
                self._cloned if self.cloner is not None else None,
                config.language,
            )
        except TtsUnavailable as exc:
            raise SessionError("unavailable", str(exc)) from exc

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

    def _check_plan(self, plan_id: str) -> Document:
        """Check the interview plan exists."""
        if self.interview is None:
            raise SessionError("unavailable", "no interview plans here")
        try:
            return self.interview.planner.get(plan_id)
        except PlanError as exc:
            raise SessionError("not_found", str(exc)) from exc

    def _director(
        self, session: _Session, output: "_BotFeed"
    ) -> Optional[InterviewDirector]:
        """The interview director of a ``voicebot.interview`` session."""
        section = session.product.interview
        wiring = self.interview
        if section is None or wiring is None or session.plan is None:
            return None
        plan = wiring.store.load(session.plan)
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
        )

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
                names=dict(names or {}),
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
            product, speaker, plan = self._voice(session.product)
            session.product = product
            session.feed.publish({
                "type": "voice", "clone": plan.clone is not None
                and speaker is not self.tts,
                "length_scale": plan.length_scale, "reason": plan.reason,
            })
            output = _BotFeed(session, turns)
            bot = VoiceBot(
                product, self.knowledge, self.reasoner, speaker,
                output, self.clock, self.executor(), session.roles,
                self._director(session, output),
            )
            session.assist = bot
            return bot
        log = (self.log(session.id, session.product.product)
               if self.log is not None else None)
        assist = LiveAssist(
            session.product, self.knowledge, self.reasoner,
            _FeedSink(session, log), self.clock, self.executor(),
            session.roles, about,
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
        return assist

    def _end(self, session: _Session, state: str = "ended") -> None:
        """Mark the session ended, hand it off and close its feed."""
        if isinstance(session.assist, LiveAssist):
            self._hand_off(session)
        with self._lock:
            session.state = state
        if isinstance(session.assist, VoiceBot) and isinstance(
            session.assist.tts, ClonedVoice
        ):
            session.assist.tts.close()
        session.feed.publish({"type": "status", "session": session.view()})
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
        if self.handoff_sink is not None:
            try:
                self.handoff_sink(session.id, session.product.product
                                  ).write(record)
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


def diarized(config: ProductConfig) -> bool:
    """Tell whether the product asks for mono diarization (LV-6)."""
    return config.attribution.kind == "sortformer_stream"


def transcriber_body(
    config: ProductConfig, session_id: str, speakers: List[Document],
    diarize: Optional[bool] = None,
    enrolled: Optional[List[str]] = None,
) -> Document:
    """The body of the transcriber's ``POST /api/live/sessions``.

    Speakers all on channel 0 under a ``sortformer_stream`` product
    (or with ``diarize``) make a diarized one-channel session;
    ``enrolled`` limits the voiceprints that may name a voice.
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
