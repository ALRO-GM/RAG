"""Use case: the voice bot answers out loud (P3, ``ARCHITECTURE.md`` §3).

:class:`VoiceBot` receives the ``live.event.v1`` events of a one-person
microphone session. When the user's turn ends (a ``final`` of the user,
by the product's trigger) it starts a turn: it queries the knowledge
domain with what the user said since the bot last spoke, asks the
``conversational`` reasoner and speaks the reply **sentence by
sentence while the reply is still being written**: the reasoner's
pieces go through :class:`~orchestrator.domain.speech.ReplyText` and
:class:`~orchestrator.domain.speech.SentenceChunker`, and each sentence
becomes one ``tts.request.v1`` whose audio is published at once
(``{"type": "speech"}``, base64). A reasoner that does not stream is
spoken when it answers. The reply's final text is authoritative: what
the stream missed is spoken at the end.

One turn is in flight at a time; a newer user final cancels it
(``cancel_stale``) and the page is told to stop its playback
(``{"type": "interrupt"}``). Barge-in (VB-3,
:mod:`~orchestrator.domain.bargein`): while the bot's audio is still
playing, the page's voice detector (:meth:`VoiceBot.barge_in`, ``vad``)
or a user partial or final of two words or more (``speech``) stops the
turn and the playback; what is mostly the bot's own words coming back
through the microphone is echo, and is neither a barge-in nor a turn.
Each turn ends with a
:class:`~orchestrator.domain.records.TurnRecord` with the latency of
each stage. A quota or auth answer stops the session's turns, as in
:class:`~orchestrator.application.live_assist.LiveAssist`. No audio is
kept; the text is never logged here.
"""

import base64
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Protocol

from orchestrator.application.ports import (
    BotOutput,
    CancelToken,
    Clock,
    Document,
    Executor,
    KnowledgePort,
    KnowledgeUnavailable,
    ReasonerPort,
    ReasonerUnavailable,
    TextToSpeech,
    TtsUnavailable,
)
from orchestrator.application.voice import voice_request
from orchestrator.domain.bargein import BargeInPolicy, Playback, is_echo
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import (
    CallStatus,
    TriggerInfo,
    TurnRecord,
)
from orchestrator.domain.speech import (
    ReplyText,
    SentenceChunker,
    SpeechChunk,
)
from orchestrator.domain.triggers import Firing, Skip, TriggerPolicy


class Directed(Protocol):
    """What a directed turn must do (see :class:`TurnDirector`)."""

    @property
    def payload(self) -> Document:
        """The reasoner request's ``payload``."""
        ...

    @property
    def query(self) -> str:
        """The knowledge query."""
        ...


class TurnDirector(Protocol):
    """Decides what each turn does (the AI interviewer, #208).

    Without one the bot answers the user from the knowledge domain;
    with one, every turn asks the director first: its ``payload`` goes
    to the reasoner's request and its ``query`` to the knowledge
    domain. The director also hears the user's finals, is told what
    the bot spoke and is finished when the session ends.
    """

    @property
    def closed(self) -> bool:
        """Whether no more turns are wanted."""
        ...

    def heard(self, utterance: str, text: str, start: float,
              end: float) -> None:
        """Take one final of the user."""
        ...

    def plan_turn(
        self, answer: str, utterance: Optional[str], cancel: CancelToken
    ) -> Directed:
        """Decide a turn (``utterance`` ``None``: the bot speaks first)."""
        ...

    def spoken(self, record: TurnRecord) -> None:
        """Take the record of a turn that spoke."""
        ...

    def finish(self) -> Optional[Document]:
        """The session ended."""
        ...


MAX_REQUEST_ID = 64
CHUNK_TEXT = 600
HALTING = ("quota", "auth")
BOT_SPEAKER = "BOT"
"""The bot's speaker id in the reasoner's window."""
FALLBACK = {
    "es": "Perdona, ahora mismo no puedo responder. ¿Puedes repetirlo?",
    "en": "Sorry, I cannot answer right now. Could you say it again?",
}
"""What the bot says when the reasoner gave nothing to say."""
ECHO_WINDOW_S = 1.5
"""Seconds after the playback's estimated end where echo is still
possible (the transcriber's final comes after the hang)."""
SAID_KEPT = 8
"""The bot's last sentences compared against what the mic hears."""


def _ms(seconds: float) -> float:
    """Return ``seconds`` as milliseconds, to a tenth."""
    return round(seconds * 1000.0, 1)


def _clean(text: str) -> str:
    """Return ``text`` on one line with single spaces."""
    return " ".join(text.split())


class _Turn:
    """One turn's state while it is spoken (internal)."""

    def __init__(
        self, bot: "VoiceBot", record: TurnRecord, token: CancelToken,
        started: float, interrupt_id: Optional[str],
    ):
        """Start before the first piece of the reply."""
        self.bot = bot
        self.record = record
        self.token = token
        self.started = started
        self.interrupt_id = interrupt_id
        self.reply = ReplyText()
        self.chunker = SentenceChunker()
        self.streamed = ""
        self.sentences = 0
        self.last_request: Optional[str] = None
        self.lock = threading.Lock()

    def elapsed(self) -> float:
        """Milliseconds since the turn started."""
        return _ms(self.bot.clock.now() - self.started)

    def feed(self, piece: str) -> None:
        """Take one piece of the reasoner's output; speak what ended."""
        if self.token.cancelled:
            return
        with self.lock:
            text = self.reply.feed(piece)
            if not text:
                return
            if self.record.stages.first_text_ms is None:
                self.record.stages.first_text_ms = self.elapsed()
            self.streamed += text
            for sentence in self.chunker.feed(text):
                self._say(sentence)

    def finish(self, final: Optional[str]) -> None:
        """Speak what is left once the reply is complete."""
        with self.lock:
            done = _clean(self.streamed)
            rest: List[str] = []
            if final is not None:
                whole = _clean(final)
                if whole.startswith(done):
                    rest = self.chunker.feed(whole[len(done):])
            rest += self.chunker.flush()
            for sentence in rest:
                self._say(sentence)

    def say_whole(self, text: str) -> None:
        """Speak a text of the bot's own (the fallback)."""
        with self.lock:
            for sentence in self.chunker.feed(text) + self.chunker.flush():
                self._say(sentence)

    def _say(self, sentence: str) -> None:
        """Synthesise one sentence and publish its audio."""
        if self.token.cancelled:
            return
        stages = self.record.stages
        if stages.first_sentence_ms is None:
            stages.first_sentence_ms = self.elapsed()
        seq = self.sentences
        self.sentences += 1
        request_id = f"{self.record.request_id}-s{seq}"[-MAX_REQUEST_ID:]
        tts = self.bot.config.tts
        assert tts is not None  # checked by VoiceBot
        request = voice_request(
            request_id, sentence, tts, self.bot.config.language,
            interrupt_id=self.interrupt_id if seq == 0 else None,
        )
        self.last_request = request_id
        spoken = False
        try:
            for chunk in self.bot.tts.synthesize(request, self.token):
                if self.token.cancelled:
                    return
                self._publish(chunk)
                spoken = True
        except TtsUnavailable as exc:
            self.record.error = {"kind": "tts", "message": str(exc)}
        if not spoken and not self.token.cancelled:
            self.bot.output.publish({
                "type": "speech", "turn": self.record.turn, "seq": seq,
                "text": sentence, "audio": None,
            })
            self.record.spoken.append(sentence)

    def _publish(self, chunk: SpeechChunk) -> None:
        """Send one chunk of audio to the page."""
        stages = self.record.stages
        if stages.first_audio_ms is None:
            stages.first_audio_ms = self.elapsed()
            stages.first_tts_ms = chunk.synth_ms
        bot = self.bot
        with bot._lock:
            bot.playback.add(self.record.turn, chunk.duration_ms,
                             bot.clock.now())
            bot._said.append(chunk.text)
        self.record.spoken.append(chunk.text)
        self.record.audio_ms = round(
            self.record.audio_ms + chunk.duration_ms, 1
        )
        self.bot.output.publish({
            "type": "speech", "turn": self.record.turn,
            "seq": len(self.record.spoken) - 1, "text": chunk.text,
            "format": chunk.format, "sample_rate": chunk.sample_rate,
            "duration_ms": chunk.duration_ms,
            "audio": base64.b64encode(chunk.audio).decode("ascii"),
        })


class VoiceBot:
    """Answer the user out loud, turn by turn.

    Args:
        config: The ``voicebot`` product's config (needs trigger,
            reasoner and tts).
        knowledge: The knowledge port.
        reasoner: The reasoner port (streaming when it has
            ``reason_stream``).
        tts: Text to speech.
        output: Where the messages and turn records go.
        clock: Monotonic time.
        executor: Runs a turn (inline or on a thread).
        roles: Speaker id → role (the user).
        director: Decides each turn (the AI interviewer), if any.
    """

    def __init__(
        self,
        config: ProductConfig,
        knowledge: KnowledgePort,
        reasoner: ReasonerPort,
        tts: TextToSpeech,
        output: BotOutput,
        clock: Clock,
        executor: Executor,
        roles: Dict[str, str],
        director: Optional[TurnDirector] = None,
    ):
        """Wire the ports; no turn yet."""
        if (config.trigger is None or config.reasoner is None
                or config.tts is None):
            raise ValueError("VoiceBot needs the voicebot product config")
        self.config = config
        self.user_roles = set(config.trigger.speaker_roles)
        self.barge = BargeInPolicy(self.user_roles)
        self.playback = Playback()
        self._said: Deque[str] = deque(maxlen=SAID_KEPT)
        self.reasoner_config = config.reasoner
        self.knowledge = knowledge
        self.reasoner = reasoner
        self.tts = tts
        self.output = output
        self.clock = clock
        self.executor = executor
        self.roles = {**roles, BOT_SPEAKER: "bot"}
        self.policy = TriggerPolicy(config.trigger, roles)
        self.cancel_stale = config.trigger.cancel_stale
        self.skips: Dict[str, int] = {}
        self.turns = 0
        self.halted: Optional[str] = None
        self._window: Deque[Dict[str, Any]] = deque(
            maxlen=config.reasoner.window
        )
        self._lock = threading.Lock()
        self._inflight: Optional[_Turn] = None
        self._last_turn: Optional[_Turn] = None
        self.director = director

    # ---- events ----------------------------------------------------------

    def on_event(self, event: LiveEvent) -> None:
        """Handle one event: echo, barge-in, window, maybe a turn."""
        u = event.utterance
        if u is not None and event.type in ("partial", "final"):
            role = self.roles.get(u.speaker, "unknown")
            now = self.clock.now()
            with self._lock:
                said = list(self._said)
                speaking = self.playback.speaking(now)
                echo_possible = self.playback.recent(now, ECHO_WINDOW_S)
            if (role in self.user_roles and echo_possible
                    and is_echo(u.text, said)):
                if event.type == "final":
                    self._skip("echo")
                return
            if self.barge.decide(event, role, speaking, said):
                self.barge_in("speech")
        if event.type == "final" and u is not None:
            self._keep({"id": u.id, "speaker": u.speaker, "text": u.text,
                        "start": u.start, "end": u.end})
            if (self.director is not None
                    and self.roles.get(u.speaker) in self.user_roles):
                self.director.heard(u.id, u.text, u.start, u.end)
        decision = self.policy.decide(event)
        if isinstance(decision, Skip):
            self._skip(decision.reason)
        elif isinstance(decision, Firing):
            self._fire(event, decision)

    def finish(self) -> None:
        """Wait for the turn in flight, if any; finish the director."""
        self.executor.wait_idle()
        if self.director is not None:
            self.director.finish()

    def open(self, session_id: str) -> None:
        """Start a turn without the user (the bot speaks first)."""
        self._fire_turn(session_id, Firing(
            utterance_id="", speaker=BOT_SPEAKER, role="bot", text="",
            stream_time=0.0,
        ), None, opening=True)

    def barge_in(self, by: str = "vad") -> bool:
        """The user spoke over the bot: stop the turn and the playback.

        Returns whether there was anything to stop (a turn in flight or
        audio still playing).
        """
        now = self.clock.now()
        with self._lock:
            turn = self._inflight
            last = self._last_turn
            playing = self.playback.speaking(now)
            if turn is None and not playing:
                return False
            self.playback.stop()
        if turn is not None:
            self._interrupt(turn, by)
        elif last is not None:
            last.record.interrupted_by = by
            self.output.publish({"type": "interrupt",
                                 "turn": last.record.turn, "by": by})
        return True

    # ---- internals -------------------------------------------------------

    def _keep(self, item: Dict[str, Any]) -> None:
        """Add (or revise) one line of the window."""
        with self._lock:
            for kept in self._window:
                if kept["id"] == item["id"]:
                    kept.update(item)
                    return
            self._window.append(item)

    def _skip(self, reason: str) -> None:
        """Count a trigger that did not start a turn."""
        with self._lock:
            self.skips[reason] = self.skips.get(reason, 0) + 1

    def _user_turn(self) -> str:
        """What the user said since the bot last spoke."""
        said: List[str] = []
        for item in reversed(self._window):
            if item["speaker"] == BOT_SPEAKER:
                break
            said.append(item["text"])
        return " ".join(reversed(said))

    def _fire(self, event: LiveEvent, firing: Firing) -> None:
        """Start a turn, cancelling (or keeping) the one in flight."""
        if self.director is not None and self.director.closed:
            self._skip("closed")
            return
        lag = event.timings.lag_ms if event.timings is not None else None
        self._fire_turn(event.session_id, firing, lag)

    def _fire_turn(
        self, session_id: str, firing: Firing, lag: Optional[float],
        opening: bool = False,
    ) -> None:
        """Start a turn for ``firing`` (see :meth:`_fire`)."""
        with self._lock:
            if self.halted is not None:
                self.skips["halted"] = self.skips.get("halted", 0) + 1
                return
            stale = self._inflight
            if stale is not None and not self.cancel_stale:
                self.skips["in_flight"] = self.skips.get("in_flight", 0) + 1
                return
            self.turns += 1
            number = self.turns
            window = list(self._window)
            query = "" if opening else (self._user_turn() or firing.text)
        if stale is not None and not stale.token.cancelled:
            self._interrupt(stale, "new_turn")
        previous = self._last_turn
        token = CancelToken()
        record = TurnRecord(
            session_id=session_id, turn=number,
            request_id=f"{session_id}-b{number}"[-MAX_REQUEST_ID:],
            trigger=TriggerInfo(
                utterance=firing.utterance_id, speaker=firing.speaker,
                role=firing.role, stream_time=firing.stream_time,
                text=firing.text,
            ),
            status="error",
        )
        record.stages.event_lag_ms = lag
        turn = _Turn(self, record, token, self.clock.now(),
                     previous.last_request if previous else None)
        with self._lock:
            self._inflight = turn
            self._last_turn = turn
        self.output.publish({"type": "bot", "turn": number,
                             "state": "thinking",
                             "utterance": firing.utterance_id})
        utterance = None if opening else firing.utterance_id
        self.executor.submit(
            lambda: self._run(turn, query, window, utterance)
        )

    def _interrupt(self, turn: _Turn, by: str) -> None:
        """Stop a turn and tell the page to stop its playback."""
        turn.token.cancel()
        turn.record.interrupted_by = turn.record.interrupted_by or by
        self.output.publish({"type": "interrupt",
                             "turn": turn.record.turn, "by": by})

    def _run(
        self, turn: _Turn, query: str, window: List[Dict[str, Any]],
        utterance: Optional[str] = None,
    ) -> None:
        """Run one turn: (director,) knowledge, reasoner, speech."""
        record = turn.record
        if turn.token.cancelled:
            self._close(turn, "cancelled")
            return
        payload: Optional[Document] = None
        if self.director is not None:
            asked = self.clock.now()
            directed = self.director.plan_turn(query, utterance, turn.token)
            record.stages.director_ms = _ms(self.clock.now() - asked)
            payload = directed.payload
            query = directed.query
        chunks = self._retrieve(query, record)
        if turn.token.cancelled:
            self._close(turn, "cancelled")
            return
        request = self._request(record, window, chunks)
        if payload is not None:
            request["payload"] = payload
        asked = self.clock.now()
        status: CallStatus = "error"
        final: Optional[str] = None
        stream = getattr(self.reasoner, "reason_stream", None)
        try:
            if callable(stream):
                response = stream(request, turn.token, turn.feed)
            else:
                response = self.reasoner.reason(request, turn.token)
        except ReasonerUnavailable as exc:
            record.error = {"kind": "transient", "message": str(exc)}
        else:
            status = response.get("status", "error")
            reply = response.get("reply") or {}
            final = reply.get("text") if isinstance(reply, dict) else None
            record.reply = final
            record.cites = list(reply.get("cites") or []) if isinstance(
                reply, dict) else []
            record.model = response.get("model")
            record.backend = response.get("backend")
            record.error = response.get("error")
            record.stages.reasoner = dict(response.get("latency") or {})
        record.stages.reasoner_ms = _ms(self.clock.now() - asked)
        if not turn.token.cancelled:
            turn.finish(final if status == "ok" else None)
            if not record.spoken and status not in HALTING:
                turn.say_whole(FALLBACK.get(self.config.language[:2],
                                            FALLBACK["es"]))
        if turn.token.cancelled:
            status = "cancelled"
        elif status in HALTING:
            with self._lock:
                self.halted = status
        self._close(turn, status)

    def _retrieve(self, query: str, record: TurnRecord) -> List[Document]:
        """Query the knowledge domain; a failure means no chunks."""
        knowledge = self.config.knowledge
        if knowledge is None:
            return []
        asked = self.clock.now()
        try:
            result = self.knowledge.query({
                "schema": "knowledge.query.v1",
                "request_id": record.request_id,
                "domain": knowledge.domains[0], "text": query[:4000],
                "language": self.config.language, "top_k": knowledge.top_k,
            })
        except KnowledgeUnavailable:
            record.knowledge_status = "unavailable"
            chunks: List[Document] = []
        else:
            record.knowledge_status = "ok"
            chunks = list(result.get("chunks") or [])
        record.stages.knowledge_ms = _ms(self.clock.now() - asked)
        record.chunks = len(chunks)
        record.retrieved = [
            {"id": c.get("id"), "text": str(c.get("text", ""))[:CHUNK_TEXT],
             "score": c.get("score"), "metadata": c.get("metadata") or {}}
            for c in chunks[:10]
        ]
        return chunks

    def _request(
        self, record: TurnRecord, window: List[Dict[str, Any]],
        chunks: List[Document],
    ) -> Document:
        """Build the ``reasoner.request.v1`` of a turn."""
        cfg = self.reasoner_config
        constraints: Dict[str, Any] = {"stream": True}
        if cfg.deadline_ms:
            constraints["deadline_ms"] = cfg.deadline_ms
        request: Document = {
            "schema": "reasoner.request.v1",
            "request_id": record.request_id, "role": cfg.role,
            "model": cfg.model, "language": self.config.language,
            "context": {
                "session": {"id": record.session_id},
                "speaker_roles": self.roles, "window": window,
                "focus": record.trigger.utterance,
            },
            "retrieved": [
                {"id": c["id"], "text": c["text"],
                 "metadata": c.get("metadata") or {}}
                for c in chunks[:10]
            ],
            "constraints": constraints,
        }
        if cfg.template:
            request["template"] = cfg.template
        return request

    def _close(self, turn: _Turn, status: CallStatus) -> None:
        """Finish the record, remember what was said, free the slot."""
        record = turn.record
        record.stages.total_ms = turn.elapsed()
        record.status = status
        with self._lock:
            if self._inflight is turn:
                self._inflight = None
            if record.spoken:
                self._window.append({
                    "id": f"b{record.turn}", "speaker": BOT_SPEAKER,
                    "text": " ".join(record.spoken),
                })
        if self.director is not None and record.spoken:
            self.director.spoken(record)
        self.output.record(record)
