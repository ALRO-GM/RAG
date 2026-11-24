"""Use case: follow a conversation and suggest as it goes.

:class:`LiveAssist` receives ``live.event.v1`` events one by one. The
trigger policy decides which of them ask for a suggestion; for each one
it queries the knowledge domain, asks the reasoner and writes a
:class:`~orchestrator.domain.records.SuggestionRecord` with the latency
of each stage. At most one call is in flight: when a newer trigger
arrives, the call in flight is cancelled (``cancel_stale``) and its
answer is dropped, or the new trigger is skipped when ``cancel_stale``
is off. A service failure becomes the record's status and never stops
the conversation, except a quota or auth answer: then no further call
is made in the session (skip reason ``halted``), as the design says
(``ARCHITECTURE.md`` §6). The suggestions are prepared by
:func:`~orchestrator.domain.signals.present`: expressed customer
signals are quoted and rank them in support, and are removed in every
other product. The assistance mode (``pincelada`` or ``guiado``, LV-16)
goes in the request's constraints and can change between two calls.
Transcript text is never logged here.
"""

import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from orchestrator.application.ports import (
    CancelToken,
    Clock,
    Document,
    Executor,
    KnowledgePort,
    KnowledgeUnavailable,
    ReasonerPort,
    ReasonerUnavailable,
    SuggestionSink,
)
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import (
    CallStatus,
    Stages,
    SuggestionRecord,
    TriggerInfo,
)
from orchestrator.domain.signals import present
from orchestrator.domain.triggers import Firing, Skip, TriggerPolicy

MAX_REQUEST_ID = 64
CHUNK_TEXT = 600
"""Characters of a retrieved chunk kept in the record (for citations)."""
HALTING = ("quota", "auth")
"""Statuses after which the session stops asking the reasoner."""


class LiveAssist:
    """Turn a stream of events into suggestions.

    Args:
        config: A live product's config (needs trigger and reasoner).
        knowledge: The knowledge port.
        reasoner: The reasoner port.
        sink: Where the records go.
        clock: Monotonic time.
        executor: Runs a call (inline or on a thread).
        roles: Speaker id → role.
        session: ``type`` and ``goal`` of the conversation, if known.
        screen: Drops suggestions before they are written (#235: the
            core questions a candidate already answered).
    """

    def __init__(
        self,
        config: ProductConfig,
        knowledge: KnowledgePort,
        reasoner: ReasonerPort,
        sink: SuggestionSink,
        clock: Clock,
        executor: Executor,
        roles: Dict[str, str],
        session: Optional[Dict[str, Optional[str]]] = None,
        screen: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
    ):
        """Wire the ports; no call yet."""
        if config.trigger is None or config.reasoner is None:
            raise ValueError("LiveAssist needs a live product config")
        self.config = config
        self.trigger = config.trigger
        self.reasoner_config = config.reasoner
        self.knowledge = knowledge
        self.reasoner = reasoner
        self.sink = sink
        self.clock = clock
        self.executor = executor
        self.roles = dict(roles)
        self.session = session or {}
        self.screen = screen
        self.policy = TriggerPolicy(config.trigger, roles)
        self.skips: Dict[str, int] = {}
        self.calls = 0
        self._window: Deque[Dict[str, Any]] = deque(
            maxlen=config.reasoner.window
        )
        self._lock = threading.Lock()
        self._inflight: Optional[CancelToken] = None
        self._texts: Dict[str, Tuple[str, str]] = {}
        self.signal_history: Dict[str, int] = {}
        self.halted: Optional[str] = None
        self.mode: Optional[str] = config.reasoner.mode
        """Assistance mode sent to the ``@2`` templates; the page may
        switch it during the session (the next call uses it)."""

    def on_event(self, event: LiveEvent) -> None:
        """Handle one event: keep the window, maybe start a call."""
        u = event.utterance
        if event.type == "final" and u is not None:
            item = {"id": u.id, "speaker": u.speaker, "text": u.text,
                    "start": u.start, "end": u.end}
            if u.id in self._texts:
                self._replace(item)
            else:
                self._window.append(item)
            self._texts[u.id] = (u.speaker, u.text)
        elif event.type == "speaker_update" and u is not None:
            if u.id in self._texts:
                self._texts[u.id] = (u.speaker, self._texts[u.id][1])
                self._replace({"id": u.id, "speaker": u.speaker})
        decision = self.policy.decide(event)
        if isinstance(decision, Skip):
            self.skips[decision.reason] = (
                self.skips.get(decision.reason, 0) + 1
            )
        elif isinstance(decision, Firing):
            self._fire(event, decision)

    def _replace(self, item: Dict[str, Any]) -> None:
        """Update the fields of an utterance already in the window."""
        for kept in self._window:
            if kept["id"] == item["id"]:
                kept.update(item)

    def finish(self) -> None:
        """Wait for the call in flight, if any."""
        self.executor.wait_idle()

    def _fire(self, event: LiveEvent, firing: Firing) -> None:
        """Start a call for ``firing``, cancelling or skipping a stale one."""
        with self._lock:
            if self.halted is not None:
                self.skips["halted"] = self.skips.get("halted", 0) + 1
                return
            if self._inflight is not None:
                if not self.trigger.cancel_stale:
                    self.skips["in_flight"] = (
                        self.skips.get("in_flight", 0) + 1
                    )
                    return
                self._inflight.cancel()
            self.calls += 1
            number = self.calls
            token = CancelToken()
            self._inflight = token
        window = list(self._window)
        started = self.clock.now()
        self.executor.submit(
            lambda: self._call(event, firing, number, token, window, started)
        )

    def _call(
        self,
        event: LiveEvent,
        firing: Firing,
        number: int,
        token: CancelToken,
        window: List[Dict[str, Any]],
        started: float,
    ) -> None:
        """Run the knowledge and reasoner stages of one call."""
        request_id = f"{event.session_id}-{number}"[-MAX_REQUEST_ID:]
        record = SuggestionRecord(
            session_id=event.session_id, call=number,
            request_id=request_id,
            trigger=TriggerInfo(
                utterance=firing.utterance_id, speaker=firing.speaker,
                role=firing.role, stream_time=firing.stream_time,
                text=firing.text,
            ),
            status="error",
        )
        stages = Stages()
        chunks = self._retrieve(firing, request_id, record, stages)
        if token.cancelled:
            self._close(record, stages, started, token, "cancelled")
            return
        request = self._request(event, firing, request_id, window, chunks)
        record.mode = request["constraints"].get("mode")
        asked = self.clock.now()
        status: CallStatus = "error"
        try:
            response = self.reasoner.reason(request, token)
        except ReasonerUnavailable as exc:
            record.error = {"kind": "transient", "message": str(exc)}
        else:
            status = response.get("status", "error")
            record.suggestions = self._present(
                list(response.get("suggestions") or [])
            )
            record.reply = response.get("reply")
            record.model = response.get("model")
            record.backend = response.get("backend")
            record.error = response.get("error")
            stages.reasoner = dict(response.get("latency") or {})
        stages.reasoner_ms = _ms(self.clock.now() - asked)
        if token.cancelled:
            record.suggestions, record.reply = [], None
            status = "cancelled"
        elif status in HALTING:
            with self._lock:
                self.halted = status
        self._close(record, stages, started, token, status)

    def _present(
        self, suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Quote and rank signals (support) or remove them (others)."""
        with self._lock:
            shown = present(
                suggestions, self._texts, self.roles,
                self.reasoner_config.expressed_signals, self.signal_history,
            )
        return self.screen(shown) if self.screen is not None else shown

    def _retrieve(
        self,
        firing: Firing,
        request_id: str,
        record: SuggestionRecord,
        stages: Stages,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge domain; a failure means no chunks."""
        knowledge = self.config.knowledge
        if knowledge is None:
            return []
        query: Document = {
            "schema": "knowledge.query.v1", "request_id": request_id,
            "domain": knowledge.domains[0], "text": firing.text[:4000],
            "language": self.config.language, "top_k": knowledge.top_k,
        }
        asked = self.clock.now()
        try:
            result = self.knowledge.query(query)
        except KnowledgeUnavailable:
            record.knowledge_status = "unavailable"
            chunks: List[Dict[str, Any]] = []
        else:
            record.knowledge_status = "ok"
            chunks = list(result.get("chunks") or [])
        stages.knowledge_ms = _ms(self.clock.now() - asked)
        record.chunks = len(chunks)
        record.retrieved = [
            {"id": c.get("id"), "text": str(c.get("text", ""))[:CHUNK_TEXT],
             "score": c.get("score"), "metadata": c.get("metadata") or {}}
            for c in chunks[:10]
        ]
        return chunks

    def _request(
        self,
        event: LiveEvent,
        firing: Firing,
        request_id: str,
        window: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> Document:
        """Build the ``reasoner.request.v1`` of a call."""
        cfg = self.reasoner_config
        session: Dict[str, Any] = {"id": event.session_id}
        session.update({k: v for k, v in self.session.items() if v})
        constraints: Dict[str, Any] = {"max_suggestions": cfg.max_suggestions}
        if cfg.deadline_ms:
            constraints["deadline_ms"] = cfg.deadline_ms
        if self.mode or (cfg.template or "").endswith("@2"):
            # @2 templates always get their mode, so each record says it.
            constraints["mode"] = self.mode or "pincelada"
        request: Document = {
            "schema": "reasoner.request.v1", "request_id": request_id,
            "role": cfg.role, "model": cfg.model,
            "language": self.config.language,
            "context": {
                "session": session, "speaker_roles": self.roles,
                "window": window, "focus": firing.utterance_id,
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

    def _close(
        self,
        record: SuggestionRecord,
        stages: Stages,
        started: float,
        token: CancelToken,
        status: CallStatus,
    ) -> None:
        """Finish the record, write it and free the in-flight slot."""
        stages.total_ms = _ms(self.clock.now() - started)
        record.stages = stages
        record.status = status
        with self._lock:
            if self._inflight is token:
                self._inflight = None
            self.sink.write(record)


def _ms(seconds: float) -> float:
    """Return ``seconds`` as milliseconds, to a tenth."""
    return round(seconds * 1000.0, 1)
