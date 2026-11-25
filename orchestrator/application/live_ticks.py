"""Use case: tick a live round's concepts with the model, beside it (#268).

The word rules of the topic tracker tick few concepts on real speech.
:class:`LiveTicker` asks the model which of the pending concepts
(:meth:`TopicTracker.focus_points`) the candidate's answers talked
about (template ``interview.ticks@1``) and gives the reply to
:meth:`TopicTracker.suggested`, which ticks them ``by: "live_model"``.

**Bounded cost.** One call covers **every candidate final since the
previous call**, and two calls are at least ``every_s`` apart (the
product's ``interview.tick_every_s``, 30 s by default): at most about
``minutes × 60 / every_s`` calls a round, about 16 for 8 minutes. At
most one call is in flight; finals that arrive meanwhile wait for the
next call, which covers them all. No call is made when the round ends:
the after-session attribution reads the whole round anyway. Late
speech (a final heard over 10 s late after a reconnect, #263) gets no
suggestion but is still ticked, in the next call.

It never touches the suggestions: it is its own call, on its own thread
(the session's runner), on the reasoner's **cold** backend (the
template is ``slow``) and it *yields*: it takes a slow slot only when
one is free and uses at most one of them, so it never makes the
planner, the report or the attribution wait. A call that finds no free
slot, fails or comes late ticks nothing and is logged with its status
only, never words; its finals are not asked again.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from orchestrator.application.next_round import CANDIDATE, TopicTracker
from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerPort,
    ReasonerUnavailable,
)
from orchestrator.domain.events import LiveEvent

LOG = logging.getLogger(__name__)
TICKS_TEMPLATE = "interview.ticks@1"
TICKS_DEADLINE_MS = 20000
"""Deadline of one tick call: a cold process start (2–5 s) and a short
reply; a late tick is of no use to the interviewer."""
EVERY_S = 30.0
"""Least seconds between two tick calls (the product can change it)."""
MAX_LINES = 30
"""Most lines of the conversation one call reads (the oldest go)."""
MAX_ANSWERS = 12
"""Most candidate finals one call asks about (the most recent)."""


class LiveTicker:
    """Ask, now and then, which concepts the latest answers treated.

    Args:
        reasoner: The reasoner port (its cold path serves the template).
        topics: The round's topic tracker.
        roles: Speaker id → role (the session's mapping, read live).
        start: Runs a job in the background (the session's runner).
        model: The model alias.
        language: The interview's language.
        every_s: Least seconds between two calls.
        clock: Monotonic seconds.
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        reasoner: ReasonerPort,
        topics: TopicTracker,
        roles: Dict[str, str],
        start: Callable[[Callable[[], None]], None],
        model: str = "haiku",
        language: str = "es",
        every_s: float = EVERY_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """No call yet."""
        self.reasoner = reasoner
        self.topics = topics
        self.roles = roles
        self.start = start
        self.model = model
        self.language = language
        self.every_s = max(every_s, 0.0)
        self.clock = clock
        self.sleep = sleep
        self.calls = 0
        self.failed = 0
        self.session_id = "live"
        self._lines: List[Dict[str, Any]] = []
        self._answers: List[str] = []
        self._last: Optional[float] = None
        self._lock = threading.Lock()
        self._busy = False

    def on_event(self, event: LiveEvent) -> None:
        """Keep the conversation; a candidate final waits for a call."""
        u = event.utterance
        if event.type != "final" or u is None:
            return
        item = {"id": u.id, "speaker": u.speaker, "text": u.text,
                "start": u.start, "end": u.end}
        with self._lock:
            self.session_id = event.session_id
            self._lines = [x for x in self._lines if x["id"] != u.id]
            self._lines.append(item)
            if self.roles.get(u.speaker) != CANDIDATE:
                return
            if u.id not in self._answers:
                self._answers.append(u.id)
            if self._busy:
                return  # the call in flight or waiting will cover it
            self._busy = True
        self.start(self._run)

    def _run(self) -> None:
        """Wait for the spacing, ask about every waiting answer, repeat."""
        while True:
            with self._lock:
                if not self._answers:
                    self._busy = False
                    return
                wait = 0.0 if self._last is None else (
                    self._last + self.every_s - self.clock())
            if wait > 0:
                self.sleep(wait)
            with self._lock:
                answers, self._answers = self._answers, []
                if not answers:
                    self._busy = False
                    return
                self._last = self.clock()
                lines = self._window(answers)
                session_id = self.session_id
            try:
                self._ask(session_id, answers[-MAX_ANSWERS:], lines)
            except Exception as exc:  # noqa: BLE001 - logged, goes on
                self.failed += 1
                LOG.warning("ticks %s: call not made (%s)", session_id,
                            type(exc).__name__)

    def _window(self, answers: List[str]) -> List[Dict[str, Any]]:
        """The lines from just before the first answer asked (lock held).

        The previous line (usually the question) comes too; at most
        :data:`MAX_LINES`, the most recent.
        """
        ids = [x["id"] for x in self._lines]
        first = min((ids.index(a) for a in answers if a in ids),
                    default=len(ids))
        kept = self._lines[max(first - 1, 0):][-MAX_LINES:]
        self._lines = self._lines[-MAX_LINES:]
        return [dict(x) for x in kept]

    def _ask(self, session_id: str, answers: List[str],
             lines: List[Dict[str, Any]]) -> None:
        """One call; the concepts still pending are read now."""
        payload = self.topics.focus_points()
        shown = {x["id"] for x in lines}
        answers = [a for a in answers if a in shown]
        if not payload or not answers:
            return
        self.calls += 1
        request: Document = {
            "schema": "reasoner.request.v1",
            "request_id": f"{session_id}-tick{self.calls}"[-64:],
            "role": "structured", "model": self.model,
            "template": TICKS_TEMPLATE, "language": self.language,
            "context": {"window": lines, "speaker_roles": dict(self.roles),
                        "focus": answers[-1]},
            "payload": {**payload, "answers": answers},
            "constraints": {"deadline_ms": TICKS_DEADLINE_MS},
        }
        try:
            response = self.reasoner.reason(request, CancelToken())
        except ReasonerUnavailable as exc:
            self.failed += 1
            LOG.info("ticks %s: reasoner unavailable (%s)",
                     request["request_id"], type(exc).__name__)
            return
        data = response.get("data")
        if response.get("status") != "ok" or not isinstance(data, dict):
            self.failed += 1
            error = response.get("error")
            LOG.info("ticks %s: status %s, error %s", request["request_id"],
                     response.get("status"),
                     error.get("kind") if isinstance(error, dict) else None)
            return
        self.topics.suggested(request["payload"], data)
