"""Use case: tick a live round's concepts with the model, beside it (#268).

The word rules of the topic tracker tick few concepts on real speech.
:class:`LiveTicker` asks the model, after a candidate final, which of
the pending concepts (:meth:`TopicTracker.focus_points`) the last answer
talked about (template ``interview.ticks@1``) and gives the reply to
:meth:`TopicTracker.suggested`, which ticks them ``by: "live_model"``.

It never touches the suggestions: it is its own call, on its own thread
(the session's runner), on the reasoner's **cold** backend (the
template is ``slow``), so it can neither delay a suggestion nor take or
displace the warm process the suggestions use. At most one call is in
flight; a final that arrives meanwhile waits as the next one, and a
newer final replaces it (the older is dropped as stale and counted). A
failed or late call ticks nothing and is logged with its status only,
never words. Measured on the warm path, putting the ticks into the
suggester's own reply (``suggester.interview@3``) made it miss the 1.8 s
deadline far more often, losing suggestions: see the design doc §6.3.
"""

import logging
import threading
from typing import Any, Callable, Deque, Dict, List, Optional

from collections import deque

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
WINDOW = 6
"""Lines of the conversation the call sees (the suggester's)."""


class LiveTicker:
    """Ask for the concepts each candidate final treated, one at a time.

    Args:
        reasoner: The reasoner port (its cold path serves the template).
        topics: The round's topic tracker.
        roles: Speaker id → role (the session's mapping, read live).
        start: Runs a job in the background (the session's runner).
        model: The model alias.
        language: The interview's language.
    """

    def __init__(
        self,
        reasoner: ReasonerPort,
        topics: TopicTracker,
        roles: Dict[str, str],
        start: Callable[[Callable[[], None]], None],
        model: str = "haiku",
        language: str = "es",
    ):
        """No call yet."""
        self.reasoner = reasoner
        self.topics = topics
        self.roles = roles
        self.start = start
        self.model = model
        self.language = language
        self.calls = 0
        self.stale = 0
        self._seq = 0
        self.failed = 0
        self._window: Deque[Dict[str, Any]] = deque(maxlen=WINDOW)
        self._lock = threading.Lock()
        self._busy = False
        self._next: Optional[Document] = None

    def on_event(self, event: LiveEvent) -> None:
        """Keep the window; ask about a candidate final."""
        u = event.utterance
        if event.type != "final" or u is None:
            return
        item = {"id": u.id, "speaker": u.speaker, "text": u.text,
                "start": u.start, "end": u.end}
        with self._lock:
            self._window = deque([w for w in self._window
                                  if w["id"] != u.id], maxlen=WINDOW)
            self._window.append(item)
            if self.roles.get(u.speaker) != CANDIDATE:
                return
            request = self._request(event.session_id, u.id,
                                    list(self._window))
            if self._busy:
                if self._next is not None:
                    self.stale += 1
                self._next = request
                return
            self._busy = True
        self.start(lambda: self._run(request))

    def _request(self, session_id: str, focus: str,
                 window: List[Dict[str, Any]]) -> Document:
        """The call's ``reasoner.request.v1`` (payload read when run)."""
        self._seq += 1
        return {
            "schema": "reasoner.request.v1",
            "request_id": f"{session_id}-tick{self._seq}"[-64:],
            "role": "structured", "model": self.model,
            "template": TICKS_TEMPLATE, "language": self.language,
            "context": {"window": window, "speaker_roles": dict(self.roles),
                        "focus": focus},
            "constraints": {"deadline_ms": TICKS_DEADLINE_MS},
        }

    def _run(self, request: Document) -> None:
        """Ask, tick, then run the waiting request, if any."""
        while True:
            try:
                self._ask(request)
            finally:
                with self._lock:
                    waiting, self._next = self._next, None
                    if waiting is None:
                        self._busy = False
            if waiting is None:
                return
            request = waiting

    def _ask(self, request: Document) -> None:
        """One call; the concepts still pending are read now."""
        payload = self.topics.focus_points()
        if not payload:
            return
        request = {**request, "payload": payload}
        self.calls += 1
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
        try:
            self.topics.suggested(payload, data)
        except Exception as exc:  # noqa: BLE001 - logged, nothing ticked
            self.failed += 1
            LOG.warning("ticks %s: reply not used (%s)",
                        request["request_id"], type(exc).__name__)
