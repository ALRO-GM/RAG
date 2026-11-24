"""Use case: flag the interviewer's questions better avoided (LV-17, #232).

In an interview, every final sentence of the interviewer is checked
against KN-4's rules through the RAG (:class:`AvoidPort`), off the event
loop like the mandatory items. A sentence that touches a protected
category (age, family plans, health...) is flagged with why and a
lawful alternative; a clean question soon after marks it rephrased
(:class:`~orchestrator.domain.avoid.AvoidLedger`). Each change is
published as ``{"type": "avoid", "flags": [...]}``. The candidate's
sentences are never checked and nothing here rates anyone. A check
that cannot be made turns the tracker off for the session and is said
once as a ``notice`` (``avoid_unavailable``).
"""

import threading
from typing import Callable, Dict, List, Optional, Set

from orchestrator.application.ports import (
    AvoidPort,
    AvoidUnavailable,
    Document,
    Executor,
)
from orchestrator.domain.avoid import AvoidLedger, parse_hints
from orchestrator.domain.events import LiveEvent

INTERVIEWER = "interviewer"
"""The only role whose sentences are checked."""


class AvoidTracker:
    """Check an interview's interviewer lines and keep the flags.

    Args:
        port: The RAG's check.
        roles: Speaker id → role (the employee's id may be added later).
        publish: Sends one feed message.
        executor: Runs the checks off the event loop.
        language: The language of the hints (the product's).
    """

    def __init__(
        self,
        port: AvoidPort,
        roles: Dict[str, str],
        publish: Callable[[Document], object],
        executor: Executor,
        language: Optional[str] = "es",
    ):
        """Keep the ports; nothing is checked yet."""
        self.port = port
        self.roles = roles
        self.publish = publish
        self.executor = executor
        self.language = language or "es"
        self.ledger = AvoidLedger()
        self.available = True
        self.checks = 0
        self._texts: Dict[str, str] = {}
        self._checked: Set[str] = set()
        self._lock = threading.Lock()

    def on_event(self, event: LiveEvent) -> None:
        """Check the interviewer's final sentences."""
        u = event.utterance
        if u is None or not self.available:
            return
        if event.type == "final":
            self._texts[u.id] = u.text
        elif event.type != "speaker_update":
            return
        if self.roles.get(u.speaker) != INTERVIEWER:
            return
        text = self._texts.get(u.id, "").strip()
        if not text or u.id in self._checked:
            return
        self._checked.add(u.id)
        at = u.start
        self.executor.submit(lambda: self._check(u.id, text, at))

    def _check(self, utterance: str, text: str, at: float) -> None:
        """Ask the RAG which categories ``text`` touches and keep them."""
        if not self.available:
            return
        try:
            answer = self.port.check(text, self.language)
        except AvoidUnavailable as exc:
            with self._lock:
                first = self.available
                self.available = False
            if first:
                self.publish({
                    "type": "notice", "code": "avoid_unavailable",
                    "message": f"questions are not checked: {exc}",
                })
            return
        with self._lock:
            self.checks += 1
            changed = self.ledger.record(
                utterance, at, text, parse_hints(answer)
            )
            flags = self.ledger.view()
        if changed:
            self.publish({"type": "avoid", "flags": flags})

    def view(self) -> List[Document]:
        """Return the flags as JSON."""
        with self._lock:
            return self.ledger.view()

    def finish(self) -> None:
        """Wait for the checks in flight."""
        self.executor.wait_idle()
