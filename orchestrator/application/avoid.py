"""Use case: flag the interviewer's questions better avoided (LV-17, #232).

In an interview, every final sentence of the interviewer is checked
against KN-4's rules through the RAG (:class:`AvoidPort`), off the event
loop like the mandatory items. A sentence that touches a protected
category (age, family plans, health...) is flagged with why and a
lawful alternative; a clean question soon after marks it rephrased
(:class:`~orchestrator.domain.avoid.AvoidLedger`). A line a later
``speaker_update`` gives to the candidate loses its flags. Each change
is published as ``{"type": "avoid", "flags": [...]}``. The candidate's
sentences are never checked and nothing here rates anyone.

A failed check does not stop the session's checks: after
:data:`MAX_FAILURES` failures in a row the tracker rests for
:data:`PAUSE_S` seconds of the conversation, then tries again. Each
cause is said once as a ``notice``: ``avoid_no_rules`` (the RAG has no
interview rules) or ``avoid_unreachable`` (the RAG did not answer).
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
MAX_FAILURES = 3
"""Failed checks in a row before the tracker rests."""
PAUSE_S = 60.0
"""Seconds of conversation the tracker rests after repeated failures."""
NOTICES = {
    "no_rules": "questions are not checked: the RAG has no interview rules",
    "unreachable": "questions are not checked now: the RAG did not answer",
}


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
        self.checks = 0
        self.failures = 0
        self.resting_until: Optional[float] = None
        self._said: Set[str] = set()
        self._texts: Dict[str, str] = {}
        self._checked: Set[str] = set()
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Tell whether the tracker is checking (not resting)."""
        return self.resting_until is None

    def on_event(self, event: LiveEvent) -> None:
        """Check the interviewer's final sentences."""
        u = event.utterance
        if u is None:
            return
        if event.type == "final":
            self._texts[u.id] = u.text
        elif event.type != "speaker_update":
            return
        if self.roles.get(u.speaker) != INTERVIEWER:
            if event.type == "speaker_update":
                self._forget(u.id)
            return
        text = self._texts.get(u.id, "").strip()
        if not text or u.id in self._checked or self._resting(u.start):
            return
        self._checked.add(u.id)
        at = u.start
        self.executor.submit(lambda: self._check(u.id, text, at))

    def _resting(self, at: float) -> bool:
        """Tell whether checks rest at stream time ``at`` (and wake up)."""
        with self._lock:
            if self.resting_until is None:
                return False
            if at >= self.resting_until:
                self.resting_until = None
                self.failures = 0
                return False
            return True

    def _forget(self, utterance: str) -> None:
        """Drop a line's flags: it belongs to someone else now."""
        with self._lock:
            changed = self.ledger.drop(utterance)
            flags = self.ledger.view()
        if changed:
            self.publish({"type": "avoid", "flags": flags})

    def _check(self, utterance: str, text: str, at: float) -> None:
        """Ask the RAG which categories ``text`` touches and keep them."""
        try:
            answer = self.port.check(text, self.language)
        except AvoidUnavailable as exc:
            self._failed(exc, at)
            return
        with self._lock:
            self.failures = 0
            self.checks += 1
            changed = self.ledger.record(
                utterance, at, text, parse_hints(answer)
            )
            flags = self.ledger.view()
        if changed:
            self.publish({"type": "avoid", "flags": flags})

    def _failed(self, exc: AvoidUnavailable, at: float) -> None:
        """Count a failure; rest after several; say each cause once."""
        reason = exc.reason if exc.reason in NOTICES else "unreachable"
        with self._lock:
            self.failures += 1
            if self.failures >= MAX_FAILURES and self.resting_until is None:
                self.resting_until = at + PAUSE_S
            first = reason not in self._said
            self._said.add(reason)
        if first:
            self.publish({
                "type": "notice", "code": f"avoid_{reason}",
                "message": NOTICES[reason],
            })

    def view(self) -> List[Document]:
        """Return the flags as JSON."""
        with self._lock:
            return self.ledger.view()

    def finish(self) -> None:
        """Wait for the checks in flight."""
        self.executor.wait_idle()
