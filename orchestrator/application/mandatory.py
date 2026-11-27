"""Use case: follow the mandatory items of a session (LV-16).

When a session starts, the product's knowledge domain gives its list
(:meth:`MandatoryTracker.start`); the page shows it as a red checklist.
Every final sentence of the employee (role ``agent`` or
``interviewer``, also when a ``speaker_update`` gives a sentence to the
employee later, as the mono voiceprints do) is checked against the
items still pending, on the tracker's own executor so the conversation
never waits for it; an item said is marked with the sentence and the
similarity. A manual tick needs ``confirm`` and is logged. Each change
is published as ``{"type": "mandatory", "checklist": ...}``. A service
that does not answer leaves the items pending (they can still be
ticked); its failure is published once as a ``warning``.

The check starts with the first final sentence: nothing waits for a
warm-up. Since #286 :meth:`MandatoryTracker.start` also sends one
check whose answer is ignored, on its own thread and only in a live
session, so the RAG's first real check of the session does not pay for
loading its model and the examples' vectors and nothing waits for it.

Since #294 a session may use an owner's list (``chosen``, a
:class:`~orchestrator.application.mandatory_lists.SessionList` read when
the session starts): the general list is still read from the domain
(its threshold and reminder, and its items when the owner's list adds
to it), the items the session checks are the combination, every check
sends that list with the sentence, and the checklist records which list
and version it was.
"""

import threading
from typing import Callable, Dict, Optional, Set

from orchestrator.application.mandatory_lists import SessionList
from orchestrator.application.ports import (
    Document,
    Executor,
    MandatoryPort,
    MandatoryUnavailable,
)
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.mandatory import (
    EMPLOYEE_ROLES,
    Checklist,
    said_items,
)


WARM_TEXT = "Hola."
"""The sentence of the warm-up check at the start (its answer is ignored)."""


class TickRefused(Exception):
    """A manual tick without confirmation, or of no pending item."""


class MandatoryTracker:
    """Keep a session's checklist up to date.

    Args:
        port: The RAG's mandatory items.
        domain: The product's knowledge domain.
        roles: Speaker id → role (the employee's id may be added later).
        publish: Sends one feed message.
        executor: Runs the checks off the event loop.
        warm: Send the warm-up check at the start (a live session only:
            a replay or a rebuild needs none).
        chosen: The list the session uses (#294; ``None``: the domain's,
            not recorded).
    """

    def __init__(
        self,
        port: MandatoryPort,
        domain: str,
        roles: Dict[str, str],
        publish: Callable[[Document], object],
        executor: Executor,
        warm: bool = False,
        chosen: Optional[SessionList] = None,
    ):
        """Keep the ports; the list is read by :meth:`start`."""
        self.warm = warm
        self.chosen = chosen
        self._sent: Optional[Document] = None
        self.port = port
        self.domain = domain
        self.roles = roles
        self.publish = publish
        self.executor = executor
        self.checklist = Checklist(domain=domain)
        self.checks = 0
        self._texts: Dict[str, str] = {}
        self._checked: Set[str] = set()
        self._lock = threading.Lock()
        self._warned = False

    @property
    def active(self) -> bool:
        """Tell whether the domain has mandatory items."""
        return bool(self.checklist.items)

    def start(self) -> None:
        """Read the domain's list and publish it (nothing if it is empty)."""
        try:
            listed = self.port.items(self.domain)
        except MandatoryUnavailable as exc:
            self._warn(str(exc))
            return
        ref = None
        if self.chosen is not None:
            ref = self.chosen.ref(listed)
            if self.chosen.owned is not None:
                listed = self._sent = self.chosen.effective(listed)
        with self._lock:
            self.checklist = Checklist.from_list(listed)
            self.checklist.domain = self.domain
            self.checklist.list = ref
        if self.active:
            self._publish()
            # #286: warm the RAG's check (its model and the examples'
            # vectors, computed once per process) before the employee's
            # first sentence, which is usually the one that says an item.
            # Fire and forget on its own thread: never in the checks'
            # queue, never blocking (an inline executor would wait).
            if self.warm:
                threading.Thread(target=self._warm, daemon=True,
                                 name="mandatory-warm").start()

    def _warm(self) -> None:
        """One check whose answer is ignored (a failure says nothing)."""
        try:
            self._ask(WARM_TEXT)
        except MandatoryUnavailable:
            pass

    def on_event(self, event: LiveEvent) -> None:
        """Check the employee's final sentences."""
        u = event.utterance
        if u is None or not self.active:
            return
        if event.type == "final":
            self._texts[u.id] = u.text
        elif event.type != "speaker_update":
            return
        if self.roles.get(u.speaker) not in EMPLOYEE_ROLES:
            return
        text = self._texts.get(u.id, "").strip()
        if not text or u.id in self._checked or not self.checklist.pending():
            return
        self._checked.add(u.id)
        at = event.stream_time
        self.executor.submit(lambda: self._check(u.id, text, at))

    def _check(self, utterance: str, text: str, at: float) -> None:
        """Ask the RAG which item ``text`` says and mark it."""
        try:
            found = self._ask(text)
        except MandatoryUnavailable as exc:
            self._warn(str(exc))
            return
        changed = False
        with self._lock:
            self.checks += 1
            for hit in said_items(found):
                changed |= self.checklist.mark_said(
                    hit["id"], utterance, hit["score"], at
                )
        if changed:
            self._publish()

    def _ask(self, text: str) -> Document:
        """Check ``text`` on the domain's list, or on the session's."""
        if self._sent is not None:
            return self.port.check_list(self._sent, text)
        return self.port.check(self.domain, text)

    def tick(self, item_id: str, confirm: bool, at: float) -> Document:
        """Tick an item by hand; ``confirm`` must be true."""
        if not confirm:
            raise TickRefused("a manual tick must be confirmed")
        with self._lock:
            if not self.checklist.tick(item_id, at):
                raise TickRefused(f"no pending item {item_id!r}")
        self._publish()
        return self.view()

    def view(self) -> Document:
        """Return the checklist as JSON."""
        with self._lock:
            return self.checklist.model_dump(mode="json")

    def summary(self) -> Optional[Document]:
        """Return the hand-off's account, or ``None`` without items."""
        if not self.active:
            return None
        with self._lock:
            return self.checklist.summary()

    def finish(self) -> None:
        """Wait for the checks in flight."""
        self.executor.wait_idle()

    def _publish(self) -> None:
        self.publish({"type": "mandatory", "checklist": self.view()})

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        self.publish({"type": "notice", "code": "mandatory_unavailable",
                      "message": message[:200]})
