"""Use case: play a scripted demo on the live page (LV-17, #232).

:class:`ScriptedAssist` takes the place of the live-assist use case in
a demo session: it follows the replayed events like
:class:`~orchestrator.application.live_assist.LiveAssist` does, but its
suggestions come from the demo file (``live.demo.v1``), not from the
knowledge service and the reasoner, so no model is called and every run
is the same. After each final sentence it publishes, in this order, the
prepared suggestions attached to it (as ordinary suggestion records,
backend ``demo-script``), the hint of a line better avoided through the
same :class:`~orchestrator.domain.avoid.AvoidLedger` as a live session
(so «rephrased» follows the same rule) and the topics it covers
(``{"type": "topics"}``).
"""

from typing import Callable, Dict, List

from orchestrator.application.ports import Document, SuggestionSink
from orchestrator.domain.avoid import AvoidLedger
from orchestrator.domain.demo import DemoScript
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.records import Stages, SuggestionRecord, TriggerInfo

BACKEND = "demo-script"
"""The backend named in a demo's records: prepared, not generated."""


class ScriptedAssist:
    """Publish a demo's prepared suggestions, hints and topics on time.

    Args:
        script: The demo.
        session_id: The session the records belong to.
        roles: Speaker id → role.
        sink: Where the suggestion records go.
        publish: Sends one feed message (hints, topics).
    """

    def __init__(
        self,
        script: DemoScript,
        session_id: str,
        roles: Dict[str, str],
        sink: SuggestionSink,
        publish: Callable[[Document], object],
    ):
        """Keep the script; nothing is published yet."""
        self.script = script
        self.session_id = session_id
        self.roles = dict(roles)
        self.sink = sink
        self.publish = publish
        self.skips: Dict[str, int] = {}
        self.calls = 0
        self.ledger = AvoidLedger()
        self.covered: List[str] = []

    def start(self) -> None:
        """Publish the job's topics, none covered yet."""
        if self.script.topics:
            self._topics()

    def on_event(self, event: LiveEvent) -> None:
        """After a final sentence, publish what the script attaches to it."""
        u = event.utterance
        if event.type != "final" or u is None:
            return
        role = self.roles.get(u.speaker, "unknown")
        for prepared in self.script.suggestions:
            if prepared.after != u.id:
                continue
            self.calls += 1
            suggestion: Document = {
                "kind": prepared.kind, "text": prepared.text,
                "about": [u.id], "cites": [],
            }
            if prepared.example:
                suggestion["example"] = prepared.example
            if prepared.script_phase:
                suggestion["script_phase"] = prepared.script_phase
            self.sink.write(SuggestionRecord(
                session_id=self.session_id, call=self.calls,
                request_id=f"{self.session_id}-demo-{self.calls}"[:64],
                trigger=TriggerInfo(
                    utterance=u.id, speaker=u.speaker, role=role,
                    stream_time=event.stream_time, text=u.text,
                ),
                status="ok", suggestions=[suggestion], backend=BACKEND,
                stages=Stages(),
            ))
        hints = [
            a.model_dump(exclude={"utterance"})
            for a in self.script.avoid if a.utterance == u.id
        ]
        if role == "interviewer" and self.ledger.record(
            u.id, u.start, u.text, hints
        ):
            self.publish({"type": "avoid", "flags": self.ledger.view()})
        if any(t.covered_by == u.id for t in self.script.topics):
            self.covered += [t.id for t in self.script.topics
                             if t.covered_by == u.id]
            self._topics()

    def _topics(self) -> None:
        """Publish every topic with whether it is covered."""
        self.publish({"type": "topics", "topics": [
            {"id": t.id, "name": t.name, "done": t.id in self.covered}
            for t in self.script.topics
        ]})

    def finish(self) -> None:
        """Nothing runs in the background."""
