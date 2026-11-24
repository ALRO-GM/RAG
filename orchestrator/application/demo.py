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

A support demo (#250) goes further, still from the file: a suggestion
cites the knowledge chunks it comes from (``retrieved``, copied from the
domain) and carries the customer signals quoted from the sentence it
answers, as a live support call does; the mandatory items of the
product's domain go from pending to said when the sentence the file
names has been said (``{"type": "mandatory"}``); at the end the hand-off
record says what was said and what was left unsaid.
"""

from typing import Callable, Dict, List, Optional

from orchestrator.application.ports import Document, SuggestionSink
from orchestrator.domain.avoid import AvoidLedger
from orchestrator.domain.demo import DemoScript
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.mandatory import Checklist, ChecklistItem
from orchestrator.domain.records import Stages, SuggestionRecord, TriggerInfo
from orchestrator.domain.signals import expressed

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
        self.checklist: Optional[Checklist] = None
        mandatory = script.mandatory
        if mandatory is not None:
            self.checklist = Checklist(
                domain=mandatory.domain, version=mandatory.version,
                remind_seconds=mandatory.remind_seconds,
                items=[ChecklistItem(id=i.id, text=i.text, phase=i.phase)
                       for i in mandatory.items],
            )

    def start(self) -> None:
        """Publish the job's topics and the mandatory items, none done."""
        if self.script.topics:
            self._topics()
        if self.checklist is not None:
            self._checklist()

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
                "about": [u.id], "cites": list(prepared.cites),
            }
            quoted = expressed(
                {"about": [u.id], "signals": prepared.expressed},
                {u.id: (u.speaker, u.text)}, self.roles,
            )
            if quoted:
                suggestion["expressed"] = quoted
            if prepared.example:
                suggestion["example"] = prepared.example
            if prepared.script_phase:
                suggestion["script_phase"] = prepared.script_phase
            if prepared.en:
                suggestion["en"] = dict(prepared.en)
            self.sink.write(SuggestionRecord(
                session_id=self.session_id, call=self.calls,
                request_id=f"{self.session_id}-demo-{self.calls}"[:64],
                trigger=TriggerInfo(
                    utterance=u.id, speaker=u.speaker, role=role,
                    stream_time=event.stream_time, text=u.text,
                ),
                status="ok", suggestions=[suggestion], backend=BACKEND,
                stages=Stages(), retrieved=self._retrieved(prepared.cites),
            ))
        hints = [
            a.model_dump(exclude={"utterance"}, exclude_none=True)
            for a in self.script.avoid if a.utterance == u.id
        ]
        if role == "interviewer" and self.ledger.record(
            u.id, u.start, u.text, hints
        ):
            self.publish({"type": "avoid", "flags": self.ledger.view()})
        self._said(u.id, event.stream_time)
        if any(t.covered_by == u.id for t in self.script.topics):
            self.covered += [t.id for t in self.script.topics
                             if t.covered_by == u.id]
            self._topics()

    def _retrieved(self, cites: List[str]) -> List[Document]:
        """Return the cited chunks as a live call's record lists them."""
        by_id = {s.id: s for s in self.script.sources}
        return [
            {"id": c, "text": by_id[c].text, "score": None,
             "metadata": dict(by_id[c].metadata)}
            for c in cites if c in by_id
        ]

    def _said(self, utterance: str, at: float) -> None:
        """Mark the mandatory items the file says ``utterance`` says."""
        if self.checklist is None or self.script.mandatory is None:
            return
        changed = False
        for item in self.script.mandatory.items:
            if item.said_by == utterance:
                if self.checklist.mark_said(item.id, utterance, 0.0, at):
                    # Prepared, not matched: there is no similarity to show.
                    self.checklist.items[
                        [i.id for i in self.checklist.items].index(item.id)
                    ].score = None
                    changed = True
        if changed:
            self._checklist()

    def _checklist(self) -> None:
        """Publish the mandatory items with what is said so far."""
        if self.checklist is not None:
            self.publish({"type": "mandatory",
                          "checklist": self.checklist.model_dump(mode="json")})

    def _topics(self) -> None:
        """Publish every topic with whether it is covered."""
        self.publish({"type": "topics", "topics": [
            {"id": t.id, "name": t.name, "done": t.id in self.covered,
             **({"en": dict(t.en)} if t.en else {})}
            for t in self.script.topics
        ]})

    def finish(self) -> None:
        """Publish the hand-off record of a demo with mandatory items.

        Nothing runs in the background.
        """
        if self.checklist is not None:
            self.publish({"type": "handoff", "handoff": {
                "session_id": self.session_id, "demo": True,
                "product": self.script.product,
                "mandatory": self.checklist.summary(),
            }})
