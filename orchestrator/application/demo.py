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
(``{"type": "topics"}``), with the concepts of each topic ticked by
their own sentences and the topic being asked marked ``current`` (#258);
since #286 each concept is also marked asked when the interviewer's
sentence the file names for it is said, each topic carries its level by
the same rule as a live round (:func:`~orchestrator.domain.
interview_coverage.topic_level`) and the message names the next concept
to ask (``next``).

A support demo (#250) goes further, still from the file: a suggestion
cites the knowledge chunks it comes from (``retrieved``, copied from the
domain) and carries the customer signals quoted from the sentence it
answers, as a live support call does; the mandatory items of the
product's domain go from pending to said when the sentence the file
names has been said (``{"type": "mandatory"}``); at the end the hand-off
record says what was said and what was left unsaid.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

from orchestrator.application.ports import Document, SuggestionSink
from orchestrator.domain.avoid import AvoidLedger
from orchestrator.domain.demo import DemoScript, DemoTopic
from orchestrator.domain.interview import (
    CoverageLevel,
    PointAsked,
    PointCoverage,
)
from orchestrator.domain.interview_coverage import (
    TopicFacts,
    TopicRef,
    next_proposal,
    topic_coverage,
    without_words,
)
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
        self.said: Set[str] = set()
        self.current: Optional[str] = None
        self.lines: Dict[str, Tuple[str, float]] = {}
        """Every final so far: id → (words, start)."""
        self.levels: Dict[str, CoverageLevel] = {}
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
        self.lines[u.id] = (u.text, u.start)
        named = [t for t in self.script.topics if u.id in t.sentences()]
        if named:
            self.said.add(u.id)
            self.covered += [t.id for t in named if t.covered_by == u.id]
            # The topic asked becomes the current one; a topic without
            # a question sentence becomes it when it is covered.
            moved = [t.id for t in named if t.asked_by == u.id or (
                t.asked_by is None and t.covered_by == u.id)]
            if moved:
                self.current = moved[0]
            self._topics()

    def _retrieved(self, cites: List[str]) -> List[Document]:
        """Return the cited chunks as a live call's record lists them."""
        by_id = {s.id: s for s in self.script.sources}
        return [
            {"id": c, "text": by_id[c].text, "score": None,
             "metadata": {**by_id[c].metadata, "title": by_id[c].title}}
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

    def _facts(self, t: DemoTopic) -> TopicFacts:
        """The topic's facts from the sentences said so far (#286)."""
        points = []
        for p in t.points:
            line = self.lines.get(p.asked_by or "")
            asked = (PointAsked(by="rules", at=line[1], utterance=p.asked_by)
                     if line is not None and p.asked_by in self.said
                     else None)
            points.append(PointCoverage(
                id=p.id, status="treated" if p.covered_by in self.said
                else "pending", asked=asked))
        answers = {x for x in [t.covered_by, *(p.covered_by
                                               for p in t.points)]
                   if x and x in self.said}
        return TopicFacts(
            named=any(x in self.said for x in t.sentences()),
            asked=(t.asked_by in self.said if t.asked_by
                   else t.covered_by in self.said),
            answers=len(answers),
            words=sum(len(self.lines[x][0].split()) for x in answers
                      if x in self.lines),
            complete=t.id in self.covered, points=points)

    def _topics(self) -> None:
        """Publish every topic with whether it is covered.

        A topic with concepts lists them, each ``treated`` once its
        sentence was said (``pending`` until then) and ``asked`` once
        its question was (#286); each topic has its level, and ``next``
        names the next concept to ask.
        """
        topics: List[Document] = []
        facts: Dict[str, TopicFacts] = {}
        for t in self.script.topics:
            facts[t.id] = got = self._facts(t)
            coverage = topic_coverage(t.id, got,
                                      self.levels.get(t.id, "none"))
            self.levels[t.id] = coverage.level
            topic: Document = {"id": t.id, "name": t.name,
                               "done": t.id in self.covered,
                               **({"en": dict(t.en)} if t.en else {}),
                               "coverage": {
                                   "level": coverage.level,
                                   "concepts": coverage.concepts,
                                   "asked": coverage.asked,
                                   "treated": coverage.treated,
                                   "covered": coverage.covered}}
            if t.points:
                topic["points"] = [
                    {"id": p.id, "text": p.text, "status": c.status,
                     "asked": ({"by": c.asked.by, "at": c.asked.at}
                               if c.asked else None),
                     **({"en": dict(p.en)} if p.en else {})}
                    for p, c in zip(t.points, got.points)
                ]
            if t.id == self.current:
                topic["current"] = True
            topics.append(topic)
        found = next_proposal(
            [TopicRef(t.id, t.name, tuple((p.id, p.text) for p in t.points))
             for t in self.script.topics], facts,
            {f"{t.id}.{p.id}": 1 for t in self.script.topics
             for p in t.points if p.asked_by in self.said},
            self.current)
        self.publish({"type": "topics", "topics": topics,
                      "next": found.to_message() if found else None})

    def finish(self) -> None:
        """Publish the hand-off record of a demo with mandatory items.

        Nothing runs in the background.
        """
        if self.checklist is not None:
            self.publish({"type": "handoff", "handoff": {
                "session_id": self.session_id, "demo": True,
                "product": self.script.product,
                "mandatory": self.checklist.summary(),
                # #286: the topics' levels and concepts asked, as a live
                # round's hand-off (ids and times, no words).
                **({"coverage": without_words([
                    topic_coverage(t.id, self._facts(t),
                                   self.levels.get(t.id, "none"), t.name)
                    for t in self.script.topics])}
                   if self.script.topics else {}),
            }})
