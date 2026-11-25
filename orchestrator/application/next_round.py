"""Use case: the role's topics in a live-page round (#235).

A live-page interview with a role and a candidate shows the plan's slots
as its topics (``{"type": "topics", "topics": [...]}``), in the next
round's order: what earlier rounds left below the level or to revisit
first, then what is still
open, then what is covered (each chip says which with ``earlier``).
During the round a slot is marked done when the interviewer asked about
it (two content words of the slot, or its only one) and the candidate's
answer covers it by the cheap filler of the AI interviewer
(:func:`~orchestrator.domain.interview_agenda.heuristic_fill`,
``complete``); a slot asked but not covered that way is one to
**revisit** in the next round: a prompt to come back to it, not a
judgement of the answer. No model
is called and nothing here rates the person.

A slot with concepts (#258) is followed concept by concept: each
candidate answer to it marks the concepts it names
(:func:`~orchestrator.domain.interview_points.point_hits`: ``treated``,
or ``unknown`` when they say they do not know it), and the slot is done
when none is pending and at least one was treated. The topics message
lists each slot's concepts with their status and marks the topic being
asked (``current``); ``on_focus`` gets the line of the concepts still
to treat of that topic (or ``None``) whenever it changes, for the
suggestions' goal. "Treated" only means the candidate talked about
the concept: a hint for the interviewer, never a judgement.

During the round the existing suggestion call (``suggester.interview@3``,
#268) also gets the pending concepts of the topic asked
(:meth:`TopicTracker.focus_points`) and may say which of them the last
answer treated (:meth:`TopicTracker.suggested`): those are ticked too,
marked ``by: "live_model"``, while the rules' ticks are ``by:
"rules"``. The tracker also keeps every final of the round in order
(:meth:`TopicTracker.round_lines`), so the model can attribute the
whole round after the session.
"""

import re
import threading
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from orchestrator.application.ports import Document
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.interview_attribution import RoundLine
from orchestrator.domain.interview import (
    Answer,
    InterviewPlan,
    SheetPoint,
    SheetSlot,
    Slot,
    fold,
)
from orchestrator.domain.interview_agenda import heuristic_fill
from orchestrator.domain.interview_points import (
    PointHit,
    pending,
    quote,
    point_hits,
    points_status,
    record_hit,
)
from orchestrator.domain.next_round import TopicState, focus_line

INTERVIEWER, CANDIDATE = "interviewer", "candidate"
_STOP = frozenset(
    "sobre entre desde hasta tienes puedes cuando donde porque tambien "
    "alguna alguno algun cuentame cuenta puesto persona personas "
    "hiciste ejemplo concreto concreta respuesta situacion".split()
)


def keywords(text: str) -> Set[str]:
    """The content words of ``text`` (folded, five letters or more)."""
    return {w for w in re.findall(r"[a-z0-9+#]{5,}", fold(text))
            if w not in _STOP}


def slot_asked(text: str, slots: Sequence[Slot]) -> Optional[Slot]:
    """The slot an interviewer's sentence asks about, if one clearly.

    Two shared content words with the slot's label, questions and
    concepts (one when the label has a single content word); the slot
    sharing most wins, ties go to the plan's order.
    """
    heard = keywords(text)
    best: Optional[Slot] = None
    best_n = 0
    for slot in slots:
        label = keywords(slot.label)
        topic = label | keywords(" ".join(
            [*slot.questions, *(p.text for p in slot.points)]))
        shared = len(heard & topic)
        needed = 1 if len(label) == 1 and heard & label else 2
        if shared >= needed and shared > best_n:
            best, best_n = slot, shared
    return best


class TopicTracker:
    """Follow the role's topics through one live-page round.

    Args:
        plan: The role's plan.
        states: The slots with what earlier rounds left of them.
        roles: Speaker id → role (the session's mapping).
        publish: Sends one feed message.
        language: The language of the focus line.
        on_focus: Told the focus line (:func:`focus_line`) when it
            changes (``None``: nothing pending of the topic asked).
    """

    def __init__(
        self,
        plan: InterviewPlan,
        states: Sequence[TopicState],
        roles: Dict[str, str],
        publish: Callable[[Document], object],
        language: str = "es",
        on_focus: Optional[Callable[[Optional[str]], None]] = None,
    ):
        """Nothing asked yet."""
        self.plan = plan
        self.states = list(states)
        self.roles = roles
        self.publish = publish
        self.language = language
        self.on_focus = on_focus
        self.target: Optional[Slot] = None
        self.covered: Set[str] = set()
        self.touched: Set[str] = set()
        self.points: Dict[str, List[SheetPoint]] = {
            s.id: [SheetPoint(id=p.id, text=p.text) for p in s.points]
            for s in plan.slots
        }
        self.focus: Optional[str] = None
        self.answers: Dict[str, List[Tuple[str, str]]] = {}
        """The candidate's answers to each slot asked (utterance, text),
        for the judgement of the round (#259)."""
        self.lines: Dict[str, Tuple[str, str]] = {}
        """Every final of the round in order: id → (speaker, text)."""
        self._lock = threading.Lock()

    def start(self) -> None:
        """Show the topics in the next round's order."""
        self._publish()

    def on_event(self, event: LiveEvent) -> None:
        """Follow the interviewer's questions and the candidate's answers."""
        u = event.utterance
        if event.type == "speaker_update" and u is not None:
            with self._lock:
                if u.id in self.lines:
                    self.lines[u.id] = (u.speaker, self.lines[u.id][1])
            return
        if event.type != "final" or u is None:
            return
        role = self.roles.get(u.speaker, "unknown")
        changed = False
        with self._lock:
            self.lines[u.id] = (u.speaker, u.text)
            if role == INTERVIEWER:
                asked = slot_asked(u.text, self.plan.slots)
                if asked is not None:
                    changed = self.target is None or (
                        asked.id != self.target.id)
                    self.target = asked
                    self.touched.add(asked.id)
            elif role == CANDIDATE and self.target is not None:
                changed = self._answer(self.target, u.id, u.text)
            focus = self._focus()
            moved = focus != self.focus
            self.focus = focus
        if changed:
            self._publish()
        if moved and self.on_focus is not None:
            self.on_focus(focus)

    def _answer(self, slot: Slot, utterance: str, text: str) -> bool:
        """Take a candidate answer to ``slot``; tell whether it changed.

        Called with the lock held.
        """
        kept = self.answers.setdefault(slot.id, [])
        kept[:] = [(u, t) for u, t in kept if u != utterance]
        kept.append((utterance, text))
        states = self.points[slot.id]
        if states:
            changed = False
            for hit in point_hits(text, slot.points):
                self.touched.add(slot.id)
                changed = record_hit(states, hit, utterance) or changed
            if points_status(states) in ("filled", "partial") and \
                    slot.id not in self.covered:
                self.covered.add(slot.id)
            return changed
        for fill in heuristic_fill(text, slot, self.plan.slots):
            if fill.slot != slot.id:
                continue
            self.touched.add(fill.slot)
            if fill.status == "complete" and fill.slot not in self.covered:
                self.covered.add(fill.slot)
                return True
        return False

    def round_lines(self) -> List[RoundLine]:
        """Every final of the round in order, with its speaker's role.

        The role is the one set now (roles set late in the session apply
        to the whole round); a speaker without one is ``unknown``.
        """
        with self._lock:
            return [RoundLine(uid, self.roles.get(speaker) or "unknown",
                              text)
                    for uid, (speaker, text) in self.lines.items()]

    def focus_points(self) -> Document:
        """The pending concepts of the topic asked, for the suggester.

        ``{"topic", "label", "points": [{"id", "text"}]}`` (#268), or
        ``{}`` when no topic is asked or none of its concepts is pending.
        """
        with self._lock:
            target = self.target
            if target is None:
                return {}
            left = pending(self.points[target.id])
            if not left:
                return {}
            return {"topic": target.id, "label": target.label,
                    "points": [{"id": p.id, "text": p.text} for p in left]}

    def suggested(self, asked: Document, data: Document) -> None:
        """Tick the concepts the suggestion call said the answer treated.

        ``asked`` is the request's payload (:meth:`focus_points` when it
        was sent), ``data`` the reply's (``{"topic", "points": [{"id",
        "utterance"}]}``). Only a concept of that topic not treated yet,
        backed by a candidate utterance of this round, is ticked;
        its quote is that utterance's first words. Marked ``by:
        "live_model"``.
        """
        topic = asked.get("topic")
        raw = data.get("points")
        if not isinstance(topic, str) or data.get("topic") != topic or \
                not isinstance(raw, list) or topic not in self.points:
            return
        changed = False
        with self._lock:
            for item in raw:
                if not isinstance(item, dict):
                    continue
                said = self.lines.get(str(item.get("utterance")))
                if said is None or self.roles.get(said[0]) != CANDIDATE:
                    continue
                hit = PointHit(str(item.get("id")), "treated",
                               quote(said[1]), "live_model")
                if record_hit(self.points[topic], hit,
                              str(item.get("utterance"))):
                    changed = True
                    self.touched.add(topic)
            if changed and points_status(self.points[topic]) in (
                    "filled", "partial"):
                self.covered.add(topic)
            focus = self._focus()
            moved = focus != self.focus
            self.focus = focus
        if changed:
            self._publish()
        if moved and self.on_focus is not None:
            self.on_focus(focus)

    def _focus(self) -> Optional[str]:
        """The focus line of the topic being asked (lock held)."""
        if self.target is None:
            return None
        names = [p.text for p in pending(self.points[self.target.id])]
        return focus_line(self.target.label, names, self.language)

    def sheet_slots(self) -> Tuple[List[SheetSlot], Dict[str, str]]:
        """The round's slots as sheet slots, and the whole answers.

        A slot covered is ``filled``, one asked and answered
        ``partial``, the rest ``empty``; its answers and concepts are
        those of this round. No score: the live page never scores.
        """
        with self._lock:
            slots: List[SheetSlot] = []
            texts: Dict[str, str] = {}
            for slot in self.plan.slots:
                said = self.answers.get(slot.id, [])
                texts.update(dict(said))
                status = ("filled" if slot.id in self.covered
                          else "partial" if said else "empty")
                slots.append(SheetSlot(
                    id=slot.id, label=slot.label, level=slot.level,
                    weight=slot.weight, competency=slot.competency,
                    status=status,  # type: ignore[arg-type]
                    answers=[Answer(utterance=u, quote=quote(t),
                                    status="partial") for u, t in said],
                    points=[p.model_copy() for p in self.points[slot.id]],
                ))
        return slots, texts

    def result(self) -> Tuple[List[str], List[str]]:
        """The covered slot ids and those to revisit, of this round."""
        with self._lock:
            order = [s.id for s in self.plan.slots]
            covered = [s for s in order if s in self.covered]
            revisit = [s for s in order
                       if s in self.touched and s not in self.covered]
        return covered, revisit

    def _publish(self) -> None:
        """Send the topics with their concepts and the one being asked."""
        with self._lock:
            target = self.target.id if self.target else None
            topics = [
                t.to_topic(t.slot.id in self.covered,
                           [p.model_copy() for p in self.points.get(
                               t.slot.id, [])],
                           t.slot.id == target)
                for t in self.states
            ]
        self.publish({"type": "topics", "topics": topics})
