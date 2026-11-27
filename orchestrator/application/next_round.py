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

During the round a separate call beside the suggestions
(:class:`~orchestrator.application.live_ticks.LiveTicker`, #268) gets
the pending concepts, those of the topic asked first and a few of the
other topics' (:meth:`TopicTracker.focus_points`: on real speech the
rules often never see which topic is asked), and may say which of them
the last answer treated (:meth:`TopicTracker.suggested`): those are
ticked too, marked ``by: "live_model"``, while the rules' ticks are ``by:
"rules"``. The tracker also keeps every final of the round in order
(:meth:`TopicTracker.round_lines`), so the model can attribute the
whole round after the session.

**Coverage** (#286, :mod:`orchestrator.domain.interview_coverage`).
Beside "treated", each concept records whether its **question was
asked**: an interviewer's line that asks something and names the
concept (``by: "rules"``), the tick call reading a question in other
words (``live_model``) or the interviewer's own mark (``hand``,
:meth:`TopicTracker.mark_asked`). A topic is **named** when either
speaker's line mentions it (:func:`slot_asked`). Each topic carries
its level (none → named → treated → well treated), which only grows,
and the topics message names the **next concept to ask** (``next``,
:func:`~orchestrator.domain.interview_coverage.next_proposal`); the
suggestions' focus line names it too. :meth:`TopicTracker.coverage`
and :meth:`TopicTracker.facts` give the same facts to the report.
"""

import re
import threading
import time
from functools import lru_cache
from typing import (
    Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set,
    Tuple,
)

from orchestrator.application.ports import Document
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.interview_attribution import RoundLine
from orchestrator.domain.interview import (
    Answer,
    CoverageLevel,
    InterviewPlan,
    PointAsked,
    PointCoverage,
    SheetPoint,
    SheetSlot,
    Slot,
    TopicCoverage,
    fold,
)
from orchestrator.domain.interview_agenda import heuristic_fill
from orchestrator.domain.interview_coverage import (
    Proposal,
    TopicFacts,
    TopicRef,
    asked_concepts,
    generic_only,
    next_proposal,
    point_covered,
    topic_coverage,
)
from orchestrator.domain.interview_points import (
    PointHit,
    quote,
    point_hits,
    points_status,
    record_hit,
)
from orchestrator.domain.next_round import TopicState, next_line

INTERVIEWER, CANDIDATE = "interviewer", "candidate"
FOCUS_POINTS = 9
"""Most pending concepts one tick call is given (#268)."""
OTHER_POINTS = 4
"""Of those, most from topics other than the one asked (the rules may
not know which is asked; a short list keeps the reply short)."""
POINT_MARK = "."
"""Joins a slot id and a concept id in the suggester's concept ids."""
_STOP = frozenset(
    "sobre entre desde hasta tienes puedes cuando donde porque tambien "
    "alguna alguno algun cuentame cuenta puesto persona personas "
    "hiciste ejemplo concreto concreta respuesta situacion".split()
)


def keywords(text: str) -> Set[str]:
    """The content words of ``text`` (folded, five letters or more)."""
    return {w for w in re.findall(r"[a-z0-9+#]{5,}", fold(text))
            if w not in _STOP}


@lru_cache(maxsize=512)
def _slot_words(label: str, questions: Tuple[str, ...],
                points: Tuple[str, ...],
                ) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """A slot's label words and all its words (cached: read per final)."""
    own = frozenset(keywords(label))
    return own, own | keywords(" ".join([*questions, *points]))


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
        # A concept made only of generic words («Cuánto tiempo») names
        # no topic (#286 review): any interview's questions say them.
        label, topic = _slot_words(slot.label, tuple(slot.questions),
                                   tuple(p.text for p in slot.points
                                         if not generic_only(p.text)))
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
        on_focus: Told the focus line when it changes.
        clock: Monotonic seconds (dates a mark by hand in stream time).
    """

    def __init__(
        self,
        plan: InterviewPlan,
        states: Sequence[TopicState],
        roles: Dict[str, str],
        publish: Callable[[Document], object],
        language: str = "es",
        on_focus: Optional[Callable[[Optional[str]], None]] = None,
        clock: Callable[[], float] = time.monotonic,
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
        self.lines: Dict[str, Tuple[str, str, float]] = {}
        """Every final of the round: id → (speaker, text, start)."""
        self.named: Set[str] = set()
        """Topics either speaker mentioned (#286)."""
        self.asked_topics: Set[str] = set()
        self.asking: Dict[str, Tuple[float, str]] = {}
        """Interviewer line id → (start, topic) of each line that asked a
        topic (named it, or asked its concepts unambiguously): what a
        speaker relabel recomputes (#286 review)."""
        """Topics the interviewer asked about (#286)."""
        self.asked: Dict[str, Dict[str, PointAsked]] = {
            s.id: {} for s in plan.slots}
        """Per topic, the concepts whose question was asked (#286): the
        mark shown, the first of :attr:`sources` still standing."""
        self.sources: Dict[str, List[PointAsked]] = {}
        """``"<slot>.<concept>"`` → every mark (rules, model, hand), one
        per line (a mark by hand has none); undoing one keeps the rest."""
        self.undone: Dict[str, List[PointAsked]] = {}
        """Marks the interviewer undid; their lines are never marked
        again."""
        self.asked_times: Dict[str, int] = {}
        """``"<slot>.<concept>"`` → questions that asked it."""
        self.answered: Set[str] = set()
        """``"<slot>.<concept>"`` asked and then answered by the
        candidate (the «again» rule)."""
        self.refs = [TopicRef.of(t.slot) for t in self.states]
        self.seen_at: Optional[float] = None
        """When (``clock``) the latest event arrived: a mark by hand is
        dated in stream time from it."""
        self.clock = clock
        self.levels: Dict[str, CoverageLevel] = {}
        """The highest level each topic reached (it never goes down)."""
        self.stream_time = 0.0
        self._newly_asked = False
        self._lock = threading.Lock()
        self._focus_lock = threading.Lock()
        """Serialises telling the focus (#268): the event thread and a
        tick call's thread can both change it."""

    def start(self) -> None:
        """Show the topics in the next round's order."""
        self._publish()

    def on_event(self, event: LiveEvent) -> None:
        """Follow the interviewer's questions and the candidate's answers."""
        u = event.utterance
        if event.type == "speaker_update" and u is not None:
            with self._lock:
                self.seen_at = self.clock()
                changed = False
                if u.id in self.lines:
                    _, text, start = self.lines[u.id]
                    self.lines[u.id] = (u.speaker, text, start)
                    changed = self._relabel(u.id, u.speaker, text, start)
            if changed:
                self._publish()
                self._tell_focus()
            return
        if event.type != "final" or u is None:
            return
        role = self.roles.get(u.speaker, "unknown")
        changed = False
        with self._lock:
            if self._repeated(u.id, u.speaker, u.text, u.start):
                return
            self.lines[u.id] = (u.speaker, u.text, u.start)
            self.stream_time = max(self.stream_time, event.stream_time,
                                   u.end)
            self.seen_at = self.clock()
            named = slot_asked(u.text, self.plan.slots)
            if named is not None and named.id not in self.named:
                self.named.add(named.id)
                changed = True
            if role == INTERVIEWER:
                first = self._asked_points(u.id, u.text, u.start, named)
                asked = named or first
                if asked is not None:
                    changed = changed or self.target is None or (
                        asked.id != self.target.id)
                    self.target = asked
                    self.touched.add(asked.id)
                    self.asked_topics.add(asked.id)
                    self.asking[u.id] = (u.start, asked.id)
                changed = changed or self._newly_asked
                self._newly_asked = False
            elif role == CANDIDATE and self.target is not None:
                before = len(self.answered)
                for key in self.sources:
                    if key.startswith(f"{self.target.id}{POINT_MARK}") \
                            and self.sources[key]:
                        self.answered.add(key)
                # «Otra vez» may be due now (#286 review): publish it.
                changed = changed or len(self.answered) != before
                changed = self._answer(self.target, u.id, u.text) or changed
        if changed:
            self._publish()
        self._tell_focus()

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

    def _asked_points(self, uid: str, text: str, start: float,
                      named: Optional[Slot] = None) -> Optional[Slot]:
        """Mark the concepts an interviewer's line asks (rules, #286).

        Only concepts named inside a question clause, by their
        distinctive words, in one topic: the topic the line names, else
        the best match across topics (a tie goes to the topic being
        asked, else nothing is marked:
        :func:`~orchestrator.domain.interview_coverage.asked_concepts`).
        Returns the topic only when the match may move the topic being
        asked (the line names it, or it was the only best match). Sets
        ``_newly_asked`` when a mark changed. Lock held.
        """
        refs = ([TopicRef.of(named)] if named is not None else self.refs)
        current = (named.id if named is not None
                   else self.target.id if self.target else None)
        match = asked_concepts(text, refs, current)
        if match is None:
            return None
        for pid in match.points:
            mark = PointAsked(by="rules", at=round(start, 3),
                              utterance=uid, quote=quote(text))
            if self._mark(match.topic, pid, mark):
                self._newly_asked = True
        if named is None and not match.clear:
            return None
        return next(s for s in self.plan.slots if s.id == match.topic)

    def _mark(self, slot: str, point: str, mark: PointAsked) -> bool:
        """Record one question about a concept; tell whether it changed.

        Every source is kept, one per line (a mark by hand has none); a
        line the interviewer undid is never marked again. Lock held.
        """
        key = f"{slot}{POINT_MARK}{point}"
        found = self.sources.setdefault(key, [])
        if mark.utterance is not None and (
                any(m.utterance == mark.utterance for m in found)
                or any(m.utterance == mark.utterance
                       for m in self.undone.get(key, []))):
            return False
        found.append(mark)
        self._refresh(slot, point)
        return True

    def _refresh(self, slot: str, point: str) -> None:
        """The mark shown and the count of questions of a concept."""
        key = f"{slot}{POINT_MARK}{point}"
        found = self.sources.get(key, [])
        self.asked_times[key] = len(found)
        if found:
            self.asked[slot][point] = found[0]
        else:
            self.asked[slot].pop(point, None)
            self.answered.discard(key)

    def _forget_line(self, uid: str) -> Set[str]:
        """Drop the marks a line made; return their topics. Lock held."""
        topics: Set[str] = set()
        for key, found in self.sources.items():
            kept = [m for m in found if m.utterance != uid]
            if len(kept) != len(found):
                found[:] = kept
                slot, _, point = key.partition(POINT_MARK)
                self._refresh(slot, point)
                topics.add(slot)
        return topics

    def _relabel(self, uid: str, speaker: str, text: str,
                 start: float) -> bool:
        """A line changed speaker: recompute it as a replay would (#286).

        Its marks, the topic it asked and the topic being asked are
        computed again as if the line had arrived with its new speaker
        (the topic being asked then is the one of the latest asking
        line before it); the levels of the topics it touched are
        computed again from the facts. Lock held.
        """
        affected = self._forget_line(uid)
        old = self.asking.pop(uid, None)
        if old is not None:
            affected.add(old[1])
        if self.roles.get(speaker) == INTERVIEWER:
            earlier = [v for k, v in self.asking.items() if v[0] < start]
            then = max(earlier)[1] if earlier else None
            keep = self.target
            self.target = next((s for s in self.plan.slots if s.id == then),
                               None)
            named = slot_asked(text, self.plan.slots)
            first = self._asked_points(uid, text, start, named)
            self.target = keep
            asked = named or first
            if asked is not None:
                self.asking[uid] = (start, asked.id)
                affected.add(asked.id)
            affected |= {t for t, ps in self.asked.items() if any(
                m.utterance == uid for m in ps.values())}
            self._newly_asked = False
        self.asked_topics = {t for _, t in self.asking.values()}
        latest = max(self.asking.values()) if self.asking else None
        self.target = (next(s for s in self.plan.slots if s.id == latest[1])
                       if latest is not None else None)
        for slot in self.plan.slots:
            if slot.id in affected:
                self.levels[slot.id] = topic_coverage(
                    slot.id, self._facts(slot), "none").level
        return bool(affected) or old is not None

    def seed_undone(self, earlier: Mapping[str, TopicCoverage]) -> None:
        """Take the undone marks of an earlier report of the round (#286).

        Called before a replay, so the rules never mark again a line the
        interviewer undid during the session.
        """
        with self._lock:
            for topic in earlier.values():
                if topic.id not in self.asked:
                    continue
                for point in topic.points:
                    if point.undone:
                        key = f"{topic.id}{POINT_MARK}{point.id}"
                        kept = self.undone.setdefault(key, [])
                        kept.extend(u for u in point.undone
                                    if u not in kept)

    def stream_now(self) -> float:
        """The session's time now: the latest event's plus what passed."""
        with self._lock:
            if self.seen_at is None:
                return self.stream_time
            return self.stream_time + max(self.clock() - self.seen_at, 0.0)

    def mark_asked(self, slot: str, point: str, asked: bool,
                   at: Optional[float] = None,
                   mark: Optional[Tuple[str, Optional[float]]] = None,
                   ) -> Document:
        """The interviewer marks a concept as asked, or undoes a mark.

        ``asked`` true adds a mark by hand when none is shown. ``asked``
        false undoes the mark ``mark`` names (``(by, at)``, as the page
        saw it; ``None``: the mark shown), whoever made it (rules, model
        or hand): it is recorded in ``undone`` and its line is never
        marked again; another source of the same concept, if any, is
        shown next. Both are idempotent: a repeated request (a retry)
        finds nothing to do. An undo is a correction by the person, so
        the topic's level is computed again and may go down (#286
        review). ``at``:
        stream seconds (default: now). Raises ``KeyError`` for a concept
        the plan does not have. Returns ``{"topic", "point", "asked"}``.
        """
        when = round(self.stream_now() if at is None else at, 3)
        with self._lock:
            if point not in {p.id for p in self.points.get(slot, [])}:
                raise KeyError(f"{slot}.{point}")
            key = f"{slot}{POINT_MARK}{point}"
            marks = self.asked[slot]
            changed = False
            if asked and point not in marks:
                changed = self._mark(slot, point,
                                     PointAsked(by="hand", at=when))
            elif not asked and point in marks:
                found_at = next((n for n, m in enumerate(self.sources[key])
                                 if mark is None or (m.by == mark[0] and (
                                     m.at == (None if mark[1] is None
                                              else round(mark[1], 3))))),
                                None)
                if found_at is None:
                    return {"topic": slot, "point": point,
                            "asked": marks[point].model_dump(mode="json")}
                gone = self.sources[key].pop(found_at)
                self.undone.setdefault(key, []).append(gone)
                self._refresh(slot, point)
                target = next(s for s in self.plan.slots if s.id == slot)
                self.levels[slot] = topic_coverage(
                    slot, self._facts(target), "none").level
                changed = True
            found = marks.get(point)
        if changed:
            self._publish()
            self._tell_focus()
        return {"topic": slot, "point": point,
                "asked": found.model_dump(mode="json") if found else None}

    def _repeated(self, uid: str, speaker: str, text: str,
                  start: float) -> bool:
        """Whether a final is speech already kept under another id.

        After a reconnect the same speech can arrive again with a new
        id: same speaker, same words, same start (within 50 ms). The
        same id arriving again replaces its line instead. Lock held.
        """
        return any(other != uid and who == speaker and said == text
                   and abs(at - start) <= 0.05
                   for other, (who, said, at) in self.lines.items())

    def round_lines(self) -> List[RoundLine]:
        """Every final of the round by start time, with its speaker's role.

        A late final takes its place by its start; speech re-sent after a
        reconnect is kept once (:meth:`_repeated`). The role is the one
        set now (roles set late in the session apply
        to the whole round); a speaker without one is ``unknown``.
        """
        with self._lock:
            return [RoundLine(uid, self.roles.get(speaker) or "unknown",
                              text, start)
                    for uid, (speaker, text, start) in sorted(
                        self.lines.items(), key=lambda kv: kv[1][2])]

    def ticked_count(self) -> int:
        """How many concepts of the round are not pending."""
        with self._lock:
            return sum(1 for points in self.points.values()
                       for p in points if p.status != "pending")

    def focus_points(self) -> Document:
        """The pending concepts for the tick call (#268).

        The call says which were treated and, since #286, which a
        question asked (a pending concept may have been asked).

        ``{"label", "points": [{"id": "<slot>.<concept>", "text",
        "topic"}]}``: all of the topic asked (``label``, empty when the
        rules saw none asked), then at most :data:`OTHER_POINTS` of the
        other topics in the round's order (:data:`FOCUS_POINTS` of them
        when no topic is asked), at most :data:`FOCUS_POINTS` in all;
        ``{}`` when no concept is pending.
        """
        with self._lock:
            target = self.target
            order = [t.slot for t in self.states]
            if target is not None:
                order = [target, *(s for s in order if s.id != target.id)]
            found = [{"id": f"{slot.id}{POINT_MARK}{p.id}", "text": p.text,
                      "topic": slot.label}
                     for slot in order
                     for p in self.points.get(slot.id, [])
                     if p.status == "pending"]
        if target is not None:
            mine = [f for f in found
                    if f["id"].startswith(f"{target.id}{POINT_MARK}")]
            found = mine + found[len(mine):len(mine) + OTHER_POINTS]
        if not found:
            return {}
        return {"label": target.label if target is not None else "",
                "points": found[:FOCUS_POINTS]}

    def suggested(self, asked: Document, data: Document) -> None:
        """Tick the concepts the tick call said the answer treated.

        ``asked`` is the request's payload (:meth:`focus_points` when it
        was sent), ``data`` the reply's (``{"points": [{"id",
        "utterance"}]}``). Only a concept the request listed and not
        treated yet, backed by a candidate utterance of this round, is
        ticked; its quote is that utterance's first words. Marked ``by:
        "live_model"``.
        """
        listed = asked.get("points")
        raw = data.get("points")
        if not isinstance(listed, list) or not isinstance(raw, list):
            return
        known = {p.get("id") for p in listed if isinstance(p, dict)}
        changed = False
        with self._lock:
            changed = self._model_asked(known, data.get("asked"))
            for item in raw:
                if not isinstance(item, dict) or item.get("id") not in known:
                    continue
                topic, _, point = str(item["id"]).partition(POINT_MARK)
                said = self.lines.get(str(item.get("utterance")))
                if topic not in self.points or said is None or \
                        self.roles.get(said[0]) != CANDIDATE:
                    continue
                hit = PointHit(point, "treated", quote(said[1]),
                               "live_model")
                if record_hit(self.points[topic], hit,
                              str(item.get("utterance"))):
                    changed = True
                    self.touched.add(topic)
                    if points_status(self.points[topic]) in (
                            "filled", "partial"):
                        self.covered.add(topic)
        if changed:
            self._publish()
        self._tell_focus()

    def _model_asked(self, known: Set[object], raw: object) -> bool:
        """Mark the concepts the tick call says a question asked (#286).

        ``raw``: ``[{"id", "utterance"}]``; only listed concepts, each
        backed by an interviewer line of this round. Lock held.
        """
        changed = False
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or item.get("id") not in known:
                continue
            topic, _, point = str(item["id"]).partition(POINT_MARK)
            uid = str(item.get("utterance"))
            said = self.lines.get(uid)
            if topic not in self.points or said is None or \
                    self.roles.get(said[0]) != INTERVIEWER or \
                    point not in {p.id for p in self.points[topic]}:
                continue
            changed = self._mark(topic, point, PointAsked(
                by="live_model", at=round(said[2], 3), utterance=uid,
                quote=quote(said[1]))) or changed
            self.touched.add(topic)
        return changed

    def _facts(self, slot: Slot) -> TopicFacts:
        """What the round knows of one topic (lock held)."""
        texts = dict(self.answers.get(slot.id, []))
        for p in self.points[slot.id]:
            line = self.lines.get(p.utterance or "")
            if line is not None and p.utterance not in texts and \
                    self.roles.get(line[0]) == CANDIDATE:
                texts[str(p.utterance)] = line[1]
        marks = self.asked[slot.id]
        return TopicFacts(
            named=slot.id in self.named,
            asked=slot.id in self.asked_topics or bool(marks),
            answers=len(texts),
            words=sum(len(t.split()) for t in texts.values()),
            complete=slot.id in self.covered,
            points=[PointCoverage(
                id=p.id, text=p.text, status=p.status,
                asked=marks.get(p.id),
                undone=list(self.undone.get(
                    f"{slot.id}{POINT_MARK}{p.id}", [])),
            ) for p in self.points[slot.id]],
            answered={p.id for p in self.points[slot.id]
                      if f"{slot.id}{POINT_MARK}{p.id}" in self.answered},
        )

    def facts(self) -> Dict[str, TopicFacts]:
        """Each topic's facts now, by slot id (copies)."""
        with self._lock:
            return {s.id: self._facts(s) for s in self.plan.slots}

    def reached(self) -> Dict[str, CoverageLevel]:
        """The highest level each topic reached so far."""
        with self._lock:
            self._levels()
            return dict(self.levels)

    def _levels(self, facts: Optional[Dict[str, TopicFacts]] = None,
                ) -> Dict[str, TopicCoverage]:
        """Each topic's coverage, raising the levels kept (lock held)."""
        found: Dict[str, TopicCoverage] = {}
        for slot in self.plan.slots:
            got = topic_coverage(slot.id, facts[slot.id] if facts
                                 else self._facts(slot),
                                 self.levels.get(slot.id, "none"),
                                 slot.label)
            self.levels[slot.id] = got.level
            found[slot.id] = got
        return found

    def coverage(self) -> List[TopicCoverage]:
        """How far each topic of the plan was covered (plan order)."""
        with self._lock:
            return list(self._levels().values())

    def proposal(self) -> Optional[Proposal]:
        """The next concept to ask about (#286), or ``None``."""
        with self._lock:
            return self._proposal()

    def _proposal(self, facts: Optional[Dict[str, TopicFacts]] = None,
                  ) -> Optional[Proposal]:
        """:func:`next_proposal` on the round's facts (lock held)."""
        facts = facts or {s.id: self._facts(s) for s in self.plan.slots}
        return next_proposal(
            self.refs, facts,
            self.asked_times,
            self.target.id if self.target else None,
            self.stream_time < self.plan.minutes * 60, self.language)

    def _tell_focus(self) -> None:
        """Tell ``on_focus`` the focus line when it changed.

        Under its own lock, computed and told in one step, so two
        threads never tell an older line after a newer one.
        """
        with self._focus_lock:
            with self._lock:
                focus = self._focus()
                moved = focus != self.focus
                self.focus = focus
            if moved and self.on_focus is not None:
                self.on_focus(focus)

    def _focus(self) -> Optional[str]:
        """The suggester's focus: the next concept to ask (lock held).

        ``None`` until a topic is asked, and when nothing is left to
        propose.
        """
        if self.target is None:
            return None
        found = self._proposal()
        if found is None:
            return None
        slot = next(s for s in self.plan.slots if s.id == found.topic)
        marks = self.asked[slot.id]
        rest = [p.text for p in self.points[slot.id]
                if not point_covered(p.status, marks.get(p.id))]
        return next_line(found.topic_name, found.point_text, rest,
                         found.topic == self.target.id, self.language)

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
        """Send the topics with their concepts, levels and what to ask."""
        with self._lock:
            target = self.target.id if self.target else None
            facts = {s.id: self._facts(s) for s in self.plan.slots}
            coverage = self._levels(facts)
            topics = [
                t.to_topic(t.slot.id in self.covered,
                           [p.model_copy() for p in self.points.get(
                               t.slot.id, [])],
                           t.slot.id == target,
                           coverage[t.slot.id], self.asked[t.slot.id])
                for t in self.states
            ]
            found = self._proposal(facts)
        self.publish({"type": "topics", "topics": topics,
                      "next": found.to_message() if found else None})
