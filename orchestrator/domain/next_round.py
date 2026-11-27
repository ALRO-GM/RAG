"""The next round's plan for a candidate (#235).

What earlier rounds left open feeds the next one. Each slot of the role's
plan gets a state from the candidate's rounds (the newest round that
mentions it decides):

* ``weak``: the AI interviewer's result sheet left it below the
  required level (its own scores: a ``minimo`` or ``deseable`` slot
  under 50, not known, skipped or partial);
* ``revisit``: a live-page round asked about it and the answer looked
  incomplete by the word rules: a prompt to come back to it, never a
  rating;
* ``open``: never covered;
* ``covered``: answered at the level (score 70 or more, or filled).

The next round starts from ``weak``, then ``revisit``, then ``open``;
``covered`` slots go last. :func:`round_from_sheet` reads a round of the
AI interviewer from its sheet; a live-page round is read by
:class:`~orchestrator.application.next_round.TopicTracker`. Nothing here
rates the person: the states say what is still to ask.

During a live round, :func:`focus_line` words the concepts of the topic
being asked that are still to be treated (#258) and :func:`with_focus`
adds them to the goal the suggestions get, so the existing suggester
call steers to them without one more model call.

Plain data and functions: no I/O.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from orchestrator.domain.candidate import Round
from orchestrator.domain.interview import (
    InterviewPlan,
    PointAsked,
    SheetPoint,
    Slot,
    TopicCoverage,
)
from orchestrator.domain.interview_points import join_words

Earlier = Literal["weak", "revisit", "open", "covered"]
ORDER: Tuple[Earlier, ...] = ("weak", "revisit", "open", "covered")
GOAL_CHARS = 480
"""Longest goal sent with the suggestions (``transcript.v1`` allows 500)."""
COVERED_SCORE, WEAK_SCORE = 70, 50


@dataclass(frozen=True)
class TopicState:
    """One slot of the plan and what earlier rounds left of it."""

    slot: Slot
    earlier: Earlier

    def to_topic(
        self, done: bool = False,
        points: Optional[Sequence[SheetPoint]] = None,
        current: bool = False,
        coverage: Optional[TopicCoverage] = None,
        asked: Optional[Mapping[str, PointAsked]] = None,
    ) -> Dict[str, Any]:
        """The topic chip of the live page (``{"type": "topics"}``).

        A slot with concepts (#258) lists them with their status in this
        round (``treated``, ``pending``, ``unknown``) and, since #286,
        whether their question was asked (``asked``: ``{"by", "at"}``
        or ``null``); ``current`` marks the topic being asked;
        ``coverage`` (#286) its level and counts.
        """
        topic: Dict[str, Any] = {
            "id": self.slot.id, "name": self.slot.label, "done": done,
            "earlier": self.earlier, "level": self.slot.level,
        }
        marks = asked or {}
        if points:
            topic["points"] = [
                {"id": p.id, "text": p.text, "status": p.status,
                 **({"asked": _asked(marks.get(p.id))}
                    if coverage is not None else {})}
                for p in points]
        if current:
            topic["current"] = True
        if coverage is not None:
            topic["coverage"] = {
                "level": coverage.level, "concepts": coverage.concepts,
                "asked": coverage.asked, "treated": coverage.treated,
                "covered": coverage.covered}
        return topic


def _asked(mark: Optional[PointAsked]) -> Optional[Dict[str, Any]]:
    """A concept's asked mark on the feed: who and when (no words)."""
    if mark is None:
        return None
    return {"by": mark.by, "at": mark.at}


def round_from_sheet(
    sheet: Mapping[str, Any]
) -> Tuple[List[str], List[str]]:
    """Return the covered and the weak slot ids of an interview's sheet."""
    covered: List[str] = []
    weak: List[str] = []
    for slot in sheet.get("slots") or []:
        if not isinstance(slot, Mapping) or not slot.get("id"):
            continue
        score = slot.get("score")
        status = slot.get("status")
        if (isinstance(score, int) and score >= COVERED_SCORE) or (
                score is None and status == "filled"):
            covered.append(str(slot["id"]))
        elif slot.get("level") != "opcional" and (
                (isinstance(score, int) and score < WEAK_SCORE)
                or status in ("unknown", "skipped", "partial")):
            weak.append(str(slot["id"]))
    return covered, weak


def topic_states(
    plan: InterviewPlan, rounds: Sequence[Round]
) -> List[TopicState]:
    """Each slot with its state, weak first, then open, then covered.

    Inside each state the plan's own order holds (``minimo`` first).
    """
    states: Dict[str, Earlier] = {}
    for kept in rounds:
        for sid in kept.covered:
            states[sid] = "covered"
        for sid in kept.revisit:
            states[sid] = "revisit"
        for sid in kept.weak:
            states[sid] = "weak"
    found = [TopicState(s, states.get(s.id, "open")) for s in plan.ordered()]
    return sorted(found, key=lambda t: ORDER.index(t.earlier))


def reordered(plan: InterviewPlan, states: Sequence[TopicState]
              ) -> InterviewPlan:
    """The plan with its slots in the next round's order (weak first).

    Levels are kept, so a ``minimo`` slot still comes before a
    ``deseable`` one, but inside a level what is weak or open leads.
    """
    order = [t.slot.id for t in states]
    slots = sorted(plan.slots, key=lambda s: order.index(s.id)
                   if s.id in order else len(order))
    return plan.model_copy(update={"slots": slots})


def focus_goal(
    title: str, states: Sequence[TopicState], rounds: int,
    language: str = "es",
) -> Optional[str]:
    """The goal of a later round for the suggestions, or ``None``.

    ``None`` for a first round, or when nothing is weak or open.
    """
    if rounds < 1:
        return None
    weak = [t.slot.label for t in states if t.earlier == "weak"]
    again = [t.slot.label for t in states if t.earlier == "revisit"]
    still = [t.slot.label for t in states if t.earlier == "open"]
    if not weak and not again and not still:
        return None
    if language == "en":
        parts = [f"Round {rounds + 1} for the role «{title}». Start with "
                 "what earlier rounds left:"]
        if weak:
            parts.append("below the required level in the AI interview: "
                         + "; ".join(weak))
        if again:
            parts.append("to go back to (answer looked incomplete): "
                         + "; ".join(again))
        if still:
            parts.append("not covered yet: " + "; ".join(still))
    else:
        parts = [f"Ronda {rounds + 1} del puesto «{title}». Empieza por lo "
                 "que dejaron las rondas anteriores:"]
        if weak:
            parts.append("por debajo del nivel en la entrevista de IA: "
                         + "; ".join(weak))
        if again:
            parts.append("volver a tratar (la respuesta pareció "
                         "incompleta): " + "; ".join(again))
        if still:
            parts.append("sin tratar aún: " + "; ".join(still))
    goal = " ".join([parts[0], ". ".join(parts[1:]) + "."])
    if len(goal) > GOAL_CHARS:
        goal = goal[:GOAL_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return goal


def focus_line(
    label: str, pending: Sequence[str], language: str = "es"
) -> Optional[str]:
    """The topic being asked and its concepts still to treat, or ``None``.

    ``None`` when nothing of it is pending (no concepts, or all treated).
    """
    if not pending:
        return None
    if language[:2] == "en":
        return (f"Topic being asked: «{label}»; concepts not treated yet: "
                f"{join_words(pending, language)}.")
    return (f"Tema en curso: «{label}»; conceptos aún sin tratar: "
            f"{join_words(pending, language)}.")


def next_line(
    label: str, point: Optional[str], rest: Sequence[str],
    current: bool, language: str = "es",
) -> str:
    """The suggester's focus on the next concept to ask (#286).

    ``point``: the concept proposed (``None``: the topic itself);
    ``rest``: the topic's other concepts not covered yet; ``current``:
    the topic is the one being asked (else it is the next one).
    """
    others = [r for r in rest if r != point]
    if language[:2] == "en":
        head = (f"Topic being asked: «{label}»" if current
                else f"Next topic: «{label}»")
        line = (f"{head}; next concept to ask about: «{point}»" if point
                else f"{head}; ask about it")
        tail = (f"; then: {join_words(others, language)}" if others
                else "")
        return f"{line}{tail}."
    head = (f"Tema en curso: «{label}»" if current
            else f"Tema siguiente: «{label}»")
    line = (f"{head}; siguiente concepto por preguntar: «{point}»" if point
            else f"{head}; pregunta por él")
    tail = (f"; después: {join_words(others, language)}" if others else "")
    return f"{line}{tail}."


def with_focus(goal: Optional[str], line: Optional[str]) -> Optional[str]:
    """The round's goal followed by the focus line, within the limit.

    The focus line is kept whole; the goal is cut first when both do
    not fit in :data:`GOAL_CHARS`.
    """
    if not line:
        return goal
    line = line[:GOAL_CHARS]
    if not goal:
        return line
    room = GOAL_CHARS - len(line) - 1
    if room < 20:
        return line
    if len(goal) > room:
        goal = goal[:room - 1].rsplit(" ", 1)[0] + "…"
    return f"{goal} {line}"
