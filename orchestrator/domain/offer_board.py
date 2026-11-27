"""One comparable figure per interview; the order of an offer's people.

The offer page («Ofertas» → an offer, #271) lists the people interviewed from
one interview plan, one row each, **ordered by a draft figure** a person
reviews. The AI interviewer's sheets already carry a 0–100 score per
point (the model's, capped by the accuracy judgement, #259; or the
coarse estimate from the slot's status when the model failed, #267) and
their weighted ``suitability``; the live page's reports have the
judgement of the answers but never a score. This module gives **both**
kinds one figure on the same 0–100 scale, computed on read from the
stored sheet and the **current** plan:

* **Per point** (:func:`point_result`):

  - AI interview: the point's stored ``score`` (basis ``score``, or
    ``status`` when it is the status estimate);
  - live interview (basis ``labels``): the value of the point's
    judgement (:data:`LABEL_VALUE`: ``solid`` 100, ``correct_basic``
    70, ``to_verify`` 50, ``unclear`` 40, ``wrong`` 0, ``off_topic``
    5; without a judgement of the point, the mean value of its judged
    concepts, else 40 when it was answered) times ``0.5 + 0.5 ×`` the
    share of its concepts treated (1 when it has none), and at most
    :data:`~orchestrator.domain.interview_judgement.WRONG_POINT_CAP`
    when a treated concept was judged ``wrong``. A point the candidate
    said they do not know, or skipped, is 10; a point not treated is
    10 when it is required and left out otherwise.

* **Per interview** (:func:`interview_view`): the plan's weighted mean,
  Σ weight·value / Σ weight, over the points of the current plan that
  the interview asked, with the current plan's weights and levels; a
  required (``minimo``) point without a value counts 0, any other is
  left out (the rule of :func:`~orchestrator.domain.interview.suitability`).
  For an AI sheet whose plan was not edited since, this **is** its
  stored ``suitability``. A point is «no preguntado» (left out of the
  figure) only when it was not in the plan revision the interview was
  made with (the report's ``plan_slots``; an AI sheet keeps every slot
  of its revision); a point of that revision the report has no answer
  to is not treated. An older live report without ``plan_slots``
  never has «no preguntado» points: a point added to the plan after it
  counts as not treated there (10 when required). So saving the plan
  (a new revision) never changes a figure unless a weight or a level
  changed. Points removed from the plan drop out; ids are never reused
  (``retired_slots``), so a new point never inherits a removed one's
  answers.

* **Order** (:func:`ordered`): figure, highest first; on the same figure,
  fewer required points without evidence of coverage (not covered, or
  not asked because they were added later: each counts one), then the
  latest interview, then the name. So a person missing a required point
  never comes before someone who covers them all with the same or a
  better figure. Rows without a figure go last. Nobody is left out or
  cut off.

A point is **covered** when the answer was judged ``solid``,
``correct_basic``, ``to_verify`` or ``unclear``, or answered without a
judgement; not when it was ``wrong``, ``off_topic``, not known, skipped,
not treated or not asked.

**Coverage** (#286). A live report with ``coverage`` gives each point
its level reached (``coverage``: none, named, treated, well_treated) and
how many of its concepts the interviewer asked (``asked``), so the page
tells «se preguntó, sin respuesta» from «no se llegó a preguntar» (the
interviewer's gap, not the candidate's). The **figure does not change**:
it rests on the candidate's answers and their judgement, and a concept
asked but not answered was not treated either way; ``None`` on both
fields for AI sheets and reports written before it.

Live reports whose answers were attributed to the points by the word
rules (#268: ``attribution`` ``rules``, or none: written before it) are
marked **less reliable**.

A draft from a model and short interviews, for a person to review:
never a decision, never shown to the candidate. Plain data: no I/O.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import (
    Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple,
)

from pydantic import ValidationError

from orchestrator.domain.interview import (
    InterviewPlan,
    SheetSlot,
    TopicCoverage,
    fold,
)
from orchestrator.domain.interview_judgement import (
    COUNT_KEYS,
    WRONG_POINT_CAP,
    accuracy_counts,
)

PointState = Literal[
    "solid", "correct_basic", "to_verify", "unclear", "answered", "wrong",
    "off_topic", "does_not_know", "skipped", "not_treated", "not_asked",
]
"""What the table shows for one point of one interview."""
POINT_STATES: Tuple[PointState, ...] = (
    "solid", "correct_basic", "to_verify", "unclear", "answered", "wrong",
    "off_topic", "does_not_know", "skipped", "not_treated", "not_asked",
)
COVERED: Tuple[PointState, ...] = (
    "solid", "correct_basic", "to_verify", "unclear", "answered",
)
"""States that count as covering a point."""
LABEL_VALUE: Dict[str, int] = {
    "solid": 100, "correct_basic": 70, "to_verify": 50, "unclear": 40,
    "answered": 40, "does_not_know": 10, "skipped": 10, "not_treated": 10,
    "off_topic": 5, "wrong": 0,
}
"""Value (0–100) of a live point from its judgement or state."""
Basis = Literal["score", "status", "labels"]
FigureBasis = Literal["scores", "status", "labels"]
"""How an interview's figure was computed: the AI interviewer's model
scores, its coarse status estimate (the model failed), or the live
judgement labels."""
QUOTE_CHARS = 300
MAX_REPORTS = 500
"""Stored reports one board reads at most (the route stays bounded)."""


@dataclass(frozen=True)
class PointResult:
    """One point of the plan in one interview.

    ``value`` (0–100) feeds the figure; ``None``: no value (not treated
    or not asked). ``score`` is the AI interviewer's own score, when
    there is one. ``quote`` and ``reason`` are the evidence of the
    point's judgement (the point's own, or its first judged concept).
    """

    id: str
    state: PointState
    value: Optional[float] = None
    basis: Optional[Basis] = None
    score: Optional[int] = None
    label: Optional[str] = None
    treated: int = 0
    concepts: int = 0
    quote: str = ""
    reason: str = ""
    utterance: Optional[str] = None
    asked: Optional[int] = None
    """Concepts of the point the interviewer asked (#286); ``None``: the
    report does not say."""
    coverage: Optional[str] = None
    """The level the point reached in a live round (#286), or ``None``."""


@dataclass(frozen=True)
class InterviewView:
    """One stored interview of an offer, read against the current plan."""

    session_id: str
    kind: Literal["ai", "live"]
    finished_at: Optional[str]
    candidate: Optional[str]
    candidate_id: Optional[str]
    plan_revision: int
    plan_changed: bool
    figure: Optional[float]
    figure_basis: FigureBasis
    stored_suitability: Optional[float]
    points: List[PointResult]
    missing_required: List[str]
    unasked_required: List[str]
    not_asked: List[str]
    accuracy: Dict[str, int]
    to_check: int
    less_reliable: bool
    attribution: Optional[str]
    owned: bool = False

    @property
    def covers_required(self) -> bool:
        """Whether every required point of the current plan is covered."""
        return not self.missing_required and not self.unasked_required


def _slot(raw: object) -> Optional[SheetSlot]:
    """A stored slot as a :class:`SheetSlot` (``None`` when unreadable)."""
    if not isinstance(raw, dict):
        return None
    try:
        return SheetSlot.model_validate(raw)
    except ValidationError:
        return None


def _majority(labels: Sequence[str]) -> Optional[str]:
    """The most frequent label; a tie goes to the lower value."""
    if not labels:
        return None
    return min(set(labels), key=lambda x: (-labels.count(x),
                                           LABEL_VALUE.get(x, 0), x))


def _state(slot: SheetSlot) -> PointState:
    """The state the table shows for an asked point."""
    if slot.judgement is not None:
        return slot.judgement.label
    judged = [p.judgement.label for p in slot.points
              if p.judgement is not None]
    if judged:
        return _majority(judged)  # type: ignore[return-value]
    if slot.status in ("filled", "partial") or any(
            p.status == "treated" for p in slot.points):
        return "answered"
    if slot.status == "unknown":
        return "does_not_know"
    if slot.status == "skipped":
        return "skipped"
    return "not_treated"


def _evidence(slot: SheetSlot) -> Tuple[str, str, Optional[str]]:
    """Quote, reason and utterance of the point's judgement."""
    found = slot.judgement or next(
        (p.judgement for p in slot.points if p.judgement is not None), None)
    if found is not None:
        return found.quote[:QUOTE_CHARS], found.reason, found.utterance
    if slot.answers:
        first = slot.answers[0]
        return first.quote[:QUOTE_CHARS], "", first.utterance
    return "", "", None


def live_value(slot: SheetSlot, required: bool) -> Optional[float]:
    """The value (0–100) of a live point from its judgement and concepts."""
    state = _state(slot)
    if state == "not_treated":
        return float(LABEL_VALUE["not_treated"]) if required else None
    if state in ("does_not_know", "skipped"):
        return float(LABEL_VALUE[state])
    if slot.judgement is not None:
        base = float(LABEL_VALUE[slot.judgement.label])
    else:
        judged = [LABEL_VALUE[p.judgement.label] for p in slot.points
                  if p.judgement is not None]
        base = (sum(judged) / len(judged) if judged
                else float(LABEL_VALUE["answered"]))
    share = 1.0
    if slot.points:
        share = sum(1 for p in slot.points if p.status == "treated") / len(
            slot.points)
    value = base * (0.5 + 0.5 * share)
    if any(p.judgement is not None and p.judgement.label == "wrong"
           for p in slot.points):
        value = min(value, float(WRONG_POINT_CAP))
    return round(value, 1)


def point_result(
    slot_id: str, slot: Optional[SheetSlot], kind: str, required: bool,
) -> PointResult:
    """One point of the current plan in one interview (#271).

    ``slot`` is the interview's slot of that id (``None``: the interview
    did not ask it, e.g. the point was added to the plan later).
    """
    if slot is None:
        return PointResult(id=slot_id, state="not_asked")
    quote, reason, utterance = _evidence(slot)
    treated = sum(1 for p in slot.points if p.status == "treated")
    common: Dict[str, Any] = {
        "id": slot_id, "state": _state(slot), "score": slot.score,
        "label": slot.judgement.label if slot.judgement else None,
        "treated": treated, "concepts": len(slot.points), "quote": quote,
        "reason": reason, "utterance": utterance,
    }
    if kind == "live":
        return PointResult(**common, value=live_value(slot, required),
                           basis="labels")
    if slot.score is None:
        return PointResult(**common)
    return PointResult(
        **common, value=float(slot.score),
        basis="status" if slot.scored_by == "status" else "score")


def weighted(
    plan: InterviewPlan, points: Sequence[PointResult],
) -> Optional[float]:
    """Σ weight·value / Σ weight over the asked points of the plan.

    A required point without a value counts 0, any other point without
    a value is left out, a point not asked is left out (rounded to one
    decimal; ``None`` when nothing counts).
    """
    by_id = {p.id: p for p in points}
    total = weights = 0.0
    for slot in plan.slots:
        found = by_id.get(slot.id)
        if found is None or found.state == "not_asked":
            continue
        if found.value is None and slot.level != "minimo":
            continue
        total += slot.weight * (found.value or 0.0)
        weights += slot.weight
    return round(total / weights, 1) if weights else None


def _accuracy(slots: Sequence[SheetSlot]) -> Dict[str, int]:
    """The interview's judgement labels and concept states, summed."""
    found = {k: 0 for k in COUNT_KEYS}
    for slot in slots:
        counted = slot.accuracy or accuracy_counts(slot)
        for key, n in counted.items():
            if key in found and isinstance(n, int):
                found[key] += n
    return {k: v for k, v in found.items() if v}


def _int(value: object) -> int:
    """A stored revision as an int (0 when unknown)."""
    return value if isinstance(value, int) and not isinstance(
        value, bool) else 0


def _text(value: object) -> Optional[str]:
    """A stored text field, or ``None``."""
    return value if isinstance(value, str) and value else None


def _revision_slots(
    sheet: Mapping[str, Any], kind: str,
) -> Optional[Set[str]]:
    """The slot ids of the plan revision the interview was made with.

    The report's ``plan_slots`` when it has them; else, for an AI sheet
    (which keeps every slot of its plan), the slots it has; else
    ``None``: unknown (an older live report keeps only the slots it
    judged), so a missing point counts as not treated, never as added
    later.
    """
    listed = sheet.get("plan_slots")
    if isinstance(listed, list):
        return {x for x in listed if isinstance(x, str)}
    raw = sheet.get("slots")
    ids = {str(item.get("id")) for item in (raw if isinstance(raw, list)
                                            else [])
           if isinstance(item, dict) and item.get("id")}
    if kind == "ai" and ids:
        # Every stored slot, readable or not: an unreadable one was asked.
        return ids
    return None


def _coverage(sheet: Mapping[str, Any]) -> Dict[str, TopicCoverage]:
    """A live report's coverage by point id (#286; empty when none)."""
    raw = sheet.get("coverage")
    found: Dict[str, TopicCoverage] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            topic = TopicCoverage.model_validate(item)
        except ValidationError:
            continue
        found.setdefault(topic.id, topic)
    return found


def interview_view(
    plan: InterviewPlan, sheet: Mapping[str, Any], owned: bool = False,
) -> InterviewView:
    """Read one stored interview against the current plan (#271)."""
    kind: Literal["ai", "live"] = (
        "live" if sheet.get("kind") == "live" else "ai")
    raw = sheet.get("slots")
    slots: Dict[str, SheetSlot] = {}
    for item in raw if isinstance(raw, list) else []:
        found = _slot(item)
        if found is not None and found.id not in slots:
            slots[found.id] = found
    revision = _int(sheet.get("plan_revision"))
    changed = revision != plan.revision
    known = _revision_slots(sheet, kind)
    covered = _coverage(sheet)
    points: List[PointResult] = []
    for s in plan.slots:
        result = point_result(s.id, slots.get(s.id), kind,
                              s.level == "minimo")
        if result.state == "not_asked" and (known is None or s.id in known):
            # In the plan the interview was made with, but the report has
            # no answer to it: a live round that never reached it.
            result = PointResult(
                id=s.id, state="not_treated",
                value=(float(LABEL_VALUE["not_treated"])
                       if kind == "live" and s.level == "minimo" else None))
        if s.id in covered:
            result = replace(result, asked=covered[s.id].asked,
                             coverage=covered[s.id].level)
        points.append(result)
    required = {s.id for s in plan.slots if s.level == "minimo"}
    used = [slots[s.id] for s in plan.slots if s.id in slots]
    if kind == "live":
        basis: FigureBasis = "labels"
    elif sheet.get("report_error") or any(
            s.scored_by == "status" for s in used):
        basis = "status"
    else:
        basis = "scores"
    attribution = _text(sheet.get("attribution"))
    checks = sheet.get("to_check")
    stored = sheet.get("suitability")
    return InterviewView(
        session_id=str(sheet.get("session_id") or ""),
        kind=kind, finished_at=_text(sheet.get("finished_at")),
        candidate=_text(sheet.get("candidate")),
        candidate_id=_text(sheet.get("candidate_id")),
        plan_revision=revision, plan_changed=changed,
        figure=weighted(plan, points), figure_basis=basis,
        stored_suitability=(float(stored) if isinstance(stored, (int, float))
                            and not isinstance(stored, bool) else None),
        points=points,
        missing_required=[p.id for p in points
                          if p.id in required and p.state not in COVERED
                          and p.state != "not_asked"],
        unasked_required=[p.id for p in points if p.id in required
                          and p.state == "not_asked"],
        not_asked=[p.id for p in points if p.state == "not_asked"],
        accuracy=_accuracy(used),
        to_check=len(checks) if isinstance(checks, list) else 0,
        less_reliable=kind == "live" and attribution != "after_session",
        attribution=attribution, owned=owned,
    )


def _when(view: InterviewView) -> float:
    """The interview's end as a timestamp (0 when unknown)."""
    if not view.finished_at:
        return 0.0
    try:
        return datetime.fromisoformat(view.finished_at).timestamp()
    except ValueError:
        return 0.0


def order_key(view: InterviewView) -> Tuple[Any, ...]:
    """The default order of the offer page (best figure first, #271).

    Figure, highest first (none last); on the same figure, fewer
    required points without evidence of coverage (not covered, or not
    asked because they were added to the plan later: both count one);
    then the latest interview; then the name and the session id, so the
    order is total and stable.
    """
    return (view.figure is None, -(view.figure or 0.0),
            len(view.missing_required) + len(view.unasked_required),
            -_when(view),
            fold(view.candidate or ""), view.session_id)


def ordered(views: Sequence[InterviewView]) -> List[InterviewView]:
    """The views in the default order of :func:`order_key`."""
    return sorted(views, key=order_key)


def latest_first(views: Sequence[InterviewView]) -> List[InterviewView]:
    """One person's interviews, the latest first."""
    return sorted(views, key=lambda v: (-_when(v), v.session_id))
