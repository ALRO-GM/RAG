"""Which utterances answer each point of a round, read from the model (#268).

On real speech the word rules of the live page
(:mod:`orchestrator.domain.interview_points`, the topic tracker)
attribute few answers and tick few concepts, so after a live round the
model reads the **whole round** (the interviewer's questions and the
candidate's answers in order) with the plan's points and concepts and
says, per point, which candidate utterances answer it and which
concepts they treated (template ``interview.attribution@1``). This
module cuts the round into batches that fit that template, reads each
reply defensively and merges them:

* only ids of the **candidate's** utterances of that batch count (an
  interviewer's line never answers a point; unknown ids are dropped);
* a concept's quote is always the candidate's own words: the model's
  quote when it is in the utterance it names, else that utterance's
  first words; without a known utterance, a quote of at least
  :data:`~orchestrator.domain.interview_judgement.MIN_QUOTE_WORDS`
  words names the candidate utterance that contains it, otherwise the
  concept is not ticked;
* unknown point and concept ids and statuses are dropped; a point
  listed with no valid answer and no concept is left out;
* across batches the answers are joined (in round order) and a
  concept keeps its strongest status (``treated`` over ``unknown``),
  as :func:`~orchestrator.domain.interview_points.record_hit` does;
* since #286 (template ``interview.attribution@2``) a point may also
  list the concepts the **interviewer asked** (``asked``: concept id
  and the interviewer's utterance id): only an interviewer's line of
  that batch counts, the earliest one is kept, and
  :func:`asked_marks` turns them into asked marks ``by:
  "after_session"``.

"Treated" only means the candidate talked about it; how accurate it
was is the judge's work (``interview.report@2``), and nothing here is
ever shown or said to the candidate. Plain data and functions: no I/O.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from orchestrator.domain.interview import (
    QUOTE_CHARS,
    Answer,
    InterviewPlan,
    PointAsked,
    PointStatus,
    SheetPoint,
    SheetSlot,
    fold,
)
from orchestrator.domain.interview_judgement import MIN_QUOTE_WORDS
from orchestrator.domain.interview_points import points_status, quote

INTERVIEWER, CANDIDATE = "interviewer", "candidate"
BATCH_LINES = 120
"""Most utterances in one attribution call."""
BATCH_CHARS = 16000
"""Most characters of utterance text in one attribution call (about 4
to 5 thousand tokens of Spanish), so the prompt and its reply stay well
inside the template's caps and deadline."""
LINE_CHARS = 1000
"""Characters of one utterance the model reads (a longer one is cut)."""
CONTEXT_LINES = 6
"""Lines of the previous batch repeated at the start of the next one,
from the last interviewer question, so an answer cut by the batch
boundary is read with its question."""
_TRIM = " .…\"'«»"


@dataclass(frozen=True)
class RoundLine:
    """One final utterance of the round, in order.

    ``role``: ``interviewer``, ``candidate`` or ``unknown`` (a speaker
    whose role was never set: read as context, never attributed).
    """

    id: str
    role: str
    text: str
    start: Optional[float] = None
    """Seconds from the start of the session (``None``: not known)."""


@dataclass
class PointFound:
    """A concept the model says the candidate treated (or does not know)."""

    status: PointStatus
    utterance: str
    quote: str


@dataclass
class SlotFound:
    """What the model attributed to one point of the plan."""

    answers: List[str] = field(default_factory=list)
    complete: bool = False
    points: Dict[str, PointFound] = field(default_factory=dict)
    asked: Dict[str, str] = field(default_factory=dict)
    """Concept id → the interviewer's utterance that asked it (#286)."""


Attribution = Dict[str, SlotFound]
"""Slot id → what was attributed to it."""


def batches(lines: Sequence[RoundLine], max_lines: int = BATCH_LINES,
            max_chars: int = BATCH_CHARS) -> List[List[RoundLine]]:
    """Cut the round into batches of at most ``max_lines`` / ``max_chars``.

    A batch after the first starts with up to :data:`CONTEXT_LINES` of
    the lines before it, from the last interviewer question, so an
    answer is read with its question; those lines count in the batch's
    limits. Rounds with no candidate line give no batch.
    """
    if not any(line.role == CANDIDATE for line in lines):
        return []
    found: List[List[RoundLine]] = []
    current: List[RoundLine] = []
    chars = 0
    start = 0
    for n, line in enumerate(lines):
        size = min(len(line.text), LINE_CHARS)
        fresh = n - start
        if fresh and (len(current) >= max_lines
                      or chars + size > max_chars):
            found.append(current)
            current = _context(lines, n)
            chars = sum(min(len(c.text), LINE_CHARS) for c in current)
            start = n
        current.append(line)
        chars += size
    if current:
        found.append(current)
    return [b for b in found if any(line.role == CANDIDATE for line in b)]


def _context(lines: Sequence[RoundLine], n: int) -> List[RoundLine]:
    """The lines before ``n`` that open the next batch."""
    before = list(lines[max(0, n - CONTEXT_LINES):n])
    asked = [i for i, line in enumerate(before) if line.role == INTERVIEWER]
    return before[asked[-1]:] if asked else []


def round_payload(lines: Sequence[RoundLine]) -> List[Dict[str, str]]:
    """The batch as the template reads it: id, role and words."""
    return [{"id": line.id, "role": line.role,
             "text": " ".join(line.text.split())[:LINE_CHARS]}
            for line in lines]


def _str(value: object) -> str:
    """A reply field as text (anything but a string is empty)."""
    return value if isinstance(value, str) else ""


def _id(value: object) -> str:
    """An id of the reply, normalised: stray spaces, brackets and quotes
    removed («[t3.s0]», « t3.s0 »); a number becomes its digits (and
    then matches nothing); anything else is empty."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return _str(value).strip().strip("[]\"'«» ").strip()


def _inside(said: str, text: str) -> bool:
    """Whether ``said`` is (folded) part of ``text``."""
    words = fold(" ".join(said.split())).strip(_TRIM)
    return bool(words) and words in fold(text)


def read_point(
    item: object, candidate: Mapping[str, str],
) -> Optional[Tuple[str, PointFound]]:
    """One concept of the reply, checked against the candidate's words."""
    if not isinstance(item, dict):
        return None
    pid = _id(item.get("id"))
    raw = item.get("status") or "treated"
    if not pid or raw not in ("treated", "unknown"):
        return None
    status: PointStatus = "unknown" if raw == "unknown" else "treated"
    utterance = _id(item.get("utterance"))
    said = " ".join(_str(item.get("quote")).split())
    if utterance not in candidate:
        telling = len(said.split()) >= MIN_QUOTE_WORDS
        utterance = next((u for u, text in candidate.items()
                          if telling and _inside(said, text)), "")
        if not utterance:
            return None
    text = candidate[utterance]
    if not _inside(said, text):
        said = quote(text)
    return pid, PointFound(status, utterance, said[:QUOTE_CHARS])


def read_attribution(
    data: Mapping[str, Any], plan: InterviewPlan,
    lines: Sequence[RoundLine],
    stats: Optional[Dict[str, int]] = None,
) -> Attribution:
    """Read one batch's reply against the plan and the batch's lines.

    Answer ids keep the round's order and are only the candidate's
    utterances of this batch (ids are normalised first, :func:`_id`);
    concepts follow :func:`read_point`. ``stats``, when given, gets the
    counts the caller logs and decides on (no words): ``shaped`` (1 when
    ``slots`` is a list), ``given`` (answer and concept entries the
    reply gave) and ``kept``.
    """
    candidate = {line.id: line.text for line in lines
                 if line.role == CANDIDATE}
    interviewer = {line.id for line in lines if line.role == INTERVIEWER}
    order = {line.id: n for n, line in enumerate(lines)}
    known = {s.id: {p.id for p in s.points} for s in plan.slots}
    found: Attribution = {}
    counts = {"shaped": 0, "given": 0, "kept": 0}
    raw = data.get("slots")
    counts["shaped"] = int(isinstance(raw, list))
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        sid = _id(item.get("id"))
        given = item.get("answers")
        given = given if isinstance(given, list) else []
        points = item.get("points")
        points = points if isinstance(points, list) else []
        counts["given"] += len(given) + len(points)
        if sid not in known:
            continue
        slot = found.setdefault(sid, SlotFound())
        ids = {_id(a) for a in given} & set(candidate)
        counts["kept"] += len(ids)
        slot.answers = sorted(set(slot.answers) | ids, key=order.__getitem__)
        slot.complete = slot.complete or item.get("status") == "complete"
        for raw_point in points:
            read = read_point(raw_point, candidate)
            if read is None or read[0] not in known[sid]:
                continue
            counts["kept"] += 1
            _keep_point(slot.points, *read)
        asked = item.get("asked")
        for raw_asked in asked if isinstance(asked, list) else []:
            if not isinstance(raw_asked, dict):
                continue
            counts["given"] += 1
            pid, by = _id(raw_asked.get("id")), _id(raw_asked.get("by"))
            if pid in known[sid] and by in interviewer and (
                    pid not in slot.asked
                    or order[by] < order[slot.asked[pid]]):
                counts["kept"] += 1
                slot.asked[pid] = by
        if not slot.answers and not slot.points and not slot.asked:
            del found[sid]
    if stats is not None:
        for key, value in counts.items():
            stats[key] = stats.get(key, 0) + value
    return found


def _keep_point(points: Dict[str, PointFound], pid: str,
                new: PointFound) -> None:
    """Keep a concept's strongest status: treated over unknown."""
    old = points.get(pid)
    if old is None or (old.status == "unknown" and new.status == "treated"):
        points[pid] = new


def merged(parts: Sequence[Attribution],
           order: Sequence[str]) -> Attribution:
    """Join the batches' attributions (answers in round ``order``)."""
    at = {u: n for n, u in enumerate(order)}
    found: Attribution = {}
    for part in parts:
        for sid, slot in part.items():
            into = found.setdefault(sid, SlotFound())
            into.answers = sorted(set(into.answers) | set(slot.answers),
                                  key=lambda u: at.get(u, len(at)))
            into.complete = into.complete or slot.complete
            for pid, point in slot.points.items():
                _keep_point(into.points, pid, point)
            for pid, by in slot.asked.items():
                if pid not in into.asked or at.get(by, len(at)) < at.get(
                        into.asked[pid], len(at)):
                    into.asked[pid] = by
    return found


def asked_marks(
    found: Attribution, lines: Sequence[RoundLine],
) -> Dict[str, Dict[str, PointAsked]]:
    """The concepts the model says were asked, as asked marks (#286).

    Slot id → concept id → ``by: "after_session"``, with the
    interviewer's line, its start and its first words.
    """
    by_id = {line.id: line for line in lines}
    marks: Dict[str, Dict[str, PointAsked]] = {}
    for sid, slot in found.items():
        for pid, uid in slot.asked.items():
            line = by_id.get(uid)
            if line is None:
                continue
            marks.setdefault(sid, {})[pid] = PointAsked(
                by="after_session", at=line.start, utterance=uid,
                quote=quote(line.text))
    return marks


def attributed_slots(
    plan: InterviewPlan, found: Attribution, texts: Mapping[str, str],
) -> List[SheetSlot]:
    """The plan's slots as a live sheet's, from the model's attribution.

    A concept's utterance also counts as an answer to its point. A slot
    with concepts takes the completion rule of
    :func:`~orchestrator.domain.interview_points.points_status` (still
    open → ``partial`` when answered); one without concepts is
    ``filled`` when the model said the answers are complete,
    ``partial`` when answered, else ``empty``. Every concept the model
    ticked is marked ``by: "after_session"``; the others are pending.
    """
    slots: List[SheetSlot] = []
    for slot in plan.slots:
        got = found.get(slot.id, SlotFound())
        points = []
        for p in slot.points:
            hit = got.points.get(p.id)
            points.append(SheetPoint(id=p.id, text=p.text) if hit is None
                          else SheetPoint(id=p.id, text=p.text,
                                          status=hit.status,
                                          quote=hit.quote,
                                          utterance=hit.utterance,
                                          by="after_session"))
        said = list(got.answers)
        said += [h.utterance for h in got.points.values()
                 if h.utterance not in said]
        settled = points_status(points)
        status = (settled if settled is not None
                  else "filled" if got.complete and not points and said
                  else "partial" if said else "empty")
        slots.append(SheetSlot(
            id=slot.id, label=slot.label, level=slot.level,
            weight=slot.weight, competency=slot.competency,
            status=status,
            answers=[Answer(utterance=u, quote=quote(texts.get(u, "")),
                            status="partial") for u in said],
            points=points,
        ))
    return slots


def used_ids(found: Attribution) -> Set[str]:
    """Every utterance id the attribution uses (for counts in the log)."""
    return {u for s in found.values()
            for u in [*s.answers, *(p.utterance for p in s.points.values())]}


def with_session_ticks(
    slots: Sequence[SheetSlot], session: Sequence[SheetSlot],
    texts: Mapping[str, str],
) -> List[SheetSlot]:
    """Keep what the session ticked and the model's reading missed.

    A concept still pending after the model's attribution but treated
    (or said unknown) during the session keeps that status, with who
    decided it (``rules`` or ``live_model``): the session's tick was
    backed by the candidate's words, and dropping it silently would
    hide it. Its utterance then also answers the point, and the point's
    status follows :func:`points_status` again. A tick whose utterance is
    no longer one of the candidate's (``texts``: a role changed later) is
    not kept.
    """
    during = {s.id: {p.id: p for p in s.points} for s in session}
    kept: List[SheetSlot] = []
    for slot in slots:
        ticked = during.get(slot.id, {})
        points = [
            ticked[p.id].model_copy() if p.status == "pending"
            and p.id in ticked and ticked[p.id].status != "pending"
            and ticked[p.id].utterance in texts else p
            for p in slot.points
        ]
        if points == slot.points:
            kept.append(slot)
            continue
        said = [a.utterance for a in slot.answers]
        extra = [p.utterance for p in points
                 if p.utterance and p.utterance not in said]
        answers = [*slot.answers, *(
            Answer(utterance=u, quote=quote(texts.get(u, "")),
                   status="partial") for u in dict.fromkeys(extra))]
        settled = points_status(points)
        kept.append(slot.model_copy(update={
            "points": points, "answers": answers,
            "status": settled if settled is not None
            else "partial" if answers else "empty"}))
    return kept
