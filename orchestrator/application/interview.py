"""Use cases of the AI interviewer (product ``voicebot.interview``, #208).

* :class:`InterviewPlanner` turns a job offer (pasted text, or a
  document uploaded to a knowledge domain) and the interview-method
  corpus (``interview.generic``) into an ``interview.plan.v1`` with the
  ``structured`` reasoner (template ``interview.planner@2``, which also
  gives each slot its concrete points, #258), keeps the
  plans, takes the recruiter's edits and lists the candidates
  interviewed for one plan (the comparison view).
* :class:`InterviewDirector` drives one interview: the voice bot asks
  it what each turn must do. After every candidate final the **slot
  filler** (``interview.slots@2`` on the reasoner, or
  :func:`~orchestrator.domain.interview_agenda.heuristic_fill`) says
  which slots the answer covers and which of their points it treats;
  the
  :class:`~orchestrator.domain.interview_agenda.Agenda` decides the next
  step; the coverage goes to the page (``{"type": "coverage"}``). When
  the session ends it scores the slots against the offer and judges how
  accurate each answer and treated concept was (``interview.report@2``,
  #259, with chunks of the plan's knowledge domain when it has one; or
  a coarse estimate from each slot's status), writes the
  ``interview.sheet.v1``, keeps it with the plan,
  and hands the ``transcript.v1`` (both voices) and the sheet to the
  post-call analysis (``{"type": "interview_result"}``).

Scores are for the recruiter only: nothing here is ever spoken.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Set, Tuple,
)

from orchestrator.application.ports import (
    BotOutput,
    CancelToken,
    Clock,
    Document,
    KnowledgePort,
    KnowledgeUnavailable,
    ReasonerPort,
    ReasonerUnavailable,
)
from orchestrator.domain.candidate import CoreQuestion
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.interview import (
    COMPETENCIES,
    QUOTE_CHARS,
    STATUS_CONFIDENCE,
    STATUS_SCORE,
    CheckItem,
    InterviewPlan,
    Judgement,
    OtherSkill,
    PlannerInfo,
    PlanSource,
    Sheet,
    SheetSlot,
    Slot,
    counts,
    edited_plan,
    fallback_plan,
    plan_from_draft,
    suitability,
)
from orchestrator.domain.interview_agenda import (
    MAX_TURNS,
    Agenda,
    Fill,
    Step,
    heuristic_fill,
    quote,
)
from orchestrator.domain.interview_judgement import clean_text, judge_slots
from orchestrator.domain.interview_points import PointHit, point_hits
from orchestrator.domain.interview_transcript import (
    BOT_ID,
    CANDIDATE_ID,
    Said,
    analysis_job,
    interview_transcript,
)
from orchestrator.domain.records import TurnRecord
from orchestrator.domain.turn_taking import BotTurn

if TYPE_CHECKING:  # pragma: no cover
    from orchestrator.application.candidates import CandidateBook

LOG = logging.getLogger(__name__)
CORE_EARLY = ("opening", "experience")
"""Phases of the core questions the AI interviewer asks first (#235)."""
CORE_LATE = ("closing",)
"""Phases of those it asks just before closing."""

OFFER_CHARS = 12000
MIN_OFFER_CHARS = 40
PLANNER_DEADLINE_MS = 120000
REPORT_DEADLINE_MS = 90000
JUDGE_CHUNKS = 3
"""Knowledge chunks the judge gets per slot (#259)."""
PLAIN_REPORT = "interview.report@1"
"""The report template that scores without judging the answers."""
FILLER_DEADLINE_MS = 6000
ANSWER_SO_FAR = 400
"""Characters of a stuck answer a hint is given (its end)."""


class PlanError(Exception):
    """A plan request that cannot be served; ``code`` says which."""

    def __init__(self, code: str, message: str):
        """Keep the stable code and the message."""
        super().__init__(message)
        self.code = code


class OfferUnavailable(Exception):
    """The uploaded job offer could not be read."""


class PlanStore(Protocol):
    """Where plans and the sheets of their interviews are kept."""

    def save(self, plan: InterviewPlan) -> None:
        """Keep ``plan`` (replacing its older revision)."""
        ...

    def load(self, plan_id: str) -> InterviewPlan:
        """Return a plan (``KeyError`` when there is none)."""
        ...

    def list(self) -> List[InterviewPlan]:
        """Every plan, newest first."""
        ...

    def save_result(self, plan_id: str, sheet: Document) -> None:
        """Keep the sheet of one interview of a plan."""
        ...

    def results(self, plan_id: str) -> List[Document]:
        """The sheets of a plan's interviews, newest first."""
        ...


class OfferSource(Protocol):
    """The text of a job offer uploaded to a knowledge domain."""

    def document_text(self, domain: str, document: str) -> Tuple[str, str]:
        """Return the document's name and text (``OfferUnavailable``)."""
        ...


class InterviewHandoff(Protocol):
    """The post-call analysis of the RAG."""

    def publish_transcript(self, document: Document) -> None:
        """Store a ``transcript.v1`` (raises ``OfferUnavailable``-like
        :class:`HandoffFailed` on failure)."""
        ...

    def start_analysis(self, session_id: str, job: str) -> None:
        """Start the post-call analysis with ``job`` as the job sought."""
        ...


class HandoffFailed(Exception):
    """The RAG did not take the transcript or the analysis."""


def _ok_data(response: Document) -> Optional[Dict[str, Any]]:
    """Return a structured response's ``data`` when the call went well."""
    data = response.get("data")
    if response.get("status") == "ok" and isinstance(data, dict):
        return data
    return None


class InterviewPlanner:
    """Draft, keep and edit interview plans.

    Args:
        store: Where plans are kept.
        knowledge: The knowledge port (the method corpus).
        reasoner: The reasoner port for the planner (a long time-out).
        offers: Reads an uploaded offer.
        section: Returns the product's ``interview`` section as the
            saved settings leave it (read at every draft).
        language: The product's language.
        now: Wall-clock time (for the plan's dates).
        new_id: Makes a plan id.
        model: The planner's model alias.
    """

    def __init__(
        self,
        store: PlanStore,
        knowledge: KnowledgePort,
        reasoner: ReasonerPort,
        offers: Optional[OfferSource],
        section: Callable[[], InterviewSection],
        language: str = "es",
        now: Callable[[], datetime] = datetime.now,
        new_id: Callable[[], str] = lambda: datetime.now().strftime(
            "plan-%Y%m%d-%H%M%S"
        ),
        model: str = "haiku",
    ):
        """Wire the ports."""
        self.store = store
        self.knowledge = knowledge
        self.reasoner = reasoner
        self.offers = offers
        self.section = section
        self.language = language
        self.now = now
        self.new_id = new_id
        self.model = model
        self._lock = threading.Lock()

    # ---- queries ------------------------------------------------------

    def list(self) -> List[Document]:
        """The plans, newest first, as short rows."""
        return [
            {"id": p.id, "title": p.title, "role_type": p.role_type,
             "minutes": p.minutes, "slots": len(p.slots),
             "minimums": sum(1 for s in p.slots if s.level == "minimo"),
             "revision": p.revision,
             "updated_at": p.updated_at.isoformat() if p.updated_at
             else None,
             "interviews": len(self.store.results(p.id))}
            for p in self.store.list()
        ]

    def get(self, plan_id: str) -> Document:
        """One plan."""
        return self._load(plan_id).to_contract()

    def results(self, plan_id: str) -> Document:
        """The candidates interviewed for a plan, for the comparison view.

        Rows are sorted by weighted suitability (best first, unscored
        last); each carries the per-slot scores.
        """
        plan = self._load(plan_id)
        rows: List[Dict[str, Any]] = []
        for sheet in self.store.results(plan_id):
            rows.append({
                "session_id": sheet.get("session_id"),
                "candidate": sheet.get("candidate"),
                "finished_at": sheet.get("finished_at"),
                "suitability": sheet.get("suitability"),
                "closed_by": sheet.get("closed_by"),
                "plan_revision": sheet.get("plan_revision"),
                "slots": [
                    {k: s.get(k) for k in ("id", "label", "level", "weight",
                                           "status", "score", "above")}
                    for s in sheet.get("slots") or []
                    if isinstance(s, dict)
                ],
            })
        rows.sort(key=lambda r: (r["suitability"] is None,
                                 -(r["suitability"] or 0.0)))
        return {"plan": plan.to_contract(), "candidates": rows}

    # ---- commands -----------------------------------------------------

    def create(self, body: Document) -> Document:
        """Draft a plan from a job offer (``body``: offer or document).

        ``body``: ``offer`` (text) or ``document`` (``{"domain",
        "id"}``), optional ``title``, ``role_type`` and ``minutes``.
        The planner's failure is not an error: the plan is then the
        generic one (``planner.status: "fallback"``) for the recruiter
        to edit.
        """
        offer, source = self._offer(body)
        config = self.section()
        minutes = _int(body.get("minutes"), config.minutes)
        title = str(body.get("title") or "").strip() or None
        role_type = str(body.get("role_type") or "").strip() or None
        plan_id = self.new_id()
        chunks = self._corpus(config, offer, role_type, plan_id)
        request: Document = {
            "schema": "reasoner.request.v1", "request_id": plan_id[-64:],
            "role": "structured", "model": self.model,
            "template": config.planner_template,
            "language": self.language, "context": {"window": []},
            "retrieved": chunks,
            "payload": {
                "offer": offer[:OFFER_CHARS], "minutes": minutes,
                "role_type": role_type or "",
                "competencies": [{"id": c, "name": n}
                                 for c, n in COMPETENCIES],
            },
            "constraints": {"deadline_ms": PLANNER_DEADLINE_MS},
        }
        now = self.now()
        error: Optional[str] = None
        data: Optional[Dict[str, Any]] = None
        model: Optional[str] = None
        try:
            response = self.reasoner.reason(request, CancelToken())
        except ReasonerUnavailable as exc:
            error = f"reasoner unavailable: {exc}"
        else:
            model = response.get("model")
            data = _ok_data(response)
            if data is None:
                failure = response.get("error") or {}
                error = (f"{response.get('status')}: "
                         f"{failure.get('message', '')}"[:300])
        plan: Optional[InterviewPlan] = None
        if data is not None:
            try:
                plan = plan_from_draft(
                    data, plan_id, source, minutes, self.language, now,
                    PlannerInfo(template=config.planner_template,
                                model=model,
                                chunks=[c["id"] for c in chunks]),
                    title, role_type,
                )
            except ValueError as exc:
                error = f"unusable draft: {exc}"[:300]
        if plan is None:
            plan = fallback_plan(plan_id, title or "Entrevista", source,
                                 minutes, self.language, now, error,
                                 role_type)
        self.store.save(plan)
        return plan.to_contract()

    def update(self, plan_id: str, document: Document) -> Document:
        """Save the recruiter's edit of a plan (revision checked)."""
        with self._lock:
            current = self._load(plan_id)
            try:
                plan = edited_plan(current, document, self.now())
            except ValueError as exc:
                stale = type(exc).__name__ == "StaleRevision"
                raise PlanError("conflict" if stale else "invalid_request",
                                str(exc)[:300]) from exc
            self.store.save(plan)
        return plan.to_contract()

    # ---- internals ----------------------------------------------------

    def _load(self, plan_id: str) -> InterviewPlan:
        try:
            return self.store.load(plan_id)
        except (KeyError, ValueError) as exc:
            raise PlanError("not_found", f"no plan {plan_id!r}") from exc

    def _offer(self, body: Document) -> Tuple[str, PlanSource]:
        """Read the offer of a create request."""
        document = body.get("document")
        if isinstance(document, dict):
            domain = str(document.get("domain") or "")
            doc = str(document.get("id") or "")
            if not domain or not doc or self.offers is None:
                raise PlanError("invalid_request",
                                "document needs domain and id")
            try:
                name, text = self.offers.document_text(domain, doc)
            except OfferUnavailable as exc:
                raise PlanError("offer", str(exc)) from exc
            source = PlanSource(kind="document", domain=domain,
                                document=doc, name=name[:120],
                                chars=len(text))
        else:
            text = str(body.get("offer") or "")
            source = PlanSource(kind="text", chars=len(text))
        text = text.strip()
        if len(text) < MIN_OFFER_CHARS:
            raise PlanError(
                "invalid_request",
                f"the job offer needs at least {MIN_OFFER_CHARS} characters",
            )
        return text[:OFFER_CHARS], source

    def _corpus(
        self, config: InterviewSection, offer: str,
        role_type: Optional[str], plan_id: str,
    ) -> List[Document]:
        """Method chunks for the planner (none when unavailable)."""
        query: Document = {
            "schema": "knowledge.query.v1", "request_id": f"{plan_id}-k",
            "domain": config.corpus, "text": offer[:1500],
            "language": self.language, "top_k": config.corpus_top_k,
        }
        if role_type:
            query["filters"] = {"tags": [f"puesto_{role_type}",
                                         "puesto_todos"]}
        try:
            result = self.knowledge.query(query)
        except KnowledgeUnavailable:
            return []
        return [
            {"id": str(c["id"]), "text": str(c["text"])[:800],
             "metadata": c.get("metadata") or {}}
            for c in (result.get("chunks") or [])[:10]
        ]


def _int(value: object, default: int) -> int:
    """Return ``value`` as an int, or ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return default


# ---- the slot filler ---------------------------------------------------


class SlotFiller(Protocol):
    """Says which slots an answer covers."""

    name: str

    def fill(
        self, text: str, target: Optional[Slot], slots: List[Slot],
        question: str, cancel: CancelToken,
    ) -> List[Fill]:
        """Return the slots ``text`` gives evidence for."""
        ...


class HeuristicFiller:
    """The slot filler without a model (:func:`heuristic_fill`)."""

    name = "heuristic"

    def fill(
        self, text: str, target: Optional[Slot], slots: List[Slot],
        question: str, cancel: CancelToken,
    ) -> List[Fill]:
        """Return the heuristic's fills."""
        return heuristic_fill(text, target, slots)


class ReasonerFiller:
    """The slot filler on the reasoner (``interview.slots@2``).

    A failed call falls back to the heuristic for that answer and is
    counted in ``fallbacks``. When the reply gives the slot asked but no
    ``points`` field for it (an ``@1`` template, or the model left the
    field out), the rules of
    :func:`~orchestrator.domain.interview_points.point_hits` say which
    of its points the answer treats. When the reply leaves the slot out
    ("covers none"), the model's word stands: no rule adds it back.

    Args:
        reasoner: The reasoner port.
        template: The filler's template.
        model: The model alias (Haiku by default).
        language: The interview's language.
    """

    name = "haiku"

    def __init__(
        self, reasoner: ReasonerPort, template: str = "interview.slots@2",
        model: str = "haiku", language: str = "es",
    ):
        """Keep the port."""
        self.reasoner = reasoner
        self.template = template
        self.model = model
        self.language = language
        self.calls = 0
        self.fallbacks = 0

    def fill(
        self, text: str, target: Optional[Slot], slots: List[Slot],
        question: str, cancel: CancelToken,
    ) -> List[Fill]:
        """Ask the reasoner; check what it says against the plan."""
        self.calls += 1
        request: Document = {
            "schema": "reasoner.request.v1",
            "request_id": f"fill-{int(time.time() * 1000)}-{self.calls}",
            "role": "structured", "model": self.model,
            "template": self.template, "language": self.language,
            "context": {"window": []},
            "payload": {
                "slots": [{"id": s.id, "label": s.label,
                           "evidence": s.evidence,
                           **({"points": [p.model_dump() for p in s.points]}
                              if s.points else {})} for s in slots],
                "target": target.id if target else "",
                "question": question[:600], "answer": text[:2000],
            },
            "constraints": {"deadline_ms": FILLER_DEADLINE_MS},
        }
        try:
            data = _ok_data(self.reasoner.reason(request, cancel))
        except ReasonerUnavailable:
            data = None
        if data is None:
            self.fallbacks += 1
            return heuristic_fill(text, target, slots)
        fills = read_fills(data, text, {s.id for s in slots},
                           {s.id: {p.id for p in s.points} for s in slots})
        if target is None or not target.points or _has_points(data,
                                                              target.id):
            return fills
        hits = tuple(point_hits(text, target.points))
        return [f if f.slot != target.id or not hits else Fill(
            f.slot, f.status, f.quote, hits) for f in fills]


def _has_points(data: Dict[str, Any], sid: str) -> bool:
    """Whether the reply leaves the rules nothing to add for a slot.

    True when it gives the slot's ``points`` list, and when it leaves
    the slot out (the model said the answer does not cover it).
    """
    items = data.get("slots")
    given = [item for item in (items if isinstance(items, list) else [])
             if isinstance(item, dict) and item.get("id") == sid]
    return not given or any(isinstance(item.get("points"), list)
                            for item in given)


def _said(value: object, text: str, plain: str) -> str:
    """The model's quote when it is the candidate's words, else theirs."""
    said = " ".join(str(value or "").split())
    if not said or said.lower().strip(" .…\"'«»") not in plain:
        said = quote(text)
    return said[:QUOTE_CHARS]


def read_fills(
    data: Dict[str, Any], text: str, known: "set[str]",
    points: Optional[Dict[str, Set[str]]] = None,
) -> List[Fill]:
    """Read the filler's JSON: known slot ids, valid status, real quote.

    A quote that is not the candidate's words is replaced by the
    answer's first words. Each slot's ``points`` (#258) keep only the
    ids of ``points[slot]`` with status ``treated`` or ``unknown``; a
    slot given only points (no status) counts as ``partial``.
    """
    found: List[Fill] = []
    items = data.get("slots")
    plain = " ".join(text.lower().split())
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "")
        hits = _point_hits(item.get("points"),
                           (points or {}).get(sid, set()), text, plain)
        status = item.get("status")
        if hits and status not in ("complete", "partial"):
            status = "partial"
        if sid not in known or status not in ("complete", "partial"):
            continue
        found.append(Fill(sid, status, _said(item.get("quote"), text, plain),
                          hits))
    return found


def _point_hits(
    raw: object, known: Set[str], text: str, plain: str
) -> Tuple[PointHit, ...]:
    """Read one slot's ``points`` of the filler's JSON (known ids only)."""
    hits: List[PointHit] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id") or "")
        status = item.get("status")
        if pid not in known or status not in ("treated", "unknown") or any(
            h.point == pid for h in hits
        ):
            continue
        hits.append(PointHit(pid, status,
                             _said(item.get("quote"), text, plain)))
    return tuple(hits)


# ---- the scorer --------------------------------------------------------


def status_scores(slots: List[SheetSlot]) -> List[SheetSlot]:
    """Give every slot without a score the coarse score of its status."""
    scored = []
    for slot in slots:
        value = STATUS_SCORE[slot.status]
        scored.append(slot.model_copy(update={
            "score": value,
            "confidence": STATUS_CONFIDENCE if value is not None else None,
            "scored_by": "status" if value is not None else None,
        }))
    return scored


def read_report(
    data: Dict[str, Any], slots: List[SheetSlot]
) -> Tuple[List[SheetSlot], List[OtherSkill], str]:
    """Merge the report's JSON into the slots (unknown ids ignored).

    A slot the candidate said they do not know keeps 0; a slot without
    any answer (never asked, or asked and not answered) keeps no score
    whatever the model says: not asked is not "does not know".
    """
    given: Dict[str, Dict[str, Any]] = {}
    items = data.get("slots")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            given[item["id"]] = item
    merged = []
    for slot in slots:
        item = given.get(slot.id)
        if slot.status == "empty" and not slot.answers:
            merged.append(slot)
            continue
        score = item.get("score") if item else None
        if (item is None or isinstance(score, bool)
                or not isinstance(score, (int, float))):
            merged.append(status_scores([slot])[0])
            continue
        confidence = item.get("confidence")
        notes = item.get("notes")
        merged.append(slot.model_copy(update={
            "score": 0 if slot.status == "unknown"
            else max(0, min(100, int(round(score)))),
            "above": bool(item.get("above")) and slot.status == "filled",
            "notes": clean_text(notes if isinstance(notes, str) else "",
                                300),
            "confidence": (max(0.0, min(1.0, float(confidence)))
                           if isinstance(confidence, (int, float))
                           and not isinstance(confidence, bool) else None),
            "scored_by": "model",
        }))
    others: List[OtherSkill] = []
    raw = data.get("other_skills")
    for item in raw if isinstance(raw, list) else []:
        label = item.get("label") if isinstance(item, dict) else None
        said = item.get("quote") if isinstance(item, dict) else None
        if isinstance(label, str) and label.strip():
            others.append(OtherSkill(
                label=" ".join(label.split())[:120],
                quote=" ".join(said.split())[:300]
                if isinstance(said, str) else "",
            ))
    raw_summary = data.get("summary")
    summary = clean_text(raw_summary if isinstance(raw_summary, str)
                         else "", 1200)
    return merged, others[:8], summary


# ---- one interview -----------------------------------------------------


@dataclass
class InterviewWiring:
    """What the live sessions need to run interviews.

    Args:
        planner: The plans (a session starts from one).
        store: Where the sheets are kept.
        reasoner: The reasoner port of the slot filler and the report.
        handoff: The post-call analysis (``None``: not handed over).
    """

    planner: "InterviewPlanner"
    store: PlanStore
    reasoner: ReasonerPort
    handoff: Optional[InterviewHandoff] = None
    candidates: Optional["CandidateBook"] = None
    """The candidate profiles (#235; ``None``: not kept)."""


@dataclass(frozen=True)
class TurnPlan:
    """What one turn of the interviewer must do."""

    payload: Document
    query: str
    step: Step


class InterviewDirector:
    """Drive one interview: what each turn does, coverage, the result.

    Args:
        plan: The plan of this interview.
        session_id: The live session.
        config: The product's ``interview`` section.
        filler: The slot filler.
        reasoner: The reasoner port (the report at the end).
        output: The session's feed (coverage and result messages).
        clock: Monotonic time.
        store: Where the sheet is kept with the plan.
        handoff: The post-call analysis (``None``: not handed over).
        post_call: Start the analysis when the interview ends.
        candidate: How the recruiter calls the candidate (optional).
        model: The report's model alias.
        now: Wall-clock time.
        core_open: The candidate's core questions still open (#235;
            ``None``: none are asked). Those with ``[...]`` blanks the
            plan cannot fill (work mode and city) are left out.
        on_core: Told which core question each turn asks (``None``: the
            turn asks something else), so the answer can be read even
            when the model words the question its own way.
        knowledge: Where the judge's chunks come from (#259; ``None``:
            the judge uses general knowledge).
    """

    def __init__(
        self,
        plan: InterviewPlan,
        session_id: str,
        config: InterviewSection,
        filler: SlotFiller,
        reasoner: ReasonerPort,
        output: BotOutput,
        clock: Clock,
        store: Optional[PlanStore] = None,
        handoff: Optional[InterviewHandoff] = None,
        post_call: bool = False,
        candidate: Optional[str] = None,
        model: str = "haiku",
        now: Callable[[], datetime] = datetime.now,
        core_open: Optional[Callable[[], List[CoreQuestion]]] = None,
        on_core: Optional[Callable[[Optional[str]], None]] = None,
        knowledge: Optional[KnowledgePort] = None,
    ):
        """Start the agenda; nothing has been said yet."""
        self.plan = plan
        self.session_id = session_id
        self.config = config
        self.filler = filler
        self.reasoner = reasoner
        self.output = output
        self.clock = clock
        self.store = store
        self.handoff = handoff
        self.post_call = post_call
        self.candidate = candidate
        self.model = model
        self.now = now
        self.started = clock.now()
        self.started_at = now()
        self.agenda = Agenda(plan, self.started, config.max_probes)
        self.said: List[Said] = []
        self.heard_ids: Dict[str, int] = {}
        self.last_question = ""
        self.closing = False
        self.finished: Optional[Document] = None
        self.filler_ms: List[float] = []
        self.core_open = core_open
        self.core_asked: Set[str] = set()
        self.core_target: Optional[str] = None
        self.on_core = on_core
        self.knowledge = knowledge
        self.deferred_close: Optional[str] = None
        """Why the agenda closed while the salary question was asked."""
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Whether the closing turn was decided (no more turns)."""
        return self.agenda.closed_by is not None

    @property
    def target(self) -> Optional[str]:
        """The slot being asked, if any."""
        return self.agenda.target

    # ---- the voice bot asks ------------------------------------------

    def heard(self, utterance: str, text: str, start: float,
              end: float) -> None:
        """Keep one candidate final (a revision replaces it)."""
        with self._lock:
            item = Said(CANDIDATE_ID, text, start, end, utterance)
            if utterance in self.heard_ids:
                self.said[self.heard_ids[utterance]] = item
            else:
                self.heard_ids[utterance] = len(self.said)
                self.said.append(item)

    def plan_turn(
        self, answer: str, utterance: Optional[str], cancel: CancelToken,
        kind: BotTurn = "question",
    ) -> TurnPlan:
        """Decide one turn: fill the slots with the answer, then step.

        A ``reask`` or ``hint`` turn (#213) does not move the agenda: it
        words the slot being asked again, or a hint on how to go on
        with ``answer`` (what the candidate said so far).
        """
        if kind != "question":
            return self._aside(kind, answer)
        with self._lock:
            agenda = self.agenda
            core_answer = self.core_target is not None
            self.core_target = None
            if utterance is not None and answer.strip() and not core_answer:
                target = (agenda.slots.get(agenda.target)
                          if agenda.target else None)
                asked = self.clock.now()
                fills = self.filler.fill(
                    answer, target, self.plan.slots, self.last_question,
                    cancel,
                )
                self.filler_ms.append(
                    round((self.clock.now() - asked) * 1000.0, 1)
                )
                if cancel.cancelled:
                    # The candidate went on talking while the filler
                    # ran (#213): the turn is dropped, the agenda waits
                    # for the whole answer.
                    step = Step("ask", None,
                                minutes_left=agenda.minutes_left(
                                    self.clock.now()))
                    return TurnPlan(self._payload(step), answer[:1000],
                                    step)
                agenda.answer(utterance, answer, fills)
            if self.deferred_close is not None:
                # The salary question was answered: close as decided.
                agenda.closed_by = self.deferred_close
                self.deferred_close = None
            core = self._core_turn(CORE_EARLY, opened=agenda.opened)
            if core is not None:
                return core
            step = agenda.step(self.clock.now())
            if step.action == "close" and agenda.turns <= MAX_TURNS:
                core = self._core_turn(CORE_LATE, opened=True)
                if core is not None:
                    # Ask it with the interview still open (so its answer
                    # is heard), then close on the next turn.
                    self.deferred_close = agenda.closed_by
                    agenda.closed_by = None
                    return core
            if step.action == "open" and self.core_asked:
                step = Step("ask", step.slot, step.missing,
                            step.minutes_left)
            self.closing = step.action == "close"
            payload = self._payload(step)
            self._publish_coverage(step)
        if self.on_core is not None:
            self.on_core(None)
        slot = step.slot
        query = (f"{slot.label}. {slot.questions[0]}" if slot is not None
                 else answer or self.plan.title)
        return TurnPlan(payload, query[:1000], step)

    def spoken(self, record: TurnRecord) -> None:
        """Keep the bot's turn (what was actually spoken)."""
        if not record.spoken:
            return
        text = " ".join(record.spoken)
        end = self.clock.now() - self.started
        start = max(0.0, end - (record.stages.total_ms or 0.0) / 1000.0
                    + (record.stages.first_audio_ms or 0.0) / 1000.0)
        with self._lock:
            self.said.append(Said(BOT_ID, text, start,
                                  start + record.audio_ms / 1000.0))
            self.last_question = text
            closing = self.closing
        if closing:
            self.output.publish({"type": "interview", "state": "closed",
                                 "closed_by": self.agenda.closed_by})

    def finish(self) -> Optional[Document]:
        """Score, keep and hand over the interview (once)."""
        with self._lock:
            if self.finished is not None:
                return self.finished
            self.finished = {}
        self.output.publish({"type": "interview", "state": "reporting"})
        sheet = self._sheet()
        transcript, refs = interview_transcript(
            self.session_id, list(self.said), self.plan.language,
            self.plan.title, self.candidate, self.started_at,
        )
        sheet = _with_refs(sheet, refs)
        handed = self._handoff(transcript, sheet)
        document = sheet.to_contract()
        if self.store is not None:
            try:
                self.store.save_result(self.plan.id, document)
            except OSError as exc:
                handed["error"] = f"sheet not kept: {exc}"[:200]
        result = {"type": "interview_result", "sheet": document,
                  "counts": counts(sheet.slots), "handoff": handed}
        self.output.publish(result)
        self.finished = result
        return result

    # ---- internals ---------------------------------------------------

    def _core_turn(
        self, phases: Tuple[str, ...], opened: bool
    ) -> Optional[TurnPlan]:
        """Ask the next open core question of ``phases`` (#235), once.

        Early ones are asked before the plan's first slot (the opening
        turn greets and asks the first of them); late ones (the salary
        expectation) just before closing. Questions with blanks are
        left to a person. Called with the lock held.
        """
        if self.core_open is None or (opened and phases == CORE_EARLY):
            return None
        question = next((
            q for q in self.core_open()
            if q.phase in phases and not q.has_blanks
            and q.id not in self.core_asked
        ), None)
        if question is None:
            return None
        self.core_asked.add(question.id)
        self.core_target = question.id
        if self.on_core is not None:
            self.on_core(question.id)
        text = question.text(self.plan.language)
        first = phases == CORE_EARLY and len(self.core_asked) == 1
        payload: Document = {
            "title": self.plan.title, "summary": self.plan.summary,
            "minutes": self.plan.minutes,
            "minutes_left": self.agenda.minutes_left(self.clock.now()),
            "action": "open" if first else "ask",
            "topic": text, "questions": [text], "evidence": "",
        }
        step = Step("open" if first else "ask", None,
                    minutes_left=payload["minutes_left"])
        return TurnPlan(payload, text[:1000], step)

    def _aside(self, kind: BotTurn, answer: str) -> TurnPlan:
        """A re-ask or a hint on the slot being asked (no agenda step).

        After a core question it rewords that question (#235).
        """
        with self._lock:
            agenda = self.agenda
            core = next((q for q in (self.core_open() if self.core_open
                                     else [])
                         if q.id == self.core_target), None)
            if core is not None:
                text = core.text(self.plan.language)
                step = Step("hint" if kind == "hint" else "reask", None,
                            minutes_left=agenda.minutes_left(
                                self.clock.now()))
                payload = self._payload(step)
                payload.update({"topic": text, "questions": [text]})
                if kind == "hint" and answer.strip():
                    payload["answer_so_far"] = (
                        answer.strip()[-ANSWER_SO_FAR:])
                return TurnPlan(payload, text[:1000], step)
            slot = agenda.slots.get(agenda.target) if agenda.target else None
            step = Step("hint" if kind == "hint" else "reask", slot,
                        minutes_left=agenda.minutes_left(self.clock.now()))
            payload = self._payload(step)
            if kind == "hint" and answer.strip():
                payload["answer_so_far"] = answer.strip()[-ANSWER_SO_FAR:]
            self._publish_coverage(step)
        query = (f"{slot.label}. {slot.questions[0]}" if slot is not None
                 else answer or self.plan.title)
        return TurnPlan(payload, query[:1000], step)

    def _payload(self, step: Step) -> Document:
        """The interviewer template's payload for ``step``."""
        slot = step.slot
        payload: Document = {
            "title": self.plan.title, "summary": self.plan.summary,
            "minutes": self.plan.minutes,
            "minutes_left": step.minutes_left, "action": step.action,
        }
        if slot is not None:
            payload.update({
                "slot": slot.id, "topic": slot.label,
                "questions": list(slot.questions),
                "evidence": slot.evidence,
            })
        if step.missing:
            payload["missing"] = step.missing
        if step.points:
            payload["pending_points"] = list(step.points)
            payload["treated_points"] = list(step.treated)
        return payload

    def coverage(self) -> Document:
        """The live coverage: each slot's status, answers and attempts."""
        agenda = self.agenda
        return {
            "plan_id": self.plan.id, "title": self.plan.title,
            "minutes": self.plan.minutes,
            "minutes_left": agenda.minutes_left(self.clock.now()),
            "target": agenda.target, "closed_by": agenda.closed_by,
            "turns": agenda.turns,
            "minimums_covered": agenda.minimums_settled(),
            "slots": [s.model_dump(mode="json")
                      for s in agenda.sheet_slots()],
        }

    def _publish_coverage(self, step: Step) -> None:
        """Send the coverage to the page."""
        self.output.publish({"type": "coverage", "action": step.action,
                             "coverage": self.coverage()})

    def _sheet(self) -> Sheet:
        """The sheet of the interview, scored."""
        agenda = self.agenda
        slots = agenda.sheet_slots()
        others: List[OtherSkill] = []
        summary = ""
        scorer = "status"
        to_check: List[CheckItem] = []
        judge: Optional[str] = None
        try:
            chunks = self._chunks(slots)
        except Exception:  # noqa: BLE001 - logged, general knowledge
            LOG.exception("report: chunks of %s not read", self.session_id)
            chunks = {}
        data = self._report(slots, chunks)
        if data is not None:
            plain = slots
            try:
                slots, others, summary = read_report(data, plain)
                scorer = "haiku"
            except Exception:  # noqa: BLE001 - logged, status scores
                LOG.exception("report: reply of %s not read",
                              self.session_id)
                slots, others, summary = status_scores(plain), [], ""
                data = None
            if data is not None and (
                    self.config.report_template != PLAIN_REPORT):
                scored = slots
                try:
                    slots, to_check = judge_slots(
                        data, scored,
                        {sid: [c["id"] for c in found]
                         for sid, found in chunks.items()},
                        self._texts())
                    judge = self.config.report_template
                except Exception:  # noqa: BLE001 - logged, plain scores
                    LOG.exception("report: judgement of %s not read",
                                  self.session_id)
                    slots, to_check = scored, []
        else:
            slots = status_scores(slots)
        # Stopped while the salary question waited: the plan was covered.
        closed_by = agenda.closed_by or self.deferred_close
        return Sheet(
            session_id=self.session_id, plan_id=self.plan.id,
            plan_revision=self.plan.revision, title=self.plan.title,
            state="closed" if closed_by else "stopped",
            closed_by=closed_by or "stopped",  # type: ignore[arg-type]
            turns=agenda.turns,
            elapsed_s=round(self.clock.now() - self.started, 1),
            minutes=self.plan.minutes,
            minimums_covered=agenda.minimums_settled(),
            turns_to_minimums=agenda.turns_to_minimums,
            slots=slots, filler=self.filler.name,
            candidate=self.candidate, suitability=suitability(slots),
            summary=summary, other_skills=others, scorer=scorer,
            finished_at=self.now(), to_check=to_check, judge=judge,
        )

    def _texts(self) -> Dict[str, str]:
        """The candidate's whole words per utterance id."""
        return {s.ref: s.text for s in self.said
                if s.speaker == CANDIDATE_ID and s.ref}

    def _chunks(self, slots: List[SheetSlot]) -> Dict[str, List[Document]]:
        """Chunks of the plan's knowledge domain for each answered slot."""
        if self.config.report_template == PLAIN_REPORT:
            return {}
        return judge_chunks(self.knowledge, self.plan, slots, self._texts(),
                            self.session_id)

    def _report(
        self, slots: List[SheetSlot],
        chunks: Optional[Dict[str, List[Document]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ask the reasoner to score (and judge) the slots.

        ``None``: not scored.
        """
        if not self.config.report or not any(s.answers for s in slots):
            return None
        quoted = {a.quote for s in slots for a in s.answers}
        other = " ".join(
            s.text for s in self.said
            if s.speaker == CANDIDATE_ID and s.text not in quoted
        )
        request = report_request(
            self.plan, slots, self._texts(), chunks or {}, other,
            self.config.report_template, self.model,
            f"{self.session_id}-report"[-64:],
        )
        try:
            return _ok_data(self.reasoner.reason(request, CancelToken()))
        except ReasonerUnavailable:
            return None

    def _handoff(self, transcript: Document, sheet: Sheet) -> Document:
        """Publish the transcript and maybe start the analysis."""
        handed: Document = {"transcript": False, "analysis": None,
                            "error": None}
        if self.handoff is None:
            return handed
        if not transcript["utterances"]:
            handed["error"] = "nothing was said"
            return handed
        try:
            self.handoff.publish_transcript(transcript)
            handed["transcript"] = True
            if self.post_call:
                self.handoff.start_analysis(
                    self.session_id, analysis_job(self.plan, sheet)
                )
                handed["analysis"] = "started"
        except HandoffFailed as exc:
            handed["error"] = str(exc)[:300]
        return handed


def _ref(refs: Dict[str, str], utterance: Optional[str]) -> Optional[str]:
    """One utterance id in the transcript's terms."""
    return refs.get(utterance, utterance) if utterance else None


def _judged(
    judgement: Optional[Judgement], refs: Dict[str, str]
) -> Optional[Judgement]:
    """A judgement pointing at the transcript's utterance id."""
    if judgement is None:
        return None
    return judgement.model_copy(update={
        "utterance": _ref(refs, judgement.utterance)})


def _with_refs(sheet: Sheet, refs: Dict[str, str]) -> Sheet:
    """Point the sheet's answers, points and judgements at the
    transcript's ids."""
    slots = []
    for slot in sheet.slots:
        answers = [a.model_copy(update={
            "utterance": refs.get(a.utterance, a.utterance)
        }) for a in slot.answers]
        points = [p.model_copy(update={
            "utterance": _ref(refs, p.utterance),
            "judgement": _judged(p.judgement, refs),
        }) for p in slot.points]
        slots.append(slot.model_copy(update={
            "answers": answers, "points": points,
            "judgement": _judged(slot.judgement, refs)}))
    checks = [c.model_copy(update={"utterance": _ref(refs, c.utterance)})
              for c in sheet.to_check]
    return sheet.model_copy(update={"slots": slots, "to_check": checks})


def judge_chunks(
    knowledge: Optional[KnowledgePort], plan: InterviewPlan,
    slots: List[SheetSlot], texts: Dict[str, str], request_id: str,
) -> Dict[str, List[Document]]:
    """Chunks of the plan's knowledge domain for each answered slot.

    The plan's domain is the one its offer was uploaded to; a plan made
    from pasted text has none, and then the judge uses its general
    knowledge (an empty map). A failed query is the same.
    """
    domain = plan.source.domain if plan.source.kind == "document" else None
    if knowledge is None or not domain:
        return {}
    found: Dict[str, List[Document]] = {}
    for n, slot in enumerate(slots):
        said = [texts.get(a.utterance, a.quote) for a in slot.answers]
        if not said:
            continue
        text = " ".join([slot.label, *(p.text for p in slot.points),
                         *said])[:1500]
        query: Document = {
            "schema": "knowledge.query.v1",
            "request_id": f"{request_id}-j{n}"[-64:], "domain": domain,
            "text": text, "language": plan.language, "top_k": JUDGE_CHUNKS,
        }
        try:
            result = knowledge.query(query)
        except KnowledgeUnavailable:
            return {}
        raw = result.get("chunks") if isinstance(result, dict) else None
        chunks = [
            {"id": c["id"], "text": c["text"][:600]}
            for c in (raw if isinstance(raw, list) else [])
            if isinstance(c, dict) and isinstance(c.get("id"), str)
            and isinstance(c.get("text"), str)
        ][:JUDGE_CHUNKS]
        if chunks:
            found[slot.id] = chunks
    return found


def report_request(
    plan: InterviewPlan, slots: List[SheetSlot], texts: Dict[str, str],
    chunks: Dict[str, List[Document]], other: str, template: str,
    model: str, request_id: str,
) -> Document:
    """The report's ``reasoner.request.v1`` (``@1`` and ``@2`` fields).

    ``quotes`` serve ``@1``; ``answers`` (utterance id and whole words),
    ``points`` and ``chunks`` serve the judge of ``@2``.
    """
    by_id = {s.id: s for s in plan.slots}
    return {
        "schema": "reasoner.request.v1", "request_id": request_id,
        "role": "structured", "model": model, "template": template,
        "language": plan.language, "context": {"window": []},
        "payload": {
            "title": plan.title, "summary": plan.summary,
            "slots": [
                {"id": s.id, "label": s.label, "level": s.level,
                 "status": s.status,
                 "evidence": by_id[s.id].evidence if s.id in by_id else "",
                 "quotes": [a.quote for a in s.answers],
                 "answers": [{"utterance": a.utterance,
                              "text": texts.get(a.utterance, a.quote)}
                             for a in s.answers],
                 "points": [{"id": p.id, "text": p.text,
                             "status": p.status} for p in s.points],
                 "chunks": chunks.get(s.id, [])}
                for s in slots
            ],
            "other": other[:3000],
        },
        "constraints": {"deadline_ms": REPORT_DEADLINE_MS},
    }
