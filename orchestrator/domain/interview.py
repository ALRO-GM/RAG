"""The interview plan (``interview.plan.v1``) and its slot sheet.

An AI interviewer (product ``voicebot.interview``, #208) works from a
plan made from a job offer: **slots**, each a requirement of the offer
or a competency of the rubric (``knowledge/rubrics/core.es.json``),
with question ideas, the evidence a complete answer shows, the concrete
**points** (concepts) a complete answer talks about (#258) and a level:
``minimo`` (must be answered before the interview closes), ``deseable``
(asked when there is time) or ``opcional``. The planner model writes a
draft; :func:`plan_from_draft` keeps only what fits the contract and
drops any slot about a protected characteristic; the recruiter then
edits it on the page. :func:`fallback_plan` is the plan used when the
planner gives nothing usable.

The **sheet** (``interview.sheet.v1``) is the plan's slots after the
interview: each with its status (``filled``, ``partial``, ``empty``)
and the candidate's answers with their utterance ids and quotes; a
slot with points also says, point by point, whether the candidate
talked about it (``treated``), not yet (``pending``) or said they do
not know it (``unknown``), with the quote and utterance that settled it.

Plain data and validation: no I/O.
"""

import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

PLAN_SCHEMA = "interview.plan.v1"
SHEET_SCHEMA = "interview.sheet.v1"
Level = Literal["minimo", "deseable", "opcional"]
LEVELS: Tuple[Level, ...] = ("minimo", "deseable", "opcional")
SlotKind = Literal["requirement", "competency"]
Phase = Literal["experience", "technical", "behavioural"]
PHASES: Tuple[Phase, ...] = ("experience", "technical", "behavioural")
RoleType = Literal[
    "tecnico", "comercio", "atencion_cliente", "hosteleria", "almacen",
    "oficina", "otro",
]
ROLE_TYPES: Tuple[RoleType, ...] = (
    "tecnico", "comercio", "atencion_cliente", "hosteleria", "almacen",
    "oficina", "otro",
)
SlotStatus = Literal["filled", "partial", "empty", "skipped", "unknown"]
"""``skipped``: the candidate chose to move on after the clarification
offer; ``unknown``: they said they do not know the topic («no lo
conoce»)."""
MIN_WEIGHT, MAX_WEIGHT, DEFAULT_WEIGHT = 1, 5, 3

COMPETENCIES: Tuple[Tuple[str, str], ...] = (
    ("COMM", "Comunicación clara"),
    ("LISTEN", "Escucha activa"),
    ("INFLU", "Influencia con argumentos"),
    ("COLLAB", "Colaboración y apoyo mutuo"),
    ("INCL", "Facilitación inclusiva"),
    ("CONFL", "Desacuerdo constructivo y gestión del conflicto"),
    ("FEEDB", "Feedback y desarrollo de otros"),
    ("COORD", "Coordinación y seguimiento del trabajo"),
    ("SOLVE", "Resolución de problemas y toma de decisiones en grupo"),
    ("INIT", "Iniciativa y compromiso con la acción"),
    ("ADAPT", "Apertura al aprendizaje y adaptabilidad"),
)
"""The rubric's 11 competencies (id, Spanish name), as in
``knowledge/rubrics/core.es.json`` (a test keeps them equal)."""
COMPETENCY_IDS = tuple(c for c, _ in COMPETENCIES)

MAX_SLOTS = 12
MAX_POINTS, POINT_CHARS = 6, 80
"""Concepts per slot, and characters per concept (#258)."""
PointStatus = Literal["treated", "pending", "unknown"]
DecidedBy = Literal["rules", "live_model", "after_session"]
"""Who decided a concept's status (#268): the word rules during the
session; a model call during the session (the AI interviewer's slot
filler, the live page's tick call); or the model reading the
whole round after the session."""
POINT_STATUSES: Tuple[PointStatus, ...] = ("treated", "pending", "unknown")
MIN_MINUTES, MAX_MINUTES = 5, 90
LABEL_CHARS, QUESTION_CHARS, EVIDENCE_CHARS = 120, 300, 300
SUMMARY_CHARS = 400
QUOTE_CHARS = 300

_PROTECTED = re.compile(
    r"\b(edad|anos tienes|fecha de nacimiento|estado civil|casad[oa]|"
    r"solter[oa]|divorciad|embaraz|tienes hijos|hijos o hijas|"
    r"planes familiares|maternidad|paternidad|religi|creencias|"
    r"ideologia|ideas politicas|opinion(es)? politica|partido politico|"
    r"afiliacion|sindica|orientacion sexual|identidad de genero|"
    r"salud(?! laboral)|enfermedad|discapacidad|minusval|baja medica|"
    r"nacionalidad|pais de origen|origen etnico|origen racial|etnia|raza|"
    r"pais de nacimiento)"
)
"""Words of a protected characteristic (accents folded): a planner slot
or question that names one is dropped."""

_LEVEL_WORDS = {
    "minimo": "minimo", "minimum": "minimo", "imprescindible": "minimo",
    "obligatorio": "minimo", "required": "minimo", "must": "minimo",
    "deseable": "deseable", "desirable": "deseable", "nice": "deseable",
    "opcional": "opcional", "optional": "opcional", "maximo": "opcional",
}


def fold(text: str) -> str:
    """Return ``text`` in lower case without accents (for matching)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_protected(text: str) -> bool:
    """Tell whether ``text`` asks about a protected characteristic."""
    return bool(_PROTECTED.search(fold(text)))


def _line(value: object, limit: int) -> str:
    """Return ``value`` as one trimmed line cut to ``limit`` characters."""
    return " ".join(str(value or "").split())[:limit]


class _Strict(BaseModel):
    """Base of the plan models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class Point(_Strict):
    """A concrete concept a complete answer to a slot talks about (#258).

    For a knowledge, technical or requirement slot, what a person who
    knows the topic would mention («tipos de pisada»); for a
    competency slot, a part of a complete example (situation, own
    action, result) or what is specific to this job.
    """

    id: str = Field(pattern=r"^p[0-9]{1,2}$")
    text: str = Field(min_length=1, max_length=POINT_CHARS)


class Slot(_Strict):
    """One thing the interview must, should or may find out."""

    id: str = Field(pattern=r"^s[0-9]{1,3}$")
    label: str = Field(min_length=1, max_length=LABEL_CHARS)
    kind: SlotKind = "requirement"
    competency: Optional[str] = None
    level: Level = "deseable"
    phase: Phase = "experience"
    questions: List[str] = Field(min_length=1, max_length=4)
    evidence: str = Field(default="", max_length=EVIDENCE_CHARS)
    weight: int = Field(default=DEFAULT_WEIGHT, ge=MIN_WEIGHT, le=MAX_WEIGHT)
    points: List[Point] = Field(default_factory=list, max_length=MAX_POINTS)
    """The concepts to be treated (#258); none: the slot works as before."""

    @model_validator(mode="after")
    def _valid(self) -> "Slot":
        if self.competency is not None and (
            self.competency not in COMPETENCY_IDS
        ):
            raise ValueError(f"unknown competency {self.competency!r}")
        if self.kind == "competency" and self.competency is None:
            raise ValueError("a competency slot names its competency")
        for question in self.questions:
            if not question.strip() or len(question) > QUESTION_CHARS:
                raise ValueError("each question has 1 to 300 characters")
        ids = [p.id for p in self.points]
        if len(set(ids)) != len(ids):
            raise ValueError("point ids must be unique in a slot")
        return self


class PlanSource(_Strict):
    """Where the job offer came from (its text is not kept here)."""

    kind: Literal["text", "document", "none"] = "text"
    domain: Optional[str] = None
    document: Optional[str] = None
    name: Optional[str] = None
    chars: int = Field(default=0, ge=0)


PlanState = Literal["open", "closed"]


class Closure(_Strict):
    """One time the position was closed (#262).

    Closing deletes nothing: it only stops new interviews from the plan.
    """

    closed_at: datetime
    reopened_at: Optional[datetime] = None


class PlannerInfo(_Strict):
    """How the plan was drafted."""

    template: str = "interview.planner@1"
    model: Optional[str] = None
    status: Literal["ok", "fallback", "manual"] = "ok"
    error: Optional[str] = None
    chunks: List[str] = Field(default_factory=list)


class InterviewPlan(_Strict):
    """``interview.plan.v1``: what an AI interview has to cover."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["interview.plan.v1"] = Field(
        default="interview.plan.v1", alias="schema"
    )
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    revision: int = Field(default=0, ge=0)
    title: str = Field(min_length=1, max_length=LABEL_CHARS)
    role_type: RoleType = "otro"
    language: str = Field(default="es", min_length=2, max_length=8)
    summary: str = Field(default="", max_length=SUMMARY_CHARS)
    minutes: int = Field(default=20, ge=MIN_MINUTES, le=MAX_MINUTES)
    source: PlanSource = Field(default_factory=PlanSource)
    planner: PlannerInfo = Field(default_factory=PlannerInfo)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    slots: List[Slot] = Field(min_length=1, max_length=MAX_SLOTS)
    state: PlanState = "open"
    """Open, or closed: the position was filled or dropped (#262); a
    closed plan starts no interview. A plan stored before this field
    loads as open."""
    closures: List[Closure] = Field(default_factory=list)
    """Every time it was closed (and reopened)."""

    @model_validator(mode="after")
    def _unique(self) -> "InterviewPlan":
        ids = [s.id for s in self.slots]
        if len(set(ids)) != len(ids):
            raise ValueError("slot ids must be unique")
        return self

    def to_contract(self) -> Dict[str, Any]:
        """Return the JSON document of the contract."""
        return self.model_dump(mode="json", by_alias=True)

    def ordered(self) -> List[Slot]:
        """Return the slots by level (minimo first), then plan order."""
        return sorted(self.slots, key=lambda s: LEVELS.index(s.level))


def _level(value: object) -> Level:
    """Map a planner's level word to a level (``deseable`` if unknown)."""
    word = fold(str(value or "")).strip()
    for key, level in _LEVEL_WORDS.items():
        if word.startswith(key):
            return level  # type: ignore[return-value]
    return "deseable"


def _competency(value: object) -> Optional[str]:
    """Return a competency id from a planner's value, or ``None``."""
    text = str(value or "").strip().upper()
    return text if text in COMPETENCY_IDS else None


def _weight(value: object, level: Level) -> int:
    """Return a planner's weight (1–5), or one from the level."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(MIN_WEIGHT, min(MAX_WEIGHT, int(round(value))))
    return {"minimo": 4, "deseable": 3, "opcional": 2}[level]


def _questions(value: object) -> List[str]:
    """Return 1 to 3 usable questions from a planner's value."""
    items = value if isinstance(value, list) else [value]
    found: List[str] = []
    for item in items:
        text = _line(item, QUESTION_CHARS)
        if text and not is_protected(text) and text not in found:
            found.append(text)
    return found[:3]


def draft_points(value: object) -> List[Point]:
    """Return up to :data:`MAX_POINTS` usable concepts from a draft.

    Each item is a string (or an object with ``text``); empty, repeated
    and protected-characteristic ones are dropped, and ids are given in
    order (``p1``, ``p2``…).
    """
    items = value if isinstance(value, list) else []
    found: List[str] = []
    for item in items:
        raw = item.get("text") if isinstance(item, dict) else item
        if not isinstance(raw, str):
            continue
        text = _line(raw, POINT_CHARS).strip(" .;,")
        if not text or is_protected(text) or fold(text) in {
            fold(f) for f in found
        }:
            continue
        found.append(text)
    return [Point(id=f"p{i}", text=text)
            for i, text in enumerate(found[:MAX_POINTS], start=1)]


def draft_slots(raw: object) -> Tuple[List[Slot], int]:
    """Read a planner's ``slots`` into valid slots; count those dropped.

    A slot without a label or a usable question, or about a protected
    characteristic, is dropped; at most :data:`MAX_SLOTS` are kept.
    Its ``points`` are read by :func:`draft_points`.
    """
    slots: List[Slot] = []
    dropped = 0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or len(slots) >= MAX_SLOTS:
            dropped += 1
            continue
        label = _line(item.get("label"), LABEL_CHARS)
        questions = _questions(item.get("questions"))
        if not label or not questions or is_protected(label):
            dropped += 1
            continue
        competency = _competency(item.get("competency"))
        phase = str(item.get("phase") or "")
        level = _level(item.get("level"))
        slots.append(Slot(
            id=f"s{len(slots) + 1}", label=label,
            kind="competency" if competency else "requirement",
            competency=competency, level=level,
            weight=_weight(item.get("weight"), level),
            phase=(phase if phase in PHASES
                   else "behavioural" if competency else "experience"),
            questions=questions,
            evidence=_line(item.get("evidence"), EVIDENCE_CHARS),
            points=draft_points(item.get("points")),
        ))
    if slots and not any(s.level == "minimo" for s in slots):
        slots[0] = slots[0].model_copy(update={"level": "minimo"})
    return slots, dropped


def plan_from_draft(
    draft: Dict[str, Any],
    plan_id: str,
    source: PlanSource,
    minutes: int,
    language: str,
    now: datetime,
    planner: PlannerInfo,
    title: Optional[str] = None,
    role_type: Optional[str] = None,
) -> InterviewPlan:
    """Build a plan from the planner's JSON draft.

    Raises ``ValueError`` when the draft has no usable slot. ``title``
    and ``role_type`` given by the recruiter win over the draft's.
    """
    slots, _ = draft_slots(draft.get("slots"))
    if not slots:
        raise ValueError("the planner's draft has no usable slot")
    kind = role_type or str(draft.get("role_type") or "")
    return InterviewPlan(
        id=plan_id, title=_line(title or draft.get("title"), LABEL_CHARS)
        or "Entrevista",
        role_type=kind if kind in ROLE_TYPES else "otro",
        language=language,
        summary=_line(draft.get("summary"), SUMMARY_CHARS),
        minutes=max(MIN_MINUTES, min(MAX_MINUTES, minutes)),
        source=source, planner=planner, created_at=now, updated_at=now,
        slots=slots,
    )


_STAR = ("Situación concreta", "Lo que hizo él o ella", "Resultado")
"""The points of a competency slot: the parts of a complete example."""
_FALLBACK = (
    ("Experiencia relacionada con el puesto", None, "minimo", "experience",
     "¿Qué experiencia tienes en un puesto parecido y qué tareas hacías "
     "allí?", "Puesto, tiempo y tareas concretas que hacía.",
     ("Dónde trabajó y en qué puesto", "Cuánto tiempo", "Tareas que hacía")),
    ("Disponibilidad y condiciones", None, "minimo", "experience",
     "¿Qué disponibilidad tienes de horario y de incorporación?",
     "Horarios, turnos y fecha en que puede empezar.",
     ("Horario o turnos", "Fecha de incorporación")),
    ("Comunicación clara", "COMM", "minimo", "behavioural",
     "Cuéntame una vez en la que tuviste que explicar algo complicado a "
     "alguien. ¿Qué hiciste tú y cómo salió?",
     "Situación real, lo que explicó y cómo comprobó que se entendió.",
     _STAR),
    ("Resolución de problemas", "SOLVE", "deseable", "behavioural",
     "Cuéntame un problema del trabajo que te tocó resolver. ¿Qué "
     "hiciste paso a paso y qué resultado tuvo?",
     "Problema concreto, sus pasos y el resultado.", _STAR),
    ("Colaboración y apoyo mutuo", "COLLAB", "deseable", "behavioural",
     "Cuéntame una vez en la que ayudaste a un compañero o te apoyaste "
     "en el equipo para sacar algo adelante.",
     "Qué hizo él o ella por el equipo y qué se consiguió.", _STAR),
)


def fallback_plan(
    plan_id: str,
    title: str,
    source: PlanSource,
    minutes: int,
    language: str,
    now: datetime,
    error: Optional[str] = None,
    role_type: Optional[str] = None,
) -> InterviewPlan:
    """A generic plan, for when the planner gives nothing usable.

    Its points are generic too: where, how long and which tasks for
    the experience; the parts of a complete example for a competency.
    """
    slots = [
        Slot(id=f"s{i}", label=label,
             kind="competency" if comp else "requirement",
             competency=comp, level=level, phase=phase,  # type: ignore
             questions=[question], evidence=evidence,
             weight=_weight(None, level),  # type: ignore[arg-type]
             points=draft_points(list(points)))
        for i, (label, comp, level, phase, question, evidence, points)
        in enumerate(_FALLBACK, start=1)
    ]
    return InterviewPlan(
        id=plan_id, title=_line(title, LABEL_CHARS) or "Entrevista",
        role_type=(role_type if role_type in ROLE_TYPES
                   else "otro"),
        language=language, minutes=max(MIN_MINUTES, min(MAX_MINUTES,
                                                        minutes)),
        source=source,
        planner=PlannerInfo(status="fallback", error=error),
        created_at=now, updated_at=now, slots=slots,
    )


def edited_plan(
    current: InterviewPlan, document: Dict[str, Any], now: datetime
) -> InterviewPlan:
    """Return the recruiter's edit of ``current`` (``ValueError`` if bad).

    The edit carries the ``revision`` it was based on; an older one is
    refused (two people editing). Id, source, planner and creation time
    are kept; the revision grows by one.
    """
    if document.get("revision") != current.revision:
        raise StaleRevision(
            f"plan {current.id} is at revision {current.revision}"
        )
    keep = {"schema", "id", "source", "planner", "created_at",
            "updated_at", "revision", "state", "closures"}
    merged = {
        **{k: v for k, v in document.items() if k not in keep},
        "schema": PLAN_SCHEMA, "id": current.id,
        "source": current.source.model_dump(mode="json"),
        "planner": current.planner.model_dump(mode="json"),
        "created_at": current.created_at, "updated_at": now,
        "revision": current.revision + 1,
        "state": current.state,
        "closures": [c.model_dump(mode="json") for c in current.closures],
    }
    plan = InterviewPlan.model_validate(merged)
    for slot in plan.slots:
        if is_protected(slot.label) or any(
            is_protected(q) for q in slot.questions
        ) or any(is_protected(p.text) for p in slot.points):
            raise ValueError(
                f"slot {slot.id} asks about a protected characteristic"
            )
    if not any(s.level == "minimo" for s in plan.slots):
        raise ValueError("mark at least one slot as minimo")
    return plan


class StaleRevision(ValueError):
    """An edit based on an older revision of the plan."""


class Answer(_Strict):
    """One candidate utterance that answers a slot."""

    utterance: str
    quote: str = Field(max_length=QUOTE_CHARS)
    status: Literal["complete", "partial"]


JudgeLabel = Literal[
    "solid", "correct_basic", "wrong", "off_topic", "unclear", "to_verify",
]
"""How accurate an answer was (#259): ``solid`` correct and beyond the
surface for the time available; ``correct_basic`` correct but
superficial; ``wrong`` contains something incorrect; ``off_topic`` does
not answer what was asked; ``unclear`` cannot be judged from what was
said; ``to_verify`` a claim about the candidate's own experience, which
cannot be checked from the interview (never ``wrong``)."""
JUDGE_LABELS: Tuple[JudgeLabel, ...] = (
    "solid", "correct_basic", "wrong", "off_topic", "unclear", "to_verify",
)
REASON_CHARS = 300
Basis = Literal["documents", "general"]


class Judgement(_Strict):
    """A model's judgement of one answer, for the recruiter (#259).

    ``basis`` says what it rests on: retrieved ``documents`` of the
    plan's knowledge domain (``cites``), or the model's ``general``
    knowledge. A draft: the model can be wrong about facts.
    """

    label: JudgeLabel
    reason: str = Field(default="", max_length=REASON_CHARS)
    quote: str = Field(default="", max_length=QUOTE_CHARS)
    utterance: Optional[str] = None
    basis: Basis = "general"
    cites: List[str] = Field(default_factory=list)


class SheetPoint(_Strict):
    """One concept of a slot after the interview (#258).

    ``treated``: the candidate talked about it; ``pending``: not yet;
    ``unknown``: they said they do not know it. ``quote`` and
    ``utterance`` are the words that settled it. ``judgement`` (#259):
    how accurate what they said about it was, for a treated concept
    the report judged. ``by`` (#268): who decided the status
    (:data:`DecidedBy`; ``None`` on sheets written before it).
    """

    id: str
    text: str
    status: PointStatus = "pending"
    quote: str = Field(default="", max_length=QUOTE_CHARS)
    utterance: Optional[str] = None
    judgement: Optional[Judgement] = None
    by: Optional[DecidedBy] = None


class SheetSlot(_Strict):
    """One slot of the sheet: its status, answers and score.

    ``score`` (0–100) is the level shown **relative to what the offer
    asks** (100: meets the requirement for this job), ``above`` says the
    evidence goes beyond it, ``confidence`` (0–1) how sure the scorer
    is and ``scored_by`` which scorer gave it (``model`` or
    ``status``, a coarse estimate from the slot's status). For the
    recruiter only; never said to the candidate.
    """

    id: str
    label: str
    level: Level
    weight: int = DEFAULT_WEIGHT
    competency: Optional[str] = None
    status: SlotStatus = "empty"
    asked: int = 0
    probes: int = 0
    answers: List[Answer] = Field(default_factory=list)
    score: Optional[int] = Field(default=None, ge=0, le=100)
    above: bool = False
    notes: str = Field(default="", max_length=EVIDENCE_CHARS)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scored_by: Optional[Literal["model", "status"]] = None
    points: List[SheetPoint] = Field(default_factory=list)
    """The slot's concepts and whether each was treated (#258)."""
    judgement: Optional[Judgement] = None
    """How accurate the answers to the slot were (#259)."""
    accuracy: Dict[str, int] = Field(default_factory=dict)
    """Counts of the slot's judgements and concept statuses (#259):
    the labels plus ``not_treated`` and ``does_not_know``."""


CheckWhy = Literal["wrong", "contradiction", "vague"]


class CheckItem(_Strict):
    """A statement that is worth checking (#259): a prompt, not a verdict.

    ``why``: it looks incorrect, it contradicts another answer of the
    same interview, or it is confident but vague where specifics were
    asked.
    """

    slot: Optional[str] = None
    quote: str = Field(min_length=1, max_length=QUOTE_CHARS)
    utterance: Optional[str] = None
    why: CheckWhy
    note: str = Field(default="", max_length=REASON_CHARS)


class OtherSkill(_Strict):
    """A skill the candidate showed that the offer did not ask for."""

    label: str = Field(min_length=1, max_length=LABEL_CHARS)
    quote: str = Field(default="", max_length=QUOTE_CHARS)


class Sheet(_Strict):
    """``interview.sheet.v1``: the plan's slots after the interview."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["interview.sheet.v1"] = Field(
        default="interview.sheet.v1", alias="schema"
    )
    session_id: str
    plan_id: str
    plan_revision: int = 0
    title: str
    state: Literal["running", "closed", "stopped"] = "running"
    closed_by: Optional[Literal["covered", "time", "stopped"]] = None
    turns: int = 0
    elapsed_s: float = 0.0
    minutes: int = 20
    minimums_covered: bool = False
    turns_to_minimums: Optional[int] = None
    slots: List[SheetSlot] = Field(default_factory=list)
    kind: Literal["ai", "live"] = "ai"
    """Who held the interview (#262): the AI interviewer (a scored
    sheet), or a person on the live page (the judgement of the answers,
    never a score)."""
    candidate_id: Optional[str] = None
    """The candidate profile of the round, when there is one."""
    filler: str = "heuristic"
    candidate: Optional[str] = Field(default=None, max_length=80)
    suitability: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    summary: str = Field(default="", max_length=1200)
    other_skills: List[OtherSkill] = Field(default_factory=list)
    scorer: Optional[str] = None
    to_check: List[CheckItem] = Field(default_factory=list)
    """«Afirmaciones que conviene comprobar» (#259)."""
    judge: Optional[str] = None
    """The template that judged the answers (none: not judged)."""
    report_error: Optional[str] = Field(default=None, max_length=300)
    """Why the model did not score or judge it (#267): the reasoner's
    status and error kind, never the candidate's words; the scores are
    then the coarse estimate from each slot's status."""
    attribution: Optional[Literal["rules", "after_session"]] = None
    """Who said which utterances answer each point of a live round
    (#268): the model reading the whole round after the session
    (``after_session``), or the word rules of the session (``rules``:
    the model was not asked or failed). ``None``: an AI interviewer's
    sheet, or one written before it."""
    attribution_error: Optional[str] = Field(default=None, max_length=300)
    """Why the model's attribution was not used (its status and error
    kind, never the candidate's words)."""
    finished_at: Optional[datetime] = None

    def to_contract(self) -> Dict[str, Any]:
        """Return the JSON document of the contract."""
        return self.model_dump(mode="json", by_alias=True)


STATUS_SCORE: Dict[str, Optional[int]] = {
    "filled": 70, "partial": 40, "skipped": 10, "unknown": 0,
    "empty": None,
}
"""Coarse score of a slot from its status alone (when no model scored
it); confidence :data:`STATUS_CONFIDENCE`."""
STATUS_CONFIDENCE = 0.2


def suitability(slots: List[SheetSlot]) -> Optional[float]:
    """Return the weighted suitability (0–100) of a sheet's slots.

    Every slot weighs its ``weight``; a slot with no score counts 0
    when it is a ``minimo`` (the interview had to cover it) and is
    left out otherwise. ``None`` when nothing counts.
    """
    total = weights = 0.0
    for slot in slots:
        if slot.score is None and slot.level != "minimo":
            continue
        total += slot.weight * (slot.score or 0)
        weights += slot.weight
    return round(total / weights, 1) if weights else None


def counts(slots: List[SheetSlot]) -> Dict[str, int]:
    """Count what the report shows at a glance.

    ``meets``: score ≥ 70; ``gaps``: a minimo or deseable under 50 or
    without score, or not known/skipped; ``above``: over the
    requirement.
    """
    meets = sum(1 for s in slots if (s.score or 0) >= 70)
    gaps = sum(
        1 for s in slots
        if s.level != "opcional" and (
            s.score is None or s.score < 50
            or s.status in ("unknown", "skipped")
        )
    )
    return {"meets": meets, "gaps": gaps,
            "above": sum(1 for s in slots if s.above),
            "unknown": sum(1 for s in slots if s.status == "unknown"),
            "skipped": sum(1 for s in slots if s.status == "skipped")}


def closed_plan(plan: InterviewPlan, now: datetime) -> InterviewPlan:
    """The plan marked closed (already closed: as it is)."""
    if plan.state == "closed":
        return plan
    return plan.model_copy(update={
        "state": "closed",
        "closures": [*plan.closures, Closure(closed_at=now)],
        "updated_at": now,
    })


def reopened_plan(plan: InterviewPlan, now: datetime) -> InterviewPlan:
    """The plan open again."""
    if plan.state == "open":
        return plan
    closures = list(plan.closures)
    if closures:
        closures[-1] = closures[-1].model_copy(update={"reopened_at": now})
    return plan.model_copy(update={"state": "open", "closures": closures,
                                   "updated_at": now})
