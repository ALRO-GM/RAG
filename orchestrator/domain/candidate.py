"""The candidate profile across interview rounds (#235).

``interview.candidate.v1``.

A candidate for one role (an interview plan) is asked the **core
questions** of KN-4 (``knowledge/interview/core-questions.json``,
``interview.core.v1``) once: the same few questions, with the same
words, to every candidate. Their answers are kept in the candidate's
profile, so a later round shows them as answered and never suggests
them again.

Capturing is deterministic and conservative:

* :func:`asked_core` says which core question an interviewer's sentence
  asks (word patterns per question, Spanish and English, on the folded
  text). Nothing is captured unless the question was heard first.
* :func:`capture` reads the candidate's answer with fixed rules: a clear
  yes or no, a notice period or a start phrase, a salary figure or
  range in euros per year, or the motivation in the candidate's words.
  When the rules are not sure, nothing is captured and the question
  stays open (the interviewer can type the answer).
* A captured answer never replaces an answer already kept, and a
  corrected one (typed by the interviewer) is only changed by another
  correction.
* A «no» to the work mode or location, and every yes/no answer, is kept
  without the candidate's words, so a reason they may have given is
  never stored; a motivation that names a protected characteristic
  (family, health, origin...) is not recorded.

Plain data and functions: no I/O.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.domain.interview import is_protected

SCHEMA = "interview.candidate.v1"
CORE_SCHEMA = "interview.core.v1"
LANGUAGES = ("es", "en")
NAME_CHARS = 80
SUMMARY_CHARS = 280
QUOTE_CHARS = 160
MAX_ROUNDS = 20
"""Rounds kept per candidate (the oldest are dropped)."""
MIN_SALARY, MAX_SALARY = 6000, 500000
"""A yearly figure outside this range is not taken as a salary."""
MIN_MOTIVATION_WORDS = 6

Value = Union[bool, int, float, str, None]
Source = Literal["captured", "corrected"]


def fold(text: str) -> str:
    """Return ``text`` in lower case, without accents, one space apart."""
    parts = unicodedata.normalize("NFKD", str(text).lower())
    plain = "".join(c for c in parts if not unicodedata.combining(c))
    return " ".join(plain.replace("’", "'").split())


# ---- the core questions (interview.core.v1, read as plain data) ------


class CoreField(BaseModel):
    """Where one part of an answer goes in the profile."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    path: str = Field(pattern=r"^[a-z_]+(\.[a-z_]+)+$")
    type: str
    optional: bool = False
    unit: Optional[str] = None
    max_chars: Optional[int] = None


class CoreQuestion(BaseModel):
    """One core question: asked once per candidate, whatever the role."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    order: int = Field(ge=1)
    phase: str
    question: Dict[str, str]
    profile_fields: List[CoreField] = Field(min_length=1)

    def text(self, language: str) -> str:
        """The question's words in ``language`` (Spanish otherwise)."""
        return self.question.get(language) or self.question.get("es", "")

    @property
    def has_blanks(self) -> bool:
        """Whether the words have ``[...]`` blanks the job must fill."""
        return "[" in self.text("es")


def core_questions(data: Mapping[str, Any]) -> List[CoreQuestion]:
    """Read the parsed ``interview.core.v1`` file, in order.

    Raises:
        ValueError: Not that schema, or a question is malformed.
    """
    if data.get("schema") != CORE_SCHEMA:
        raise ValueError(f"not an {CORE_SCHEMA} document")
    raw = data.get("questions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("no core questions")
    questions = [CoreQuestion.model_validate(q) for q in raw]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("core question ids must be unique")
    return sorted(questions, key=lambda q: q.order)


# ---- which core question an interviewer asks -------------------------

ASKED: Dict[str, Tuple[str, ...]] = {
    "work_mode_location": (
        r"(modalidad|presencial|hibrido|teletrabajo|remoto|ubicacion|"
        r"desplazarte|desplazamiento).{0,80}\b(encaja|te va|viene bien|"
        r"te cuadra|te parece|puedes|podrias|ok)\b",
        r"\b(encaja|viene bien|te cuadra).{0,60}(modalidad|presencial|"
        r"hibrido|remoto|ubicacion)",
        r"\b(on-site|hybrid|remote|location|commute)\b.{0,80}\b(work for "
        r"you|ok for you|suit you|fine)\b",
    ),
    "work_authorization": (
        r"\b(autorizacion|permiso) (para|de) trabaj",
        r"\bautorizad[oa] para trabajar\b",
        r"\b(authori[sz]ed|permit|right) to work\b",
        r"\bwork (permit|authori[sz]ation)\b",
    ),
    "start_date_notice": (
        r"\b(incorporarte|incorporacion|incorporar(te)? con nosotros)\b",
        r"\bpreaviso\b",
        r"\bcuando (podrias|puedes) (empezar|comenzar)\b",
        r"\bwhen could you (start|join)\b",
        r"\bnotice (period|you have to give)\b",
    ),
    "motivation": (
        r"\bque te (ha llevado|lleva|llevo|ha animado|motiva|ha motivado)"
        r".{0,40}(presentarte|puesto|candidatura|oferta)",
        r"\bpor que (te has presentado|te presentas|te interesa|quieres)"
        r".{0,30}(puesto|oferta|trabajo)",
        r"\b(what|why) (led|made|brought) you (to )?(apply|here)\b",
        r"\bwhy (did you apply|this job|are you interested)\b",
    ),
    "salary_expectation": (
        r"\bexpectativas? (salarial|economica|de salario|de sueldo)",
        r"\bpretensiones (salariales|economicas)\b",
        r"\bcuanto (esperas|te gustaria|quieres) (cobrar|ganar)\b",
        r"\bbanda salarial\b.{0,40}\?",
        r"\bsalary (do you expect|expectation)",
        r"\bexpected salary\b",
    ),
}
"""Patterns (on the folded sentence) of each core question being asked."""

_COMPILED = {k: [re.compile(p) for p in v] for k, v in ASKED.items()}


def asked_core(
    text: str, questions: List[CoreQuestion]
) -> Optional[str]:
    """Return the id of the core question ``text`` asks, if any.

    Only the questions listed are looked for (the open ones, say); the
    first one in their order that matches wins.
    """
    folded = fold(text)
    for question in questions:
        for pattern in _COMPILED.get(question.id, []):
            if pattern.search(folded):
                return question.id
    return None


# ---- reading an answer ------------------------------------------------


@dataclass(frozen=True)
class Captured:
    """What an answer gave for one core question."""

    fields: Dict[str, Value]
    summary: str = ""
    quote: str = ""


_POSITIVE = (
    r"\bno hay (ningun )?problema\b", r"\bningun problema\b",
    r"\bno me importa\b", r"\bsin (ningun )?problema\b",
    r"\bno problem\b",
)
_YES = re.compile(
    r"^(si|claro|por supuesto|perfecto|vale|de acuerdo|correcto|exacto|"
    r"efectivamente|yes|sure|of course|definitely|absolutely)\b"
    r"|\b(si|me encaja|me viene bien|me cuadra|me va bien|encaja "
    r"perfectamente|sin problema|ningun problema|tengo (la |el )?"
    r"(autorizacion|permiso)|estoy autorizad[oa]|works for me|"
    r"i am authori[sz]ed|i'm authori[sz]ed|i have (a |the )?"
    r"(permit|authori[sz]ation))\b"
)
_NO = re.compile(
    r"\bno (me )?(encaja|puedo|podria|tengo|dispongo|cuento|estoy|"
    r"me viene bien|me viene|viene bien|me va bien|me va|me cuadra)\b|"
    r"\b(doesn't|does not|don't|do not|can't|cannot) (work|suit|have)\b|"
    r"\bnot authori[sz]ed\b|^(no|nope)\b"
)
_DOUBT = re.compile(
    r"\b(tramite|tramitando|pendiente|depende|no se|no estoy segur|"
    r"tendria que (ver|mirarlo|pensarlo)|quizas|quiza|a lo mejor|tal vez|"
    r"pending|in progress|not sure|it depends|maybe)\b"
)


def yes_no(text: str) -> Optional[bool]:
    """Return a clear yes or no, ``None`` when unclear or both."""
    folded = fold(text).strip(" .,!¡¿?")
    if _DOUBT.search(folded):
        return None
    cleaned = folded
    for phrase in _POSITIVE:
        cleaned = re.sub(phrase, " sin problema ", cleaned)
    no = bool(_NO.search(cleaned))
    yes = bool(_YES.search(_NO.sub(" ", cleaned)))
    if yes == no:
        return None
    return yes


_NUMBER_WORDS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "quince": 15, "veinte": 20, "treinta": 30, "a": 1, "an": 1, "one": 1,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8,
    "ten": 10, "fifteen": 15, "thirty": 30,
}
_UNIT_DAYS = {
    "dia": 1, "dias": 1, "day": 1, "days": 1, "semana": 7, "semanas": 7,
    "week": 7, "weeks": 7, "mes": 30, "meses": 30, "month": 30,
    "months": 30, "quincena": 15,
}
_WORDS = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_AMOUNT = r"(\d{1,3}|" + _WORDS + r")"
_UNIT = r"(dias?|semanas?|mes(?:es)?|quincena|days?|weeks?|months?)"
_NOTICE = (
    re.compile(_AMOUNT + r" " + _UNIT + r"(?: \w+){0,2} (?:de )?preaviso"),
    re.compile(r"preaviso (?:\w+ ){0,3}?(?:de )?" + _AMOUNT + r" " + _UNIT),
    re.compile(_AMOUNT + r" " + _UNIT + r"'? notice"),
    re.compile(r"notice (?:\w+ ){0,3}?(?:of )?" + _AMOUNT + r" " + _UNIT),
)
_NO_NOTICE = re.compile(
    r"\b(no tengo (que dar )?preaviso|sin preaviso|no (hay|necesito) "
    r"preaviso|no notice)\b"
)
_NOW = re.compile(
    r"\b(de inmediato|inmediatamente|incorporacion inmediata|ya mismo|"
    r"desde ya|desde manana|cuando querais|cuando quieras|manana mismo|"
    r"immediately|right away|straight away|tomorrow)\b"
)
_START = (
    re.compile(r"\b(en|dentro de) " + _AMOUNT + r" " + _UNIT + r"\b"),
    re.compile(r"\ba (principios|mediados|finales) de \w+"),
    re.compile(r"\b(a partir del?|desde el) (\d{1,2} de \w+|lunes|martes|"
               r"miercoles|jueves|viernes|\w+ que viene|proximo mes)"),
    re.compile(r"\bin " + _AMOUNT + r" " + _UNIT + r"\b"),
    re.compile(r"\b(from|on|at the (start|beginning|end) of) "
               r"(january|february|march|april|may|june|july|august|"
               r"september|october|november|december|monday|next month)"),
)


def _days(amount: str, unit: str) -> int:
    """Return ``amount`` ``unit`` (``dos semanas``) as days."""
    number = (int(amount) if amount.isdigit()
              else _NUMBER_WORDS.get(amount, 0))
    return number * _UNIT_DAYS.get(unit.rstrip("'"), 0)


def start_and_notice(text: str) -> Optional[Captured]:
    """Read when the candidate could start and their notice period."""
    folded = fold(text)
    notice: Optional[int] = 0 if _NO_NOTICE.search(folded) else None
    if notice is None:
        for pattern in _NOTICE:
            found = pattern.search(folded)
            if found:
                notice = _days(found.group(1), found.group(2)) or None
                break
    start: Optional[str] = None
    now = _NOW.search(folded)
    if now:
        start = now.group(0)
    else:
        for pattern in _START:
            found = pattern.search(folded)
            if found and "preaviso" not in folded[found.end():][:12]:
                start = found.group(0)
                break
    if start is None and notice is None:
        return None
    fields: Dict[str, Value] = {"availability.earliest_start": start}
    if notice is not None:
        fields["availability.notice_days"] = notice
    return Captured(fields, summary=start or "", quote=_quote(text))


_THOUSANDS = re.compile(r"(?<=\d)[.\s](?=\d{3}\b)")
_FIGURE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(k\b|mil\b|000\b)?"
)
_MONTHLY = re.compile(r"\b(al mes|mensual(es)?|por mes|a month|per month|"
                      r"monthly)\b")
_DECLINE = re.compile(
    r"\b(prefiero no|no quiero (decir|dar)|no lo se|no sabria|depende|"
    r"lo hablamos|rather not|prefer not)\b"
)
_SPAN = re.compile(r"\b(entre|between|de)\b|\d\s*(-|a|y|to|and)\s*\d")


def salary(text: str) -> Optional[Captured]:
    """Read a yearly salary figure or range in euros.

    ``30.000``, ``30 mil``, ``30k``, ``entre 28 y 32 mil``: a range when
    two figures are said with «entre», «y», «a» or a dash; a «mil» or
    «k» after the last figure applies to the bare figures before it.
    A monthly figure, a refusal or anything outside 6 000–500 000 is not
    taken.
    """
    folded = fold(text)
    if _DECLINE.search(folded) or _MONTHLY.search(folded):
        return None
    plain = _THOUSANDS.sub("", folded)
    figures: List[Tuple[float, bool]] = []
    for found in _FIGURE.finditer(plain):
        number = float(found.group(1).replace(",", "."))
        scaled = found.group(2) in ("k", "mil")
        figures.append((number * 1000 if scaled else number, scaled))
    if not figures or len(figures) > 2:
        return None
    if len(figures) == 2 and figures[1][1] and not figures[0][1] \
            and figures[0][0] < 1000:
        figures[0] = (figures[0][0] * 1000, True)
    values = sorted(v for v, _ in figures)
    if any(not MIN_SALARY <= v <= MAX_SALARY for v in values):
        return None
    if len(values) == 2 and not _SPAN.search(plain):
        return None
    low, high = int(values[0]), int(values[-1])
    return Captured(
        {"salary.expectation_min": low, "salary.expectation_max": high},
        quote=_quote(text),
    )


_PRIVATE = re.compile(
    r"\b(hij[oa]s?|bebe|embaraz\w*|pareja|marido|mujer|esposa?|novi[oa]|"
    r"casad[oa]|divorci\w*|separad[oa]|viud[oa]|madre|padre|familia\w*|"
    r"cuidar|cuido|salud|enferm\w*|baja|tratamiento|discapacidad|"
    r"religi\w*|iglesia|mezquita|sindicat\w*|afiliad[oa]|partido|"
    r"nacionalidad|extranjer[oa]|inmigra\w*|anos tengo|mi edad|"
    r"children|kids?|pregnan\w*|husband|wife|partner|family|health|"
    r"illness|religion|union|nationality|my age)\b"
)
"""Words of a personal circumstance an answer may bring up: the
motivation is then not recorded (KN-4's legal note)."""


def motivation(text: str) -> Optional[Captured]:
    """Keep the motivation in the candidate's words (280 characters).

    Nothing is kept for a short answer or when the answer names a
    protected characteristic (it is not followed up nor recorded).
    """
    words = text.split()
    if (len(words) < MIN_MOTIVATION_WORDS or is_protected(text)
            or _PRIVATE.search(fold(text))):
        return None
    summary = " ".join(words)
    if len(summary) > SUMMARY_CHARS:
        summary = summary[:SUMMARY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return Captured({"motivation.summary": summary}, summary=summary)


def _quote(text: str) -> str:
    """The first characters of an answer, cut at a word."""
    clean = " ".join(text.split())
    if len(clean) <= QUOTE_CHARS:
        return clean
    return clean[:QUOTE_CHARS - 1].rsplit(" ", 1)[0] + "…"


def capture(question_id: str, text: str) -> Optional[Captured]:
    """Read the answer ``text`` to a core question, conservatively.

    ``None`` when the rules are not sure: the question stays open.
    Yes/no answers keep no quote (a reason given is never stored).
    """
    if not text.strip():
        return None
    if question_id == "work_mode_location":
        answer = yes_no(text)
        if answer is None:
            return None
        return Captured({"availability.work_mode_ok": answer,
                         "availability.location_note": None})
    if question_id == "work_authorization":
        answer = yes_no(text)
        if answer is None:
            return None
        return Captured({"eligibility.work_authorization": answer})
    if question_id == "start_date_notice":
        return start_and_notice(text)
    if question_id == "motivation":
        return motivation(text)
    if question_id == "salary_expectation":
        return salary(text)
    return None


# ---- the profile -------------------------------------------------------


class _Strict(BaseModel):
    """Base of the profile models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class CoreAnswer(_Strict):
    """The kept answer to one core question.

    ``source`` is ``captured`` (read from the conversation) or
    ``corrected`` (typed by the interviewer); ``quote`` is the
    candidate's words when they may be kept (never for a yes/no).
    """

    question: str
    fields: Dict[str, Value] = Field(default_factory=dict)
    summary: str = Field(default="", max_length=SUMMARY_CHARS)
    quote: str = Field(default="", max_length=QUOTE_CHARS)
    source: Source = "captured"
    session_id: Optional[str] = None
    at: str


class Round(_Strict):
    """One interview of the candidate for this role.

    ``covered`` are the plan's slots the round gave evidence for;
    ``weak`` those left open or below the required level (they lead the
    next round); ``talk`` the share of speaking time per role (0–1).
    """

    session_id: str
    at: str
    kind: Literal["live", "bot"] = "live"
    covered: List[str] = Field(default_factory=list)
    weak: List[str] = Field(default_factory=list)
    talk: Dict[str, float] = Field(default_factory=dict)


class CandidateProfile(_Strict):
    """``interview.candidate.v1``: one candidate for one role."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["interview.candidate.v1"] = Field(
        default="interview.candidate.v1", alias="schema"
    )
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    plan_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    name: str = Field(min_length=1, max_length=NAME_CHARS)
    created_at: str
    updated_at: str
    answers: Dict[str, CoreAnswer] = Field(default_factory=dict)
    rounds: List[Round] = Field(default_factory=list)

    def to_contract(self) -> Dict[str, Any]:
        """Return the JSON document."""
        return self.model_dump(mode="json", by_alias=True)


def stamp(at: datetime) -> str:
    """Return ``at`` as ISO 8601 to the second."""
    return at.isoformat(timespec="seconds")


def clean_name(name: object) -> str:
    """Return a candidate's name on one line (``ValueError`` if empty)."""
    text = " ".join(str(name or "").split())[:NAME_CHARS]
    if not text:
        raise ValueError("a candidate needs a name")
    return text


def candidate_id(name: str, taken: List[str]) -> str:
    """A new id from the name: ``lucia-marin``, then ``lucia-marin-2``."""
    slug = re.sub(r"[^a-z0-9]+", "-", fold(name)).strip("-")[:56] or "c"
    found, number = slug, 1
    while found in taken:
        number += 1
        found = f"{slug}-{number}"
    return found


def new_profile(
    plan_id: str, name: str, taken: List[str], now: datetime
) -> CandidateProfile:
    """A candidate with no answer and no round yet."""
    clean = clean_name(name)
    return CandidateProfile(
        id=candidate_id(clean, taken), plan_id=plan_id, name=clean,
        created_at=stamp(now), updated_at=stamp(now),
    )


def with_capture(
    profile: CandidateProfile, question_id: str, found: Captured,
    session_id: Optional[str], now: datetime,
) -> Optional[CandidateProfile]:
    """Keep a captured answer; ``None`` when the question has one.

    A capture never replaces an answer already kept (captured or
    corrected): a wrong one is fixed by a correction.
    """
    if question_id in profile.answers:
        return None
    answer = CoreAnswer(
        question=question_id, fields=dict(found.fields),
        summary=found.summary[:SUMMARY_CHARS],
        quote=found.quote[:QUOTE_CHARS], source="captured",
        session_id=session_id, at=stamp(now),
    )
    return profile.model_copy(update={
        "answers": {**profile.answers, question_id: answer},
        "updated_at": stamp(now),
    })


def with_correction(
    profile: CandidateProfile, question: CoreQuestion, text: str,
    now: datetime,
) -> CandidateProfile:
    """Keep the interviewer's answer to ``question`` (typed words).

    The typed words are read with the same rules as a capture to fill
    the fields; when they say nothing the rules understand, they are
    kept as the summary. A yes/no keeps no other words.
    """
    words = " ".join(str(text or "").split())[:SUMMARY_CHARS]
    if not words:
        raise ValueError("a correction needs the answer")
    found = capture(question.id, words)
    if question.id == "motivation":
        found = Captured({"motivation.summary": words}, summary=words)
    fields = dict(found.fields) if found else {}
    boolean = any(f.type == "boolean" for f in question.profile_fields)
    summary = "" if boolean and found else words
    answer = CoreAnswer(
        question=question.id, fields=fields, summary=summary,
        source="corrected", session_id=None, at=stamp(now),
    )
    return profile.model_copy(update={
        "answers": {**profile.answers, question.id: answer},
        "updated_at": stamp(now),
    })


def without_answer(
    profile: CandidateProfile, question_id: str, now: datetime
) -> CandidateProfile:
    """Forget the answer to one question (it is open again)."""
    answers = {k: v for k, v in profile.answers.items() if k != question_id}
    return profile.model_copy(update={"answers": answers,
                                      "updated_at": stamp(now)})


def with_round(
    profile: CandidateProfile, kept: Round, now: datetime
) -> CandidateProfile:
    """Add (or replace, by session) one round; the newest are kept."""
    rounds = [r for r in profile.rounds if r.session_id != kept.session_id]
    rounds = (rounds + [kept])[-MAX_ROUNDS:]
    return profile.model_copy(update={"rounds": rounds,
                                      "updated_at": stamp(now)})


def open_questions(
    profile: CandidateProfile, questions: List[CoreQuestion]
) -> List[CoreQuestion]:
    """The core questions the profile has no answer to, in order."""
    return [q for q in questions if q.id not in profile.answers]


def expired(
    profile: CandidateProfile, now: datetime, days: Optional[int]
) -> bool:
    """Whether the profile is older than the retention (``None``: never)."""
    if days is None:
        return False
    try:
        updated = datetime.fromisoformat(profile.updated_at)
    except ValueError:
        return True
    if (updated.tzinfo is None) != (now.tzinfo is None):
        updated = updated.replace(tzinfo=now.tzinfo)
    return now - updated > timedelta(days=days)


def view(
    profile: CandidateProfile, questions: List[CoreQuestion]
) -> Dict[str, Any]:
    """The profile for the pages: each core question with its answer."""
    rows = []
    for q in questions:
        kept = profile.answers.get(q.id)
        rows.append({
            "id": q.id, "order": q.order, "phase": q.phase,
            "question": dict(q.question),
            "fields": [f.model_dump() for f in q.profile_fields],
            "answer": kept.model_dump(mode="json") if kept else None,
        })
    return {
        "schema": SCHEMA, "id": profile.id, "plan_id": profile.plan_id,
        "name": profile.name, "created_at": profile.created_at,
        "updated_at": profile.updated_at, "questions": rows,
        "answered": sum(1 for r in rows if r["answer"]),
        "rounds": [r.model_dump(mode="json") for r in profile.rounds],
    }
