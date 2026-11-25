"""Which concepts of a slot an answer treats, without a model (#258).

A slot of the plan can carry **points**: the concrete concepts a
complete answer talks about («tipos de pisada», «amortiguación y
drop»). :func:`point_hits` says which of them a candidate's words
treat, sentence part by sentence part (split at ``.``, ``;``, ``?``,
``!`` and «pero»):

* a point is **treated** when a part names it: at least half of the
  point's content words (stems of their first five letters, at least
  one) appear in it, or a cue of the point's kind does (a duration for
  «cuánto tiempo», a first-person past action for «lo que hizo», a
  result for «resultado»…; see :data:`_CUES`);
* it is **unknown** («no lo conoce») when the part that names it also
  says they do not know it («el drop no lo conozco»).

This is a hint for the interviewer, not a judgement: "treated" only
means the candidate talked about the concept, never whether what they
said was right. The model filler (``interview.slots@2``) reads the
same points with the meaning of the words; this rule is its fallback
and the live page's matcher.

:func:`points_status` is the completion rule of a slot with points:
filled when every point is treated, ``unknown`` when every point is
unknown, settled ``partial`` when none is pending but some are unknown,
``None`` (still open) while any is pending.

Plain data and functions: no I/O.
"""

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple

from orchestrator.domain.interview import (
    QUOTE_CHARS,
    Point,
    PointStatus,
    SheetPoint,
    SlotStatus,
    fold,
)

MAX_ASKED_POINTS = 3
"""Most concepts one follow-up names at once (natural spoken Spanish)."""
QUOTE_WORDS = 30

_UNKNOWN = re.compile(
    r"\b(no (lo|la|los|las) (conozco|domino|he usado|he utilizado|he "
    r"tocado)|no se (nada|que es|lo que es|de eso|sobre eso)|desconozco|"
    r"no tengo (ni idea|experiencia (con|en|de))|nunca (he|lo he|la he) "
    r"(trabajado|usado|utilizado|tocado)|no sabria (decirte|responder|"
    r"contestar)|no he trabajado (nunca )?(con|en) (eso|esto|ello))"
)
_PARTS = re.compile(r"[.;!?¿¡]+|,?\s+pero\s+|\s+aunque\s+")
_STOP = frozenset(
    "tipo tipos como cual cuales forma formas manera cosas cosa nivel "
    "sobre entre desde hasta para segun algun alguna alguno todos todas "
    "cada otro otra otros otras usar saber hacer tener".split()
)
_PAST = re.compile(
    r"\b(\w{2,}(é|í)|hice|estuve|tuve|fui|puse|propuse|dije|traje|"
    r"pude|supe|quise|conduje|hicimos|estuvimos|tuvimos|fuimos)\b"
)
_RESULT = (
    r"\b(al final|consegui|conseguimos|logre|logramos|resultado|se "
    r"resolvio|funciono|redujimos|reduje|mejoro|mejoramos|gracias a eso|"
    r"desde entonces|aprendi|acabamos|terminamos|cerramos|quedo "
    r"(resuelto|contento|satisfecho)|nos quedamos)"
)
_NUMBER = (r"(\d+|un|una|medio|dos|tres|cuatro|cinco|seis|siete|ocho|"
           r"nueve|diez)")
_CUES: Tuple[Tuple[Pattern[str], Pattern[str]], ...] = tuple(
    (re.compile(point), re.compile(answer)) for point, answer in (
        (r"\b(situacion|contexto)\b",
         r"\b(una vez|cuando|en mi (anterior|ultimo|antiguo)|el ano pasado|"
         r"hace \w+ (anos?|meses)|teniamos|habia una?)\b"),
        (r"\b(resultado|como acabo|como termino)\b", _RESULT),
        (r"\b(cuanto tiempo|duracion|anos de)\b",
         rf"\b{_NUMBER} (anos?|meses|semanas)\b"),
        (r"\b(horario|turnos?)\b",
         r"\b(horarios?|turnos?|mananas?|tardes?|noches?|fines? de semana|"
         r"jornada)\b"),
        (r"\b(incorporacion|empezar)\b",
         r"\b(empezar|incorpor\w*|inmediat\w*|desde ya|en \w+ semanas?)\b"),
        (r"\b(donde|puesto|empresa)\b",
         r"\b(trabaje|trabajaba|estuve|he estado|he trabajado|trabajo)\b"
         r".*\b(en|de|como)\b"),
        (r"\b(tareas|funciones)\b",
         r"\b(me encargaba|me encargue|hacia|atendia|llevaba|tareas|"
         r"funciones|\w{3,}(ando|iendo))\b"),
    )
)
"""(point words, answer words): a point of that kind is treated when the
answer has the cue, whatever the point's own words."""
_ACTION = re.compile(r"\b(lo que hizo|accion|que hizo|su papel|su parte)\b")
"""A point about the candidate's own action: a past-tense verb treats it."""


def says_unknown(text: str) -> bool:
    """Tell whether the candidate says they do not know the topic."""
    folded = fold(text)
    if _UNKNOWN.search(folded):
        return True
    words = folded.replace(",", " ").replace(".", " ").split()
    return len(words) <= 6 and "no" in words and "se" in words


def quote(text: str) -> str:
    """Return the first words of ``text`` as a quote."""
    words = text.split()
    cut = " ".join(words[:QUOTE_WORDS])
    return (cut + (" …" if len(words) > QUOTE_WORDS else ""))[:QUOTE_CHARS]


def _stems(text: str) -> List[str]:
    """The point's content stems: words of 4+ letters, first five."""
    words = re.findall(r"[a-z0-9+#]{4,}", fold(text))
    return list(dict.fromkeys(w[:5] for w in words if w not in _STOP))


def names_point(point: str, part: str) -> bool:
    """Tell whether ``part`` (some words of an answer) names ``point``."""
    folded = fold(part)
    concept = fold(point)
    for cue, answer in _CUES:
        if cue.search(concept) and answer.search(folded):
            return True
    if _ACTION.search(concept) and _PAST.search(part.lower()):
        return True
    stems = _stems(point)
    if not stems:
        return False
    heard = re.findall(r"[a-z0-9+#]{3,}", folded)
    found = sum(1 for s in stems if any(w.startswith(s) for w in heard))
    return found >= max(1, math.ceil(len(stems) / 2))


@dataclass(frozen=True)
class PointHit:
    """One concept of a slot an answer treats or says it does not know."""

    point: str
    status: PointStatus
    quote: str


def parts(text: str) -> List[str]:
    """Split an answer into the parts a point is looked for in."""
    return [p.strip(" ,") for p in _PARTS.split(text) if p.strip(" ,")]


def point_hits(text: str, points: Sequence[Point]) -> List[PointHit]:
    """Say which of ``points`` the answer ``text`` treats (or not knows).

    The first part naming a point decides; a part that names it and
    says they do not know it gives ``unknown``.
    """
    found: List[PointHit] = []
    pieces = parts(text)
    for point in points:
        for piece in pieces:
            if names_point(point.text, piece):
                status: PointStatus = (
                    "unknown" if says_unknown(piece) else "treated")
                found.append(PointHit(point.id, status, quote(piece)))
                break
    return found


def record_hit(
    states: Sequence[SheetPoint], hit: PointHit, utterance: str
) -> bool:
    """Apply one hit to a slot's points; tell whether anything changed.

    A pending point takes the hit's status; an unknown one can become
    treated (they talked about it after all); a treated one stays.
    """
    for state in states:
        if state.id != hit.point:
            continue
        if state.status == "treated" or state.status == hit.status:
            return False
        if state.status == "unknown" and hit.status != "treated":
            return False
        state.status = hit.status
        state.quote = hit.quote[:QUOTE_CHARS]
        state.utterance = utterance
        return True
    return False


def points_status(states: Sequence[SheetPoint]) -> Optional[SlotStatus]:
    """The completion rule of a slot with points (``None``: still open).

    ``filled`` when every point is treated; ``unknown`` when every
    point is unknown; ``partial`` (settled: nothing left to ask) when
    none is pending and some are unknown; ``None`` while any is pending
    or when the slot has no points.
    """
    if not states or any(s.status == "pending" for s in states):
        return None
    treated = sum(1 for s in states if s.status == "treated")
    if treated == len(states):
        return "filled"
    return "partial" if treated else "unknown"


def pending(states: Sequence[SheetPoint]) -> List[SheetPoint]:
    """The points not treated yet, in plan order."""
    return [s for s in states if s.status == "pending"]


def join_words(items: Iterable[str], language: str = "es") -> str:
    """Join ``items`` as spoken: «A», «A y B», «A, B y C»."""
    words = [i for i in items if i]
    last = " and " if language[:2] == "en" else " y "
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + last + words[-1]
