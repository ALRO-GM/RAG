"""Which concepts of a slot an answer treats, without a model (#258).

A slot of the plan can carry **points**: the concrete concepts a
complete answer talks about («tipos de pisada», «amortiguación y
drop»). :func:`point_hits` says which of them a candidate's words
treat, sentence part by sentence part (split at ``.``, ``;``, ``?``,
``!`` and «pero»):

* a point is **treated** when a part names it: **every** content word
  of the point when it has one or two (stems of their first five
  letters; «tipo», «cómo», «tiempo»… do not count), at least half when
  it has more; or a cue of the point's kind (a figure with years or
  months for «cuánto tiempo», a first-person past verb for «lo que
  hizo», a result verb for «resultado»…; see :data:`_CUES`). Filler
  words never count: «así», «aquí», «qué» are not past verbs, «cuando»
  alone is not a situation, «hacia» is not «hacía», a gerund is not a
  task, «esta tarde» is not a shift;
* it is **unknown** («no lo conoce») when a part says they do not know
  it and names one of its content words («el drop no lo conozco»).

The rules prefer missing a concept (it stays pending and is asked
about) to ticking one the candidate did not talk about.

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
    DecidedBy,
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
    "cada otro otra otros otras usar saber hacer tener cuanto cuantos "
    "cuanta cuantas donde cuando quien bien mucho mucha muchos muchas "
    "poco algo todo esta este estos estas aqui alli hacia tiempo veces "
    "siempre nunca".split()
)
_PAST = re.compile(
    r"\b(\w{3,}(é|í)|hice|estuve|tuve|fui|puse|propuse|dije|traje|"
    r"pude|supe|quise|conduje|hicimos|estuvimos|tuvimos|fuimos)\b"
)
"""First-person past verbs (on the unfolded text: «expliqué», «decidí»)."""
_NOT_PAST = frozenset(
    "aquí allí ahí así qué porqué café bebé comité esquí maní rubí "
    "jabalí colibrí también".split()
)
"""Words ending in «é»/«í» that are not past verbs."""
_RESULT = (
    r"\b(consegui|conseguimos|logre|logramos|resultado|se "
    r"resolvio|funciono|redujimos|reduje|mejoro|mejoramos|gracias a eso|"
    r"desde entonces|aprendi|acabamos|terminamos|cerramos|quedo "
    r"(resuelto|contento|satisfecha?)|nos quedamos)"
)
_NUMBER = (r"(\d+|un|una|medio|dos|tres|cuatro|cinco|seis|siete|ocho|"
           r"nueve|diez)")
_CUES: Tuple[Tuple[Pattern[str], Pattern[str]], ...] = tuple(
    (re.compile(point), re.compile(answer)) for point, answer in (
        (r"\b(situacion|contexto)\b",
         r"\b(una vez|cuando (trabajaba|estaba|estuve|era|tenia|llegue|"
         r"empece)|en mi (anterior|ultimo|antiguo)|el ano pasado|"
         r"hace \w+ (anos?|meses))\b"),
        (r"\b(resultado|como acabo|como termino)\b", _RESULT),
        (r"\b(cuanto tiempo|duracion|anos de)\b",
         rf"\b((durante|estuve|trabaje|trabajaba|he estado|he trabajado|"
         rf"pase)( \w+){{0,3}} {_NUMBER} (anos?|meses|semanas)|"
         rf"llevo {_NUMBER} (anos?|meses|semanas) (trabajando|en|con|"
         rf"como)|{_NUMBER} (anos?|meses|semanas) (trabajando|de "
         rf"experiencia|en (la|el|una?) \w+))\b"),
        (r"\b(horario|turnos?)\b",
         r"\b(horarios?|turnos?|(de|por las) (mananas|tardes|noches)|"
         r"fines? de semana|sabados|domingos|festivos|jornada)\b"),
        (r"\b(incorporacion|empezar)\b",
         r"\b((empezar|incorporarme) (en|el|la|ya|manana|a partir|de "
         r"inmediato|cuando (querais|necesiteis))|incorporacion inmediata|"
         r"de inmediato|inmediatamente|desde ya)\b"),
        (r"\b(donde|puesto|empresa)\b",
         r"\b((trabaje|trabajaba|he trabajado) (de|como|en)|"
         r"estuve( \w+){0,3} en (un|una|el|la|caja))\b"),
        (r"\b(tareas|funciones)\b",
         r"\b(me encargaba|me encargue|me ocupaba|atendia|mis tareas|"
         r"mis funciones|(atendiendo|reponiendo|cobrando|vendiendo|"
         r"preparando|limpiando|gestionando|organizando|cargando|"
         r"descargando|montando|cocinando|sirviendo) \w+)\b"),
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


def _past(part: str) -> bool:
    """Whether ``part`` has a first-person past verb (not «así», «qué»)."""
    return any(m.group(0) not in _NOT_PAST
               for m in _PAST.finditer(part.lower()))


def names_point(point: str, part: str, loose: bool = False) -> bool:
    """Tell whether ``part`` (some words of an answer) names ``point``.

    Strict (the default, for "treated"): all the point's stems when it
    has one or two, half of them when more, or a cue of its kind.
    ``loose`` (for "does not know it"): one stem is enough.
    """
    folded = fold(part)
    concept = fold(point)
    for cue, answer in _CUES:
        if cue.search(concept) and answer.search(folded):
            return True
    if _ACTION.search(concept) and _past(part):
        return True
    stems = _stems(point)
    if not stems:
        return False
    heard = re.findall(r"[a-z0-9+#]{3,}", folded)
    found = sum(1 for s in stems if any(w.startswith(s) for w in heard))
    if loose:
        return found >= 1
    needed = len(stems) if len(stems) <= 2 else math.ceil(len(stems) / 2)
    return found >= needed


@dataclass(frozen=True)
class PointHit:
    """One concept of a slot an answer treats or says it does not know.

    ``by``: who decided it (#268), the rules unless a model said so.
    """

    point: str
    status: PointStatus
    quote: str
    by: DecidedBy = "rules"


def parts(text: str) -> List[str]:
    """Split an answer into the parts a point is looked for in."""
    return [p.strip(" ,") for p in _PARTS.split(text) if p.strip(" ,")]


def point_hits(text: str, points: Sequence[Point]) -> List[PointHit]:
    """Say which of ``points`` the answer ``text`` treats (or not knows).

    The first part naming a point decides: a part saying they do not
    know it and naming one of its words gives ``unknown``; any other
    part naming it (strictly) gives ``treated``.
    """
    found: List[PointHit] = []
    pieces = parts(text)
    for point in points:
        for piece in pieces:
            unknown = says_unknown(piece)
            if names_point(point.text, piece, loose=unknown):
                status: PointStatus = "unknown" if unknown else "treated"
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
        state.by = hit.by
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
