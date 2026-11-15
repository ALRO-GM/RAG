"""Speakers' names from what they say: candidates, evidence, decision.

After a transcription the voices nobody enrolled are ``S1``, ``S2``…
People often say who they are ("soy Ana", "me llamo Pedro", "aquí Luis
de ventas", "I'm Kate") or name the person they talk to ("gracias,
Marta", "os presento a Ana"). This module (TR-20, issue #102) holds the
pure part of turning that into names:

1. :func:`find_candidates` picks, with rules, the sentences that may
   hold a name, each with a little context, so a model reads a few
   fragments instead of the whole transcript. The rules also make
   their own :class:`NameEvidence` (:func:`rule_evidence`), the
   baseline the model is measured against.
2. :func:`model_prompt` and :func:`parse_reply` are what is asked of a
   local chat model about one candidate and how its JSON answer is
   checked. The model only says whether there is a name, of which kind
   and how sure it is; *who* the name belongs to is fixed here
   (:func:`target_speaker`): a self-introduction names the speaker of
   the sentence, a name said to someone else names the neighbour
   spoken to. A name the sentence does not contain is dropped, so the
   model cannot invent one.
3. :func:`decide` applies the precision-first policy: a name is given
   automatically only from a clear self-introduction that conflicts
   with nothing (one name for the voice, no other voice claiming it,
   not already the name of another speaker, no name said to the voice
   that differs). Everything else is a :class:`NameSuggestion` with
   its quote and time, for one click. Enrolled speakers and names set
   by hand are never touched.

Nothing here does I/O; the use case (``application/speaker_names.py``)
reads the transcript, calls the model and applies the names.
"""

import json
import re
import unicodedata
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from transcriber.domain.models import Speaker, Transcript, Utterance

Kind = Literal["self", "addressed"]
"""``self``: the speaker says their own name; ``addressed``: the
speaker says the name of someone else present (thanking, asking or
introducing them)."""

Source = Literal["rules", "model"]
"""Who produced a piece of evidence."""

Towards = Literal["self", "previous", "next"]
"""Whose name it is: the speaker's own, or the person who spoke just
before or speaks just after."""

Reason = Literal[
    "addressed", "unsure", "several_names", "same_name", "name_taken",
    "conflict", "rules_only", "undone",
]
"""Why a name is only suggested:

``addressed``
    only said by someone else (weaker evidence);
``unsure``
    a self-introduction below the confidence for automatic names;
``several_names``
    the voice introduced itself with more than one name;
``same_name``
    another voice was also given this name (by itself or by others);
``name_taken``
    another speaker of the transcript already has this name;
``conflict``
    others called this voice by a different name;
``rules_only``
    the model could not run: rules alone never name anyone;
``undone``
    it was given automatically and a person undid it (it is never given
    automatically to that voice again, see :class:`UndoneName`).
"""

NAME_WORD = r"[A-ZÁÉÍÓÚÑÜ][a-záéíóúñüçàèòï'-]+"
"""A capitalised word: names are written with a capital by the
recogniser, and lower-case look-alikes ("me llamo la atención") are not
names."""

NAME = rf"(?P<name>{NAME_WORD}(?:\s+{NAME_WORD})?)"
"""A first name, optionally followed by a second capitalised word."""

NOT_NAMES = frozenset({
    # Spanish words that start sentences or follow "soy"/"aquí"
    "a", "ah", "además", "ahora", "al", "algo", "alguien", "allí",
    "aquí", "así", "bien", "bueno", "buenas", "buenos", "claro", "como",
    "cómo", "con", "creo", "cuando", "de", "del", "dime", "disculpa",
    "disculpad", "ejemplo", "el", "él", "ella", "ellos", "en", "entonces",
    "es", "esa", "ese", "eso", "esta", "está", "este", "esto", "exacto",
    "gente", "gracias", "hola", "hombre", "hoy", "igual", "la", "las",
    "lo", "los", "luego", "mañana", "mira", "mire", "mujer", "muy",
    "nada", "no", "nos", "nosotros", "oye", "para", "pero", "perdón",
    "perdona", "perdone", "por", "porque", "pues", "que", "qué", "quien",
    "quién", "se", "señor", "señora", "si", "sí", "sin", "sobre",
    "también", "tampoco", "te", "todo", "todos", "tú", "tu", "un",
    "una", "uno", "usted", "vale", "vamos", "venga", "vosotros", "y",
    "ya", "yo", "soy", "me", "mi", "i'm", "equipo", "chicos", "chicas",
    "compañeros", "todas",
    "jefe", "jefa", "doctor", "doctora", "profe", "profesor",
    "profesora", "madrid", "barcelona", "españa",
    # English
    "and", "but", "everyone", "everybody", "fine", "good", "great",
    "guys", "hello", "hey", "hi", "i", "it", "just", "no", "not", "now",
    "oh", "ok", "okay", "right", "so", "sorry", "sure", "thanks", "the",
    "then", "there", "this", "well", "yes", "yeah", "you", "team",
    "here", "sir", "madam",
    # numbers and other words that open answers ("Doce, en realidad.")
    "cero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
    "nueve", "diez", "once", "doce", "trece", "catorce", "quince",
    "veinte", "treinta", "cien", "mil", "primero", "segundo", "tercero",
    "total", "perfecto", "genial", "fenomenal", "estupendo", "cierto",
    "verdad", "precisamente", "ojo", "fíjate", "fijate", "después",
    "antes", "aunque", "igualmente", "obvio", "seguro", "bastante",
    "mucho", "poco", "otra", "otro", "lunes", "martes", "miércoles",
    "jueves", "viernes", "sábado", "domingo", "enero", "febrero",
    "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
    "octubre", "noviembre", "diciembre", "one", "two", "three", "first",
    "second", "maybe", "exactly", "absolutely", "true", "agreed",
    # interjections and greetings that open a sentence before a comma
    "ja", "jaja", "je", "eh", "uf", "bah", "anda", "vaya", "ostras",
    "encantado", "encantada", "tranquilo", "tranquila", "bienvenido",
    "bienvenida", "mejor", "welcome", "nice", "please", "cheers",
})
"""Capitalised words that are never taken as a name (lower case, no
accents are folded: compare with :func:`fold`)."""

SELF_PATTERNS: Tuple[Tuple[str, float], ...] = (
    (rf"\bme llamo\s+{NAME}", 0.9),
    (rf"\bmi nombre es\s+{NAME}", 0.9),
    (rf"\bmy name is\s+{NAME}", 0.9),
    (rf"\bsoy\s+(?:yo,?\s+)?{NAME}", 0.8),
    (rf"\b(?:I'm|I am)\s+{NAME}", 0.8),
    (rf"\b(?:aquí|aqui)\s+{NAME}", 0.6),
    (rf"\b(?:os|le|les|te)?\s*habla\s+{NAME}", 0.6),
    (rf"\bthis is\s+{NAME}\s+(?:speaking|here)\b", 0.7),
    (rf"^\s*{NAME}\s+(?:here|aquí|al habla)\b", 0.6),
)
"""Self-introduction patterns with the rules' confidence. The trigger
is matched without regard to case; the name must be capitalised."""

ADDRESSED_PATTERNS: Tuple[Tuple[str, Literal["previous", "next"]], ...] = (
    (rf"\b(?:os|te|les|le) presento a\s+{NAME}", "next"),
    (rf"\b(?:let me introduce|meet|this is)\s+{NAME}", "next"),
    (rf"\b(?:gracias|muchas gracias|thanks|thank you|vale|de acuerdo|"
     rf"perdona|perdón|sorry|exacto|claro)\s*,\s*{NAME}\b", "previous"),
    (rf"\b(?:hola|buenas|buenos días|hi|hello|hey)\s*,?\s+{NAME}\b",
     "next"),
    (rf"^\s*¿?\s*{NAME}\s*,", "next"),
    (rf"\b(?:y tú|y usted|and you|what about you)\s*,\s*{NAME}\b", "next"),
    (rf",\s*{NAME}\s*[?¿]", "next"),
    (rf",\s*{NAME}\s*[.!]*\s*$", "previous"),
)
"""Patterns of a name said to someone else, and which neighbour it
points at: thanks and acknowledgements answer the person who just
spoke (``previous``); questions, greetings and introductions call on
the person who speaks next (``next``)."""

NAME_MODELS: Tuple[str, ...] = ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
"""The only local models allowed to read transcripts for names, in
preference order: the 4B only when the 8B does not fit, never a smaller
one (the owner's rule on #102)."""

ADDRESSED_CONFIDENCE = 0.5
"""The rules' confidence in a name said to someone else."""


def fold(text: str) -> str:
    """Return ``text`` in lower case without accents, for comparisons."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _never_a_name(word: str) -> bool:
    """Tell whether ``word`` is never taken as a name."""
    folded = fold(word)
    return folded in NOT_NAMES or folded.endswith("mente")


def clean_name(name: str) -> Optional[str]:
    """Return ``name`` tidied, or ``None`` if it is not a usable name.

    The name is its first one or two words that are capitalised and
    are not words that are never names (articles, greetings, numbers,
    discourse markers: :data:`NOT_NAMES`, adverbs in *-mente*); a name
    whose first word is not one is refused.
    """
    words: List[str] = []
    for raw in name.split():
        word = raw.strip(".,;:!?¿¡\"'«»()")
        if not re.fullmatch(NAME_WORD, word) or _never_a_name(word):
            break
        words.append(word)
        if len(words) == 2 or word != raw:
            break
    return " ".join(words) if words else None


def same_name(a: str, b: str) -> bool:
    """Tell whether two names are the same person's name.

    Case and accents do not count, and a first name matches the same
    first name with a surname ("Ana" and "Ana García").
    """
    first_a, first_b = fold(a).split(), fold(b).split()
    if not first_a or not first_b:
        return False
    if len(first_a) > 1 and len(first_b) > 1:
        return first_a == first_b
    return first_a[0] == first_b[0]


def contains_name(text: str, name: str) -> bool:
    """Tell whether ``name`` appears in ``text`` as whole words."""
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in fold(name).split())
    return re.search(pattern + r"\b", fold(text)) is not None


class NameSearchParameters(BaseModel):
    """How names are looked for and when one is given automatically.

    ``auto_confidence``: the model's confidence a self-introduction
    needs for an automatic name. ``context``: sentences shown to the
    model before and after a candidate. ``max_candidates``: at most so
    many sentences are read per session. ``max_tokens``: the length of
    the model's answer. Untuned starting values; the measurement of
    ``docs/transcriber/NAMES.md`` uses them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auto_confidence: float = Field(default=0.8, ge=0, le=1)
    context: int = Field(default=1, ge=0, le=5)
    max_candidates: int = Field(default=60, ge=1, le=500)
    max_tokens: int = Field(default=96, ge=16, le=512)


class NameCandidate(NamedTuple):
    """A sentence that may hold a name, with its context.

    ``kind`` and ``name`` are what the rules saw (the model decides);
    ``towards`` is which neighbour an ``addressed`` name points at.
    ``before`` and ``after`` are the neighbouring sentences.
    """

    utterance: Utterance
    kind: Kind
    name: str
    confidence: float
    towards: Towards
    before: Tuple[Utterance, ...]
    after: Tuple[Utterance, ...]
    span: Tuple[int, int] = (0, 0)


class NameEvidence(BaseModel):
    """One sign that a voice has a name.

    ``speaker`` is the voice the name belongs to (not necessarily who
    said it), ``utterance`` and ``start`` where it was said, ``quote``
    the words that show it.
    """

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    kind: Kind
    quote: str = Field(min_length=1, max_length=300)
    utterance: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    source: Source


class AutoName(BaseModel):
    """A name given automatically, kept so it can be shown and undone."""

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=300)
    utterance: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class NameSuggestion(BaseModel):
    """A name proposed for a voice, accepted or not by a person."""

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    kind: Kind
    quote: str = Field(min_length=1, max_length=300)
    utterance: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: Reason
    count: int = Field(default=1, ge=1)


class UndoneName(BaseModel):
    """An automatic name a person undid for one voice.

    :func:`decide` never gives that name (or the same first name)
    automatically to ``speaker`` again; it may still be suggested.
    ``run`` is the transcription run of the transcript it was undone
    in: after a new transcription the labels may belong to other
    people, so the entry no longer applies (:func:`kept_undone`).
    """

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=100)
    run: Optional[str] = None


class SpeakerNames(BaseModel):
    """The outcome of the last search for names in a session.

    ``model`` is the model that read the candidates (``None``: rules
    only), ``seconds`` how long the search took, ``candidates`` how
    many sentences were read, ``note`` why the model was not used, if
    it was not. ``undone`` lists the automatic names people undid,
    which later searches never give automatically again.
    """

    model_config = ConfigDict(frozen=True)

    session: str = Field(min_length=1)
    searched_at: AwareDatetime
    model: Optional[str] = None
    seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    candidates: int = Field(default=0, ge=0)
    auto: List[AutoName] = Field(default_factory=list)
    suggestions: List[NameSuggestion] = Field(default_factory=list)
    note: Optional[str] = None
    undone: List[UndoneName] = Field(default_factory=list)


class Decision(NamedTuple):
    """What :func:`decide` concluded: names to give and to suggest.

    ``clear`` lists voices whose automatic name no longer holds (it is
    removed, the voice becomes ``S#`` again).
    """

    auto: List[AutoName]
    suggestions: List[NameSuggestion]
    clear: List[str]


def _quote(text: str, start: int, end: int, limit: int = 160) -> str:
    """Return the part of ``text`` around ``[start, end)``.

    At most ``limit`` characters, cut at word boundaries.
    """
    if len(text) <= limit:
        return text.strip()
    left = max(0, start - (limit - (end - start)) // 2)
    right = min(len(text), left + limit)
    left = max(0, right - limit)
    piece = text[left:right]
    if left > 0 and " " in piece:
        piece = piece[piece.index(" ") + 1:]
    if right < len(text) and " " in piece:
        piece = piece[:piece.rindex(" ")]
    return piece.strip()


def _matches(
    text: str,
) -> Iterable[Tuple[Kind, str, float, Towards, int, int]]:
    """Yield ``(kind, name, confidence, towards, start, end)`` per match."""
    for pattern, confidence in SELF_PATTERNS:
        for found in re.finditer(pattern, text, re.IGNORECASE):
            name = clean_name(found.group("name"))
            if name and re.match(NAME_WORD, found.group("name")):
                yield ("self", name, confidence, "self", found.start(),
                       found.end())
    for pattern, towards in ADDRESSED_PATTERNS:
        for found in re.finditer(pattern, text, re.IGNORECASE):
            name = clean_name(found.group("name"))
            if name and re.match(NAME_WORD, found.group("name")):
                yield ("addressed", name, ADDRESSED_CONFIDENCE, towards,
                       found.start(), found.end())


def _case_sensitive_name(text: str, name: str) -> bool:
    """Tell whether ``name`` is written capitalised in ``text``."""
    return re.search(
        r"\b" + r"\s+".join(re.escape(w) for w in name.split()) + r"\b",
        text,
    ) is not None


def find_candidates(
    transcript: Transcript, context: int = 1, limit: int = 60
) -> List[NameCandidate]:
    """Return the sentences that may hold a name, in transcript order.

    One candidate per sentence and name: a self-introduction wins over
    a name said to someone else in the same sentence. ``context``
    sentences before and after are attached; at most ``limit``
    candidates are returned (self-introductions first, then the
    earliest).
    """
    utterances = transcript.utterances
    found: Dict[Tuple[str, str], NameCandidate] = {}
    for index, utterance in enumerate(utterances):
        before = tuple(utterances[max(0, index - context):index])
        after = tuple(utterances[index + 1:index + 1 + context])
        for kind, name, confidence, towards, start, end in _matches(
            utterance.text
        ):
            if not _case_sensitive_name(utterance.text, name):
                continue
            key = (utterance.id, fold(name).split()[0])
            if key in found and (
                found[key].kind == "self" or kind != "self"
            ):
                continue
            found[key] = NameCandidate(
                utterance, kind, name, confidence, towards, before, after,
                (start, end),
            )
    order = {u.id: i for i, u in enumerate(utterances)}
    chosen = sorted(
        found.values(),
        key=lambda c: (c.kind != "self", order[c.utterance.id]),
    )[:limit]
    return sorted(chosen, key=lambda c: order[c.utterance.id])


def target_speaker(
    transcript: Transcript,
    utterance: Utterance,
    kind: Kind,
    towards: Towards,
) -> Optional[str]:
    """Return the voice a name said in ``utterance`` belongs to.

    ``self``: the speaker of the sentence. ``addressed``: the nearest
    other speaker before it (``previous``) or after it (``next``), or,
    when there is none on that side, on the other side; ``None`` when
    nobody else speaks.
    """
    if kind == "self":
        return utterance.speaker
    utterances = transcript.utterances
    index = next(
        i for i, u in enumerate(utterances) if u.id == utterance.id
    )
    earlier = [
        u.speaker for u in reversed(utterances[:index])
        if u.speaker != utterance.speaker
    ]
    later = [
        u.speaker for u in utterances[index + 1:]
        if u.speaker != utterance.speaker
    ]
    sides = (earlier, later) if towards == "previous" else (later, earlier)
    for side in sides:
        if side:
            return side[0]
    return None


def rule_evidence(
    transcript: Transcript, candidate: NameCandidate
) -> Optional[NameEvidence]:
    """Return what the rules alone conclude from ``candidate``."""
    speaker = target_speaker(
        transcript, candidate.utterance, candidate.kind, candidate.towards
    )
    if speaker is None:
        return None
    text = candidate.utterance.text
    return NameEvidence(
        speaker=speaker, name=candidate.name, kind=candidate.kind,
        quote=_quote(text, *candidate.span), utterance=candidate.utterance.id,
        start=candidate.utterance.start, confidence=candidate.confidence,
        source="rules",
    )


PROMPT_VERSION = "v2"
"""Which wording of the prompt is in use (the measurement records it;
``docs/transcriber/NAMES.md`` §4 says what changed from v1)."""

SYSTEM_PROMPT = (
    "You read a few lines of a transcribed conversation (usually "
    "Spanish, sometimes English) and decide about ONE word in ONE line. "
    "Each line starts with the label of its speaker (S1, S2...); the "
    "line to judge is marked with >>. The other lines are only context.\n"
    "Question: in the marked line, is the given word the speaker's OWN "
    "name, the name of ANOTHER person taking part, or not a "
    "participant's name?\n"
    '- "self": the speaker of the marked line says it is their own name '
    '("soy Ana", "me llamo Pedro", "habla Luis", "aquí Luis de '
    'ventas", "I\'m Kate"). It is still "self" when the same line also '
    'names someone else ("Buenas, Rosa, soy Tomás": Tomás is self, '
    "Rosa is addressed).\n"
    '- "addressed": the speaker says it to or about another person who '
    'takes part: thanks, asks, greets or introduces them ("gracias, '
    'Marta", "Pedro, ¿qué opinas?", "os presento a Ana").\n'
    '- "none": not the name of a participant: words that only look '
    'like an introduction ("soy de Madrid", "me llamo la atención", '
    '"soy consciente"), a team, a star sign, a place, a company, '
    "someone absent or only talked about, someone quoted "
    '("mi padre decía: soy Julián y punto").\n'
    "Answer with one JSON object and nothing else: "
    '{"name": <the word as written in the marked line, or null>, '
    '"kind": "self" | "addressed" | "none", '
    '"quote": <the words of the marked line that show it>, '
    '"confidence": <a number from 0 to 1>}'
)
"""What the model is told; it answers about one candidate at a time."""


def model_prompt(candidate: NameCandidate) -> str:
    """Return the user message for ``candidate``.

    The lines around it (the candidate's marked with ``>>``), then the
    marked line again and the word to judge.
    """
    lines = ["Lines:"]
    lines += [f"   {u.speaker}: {u.text}" for u in candidate.before]
    lines.append(f">> {candidate.utterance.speaker}: "
                 f"{candidate.utterance.text}")
    lines += [f"   {u.speaker}: {u.text}" for u in candidate.after]
    lines.append(f"Marked line: {candidate.utterance.speaker}: "
                 f"{candidate.utterance.text}")
    lines.append(f"Word to judge: {candidate.name}")
    return "\n".join(lines)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_reply(
    transcript: Transcript, candidate: NameCandidate, reply: str
) -> Optional[NameEvidence]:
    """Return the evidence in the model's ``reply`` about ``candidate``.

    ``None`` when the reply is not the JSON asked for, says ``none``,
    or names someone the marked line does not name (the model cannot
    invent a name). The voice the name belongs to is fixed by
    :func:`target_speaker` (for ``addressed``, towards the side the
    rules saw); a quote that is not in the line is replaced by the
    line.
    """
    found = _JSON_OBJECT.search(reply)
    if found is None:
        return None
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    kind, name = data.get("kind"), data.get("name")
    if kind not in ("self", "addressed") or not isinstance(name, str):
        return None
    tidy = clean_name(name)
    text = candidate.utterance.text
    if tidy is None or not contains_name(text, tidy):
        return None
    try:
        confidence = min(max(float(data.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        return None
    towards: Towards = candidate.towards if kind == candidate.kind else (
        "self" if kind == "self" else "next"
    )
    speaker = target_speaker(
        transcript, candidate.utterance, kind, towards
    )
    if speaker is None:
        return None
    quote = data.get("quote")
    if not isinstance(quote, str) or not quote.strip() or (
        fold(quote.strip()) not in fold(text)
    ):
        quote = _quote(text, *candidate.span)
    return NameEvidence(
        speaker=speaker, name=tidy, kind=kind,
        quote=_quote(quote.strip(), 0, len(quote.strip())),
        utterance=candidate.utterance.id,
        start=candidate.utterance.start, confidence=confidence,
        source="model",
    )


def _suggestion(
    evidence: NameEvidence, reason: Reason, count: int
) -> NameSuggestion:
    """Return ``evidence`` as a suggestion for ``reason``."""
    return NameSuggestion(
        speaker=evidence.speaker, name=evidence.name, kind=evidence.kind,
        quote=evidence.quote, utterance=evidence.utterance,
        start=evidence.start, confidence=evidence.confidence,
        reason=reason, count=count,
    )


def _groups(
    evidence: Sequence[NameEvidence],
) -> Dict[Tuple[str, str], List[NameEvidence]]:
    """Group evidence by voice and first name, in order of appearance."""
    groups: Dict[Tuple[str, str], List[NameEvidence]] = {}
    for item in evidence:
        key = (item.speaker, fold(item.name).split()[0])
        groups.setdefault(key, []).append(item)
    return groups


def _best(items: Sequence[NameEvidence]) -> NameEvidence:
    """Return the strongest piece: self first, then the most confident."""
    return max(
        items, key=lambda e: (e.kind == "self", e.confidence, -e.start)
    )


def decide(
    speakers: Sequence[Speaker],
    evidence: Sequence[NameEvidence],
    previous: Sequence[AutoName] = (),
    auto_confidence: float = 0.8,
    allow_auto: bool = True,
    undone: Sequence[UndoneName] = (),
) -> Decision:
    """Decide which names to give and which only to suggest.

    ``speakers`` are the transcript's; ``previous`` the automatic names
    of an earlier search (a voice whose name is still that one is not
    named by hand). A voice gets a name automatically only when
    ``allow_auto`` and:

    - it is anonymous (not enrolled) and has no name, or only the
      automatic one;
    - it introduced itself (``self``) with confidence of at least
      ``auto_confidence``, always with the same name;
    - no other voice was given that name, by itself or by others (two
      people with the same name: both are suggestions);
    - no other speaker of the transcript already has that name;
    - nobody called it by a different name;
    - a person did not undo that name for that voice (``undone``).

    Every other piece of evidence about a voice still unnamed becomes a
    suggestion (one per voice and name, the strongest quote, with how
    many times it was seen). Voices that are enrolled or named by hand
    get nothing. A voice whose earlier automatic name is not given
    again is listed in ``clear`` (only when ``allow_auto``: without the
    model earlier names are kept).
    """
    known = {s.id: s for s in speakers}
    previous_names = {a.speaker: a.name for a in previous}
    open_ids = [
        s.id for s in speakers if not s.enrolled and (
            s.name is None or s.name == previous_names.get(s.id)
        )
    ]
    open_set = set(open_ids)
    usable = [e for e in evidence if e.speaker in known]
    strong_self = [
        e for e in usable
        if e.kind == "self" and e.confidence >= auto_confidence
    ]
    auto: List[AutoName] = []
    named_auto: Dict[str, str] = {}
    for speaker_id in open_ids:
        own = [e for e in strong_self if e.speaker == speaker_id]
        if not own or not allow_auto:
            continue
        names = {fold(e.name).split()[0] for e in own}
        if len(names) != 1:
            continue
        best = _best(own)
        claimed_elsewhere = any(
            same_name(e.name, best.name) for e in usable
            if e.speaker != speaker_id
        )
        taken = any(
            s.name is not None and same_name(s.name, best.name)
            for s in speakers if s.id != speaker_id and s.id not in open_set
        )
        called_otherwise = any(
            not same_name(e.name, best.name) for e in usable
            if e.kind == "addressed" and e.speaker == speaker_id
        )
        if (
            claimed_elsewhere or taken or called_otherwise
            or _was_undone(undone, speaker_id, best.name)
        ):
            continue
        named_auto[speaker_id] = best.name
        auto.append(AutoName(
            speaker=speaker_id, name=best.name, quote=best.quote,
            utterance=best.utterance, start=best.start,
            confidence=best.confidence,
        ))
    suggestions: List[NameSuggestion] = []
    for (speaker_id, first), items in _groups(usable).items():
        if speaker_id not in open_set or speaker_id in named_auto:
            continue
        best = _best(items)
        suggestions.append(_suggestion(
            best, _reason(speaker_id, best, usable, speakers, open_set,
                          auto_confidence, allow_auto, undone),
            len(items),
        ))
    clear = [
        s for s in open_ids
        if allow_auto and s in previous_names
        and known[s].name == previous_names[s] and s not in named_auto
    ]
    return Decision(auto, suggestions, clear)


def _reason(
    speaker_id: str,
    best: NameEvidence,
    evidence: Sequence[NameEvidence],
    speakers: Sequence[Speaker],
    open_set: "set[str]",
    auto_confidence: float,
    allow_auto: bool,
    undone: Sequence[UndoneName] = (),
) -> Reason:
    """Return why ``best`` is only a suggestion for ``speaker_id``."""
    if best.kind == "addressed":
        return "addressed"
    if _was_undone(undone, speaker_id, best.name):
        return "undone"
    if not allow_auto:
        return "rules_only"
    if best.confidence < auto_confidence:
        return "unsure"
    own = {
        fold(e.name).split()[0] for e in evidence
        if e.speaker == speaker_id and e.kind == "self"
        and e.confidence >= auto_confidence
    }
    if len(own) > 1:
        return "several_names"
    if any(
        same_name(e.name, best.name) for e in evidence
        if e.speaker != speaker_id
    ):
        return "same_name"
    if any(
        s.name is not None and same_name(s.name, best.name)
        for s in speakers if s.id != speaker_id and s.id not in open_set
    ):
        return "name_taken"
    return "conflict"


def _was_undone(
    undone: Sequence[UndoneName], speaker_id: str, name: str
) -> bool:
    """Tell whether a person undid ``name`` for ``speaker_id``."""
    return any(
        u.speaker == speaker_id and same_name(u.name, name) for u in undone
    )


def kept_undone(
    undone: Sequence[UndoneName], run: Optional[str]
) -> List[UndoneName]:
    """Return the undone names that apply to transcription run ``run``.

    Those undone in another run are dropped: a new transcription may
    give the labels to other people.
    """
    return [u for u in undone if u.run == run]


def valid_auto(
    speakers: Sequence[Speaker], auto: Sequence[AutoName]
) -> List[AutoName]:
    """Return the automatic names the speakers still carry.

    One a person changed by hand (or undid) is no longer automatic.
    """
    names = {s.id: s.name for s in speakers}
    return [a for a in auto if names.get(a.speaker) == a.name]


def open_suggestions(
    speakers: Sequence[Speaker], suggestions: Sequence[NameSuggestion]
) -> List[NameSuggestion]:
    """Return the suggestions for voices that are still unnamed."""
    unnamed = {s.id for s in speakers if not s.enrolled and s.name is None}
    return [s for s in suggestions if s.speaker in unnamed]
