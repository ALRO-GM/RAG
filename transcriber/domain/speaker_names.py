"""Speakers' names from what they say: candidates, evidence, decision.

After a transcription the voices nobody enrolled are ``S1``, ``S2``…
People often say who they are ("soy Ana", "me llamo Pedro", "aquí Luis
de ventas", "I'm Kate") or name the person they talk to ("gracias,
Marta", "os presento a Ana", "tiene la palabra el señor Ruiz"). This
module (TR-20, issue #102; multi-party recordings, TR-21, issue #116)
holds the pure part of turning that into names:

1. :func:`find_candidates` picks, with rules, the sentences that may
   hold a name, each with a little context. Words that are never a
   person's name (pronouns, imperatives used as fillers, places,
   institutions, parties, months…: :data:`NOT_NAMES`) are refused and
   honorifics ("señor", "Mr") are not part of the name. The rules also
   make their own :class:`NameEvidence` (:func:`rule_evidence`), the
   baseline the model is measured against.
2. :func:`pick_for_model` chooses the candidates a local chat model
   reads (self-introductions first, then a few of every name);
   :func:`model_prompt` and :func:`parse_reply` are what is asked about
   one candidate and how its JSON answer is checked (it must say the
   word is a person's name; a name the line does not contain is
   dropped, so the model cannot invent one). :func:`gated_evidence`
   keeps the unread mentions of a name the model confirmed.
3. :func:`attribute` turns every piece into votes for voices over the
   whole session (``domain/name_votes.py``): a self-introduction for
   its speaker, a cue for the next speaker, thanks for the previous
   one, a plain mention mostly for the last participants who spoke;
   the moderators of a debate or a panel are found and their cues hand
   the floor to the next speaker.
4. :func:`decide` applies the precision-first policy. The spellings of
   a name are grouped (``domain/name_variants.py``), names are given to
   voices one to one over the session, and a name is given
   automatically only from a clear self-introduction that conflicts
   with nothing, or from enough votes with a clear margin over every
   rival. Everything else is a :class:`NameSuggestion` per voice and
   name, with its strongest quote and how many pieces support it.
   Enrolled speakers and names set by hand are never touched.

Nothing here does I/O; the use case (``application/speaker_names.py``)
reads the transcript, calls the model and applies the names.
"""

import json
import re
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
from transcriber.domain.name_variants import (
    cluster_names,
    fold,
    similar_names,
    strip_honorifics,
)
from transcriber.domain.name_votes import (
    CUE_WEIGHT,
    MENTION_WEIGHT,
    SELF_WEIGHT,
    Towards,
    best_matching,
    find_moderators,
    margin,
    votes_for,
)

Kind = Literal["self", "addressed"]
"""``self``: the speaker says their own name; ``addressed``: the
speaker says the name of someone else present (thanking, asking or
introducing them)."""

Source = Literal["rules", "model"]
"""Who produced a piece of evidence."""

Reason = Literal[
    "addressed", "unsure", "several_names", "same_name", "name_taken",
    "conflict", "rules_only", "undone", "few_votes", "close",
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
    automatically to that voice again, see :class:`UndoneName`);
``few_votes``
    said by others, but not often or clearly enough;
``close``
    another voice for this name, or another name for this voice, is
    nearly as likely.
"""

NAME_WORD = r"[A-ZÁÉÍÓÚÑÜ][a-záéíóúñüçàèòï'-]+"
"""A capitalised word: names are written with a capital by the
recogniser, and lower-case look-alikes ("me llamo la atención") are not
names."""

HONORIFIC = (
    r"(?:señor|señora|señorita|sr\.?|sra\.?|srta\.?|don|doña|doctor|"
    r"doctora|profesor|profesora|mr\.?|mrs\.?|ms\.?|miss|dr\.?|dra\.?)"
)
"""Words before a name that are not part of it ("señor Ruiz")."""

NAME = (
    rf"(?:{HONORIFIC}\s+)?(?P<name>{NAME_WORD}(?:\s+{NAME_WORD})?)"
)
"""A first name, optionally followed by a second capitalised word,
after an optional honorific that is left out of the name."""

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
    # pronouns and determiners (#116: "Ustedes, …", "Le pregunto")
    "ustedes", "le", "les", "os", "vosotras", "ellas", "nosotras",
    "usted", "su", "sus", "mis", "tus", "nuestro", "nuestra", "vuestro",
    "vuestra", "aquel", "aquella", "cada", "cualquier", "ningún",
    "ninguno", "ninguna", "alguno", "alguna", "varios", "muchos",
    "we", "they", "he", "she", "your", "our", "their", "everyone's",
    # imperatives and discourse markers used as fillers
    "miren", "mire", "fíjense", "fíjese", "oiga", "oigan", "escuche",
    "escuchen", "escucha", "vean", "sabe", "sabes", "digo",
    "insisto", "repito", "perdonen", "disculpe", "disculpen", "dígame",
    "permítame", "déjeme", "déjenme", "créame", "créanme", "hombre",
    "mujer", "señores", "señoras", "señorías", "señoría", "look",
    "listen", "see", "actually", "anyway", "honestly", "basically",
    # words that open a sentence before a comma ("Mal, sinceramente.")
    "mal", "adelante", "perdonad", "va", "muchas", "muchísimas",
    "espera", "esperad", "espere", "dale", "hecho", "listo", "lista",
    "tal", "solo", "sólo", "quizá", "quizás", "correcto", "falso",
    "mentira", "cuidado", "atención", "silencio", "enhorabuena",
    "felicidades", "efectivamente", "ojalá", "venid", "mirad", "oíd",
    "decía", "dice", "digamos", "pasa", "pasad", "sigue", "seguimos",
    "empezamos", "terminamos", "hala", "toma",
    # verbs that open a sentence in debates and meetings
    "tiene", "tienen", "quiero", "queremos", "ha", "han", "he", "hemos",
    "sale", "salen", "espero", "esperamos", "hablan", "hablamos",
    "termine", "terminen", "concluya", "concluyan", "acabe", "acaben",
    "construir", "pocas", "pocos", "cuánto", "cuántos", "cuánta",
    "cuántas", "turno", "dicen", "saben", "pregunto", "recuerdo",
    "permítame", "déjeme", "ne",
    # places
    "europa", "francia", "alemania", "italia", "portugal", "grecia",
    "bruselas", "londres", "parís", "berlín", "roma", "lisboa",
    "américa", "méxico", "argentina", "chile", "colombia", "perú",
    "venezuela", "cuba", "china", "rusia", "ucrania", "marruecos",
    "andalucía", "cataluña", "galicia", "euskadi", "valencia", "murcia",
    "aragón", "navarra", "asturias", "cantabria", "extremadura",
    "canarias", "baleares", "castilla", "sevilla", "bilbao", "málaga",
    "zaragoza", "toledo", "granada", "córdoba", "cádiz", "vigo",
    "spain", "europe", "france", "germany", "london", "america",
    "england", "britain", "london", "york", "washington",
    # institutions, parties and companies talked about
    "gobierno", "congreso", "senado", "parlamento", "ministerio",
    "ministro", "ministra", "presidente", "presidenta", "vicepresidente",
    "vicepresidenta", "ayuntamiento", "estado", "banco", "hacienda",
    "seguridad", "unión", "comisión", "tribunal", "constitución",
    "partido", "popular", "socialista", "pp", "psoe", "vox", "podemos",
    "sumar", "ciudadanos", "bildu", "junts", "erc", "pnv", "ue", "otan",
    "government", "congress", "parliament", "president", "minister",
    "company", "board",
})
"""Capitalised words that are never taken as a name. Compared folded
(lower case, no accents: :func:`fold`), as :data:`_NOT_NAMES` holds
them."""

_NOT_NAMES = frozenset(fold(word) for word in NOT_NAMES)
"""The words of :data:`NOT_NAMES` folded."""

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

CUE_PATTERNS: Tuple[str, ...] = (
    rf"\btiene (?:usted |ahora )?la palabra,?\s+(?:su minuto,?\s+)?"
    rf"(?:el |la )?{NAME}",
    rf"\b(?:el )?turno (?:de palabra )?(?:para|de|del)\s+(?:el |la )?{NAME}",
    rf"\b(?:le|les) (?:pregunto|doy la palabra|cedo la palabra) "
    rf"(?:a|al|a la)\s+{NAME}",
    rf"\b(?:pregunta|palabra)(?: es)? para\s+(?:el |la )?{NAME}",
    rf"\b(?:con nosotros|nos acompañan?(?: también)?|bienvenid[oa]"
    rf"(?: también)?|damos la bienvenida a)\s*,?\s+(?:el |la )?{NAME}",
    rf"\b(?:adelante|cuando quiera|cuando quieras|su turno|tu turno)"
    rf"\s*,\s*{NAME}",
    rf"\b(?:over to you|the floor is yours|go ahead|your turn)\s*,\s*"
    rf"{NAME}",
    rf"\b(?:over to|let'?s hear from|let us hear from|welcome)\s+{NAME}",
)
"""Moderators' phrases that hand the floor to the person named: the
next speaker."""

WELCOME = re.compile(
    r"\b(?:bienvenid[oa]s?|welcome|hola|buenas|hi|hello)\b", re.IGNORECASE
)
"""A welcome or a greeting: from a moderator it introduces the next
speaker, from anyone else it answers the person who just spoke."""

ADDRESSED_PATTERNS: Tuple[Tuple[str, Towards], ...] = (
    (rf"\b(?:os|te|les|le) presento a\s+{NAME}", "next"),
    (rf"\b(?:let me introduce|meet|this is)\s+{NAME}", "next"),
    *((pattern, "next") for pattern in CUE_PATTERNS),
    (rf"\b(?:gracias|muchas gracias|thanks|thank you|vale|de acuerdo|"
     rf"perdona|perdón|sorry|exacto|claro)\s*,?\s*{NAME}\b", "previous"),
    (rf"\b(?:hola|buenas|buenos días|hi|hello|hey)\s*,?\s+{NAME}\b",
     "next"),
    (rf"^\s*¿?\s*{NAME}\s*,", "next"),
    (rf"\b(?:y tú|y usted|and you|what about you)\s*,\s*{NAME}\b", "next"),
    (rf",\s*{NAME}\s*[?¿]", "next"),
    (rf",\s*{NAME}\s*[.!]*\s*$", "previous"),
    (rf"\b{HONORIFIC}\s+(?P<name>{NAME_WORD}(?:\s+{NAME_WORD})?)",
     "around"),
)
"""Patterns of a name said to someone else, and which voice it points
at: thanks and acknowledgements answer the person who just spoke
(``previous``); questions, greetings, introductions and moderators'
cues call on the person who speaks next (``next``); a name after an
honorific anywhere else ("como dice el señor Ruiz") is a plain mention
of someone taking part (``around``). The first pattern that matches a
name in a sentence decides its direction."""

NAME_MODELS: Tuple[str, ...] = ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
"""The only local models allowed to read transcripts for names, in
preference order: the 4B only when the 8B does not fit, never a smaller
one (the owner's rule on #102)."""

ADDRESSED_CONFIDENCE = 0.5

MAX_ALTERNATIVES = 2
"""Names suggested for a voice besides the one matched to it."""

SAID_TOLERANCE = 0.1
"""How often a voice may say its own matched name to someone, as a
share of its votes, and still get it automatically: one sentence given
to the wrong voice by the diarizer must not veto thirty clear votes,
but a voice that keeps naming someone is not that someone."""
"""The rules' confidence in a name said to someone else."""


CLITIC_ENDINGS = ("me", "le", "nse", "nos", "selo", "sela", "melo")
"""Endings of an imperative with a pronoun attached ("Déjele",
"Recuérdeme", "Fíjense"): with a written accent, never a name."""


def _never_a_name(word: str) -> bool:
    """Tell whether ``word`` is never taken as a name.

    A word of :data:`NOT_NAMES`, an adverb in *-mente*, or an
    imperative with a pronoun attached (an accent and one of
    :data:`CLITIC_ENDINGS`).
    """
    folded = fold(word)
    accented = any(c in "áéíóú" for c in word.lower())
    return (
        folded in _NOT_NAMES or folded.endswith("mente")
        or (accented and len(folded) >= 5
            and folded.endswith(CLITIC_ENDINGS))
    )


def clean_name(name: str) -> Optional[str]:
    """Return ``name`` tidied, or ``None`` if it is not a usable name.

    The name is its first one or two words that are capitalised and
    are not words that are never names (articles, greetings, numbers,
    discourse markers: :data:`NOT_NAMES`, adverbs in *-mente*); a name
    whose first word is not one is refused. Leading honorifics
    ("señor", "Mr.") are left out.
    """
    words: List[str] = []
    for raw in strip_honorifics(name).split():
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
    many sentences are read by the model per session (the others of a
    name it confirmed count as the rules saw them). ``max_tokens``: the
    length of the model's answer. ``min_votes``: the votes a name said
    by others needs for an automatic name; ``margin``: by how many
    votes it must beat every rival (another voice for the name, another
    name for the voice). Untuned starting values (``min_votes`` and
    ``margin`` chosen on the synthetic set of #116, see
    ``docs/transcriber/NAMES.md``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auto_confidence: float = Field(default=0.8, ge=0, le=1)
    context: int = Field(default=1, ge=0, le=5)
    max_candidates: int = Field(default=60, ge=1, le=500)
    max_tokens: int = Field(default=96, ge=16, le=512)
    min_votes: float = Field(default=6.0, ge=0, le=100)
    margin: float = Field(default=3.0, ge=0, le=100)


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

    ``speaker`` is the voice the name most likely belongs to (not
    necessarily who said it), ``utterance`` and ``start`` where it was
    said, ``quote`` the words that show it. ``said_by`` is who said it
    and ``towards`` which voice it points at (:data:`Towards`);
    ``votes`` how it counts for each voice over the session, filled by
    :func:`attribute` (empty: it counts for ``speaker`` only).
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
    said_by: Optional[str] = None
    towards: Towards = "self"
    votes: Tuple[Tuple[str, float], ...] = ()


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
    """A name proposed for a voice, accepted or not by a person.

    One per voice and name: ``quote`` is the strongest piece,
    ``count`` how many pieces support it, ``votes`` their weight for
    this voice and ``spellings`` the other ways the name was written
    (recognition variants grouped with it).
    """

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
    votes: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    spellings: List[str] = Field(default_factory=list)


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
    transcript: Transcript, context: int = 1, limit: Optional[int] = 60
) -> List[NameCandidate]:
    """Return the sentences that may hold a name, in transcript order.

    One candidate per sentence and name: a self-introduction wins over
    a name said to someone else in the same sentence. ``context``
    sentences before and after are attached; at most ``limit``
    candidates are returned (chosen by :func:`pick_for_model`), all of
    them when ``limit`` is ``None``.
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
    every = sorted(found.values(), key=lambda c: order[c.utterance.id])
    return every if limit is None else pick_for_model(every, limit)


def target_speaker(
    transcript: Transcript,
    utterance: Utterance,
    kind: Kind,
    towards: Towards,
) -> Optional[str]:
    """Return the voice a name said in ``utterance`` most likely names.

    ``self``: the speaker of the sentence. ``addressed``: the nearest
    other speaker before it (``previous``, and ``around``: a plain
    mention) or after it (``next``), or, when there is none on that
    side, on the other side; ``None`` when nobody else speaks. Over a
    whole session :func:`attribute` spreads it more carefully.
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
    sides = (later, earlier) if towards == "next" else (earlier, later)
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
        source="rules", said_by=candidate.utterance.speaker,
        towards=candidate.towards,
    )


def pick_for_model(
    candidates: Sequence[NameCandidate], limit: int
) -> List[NameCandidate]:
    """Return at most ``limit`` candidates for the model, in order.

    Self-introductions first; then the names said to others in turns,
    one sentence of every name (its spellings grouped) per round, the
    ones with a direction (a cue, thanks, a question) before plain
    mentions, so every name is read at least once before any is read
    twice.
    """
    order = {id(c): i for i, c in enumerate(candidates)}
    chosen = [c for c in candidates if c.kind == "self"][:limit]
    shown = cluster_names([c.name for c in candidates])
    queues: Dict[str, List[NameCandidate]] = {}
    for candidate in candidates:
        if candidate.kind != "self":
            queues.setdefault(shown[candidate.name], []).append(candidate)
    for queue in queues.values():
        queue.sort(key=lambda c: (c.towards == "around", order[id(c)]))
    rounds = max((len(q) for q in queues.values()), default=0)
    for step in range(rounds):
        for queue in queues.values():
            if len(chosen) >= limit:
                break
            if step < len(queue):
                chosen.append(queue[step])
    return sorted(chosen, key=lambda c: order[id(c)])


def gated_evidence(
    transcript: Transcript,
    candidates: Sequence[NameCandidate],
    read: Sequence[Tuple[NameCandidate, Optional[NameEvidence]]],
) -> List[NameEvidence]:
    """Return the model's evidence and the mentions it vouches for.

    ``read`` pairs every candidate the model read with what it
    concluded (``None``: not a participant's name). A candidate it did
    not read counts as the rules saw it when the model confirmed that
    name (its spellings grouped) at least once and at least as often as
    it refused it; names it never read, or refused more often, are
    dropped.
    """
    shown = cluster_names(
        [c.name for c in candidates]
        + [c.name for c, _ in read]
        + [e.name for _, e in read if e is not None]
    )
    done = {id(c) for c, _ in read}
    confirmed: Dict[str, int] = {}
    refused: Dict[str, int] = {}
    found: List[NameEvidence] = []
    for candidate, item in read:
        key = shown[candidate.name]
        if item is None:
            refused[key] = refused.get(key, 0) + 1
            continue
        confirmed[key] = confirmed.get(key, 0) + 1
        found.append(item)
    for candidate in candidates:
        key = shown[candidate.name]
        if id(candidate) in done or confirmed.get(key, 0) < max(
            1, refused.get(key, 0)
        ):
            continue
        item = rule_evidence(transcript, candidate)
        if item is not None:
            found.append(item)
    return found


def attribute(
    transcript: Transcript, evidence: Sequence[NameEvidence]
) -> List[NameEvidence]:
    """Return ``evidence`` with its votes for voices over the session.

    The moderators are found (:func:`~transcriber.domain.name_votes.
    find_moderators`); then each piece is spread by its direction
    (:func:`~transcriber.domain.name_votes.votes_for`): a
    self-introduction weighs most, a name with a direction (thanks, a
    cue, a question) next, a plain mention least. In a moderated
    session a participant's vocative that is not a question ("Señor
    Ruiz, usted dijo…") is a rebuttal, not a hand-over, so it counts
    as a plain mention; a welcome or greeting by name from someone who
    does not run the conversation ("Bienvenido, Joaquín") answers the
    person who just spoke. A piece without ``said_by`` (or whose
    sentence is not in the transcript) keeps its ``speaker``.
    """
    utterances = transcript.utterances
    speakers = [u.speaker for u in utterances]
    position = {u.id: i for i, u in enumerate(utterances)}
    moderators = find_moderators([(u.speaker, u.text) for u in utterances])
    out: List[NameEvidence] = []
    for item in evidence:
        index = position.get(item.utterance)
        if item.said_by is None or index is None:
            out.append(item)
            continue
        towards: Towards = item.towards
        text = utterances[index].text
        if item.kind == "self":
            towards = "self"
        elif towards == "self" or (
            towards == "next" and moderators
            and item.said_by not in moderators and "?" not in text
            and not WELCOME.search(text)
        ):
            towards = "around"
        elif (
            towards == "next" and item.said_by not in moderators
            and WELCOME.search(text) and "?" not in text
        ):
            towards = "previous"
        weight = {
            "self": SELF_WEIGHT, "around": MENTION_WEIGHT,
        }.get(towards, CUE_WEIGHT)
        votes = votes_for(speakers, index, towards, moderators, weight)
        out.append(item.model_copy(update={
            "towards": towards,
            "votes": tuple((v, round(w, 4)) for v, w in votes),
        }))
    return out


PROMPT_VERSION = "v3"
"""Which wording of the prompt is in use (the measurement records it;
``docs/transcriber/NAMES.md`` §4 says what changed)."""

SYSTEM_PROMPT = (
    "You read a few lines of a transcribed conversation (usually "
    "Spanish, sometimes English) and decide about ONE word in ONE line. "
    "Each line starts with the label of its speaker (S1, S2...); the "
    "line to judge is marked with >>. The other lines are only context. "
    "The transcript comes from speech recognition: names may be "
    "misspelt.\n"
    "First question: is the given word a PERSON'S first name or "
    "surname? It is NOT when it is a pronoun (\"Ustedes\", \"Le\"), a "
    "verb or filler (\"Miren\", \"Fíjense\", \"Oiga\"), a country, "
    "region or city, an institution, a party, a company, a month or "
    "any ordinary word that is capitalised only because it starts the "
    "sentence.\n"
    "Second question: in the marked line, is it the speaker's OWN name, "
    "the name of ANOTHER person taking part, or not a participant's "
    "name?\n"
    '- "self": the speaker of the marked line says it is their own name '
    '("soy Ana", "me llamo Pedro", "habla Luis", "aquí Luis de '
    'ventas", "I\'m Kate"). It is still "self" when the same line also '
    'names someone else ("Buenas, Rosa, soy Tomás": Tomás is self, '
    "Rosa is addressed).\n"
    '- "addressed": the speaker says it to or about another person who '
    "takes part in this conversation: thanks, asks, greets, rebuts, "
    'introduces them or gives them the floor ("gracias, Marta", '
    '"Pedro, ¿qué opinas?", "os presento a Ana", "señor Ruiz, usted '
    'dijo...", "tiene la palabra la señora Gil").\n'
    '- "none": not the name of a participant: not a person\'s name '
    '(see above), words that only look like an introduction ("soy de '
    'Madrid", "me llamo la atención", "soy consciente"), a team, a '
    "star sign, someone absent or only talked about, someone quoted "
    '("mi padre decía: soy Julián y punto").\n'
    "Leave out honorifics: for \"señor Ruiz\" the name is \"Ruiz\".\n"
    "Answer with one JSON object and nothing else: "
    '{"is_person_name": true | false, '
    '"name": <the word as written in the marked line, or null>, '
    '"kind": "self" | "addressed" | "none", '
    '"quote": <the words of the marked line that show it>, '
    '"confidence": <a number from 0 to 1: 1 only when the line leaves '
    "no doubt; lower it when the word could be something else or the "
    "person might not be present>}"
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

    ``None`` when the reply is not the JSON asked for, does not say the
    word is a person's name (``is_person_name`` must be ``true``), says
    ``none``, or names someone the marked line does not name (the model
    cannot invent a name) or a word that is never a name. The voice the
    name belongs to is fixed by the rules (for ``addressed``, towards
    the side the rules saw); a quote that is not in the line is
    replaced by the line.
    """
    found = _JSON_OBJECT.search(reply)
    if found is None:
        return None
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("is_person_name") is not True:
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
        source="model", said_by=candidate.utterance.speaker,
        towards=towards,
    )


def _votes(item: NameEvidence) -> Tuple[Tuple[str, float], ...]:
    """Return the votes of ``item`` (its ``speaker`` when not filled)."""
    if item.votes:
        return item.votes
    weight = SELF_WEIGHT if item.kind == "self" else CUE_WEIGHT
    return ((item.speaker, weight),)


def _best(items: Sequence[NameEvidence]) -> NameEvidence:
    """Return the strongest piece: self first, then the most confident."""
    return max(
        items, key=lambda e: (e.kind == "self", e.confidence, -e.start)
    )


class _Table(NamedTuple):
    """The votes of a session per voice and name (spellings grouped).

    ``votes`` add up the evidence for a voice; ``scores`` are the votes
    minus how often the voice itself said the name to someone (people
    do not call themselves "señor Ruiz"); ``pieces`` the evidence
    behind each pair; ``shown`` how each spelling is shown and
    ``spellings`` the spellings of each shown name.
    """

    scores: Dict[Tuple[str, str], float]
    votes: Dict[Tuple[str, str], float]
    pieces: Dict[Tuple[str, str], List[NameEvidence]]
    shown: Dict[str, str]
    spellings: Dict[str, List[str]]


def _table(evidence: Sequence[NameEvidence]) -> _Table:
    """Count the votes of ``evidence`` per voice and grouped name."""
    shown = cluster_names([e.name for e in evidence])
    votes: Dict[Tuple[str, str], float] = {}
    pieces: Dict[Tuple[str, str], List[NameEvidence]] = {}
    said: Dict[Tuple[str, str], float] = {}
    spellings: Dict[str, List[str]] = {}
    for item in evidence:
        name = shown[item.name]
        if item.name not in spellings.setdefault(name, []):
            spellings[name].append(item.name)
        for voice, weight in _votes(item):
            key = (voice, name)
            votes[key] = votes.get(key, 0.0) + weight
            pieces.setdefault(key, []).append(item)
        if item.kind == "addressed" and item.said_by is not None:
            key = (item.said_by, name)
            said[key] = said.get(key, 0.0) + MENTION_WEIGHT
    keys = sorted(
        set(votes) | set(said),
        key=lambda k: (_first_start(pieces.get(k, [])), k),
    )
    scores = {k: votes.get(k, 0.0) - said.get(k, 0.0) for k in keys}
    return _Table(scores, votes, pieces, shown, spellings)


def _first_start(items: Sequence[NameEvidence]) -> float:
    """Return when the earliest of ``items`` was said (inf: none)."""
    return min((e.start for e in items), default=float("inf"))


def decide(
    speakers: Sequence[Speaker],
    evidence: Sequence[NameEvidence],
    previous: Sequence[AutoName] = (),
    auto_confidence: float = 0.8,
    allow_auto: bool = True,
    undone: Sequence[UndoneName] = (),
    min_votes: float = 6.0,
    min_margin: float = 3.0,
) -> Decision:
    """Decide which names to give and which only to suggest.

    ``speakers`` are the transcript's; ``previous`` the automatic names
    of an earlier search (a voice whose name is still that one is not
    named by hand); ``evidence`` with its votes (:func:`attribute`;
    without them a piece counts for its ``speaker``). The spellings of
    a name are grouped first. A voice gets a name automatically only
    when ``allow_auto``, it is anonymous (not enrolled) and has no
    name, or only the automatic one, a person did not undo that name
    for that voice (``undone``), no other speaker of the transcript
    already has it, and either:

    1. it introduced itself (``self``) with confidence of at least
       ``auto_confidence``, always with the same name; no other voice
       was given that name (by itself, or by others with at least one
       vote: two people with the same name), and nobody called it by a
       different name with a clear vote; or
    2. names are matched to voices one to one over the session (the
       highest total of votes); the match has at least ``min_votes``
       votes, beats every rival (another voice for the name, another
       name for the voice) by ``min_margin``, the voice said that name
       to someone else hardly ever (:data:`SAID_TOLERANCE`) and never
       introduced itself with another name.

    Every other match of a voice still unnamed becomes a suggestion,
    and so do up to :data:`MAX_ALTERNATIVES` other names per voice
    that it introduced itself with or that have a clear vote for it
    (one per voice and name, with the strongest quote, how many pieces
    support it and their votes).
    Voices that are enrolled or named by hand get nothing. A voice
    whose earlier automatic name is not given again is listed in
    ``clear`` (only when ``allow_auto``: without the model earlier
    names are kept).
    """
    known = {s.id: s for s in speakers}
    previous_names = {a.speaker: a.name for a in previous}
    open_ids = [
        s.id for s in speakers if not s.enrolled and (
            s.name is None or s.name == previous_names.get(s.id)
        )
    ]
    open_set = set(open_ids)
    usable = [
        e.model_copy(update={"votes": tuple(
            (v, w) for v, w in e.votes if v in known
        )}) if e.votes else e
        for e in evidence if e.speaker in known
    ]
    table = _table(usable)
    taken = [
        s.name for s in speakers
        if s.name is not None and s.id not in open_set
    ]
    auto: List[AutoName] = []
    named: Dict[str, str] = {}
    if allow_auto:
        for found in _self_names(open_ids, usable, table, auto_confidence):
            speaker_id, name = found.speaker, table.shown[found.name]
            if not (
                any(similar_names(t, name) for t in taken)
                or _was_undone(undone, speaker_id, name)
            ):
                named[speaker_id] = name
                auto.append(found)
    given = set(named.values())
    open_scores = {
        key: score for key, score in table.scores.items()
        if key[0] in open_set and key[0] not in named
        and key[1] not in given
    }
    suggestions: List[NameSuggestion] = []
    for speaker_id, name in best_matching(open_scores):
        items = table.pieces.get((speaker_id, name), [])
        if not items:
            continue
        best = _best(items)
        score = table.scores[(speaker_id, name)]
        lead = margin(open_scores, speaker_id, name)
        own = {
            table.shown[e.name] for e in usable
            if e.kind == "self" and e.speaker == speaker_id
        }
        said = table.votes.get((speaker_id, name), 0.0) - score
        if (
            allow_auto and score >= min_votes and lead >= min_margin
            and own <= {name}
            and said <= SAID_TOLERANCE * table.votes.get(
                (speaker_id, name), 0.0
            )
            and not any(similar_names(t, name) for t in taken)
            and not _was_undone(undone, speaker_id, name)
        ):
            named[speaker_id] = name
            auto.append(AutoName(
                speaker=speaker_id, name=name, quote=best.quote,
                utterance=best.utterance, start=best.start,
                confidence=round(min(1.0, lead / (lead + min_margin)), 3),
            ))
            continue
        reason = _reason(
            speaker_id, name, best, table, (score, lead), usable, taken,
            (auto_confidence, min_votes, min_margin), allow_auto, undone,
        )
        suggestions.append(
            _suggestion(best, name, reason, items, table, speaker_id)
        )
    seen = {(s.speaker, table.shown.get(s.name, s.name)) for s in suggestions}
    extra: Dict[str, int] = {}
    for (speaker_id, name), score in sorted(
        table.scores.items(), key=lambda kv: -kv[1]
    ):
        items = table.pieces.get((speaker_id, name), [])
        if (
            speaker_id not in open_set or speaker_id in named
            or (speaker_id, name) in seen or name in given or not items
            or extra.get(speaker_id, 0) >= MAX_ALTERNATIVES
            or (score < CUE_WEIGHT and not any(
                e.kind == "self" for e in items
            ))
        ):
            continue
        seen.add((speaker_id, name))
        extra[speaker_id] = extra.get(speaker_id, 0) + 1
        best = _best(items)
        reason = _reason(
            speaker_id, name, best, table,
            (score, margin(table.scores, speaker_id, name)),
            usable, taken, (auto_confidence, min_votes, min_margin),
            allow_auto, undone,
        )
        suggestions.append(
            _suggestion(best, name, reason, items, table, speaker_id)
        )
    clear = [
        s for s in open_ids
        if allow_auto and s in previous_names
        and known[s].name == previous_names[s] and s not in named
    ]
    return Decision(auto, suggestions, clear)


def _self_names(
    open_ids: Sequence[str],
    evidence: Sequence[NameEvidence],
    table: _Table,
    auto_confidence: float,
) -> List[AutoName]:
    """Return the clear self-introductions that conflict with nothing.

    For each open voice: introduced itself with confidence of at least
    ``auto_confidence``, always with the same name; no other voice has
    a vote for that name or introduced itself with it; no other name
    has a clear vote (:data:`CUE_WEIGHT`) for this voice.
    """
    out: List[AutoName] = []
    for speaker_id in open_ids:
        own = [
            e for e in evidence if e.speaker == speaker_id
            and e.kind == "self" and e.confidence >= auto_confidence
        ]
        names = {table.shown[e.name] for e in own}
        if len(names) != 1:
            continue
        (name,) = names
        claimed_elsewhere = any(
            n == name and v != speaker_id and (
                s >= MENTION_WEIGHT or any(
                    e.kind == "self" for e in table.pieces.get((v, n), [])
                )
            )
            for (v, n), s in table.scores.items()
        )
        called_otherwise = any(
            v == speaker_id and n != name and s >= CUE_WEIGHT
            for (v, n), s in table.scores.items()
        )
        if claimed_elsewhere or called_otherwise:
            continue
        best = _best(own)
        out.append(AutoName(
            speaker=speaker_id, name=best.name, quote=best.quote,
            utterance=best.utterance, start=best.start,
            confidence=best.confidence,
        ))
    return out


def _suggestion(
    best: NameEvidence,
    name: str,
    reason: Reason,
    items: Sequence[NameEvidence],
    table: _Table,
    speaker_id: str,
) -> NameSuggestion:
    """Return the suggestion of ``name`` (grouped) for ``speaker_id``.

    A self-introduction is shown as the voice said it; a name said by
    others with its most frequent spelling.
    """
    shown = best.name if best.kind == "self" else name
    return NameSuggestion(
        speaker=speaker_id, name=shown, kind=best.kind,
        quote=best.quote, utterance=best.utterance,
        start=best.start, confidence=best.confidence,
        reason=reason, count=max(1, len(items)),
        votes=round(max(0.0, table.votes.get((speaker_id, name), 0.0)), 2),
        spellings=[s for s in table.spellings.get(name, []) if s != shown],
    )


def _reason(
    speaker_id: str,
    name: str,
    best: NameEvidence,
    table: _Table,
    score_and_lead: Tuple[float, float],
    evidence: Sequence[NameEvidence],
    taken: Sequence[str],
    thresholds: Tuple[float, float, float],
    allow_auto: bool,
    undone: Sequence[UndoneName],
) -> Reason:
    """Return why ``name`` is only a suggestion for ``speaker_id``.

    ``thresholds`` are the confidence for a self-introduction, the
    votes and the margin for a name said by others.
    """
    score, lead = score_and_lead
    auto_confidence, min_votes, min_margin = thresholds
    if _was_undone(undone, speaker_id, name):
        return "undone"
    if not allow_auto:
        return "rules_only"
    if any(similar_names(t, name) for t in taken):
        return "name_taken"
    if best.kind == "self":
        if best.confidence < auto_confidence:
            return "unsure"
        own = {
            table.shown[e.name] for e in evidence
            if e.speaker == speaker_id and e.kind == "self"
            and e.confidence >= auto_confidence
        }
        if len(own) > 1:
            return "several_names"
        if any(
            n == name and v != speaker_id and s >= MENTION_WEIGHT
            for (v, n), s in table.scores.items()
        ) or any(
            e.kind == "self" and e.speaker != speaker_id
            and table.shown[e.name] == name for e in evidence
        ):
            return "same_name"
        return "conflict"
    if score < min_votes:
        return "few_votes"
    if lead < min_margin:
        return "close"
    return "conflict"


def _was_undone(
    undone: Sequence[UndoneName], speaker_id: str, name: str
) -> bool:
    """Tell whether a person undid ``name`` for ``speaker_id``."""
    return any(
        u.speaker == speaker_id and (
            same_name(u.name, name) or similar_names(u.name, name)
        )
        for u in undone
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
