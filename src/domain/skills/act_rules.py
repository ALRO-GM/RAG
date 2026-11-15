"""Approach A of the dialogue-act tagger: surface rules (#82).

Each sentence gets one **rule key**, the first rule that fires in a
fixed order: what the sentence looks like (question mark, interrogative
word, a back-channel alone, a greeting or thanks, a commitment, a
proposal, a request, meeting management, agreement or disagreement
openers, opinion markers) and, for short sentences and the default,
whether it follows somebody else's question. The key is not a label:
an :class:`~src.domain.skills.act_calibration.OutcomeTable` learned on
held-out sentences turns it into calibrated probabilities, so a rule
that is right 60 % of the time says 60 %.

The cue lists are Spanish and English (the public corpora are English;
``docs/skills/CONVERSATION.md`` §6.1 option A). They match words in
lower case without accents, so "Qué" and "que" fall together on
purpose: the question mark decides when there is one.
"""

import re
import unicodedata
from typing import Iterable, Tuple

from src.domain.skills.act_tagging import ActInput

RULES_VERSION = "rules-1"
"""Changes whenever a cue list or the order of the rules changes."""


def normalise(text: str) -> str:
    """Return ``text`` in lower case, without accents or extra spaces."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.split())


def _words(text: str) -> str:
    """Return the normalised words only (punctuation becomes space)."""
    return " ".join(re.findall(r"[a-z0-9ñ']+", normalise(text)))


def _pattern(cues: Iterable[str]) -> "re.Pattern[str]":
    """Return a regex matching any cue as whole words."""
    joined = "|".join(sorted((re.escape(c) for c in cues), key=len,
                             reverse=True))
    return re.compile(rf"(?:^|\s)(?:{joined})(?:\s|$)")


FILLERS = frozenset({
    "eh", "em", "mm", "mmm", "um", "uh", "hmm", "ehm", "este", "pues",
    "bueno", "o sea", "ah",
})
BACKCHANNELS = frozenset({
    "aja", "aha", "mhm", "mm hmm", "uh huh", "ya", "ya ya", "si si",
    "vale vale", "claro claro", "ok", "okay", "vale", "si", "claro",
    "ya veo", "entiendo", "oh", "ah vale", "ah ya", "yeah", "yes",
    "right", "i see", "sure", "uh huh yeah", "yep", "yup", "hm",
    "mm hm", "de acuerdo", "perfecto", "bien", "muy bien", "okay okay",
    "right right", "yeah yeah", "no", "nope",
})
YES_NO = frozenset({
    "si", "no", "yes", "yeah", "yep", "nope", "claro", "si si", "no no",
    "claro que si", "por supuesto", "of course", "exacto",
})
_RELATION = _pattern((
    "gracias", "muchas gracias", "buenos dias", "buenas tardes",
    "buenas", "hola", "perdona", "perdon", "disculpa", "lo siento",
    "enhorabuena", "felicidades", "adios", "hasta luego", "hasta manana",
    "bienvenido", "bienvenida", "bienvenidos", "encantado", "encantada",
    "thanks", "thank you", "sorry", "hello", "hi", "bye", "goodbye",
    "welcome", "congratulations", "nice to meet", "great job",
    "buen trabajo", "que tal estas",
))
_INTERROGATIVE = re.compile(
    r"^(que|como|cuando|donde|quien|quienes|cual|cuales|cuanto|cuanta|"
    r"cuantos|cuantas|por que|para que|what|how|why|when|where|who|"
    r"which|whose|do you|does|did you|are you|is it|is there|are there|"
    r"have you|has|can we|should we|would it|is that)\b")
_REQUEST_QUESTION = re.compile(
    r"\b(puedes|podrias|puede|podria|te importa|me pasas|me mandas|"
    r"me envias|can you|could you|would you|will you)\b")
_PROPOSAL = _pattern((
    "y si", "que tal si", "podriamos", "propongo", "sugiero", "mi propuesta",
    "deberiamos", "hagamos", "por que no", "lo suyo seria", "una opcion",
    "se podria", "podemos", "why don't we", "we could", "we should",
    "let's", "lets", "maybe we", "i suggest", "how about", "what about",
    "i propose", "we might", "it would be good to",
))
_REQUEST = _pattern((
    "por favor", "necesito que", "necesitamos que", "quiero que",
    "mandame", "enviame", "pasame", "revisa", "revisad", "acuerdate",
    "no te olvides", "please", "i need you to", "make sure", "send me",
    "have a look", "take a look", "tienes que", "teneis que",
))
_COMMITMENT = _pattern((
    "me encargo", "me ocupo", "lo hago yo", "yo lo hago", "lo preparo",
    "te lo mando", "os lo mando", "te lo envio", "os lo envio",
    "te lo paso", "os lo paso", "yo me encargo", "yo me ocupo",
    "lo miro yo", "lo reviso yo", "cuenta conmigo", "lo tendras",
    "lo tendreis", "i'll", "i will", "i'll do", "i can do that",
    "leave it to me", "i'll take care", "me comprometo", "voy a",
))
_MANAGEMENT = _pattern((
    "siguiente punto", "pasamos a", "orden del dia", "nos quedan",
    "minutos", "resumiendo", "para terminar", "empezamos",
    "vamos a empezar", "el siguiente tema", "volviendo a", "en resumen",
    "next item", "moving on", "let's start", "to sum up", "agenda",
    "let's move on", "back to", "first of all", "the next point",
    "se nos acaba el tiempo", "dejamos aqui", "cerramos",
    "te doy la palabra", "turno de", "a ver, recapitulando",
))
_AGREEMENT = re.compile(
    r"^(de acuerdo|estoy de acuerdo|me parece bien|perfecto|exacto|"
    r"totalmente|me parece genial|me vale|hecho|trato hecho|vale pues|"
    r"si claro|buena idea|tienes razon|i agree|exactly|sounds good|"
    r"good idea|absolutely|that's right|fair enough|agreed|ok let's|"
    r"okay let's)\b")
_DISAGREEMENT = re.compile(
    r"^(no estoy de acuerdo|no creo|eso no|no no|pero|no lo veo|"
    r"yo no|ni hablar|no me parece|i disagree|i don't think|but|no but|"
    r"not really|i'm not sure|that's not)\b")
_OPINION = _pattern((
    "creo que", "me parece", "pienso que", "en mi opinion", "opino",
    "para mi", "es mejor", "es peor", "es un error", "es importante",
    "me gusta", "no me gusta", "es genial", "es un problema",
    "i think", "i believe", "in my opinion", "i feel", "i like",
    "i don't like", "it's better", "it's important", "it's good",
    "that's great", "too expensive", "es caro", "es muy",
))


def _after_question(item: ActInput) -> bool:
    """Tell whether the sentence follows another speaker's question."""
    if not item.context:
        return False
    last = item.context[-1]
    text = last.text.strip()
    asked = text.endswith("?") or bool(_INTERROGATIVE.match(_words(text)))
    return last.speaker != item.speaker and asked


def _after_other(item: ActInput) -> bool:
    """Tell whether the sentence follows somebody else's sentence."""
    return bool(item.context) and item.context[-1].speaker != item.speaker


def _cut(text: str) -> bool:
    """Tell whether a sentence was left unfinished."""
    stripped = text.strip()
    return stripped.endswith(("...", "…", "-", "—")) and len(
        _words(stripped).split()) <= 6


def rule_key(item: ActInput) -> str:
    """Return the key of the first rule that fires for ``item``.

    Keys are ``<category the rule points at>:<rule>``; the category is a
    hint for a reader, the calibrated table decides the probabilities.
    """
    raw = item.text.strip()
    words = _words(raw)
    n = len(words.split())
    after_q = _after_question(item)
    after_other = _after_other(item)
    if not words:
        return "otro:empty"
    if _cut(raw):
        return "otro:cut"
    if all(w in FILLERS for w in words.split()):
        return "otro:filler"
    if words in YES_NO and after_q:
        return "respuesta:yes_no_after_question"
    if words in BACKCHANNELS:
        return ("escucha:backchannel_after_other" if after_other
                else "escucha:backchannel")
    if _RELATION.search(words) and n <= 8:
        return "relacion:formula"
    question = raw.endswith("?") or raw.startswith("¿") or "?" in raw
    if question and _REQUEST_QUESTION.search(words):
        return "peticion:can_you_question"
    if question and _PROPOSAL.search(words):
        return "propuesta:question"
    if question:
        return ("pregunta:question_mark_after_question" if after_q
                else "pregunta:question_mark")
    if _INTERROGATIVE.match(words) and n <= 12:
        return "pregunta:interrogative_word"
    if _MANAGEMENT.search(words):
        return "gestion:cue"
    if _COMMITMENT.search(words):
        return "compromiso:cue"
    if after_other and _AGREEMENT.match(words):
        return "aceptacion:opener"
    if after_other and _DISAGREEMENT.match(words):
        return "rechazo:opener"
    if _PROPOSAL.search(words):
        return "propuesta:cue"
    if _REQUEST.search(words):
        return "peticion:cue"
    if _OPINION.search(words):
        return "opinion:cue"
    if after_q:
        return "default:after_question"
    return "default:short" if n <= 3 else "default:statement"


def rule_keys(items: Iterable[ActInput]) -> Tuple[str, ...]:
    """Return :func:`rule_key` of every item."""
    return tuple(rule_key(item) for item in items)
