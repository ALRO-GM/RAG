"""Language-aware lexical analysis for natural-language text.

The code-oriented tokenizer of :mod:`src.domain.text_analysis` stays
the analyzer of English; this module adds Spanish and the choice of an
analyzer from a language code, so that lexical retrieval works on
transcripts in either language.
"""

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple

from src.domain.errors import InvalidInputError
from src.domain.text_analysis import STOPWORDS as STOPWORDS_EN
from src.domain.text_analysis import tokenize as tokenize_en

Analyzer = Callable[[str], List[str]]

_WORD = re.compile(r"[a-z0-9ñ]+")
_VOWELS = "aeiou"
_VERB_PLURALS = ("amos", "emos", "imos", "ais", "eis")
_SUFFIXES = (
    ("iendo", 8), ("ando", 7), ("amos", 7), ("emos", 7), ("imos", 7),
    ("ado", 6), ("ido", 6), ("ada", 6), ("ida", 6), ("ais", 6),
    ("eis", 6), ("ion", 7), ("ar", 5), ("er", 5), ("ir", 5),
)

STOPWORDS_ES = frozenset(
    """a al algo algun alguna algunas alguno algunos ante antes aqui asi
    aun aunque bien cada casi como con contra cual cuales cuando de del
    desde donde dos el ella ellas ello ellos en entre era eran eres es
    esa esas ese eso esos esta estaba estaban estamos estan estar estas
    este esto estos estoy fue fueron ha habia han has hasta hay he la
    las le les lo los mas me mi mis mucho muy nada ni no nos nosotros o
    os otra otras otro otros para pero poco por porque pues que quien
    se sea ser si sido sin sobre solo somos son soy su sus tambien tan
    te tengo ti tiene tienen todo todos tu tus un una unas uno unos va
    vamos van vosotros voy y ya yo eh em pues vale bueno""".split()
)


def fold(text: str) -> str:
    """Lowercase ``text`` and remove accents, keeping ``ñ``."""
    composed = unicodedata.normalize("NFC", text)
    lowered = composed.lower().replace("ñ", "\0")
    plain = "".join(
        c for c in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(c) != "Mn"
    )
    return plain.replace("\0", "ñ")


def stem_es(token: str) -> str:
    """Apply a light Spanish stemmer to an accent-folded token.

    It removes ``-mente``, plural endings, the most regular verb
    endings (infinitive, gerund, participle, first and second person
    plural), ``-ion`` and a final vowel, so that ``escuchar``,
    ``escuchando``, ``escuchado`` and ``escucha`` meet on ``escuch``.
    Words of four letters or fewer are kept as they are.
    """
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith("mente") and len(token) > 8:
        token = token[:-5]
    if token.endswith("ces") and len(token) >= 5:
        token = token[:-3] + "z"
    elif token.endswith("es") and token[-3] not in _VOWELS:
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(_VERB_PLURALS):
        token = token[:-1]
    for suffix, minimum in _SUFFIXES:
        if token.endswith(suffix) and len(token) >= minimum:
            return token[: -len(suffix)]
    if len(token) > 4 and token[-1] in _VOWELS:
        return token[:-1]
    return token


def tokenize_es(text: str) -> List[str]:
    """Split Spanish text into accent-folded, stemmed lexical terms."""
    return [
        stem_es(word)
        for word in _WORD.findall(fold(text))
        if word not in STOPWORDS_ES and len(word) > 1
    ]


ANALYZERS: Dict[str, Analyzer] = {"en": tokenize_en, "es": tokenize_es}
_STOPWORDS: Tuple[Tuple[str, frozenset[str]], ...] = (
    ("en", STOPWORDS_EN), ("es", STOPWORDS_ES),
)


def analyzer_for(language: str) -> Analyzer:
    """Return the lexical analyzer of a language code (``en``, ``es``)."""
    code = language.strip().lower()[:2]
    if code not in ANALYZERS:
        raise InvalidInputError(
            f"Unsupported language {language!r}; choose one of "
            f"{', '.join(sorted(ANALYZERS))}."
        )
    return ANALYZERS[code]


def detect_language(text: str, default: str = "en") -> str:
    """Guess the language of ``text`` from its function words.

    Returns the supported language whose stopwords cover most words,
    or ``default`` when the text has no function word at all.
    """
    words = _WORD.findall(fold(text))
    best: Optional[str] = None
    best_hits = 0
    for code, stopwords in _STOPWORDS:
        hits = sum(1 for w in words if w in stopwords)
        if hits > best_hits:
            best, best_hits = code, hits
    return best or default
