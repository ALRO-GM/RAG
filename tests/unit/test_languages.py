"""Spanish lexical analysis, language detection and analyzer choice."""

import unicodedata
from typing import List

import pytest

from src.domain.errors import InvalidInputError
from src.domain.languages import (
    STOPWORDS_ES,
    analyzer_for,
    detect_language,
    fold,
    stem_es,
    tokenize_es,
)
from src.domain.text_analysis import tokenize as tokenize_en


def test_fold_lowercases_and_removes_accents() -> None:
    """Accented vowels and the dieresis lose their mark, case is dropped."""
    assert fold("DÉJALA Terminar, pingüino ¿Qué?") == (
        "dejala terminar, pinguino ¿que?"
    )


def test_fold_keeps_the_enye() -> None:
    """``ñ`` is a letter of its own, in either case."""
    assert fold("Año, AÑO, niño") == "año, año, niño"


def test_fold_keeps_a_decomposed_enye() -> None:
    """The NFD spelling of ``ñ`` folds like the NFC one."""
    assert fold(unicodedata.normalize("NFD", "año")) == "año"


@pytest.mark.parametrize("forms, stem", [
    (["escuchar", "escuchando", "escuchado", "escucha", "escuchamos",
      "escucháis"], "escuch"),
    (["pregunta", "preguntas", "preguntar", "preguntando"], "pregunt"),
    (["comunicación", "comunicaciones"], "comunicac"),
    (["lápiz", "lápices"], "lapiz"),
    (["claro", "clara", "claramente", "claros"], "clar"),
    (["resumen", "resúmenes"], "resumen"),
])
def test_tokenize_es_inflections_meet_on_one_stem(
    forms: List[str], stem: str
) -> None:
    """Plural, verb and adverb forms of a word share their stem."""
    assert tokenize_es(" ".join(forms)) == [stem] * len(forms)


def test_tokenize_es_adverb_meets_its_ido_adjective() -> None:
    """``rápidamente`` and ``rápido`` share a stem."""
    assert len(set(tokenize_es("rápido rápidamente"))) == 1


def test_tokenize_es_drops_stopwords_and_single_letters() -> None:
    """Function words (accented or not) and one-letter tokens vanish."""
    assert tokenize_es("Yo también estoy aquí y él o ella, ¿qué?") == []
    assert "tambien" in STOPWORDS_ES


def test_tokenize_es_keeps_numbers_and_short_words() -> None:
    """Numbers and words of four letters or fewer are kept unstemmed."""
    assert tokenize_es("En 2024 leyó 12 años poll") == [
        "2024", "leyo", "12", "años", "poll",
    ]


def test_tokenize_es_splits_on_punctuation() -> None:
    """Punctuation, inverted marks and quotes only separate words."""
    assert tokenize_es("¡Servidor!¿bucle?«eventos»") == [
        "servidor", "bucl", "event",
    ]


def test_tokenize_es_of_empty_text_is_empty() -> None:
    """No text gives no term."""
    assert tokenize_es("") == []
    assert tokenize_es(" \n\t ") == []


def test_stem_es_leaves_short_tokens_and_digits() -> None:
    """Short words and numbers are returned unchanged."""
    assert stem_es("casa") == "casa"
    assert stem_es("1234567") == "1234567"


def test_stem_es_does_not_strip_verb_plural_s() -> None:
    """First and second person plurals lose the whole ending, not an s."""
    assert stem_es("hablamos") == "habl"
    assert stem_es("comeis") == "com"


def test_stem_es_never_returns_an_empty_stem() -> None:
    """Every word of five letters or more keeps a non-empty stem."""
    words = ["mente", "yendo", "ando", "iendo", "amente", "oseis", "ceses",
             "aaaaa", "eses", "clases", "ciones", "acion"]
    assert all(stem_es(w) for w in words)


@pytest.mark.parametrize("text, expected", [
    ("El servidor no puede bloquearse mientras atiende a un cliente", "es"),
    ("The server must not block while it serves a client", "en"),
    ("Bueno, vale, pues eso", "es"),
])
def test_detect_language_from_function_words(
    text: str, expected: str
) -> None:
    """The language whose stopwords cover most words wins."""
    assert detect_language(text) == expected


def test_detect_language_defaults_without_function_words() -> None:
    """Empty or symbol-only text yields the default language."""
    assert detect_language("") == "en"
    assert detect_language("", default="es") == "es"
    assert detect_language("12345 !!! ###", default="es") == "es"


def test_detect_language_of_mixed_text_follows_the_majority() -> None:
    """A few foreign words do not change the detected language."""
    mixed = "Vale, hacemos el deploy y luego el merge de la branch, ok"
    assert detect_language(mixed) == "es"
    assert detect_language("We call it la casa and the team likes it") == "en"


def test_detect_language_tie_goes_to_english() -> None:
    """With as many hits in both languages the first one (en) is kept."""
    assert detect_language("the el") == "en"


def test_analyzer_for_accepts_codes_in_any_case_and_region() -> None:
    """``ES``, ``es-MX`` and padded codes select the same analyzer."""
    assert analyzer_for("es") is tokenize_es
    assert analyzer_for("ES") is tokenize_es
    assert analyzer_for(" es-MX ") is tokenize_es
    assert analyzer_for("en_US") is tokenize_en


@pytest.mark.parametrize("language", ["fr", "", "  ", "x"])
def test_analyzer_for_rejects_unsupported_languages(language: str) -> None:
    """An unknown or empty code is a user error naming the choices."""
    with pytest.raises(InvalidInputError, match="choose one of en, es"):
        analyzer_for(language)
