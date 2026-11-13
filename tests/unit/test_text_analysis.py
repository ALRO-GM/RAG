"""Lexical tokenizer: identifier splitting, stopwords, stemming."""

import pytest

from src.domain.text_analysis import STOPWORDS, tokenize


def test_snake_case_identifier_kept_and_split() -> None:
    """The whole identifier and its parts are emitted."""
    assert tokenize("get_kv_cache") == ["get_kv_cache", "get", "kv", "cache"]


def test_camel_case_identifier_kept_and_split() -> None:
    """CamelCase and acronyms are split into lowercase parts."""
    assert tokenize("KVCacheManager") == [
        "kvcachemanager", "kv", "cache", "manager",
    ]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_paraphrase_matches_identifier_parts() -> None:
    """A natural-language query shares terms with the identifier."""
    assert set(tokenize("kv cache")) <= set(tokenize("get_kv_cache"))


def test_stopwords_and_single_letters_removed() -> None:
    """Stopwords (including project words) and 1-char parts are dropped."""
    assert tokenize("the vllm is used by a model") == ["model"]
    assert "vllm" in STOPWORDS
    assert tokenize("x y z") == []


def test_numbers_kept() -> None:
    """Digits are kept as terms; alphanumerics are also split."""
    assert tokenize("port 8000 fp8") == ["port", "8000", "fp8", "fp"]


@pytest.mark.parametrize("word, stem", [
    ("models", "model"),
    ("loading", "load"),
    ("quantized", "quantiz"),
    ("policies", "policy"),
    ("status", "status"),
    ("analysis", "analysis"),
    ("class", "class"),
    ("bus", "bus"),
    ("tokens", "token"),
    ("sing", "sing"),
])
def test_light_stemmer(word: str, stem: str) -> None:
    """Common plural and verb suffixes are removed."""
    assert tokenize(word) == [stem]


def test_sses_plural_matches_singular() -> None:
    """``classes`` and ``class`` share a stem."""
    assert tokenize("classes") == tokenize("class")
    assert tokenize("libraries") == tokenize("library")


def test_query_and_document_consistency() -> None:
    """Plural/singular and inflections meet on the same term."""
    assert tokenize("Loading adapters") == tokenize("load adapter")


def test_empty_and_punctuation() -> None:
    """Text without words yields no term."""
    assert tokenize("") == []
    assert tokenize("!!! --- ???") == []


def test_bigrams_add_consecutive_stem_pairs() -> None:
    """With bigrams, adjacent word stems also form a term."""
    assert tokenize("data parallel deployment", bigrams=True) == [
        "data", "parallel", "deployment",
        "data~parallel", "parallel~deployment",
    ]
    assert tokenize("data parallel") == ["data", "parallel"]


def test_bigrams_skip_stopwords_and_use_stems() -> None:
    """Pairs are built from the kept stems, across removed stopwords."""
    assert "load~adapter" in tokenize("loading the adapters", bigrams=True)
    assert tokenize("model", bigrams=True) == ["model"]
