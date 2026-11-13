"""Lexical tokenizer: identifier splitting, stopwords, stemming."""

import pytest

from src.domain.text_analysis import (
    STOPWORDS,
    adaptive_weight,
    identifier_share,
    tokenize,
)


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


@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("How does the sampler work?", 0.0),
    ("Is there any evidence that vitamin D helps?", 0.0),
    ("What does KVCacheManager do?", 1 / 4),
    ("call load_lora_adapter() in engine.py with --max-loras", 3 / 6),
    ("defaults of FP8_MIN in vLLM's triton_flash_attention module?",
     3 / 7),
    ("see `max_tokens` here", 1 / 3),
    ("open src/domain/ranking.py please", 1 / 3),
    ("the camelCase name", 1 / 3),
    ("A sentence. Another one, with commas; done!", 0.0),
])
def test_identifier_share(text: str, expected: float) -> None:
    """Code-like tokens are counted; prose and punctuation are not."""
    assert identifier_share(text) == pytest.approx(expected)


@pytest.mark.parametrize("share, expected", [
    (0.0, 4.0), (0.05, 3.0), (0.1, 2.0), (0.5, 2.0),
])
def test_adaptive_weight_slides_to_the_identifier_weight(
    share: float, expected: float
) -> None:
    """Linear from the prose weight to the identifier weight."""
    assert adaptive_weight(4.0, 2.0, share, 0.1) == pytest.approx(expected)


def test_adaptive_weight_without_saturation_is_a_switch() -> None:
    """Saturation 0: any identifier switches to the identifier weight."""
    assert adaptive_weight(4.0, 1.0, 0.0, 0.0) == 4.0
    assert adaptive_weight(4.0, 1.0, 0.01, 0.0) == 1.0
