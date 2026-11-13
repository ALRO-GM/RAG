"""Query routing features, the router model and the test-path prior."""

from typing import List, Tuple

import pytest

from src.domain.priors import (
    FEATURE_NAMES,
    RouterModel,
    fit_router,
    is_test_path,
    query_features,
    query_words,
)

CODE_QUERY = "What is the default value of foo_bar in BazClass?"
DOCS_QUERY = "How do you install X with Docker?"

IDENTIFIERS = [
    ("max_tokens", "SamplingParams"), ("block_size", "CacheConfig"),
    ("num_workers", "EngineArgs"), ("top_k", "BeamSearch"),
    ("seed_value", "ModelConfig"), ("lora_rank", "LoraRequest"),
    ("gpu_fraction", "DeviceConfig"), ("swap_space", "SchedulerConfig"),
    ("max_len", "PromptAdapter"), ("log_level", "ServerArgs"),
]
PRODUCTS = [
    "vLLM", "the server", "the chart", "the proxy", "the gateway",
    "the client", "the plugin", "the worker", "the router", "the cache",
]


def training_set() -> Tuple[List[str], List[int]]:
    """Return code-style (label 1) and docs-style (label 0) questions."""
    queries = [
        f"What is the default value of {name} in {cls}?"
        for name, cls in IDENTIFIERS
    ] + [f"How do you install {p} with Docker?" for p in PRODUCTS]
    return queries, [1] * len(IDENTIFIERS) + [0] * len(PRODUCTS)


def feature(query: str, name: str) -> float:
    """Return one named routing feature of ``query``."""
    return query_features(query)[FEATURE_NAMES.index(name)]


def test_query_features_of_a_code_question() -> None:
    """Identifiers and 'default value' are detected; no docs word."""
    assert query_features(CODE_QUERY) == pytest.approx(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 9 / 30]
    )


def test_query_features_of_a_docs_question() -> None:
    """'how do', 'install' and 'docker' saturate the docs-word feature."""
    assert query_features(DOCS_QUERY) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 7 / 30]
    )


def test_query_features_has_one_value_per_name() -> None:
    """The vector is aligned with FEATURE_NAMES, also when empty."""
    assert len(query_features("")) == len(FEATURE_NAMES)
    assert query_features("") == [0.0] * len(FEATURE_NAMES)


@pytest.mark.parametrize("query, name, expected", [
    ("Where is MAX_NUM_SEQS set?", "constant", 1.0),
    ("Where is max_num_seqs set?", "constant", 0.0),
    ("What does --enable-lora do?", "cli_flag", 1.0),
    ("A range -- like 1 to 2", "cli_flag", 0.0),
    ("Which class has the method?", "code_words", 2 / 3),
    ("function method class parameter argument", "code_words", 1.0),
    ("Is there a tutorial?", "doc_words", 1 / 3),
    ("What happens when it fails?", "default_or_condition", 1.0),
    ("getValue", "camel_case", 1.0),
    ("Docker", "camel_case", 0.0),
    (" ".join(["word"] * 45), "length", 1.0),
])
def test_query_feature_values(
    query: str, name: str, expected: float
) -> None:
    """Each surface feature fires on its pattern and is capped at 1."""
    assert feature(query, name) == pytest.approx(expected)


def test_query_words_are_lowercase_alphabetic() -> None:
    """Words are lowercased; digits and punctuation split them."""
    assert query_words("How do-you Install X2, how?") == {
        "how", "do", "you", "install", "x",
    }
    assert query_words("") == set()


def test_router_features_append_vocabulary_flags() -> None:
    """The feature vector ends with one 0/1 flag per vocabulary word."""
    model = RouterModel(
        weights=[0.0] * (len(FEATURE_NAMES) + 2),
        vocabulary=["docker", "tensor"],
    )
    features = model.features(DOCS_QUERY)
    assert features[:len(FEATURE_NAMES)] == query_features(DOCS_QUERY)
    assert features[len(FEATURE_NAMES):] == [1.0, 0.0]


def test_router_p_code_is_a_logistic_of_the_features() -> None:
    """Weights and bias combine into a probability."""
    weights = [0.0] * len(FEATURE_NAMES)
    assert RouterModel(weights=weights).p_code(CODE_QUERY) == 0.5
    weights[FEATURE_NAMES.index("snake_case")] = 5.0
    model = RouterModel(weights=weights, bias=-2.0)
    assert model.p_code(CODE_QUERY) == pytest.approx(0.95257, abs=1e-4)
    assert model.p_code(DOCS_QUERY) == pytest.approx(0.11920, abs=1e-4)


def test_router_vocabulary_weight_moves_the_probability() -> None:
    """A negative weight on a word lowers p_code when it is present."""
    model = RouterModel(
        weights=[0.0] * len(FEATURE_NAMES) + [-4.0], vocabulary=["docker"]
    )
    assert model.p_code(DOCS_QUERY) < 0.05
    assert model.p_code(CODE_QUERY) == 0.5


@pytest.mark.parametrize("bias, expected", [(1000.0, 1.0), (-1000.0, 0.0)])
def test_router_p_code_is_safe_for_large_magnitudes(
    bias: float, expected: float
) -> None:
    """Huge logits saturate instead of overflowing."""
    model = RouterModel(weights=[0.0] * len(FEATURE_NAMES), bias=bias)
    assert model.p_code("anything") == pytest.approx(expected)


def test_fit_router_separates_code_from_docs_questions() -> None:
    """Training and unseen questions are routed to the right side."""
    queries, labels = training_set()
    model = fit_router(queries, labels)
    for query, label in zip(queries, labels):
        assert (model.p_code(query) >= 0.5) == bool(label)
    assert model.p_code(CODE_QUERY) > 0.8
    assert model.p_code(DOCS_QUERY) < 0.2
    assert model.p_code("What is the default of cache_dir?") > 0.5
    assert model.p_code("How to deploy with Kubernetes?") < 0.5


def test_fit_router_vocabulary_keeps_frequent_words_only() -> None:
    """Words below ``min_count`` questions are left out; one weight each."""
    queries, labels = training_set()
    model = fit_router(queries, labels, min_count=5)
    assert model.vocabulary == sorted(model.vocabulary)
    assert {"default", "value", "docker", "install"} <= set(model.vocabulary)
    assert "samplingparams" not in model.vocabulary
    assert "gateway" not in model.vocabulary
    assert len(model.weights) == len(FEATURE_NAMES) + len(model.vocabulary)
    everything = fit_router(queries, labels, min_count=1, epochs=5)
    assert "gateway" in everything.vocabulary
    nothing = fit_router(queries, labels, min_count=1000, epochs=5)
    assert nothing.vocabulary == []
    assert len(nothing.weights) == len(FEATURE_NAMES)


def test_fit_router_is_deterministic() -> None:
    """Two fits on the same data give the same model."""
    queries, labels = training_set()
    assert fit_router(queries, labels, epochs=50) == fit_router(
        queries, labels, epochs=50
    )


def test_fit_router_learns_the_majority_class_in_the_bias() -> None:
    """With a single class the bias alone pushes towards that class."""
    model = fit_router(["alpha", "beta", "gamma"], [1, 1, 1], epochs=200)
    assert model.bias > 0
    assert model.p_code("unrelated words") > 0.5


@pytest.mark.parametrize("queries, labels", [
    ([], []),
    (["one question"], []),
    (["one", "two"], [1]),
])
def test_fit_router_rejects_bad_input(
    queries: List[str], labels: List[int]
) -> None:
    """No query, or a label count that differs, is a ValueError."""
    with pytest.raises(ValueError, match="one label per query"):
        fit_router(queries, labels)


@pytest.mark.parametrize("path, expected", [
    ("tests/test_engine.py", True),
    ("pkg/tests/helpers.py", True),
    ("test/conftest.py", True),
    ("pkg/testing/utils.py", True),
    ("pkg/test_engine.py", True),
    ("test_engine.py", True),
    ("pkg/engine_test.py", True),
    ("pkg/engine.py", False),
    ("pkg/latest/engine.py", False),
    ("docs/testing.md", False),
    ("pkg/contest.py", False),
    ("pkg/attest_util.py", False),
    ("", False),
])
def test_is_test_path(path: str, expected: bool) -> None:
    """Test directories and test_*/*_test.py files are recognised."""
    assert is_test_path(path) is expected
