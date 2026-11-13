"""Query routing and chunk priors: where is the answer likely to live?

A tiny logistic model reads surface features of the question (an
identifier, the words "default value", "how do you"...) and estimates
the probability that the answer is in source code rather than in the
documentation. The retriever uses it to scale chunk scores by kind.
"""

import math
import re
from typing import Dict, List, Sequence, Set

import numpy as np
from pydantic import BaseModel, Field

_SNAKE = re.compile(r"\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b")
_CAMEL = re.compile(r"\b[A-Za-z]+[a-z][A-Z][A-Za-z0-9]*\b")
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]+_[A-Z0-9_]+\b")
_FLAG = re.compile(r"(?<!\w)--[a-z]")
_CODE_WORDS = re.compile(
    r"\b(function|method|class|parameter|argument|attribute|constructor"
    r"|return|returns|returned|type hint|annotation|constant|variable"
    r"|none|raise|raises|raised|exception|decorator|enum|property"
    r"|tensor|dtype|instance|subclass|inherit|inherits|field|module)\b"
)
_DOC_WORDS = re.compile(
    r"\b(how do|how to|how can|how many|where can|install|deploy|docker"
    r"|kubernetes|helm|command|recommended|supported|support|guide"
    r"|documentation|example|tutorial|configure|enable|version|use)\b"
)
_DEFAULT = re.compile(r"\b(default|what happens|what condition)\b")

FEATURE_NAMES = (
    "snake_case", "camel_case", "constant", "cli_flag", "code_words",
    "doc_words", "default_or_condition", "length",
)
_TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$")


def query_features(query: str) -> List[float]:
    """Return the routing features of a question (see FEATURE_NAMES)."""
    lower = query.lower()
    return [
        1.0 if _SNAKE.search(query) else 0.0,
        1.0 if _CAMEL.search(query) else 0.0,
        1.0 if _CONSTANT.search(query) else 0.0,
        1.0 if _FLAG.search(query) else 0.0,
        min(len(_CODE_WORDS.findall(lower)), 3) / 3.0,
        min(len(_DOC_WORDS.findall(lower)), 3) / 3.0,
        1.0 if _DEFAULT.search(lower) else 0.0,
        min(len(lower.split()), 30) / 30.0,
    ]


_WORD = re.compile(r"[a-z]+")


def query_words(query: str) -> Set[str]:
    """Return the lowercase words of a question."""
    return set(_WORD.findall(query.lower()))


def _sigmoid(x: float) -> float:
    """Return the logistic function, safe for large magnitudes."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class RouterModel(BaseModel):
    """Logistic regression over surface features and frequent words.

    ``weights`` holds one weight per entry of ``FEATURE_NAMES`` followed
    by one per word of ``vocabulary`` (1 when the question has it).
    """

    weights: List[float]
    bias: float = 0.0
    vocabulary: List[str] = Field(default_factory=list)

    def features(self, query: str) -> List[float]:
        """Return the feature vector of a question."""
        words = query_words(query)
        return query_features(query) + [
            1.0 if word in words else 0.0 for word in self.vocabulary
        ]

    def p_code(self, query: str) -> float:
        """Return the probability that the answer is in source code."""
        z = self.bias + sum(
            w * x for w, x in zip(self.weights, self.features(query))
        )
        return _sigmoid(z)


def fit_router(
    queries: Sequence[str],
    labels: Sequence[int],
    l2: float = 0.003,
    epochs: int = 1500,
    learning_rate: float = 0.5,
    min_count: int = 5,
) -> RouterModel:
    """Fit the router by gradient descent (label 1 = answer in code).

    Words seen in at least ``min_count`` questions join the features;
    rarer words are ignored so the model cannot memorise questions.
    """
    if len(queries) != len(labels) or not queries:
        raise ValueError("Need one label per query and at least one query.")
    counts: Dict[str, int] = {}
    for query in queries:
        for word in query_words(query):
            counts[word] = counts.get(word, 0) + 1
    vocabulary = sorted(w for w, c in counts.items() if c >= min_count)
    model = RouterModel(
        weights=[0.0] * (len(FEATURE_NAMES) + len(vocabulary)),
        vocabulary=vocabulary,
    )
    x = np.asarray([model.features(q) for q in queries], dtype=float)
    y = np.asarray(labels, dtype=float)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(epochs):
        error = 1.0 / (1.0 + np.exp(-(x @ weights + bias))) - y
        weights -= learning_rate * (x.T @ error / len(y) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    model.weights = [float(w) for w in weights]
    model.bias = bias
    return model


def is_test_path(path: str) -> bool:
    """Tell whether ``path`` is a test file (``tests/``, ``test_*.py``)."""
    return bool(_TEST_PATH.search(path))
