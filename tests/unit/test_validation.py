"""Validation of user-supplied values."""

import pytest

from src.domain.errors import InvalidInputError, RagError
from src.domain.validation import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    validate_chunk_size,
    validate_k,
    validate_query,
)


@pytest.mark.parametrize("k", [1, 10, 1000])
def test_validate_k_accepts_positive_ints(k: int) -> None:
    """Positive integers are returned unchanged."""
    assert validate_k(k) == k


@pytest.mark.parametrize("k", [0, -1, True, False, 1.5, "3", None])
def test_validate_k_rejects(k: object) -> None:
    """Zero, negatives, booleans and non-integers are rejected."""
    with pytest.raises(InvalidInputError, match="k must be"):
        validate_k(k)


@pytest.mark.parametrize("query", ["a", "  lora  ", "é"])
def test_validate_query_accepts(query: str) -> None:
    """Non-blank strings are returned unchanged."""
    assert validate_query(query) == query


@pytest.mark.parametrize("query", ["", "   ", "\n\t", None, 42])
def test_validate_query_rejects(query: object) -> None:
    """Blank strings and non-strings are rejected."""
    with pytest.raises(InvalidInputError, match="empty"):
        validate_query(query)


@pytest.mark.parametrize("size", [MIN_CHUNK_SIZE, 500, MAX_CHUNK_SIZE])
def test_validate_chunk_size_accepts(size: int) -> None:
    """Sizes within the bounds are accepted."""
    assert validate_chunk_size(size) == size


@pytest.mark.parametrize("size", [MIN_CHUNK_SIZE - 1, MAX_CHUNK_SIZE + 1,
                                  0, -5, 5000])
def test_validate_chunk_size_out_of_range(size: int) -> None:
    """Sizes outside the bounds are rejected."""
    with pytest.raises(InvalidInputError, match="between"):
        validate_chunk_size(size)


@pytest.mark.parametrize("size", [True, 100.0, "100", None])
def test_validate_chunk_size_type(size: object) -> None:
    """Non-integers are rejected."""
    with pytest.raises(InvalidInputError, match="integer"):
        validate_chunk_size(size)


def test_errors_are_rag_errors() -> None:
    """Validation errors are user-facing RagErrors."""
    assert issubclass(InvalidInputError, RagError)
