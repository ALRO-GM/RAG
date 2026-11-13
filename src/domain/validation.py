"""Validation of user-supplied values shared by every entry point."""

from src.domain.errors import InvalidInputError

MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 50


def validate_k(k: object) -> int:
    """Return ``k`` if it is a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise InvalidInputError(f"k must be a positive integer (got {k!r}).")
    return k


def validate_query(query: object) -> str:
    """Return ``query`` as a non-blank string."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("The query is empty.")
    return query


def validate_chunk_size(size: object) -> int:
    """Return ``size`` if it is an integer within the allowed range."""
    if isinstance(size, bool) or not isinstance(size, int):
        raise InvalidInputError(
            f"max_chunk_size must be an integer (got {size!r})."
        )
    if not MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE:
        raise InvalidInputError(
            f"max_chunk_size must be between {MIN_CHUNK_SIZE} and "
            f"{MAX_CHUNK_SIZE}."
        )
    return size
