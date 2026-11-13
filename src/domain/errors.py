"""User-facing errors raised anywhere in the system.

Inbound adapters translate them into a clean CLI message or an HTTP
status code; they never surface as tracebacks.
"""


class RagError(Exception):
    """Base class of every expected, user-facing error."""


class InvalidInputError(RagError):
    """A user-supplied value is invalid (empty query, k <= 0, ...)."""


class NotFoundError(RagError):
    """A required file, directory or index does not exist."""


class CorruptedDataError(RagError):
    """A file exists but cannot be parsed or is inconsistent."""


class ModelError(RagError):
    """A machine-learning model could not be loaded or run."""


def short_reason(exc: Exception) -> str:
    """Return a one-line reason for a (pydantic) validation error."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ()))
            return f"{where}: {first.get('msg', '')}".strip(": ")
        except (IndexError, TypeError, AttributeError):
            pass
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
