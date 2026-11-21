"""User-facing errors raised anywhere in the system.

Inbound adapters translate them into a clean CLI message or an HTTP
status code; they never surface as tracebacks.
"""


from datetime import datetime
from typing import Optional


class RagError(Exception):
    """Base class of every expected, user-facing error."""


class InvalidInputError(RagError):
    """A user-supplied value is invalid (empty query, k <= 0, ...)."""


class UnsupportedDocumentError(InvalidInputError):
    """An uploaded file is not of a supported type (by its content)."""


class TooLargeError(InvalidInputError):
    """An upload is larger than allowed."""


class NotFoundError(RagError):
    """A required file, directory or index does not exist."""


class BusyError(RagError):
    """The work asked for is already running."""


class LimitError(RagError):
    """A usage limit refuses the request now (try again later)."""


class CorruptedDataError(RagError):
    """A file exists but cannot be parsed or is inconsistent."""


class ModelError(RagError):
    """A machine-learning model could not be loaded or run."""


class QuotaExhaustedError(ModelError):
    """A hosted model refused because the usage quota is used up.

    ``resets`` says when the quota comes back, as the service wrote it
    (empty when it did not say); ``resets_at`` is that moment when the
    service gave it as a timestamp. A judge that can fall back to the
    local model does so instead of failing.
    """

    def __init__(
        self,
        message: str,
        resets: str = "",
        resets_at: Optional[datetime] = None,
    ):
        """Keep the message and when the quota resets."""
        super().__init__(message)
        self.resets = resets
        self.resets_at = resets_at


class ModelAuthError(ModelError):
    """A hosted model refused the credentials (not logged in, expired)."""


class TransientModelError(ModelError):
    """A hosted model failed in a way worth retrying (overload, 5xx)."""


class OutputTooLongError(ModelError):
    """A model's reply went past the output-token cap it was given."""


class ModelTimeoutError(ModelError):
    """A model did not answer within the time it was given."""


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
