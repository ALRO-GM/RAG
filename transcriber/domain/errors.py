"""Expected errors of the transcriber.

Inbound adapters turn them into a clean CLI message or an HTTP status
code; they never surface as tracebacks.
"""


class TranscriberError(Exception):
    """Base class of every expected, user-facing transcriber error."""


class InvalidInputError(TranscriberError):
    """A value given to the transcriber is invalid."""


class ContractError(TranscriberError):
    """A document does not follow the ``transcript.v1`` contract."""
