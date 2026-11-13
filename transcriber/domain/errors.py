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


class SampleRejectedError(InvalidInputError):
    """An enrolment sample cannot give a reliable voiceprint.

    ``reason`` is a :class:`~transcriber.domain.enrolment.Rejection`
    value, so a caller can show the person what to do differently.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Keep the machine-readable ``reason`` beside the message."""
        super().__init__(message)
        self.reason = reason
