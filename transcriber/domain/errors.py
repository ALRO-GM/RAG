"""Expected errors of the transcriber.

Inbound adapters turn them into a clean CLI message or an HTTP status
code; they never surface as tracebacks.
"""


class TranscriberError(Exception):
    """Base class of every expected, user-facing transcriber error."""


class InvalidInputError(TranscriberError):
    """A value given to the transcriber is invalid."""


class NotFoundError(TranscriberError):
    """A session, transcript, utterance or speaker does not exist."""


class StateError(TranscriberError):
    """An action is not allowed in the current state of a session."""


class ConsentRequiredError(InvalidInputError):
    """A voice cannot be enrolled without the person's explicit consent."""


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


class VoiceModelMismatchError(InvalidInputError):
    """A voiceprint was made with another voice model than the one in use.

    Embeddings of different models are not comparable; the person has
    to enrol again with the current model.
    """


class StorageError(TranscriberError):
    """Stored data cannot be read back (a damaged or foreign file)."""
