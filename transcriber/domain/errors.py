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


class BusyError(StateError):
    """The session has a job queued or running; try again when it ends.

    For instance a publish while the speakers' names are still searched:
    the transcript is about to change.
    """


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


class PublishError(TranscriberError):
    """A transcript could not be handed over to the RAG."""


class AnalysisUnavailableError(TranscriberError):
    """The RAG's analysis settings cannot be reached from here."""


class TooLargeError(InvalidInputError):
    """An upload is larger than the RAG accepts."""


class UnsupportedMediaError(InvalidInputError):
    """An uploaded file is not of a type the RAG can read."""


class RagUnavailableError(TranscriberError):
    """The RAG's knowledge routes cannot be reached or failed."""


class LimitReachedError(TranscriberError):
    """A usage limit refuses the request now; it can be tried later."""


class StorageError(TranscriberError):
    """Stored data cannot be read back (a damaged or foreign file)."""


class ModelLoadError(TranscriberError):
    """A model cannot be loaded (not downloaded, or not that kind)."""


class DiarizerError(TranscriberError):
    """The streaming diarizer of a mono live session failed or is absent."""


class GpuBusyError(TranscriberError):
    """The GPU stayed out of memory after the models were freed.

    Other processes hold the memory a stage needs; the stage was tried
    twice. The message says who holds the GPU.
    """


class AccessCodeError(TranscriberError):
    """The kept access code cannot be used."""


class LabStepError(TranscriberError):
    """A lab tool could not produce its stage result.

    ``code`` is one of :data:`transcriber.domain.lab.ErrorCode`
    (``needs_token``, ``out_of_memory``...), stable for the pages.
    """

    def __init__(self, code: str, message: str) -> None:
        """Keep the code with the message."""
        super().__init__(message)
        self.code = code


class CorpusAccessError(TranscriberError):
    """A public corpus cannot be read: gated and not accepted, or absent.

    The message says what the owner has to do (accept the terms on the
    corpus's page and log in).
    """


class PasswordError(TranscriberError):
    """The password cannot be set or its stored hash cannot be used."""
