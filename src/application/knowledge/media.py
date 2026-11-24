"""Use case: the text of an audio or video file, from the transcriber (KN-3a).

The RAG loads no speech model. A recording uploaded to a domain is sent
to the transcriber service (:class:`~src.application.knowledge.ports.
MediaTranscriber`), which runs it through its own pipeline and GPU
rules; this use case waits for the job, turns the ``transcript.v1`` it
returns into the document's timed text
(:func:`~src.domain.knowledge.media.transcript_markdown`) and deletes
the transcriber's copy. Every failure is an ``InvalidInputError`` whose
message is the one the library page shows.
"""

import logging
import time
from typing import Callable, Optional

from src.application.knowledge.ports import MediaJob, MediaTranscriber
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    RagError,
    ServiceUnavailableError,
)
from src.domain.knowledge.media import (
    clock,
    media_failure,
    transcript_markdown,
)
from src.domain.skills.transcript_contract import read_contract

LOG = logging.getLogger(__name__)

UNREACHABLE = (
    "No se pudo contactar con el servicio de transcripción; vuelve a "
    "intentarlo cuando esté en marcha. / The transcription service "
    "cannot be reached; try again once it is running."
)
LOST = (
    "El servicio de transcripción perdió el trabajo dos veces (¿se "
    "reinició?); vuelve a subir el archivo. / The transcription service "
    "lost the job twice (restarted?); upload the file again."
)
NO_SPEECH = (
    "No se oye voz en el archivo. / No speech was found in the file."
)
TERMINAL = ("done", "failed")


class MediaTranscription:
    """Wait for the transcription of one recording; return its text.

    Args:
        transcriber: The transcriber service.
        max_seconds: The longest recording accepted (the transcriber
            fails a longer one before any model runs).
        poll_seconds: Time between two questions about the job.
        max_wait_seconds: Longest wait for a job, queue included (a
            live session keeps the GPU and delays batch jobs).
        unreachable_seconds: How long the service may stay unreachable
            while a job runs before the document fails.
        sleep: Waits a number of seconds (tests pass a fake).
        now: A monotonic clock in seconds (tests pass a fake).
    """

    def __init__(
        self,
        transcriber: MediaTranscriber,
        max_seconds: float,
        poll_seconds: float = 5.0,
        max_wait_seconds: float = 6 * 3600.0,
        unreachable_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep the port, the limits and the clock."""
        self.transcriber = transcriber
        self.max_seconds = max_seconds
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.unreachable_seconds = unreachable_seconds
        self.sleep = sleep
        self.now = now

    def transcribe(
        self,
        job_id: str,
        data: bytes,
        file_format: str,
        language: str,
        wanted: Callable[[], bool] = lambda: True,
    ) -> Optional[str]:
        """Return the timed text of the recording ``data``.

        ``wanted`` tells whether the document still exists: once it is
        gone the job is deleted at the transcriber (as soon as it is no
        longer running) and ``None`` is returned. A job the transcriber
        lost (it restarted) is sent once more.
        """
        def submit() -> MediaJob:
            """Send the file."""
            return self.transcriber.submit(
                job_id, data, file_format, language, self.max_seconds
            )

        try:
            job = submit()
        except ServiceUnavailableError:
            raise InvalidInputError(UNREACHABLE) from None
        started = seen = self.now()
        resent = False
        while job.state not in TERMINAL:
            if not wanted():
                self._forget(job_id)
                return None
            if self.now() - started > self.max_wait_seconds:
                self._forget(job_id)
                raise InvalidInputError(
                    f"La transcripción no terminó en "
                    f"{clock(self.max_wait_seconds)}. / The transcription "
                    f"did not end within {clock(self.max_wait_seconds)}."
                )
            self.sleep(self.poll_seconds)
            try:
                job = self.transcriber.status(job_id)
                seen = self.now()
            except NotFoundError:
                if resent:
                    raise InvalidInputError(LOST) from None
                resent = True
                job = submit()
            except ServiceUnavailableError:
                if self.now() - seen > self.unreachable_seconds:
                    raise InvalidInputError(UNREACHABLE) from None
        self._forget(job_id)
        if not wanted():
            return None
        if job.state == "failed":
            raise InvalidInputError(media_failure(job.message or ""))
        return self._text(job, language)

    @staticmethod
    def _text(job: MediaJob, language: str) -> str:
        """Return the timed text of a done job's transcript."""
        try:
            transcript = read_contract(job.transcript)
        except InvalidInputError as exc:
            raise InvalidInputError(media_failure(str(exc))) from None
        text = transcript_markdown(transcript, language)
        if not text.strip():
            raise InvalidInputError(NO_SPEECH)
        return text

    def _forget(self, job_id: str) -> None:
        """Delete the transcriber's copy; wait while the job still runs.

        A failure is logged: the transcriber also deletes what is left
        when it starts.
        """
        for _ in range(int(self.max_wait_seconds // self.poll_seconds) + 1):
            try:
                self.transcriber.forget(job_id)
                return
            except BusyError:
                self.sleep(self.poll_seconds)
            except RagError as exc:
                LOG.warning("Cannot delete the transcriber's job %s: %s",
                            job_id, exc)
                return
