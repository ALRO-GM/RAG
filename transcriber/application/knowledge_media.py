"""Use case: transcribe an audio or video file of the knowledge library.

The RAG keeps the knowledge library (``docs/KNOWLEDGE.md``) and has no
speech model. When an audio or video file is uploaded to a domain, the
RAG's worker sends it here (KN-3a, #244) and this use case runs it
through the normal pipeline (:class:`~transcriber.application.
transcribe.TranscribeSession`, in the same job runner, so the GPU
lease of the live mode is honoured as for any batch job).

The jobs live in a session repository of their own
(``<data_dir>/knowledge-media``), never in the one the session list,
the player, publishing and the analysis read: a knowledge transcription
does not appear in «Sesiones», is never published as ``transcript.v1``
to ``POST /transcripts`` and is never analysed. Its voices are not
compared with the enrolled voiceprints (``expected_speakers`` empty):
the speakers of a training video stay anonymous labels. The RAG polls
the job, reads the transcript (``transcript.v1``, the contract it
already reads) and deletes the job; whatever a stopped process left is
deleted when the service starts (:meth:`KnowledgeMediaJobs.clear`).
"""

import re
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from transcriber.application.ports import (
    RecordedAudio,
    SessionLocks,
    SessionRepository,
)
from transcriber.application.record import ImportRecording
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscriptionOptions,
    TranscriptionStatus,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StateError,
    UnsupportedMediaError,
)
from transcriber.domain.models import SessionState

JOB_ID_PATTERN = r"kn-[0-9a-f]{32}"
"""Ids of knowledge-media jobs (the RAG derives them from the document)."""

KNOWLEDGE_MEDIA_DIR = "knowledge-media"
"""Folder of the jobs' repository, under the transcriber's data dir."""

KEPT = (SessionState.QUEUED, SessionState.RUNNING, SessionState.DONE)
"""States in which a job sent again is left as it is."""


class MediaJob(BaseModel):
    """Where a knowledge-media job is, as the RAG reads it.

    ``state`` is the session state (``queued``, ``running``, ``done``,
    ``failed``); ``message`` the reason of a failure or the job's last
    note; ``transcript`` the ``transcript.v1`` document once done.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    state: str
    stage: Optional[str] = None
    fraction: float = 0.0
    message: Optional[str] = None
    duration: Optional[float] = None
    transcript: Optional[Dict[str, Any]] = None


def check_job_id(job_id: str) -> str:
    """Return ``job_id`` if it is a knowledge-media job id, else raise."""
    if not re.fullmatch(JOB_ID_PATTERN, job_id or ""):
        raise InvalidInputError(f"Invalid knowledge-media job {job_id!r}.")
    return job_id


class KnowledgeMediaJobs:
    """Queue, report and delete the transcription of a library file.

    Args:
        sessions: The repository of these jobs only (not the sessions').
        locks: Per-id locks of that repository.
        importer: Stores a recording as an ``uploaded`` session there.
        queue: Queues a session of that repository in the job runner.
        status: Tells where a session of that repository is.
        formats: The file formats accepted (the API's upload formats).
    """

    def __init__(
        self,
        sessions: SessionRepository,
        locks: SessionLocks,
        importer: ImportRecording,
        queue: QueueTranscription,
        status: TranscriptionStatus,
        formats: Tuple[str, ...],
    ) -> None:
        """Wire the use cases of the jobs' own repository."""
        self._sessions = sessions
        self._locks = locks
        self._import = importer
        self._queue = queue
        self._status = status
        self._formats = formats

    def submit(
        self,
        job_id: str,
        file_format: str,
        data: bytes,
        language: str,
        max_seconds: Optional[float] = None,
    ) -> MediaJob:
        """Start transcribing ``data``; return the job.

        Sending a job that is queued, running or done again changes
        nothing (the RAG's worker may retry after a lost answer); a
        failed one is started again. ``max_seconds`` is the longest
        recording accepted: a longer one fails at the decode stage.
        """
        check_job_id(job_id)
        if file_format not in self._formats:
            raise UnsupportedMediaError(
                f"Unsupported recording format {file_format!r}; use "
                f"{', '.join(self._formats)}."
            )
        with self._locks.hold(job_id):
            existing = self._sessions.get(job_id)
            if existing is not None and existing.state in KEPT:
                return self.status(job_id)
            if existing is not None:
                self._sessions.delete(job_id)
        self._import(job_id, language, RecordedAudio(data, file_format))
        self._queue(job_id, TranscriptionOptions(
            expected_speakers=[], max_seconds=max_seconds,
        ))
        return self.status(job_id)

    def status(self, job_id: str) -> MediaJob:
        """Return the job; with its ``transcript.v1`` once done."""
        session = self._sessions.get(check_job_id(job_id))
        if session is None:
            raise NotFoundError(f"No knowledge-media job {job_id}.")
        job = self._status.of(session)
        transcript: Optional[Dict[str, Any]] = None
        if session.state is SessionState.DONE:
            stored = self._sessions.transcript(job_id)
            if stored is None:
                raise NotFoundError(f"The job {job_id} has no transcript.")
            transcript = stored.to_contract()
        failed = session.state is SessionState.FAILED
        return MediaJob(
            id=job_id, state=session.state.value, stage=job.stage,
            fraction=job.fraction,
            message=session.failure if failed else job.message,
            duration=session.duration or None, transcript=transcript,
        )

    def forget(self, job_id: str) -> None:
        """Delete a job and everything stored for it (absent: nothing).

        A running job is refused (:class:`StateError`): the RAG deletes
        it once it has ended.
        """
        with self._locks.hold(check_job_id(job_id)):
            session = self._sessions.get(job_id)
            if session is None:
                return
            if session.state is SessionState.RUNNING:
                raise StateError(
                    f"The job {job_id} is running; delete it when it ends."
                )
            self._sessions.delete(job_id)

    def clear(self) -> int:
        """Delete every job a previous process left; return how many.

        Only for the start of the process that runs the jobs: its runner
        is empty, so no job is in progress. A RAG still waiting for one
        of them finds it gone and sends the file again.
        """
        count = 0
        for job_id in self._sessions.ids():
            with self._locks.hold(job_id):
                self._sessions.delete(job_id)
            count += 1
        return count
