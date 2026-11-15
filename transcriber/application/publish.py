"""Publishing: hand a finished transcript over to the RAG."""

from typing import Any, Dict

from transcriber.application.common import (
    job_active,
    require_session,
    require_transcript,
)
from transcriber.application.ports import (
    JobRunner,
    SessionLocks,
    SessionRepository,
    TranscriptPublisher,
)
from transcriber.domain.errors import BusyError
from transcriber.domain.models import SessionState
from transcriber.domain.sessions import require_state

BUSY_MESSAGE = (
    "Session {session} still has a job queued or running (the speakers'"
    " names are being searched); publish it when the job ends."
)
"""Why a publish is refused while the session's job has not ended."""


class PublishTranscript:
    """Send the ``transcript.v1`` document of a finished session."""

    def __init__(
        self, sessions: SessionRepository, publisher: TranscriptPublisher,
        locks: SessionLocks, runner: JobRunner,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._publisher = publisher
        self._locks = locks
        self._runner = runner

    def __call__(self, session_id: str) -> Dict[str, Any]:
        """Publish the transcript of a ``done`` session; return it.

        Only text, times and ids leave: the contract carries no audio,
        no words and no voiceprint. While the session's job (its
        transcription, or a search for the speakers' names, which runs
        after the session is already ``done``) is queued or running the
        publish is refused with :class:`BusyError`: the names the job
        is about to give would be missing from the RAG's copy. The
        session's lock is held until the RAG has the document, so a
        deletion (which takes the same lock and then deletes the RAG's
        copy) never runs while a publish is in flight and cannot leave
        a copy behind; the names are applied under the same lock.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            if job_active(self._runner, session_id):
                raise BusyError(BUSY_MESSAGE.format(session=session_id))
            require_state(session, SessionState.DONE)
            document = require_transcript(
                self._sessions, session_id
            ).to_contract()
            self._publisher.publish(document)
        return document
