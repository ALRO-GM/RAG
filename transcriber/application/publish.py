"""Publishing: hand a finished transcript over to the RAG."""

from typing import Any, Dict

from transcriber.application.common import require_session, require_transcript
from transcriber.application.ports import (
    SessionLocks,
    SessionRepository,
    TranscriptPublisher,
)
from transcriber.domain.models import SessionState
from transcriber.domain.sessions import require_state


class PublishTranscript:
    """Send the ``transcript.v1`` document of a finished session."""

    def __init__(
        self, sessions: SessionRepository, publisher: TranscriptPublisher,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._publisher = publisher
        self._locks = locks

    def __call__(self, session_id: str) -> Dict[str, Any]:
        """Publish the transcript of a ``done`` session; return it.

        Only text, times and ids leave: the contract carries no audio,
        no words and no voiceprint. The session's lock is held until the
        RAG has the document, so a deletion (which takes the same lock
        and then deletes the RAG's copy) never runs while a publish is
        in flight and cannot leave a copy behind.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            require_state(session, SessionState.DONE)
            document = require_transcript(
                self._sessions, session_id
            ).to_contract()
            self._publisher.publish(document)
        return document
