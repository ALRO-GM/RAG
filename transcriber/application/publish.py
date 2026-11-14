"""Publishing: hand a finished transcript over to the RAG."""

from typing import Any, Dict

from transcriber.application.common import require_session, require_transcript
from transcriber.application.ports import (
    SessionRepository,
    TranscriptPublisher,
)
from transcriber.domain.models import SessionState
from transcriber.domain.sessions import require_state


class PublishTranscript:
    """Send the ``transcript.v1`` document of a finished session."""

    def __init__(
        self, sessions: SessionRepository, publisher: TranscriptPublisher
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._publisher = publisher

    def __call__(self, session_id: str) -> Dict[str, Any]:
        """Publish the transcript of a ``done`` session; return it.

        Only text, times and ids leave: the contract carries no audio,
        no words and no voiceprint.
        """
        session = require_session(self._sessions, session_id)
        require_state(session, SessionState.DONE)
        document = require_transcript(self._sessions, session_id).to_contract()
        self._publisher.publish(document)
        return document
