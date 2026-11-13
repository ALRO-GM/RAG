"""Lookups shared by the use cases."""

from transcriber.application.ports import SessionRepository
from transcriber.domain.errors import NotFoundError
from transcriber.domain.models import Session, Transcript


def require_session(sessions: SessionRepository, session_id: str) -> Session:
    """Return the session ``session_id`` or raise :class:`NotFoundError`."""
    session = sessions.get(session_id)
    if session is None:
        raise NotFoundError(f"No session {session_id}.")
    return session


def require_transcript(
    sessions: SessionRepository, session_id: str
) -> Transcript:
    """Return the transcript of ``session_id`` or raise NotFoundError."""
    transcript = sessions.transcript(session_id)
    if transcript is None:
        raise NotFoundError(f"Session {session_id} has no transcript.")
    return transcript
