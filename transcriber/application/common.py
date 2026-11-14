"""Lookups and checks shared by the use cases.

Every id that reaches a repository is first checked against the domain
pattern of its kind (:func:`check_session_id`, :func:`check_speaker_id`),
so a path such as ``../x`` is refused as invalid input whatever the
adapter behind the port does with it.
"""

import re

from transcriber.application.ports import SessionRepository
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.models import (
    SESSION_ID_PATTERN,
    SPEAKER_ID_PATTERN,
    Session,
    Transcript,
)


def check_id(value: str, pattern: str, kind: str) -> str:
    """Return ``value`` if it matches ``pattern``, else raise.

    ``kind`` names the id in the :class:`InvalidInputError` message.
    """
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise InvalidInputError(f"Invalid {kind} id {value!r}.")
    return value


def check_session_id(session_id: str) -> str:
    """Return ``session_id`` if it is a valid session id, else raise."""
    return check_id(session_id, SESSION_ID_PATTERN, "session")


def check_speaker_id(speaker_id: str) -> str:
    """Return ``speaker_id`` if it is a valid speaker id, else raise."""
    return check_id(speaker_id, SPEAKER_ID_PATTERN, "speaker")


def require_session(sessions: SessionRepository, session_id: str) -> Session:
    """Return the session ``session_id`` or raise :class:`NotFoundError`.

    An invalid id is an :class:`InvalidInputError`.
    """
    session = sessions.get(check_session_id(session_id))
    if session is None:
        raise NotFoundError(f"No session {session_id}.")
    return session


def require_transcript(
    sessions: SessionRepository, session_id: str
) -> Transcript:
    """Return the transcript of ``session_id`` or raise NotFoundError."""
    transcript = sessions.transcript(check_session_id(session_id))
    if transcript is None:
        raise NotFoundError(f"Session {session_id} has no transcript.")
    return transcript
