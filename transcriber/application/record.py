"""Recording use cases: open a session, upload its audio, close it.

A session recorded from the phone is opened, receives numbered audio
chunks (:class:`AppendAudio`) and is closed, which makes it ready to
transcribe. A recording made with another app is imported whole
(:class:`ImportRecording`). :class:`DeleteSession` removes a session
with everything derived from it.
"""

import hashlib
from typing import Any, Optional

from pydantic import ValidationError

from transcriber.application.common import require_session
from transcriber.application.ports import (
    Clock,
    RecordedAudio,
    SessionRepository,
)
from transcriber.domain.errors import InvalidInputError, StateError
from transcriber.domain.models import Session, SessionState
from transcriber.domain.sessions import (
    ChunkAction,
    accept_chunk,
    move,
    with_chunk,
)


def _make_session(**fields: Any) -> Session:
    """Build a session, turning a validation error into an input error."""
    try:
        return Session(**fields)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise InvalidInputError(
            f"Invalid session {where}: {first.get('msg', '')}"
        ) from exc


def _new_session(
    sessions: SessionRepository, session: Session
) -> Session:
    """Store ``session`` unless its id is taken."""
    if sessions.get(session.id) is not None:
        raise InvalidInputError(f"Session {session.id} already exists.")
    sessions.create(session)
    return session


class OpenSession:
    """Start recording a new session."""

    def __init__(self, sessions: SessionRepository, clock: Clock) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._clock = clock

    def __call__(
        self, session_id: str, language: str, title: Optional[str] = None
    ) -> Session:
        """Create session ``session_id`` in state ``recording``.

        ``started_at`` is the clock's current time.
        """
        return _new_session(self._sessions, _make_session(
            id=session_id, title=title, language=language,
            started_at=self._clock.now(),
        ))


class AppendAudio:
    """Receive one numbered chunk of a recording in progress."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self, session_id: str, number: int, data: bytes) -> bool:
        """Append chunk ``number`` (from 0); tell whether it was new.

        Repeating the last accepted chunk with the same bytes is a
        no-op that returns ``False``; see
        :func:`~transcriber.domain.sessions.accept_chunk` for the
        errors. Only a session in ``recording`` accepts chunks.
        """
        if not data:
            raise InvalidInputError("An audio chunk cannot be empty.")
        session = require_session(self._sessions, session_id)
        digest = hashlib.sha256(data).hexdigest()
        if accept_chunk(session, number, digest) is ChunkAction.DUPLICATE:
            return False
        self._sessions.append_chunk(session_id, number, data)
        self._sessions.save(with_chunk(session, digest))
        return True


class CloseSession:
    """End the recording; the session is then ready to transcribe."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self, session_id: str) -> Session:
        """Move the session from ``recording`` to ``uploaded``.

        A session without any audio cannot be closed.
        """
        session = require_session(self._sessions, session_id)
        if session.state is SessionState.RECORDING and session.chunks == 0:
            raise InvalidInputError(
                f"Session {session_id} has no audio to close."
            )
        closed = move(session, SessionState.UPLOADED)
        self._sessions.save(closed)
        return closed


class ImportRecording:
    """Create a session from a recording made with another app."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(
        self,
        session_id: str,
        language: str,
        recording: RecordedAudio,
        title: Optional[str] = None,
    ) -> Session:
        """Store ``recording`` as session ``session_id``, ``uploaded``."""
        if not recording.data:
            raise InvalidInputError("The recording is empty.")
        if not recording.format.strip():
            raise InvalidInputError("The recording needs a format.")
        session = _new_session(self._sessions, _make_session(
            id=session_id, title=title, language=language,
            state=SessionState.UPLOADED,
        ))
        self._sessions.store_recording(session_id, recording)
        return session


class DeleteSession:
    """Remove a session, its audio, its words and its transcript."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self, session_id: str) -> None:
        """Delete the session; not while it is being transcribed."""
        session = require_session(self._sessions, session_id)
        if session.state is SessionState.RUNNING:
            raise StateError(
                f"Session {session_id} is being transcribed; delete it"
                " when the job ends."
            )
        self._sessions.delete(session_id)
