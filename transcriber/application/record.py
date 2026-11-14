"""Recording use cases: open a session, upload its audio, close it.

A session recorded from the phone is opened, receives numbered audio
chunks (:class:`AppendAudio`) and is closed, which makes it ready to
transcribe. A recording made with another app is imported whole
(:class:`ImportRecording`). :class:`DeleteSession` removes a session
with everything derived from it. Each use case reads and writes the
session while holding its lock (:class:`~transcriber.application.ports.
SessionLocks`), so concurrent requests and the job never interleave.
"""

import hashlib
from typing import Any, Optional

from pydantic import ValidationError

from transcriber.application.common import require_session
from transcriber.application.ports import (
    PCM_SAMPLE_WIDTH,
    Clock,
    RecordedAudio,
    SessionLocks,
    SessionRepository,
)
from transcriber.domain.errors import InvalidInputError, StateError
from transcriber.domain.models import Session, SessionState
from transcriber.domain.sessions import (
    ChunkAction,
    accept_chunk,
    can_delete,
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

    def __init__(
        self, sessions: SessionRepository, clock: Clock, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._clock = clock
        self._locks = locks

    def __call__(
        self, session_id: str, language: str, title: Optional[str] = None
    ) -> Session:
        """Create session ``session_id`` in state ``recording``.

        ``started_at`` is the clock's current time.
        """
        session = _make_session(
            id=session_id, title=title, language=language,
            started_at=self._clock.now(),
        )
        with self._locks.hold(session.id):
            return _new_session(self._sessions, session)


class AppendAudio:
    """Receive one numbered chunk of a recording in progress."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(self, session_id: str, number: int, data: bytes) -> bool:
        """Append chunk ``number`` (from 0); tell whether it was new.

        Repeating the last accepted chunk with the same bytes is a
        no-op that returns ``False``; see
        :func:`~transcriber.domain.sessions.accept_chunk` for the
        errors. Only a session in ``recording`` accepts chunks, and a
        chunk holds whole :data:`PCM_SAMPLE_WIDTH`-byte samples (an odd
        byte would shift every later sample).
        """
        if not data:
            raise InvalidInputError("An audio chunk cannot be empty.")
        if len(data) % PCM_SAMPLE_WIDTH:
            raise InvalidInputError(
                f"An audio chunk holds whole {PCM_SAMPLE_WIDTH}-byte"
                f" samples; got {len(data)} bytes."
            )
        digest = hashlib.sha256(data).hexdigest()
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            action = accept_chunk(session, number, digest)
            if action is ChunkAction.DUPLICATE:
                return False
            self._sessions.append_chunk(session_id, number, data)
            self._sessions.save(with_chunk(session, digest))
            return True


class CloseSession:
    """End the recording; the session is then ready to transcribe."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(self, session_id: str) -> Session:
        """Move the session from ``recording`` to ``uploaded``.

        A session without any audio cannot be closed.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            if (
                session.state is SessionState.RECORDING
                and session.chunks == 0
            ):
                raise InvalidInputError(
                    f"Session {session_id} has no audio to close."
                )
            closed = move(session, SessionState.UPLOADED)
            self._sessions.save(closed)
            return closed


class ImportRecording:
    """Create a session from a recording made with another app."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(
        self,
        session_id: str,
        language: str,
        recording: RecordedAudio,
        title: Optional[str] = None,
    ) -> Session:
        """Store ``recording`` as session ``session_id``, ``uploaded``.

        Creating the session and storing its audio succeed or fail
        together: if the audio cannot be stored the new session is
        removed again, so the same id can be used for another try.
        """
        if not recording.data:
            raise InvalidInputError("The recording is empty.")
        if not recording.format.strip():
            raise InvalidInputError("The recording needs a format.")
        session = _make_session(
            id=session_id, title=title, language=language,
            state=SessionState.UPLOADED,
        )
        with self._locks.hold(session.id):
            _new_session(self._sessions, session)
            try:
                self._sessions.store_recording(session.id, recording)
            except BaseException:
                self._sessions.delete(session.id)
                raise
        return session


class DeleteSession:
    """Remove a session, its audio, its words and its transcript."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(self, session_id: str) -> None:
        """Delete the session in any state but ``running``.

        A queued session may be deleted: its job finds the session gone
        (or replaced, with another ``run_id``) and does nothing.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            if not can_delete(session):
                raise StateError(
                    f"Session {session_id} is being transcribed; delete"
                    " it when the job ends."
                )
            self._sessions.delete(session_id)
