"""Browsing: list sessions and enrolled people, show one, play its audio.

Read-only use cases for the pages. A listing never fails because one
stored record cannot be read: that record is left out and counted, so
one damaged file does not hide every other session.
"""

from typing import List, NamedTuple, Optional

from transcriber.application.common import (
    require_session,
    require_transcript,
)
from transcriber.application.ports import (
    JobStatus,
    PlaybackAudio,
    SessionRepository,
    VoiceprintRepository,
)
from transcriber.application.transcribe import TranscriptionStatus
from transcriber.domain.errors import StorageError
from transcriber.domain.models import Session, Transcript


class SessionList(NamedTuple):
    """Readable sessions, by id, and how many could not be read."""

    sessions: List[Session]
    unreadable: int


class EnrolledSpeaker(NamedTuple):
    """An enrolled person as the pages show them (no vector)."""

    id: str
    name: Optional[str]
    model: str
    enrolled_at: str


class ListSessions:
    """Every stored session that can be read."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self) -> SessionList:
        """Return the readable sessions sorted by id.

        A session whose record cannot be read (:class:`StorageError`)
        or that disappears while listing is left out; the former are
        counted in ``unreadable``.
        """
        found: List[Session] = []
        unreadable = 0
        for session_id in sorted(self._sessions.ids()):
            try:
                session = self._sessions.get(session_id)
            except StorageError:
                unreadable += 1
                continue
            if session is not None:
                found.append(session)
        return SessionList(found, unreadable)


class ListSpeakers:
    """The people enrolled with the voice model in use."""

    def __init__(self, voiceprints: VoiceprintRepository) -> None:
        """Wire the port."""
        self._voiceprints = voiceprints

    def __call__(self) -> List[EnrolledSpeaker]:
        """Return each enrolled person with the name they gave."""
        speakers: List[EnrolledSpeaker] = []
        for voiceprint in self._voiceprints.all():
            try:
                consent = self._voiceprints.consent(voiceprint.speaker_id)
            except StorageError:
                consent = None
            speakers.append(EnrolledSpeaker(
                id=voiceprint.speaker_id,
                name=consent.name if consent is not None else None,
                model=voiceprint.model,
                enrolled_at=voiceprint.enrolled_at.isoformat(),
            ))
        return speakers


class SessionView(NamedTuple):
    """A session and the status of its job."""

    session: Session
    status: JobStatus


class ShowSession:
    """One session with where its job is."""

    def __init__(
        self, sessions: SessionRepository, status: TranscriptionStatus
    ) -> None:
        """Wire the port and the status use case."""
        self._sessions = sessions
        self._status = status

    def __call__(self, session_id: str) -> SessionView:
        """Return the session and its status; NotFound if it is unknown."""
        session = require_session(self._sessions, session_id)
        return SessionView(session, self._status(session_id))


class ShowTranscript:
    """The transcript of a session, words included (for the player)."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self, session_id: str) -> Transcript:
        """Return the transcript; NotFound for no session or transcript."""
        require_session(self._sessions, session_id)
        return require_transcript(self._sessions, session_id)


class SessionAudio:
    """The audio of a session as WAV, for the player."""

    def __init__(self, sessions: SessionRepository) -> None:
        """Wire the port."""
        self._sessions = sessions

    def __call__(self, session_id: str) -> PlaybackAudio:
        """Return the session's audio rendered for the player.

        The caller closes it. :class:`NotFoundError` for an unknown
        session or one without audio.
        """
        require_session(self._sessions, session_id)
        return self._sessions.playback(session_id)
