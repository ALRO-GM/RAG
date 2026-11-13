"""Transcriber recording use cases: open, append, close, import, delete."""

import pytest

from tests.unit.transcriber.fakes import (
    NOW,
    FakeSessionRepository,
    FixedClock,
)
from transcriber.application.ports import PCM_FORMAT, RecordedAudio
from transcriber.application.record import (
    AppendAudio,
    CloseSession,
    DeleteSession,
    ImportRecording,
    OpenSession,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StateError,
)
from transcriber.domain.models import SessionState


@pytest.fixture
def sessions() -> FakeSessionRepository:
    """An empty session store."""
    return FakeSessionRepository()


def opened(sessions: FakeSessionRepository) -> str:
    """Open session ``s1`` and return its id."""
    OpenSession(sessions, FixedClock())("s1", "es", title="Equipo A")
    return "s1"


class TestOpenSession:
    def test_opens_in_recording_with_the_clock_time(
        self, sessions: FakeSessionRepository
    ) -> None:
        session = OpenSession(sessions, FixedClock())("s1", "es", "Equipo")
        assert session.state is SessionState.RECORDING
        assert session.started_at == NOW
        assert sessions.get("s1") == session

    def test_an_id_is_used_once(self, sessions: FakeSessionRepository) -> None:
        opened(sessions)
        with pytest.raises(InvalidInputError, match="already exists"):
            OpenSession(sessions, FixedClock())("s1", "es")

    def test_invalid_id_is_input_error(
        self, sessions: FakeSessionRepository
    ) -> None:
        with pytest.raises(InvalidInputError, match="id"):
            OpenSession(sessions, FixedClock())("../etc", "es")


class TestAppendAudio:
    def test_chunks_are_numbered_from_zero(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        assert append("s1", 0, b"aa") is True
        assert append("s1", 1, b"bb") is True
        assert sessions.chunks["s1"] == [b"aa", b"bb"]
        assert sessions.sessions["s1"].chunks == 2

    def test_repeating_the_last_chunk_is_a_no_op(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        append("s1", 0, b"aa")
        append("s1", 1, b"bb")
        assert append("s1", 1, b"bb") is False
        assert sessions.chunks["s1"] == [b"aa", b"bb"]
        assert sessions.sessions["s1"].chunks == 2

    def test_other_bytes_for_an_accepted_number_is_an_error(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        append("s1", 0, b"aa")
        append("s1", 1, b"bb")
        with pytest.raises(InvalidInputError, match="other content"):
            append("s1", 1, b"cc")
        with pytest.raises(InvalidInputError, match="already accepted"):
            append("s1", 0, b"aa")
        assert sessions.chunks["s1"] == [b"aa", b"bb"]

    def test_a_gap_is_an_error(self, sessions: FakeSessionRepository) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        with pytest.raises(InvalidInputError, match="gap"):
            append("s1", 1, b"bb")
        assert "s1" not in sessions.chunks

    def test_only_while_recording(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        append("s1", 0, b"aa")
        CloseSession(sessions)("s1")
        with pytest.raises(StateError, match="recording"):
            append("s1", 1, b"bb")

    def test_empty_chunk_and_unknown_session(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions)
        opened(sessions)
        with pytest.raises(InvalidInputError, match="empty"):
            append("s1", 0, b"")
        with pytest.raises(NotFoundError):
            append("nope", 0, b"aa")


class TestCloseSession:
    def test_close_moves_to_uploaded(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        AppendAudio(sessions)("s1", 0, b"aa")
        closed = CloseSession(sessions)("s1")
        assert closed.state is SessionState.UPLOADED
        assert sessions.recording("s1") == RecordedAudio(b"aa", PCM_FORMAT)

    def test_nothing_recorded_cannot_close(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        with pytest.raises(InvalidInputError, match="no audio"):
            CloseSession(sessions)("s1")

    def test_closing_twice_is_a_state_error(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        AppendAudio(sessions)("s1", 0, b"aa")
        CloseSession(sessions)("s1")
        with pytest.raises(StateError):
            CloseSession(sessions)("s1")


class TestImportRecording:
    def test_import_is_uploaded_at_once(
        self, sessions: FakeSessionRepository
    ) -> None:
        recording = RecordedAudio(b"RIFF....", "wav")
        session = ImportRecording(sessions)("s2", "en", recording, "Demo")
        assert session.state is SessionState.UPLOADED
        assert session.title == "Demo"
        assert sessions.recording("s2") == recording

    def test_invalid_imports(self, sessions: FakeSessionRepository) -> None:
        importer = ImportRecording(sessions)
        with pytest.raises(InvalidInputError, match="empty"):
            importer("s2", "en", RecordedAudio(b"", "wav"))
        with pytest.raises(InvalidInputError, match="format"):
            importer("s2", "en", RecordedAudio(b"x", " "))
        opened(sessions)
        with pytest.raises(InvalidInputError, match="exists"):
            importer("s1", "en", RecordedAudio(b"x", "wav"))
        assert "s1" not in sessions.recordings


class TestDeleteSession:
    def test_delete_removes_everything(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        AppendAudio(sessions)("s1", 0, b"aa")
        DeleteSession(sessions)("s1")
        assert sessions.get("s1") is None
        assert "s1" not in sessions.chunks
        with pytest.raises(NotFoundError):
            DeleteSession(sessions)("s1")

    def test_not_while_running(self, sessions: FakeSessionRepository) -> None:
        opened(sessions)
        sessions.save(sessions.sessions["s1"].model_copy(
            update={"state": SessionState.RUNNING}
        ))
        with pytest.raises(StateError, match="being transcribed"):
            DeleteSession(sessions)("s1")
