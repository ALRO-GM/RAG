"""Transcriber recording use cases: open, append, close, import, delete."""

from typing import Any, Dict

import pytest

from tests.unit.transcriber.fakes import (
    NOW,
    FakeSessionLocks,
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
    """An empty session store whose writes are checked for the lock."""
    return FakeSessionRepository(locks=FakeSessionLocks())


def locks(sessions: FakeSessionRepository) -> FakeSessionLocks:
    """The locks the store checks its writes against."""
    assert sessions.locks is not None
    return sessions.locks


def opener(sessions: FakeSessionRepository) -> OpenSession:
    """The open use case over ``sessions`` with a stopped clock."""
    return OpenSession(sessions, FixedClock(), locks(sessions))


def opened(sessions: FakeSessionRepository) -> str:
    """Open session ``s1`` and return its id."""
    opener(sessions)("s1", "es", title="Equipo A")
    return "s1"


class TestOpenSession:
    def test_opens_in_recording_with_the_clock_time(
        self, sessions: FakeSessionRepository
    ) -> None:
        session = opener(sessions)("s1", "es", "Equipo")
        assert session.state is SessionState.RECORDING
        assert session.started_at == NOW
        assert sessions.get("s1") == session

    def test_an_id_is_used_once(self, sessions: FakeSessionRepository) -> None:
        opened(sessions)
        with pytest.raises(InvalidInputError, match="already exists"):
            opener(sessions)("s1", "es")

    def test_invalid_id_is_input_error(
        self, sessions: FakeSessionRepository
    ) -> None:
        with pytest.raises(InvalidInputError, match="id"):
            opener(sessions)("../etc", "es")


class TestAppendAudio:
    def test_chunks_are_numbered_from_zero(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        assert append("s1", 0, b"aa") is True
        assert append("s1", 1, b"bb") is True
        assert sessions.chunks["s1"] == [b"aa", b"bb"]
        assert sessions.sessions["s1"].chunks == 2

    def test_repeating_the_last_chunk_is_a_no_op(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        append("s1", 0, b"aa")
        append("s1", 1, b"bb")
        assert append("s1", 1, b"bb") is False
        assert sessions.chunks["s1"] == [b"aa", b"bb"]
        assert sessions.sessions["s1"].chunks == 2

    def test_other_bytes_for_an_accepted_number_is_an_error(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        append("s1", 0, b"aa")
        append("s1", 1, b"bb")
        with pytest.raises(InvalidInputError, match="other content"):
            append("s1", 1, b"cc")
        with pytest.raises(InvalidInputError, match="already accepted"):
            append("s1", 0, b"aa")
        assert sessions.chunks["s1"] == [b"aa", b"bb"]

    @pytest.mark.parametrize("data", [b"a", b"abc", b"\x00" * 31999])
    def test_a_chunk_of_half_a_sample_is_refused(
        self, sessions: FakeSessionRepository, data: bytes
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        with pytest.raises(InvalidInputError, match="whole 2-byte"):
            append("s1", 0, data)
        assert "s1" not in sessions.chunks
        assert sessions.sessions["s1"].chunks == 0

    def test_storing_a_chunk_number_again_keeps_it_once(
        self, sessions: FakeSessionRepository
    ) -> None:
        """The port's contract, as the fake keeps it (see the adapter's
        integration test for the file system)."""
        opened(sessions)
        sessions.append_chunk("s1", 0, b"aa")
        sessions.append_chunk("s1", 1, b"bb")
        sessions.append_chunk("s1", 1, b"dd")
        assert sessions.chunks["s1"] == [b"aa", b"dd"]
        sessions.append_chunk("s1", 2, b"cc")
        with pytest.raises(InvalidInputError, match="drop the chunks"):
            sessions.append_chunk("s1", 1, b"ee")
        assert sessions.chunks["s1"] == [b"aa", b"dd", b"cc"]
        with pytest.raises(InvalidInputError, match="gap"):
            sessions.append_chunk("s1", 4, b"ee")

    def test_a_gap_is_an_error(self, sessions: FakeSessionRepository) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        with pytest.raises(InvalidInputError, match="gap"):
            append("s1", 1, b"bb")
        assert "s1" not in sessions.chunks

    def test_only_while_recording(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
        opened(sessions)
        append("s1", 0, b"aa")
        CloseSession(sessions, locks(sessions))("s1")
        with pytest.raises(StateError, match="recording"):
            append("s1", 1, b"bb")

    def test_empty_chunk_and_unknown_session(
        self, sessions: FakeSessionRepository
    ) -> None:
        append = AppendAudio(sessions, locks(sessions))
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
        AppendAudio(sessions, locks(sessions))("s1", 0, b"aa")
        closed = CloseSession(sessions, locks(sessions))("s1")
        assert closed.state is SessionState.UPLOADED
        assert sessions.recording("s1") == RecordedAudio(b"aa", PCM_FORMAT)

    def test_nothing_recorded_cannot_close(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        with pytest.raises(InvalidInputError, match="no audio"):
            CloseSession(sessions, locks(sessions))("s1")

    def test_closing_again_returns_the_session_as_it_is(
        self, sessions: FakeSessionRepository
    ) -> None:
        # The page retries a close whose answer was lost.
        opened(sessions)
        AppendAudio(sessions, locks(sessions))("s1", 0, b"aa")
        first = CloseSession(sessions, locks(sessions))("s1")
        again = CloseSession(sessions, locks(sessions))("s1")
        assert again == first
        assert again.state is SessionState.UPLOADED

    @pytest.mark.parametrize("state", [
        SessionState.QUEUED, SessionState.RUNNING, SessionState.DONE,
        SessionState.FAILED,
    ])
    def test_a_session_further_on_is_left_as_it_is(
        self, sessions: FakeSessionRepository, state: SessionState
    ) -> None:
        opened(sessions)
        AppendAudio(sessions, locks(sessions))("s1", 0, b"aa")
        closed = CloseSession(sessions, locks(sessions))("s1")
        later = closed.model_copy(update={
            "state": state,
            "failure": "interrupted" if state is SessionState.FAILED
            else None,
        })
        sessions.save(later)
        assert CloseSession(sessions, locks(sessions))("s1") == later
        assert sessions.get("s1") == later


class TestStartedWith:
    def test_consent_and_options_are_kept_on_the_session(
        self, sessions: FakeSessionRepository
    ) -> None:
        session = opener(sessions)(
            "s1", "es", "Reparto", consent_version="consent-1",
            num_speakers=3, expected_speakers=["ana", "luis"],
        )
        assert session.consent_version == "consent-1"
        assert session.num_speakers == 3
        assert session.expected_speakers == ("ana", "luis")
        assert sessions.sessions["s1"] == session

    def test_without_them_nothing_is_kept(
        self, sessions: FakeSessionRepository
    ) -> None:
        session = opener(sessions)("s1", "es")
        assert (session.consent_version, session.num_speakers,
                session.expected_speakers) == (None, None, None)

    @pytest.mark.parametrize("fields, message", [
        ({"consent_version": "consent-99"}, "consent version"),
        ({"expected_speakers": ["../x"]}, "speaker id"),
        ({"num_speakers": 0}, "num_speakers"),
    ])
    def test_invalid_choices_are_refused(
        self, sessions: FakeSessionRepository, fields: Dict[str, Any],
        message: str,
    ) -> None:
        with pytest.raises(InvalidInputError, match=message):
            opener(sessions)("s1", "es", **fields)
        assert sessions.get("s1") is None

    def test_an_import_keeps_the_consent_shown(
        self, sessions: FakeSessionRepository
    ) -> None:
        importer = ImportRecording(sessions, locks(sessions))
        made = importer("s2", "es", RecordedAudio(b"x", "wav"), None,
                        "consent-1")
        assert made.consent_version == "consent-1"
        with pytest.raises(InvalidInputError, match="consent version"):
            importer("s3", "es", RecordedAudio(b"x", "wav"), None, "v0")
        assert sessions.get("s3") is None


class TestImportRecording:
    def test_import_is_uploaded_at_once(
        self, sessions: FakeSessionRepository
    ) -> None:
        recording = RecordedAudio(b"RIFF....", "wav")
        importer = ImportRecording(sessions, locks(sessions))
        session = importer("s2", "en", recording, "Demo")
        assert session.state is SessionState.UPLOADED
        assert session.title == "Demo"
        assert sessions.recording("s2") == recording

    def test_invalid_imports(self, sessions: FakeSessionRepository) -> None:
        importer = ImportRecording(sessions, locks(sessions))
        with pytest.raises(InvalidInputError, match="empty"):
            importer("s2", "en", RecordedAudio(b"", "wav"))
        with pytest.raises(InvalidInputError, match="format"):
            importer("s2", "en", RecordedAudio(b"x", " "))
        opened(sessions)
        with pytest.raises(InvalidInputError, match="exists"):
            importer("s1", "en", RecordedAudio(b"x", "wav"))
        assert "s1" not in sessions.recordings

    def test_a_failed_store_leaves_no_session(
        self, sessions: FakeSessionRepository
    ) -> None:
        importer = ImportRecording(sessions, locks(sessions))
        sessions.store_failures.append(
            InvalidInputError("Unsupported recording format")
        )
        with pytest.raises(InvalidInputError, match="Unsupported"):
            importer("s2", "en", RecordedAudio(b"x", "wav\n"))
        assert sessions.get("s2") is None
        assert sessions.unlocked_writes == []
        again = importer("s2", "en", RecordedAudio(b"x", "wav"))
        assert again.state is SessionState.UPLOADED


class TestDeleteSession:
    def test_delete_removes_everything(
        self, sessions: FakeSessionRepository
    ) -> None:
        opened(sessions)
        AppendAudio(sessions, locks(sessions))("s1", 0, b"aa")
        DeleteSession(sessions, locks(sessions))("s1")
        assert sessions.get("s1") is None
        assert "s1" not in sessions.chunks
        with pytest.raises(NotFoundError):
            DeleteSession(sessions, locks(sessions))("s1")

    def test_not_while_running(self, sessions: FakeSessionRepository) -> None:
        opened(sessions)
        sessions.save(sessions.sessions["s1"].model_copy(
            update={"state": SessionState.RUNNING}
        ))
        with pytest.raises(StateError, match="being transcribed"):
            DeleteSession(sessions, locks(sessions))("s1")
