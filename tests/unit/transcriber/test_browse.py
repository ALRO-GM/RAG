"""Browsing use cases and the language given when queueing."""

import pytest

from tests.unit.transcriber.fakes import (
    FakeDecoder,
    FakeVoiceprintRepository,
    meeting,
    voice,
)
from tests.unit.transcriber.test_transcribe import world
from transcriber.application.browse import (
    ListSessions,
    ListSpeakers,
    SessionAudio,
    ShowSession,
    ShowTranscript,
)
from transcriber.application.ports import RecordedAudio
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.models import SessionState

S = SessionState


class TestListSessions:
    def test_sorted_and_damaged_ones_counted(self) -> None:
        w = world()
        for session_id in ("b", "a", "c"):
            w.upload(session_id)
        w.sessions.damaged.add("c")
        listing = ListSessions(w.sessions)()
        assert [s.id for s in listing.sessions] == ["a", "b"]
        assert listing.unreadable == 1

    def test_empty(self) -> None:
        listing = ListSessions(world().sessions)()
        assert (listing.sessions, listing.unreadable) == ([], 0)


class TestListSpeakers:
    def test_names_come_from_the_consent(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        voiceprints.enrol("ana", voice(0), name="Ana")
        voiceprints.enrol("luis", voice(1))
        found = ListSpeakers(voiceprints)()
        assert [(s.id, s.name, s.model) for s in found] == [
            ("ana", "Ana", "fake-voice"), ("luis", None, "fake-voice"),
        ]
        assert found[0].enrolled_at.startswith("2026-11-14")

    def test_an_unreadable_consent_leaves_the_name_out(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        voiceprints.enrol("ana", voice(0), name="Ana")

        def damaged(speaker_id: str) -> None:
            raise StorageError("Damaged ana.json")

        voiceprints.consent = damaged  # type: ignore[method-assign]
        assert [s.name for s in ListSpeakers(voiceprints)()] == [None]


class TestShowSession:
    def test_session_and_status(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        view = ShowSession(w.sessions, w.status)("s1")
        assert view.session.state is S.QUEUED
        assert view.status.state is S.QUEUED
        w.runner.run_all()
        view = ShowSession(w.sessions, w.status)("s1")
        assert (view.status.state, view.status.fraction) == (S.DONE, 1.0)

    def test_unknown_or_invalid(self) -> None:
        w = world()
        with pytest.raises(NotFoundError):
            ShowSession(w.sessions, w.status)("nope")
        with pytest.raises(InvalidInputError):
            ShowSession(w.sessions, w.status)("../x")


class TestShowTranscript:
    def test_only_once_transcribed(self) -> None:
        w = world()
        w.upload()
        show = ShowTranscript(w.sessions)
        with pytest.raises(NotFoundError, match="no transcript"):
            show("s1")
        w.queue("s1")
        w.runner.run_all()
        assert show("s1") == w.sessions.transcript("s1")
        with pytest.raises(NotFoundError):
            show("nope")


class TestSessionAudio:
    def test_recording_and_decoding(self) -> None:
        w = world()
        w.upload()
        decoder = FakeDecoder(meeting())
        audio = SessionAudio(w.sessions, decoder)
        recording = audio.recording("s1")
        assert recording == RecordedAudio(b"audio", "wav")
        assert audio.decode(recording).sample_rate > 0
        assert decoder.decoded == [recording]
        with pytest.raises(NotFoundError):
            audio.recording("nope")


class TestQueueLanguage:
    def test_the_language_given_replaces_the_session_one(self) -> None:
        w = world()
        w.upload()
        queued = w.queue("s1", TranscriptionOptions(language="en"))
        assert queued.language == "en"
        w.runner.run_all()
        assert w.recogniser.languages == ["en"]
        transcript = w.sessions.transcript("s1")
        assert transcript is not None and transcript.session.language == "en"

    def test_without_one_the_session_keeps_its_own(self) -> None:
        w = world()
        w.upload()
        assert w.queue("s1").language == "es"

    def test_an_invalid_language_is_refused_before_queueing(self) -> None:
        w = world()
        w.upload()
        with pytest.raises(InvalidInputError, match="Invalid language"):
            w.queue("s1", TranscriptionOptions(language="e"))
        assert w.sessions.sessions["s1"].state is S.UPLOADED
        assert w.runner.pending == []
