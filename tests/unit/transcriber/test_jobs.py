"""Transcriber jobs: run ids, locks, interruptions, recovery, re-runs."""

from typing import Optional

import pytest

from tests.unit.transcriber.fakes import (
    FakeSessionLocks,
    FakeSessionRepository,
    FixedClock,
    RecordingProgress,
    meeting,
    voice,
)
from tests.unit.transcriber.test_transcribe import World, turns, world
from transcriber.application.ports import RecordedAudio
from transcriber.application.record import (
    AppendAudio,
    CloseSession,
    DeleteSession,
    ImportRecording,
    OpenSession,
)
from transcriber.application.recover import RecoverSessions
from transcriber.application.review import RenameSpeaker, ReassignUtterance
from transcriber.application.transcribe import (
    STOPPED_NOTE,
    TranscriptionOptions,
)
from transcriber.domain.errors import StateError
from transcriber.domain.models import Session, SessionState
from transcriber.domain.sessions import INTERRUPTED

S = SessionState


def session_in(w: World, state: SessionState, run_id: Optional[str] = None
               ) -> Session:
    """Store session ``s1`` of ``w`` in ``state`` (a test shortcut)."""
    stored = w.sessions.sessions["s1"].model_copy(
        update={"state": state, "run_id": run_id}
    )
    w.sessions.sessions["s1"] = stored
    return stored


class TestRunIds:
    def test_each_queueing_draws_a_new_run_id(self) -> None:
        w = world()
        w.upload()
        first = w.queue("s1")
        w.runner.run_all()
        second = w.queue("s1")
        assert first.run_id and second.run_id
        assert first.run_id != second.run_id
        assert w.sessions.sessions["s1"].run_id == second.run_id

    def test_a_stale_job_does_not_run_on_a_new_session(self) -> None:
        """The reviewer's scenario: the transcript is the second run's."""
        w = world()
        w.upload("s")
        w.queue("s", TranscriptionOptions(num_speakers=5))
        DeleteSession(w.sessions, w.locks)("s")
        w.upload("s")
        w.queue("s", TranscriptionOptions(num_speakers=2))
        assert len(w.runner.pending) == 2
        w.runner.run_all()
        assert w.runner.errors == {}
        transcript = w.sessions.transcript("s")
        assert transcript is not None
        assert [sp.id for sp in transcript.speakers] == ["S1", "S2"]
        assert turns(transcript) == [
            ("S1", None), ("S2", "S1"), ("S1", "S2"),
        ]
        # The stale job did nothing: one run, one running state saved.
        states = [st for sid, st in w.sessions.saved_states if sid == "s"]
        assert states.count(S.RUNNING) == 1

    def test_five_speakers_would_give_another_transcript(self) -> None:
        """Control for the scenario above: the stale options differ."""
        w = world()
        transcript = w.run(TranscriptionOptions(num_speakers=5))
        assert len(transcript.speakers) > 2

    def test_a_job_whose_run_id_is_gone_reports_and_stops(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        progress = RecordingProgress()
        assert w.transcribe("s1", progress=progress,
                            run_id="0" * 32) is None
        assert progress.notes == [STOPPED_NOTE]
        assert w.sessions.sessions["s1"].state is S.QUEUED
        assert w.transcribe("gone", run_id="0" * 32) is None


class TestTheJobChecksBeforeSaving:
    def interrupt_recognition(self, w: World, action: str) -> None:
        """While recognising, the session is deleted or queued again."""
        original = w.recogniser.recognise

        def meddle(audio, language=None):  # type: ignore[no-untyped-def]
            if action == "delete":
                w.sessions.delete("s1")
            else:
                w.sessions.sessions["s1"] = w.sessions.sessions[
                    "s1"
                ].model_copy(update={"run_id": "f" * 32})
            return original(audio, language)

        w.recogniser.recognise = meddle  # type: ignore[method-assign]

    def test_a_session_deleted_during_the_job_is_not_resurrected(
        self,
    ) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        self.interrupt_recognition(w, "delete")
        w.runner.run_all()
        assert w.runner.errors == {}
        assert w.sessions.get("s1") is None
        assert w.sessions.transcript("s1") is None
        assert w.sessions.words("s1") == []
        assert w.runner.statuses["s1"].message == STOPPED_NOTE

    def test_another_run_is_left_alone(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        self.interrupt_recognition(w, "requeue")
        w.runner.run_all()
        session = w.sessions.sessions["s1"]
        assert (session.state, session.run_id) == (S.RUNNING, "f" * 32)
        assert w.sessions.transcript("s1") is None

    def test_a_failure_of_another_run_is_not_saved(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        self.interrupt_recognition(w, "requeue")
        w.encoder.embed = None  # type: ignore[assignment]  # fails later
        w.runner.run_all()
        assert isinstance(w.runner.errors["s1"], TypeError)
        session = w.sessions.sessions["s1"]
        assert (session.state, session.failure) == (S.RUNNING, None)


class TestInterruptions:
    @pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
    def test_an_interruption_fails_the_session_and_propagates(
        self, error: type
    ) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        w.runner.pending.clear()
        w.recogniser.failures.append(error())
        with pytest.raises(error):
            w.transcribe("s1")
        session = w.sessions.sessions["s1"]
        assert (session.state, session.failure) == (
            S.FAILED, f"recognise: {error.__name__}",
        )
        w.queue("s1")  # and it can be queued again
        w.runner.run_all()
        assert w.sessions.sessions["s1"].state is S.DONE


class TestRecoverSessions:
    @pytest.mark.parametrize("state", [S.QUEUED, S.RUNNING])
    def test_interrupted_sessions_fail_and_can_be_retried_or_deleted(
        self, state: SessionState
    ) -> None:
        w = world()
        w.upload()
        session_in(w, state, "a" * 32)
        assert RecoverSessions(w.sessions, w.locks)() == ["s1"]
        session = w.sessions.sessions["s1"]
        assert (session.state, session.failure) == (S.FAILED, INTERRUPTED)
        w.queue("s1")
        w.runner.run_all()
        assert w.sessions.sessions["s1"].state is S.DONE
        DeleteSession(w.sessions, w.locks)("s1")
        assert w.sessions.get("s1") is None

    def test_other_sessions_are_left_alone(self) -> None:
        w = world()
        w.upload("a")
        w.upload("b")
        w.queue("b")
        w.runner.run_all()
        w.upload("c")
        w.queue("c")
        before = dict(w.sessions.sessions)
        assert RecoverSessions(w.sessions, w.locks)() == ["c"]
        assert w.sessions.sessions["a"] == before["a"]
        assert w.sessions.sessions["b"] == before["b"]
        assert w.sessions.sessions["c"].state is S.FAILED
        assert RecoverSessions(w.sessions, w.locks)() == []


class TestRequeue:
    def test_a_done_session_is_transcribed_again(self) -> None:
        w = world()
        first = w.run()
        assert [s.id for s in first.speakers] == ["S1", "S2"]
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        queued = w.queue("s1")
        assert queued.state is S.QUEUED
        # The old transcript stays until the new one is saved.
        assert w.sessions.transcript("s1") == first
        w.runner.run_all()
        second = w.sessions.transcript("s1")
        assert second is not None
        assert [s.id for s in second.speakers] == ["S1", "luis"]
        assert w.sessions.sessions["s1"].state is S.DONE

    def test_a_failed_re_run_keeps_the_previous_transcript(self) -> None:
        w = world()
        first = w.run()
        w.queue("s1")
        w.recogniser.failures.append(RuntimeError("boom"))
        w.runner.run_all()
        assert w.sessions.sessions["s1"].state is S.FAILED
        assert w.sessions.transcript("s1") == first


class TestDeleteSession:
    @pytest.mark.parametrize("state", [
        S.RECORDING, S.UPLOADED, S.QUEUED, S.DONE, S.FAILED,
    ])
    def test_every_state_but_running(self, state: SessionState) -> None:
        w = world()
        w.upload()
        session_in(w, state)
        DeleteSession(w.sessions, w.locks)("s1")
        assert w.sessions.get("s1") is None

    def test_not_while_running(self) -> None:
        w = world()
        w.upload()
        session_in(w, S.RUNNING)
        with pytest.raises(StateError, match="being transcribed"):
            DeleteSession(w.sessions, w.locks)("s1")
        assert w.sessions.get("s1") is not None


class TestLocks:
    def test_every_write_holds_the_session_lock(self) -> None:
        """A whole life of two sessions: no write outside the lock."""
        w = world(meeting())
        locks: FakeSessionLocks = w.locks
        sessions: FakeSessionRepository = w.sessions
        OpenSession(sessions, FixedClock(), locks)("rec", "es")
        AppendAudio(sessions, locks)("rec", 0, b"\x01\x00")
        AppendAudio(sessions, locks)("rec", 0, b"\x01\x00")
        CloseSession(sessions, locks)("rec")
        ImportRecording(sessions, locks)(
            "s1", "es", RecordedAudio(b"audio", "wav")
        )
        w.queue("s1")
        w.runner.run_all()
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        RenameSpeaker(sessions, w.voiceprints, locks)("s1", "S2",
                                                      name="Luis")
        transcript = sessions.transcript("s1")
        assert transcript is not None
        ReassignUtterance(sessions, w.voiceprints, locks)(
            "s1", transcript.utterances[0].id, "luis"
        )
        w.queue("rec")
        RecoverSessions(sessions, locks)()
        DeleteSession(sessions, locks)("rec")
        DeleteSession(sessions, locks)("s1")
        assert sessions.unlocked_writes == []
        assert locks.held == {}
        assert {"rec", "s1"} <= set(locks.taken)

    def test_the_fake_reports_a_write_outside_the_lock(self) -> None:
        """Control: the check above would see a missing lock."""
        w = world()
        w.upload()
        w.sessions.delete("s1")
        assert w.sessions.unlocked_writes == [("delete", "s1")]
