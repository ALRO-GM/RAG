"""The conversation type and goal of a session (#65).

Domain (:func:`describe`, :func:`carry_description`), the use cases
that set them (open, import, :class:`DescribeSession`), the job that
must not lose an edit made while it ran, and the publish that carries
them in ``transcript.v1``.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import pytest

from tests.unit.transcriber.fakes import (
    FakeSessionLocks,
    FakeSessionRepository,
    FixedClock,
    RecordingPublisher,
)
from tests.unit.transcriber.test_transcribe import world
from transcriber.application.ports import RecordedAudio
from transcriber.application.publish import PublishTranscript
from transcriber.application.record import (
    DescribeSession,
    ImportRecording,
    OpenSession,
)
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.models import (
    GOAL_MAX_LENGTH,
    SESSION_TYPES,
    Session,
    SessionState,
)
from transcriber.domain.sessions import carry_description, describe

SESSION = Session(id="s1", language="es")


class TestDescribe:
    def test_the_types_are_the_eleven_of_the_study(self) -> None:
        schema = json.loads(
            (Path(__file__).parents[3] / "contracts"
             / "transcript.v1.schema.json").read_text(encoding="utf-8")
        )
        enum = schema["properties"]["session"]["properties"]["type"]["enum"]
        assert list(SESSION_TYPES) == [v for v in enum if v is not None]
        assert len(SESSION_TYPES) == 11

    def test_unset_by_default(self) -> None:
        assert SESSION.type is None and SESSION.goal is None

    def test_sets_type_and_trimmed_goal(self) -> None:
        out = describe(SESSION, {"type": "negociacion", "goal": "  Precio  "})
        assert (out.type, out.goal) == ("negociacion", "Precio")

    def test_a_field_left_out_keeps_its_value(self) -> None:
        first = describe(SESSION, {"type": "venta", "goal": "Cerrar"})
        assert describe(first, {"goal": "Otra"}).type == "venta"
        assert describe(first, {"type": "coaching"}).goal == "Cerrar"

    def test_none_or_blank_clears(self) -> None:
        first = describe(SESSION, {"type": "venta", "goal": "Cerrar"})
        out = describe(first, {"type": None, "goal": "   "})
        assert (out.type, out.goal) == (None, None)

    @pytest.mark.parametrize("changes, message", [
        ({"type": "meeting"}, "Unknown conversation type"),
        ({"goal": "x" * (GOAL_MAX_LENGTH + 1)}, "at most"),
        ({"title": "x"}, "Unknown session field"),
        ({"goal": "a\nb"}, "control characters"),
        ({"goal": "a\x1b[31mb"}, "control characters"),
        ({"goal": "a\x7fb"}, "control characters"),
    ])
    def test_invalid_changes(
        self, changes: dict[str, Optional[str]], message: str
    ) -> None:
        with pytest.raises(InvalidInputError, match=message):
            describe(SESSION, changes)

    def test_a_goal_of_the_longest_length_is_kept(self) -> None:
        goal = "x" * GOAL_MAX_LENGTH
        assert describe(SESSION, {"goal": goal}).goal == goal

    def test_carry_takes_both_fields_and_nothing_else(self) -> None:
        source = describe(
            SESSION.model_copy(update={"title": "Otra"}),
            {"type": "feedback", "goal": "Revisar"},
        )
        out = carry_description(SESSION, source)
        assert (out.type, out.goal, out.title) == ("feedback", "Revisar", None)


def store() -> FakeSessionRepository:
    """An empty store whose writes are checked against its locks."""
    return FakeSessionRepository(locks=FakeSessionLocks())


def locks(sessions: FakeSessionRepository) -> FakeSessionLocks:
    """The locks ``sessions`` checks its writes against."""
    assert sessions.locks is not None
    return sessions.locks


class TestStartingWithAType:
    def test_open_keeps_type_and_goal(self) -> None:
        sessions = store()
        session = OpenSession(sessions, FixedClock(), locks(sessions))(
            "s1", "es", kind="lluvia_ideas", goal=" Ideas ",
        )
        assert (session.type, session.goal) == ("lluvia_ideas", "Ideas")
        assert sessions.get("s1") == session

    def test_open_without_them_leaves_them_unset(self) -> None:
        sessions = store()
        session = OpenSession(sessions, FixedClock(), locks(sessions))(
            "s1", "es"
        )
        assert (session.type, session.goal) == (None, None)

    def test_open_refuses_an_unknown_type_and_stores_nothing(self) -> None:
        sessions = store()
        with pytest.raises(InvalidInputError, match="conversation type"):
            OpenSession(sessions, FixedClock(), locks(sessions))(
                "s1", "es", kind="party"
            )
        assert sessions.get("s1") is None

    def test_import_keeps_type_and_goal(self) -> None:
        sessions = store()
        session = ImportRecording(sessions, locks(sessions))(
            "s1", "es", RecordedAudio(b"audio", "wav"),
            kind="entrevista", goal="Puesto de soporte",
        )
        assert (session.type, session.goal) == (
            "entrevista", "Puesto de soporte"
        )


class TestDescribeSession:
    def test_changes_a_recording_session(self) -> None:
        sessions = store()
        OpenSession(sessions, FixedClock(), locks(sessions))("s1", "es")
        out = DescribeSession(sessions, locks(sessions))(
            "s1", {"type": "conflicto"}
        )
        assert out.type == "conflicto"
        assert out.state is SessionState.RECORDING
        assert sessions.get("s1") == out

    def test_unknown_session(self) -> None:
        sessions = store()
        with pytest.raises(NotFoundError):
            DescribeSession(sessions, locks(sessions))("nope", {"type": None})

    def test_a_done_session_and_its_transcript_change_together(self) -> None:
        w = world()
        w.run()
        out = DescribeSession(w.sessions, w.locks)(
            "s1", {"type": "reunion_decision", "goal": "Reparto"}
        )
        transcript = w.sessions.transcript("s1")
        assert transcript is not None
        assert transcript.session == out
        assert out.state is SessionState.DONE
        document = PublishTranscript(
            w.sessions, RecordingPublisher(), w.locks
        )("s1")
        assert document["session"]["type"] == "reunion_decision"
        assert document["session"]["goal"] == "Reparto"

    def test_a_publish_without_them_leaves_them_out(self) -> None:
        w = world()
        w.run()
        document = PublishTranscript(
            w.sessions, RecordingPublisher(), w.locks
        )("s1")
        assert "type" not in document["session"]
        assert "goal" not in document["session"]


class EditWhileRunning:
    """A job's progress that edits the session at one stage.

    With ``fail`` it raises right after the edit, so the job fails.
    """

    def __init__(self, edit: Callable[[], object], fail: bool = False) -> None:
        """Keep the edit to make and whether to fail after it."""
        self.edit = edit
        self.fail = fail
        self.done = False

    def report(self, stage: str, fraction: float) -> None:
        """Edit on the first stage after the session is running."""
        if not self.done:
            self.done = True
            self.edit()
            if self.fail:
                raise RuntimeError("stopped")

    def note(self, message: str) -> None:
        """Ignore notes."""


class TestAJobKeepsAnEditMadeWhileItRan:
    def test_the_saved_transcript_and_session_keep_the_edit(self) -> None:
        w = world()
        w.upload()
        queued = w.queue("s1")
        describe_session = DescribeSession(w.sessions, w.locks)
        progress = EditWhileRunning(
            lambda: describe_session("s1", {"type": "coaching"})
        )
        transcript = w.transcribe("s1", progress=progress,
                                  run_id=queued.run_id)
        assert transcript is not None
        assert transcript.session.type == "coaching"
        stored = w.sessions.transcript("s1")
        assert stored is not None and stored.session.type == "coaching"
        session = w.sessions.get("s1")
        assert session is not None
        assert session.type == "coaching"
        assert session.state is SessionState.DONE

    def test_a_failed_job_keeps_the_edit(self) -> None:
        w = world()
        w.upload()
        queued = w.queue("s1")
        describe_session = DescribeSession(w.sessions, w.locks)
        progress = EditWhileRunning(
            lambda: describe_session("s1", {"goal": "Algo"}), fail=True
        )
        with pytest.raises(RuntimeError):
            w.transcribe("s1", progress=progress, run_id=queued.run_id)
        session = w.sessions.get("s1")
        assert session is not None
        assert session.state is SessionState.FAILED
        assert session.goal == "Algo"
