"""The lab's progress overview and audio excerpts, with fakes."""

from datetime import datetime, timedelta, timezone
from typing import Any, List

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    FakeJobRunner,
    FakeSessionRepository,
)
from tests.unit.transcriber.lab_fakes import FakeLabRepository, FakeProcesses
from transcriber.application.lab import (
    MAX_EXCERPT_SECONDS,
    LabAudio,
    LabRunStatus,
)
from transcriber.application.lab_progress import LabOverview, ReferenceTurns
from transcriber.application.ports import Audio
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.lab import LabRequest, LabRun, Segment, Step
from transcriber.domain.lab_progress import (
    UNKNOWN_ERROR,
    progress_state,
    tally_steps,
)
from transcriber.domain.lab_reference import CORPORA, make_reference
from transcriber.domain.models import Session, SessionState
from transcriber.domain.speech_check import NO_SPEECH, SpeechCheck

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
RUN_A = "a" * 32
RUN_B = "b" * 32
RUN_C = "c" * 32


def step(model: str, state: str, code: str = "", denoiser: str = "none",
         stage: str = "diarize") -> Step:
    """A step of ``model`` in ``state`` (failed with ``code`` if given)."""
    error = {"code": code, "message": f"{code} happened"} if code else None
    return Step.model_validate({
        "stage": stage, "model": model, "denoiser": denoiser,
        "state": state, "error": error,
    })


STEPS = [
    step("dn", "done", stage="denoise", denoiser="none"),
    step("a", "done"),
    step("b", "running"),
    step("c", "pending"),
    step("d", "failed", "out_of_memory"),
    step("e", "failed", "out_of_memory", denoiser="dn"),
    step("f", "failed", "timeout"),
    step("g", "cancelled"),
]


class TestTally:
    def test_an_active_run(self) -> None:
        tally = tally_steps(STEPS, active=True)
        assert (tally.done, tally.running, tally.pending, tally.failed) \
            == (2, 1, 2, 3)
        assert tally.total == len(STEPS)
        assert tally.failed_codes == {"out_of_memory": 2, "timeout": 1}
        assert [s.model for s in tally.open] == ["b", "c", "d", "e", "f", "g"]
        failed = tally.open[2]
        assert (failed.name, failed.code, failed.message) == (
            "diarize:none+d", "out_of_memory", "out_of_memory happened",
        )
        assert tally.open[0].code is None

    def test_a_stopped_run_runs_nothing(self) -> None:
        tally = tally_steps(STEPS, active=False)
        assert (tally.running, tally.pending) == (0, 3)
        assert progress_state(STEPS[2], False) == "pending"
        assert progress_state(STEPS[7], True) == "pending"

    def test_a_failure_without_an_error(self) -> None:
        tally = tally_steps([step("x", "failed")], active=False)
        assert tally.failed_codes == {UNKNOWN_ERROR: 1}

    def test_nothing(self) -> None:
        tally = tally_steps([], active=False)
        assert tally.total == 0 and tally.open == []


def run(run_id: str, session: str, state: str, minutes: int,
        steps: List[Step], pid: Any = None) -> LabRun:
    """A stored run of ``session`` made ``minutes`` after T0."""
    return LabRun.model_validate({
        "id": run_id, "session_id": session,
        "created_at": T0 + timedelta(minutes=minutes),
        "request": LabRequest(diarizers=["a"]), "state": state,
        "steps": steps, "worker_pid": pid,
    })


class Fixture:
    """The overview's ports, faked."""

    def __init__(self) -> None:
        """Three sessions: one with runs, one imported, one untouched."""
        self.sessions = FakeSessionRepository()
        self.repository = FakeLabRepository()
        self.runner = FakeJobRunner()
        self.processes = FakeProcesses()
        for ident, title in (("own", "Mine"), ("ami-X", "ami X"),
                             ("quiet", None)):
            self.sessions.create(Session(
                id=ident, title=title, language="es",
                state=SessionState.DONE, duration=30.0,
            ))
        self.status = LabRunStatus(
            self.sessions, self.repository, self.runner, self.processes
        )
        self.overview = LabOverview(
            self.sessions, self.repository, self.status
        )
        self.audio = LabAudio(self.status, self.repository)


@pytest.fixture()
def lab() -> Fixture:
    """The overview over fakes."""
    return Fixture()


class TestOverview:
    def test_sessions_with_runs_or_a_reference(self, lab: Fixture) -> None:
        lab.repository.save_run(run(RUN_A, "own", "partial", 0, STEPS[:2]))
        lab.repository.save_run(run(RUN_B, "own", "failed", 5, STEPS))
        reference = make_reference(
            CORPORA["callhome"], "data", "0001", 12.5,
            [Segment(start=0, end=5, speaker="A")], None,
        )
        lab.repository.save_reference("ami-X", reference)
        found = {entry.session_id: entry for entry in lab.overview()}
        assert sorted(found) == ["ami-X", "own"]
        own = found["own"]
        assert (own.title, own.corpus, own.run_id, own.run_state) == (
            "Mine", None, RUN_B, "failed",
        )
        assert own.results_run == RUN_A
        assert own.steps is not None and own.steps.pending == 3
        imported = found["ami-X"]
        assert (imported.corpus, imported.licence_commercial) == (
            "callhome", False,
        )
        assert imported.duration == 12.5
        assert imported.run_id is None and imported.steps is None
        assert imported.results_run is None
        assert imported.excluded is None and own.excluded is None

    def test_a_recording_without_speech_says_so(self, lab: Fixture) -> None:
        noise = SpeechCheck(autocorrelation=0.68, flatness=0.39, frames=9,
                            speech=False)
        reference = make_reference(
            CORPORA["callhome"], "data", "0001", 12.5,
            [Segment(start=0, end=5, speaker="A")], None, speech=noise,
        )
        lab.repository.save_reference("ami-X", reference)
        (entry,) = lab.overview()
        assert entry.excluded == NO_SPEECH

    def test_an_interrupted_run_is_not_running(self, lab: Fixture) -> None:
        lab.repository.save_run(
            run(RUN_C, "own", "running", 0, STEPS, pid=999)
        )
        entry = lab.overview()[0]
        assert entry.run_state == "interrupted"
        assert entry.steps is not None and entry.steps.running == 0

    def test_a_running_run(self, lab: Fixture) -> None:
        lab.processes.others.add(7)
        lab.repository.save_run(run(RUN_C, "own", "running", 0, STEPS, pid=7))
        entry = lab.overview()[0]
        assert entry.run_state == "running"
        assert entry.steps is not None and entry.steps.running == 1

    def test_a_vanished_session_is_left_out(self, lab: Fixture) -> None:
        lab.repository.save_run(run(RUN_A, "own", "done", 0, STEPS[:2]))
        lab.sessions.delete("own")
        assert lab.overview() == []


class TestExcerpt:
    def setup_audio(self, lab: Fixture) -> None:
        """Ten seconds stored as the original of a run with a denoiser."""
        samples = np.arange(160000, dtype=np.float32)
        lab.repository.store_original("own", Audio(samples, 16000))
        lab.repository.save_run(run(RUN_A, "own", "done", 0, [
            step("dn", "failed", "failed", stage="denoise"),
        ]))

    def test_a_stretch_of_the_original(self, lab: Fixture) -> None:
        self.setup_audio(lab)
        cut = lab.audio.excerpt("own", RUN_A, "none", 2.0, 1.5)
        assert cut.size == 24000 * 4
        assert lab.audio.excerpt("own", RUN_A, "none", 9.5).size \
            == 8000 * 4

    @pytest.mark.parametrize("start, seconds", [
        (-1.0, 5.0), (float("nan"), 5.0), (0.0, 0.0),
        (0.0, MAX_EXCERPT_SECONDS + 1), (0.0, float("inf")), (10.0, 5.0),
        (1e308, 5.0),
    ])
    def test_refused(self, lab: Fixture, start: float,
                     seconds: float) -> None:
        self.setup_audio(lab)
        with pytest.raises(InvalidInputError):
            lab.audio.excerpt("own", RUN_A, "none", start, seconds)

    def test_audio_that_was_not_made(self, lab: Fixture) -> None:
        self.setup_audio(lab)
        with pytest.raises(NotFoundError):
            lab.audio.excerpt("own", RUN_A, "dn", 0.0)
        with pytest.raises(NotFoundError):
            lab.audio.excerpt("own", RUN_B, "none", 0.0)


class TestReferenceTurns:
    def test_the_turns_of_an_imported_session(self, lab: Fixture) -> None:
        turns = ReferenceTurns(lab.sessions, lab.repository)
        segments = [Segment(start=0, end=5, speaker="A"),
                    Segment(start=4, end=8, speaker="B")]
        lab.repository.save_reference("ami-X", make_reference(
            CORPORA["ami"], "test", "X", 10.0, segments, None,
        ))
        assert turns("ami-X") == segments
        with pytest.raises(NotFoundError):
            turns("own")
        with pytest.raises(NotFoundError):
            turns("nobody")
