"""The lab's use cases with fakes: runs, cache, mapping, labels, scores."""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from tests.unit.transcriber.fakes import (
    FakeDecoder,
    FakeJobRunner,
    FakeRecogniser,
    FakeSessionLocks,
    FakeSessionRepository,
    FakeVoiceEncoder,
    FakeVoiceprintRepository,
    FixedClock,
    RecordingProgress,
    meeting,
    voice,
)
from tests.unit.transcriber.lab_fakes import (
    Answer,
    FakeEnvironment,
    FakeLabRepository,
    FakeProcesses,
    ScriptedTool,
    StaticCatalogue,
    crashes,
    denoises,
    fails,
    halves,
    hears,
    scene_turns,
)
from transcriber.application.lab import (
    CancelLabRun,
    LabAudio,
    LabLabels,
    LabResultsView,
    LabRunStatus,
    LabScores,
    LabSessionData,
    ListLabRuns,
    ResumeLabRun,
    RunLab,
    ShowCatalogue,
    StartLabRun,
    job_key,
    plan_steps,
)
from transcriber.application.lab_builtin import (
    InProcessTools,
    exclusive_regions,
    map_speakers,
    turn_segments,
)
from transcriber.application.lab_ports import ToolRequest
from transcriber.application.ports import RecordedAudio
from transcriber.application.record import DeleteSession
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
    StateError,
)
from transcriber.domain.lab import (
    LabRequest,
    LabRun,
    ReferenceLabel,
    Segment,
    StageOutcome,
)
from transcriber.domain.models import Session, SessionState, Word
from transcriber.domain.parameters import TranscriptionParameters

SESSION = "s1"


def entry(model_id: str, stage: str, family: str = "fam",
          **extra: Any) -> Dict[str, Any]:
    """A catalogue entry."""
    return {"id": model_id, "stage": stage, "family": family,
            "display_name": model_id, **extra}


ENTRIES = [
    entry("none", "denoise", "project"),
    entry("builtin", "diarize", "project", weights=["fake/asr"],
          parameters={"num_speakers": None, "expected_speakers": None},
          types={"num_speakers": "integer",
                 "expected_speakers": "string_list"}),
    entry("project-asr", "recognise", "project", weights=["fake/asr"],
          parameters={"language": None}, types={"language": "string"}),
    entry("dn", "denoise"),
    entry("z1", "diarize", parameters={"num_speakers": None},
          types={"num_speakers": "integer"}),
    entry("z2", "diarize", lineage="z1", licence_commercial=False),
    entry("r1", "recognise", parameters={"language": None},
          types={"language": "string"}),
    entry("gated", "diarize", gated=True, token_url="https://x/accept"),
    entry("absent", "diarize", family="gone"),
]

TURNS = [(0.0, 12.0, "X"), (13.0, 25.0, "Y"), (26.0, 36.0, "X")]


def fails_once(code: str, then: Answer = denoises) -> Answer:
    """A model that fails with ``code`` on its first call only."""
    calls: List[int] = []

    def answer(request: ToolRequest) -> StageOutcome:
        """Fail the first time, then answer as ``then``."""
        calls.append(1)
        if len(calls) == 1:
            return fails(code)(request)
        return then(request)
    return answer


@dataclass
class Lab:
    """Every port of the lab, faked, and the use cases over them."""

    scene: Any = field(default_factory=meeting)
    sessions: FakeSessionRepository = field(
        default_factory=FakeSessionRepository
    )
    voiceprints: FakeVoiceprintRepository = field(
        default_factory=FakeVoiceprintRepository
    )
    repository: FakeLabRepository = field(default_factory=FakeLabRepository)
    runner: FakeJobRunner = field(default_factory=FakeJobRunner)
    processes: FakeProcesses = field(default_factory=FakeProcesses)
    environment: FakeEnvironment = field(default_factory=FakeEnvironment)
    catalogue: StaticCatalogue = field(
        default_factory=lambda: StaticCatalogue(ENTRIES)
    )
    released: List[str] = field(default_factory=list)
    locks: Any = field(default_factory=FakeSessionLocks)
    session_locks: Any = field(default_factory=FakeSessionLocks)

    def __post_init__(self) -> None:
        """Store a finished session and enrol its two people."""
        self.decoder = FakeDecoder(self.scene)
        self.encoder = FakeVoiceEncoder(self.scene)
        self.recogniser = FakeRecogniser(self.scene)
        self.sessions.create(Session(
            id=SESSION, language="es", state=SessionState.DONE,
            duration=self.scene.duration, num_speakers=2,
            expected_speakers=("ana", "luis"),
        ))
        self.sessions.store_recording(SESSION, RecordedAudio(b"x", "wav"))
        self.voiceprints.enrol("ana", voice(0), "Ana")
        self.voiceprints.enrol("luis", voice(1), "Luis")
        self.tool = ScriptedTool(self.repository, {
            "dn": denoises, "z1": scene_turns(TURNS),
            "z2": halves(["P", "Q"]),
            "r1": hears([Word(text="Hola", start=1.0, end=1.5),
                         Word(text="vale", start=14.0, end=14.5)]),
            "gated": fails("needs_token"),
        })
        self.internal = InProcessTools(
            lambda name: self.recogniser, self.encoder, self.voiceprints,
            release=lambda: self.released.append("internal"),
        )
        clock = FixedClock()
        self.execute = RunLab(
            self.sessions, self.voiceprints, self.decoder, self.repository,
            self.catalogue, self.tool, self.internal, self.encoder, clock,
            self.processes, release=lambda: self.released.append("all"),
        )
        self.status = LabRunStatus(self.sessions, self.repository,
                                   self.runner, self.processes)
        self.start = StartLabRun(
            self.sessions, self.repository, self.catalogue, self.runner,
            self.execute, clock, self.processes, self.decoder,
            self.session_locks,
        )
        self.resume = ResumeLabRun(self.repository, self.start, self.status,
                                   self.processes, self.locks)
        self.cancel = CancelLabRun(self.repository, self.status)
        self.results = LabResultsView(self.status, self.repository)
        self.labels = LabLabels(self.sessions, self.voiceprints,
                                self.repository, self.decoder, clock)
        self.scores = LabScores(self.status, self.repository)
        self.audio = LabAudio(self.status, self.repository)
        self.runs = ListLabRuns(self.sessions, self.repository, self.status)

    def run(self, **request: Any) -> str:
        """Start a run of ``request``, run its job; return its id."""
        run = self.start(SESSION, LabRequest.model_validate(request))
        self.runner.run_all()
        return run.id


class ThreadLocks:
    """Locks that really exclude threads: one re-entrant lock per key."""

    def __init__(self) -> None:
        """Start with no lock."""
        self._guard = threading.Lock()
        self._locks: Dict[str, threading.RLock] = {}

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        """Hold the lock of ``key`` for the block."""
        with self._guard:
            lock = self._locks.setdefault(key, threading.RLock())
        with lock:
            yield


@pytest.fixture()
def lab() -> Lab:
    """A fresh lab."""
    return Lab()


FULL = dict(denoisers=["none", "dn"], diarizers=["builtin", "z1"],
            recognisers=["project-asr", "r1"])


class TestCatalogue:
    def test_states(self, lab: Lab) -> None:
        lab.environment.problems["gone"] = "not installed: no env"
        lab.environment.cached["fake/asr"] = True
        view = ShowCatalogue(lab.catalogue, lab.environment)()
        states = {i.entry.id: (i.state, i.reason) for i in view.items}
        assert states["z1"] == ("ready", None)
        assert states["absent"] == ("not_installed", "not installed: no env")
        assert states["gated"][0] == "needs_token"
        assert "https://x/accept" in str(states["gated"][1])
        cached = {i.entry.id: i.weights_cached for i in view.items}
        assert cached["project-asr"] is True and cached["z1"] is None
        assert view.token_present is False
        lab.environment.token = True
        again = ShowCatalogue(lab.catalogue, lab.environment)()
        assert {i.entry.id: i.state for i in again.items}["gated"] == "ready"


class TestStart:
    def test_the_plan_runs_denoisers_first_and_builtin_first(self) -> None:
        steps = plan_steps(LabRequest(
            denoisers=["none", "dn"], diarizers=["z1", "builtin"],
            recognisers=["r1"],
        ))
        assert [s.name for s in steps] == [
            "denoise:dn", "diarize:none+builtin", "diarize:none+z1",
            "recognise:none+r1", "diarize:dn+builtin", "diarize:dn+z1",
            "recognise:dn+r1",
        ]

    @pytest.mark.parametrize("request_, error", [
        ({"diarizers": ["nope"]}, "No diarize model"),
        ({"diarizers": ["r1"]}, "No diarize model"),
        ({"diarizers": ["z1"], "params": {"z1": {"bad": 1}}},
         "no parameter"),
        ({"diarizers": ["z1"], "params": {"z2": {}}}, "not chosen"),
        ({"diarizers": ["z1"], "params": {"z1": {"num_speakers": 0}}},
         "greater than 0"),
        ({"diarizers": ["z1"], "params": {"z1": {"num_speakers": {}}}},
         "must be null or an integer, not an object"),
        ({"diarizers": ["z1"], "params": {"z1": {"num_speakers": "auto"}}},
         "must be null or an integer, not a string"),
    ])
    def test_invalid_requests(
        self, lab: Lab, request_: Dict[str, Any], error: str
    ) -> None:
        with pytest.raises(InvalidInputError, match=error):
            lab.start(SESSION, LabRequest.model_validate(request_))
        assert lab.repository.stored_runs == {}

    def test_too_many_pipelines(self, lab: Lab, monkeypatch: Any) -> None:
        import transcriber.application.lab as module

        monkeypatch.setattr(module, "MAX_PIPELINES", 3)
        with pytest.raises(InvalidInputError, match="pipelines"):
            lab.start(SESSION, LabRequest.model_validate(FULL))

    def test_the_session_must_have_its_audio(self, lab: Lab) -> None:
        lab.sessions.create(Session(id="rec", language="es"))
        with pytest.raises(StateError):
            lab.start("rec", LabRequest())
        with pytest.raises(NotFoundError):
            lab.start("nobody", LabRequest())
        with pytest.raises(InvalidInputError):
            lab.start("../x", LabRequest())

    def test_the_original_is_stored_before_the_job(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest())
        key = lab.repository.originals[SESSION]
        assert lab.runner.pending and len(lab.decoder.decoded) == 1
        assert lab.audio(SESSION, run.id, "none").size > 0
        lab.start(SESSION, LabRequest())
        assert len(lab.decoder.decoded) == 1
        assert lab.repository.originals[SESSION] == key

    def test_the_start_holds_the_session_lock(self, lab: Lab) -> None:
        """Check, original, run and queueing all happen under the lock."""
        seen: List[str] = []
        save, store = lab.repository.save_run, lab.repository.store_original

        def saving(run: LabRun) -> None:
            """Record whether the session's lock is held, then save."""
            seen.append(f"run:{lab.session_locks.holds(SESSION)}")
            save(run)

        def storing(session_id: str, audio: Any) -> str:
            """Record whether the session's lock is held, then store."""
            seen.append(f"original:{lab.session_locks.holds(session_id)}")
            return store(session_id, audio)

        lab.repository.save_run = saving  # type: ignore[method-assign]
        lab.repository.store_original = (  # type: ignore[method-assign]
            storing
        )
        lab.start(SESSION, LabRequest())
        assert seen == ["original:True", "run:True"]
        assert lab.session_locks.taken == [SESSION]
        assert not lab.session_locks.held
        with pytest.raises(NotFoundError):
            lab.start("nobody", LabRequest())
        assert not lab.session_locks.held

    def test_a_refused_request_stores_nothing(self, lab: Lab) -> None:
        with pytest.raises(InvalidInputError):
            lab.start(SESSION, LabRequest(diarizers=["nope"]))
        assert not lab.repository.originals and not lab.decoder.decoded

    def test_the_run_is_queued_under_its_own_key(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest())
        assert lab.runner.pending[0][0] == job_key(run.id)
        assert lab.status(SESSION, run.id).state == "queued"
        assert lab.runner.statuses[job_key(run.id)].state.value == "queued"


class TestRun:
    def test_a_full_run(self, lab: Lab) -> None:
        run_id = lab.run(**FULL)
        view = lab.status(SESSION, run_id)
        assert view.state == "done"
        assert view.done_steps == view.total_steps == 9
        assert all(p["state"] == "done" for p in view.pipelines)
        assert lab.tool.calls == [
            "denoise:dn", "diarize:z1", "recognise:r1", "diarize:z1",
            "recognise:r1",
        ]
        results = lab.results(SESSION, run_id)
        assert len(results["pipelines"]) == 8
        assert results["fragment_count"] > 0
        tracks = {t["id"]: t for t in results["tracks"]}
        builtin = tracks["none+builtin"]
        assert builtin["speakers"] == ["ana", "luis"]
        assert {m["method"] for m in builtin["mapping"]} == {"pipeline"}
        named = {m["label"]: (m["speaker"], m["method"])
                 for m in tracks["none+z1"]["mapping"]}
        assert named == {"X": ("ana", "voiceprint"),
                         "Y": ("luis", "voiceprint")}
        assert tracks["none+z1"]["agreement_with_consensus"] == 1.0
        assert "released" not in results
        assert "internal" in lab.released and "all" in lab.released

    def test_steps_are_cached_across_runs(self, lab: Lab) -> None:
        lab.run(**FULL)
        calls = len(lab.tool.calls)
        again = lab.run(**FULL)
        assert len(lab.tool.calls) == calls
        steps = lab.status(SESSION, again).run.steps
        assert all(s.cached for s in steps)
        other = lab.run(denoisers=["none"], diarizers=["z1"],
                        recognisers=["r1"], params={"r1": {"language": "en"}})
        assert lab.tool.calls[calls:] == ["recognise:r1"]
        assert lab.status(SESSION, other).state == "done"

    def test_a_new_revision_of_a_model_is_computed_again(
        self, lab: Lab
    ) -> None:
        lab.run(diarizers=["z1"])
        revised = [dict(e, revision=2) if e["id"] == "z1" else e
                   for e in ENTRIES]
        lab.catalogue.catalogue = StaticCatalogue(revised).catalogue
        lab.run(diarizers=["z1"])
        assert lab.tool.calls == ["diarize:z1", "diarize:z1"]

    def test_failures_are_recorded_and_the_run_goes_on(
        self, lab: Lab
    ) -> None:
        lab.tool.answers["z1"] = crashes
        run_id = lab.run(denoisers=["none"], diarizers=["z1", "gated"],
                         recognisers=["r1"])
        view = lab.status(SESSION, run_id)
        assert view.state == "failed"  # no pipeline has every step
        errors = {s.name: s.error.code for s in view.run.steps if s.error}
        assert errors == {"diarize:none+z1": "failed",
                          "diarize:none+gated": "needs_token"}
        assert all(p["state"] == "failed" for p in view.pipelines)
        assert view.pipelines[0]["error"]["code"] == "failed"
        assert len(lab.repository.discarded) == 2

    def test_a_partial_run(self, lab: Lab) -> None:
        run_id = lab.run(denoisers=["none"], diarizers=["z1", "gated"],
                         recognisers=["r1"])
        assert lab.status(SESSION, run_id).state == "partial"
        results = lab.results(SESSION, run_id)
        assert [p["id"] for p in results["pipelines"]] == ["none+z1+r1"]

    def test_a_failed_denoiser_fails_what_reads_its_audio(
        self, lab: Lab
    ) -> None:
        lab.tool.answers["dn"] = fails("out_of_memory")
        run_id = lab.run(denoisers=["none", "dn"], diarizers=["z1"])
        steps = {s.name: s for s in lab.status(SESSION, run_id).run.steps}
        denoised = steps["denoise:dn"].error
        read = steps["diarize:dn+z1"].error
        assert denoised is not None and denoised.code == "out_of_memory"
        assert read is not None and "did not finish" in read.message
        assert steps["diarize:none+z1"].state == "done"

    def test_out_of_memory_is_tried_again_after_its_stage(
        self, lab: Lab
    ) -> None:
        lab.catalogue.catalogue = StaticCatalogue(
            ENTRIES + [entry("dn2", "denoise")]
        ).catalogue
        lab.tool.answers["dn"] = fails_once("out_of_memory")
        lab.tool.answers["dn2"] = denoises
        run_id = lab.run(denoisers=["none", "dn", "dn2"], diarizers=["z1"])
        view = lab.status(SESSION, run_id)
        assert view.state == "done"
        assert lab.tool.calls == [
            "denoise:dn", "denoise:dn2", "denoise:dn",
            "diarize:z1", "diarize:z1", "diarize:z1",
        ]
        assert all(s.error is None for s in view.run.steps)

    def test_a_diarizer_out_of_memory_waits_for_the_last_diarization(
        self, lab: Lab
    ) -> None:
        lab.tool.answers["z1"] = fails_once(
            "out_of_memory", scene_turns(TURNS)
        )
        run_id = lab.run(denoisers=["none", "dn"], diarizers=["z1"],
                         recognisers=["r1"])
        assert lab.status(SESSION, run_id).state == "done"
        assert lab.tool.calls == [
            "denoise:dn", "diarize:z1", "recognise:r1", "diarize:z1",
            "diarize:z1", "recognise:r1",
        ]

    def test_out_of_memory_is_tried_again_only_once(self, lab: Lab) -> None:
        lab.tool.answers["dn"] = fails("out_of_memory")
        lab.run(denoisers=["none", "dn"], diarizers=["z1"])
        assert lab.tool.calls.count("denoise:dn") == 2

    def test_other_failures_are_not_tried_again(self, lab: Lab) -> None:
        lab.tool.answers["z1"] = crashes
        lab.run(denoisers=["none"], diarizers=["z1", "gated"])
        assert lab.tool.calls == ["diarize:z1", "diarize:gated"]

    def test_a_cancel_stops_before_the_second_try(self, lab: Lab) -> None:
        lab.catalogue.catalogue = StaticCatalogue(
            ENTRIES + [entry("dn2", "denoise")]
        ).catalogue
        lab.tool.answers["dn"] = fails("out_of_memory")
        lab.tool.answers["dn2"] = denoises
        run = lab.start(SESSION, LabRequest.model_validate(
            dict(denoisers=["none", "dn", "dn2"], diarizers=["z1"])
        ))

        def cancel_at_dn2() -> None:
            """Ask the run to stop when the second denoiser is called."""
            if lab.tool.calls == ["denoise:dn"]:
                lab.cancel(SESSION, run.id)

        lab.tool.before = cancel_at_dn2
        lab.execute(SESSION, run.id, RecordingProgress())
        assert lab.status(SESSION, run.id).state == "cancelled"
        assert lab.tool.calls == ["denoise:dn", "denoise:dn2"]

    def test_an_unknown_error_code_becomes_failed(self, lab: Lab) -> None:
        lab.tool.answers["z1"] = fails("weird")
        run_id = lab.run(diarizers=["z1"])
        step = lab.status(SESSION, run_id).run.steps[0]
        assert step.error is not None and step.error.code == "failed"

    def test_cancel_then_resume(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest.model_validate(FULL))

        def cancel_once() -> None:
            """Ask the run to stop when the first tool runs."""
            if not lab.repository.cancels and len(lab.tool.calls) == 1:
                lab.cancel(SESSION, run.id)

        lab.tool.before = cancel_once
        progress = RecordingProgress()
        lab.execute(SESSION, run.id, progress)
        view = lab.status(SESSION, run.id)
        assert view.state == "cancelled"
        states = [s.state for s in view.run.steps]
        assert states[0] == "done" and "cancelled" in states
        with pytest.raises(StateError):
            lab.cancel(SESSION, run.id)
        lab.tool.before = lambda: None
        lab.resume(SESSION, run.id)
        lab.runner.run_all()
        resumed = lab.status(SESSION, run.id)
        assert resumed.state == "done"
        assert lab.tool.calls.count("denoise:dn") == 1
        with pytest.raises(StateError):
            lab.resume(SESSION, lab.start(SESSION, LabRequest()).id)

    def test_resume_holds_the_run_lock(self, lab: Lab) -> None:
        run_id = lab.run(diarizers=["z1"])
        seen: List[bool] = []
        save = lab.repository.save_run

        def saving(run: LabRun) -> None:
            """Record whether the run's lock is held, then save."""
            seen.append(lab.locks.holds(job_key(run_id)))
            save(run)

        lab.repository.save_run = saving  # type: ignore[method-assign]
        lab.session_locks.taken.clear()
        lab.resume(SESSION, run_id)
        assert seen == [True]
        assert lab.locks.taken == [job_key(run_id)]
        assert lab.session_locks.taken == [SESSION]
        assert not lab.locks.held and not lab.session_locks.held
        with pytest.raises(StateError):
            lab.resume(SESSION, run_id)
        assert len(lab.runner.pending) == 1

    def test_resume_stores_a_missing_original(self, lab: Lab) -> None:
        run_id = lab.run(diarizers=["z1"])
        lab.repository.originals.clear()
        lab.resume(SESSION, run_id)
        assert lab.audio(SESSION, run_id, "none").size > 0

    def test_two_concurrent_resumes_queue_one_job(self) -> None:
        lab = Lab(locks=ThreadLocks())
        run_id = lab.run(diarizers=["z1"])
        save = lab.repository.save_run

        def slow(run: LabRun) -> None:
            """Save late, so both resumes check before either saves."""
            time.sleep(0.05)
            save(run)

        lab.repository.save_run = slow  # type: ignore[method-assign]
        outcomes: List[str] = []

        def resume() -> None:
            """Resume the run; record whether it was refused."""
            try:
                lab.resume(SESSION, run_id)
                outcomes.append("queued")
            except StateError:
                outcomes.append("refused")

        threads = [threading.Thread(target=resume) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
        assert sorted(outcomes) == ["queued", "refused"]
        assert [key for key, _ in lab.runner.pending] == [job_key(run_id)]

    def test_a_cancelled_queued_run_stops_at_once(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest(diarizers=["z1"]))
        lab.cancel(SESSION, run.id)
        lab.runner.run_all()
        assert lab.status(SESSION, run.id).state == "cancelled"
        assert lab.tool.calls == []

    def test_interrupted_runs(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest(diarizers=["z1"]))
        lab.runner.statuses.clear()  # the job was lost in this process
        assert lab.status(SESSION, run.id).state == "interrupted"
        other = run.model_copy(update={"worker_pid": 7})
        lab.repository.save_run(other)
        assert lab.status(SESSION, run.id).state == "interrupted"
        lab.processes.others.add(7)
        assert lab.status(SESSION, run.id).state == "queued"
        lab.repository.save_run(run.model_copy(update={"worker_pid": None}))
        assert lab.status(SESSION, run.id).state == "interrupted"
        assert [s for _, s in lab.runs(SESSION)] == ["interrupted"]

    def test_a_crash_of_the_job_fails_the_run(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest(diarizers=["z1"]))
        lab.decoder.failures.append(RuntimeError("disk"))
        with pytest.raises(RuntimeError):
            lab.execute(SESSION, run.id, RecordingProgress())
        stored = lab.status(SESSION, run.id).run
        assert (stored.state, stored.message) == ("failed", "disk")
        with pytest.raises(NotFoundError):
            lab.execute(SESSION, "f" * 32, RecordingProgress())

    def test_unknown_runs(self, lab: Lab) -> None:
        with pytest.raises(NotFoundError):
            lab.status(SESSION, "a" * 32)
        with pytest.raises(InvalidInputError):
            lab.status(SESSION, "not-a-run")


class TestDeleteSession:
    def test_a_delete_and_a_start_do_not_interleave(self) -> None:
        """A delete waits for a start decoding the original, then refuses.

        The start holds the session's lock while it decodes, stores the
        original and saves the run, so the delete (same locks) runs
        after it, sees the queued run and is refused: no lab data is
        removed under the start, and none is left without a session.
        """
        locks = ThreadLocks()
        lab = Lab(session_locks=locks)
        decoding, release = threading.Event(), threading.Event()
        decode = lab.decoder.decode

        def slow(recording: RecordedAudio) -> Any:
            """Signal the decoding, then wait to be released."""
            decoding.set()
            assert release.wait(5.0)
            return decode(recording)

        lab.decoder.decode = slow  # type: ignore[method-assign]
        delete = DeleteSession(
            lab.sessions, locks, LabSessionData(lab.repository, lab.status),
        )
        outcomes: Dict[str, Any] = {}

        def start() -> None:
            """Start a run; record its id."""
            outcomes["run"] = lab.start(SESSION, LabRequest()).id

        def deleting() -> None:
            """Delete the session; record how it ended."""
            try:
                delete(SESSION)
                outcomes["delete"] = "deleted"
            except StateError:
                outcomes["delete"] = "refused"

        starter = threading.Thread(target=start)
        starter.start()
        assert decoding.wait(5.0)
        deleter = threading.Thread(target=deleting)
        deleter.start()
        deleter.join(0.2)
        assert deleter.is_alive()  # waiting for the start's lock
        assert "delete" not in outcomes
        release.set()
        starter.join(5.0)
        deleter.join(5.0)
        assert outcomes["delete"] == "refused"
        assert lab.sessions.get(SESSION) is not None
        assert lab.repository.run(SESSION, outcomes["run"]) is not None
        assert lab.repository.original_key(SESSION) is not None

    def test_a_start_after_a_delete_finds_no_session(self) -> None:
        """A delete first: the waiting start finds no session, stores none."""
        locks = ThreadLocks()
        lab = Lab(session_locks=locks)
        delete = DeleteSession(
            lab.sessions, locks, LabSessionData(lab.repository, lab.status),
        )
        outcomes: List[str] = []

        def start() -> None:
            """Start a run; record that it found no session."""
            try:
                lab.start(SESSION, LabRequest())
                outcomes.append("started")
            except NotFoundError:
                outcomes.append("not_found")

        with locks.hold(SESSION):  # the delete is under way
            starter = threading.Thread(target=start)
            starter.start()
            starter.join(0.2)
            assert starter.is_alive()
            delete(SESSION)
        starter.join(5.0)
        assert outcomes == ["not_found"]
        assert lab.repository.original_key(SESSION) is None
        assert lab.repository.runs(SESSION) == []

    def test_a_resume_holds_the_session_lock_against_a_delete(
        self
    ) -> None:
        """A delete waits for a resume storing the original, then refuses."""
        locks = ThreadLocks()
        lab = Lab(session_locks=locks)
        run_id = lab.run(diarizers=["z1"])
        lab.repository.originals.clear()
        storing, release = threading.Event(), threading.Event()
        store = lab.repository.store_original

        def slow(session_id: str, audio: Any) -> str:
            """Signal the store, then wait to be released."""
            storing.set()
            assert release.wait(5.0)
            return store(session_id, audio)

        lab.repository.store_original = slow  # type: ignore[method-assign]
        delete = DeleteSession(
            lab.sessions, locks, LabSessionData(lab.repository, lab.status),
        )
        outcomes: List[str] = []

        def deleting() -> None:
            """Delete the session; record how it ended."""
            try:
                delete(SESSION)
                outcomes.append("deleted")
            except StateError:
                outcomes.append("refused")

        resumer = threading.Thread(target=lambda: lab.resume(SESSION, run_id))
        resumer.start()
        assert storing.wait(5.0)
        deleter = threading.Thread(target=deleting)
        deleter.start()
        deleter.join(0.2)
        assert deleter.is_alive()
        release.set()
        resumer.join(5.0)
        deleter.join(5.0)
        assert outcomes == ["refused"]
        assert lab.status(SESSION, run_id).state == "queued"
        assert lab.repository.original_key(SESSION) is not None

    def delete(self, lab: Lab) -> None:
        """Delete the session with its lab data."""
        DeleteSession(
            lab.sessions, FakeSessionLocks(),
            LabSessionData(lab.repository, lab.status),
        )(SESSION)

    def test_an_active_run_blocks_the_deletion(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest(diarizers=["z1"]))
        data = LabSessionData(lab.repository, lab.status)
        assert data.active_runs(SESSION) == [run.id]
        with pytest.raises(StateError, match=f"lab run.*{run.id}"):
            self.delete(lab)
        assert lab.sessions.get(SESSION) is not None
        assert lab.repository.run(SESSION, run.id) is not None
        lab.repository.save_run(run.model_copy(update={"state": "running"}))
        assert data.active_runs(SESSION) == [run.id]

    def test_finished_or_interrupted_runs_go_with_the_session(
        self, lab: Lab
    ) -> None:
        done = lab.run(diarizers=["z1"])
        lab.labels.replace(SESSION, [ReferenceLabel(
            start=0.0, end=5.0, kind="person", speakers=["ana"],
        )])
        lost = lab.start(SESSION, LabRequest(diarizers=["z2"]))
        lab.runner.statuses.clear()  # its job was lost: interrupted
        assert LabSessionData(
            lab.repository, lab.status
        ).active_runs(SESSION) == []
        assert lab.repository.labels(SESSION)
        assert lab.repository.original_key(SESSION) is not None
        self.delete(lab)
        assert lab.sessions.get(SESSION) is None
        assert lab.repository.runs(SESSION) == []
        assert lab.repository.labels(SESSION) == []
        assert lab.repository.original_key(SESSION) is None
        assert lab.repository.results(SESSION, done) is None
        assert lab.repository.run(SESSION, lost.id) is None


class TestMapping:
    def test_speakers_without_own_speech_take_the_builtin_names(
        self, lab: Lab
    ) -> None:
        lab.tool.answers["z1"] = scene_turns([(0, 36, "X"), (0, 36, "Y")])
        run_id = lab.run(diarizers=["builtin", "z1"])
        tracks = {t["id"]: t for t in lab.results(SESSION, run_id)["tracks"]}
        mapping = tracks["none+z1"]["mapping"]
        assert {m["method"] for m in mapping} == {"builtin_overlap"}
        assert {m["speaker"] for m in mapping} == {"ana", "luis"}

    def test_without_builtin_they_stay_anonymous(self, lab: Lab) -> None:
        lab.tool.answers["z1"] = scene_turns([(0, 36, "X"), (0, 36, "Y")])
        run_id = lab.run(diarizers=["z1"])
        track = lab.results(SESSION, run_id)["tracks"][0]
        assert track["speakers"] == ["~X", "~Y"]
        assert {m["method"] for m in track["mapping"]} == {"none"}

    def test_mappings_are_cached(self, lab: Lab) -> None:
        lab.run(diarizers=["z1"])
        calls = len(lab.encoder.calls)
        lab.run(diarizers=["z1"])
        assert len(lab.encoder.calls) == calls
        assert len(lab.repository.mappings) == 1

    def test_map_speakers_directly(self, lab: Lab) -> None:
        audio = lab.decoder.decode(RecordedAudio(b"", "wav"))
        segments = [Segment(start=s, end=e, speaker=x) for s, e, x in TURNS]
        mapping = map_speakers(
            segments, audio, lab.encoder, lab.voiceprints.all(), 0.8,
            TranscriptionParameters(),
        )
        assert [(m.label, m.speaker) for m in mapping] == [
            ("X", "ana"), ("Y", "luis"),
        ]
        assert mapping[0].speech_seconds == pytest.approx(22.0)
        nobody = map_speakers(segments, audio, lab.encoder, [], 0.8,
                              TranscriptionParameters())
        assert {m.method for m in nobody} == {"none"}

    def test_exclusive_regions(self) -> None:
        regions = exclusive_regions(
            [Segment(start=0, end=5, speaker="a"),
             Segment(start=4, end=6, speaker="b"),
             Segment(start=6, end=6.2, speaker="a")], 10.0,
        )
        assert regions == {"a": [(0.0, 4.0)], "b": [(5.0, 6.0)]}


class TestInProcess:
    def request(self, lab: Lab, model: str, stage: str,
                **params: Any) -> ToolRequest:
        """A request for an in-process model."""
        audio = lab.decoder.decode(RecordedAudio(b"", "wav"))
        return ToolRequest(
            entry=lab.catalogue.entry(model), stage=stage,  # type: ignore
            params=params, audio=Path("/x.wav"), workspace=Path("/w"),
            load=lambda: audio,
        )

    def test_the_builtin_diarization_names_people(self, lab: Lab) -> None:
        outcome = lab.internal.run(self.request(
            lab, "builtin", "diarize", language="es", num_speakers=2,
            expected_speakers=["ana", "luis"],
        ), lambda: False)
        speakers = [s.speaker for s in outcome.segments]
        assert speakers == ["ana", "luis", "ana"]
        assert outcome.model_version == "fake-whisper"

    def test_the_project_recogniser_gives_unattributed_words(
        self, lab: Lab
    ) -> None:
        outcome = lab.internal.run(self.request(
            lab, "project-asr", "recognise", language="auto",
        ), lambda: False)
        assert outcome.words and all(w.speaker is None
                                     for w in outcome.words)
        assert outcome.language is None
        assert lab.recogniser.languages == [None]

    def test_refusals(self, lab: Lab) -> None:
        with pytest.raises(LabStepError) as cancelled:
            lab.internal.run(self.request(lab, "builtin", "diarize"),
                             lambda: True)
        assert cancelled.value.code == "cancelled"
        with pytest.raises(LabStepError) as unknown:
            lab.internal.run(self.request(lab, "none", "denoise"),
                             lambda: False)
        assert unknown.value.code == "unknown_model"
        lab.recogniser.failures.append(InvalidInputError("no speech"))
        with pytest.raises(LabStepError) as failed:
            lab.internal.run(self.request(lab, "project-asr", "recognise"),
                             lambda: False)
        assert failed.value.code == "failed"
        lab.internal.release()
        assert lab.released == ["internal"]

    def test_turn_segments(self) -> None:
        assert turn_segments([], 1.0) == []


class TestReview:
    def labelled(self, lab: Lab) -> str:
        """A finished run of two diarizers and a recogniser."""
        return lab.run(diarizers=["z1", "z2"], recognisers=["r1"])

    def test_fragments_pages(self, lab: Lab) -> None:
        run_id = self.labelled(lab)
        page = lab.results.fragments(SESSION, run_id, 0, 2)
        assert page["total"] >= 3 and len(page["fragments"]) == 2
        first = page["fragments"][0]
        assert first["disagreement"] >= page["fragments"][1]["disagreement"]
        assert set(first["pipelines"]) == {"none+z1+r1", "none+z2+r1"}
        timed = lab.results.fragments(SESSION, run_id, 0, 200, "time")
        starts = [f["start"] for f in timed["fragments"]]
        assert starts == sorted(starts)
        said = [f["pipelines"]["none+z1+r1"] for f in timed["fragments"]
                if f["pipelines"]["none+z1+r1"]["text"]]
        assert said[0]["words_by_speaker"] == [
            {"speaker": "ana", "text": "Hola"}
        ]
        for bad in ((-1, 5, "time"), (0, 0, "time"), (0, 5, "loud")):
            with pytest.raises(InvalidInputError):
                lab.results.fragments(SESSION, run_id, *bad)

    def test_spans(self, lab: Lab) -> None:
        run_id = self.labelled(lab)
        spans = lab.results.spans(SESSION, run_id)["spans"]
        timed = lab.results.fragments(SESSION, run_id, 0, 200, "time")
        assert [s["start"] for s in spans] == [
            f["start"] for f in timed["fragments"]
        ]
        assert set(spans[0]) == {"index", "start", "end", "disagreement",
                                 "distinct_answers"}

    def test_tracks(self, lab: Lab) -> None:
        run_id = self.labelled(lab)
        shown = lab.results.tracks(SESSION, run_id)
        assert set(shown["tracks"]) == {"none+z1", "none+z2"}
        assert shown["tracks"]["none+z1"][0]["speaker"] == "ana"
        assert shown["consensus"]

    def test_labels_and_scores(self, lab: Lab) -> None:
        run_id = self.labelled(lab)
        saved = lab.labels.replace(SESSION, [
            ReferenceLabel(start=0, end=12, kind="person",
                           speakers=["ana"], text="hola"),
            ReferenceLabel(start=13, end=25, kind="person",
                           speakers=["luis"]),
            ReferenceLabel(start=30, end=31, kind="unsure"),
        ])
        assert all(label.updated_at is not None for label in saved)
        assert lab.labels(SESSION) == saved
        assert lab.labels.speakers() == ["ana", "luis"]
        scores = lab.scores(SESSION, run_id)
        assert scores["labelled_seconds"] == pytest.approx(24.0)
        assert scores["unsure_seconds"] == pytest.approx(1.0)
        best = scores["scores"][0]
        assert best["pipeline"] == "none+z1+r1" and best["rank"] == 1
        assert best["error_rate"] == pytest.approx(0.0)
        assert best["wer"] == pytest.approx(0.0)
        page = lab.results.fragments(SESSION, run_id, 0, 200, "time")
        assert page["fragments"][0]["label"]["kind"] == "person"
        left = lab.results.fragments(SESSION, run_id, 0, 200, "time", True)
        assert left["total"] < page["total"]
        assert all(f["label"] is None for f in left["fragments"])
        assert left["total"] == sum(
            1 for f in page["fragments"] if f["label"] is None
        )
        with pytest.raises(InvalidInputError):
            lab.labels.replace(SESSION, [ReferenceLabel(
                start=0, end=1, kind="person", speakers=["eve"],
            )])

    def test_label_duration_falls_back_to_the_audio(self, lab: Lab) -> None:
        lab.sessions.save(lab.sessions.get(SESSION).model_copy(  # type: ignore
            update={"duration": 0.0}
        ))
        lab.labels.replace(SESSION, [ReferenceLabel(start=0, end=30,
                                                    kind="nobody")])
        with pytest.raises(InvalidInputError):
            lab.labels.replace(SESSION, [ReferenceLabel(start=0, end=99,
                                                        kind="nobody")])

    def test_no_results_yet(self, lab: Lab) -> None:
        run = lab.start(SESSION, LabRequest(diarizers=["z1"]))
        with pytest.raises(NotFoundError):
            lab.results(SESSION, run.id)
        with pytest.raises(NotFoundError):
            lab.scores(SESSION, run.id)

    def test_audio(self, lab: Lab) -> None:
        run_id = lab.run(denoisers=["none", "dn"], diarizers=["z1"])
        original = lab.audio(SESSION, run_id, "none")
        cleaned = lab.audio(SESSION, run_id, "dn")
        assert original.size == cleaned.size > 0
        assert original.read(0, 64) != cleaned.read(0, 64)
        with pytest.raises(NotFoundError):
            lab.audio(SESSION, run_id, "other")

    def test_playback_only_reads(self, lab: Lab) -> None:
        run_id = lab.run(diarizers=["z1"])
        lab.repository.originals.clear()
        decoded = len(lab.decoder.decoded)
        with pytest.raises(NotFoundError, match="original"):
            lab.audio(SESSION, run_id, "none")
        assert not lab.repository.originals
        assert len(lab.decoder.decoded) == decoded
