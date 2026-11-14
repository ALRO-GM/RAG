"""The lab's use cases with fakes: runs, cache, mapping, labels, scores."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.unit.transcriber.fakes import (
    FakeDecoder,
    FakeJobRunner,
    FakeRecogniser,
    FakeSessionRepository,
    FakeVoiceEncoder,
    FakeVoiceprintRepository,
    FixedClock,
    RecordingProgress,
    meeting,
    voice,
)
from tests.unit.transcriber.lab_fakes import (
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
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
    StateError,
)
from transcriber.domain.lab import (
    LabRequest,
    ReferenceLabel,
    Segment,
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
          parameters={"num_speakers": None, "expected_speakers": None}),
    entry("project-asr", "recognise", "project", weights=["fake/asr"],
          parameters={"language": None}),
    entry("dn", "denoise"),
    entry("z1", "diarize", parameters={"num_speakers": None}),
    entry("z2", "diarize", lineage="z1", licence_commercial=False),
    entry("r1", "recognise", parameters={"language": None}),
    entry("gated", "diarize", gated=True, token_url="https://x/accept"),
    entry("absent", "diarize", family="gone"),
]

TURNS = [(0.0, 12.0, "X"), (13.0, 25.0, "Y"), (26.0, 36.0, "X")]


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
            self.execute, clock, self.processes,
        )
        self.resume = ResumeLabRun(self.repository, self.start, self.status,
                                   self.processes)
        self.cancel = CancelLabRun(self.repository, self.status)
        self.results = LabResultsView(self.status, self.repository)
        self.labels = LabLabels(self.sessions, self.voiceprints,
                                self.repository, self.decoder, clock)
        self.scores = LabScores(self.status, self.repository)
        self.audio = LabAudio(self.status, self.repository, self.sessions,
                              self.decoder)
        self.runs = ListLabRuns(self.sessions, self.repository, self.status)

    def run(self, **request: Any) -> str:
        """Start a run of ``request``, run its job; return its id."""
        run = self.start(SESSION, LabRequest.model_validate(request))
        self.runner.run_all()
        return run.id


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
        lab.repository.originals.clear()
        assert lab.audio(SESSION, run_id, "none").size == original.size
