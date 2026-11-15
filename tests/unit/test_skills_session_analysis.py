"""SessionAnalysis: the whole flow, step records, resume, optional steps."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence

import pytest

from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.judges import FallbackJudge
from src.application.skills.ports import StepOutput
from src.application.skills.session_analysis import (
    JudgeSetup,
    SessionAnalysis,
    most_promising,
    step_table,
)
from src.application.skills.tagging import (
    SentenceTag,
    SentenceTagger,
    TagConfirmer,
    TaggedSentence,
)
from src.domain.errors import (
    InvalidInputError,
    NotFoundError,
    QuotaExhaustedError,
)
from src.domain.skills.analysis_run import (
    STEPS,
    StepRecord,
    is_step,
    new_run,
    steps_to_run,
)
from src.domain.skills.speech import Sentence
from src.domain.skills.transcript import Transcript
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryAnalysisStore,
    InMemoryTranscriptRepository,
    judge_reply,
    make_catalogue,
    make_transcript,
)

T0 = datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)


class Clock:
    """A clock that moves one second per reading."""

    def __init__(self) -> None:
        """Start at ``T0``."""
        self.ticks = 0

    def __call__(self) -> datetime:
        """Return the next second."""
        self.ticks += 1
        return T0 + timedelta(seconds=self.ticks)


def reply(system: str, user: str) -> str:
    """Confirm every offered sentence as L1/P1; rate nothing."""
    if "TARGET sentence" in user:
        named = "L1" if "listening" in user.lower() else "P1"
        return json.dumps({"indicator": named, "reason": "fits"})
    return judge_reply()


class FakeActs:
    """A dialogue-act tagger labelling every sentence ``otro``."""

    def tag(
        self, transcript: Transcript, sentences: Sequence[Sentence]
    ) -> StepOutput:
        """Label every sentence."""
        return StepOutput(
            [{"id": s.id, "labels": ["otro"]} for s in sentences],
            {"sentences": len(sentences)}, "fake-acts",
        )


class FakeEvents:
    """An extractor that finds one decision and remembers the acts."""

    def __init__(self) -> None:
        """Nothing seen yet."""
        self.acts: Optional[Any] = None

    def extract(
        self,
        transcript: Transcript,
        sentences: Sequence[Sentence],
        acts: Optional[Any],
    ) -> StepOutput:
        """Return one decision."""
        self.acts = acts
        return StepOutput({"decisions": [{"id": "d1"}]}, {"decisions": 1},
                          "fake-events")


def build(
    judge: Optional[Any] = None,
    store: Optional[InMemoryAnalysisStore] = None,
    transcripts: Optional[List[Transcript]] = None,
    **kwargs: Any,
) -> SessionAnalysis:
    """Return the flow over the English test meeting and fakes."""
    model = judge or FakeChatModel(responder=reply)
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), FakeEmbedder(dim=64), min_words=4
    )
    tagger = SentenceTagger(retriever, min_similarity=-1.0)

    def judges() -> JudgeSetup:
        """The fake judge for both steps."""
        return JudgeSetup(
            "claude_code", TagConfirmer(model),
            AssessmentService(retriever, model, min_incidents=1),
        )

    return SessionAnalysis(
        InMemoryTranscriptRepository(
            transcripts or [make_transcript(session_id="s1")]
        ),
        store or InMemoryAnalysisStore(),
        make_catalogue,
        lambda name, language: make_catalogue(language, name),
        tagger, kwargs.pop("judges", judges), Clock(),
        **kwargs,
    )


# ------------------------------------------------------------- domain


def test_a_new_run_has_every_step_pending_and_the_open_issues() -> None:
    """Consent, human review and several situations are always stated."""
    run = new_run("s1", T0)
    assert [s.name for s in run.steps] == list(STEPS)
    assert {s.status for s in run.steps} == {"pending"}
    assert [i.id for i in run.open_issues] == [
        "consent", "human_review", "situations",
    ]
    assert run.step("acts").name == "acts"
    with pytest.raises(KeyError):
        run.step("nope")
    assert is_step("reports") and not is_step("../run")


def test_resume_starts_at_the_first_step_not_finished() -> None:
    """Done and unavailable steps are kept; the rest runs again."""
    run = new_run("s1", T0)
    assert steps_to_run(None) == list(STEPS)
    for name in STEPS[:3]:
        run = run.with_step(StepRecord(name=name, status="done"))
    run = run.with_step(StepRecord(name="acts", status="unavailable"))
    run = run.with_step(StepRecord(name="events", status="unavailable"))
    run = run.with_step(StepRecord(name="confirmation", status="failed"))
    assert steps_to_run(run) == ["confirmation", "assessment", "reports"]
    assert steps_to_run(run, restart=True) == list(STEPS)
    for name in STEPS[5:]:
        run = run.with_step(StepRecord(name=name, status="done"))
    assert steps_to_run(run) == []


def test_most_promising_keeps_the_best_candidates_in_order() -> None:
    """The cap keeps the highest scores, then speaking order."""
    def item(n: int, score: float) -> TaggedSentence:
        tags = [] if score < 0 else [SentenceTag(
            indicator_id="L1", competency_id="listening",
            polarity="positive", score=score,
        )]
        return TaggedSentence(sentence=Sentence(
            id=f"t{n}.s0", turn=n, index=0, speaker="A", text="x"
        ), tags=tags)

    tagged = [item(0, 0.2), item(1, -1), item(2, 0.9), item(3, 0.5)]
    assert most_promising(tagged, 0) == [0, 2, 3]
    assert most_promising(tagged, 5) == [0, 2, 3]
    assert most_promising(tagged, 2) == [2, 3]


# ----------------------------------------------------------- the flow


def test_the_whole_flow_records_every_step_and_its_results() -> None:
    """Each step is done or unavailable, with time, who and counts."""
    store = InMemoryAnalysisStore()
    seen: List[StepRecord] = []
    run = build(store=store).run("s1", synthetic=True, on_step=seen.append)
    assert run.status == "done" and run.finished_at is not None
    assert run.synthetic
    status = {s.name: s.status for s in run.steps}
    assert status == {
        "sentences": "done", "interaction": "done", "candidates": "done",
        "acts": "unavailable", "events": "unavailable",
        "confirmation": "done", "assessment": "done", "reports": "done",
    }
    assert [r.name for r in seen] == list(STEPS)
    assert all(s.seconds is not None and s.started_at for s in run.steps)
    assert "#82" in run.step("acts").note
    assert "#83" in run.step("events").note
    sentences = run.step("sentences")
    assert sentences.counts == {"sentences": 8, "speakers": 3}
    assert run.step("candidates").model == "bm25"
    confirmation = run.step("confirmation")
    assert confirmation.judge == "claude_code"
    assert confirmation.model == "fake-judge"
    assert confirmation.counts["sent"] == confirmation.counts["confirmed"]
    assert confirmation.counts["not_sent"] == 0
    assert run.step("assessment").counts["participants"] == 3
    assert run.step("reports").counts == {"reports": 3}
    # Every step's result is stored next to the record.
    names = {name for _, name in store.files}
    assert {"sentences.json", "interaction.json", "candidates.json",
            "confirmation.json", "assessment.json", "summary.json",
            "scorecard.md", "report-1.html", "report-3.html"} <= names
    assert "acts.json" not in names
    summary = json.loads(store.files[("s1", "summary.json")])
    assert [p["name"] for p in summary["people"]] == ["Maya", "Tom", "Priya"]
    assert summary["synthetic"] is True
    assert [i["id"] for i in summary["open_issues"]] == [
        "consent", "human_review", "situations",
    ]
    assert "<html" in store.files[("s1", "report-1.html")].lower()
    # The record was saved while it ran, not only at the end.
    assert any(
        s.status == "running" for r in store.saved for s in r.steps
    )


def test_results_and_reports_are_read_back_by_step() -> None:
    """The stored results are served by step name and report number."""
    analysis = build()
    analysis.run("s1")
    assert analysis.status("s1").status == "done"
    assert analysis.result("s1", "sentences")["speakers"][0]["name"] == "Maya"
    assert analysis.result("s1", "reports")["people"][0]["report"] == (
        "report-1.html"
    )
    assert "Maya" in analysis.report("s1", 1)
    with pytest.raises(InvalidInputError):
        analysis.result("s1", "run")
    with pytest.raises(NotFoundError):
        analysis.result("s1", "acts")
    with pytest.raises(InvalidInputError):
        analysis.report("s1", 0)
    with pytest.raises(NotFoundError):
        analysis.report("s1", 9)
    with pytest.raises(InvalidInputError):
        analysis.status("../etc")
    with pytest.raises(NotFoundError):
        analysis.status("other")


def test_an_unknown_session_fails_before_anything_is_stored() -> None:
    """No transcript, no run."""
    store = InMemoryAnalysisStore()
    with pytest.raises(NotFoundError):
        build(store=store).run("missing")
    assert store.runs == {}


def test_a_failed_step_keeps_the_earlier_ones_and_resumes_there() -> None:
    """The judge breaks: steps a-e stay done; the rerun starts at f."""
    store = InMemoryAnalysisStore()
    calls: List[str] = []

    def broken(system: str, user: str) -> str:
        """Fail like a model that cannot load."""
        calls.append(user)
        raise RuntimeError("CUDA out of memory\nlong trace")

    run = build(FakeChatModel(responder=broken), store).run("s1")
    assert run.status == "failed"
    failed = run.step("confirmation")
    assert failed.status == "failed"
    assert failed.error == "CUDA out of memory"
    assert run.step("candidates").status == "done"
    assert run.step("assessment").status == "pending"
    assert store.runs["s1"].status == "failed"
    rows = step_table(run)
    assert rows[5][1] == "failed" and rows[5][4] == "CUDA out of memory"
    # Resume with a working judge: a-e are not run again.
    resumed = build(store=store).run("s1")
    assert resumed.status == "done"
    assert resumed.step("sentences").started_at == (
        run.step("sentences").started_at
    )
    assert resumed.step("confirmation").status == "done"


def test_restart_runs_every_step_again() -> None:
    """``restart`` ignores what was done."""
    store = InMemoryAnalysisStore()
    build(store=store).run("s1")
    saves = len(store.saved)
    assert build(store=store).run("s1").status == "done"
    assert len(store.saved) == saves + 2  # nothing left to do
    again = build(store=store).run("s1", restart=True)
    assert again.status == "done"
    assert len(store.saved) > saves + 2 + 2 * len(STEPS)


def test_confirmation_reads_at_most_the_cap() -> None:
    """Sentences beyond the cap keep their candidates, undecided."""
    store = InMemoryAnalysisStore()
    run = build(store=store, max_confirmations=2).run("s1")
    counts = run.step("confirmation").counts
    assert counts["sent"] == 2 and counts["not_sent"] > 0
    stored = json.loads(store.files[("s1", "confirmation.json")])
    assert stored["max_confirmations"] == 2
    decided = [s for s in stored["sentences"] if "decision" in s]
    assert len(decided) == 2
    undecided = [s for s in stored["sentences"]
                 if s["tags"] and "decision" not in s]
    assert len(undecided) == counts["not_sent"]


def test_optional_steps_run_when_their_modules_exist() -> None:
    """Acts and events are stored; events receive the acts."""
    store = InMemoryAnalysisStore()
    events = FakeEvents()
    run = build(store=store, acts=FakeActs(), events=events).run("s1")
    assert run.step("acts").status == "done"
    assert run.step("acts").model == "fake-acts"
    assert run.step("events").counts == {"decisions": 1}
    assert events.acts is not None and events.acts[0]["labels"] == ["otro"]
    assert json.loads(store.files[("s1", "events.json")]) == {
        "decisions": [{"id": "d1"}]
    }


def test_fallback_and_both_modes_are_recorded() -> None:
    """The fallback reason and the second judge reach the records."""

    class OutOfQuota:
        """Claude refusing for quota."""

        model_name = "claude-opus"

        def complete(self, system: str, user: str, max_tokens: int) -> str:
            """Refuse."""
            raise QuotaExhaustedError("usage limit reached", "3pm")

    local = FakeChatModel(responder=reply, model_name="qwen-local")
    judge = FallbackJudge(OutOfQuota(), local)
    second = FakeChatModel(responder=reply, model_name="qwen-second")
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), FakeEmbedder(dim=64), min_words=4
    )

    def judges() -> JudgeSetup:
        """Both mode: the fallback judge and a second one."""
        return JudgeSetup(
            "both", TagConfirmer(judge),
            AssessmentService(retriever, judge, min_incidents=1),
            TagConfirmer(second),
        )

    store = InMemoryAnalysisStore()
    run = build(store=store, judges=judges).run("s1")
    confirmation = run.step("confirmation")
    assert confirmation.judge == "both"
    assert "usage limit" in confirmation.fallback_reason
    assert confirmation.model == "qwen-local, qwen-second"
    assert "usage limit" in run.step("assessment").fallback_reason
    stored = json.loads(store.files[("s1", "confirmation.json")])
    assert stored["judge"]["second_models"] == ["qwen-second"]
    assert "comparison" in stored
    assert "fell back" in step_table(run)[5][4]
