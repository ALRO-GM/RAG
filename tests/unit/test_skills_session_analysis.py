"""SessionAnalysis: the whole flow, step records, resume, optional steps."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence

import pytest

from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.judges import CallBudget, FallbackJudge
from src.application.skills.ports import StepOutput
from src.application.skills.synthesis import ProfileSynthesizer
from src.application.skills.session_analysis import (
    AnalysisResults,
    JudgeSetup,
    SessionAnalysis,
    confirmed_observations,
    most_promising,
    step_table,
)
from src.application.skills.tagging import (
    SentenceTag,
    SentenceTagger,
    TagConfirmer,
    TagDecision,
    TaggedSentence,
)
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    LimitError,
    ModelError,
    NotFoundError,
    QuotaExhaustedError,
)
from src.domain.skills.analysis_run import (
    STEPS,
    AnalysisRun,
    StepRecord,
    is_step,
    new_run,
    shown_status,
    steps_to_run,
)
from src.domain.skills.speech import Sentence
from src.domain.skills.synthesis import JobSought
from src.domain.skills.transcript import Transcript
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryAnalysisLock,
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
        kwargs.pop("lock", InMemoryAnalysisLock()),
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
    assert steps_to_run(run) == [
        "confirmation", "assessment", "reports", "synthesis",
    ]
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
        "synthesis": "unavailable",
    }
    assert [r.name for r in seen] == list(STEPS)
    assert all(s.seconds is not None and s.started_at for s in run.steps)
    assert "no dialogue-act tagger" in run.step("acts").note
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


def test_step_f_confirmations_become_estimates_and_observations() -> None:
    """The judge rates nothing; what step f confirmed is not lost (#103).

    Step g makes no extra call for it: one per person and competency.
    """
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=reply)
    run = build(model, store=store).run("s1")
    confirmed = run.step("confirmation").counts["confirmed"]
    counts = run.step("assessment").counts
    assert counts["rated"] == 0 and counts["provisional"] > 0
    assert counts["observations"] == confirmed
    assert len(model.calls) == confirmed + 3 * 2
    stored = json.loads(store.files[("s1", "assessment.json")])
    assert stored["format_version"] == 2
    cell = stored["scorecards"][0]["assessments"][0]
    assert cell["status"] == "provisional"
    assert cell["estimate"]["reasons"][0] == "only_confirmations"
    assert {o["source"] for o in cell["observations"]} == {"confirmation"}
    assert all(o["sentence_id"] for o in cell["observations"])
    summary = json.loads(store.files[("s1", "summary.json")])
    maya = summary["people"][0]
    assert maya["rated"] == 0 and maya["provisional"] > 0
    assert maya["observations"] > 0 and maya["why"]
    assert set(maya["estimates"]) <= {"listening", "planning"}
    assert maya["words"] > 0 and summary["min_incidents"] == 1
    page = store.files[("s1", "report-1.html")]
    assert "Provisional estimates and observations" in page
    scorecard = store.files[("s1", "scorecard.md")]
    assert "provisional ~" in scorecard
    assert "[confirmed by the judge sentence by sentence]" in scorecard


def test_only_the_first_judges_confirmations_are_observations() -> None:
    """Rejected, unparsed and second-judge decisions are left out."""
    sentence = Sentence(id="t0.s0", turn=0, index=0, speaker="Maya",
                        text="What do you mean", start=1.5)
    tag = SentenceTag(indicator_id="L1", competency_id="listening",
                      polarity="positive", score=0.5)
    items = [
        TaggedSentence(sentence=sentence, tags=[tag], decision=TagDecision(
            status="confirmed", indicator_id="L1", competency_id="listening",
            polarity="positive",
        )),
        TaggedSentence(sentence=sentence, tags=[tag], decision=TagDecision(
            status="rejected"), second_decision=TagDecision(
            status="confirmed", indicator_id="L1", polarity="positive")),
        TaggedSentence(sentence=sentence, tags=[tag], decision=TagDecision(
            status="confirmed", indicator_id=None)),
        TaggedSentence(sentence=sentence, tags=[tag]),
    ]
    found, = confirmed_observations(items)
    assert (found.turn, found.quote, found.start, found.sentence_id) == (
        0, "What do you mean", 1.5, "t0.s0",
    )
    assert (found.source, found.polarity) == ("confirmation", "positive")


def test_results_and_reports_are_read_back_by_step() -> None:
    """The stored results are served by step name and report number."""
    store = InMemoryAnalysisStore()
    build(store=store).run("s1")
    analysis = AnalysisResults(store)
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


def test_one_run_per_session_and_the_lock_is_given_back() -> None:
    """Another worker holds the session: busy; after a run it is free."""
    lock = InMemoryAnalysisLock()
    lock.taken["s1"] = 777
    with pytest.raises(BusyError, match="process 777"):
        build(lock=lock).run("s1")
    del lock.taken["s1"]
    store = InMemoryAnalysisStore()
    run = build(store=store, lock=lock).run("s1")
    assert run.worker_pid == 4242
    assert lock.taken == {} and lock.acquired == ["s1"]


def test_the_transcript_is_read_only_once_the_session_is_held() -> None:
    """A deletion holds the session until the transcript is gone (#121)."""
    lock = InMemoryAnalysisLock()
    store = InMemoryAnalysisStore()
    analysis = build(store=store, lock=lock)
    repository = analysis.transcripts
    assert isinstance(repository, InMemoryTranscriptRepository)
    lock.taken["s1"] = 777
    with pytest.raises(BusyError):
        analysis.run("s1")
    assert repository.finds == []
    del lock.taken["s1"]
    del repository.transcripts["s1"]
    with pytest.raises(NotFoundError):
        analysis.run("s1", job=JobSought(text="Shift lead"))
    assert repository.finds == ["s1"]
    assert lock.taken == {} and store.runs == {} and store.files == {}


def test_the_lock_is_given_back_after_a_crash() -> None:
    """A store that cannot save still releases the session."""
    lock = InMemoryAnalysisLock()
    store = InMemoryAnalysisStore()

    def broken(run: Any) -> None:
        """Fail to save."""
        raise OSError("disk full")

    store.save = broken  # type: ignore[method-assign]
    with pytest.raises(OSError):
        build(store=store, lock=lock).run("s1")
    assert lock.taken == {}


def test_a_run_left_running_by_a_dead_worker_reads_interrupted() -> None:
    """``shown_status`` and the API's status say so; a rerun resumes."""
    run = new_run("s1", T0).model_copy(update={"status": "running"})
    assert shown_status(run, alive=True) == "running"
    assert shown_status(run, alive=False) == "interrupted"
    assert shown_status(run.model_copy(update={"status": "done"}), False) == (
        "done"
    )


def test_only_reruns_the_named_steps_and_keeps_the_rest() -> None:
    """Acts and events come later without asking the judge again."""
    store = InMemoryAnalysisStore()
    first = build(store=store).run("s1")
    assert first.step("acts").status == "unavailable"
    judge = FakeChatModel(responder=reply)
    analysis = build(judge, store=store, acts=FakeActs(), events=FakeEvents())
    again = analysis.run("s1", only=["events", "acts"])
    assert again.status == "done"
    assert again.step("acts").status == "done"
    assert again.step("events").status == "done"
    assert again.step("confirmation") == first.step("confirmation")
    assert judge.calls == []
    with pytest.raises(InvalidInputError, match="nope"):
        build(store=store).run("s1", only=["nope"])
    with pytest.raises(InvalidInputError):
        build(store=store).run("s1", only=[" "])


def test_the_call_budget_counts_real_calls_and_refuses_past_it() -> None:
    """Only calls reaching the model count; past the limit: LimitError."""
    judge = FakeChatModel(responder=reply, model_name="m")
    budget = CallBudget(judge, 2)
    assert budget.model_name == "m" and budget.fallback_reason == ""
    budget.complete("s", "u", 5)
    budget.complete("s", "u", 5)
    assert budget.remaining == 0
    with pytest.raises(LimitError, match=r"limit of this analysis \(2\)"):
        budget.complete("s", "u", 5)
    assert len(judge.calls) == 2
    budget.release()


def test_step_f_leaves_room_for_step_g_within_the_budget() -> None:
    """3 speakers x 2 competencies are kept for step g."""
    model = FakeChatModel(responder=reply)
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), FakeEmbedder(dim=64), min_words=4
    )

    def judges() -> JudgeSetup:
        """A budget of 8: 6 kept for step g, 2 for step f."""
        budget = CallBudget(model, 8)
        return JudgeSetup(
            "claude_code", TagConfirmer(budget),
            AssessmentService(retriever, budget, min_incidents=1),
            budget=budget,
        )

    run = build(judges=judges).run("s1")
    assert run.status == "done"
    confirmation = run.step("confirmation").counts
    assert confirmation["sent"] == 2 and confirmation["judge_calls"] == 2
    assert run.step("assessment").counts["judge_calls"] == 6
    assert len(model.calls) == 8

    def tight() -> JudgeSetup:
        """A budget smaller than step g needs: f reads nothing, g fails."""
        budget = CallBudget(model, 3)
        return JudgeSetup(
            "claude_code", TagConfirmer(budget),
            AssessmentService(retriever, budget, min_incidents=1),
            budget=budget,
        )

    failed = build(judges=tight).run("s1", restart=True)
    assert failed.step("confirmation").counts["sent"] == 0
    assert failed.step("assessment").status == "failed"
    assert "max_judge_calls" in failed.step("assessment").error


def test_a_restart_clears_results_and_results_need_a_done_step() -> None:
    """After a restart the old files are gone; failed steps serve none."""
    store = InMemoryAnalysisStore()
    build(store=store).run("s1")
    results = AnalysisResults(store)
    assert results.result("s1", "candidates")

    def broken(system: str, user: str) -> str:
        """Fail in step f."""
        raise RuntimeError("down")

    run = build(FakeChatModel(responder=broken), store).run(
        "s1", restart=True
    )
    assert run.step("confirmation").status == "failed"
    assert ("s1", "report-1.html") not in store.files
    with pytest.raises(NotFoundError):
        results.result("s1", "confirmation")
    with pytest.raises(NotFoundError):
        results.report("s1", 1)
    assert results.result("s1", "sentences")


def test_restart_with_steps_is_refused_and_steps_alone_can_be_partial() -> (
    None
):
    """``--restart --steps`` makes no sense; a lone step leaves partial."""
    store = InMemoryAnalysisStore()
    with pytest.raises(InvalidInputError, match="not both"):
        build(store=store).run("s1", restart=True, only=["acts"])
    run = build(store=store).run("s1", only=["sentences"])
    assert run.status == "partial"
    assert run.step("interaction").status == "pending"
    assert build(store=store).run("s1").status == "done"


def test_a_model_moved_to_the_cpu_is_noted_on_the_step_it_happened() -> None:
    """A shared model's device note lands on the step that added it."""
    notes: List[str] = []
    moved = "CUDA ran out of memory: the judge ran on the CPU instead."

    def responder(system: str, user: str) -> str:
        """Answer as usual; the first call "moves" the model."""
        if not notes:
            notes.append(moved)
        return reply(system, user)

    run = build(
        judge=FakeChatModel(responder=responder),
        device_notes=lambda: list(notes),
    ).run("s1")
    assert run.status == "done", "a move to the CPU does not fail the run"
    assert run.step("confirmation").note == moved
    assert run.step("candidates").note == ""
    assert run.step("assessment").note == "", "said once, where it happened"


# --------------------------------------------------- step i: synthesis


def profile_reply(system: str, user: str) -> str:
    """Confirm, rate nothing, and write a profile citing ``Q1``."""
    if "Items you may cite" in user:
        return json.dumps({
            "strengths": [{"text": "Checks what others mean.",
                           "cites": ["Q1"]},
                          {"text": "Made up.", "cites": ["Q99"]}],
            "fit": [{"text": "Useful to coordinate.", "cites": ["Q1"]}],
            "questions": ["How do you follow up?"],
        })
    return reply(system, user)


def synthesis_build(
    store: InMemoryAnalysisStore,
    model: Any,
    mode: Any = "claude_code",
    limit: int = 100,
    asked: Optional[List[bool]] = None,
    own: Optional[int] = None,
    transcripts: Optional[List[Transcript]] = None,
) -> SessionAnalysis:
    """The flow with a budget shared by steps f, g and i."""
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), FakeEmbedder(dim=64), min_words=4
    )

    def judges() -> JudgeSetup:
        """One budget over ``model`` for every judge call."""
        budget = CallBudget(model, limit)

        def writer(use_claude: bool) -> ProfileSynthesizer:
            """Record the leave to use Claude; same judge."""
            if asked is not None:
                asked.append(use_claude)
            if own is not None:
                return ProfileSynthesizer(CallBudget(model, own))
            return ProfileSynthesizer(budget.using(model))

        return JudgeSetup(
            mode, TagConfirmer(budget),
            AssessmentService(retriever, budget, min_incidents=1),
            budget=budget, synthesis=writer,
        )

    return build(model, store, judges=judges, transcripts=transcripts)


def test_synthesis_json_carries_the_opinion_contract_of_the_type() -> None:
    """Contract v2 end to end: type, overall, skills, top lists, percent."""
    store = InMemoryAnalysisStore()
    transcript = make_transcript(session_id="s1").model_copy(
        update={"conversation_type": "coaching"}
    )
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model, transcripts=[transcript]).run("s1")
    stored = json.loads(store.files[("s1", "synthesis.json")])
    assert stored["format_version"] == 2
    assert stored["conversation_type"] == "coaching"
    prompts = [u for _, u, _ in model.calls if "Items you may cite" in u]
    assert prompts and all("Conversation type: coaching" in u for u in prompts)
    for person in stored["people"]:
        assert person["format_version"] == 2
        assert person["conversation_type"] == "coaching"
        overall = person["overall"]
        assert set(overall) >= {
            "score", "maximum", "minimum", "percent", "verdict", "line",
            "frame", "confidence", "caveats", "source",
        }
        assert overall["percent"] == round(
            overall["score"] / overall["maximum"] * 100, 1
        )
        assert "acquired" in overall["frame"]
        for skill in person["skills"]:
            assert set(skill) >= {
                "competency_id", "name", "score", "verdict", "conclusion",
                "basis", "confidence", "caveats", "source", "cites",
            }


def test_step_i_writes_checked_profiles_and_puts_them_in_the_reports() -> (
    None
):
    """One call per person with quotes, counted; reports get «Synthesis»."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    run = synthesis_build(store, model).run(
        "s1", job=JobSought(text=" Shift lead \n\n tasks ")
    )
    assert run.status == "done"
    step = run.step("synthesis")
    assert step.status == "done" and step.judge == "claude_code"
    assert step.model == "fake-judge" and step.note == ""
    people = step.counts["people"]
    assert people >= 1 and step.counts["judge_calls"] == people
    assert step.counts["dropped"] == people
    assert step.counts["points"] == 2 * people
    stored = json.loads(store.files[("s1", "synthesis.json")])
    assert stored["job"] == "Shift lead\n tasks"
    assert stored["judge"]["models"] == ["fake-judge"]
    assert stored["format_version"] == 2
    assert "conversation_type" in stored
    for person in stored["people"]:
        assert person["overall"]["verdict"] in ("strong", "adequate", "weak")
        assert len(person["skills"]) >= 1
        assert len(person["top_strengths"]) <= 3
        assert len(person["top_improvements"]) <= 2
    done = [p for p in stored["people"] if p["status"] == "done"]
    assert done[0]["strengths"][0]["cites"][0]["id"] == "Q1"
    assert done[0]["dropped"] == {"no_citation": 1}
    page = store.files[("s1", "report-1.html")]
    assert "Synthesis" in page and "Job sought: Shift lead" in page
    assert page.index("Synthesis") < page.index("Summary")
    prompts = [u for _, u, _ in model.calls if "Items you may cite" in u]
    assert len(prompts) == people and "Shift lead" in prompts[0]
    analysis = AnalysisResults(store)
    assert analysis.job("s1") == JobSought(text="Shift lead\n tasks")
    assert analysis.result("s1", "synthesis")["session_id"] == "s1"


def test_a_new_job_reruns_only_the_synthesis() -> None:
    """Same job: nothing to do; another one: only step i; a restart
    keeps the job."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model).run("s1", job=JobSought(text="A"))
    calls = len(model.calls)
    same = synthesis_build(store, model).run("s1", job=JobSought(text="A"))
    assert same.status == "done" and len(model.calls) == calls
    seen: List[StepRecord] = []
    synthesis_build(store, model).run(
        "s1", job=JobSought(text="B"), on_step=seen.append
    )
    assert [r.name for r in seen] == ["synthesis"]
    assert json.loads(store.files[("s1", "synthesis.json")])["job"] == "B"
    asked: List[bool] = []
    synthesis_build(store, model, asked=asked).run(
        "s1", job=JobSought(text="B", use_claude=True)
    )
    assert asked == [True]
    synthesis_build(store, model).run("s1", restart=True)
    assert json.loads(store.files[("s1", "job.json")])["text"] == "B"
    assert json.loads(store.files[("s1", "synthesis.json")])["job"] == "B"


def test_reports_alone_keep_the_stored_profiles() -> None:
    """Rerunning step h shows the profiles of the last step i."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model).run("s1")
    store.files.pop(("s1", "report-1.html"))
    synthesis_build(store, model).run("s1", only=["reports"])
    page = store.files[("s1", "report-1.html")]
    assert "What kinds of tasks they look useful for" in page


@pytest.mark.parametrize("steps", [
    ["assessment", "reports"], ["confirmation"],
    ["confirmation", "assessment", "reports"],
    ["acts"], ["events"], ["acts", "events"],
])
def test_rerunning_d_e_f_or_g_sets_the_profiles_back_to_pending(
    steps: List[str],
) -> None:
    """Profiles of earlier inputs are never shown (#115, #121)."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model).run("s1")
    run = synthesis_build(store, model).run("s1", only=steps)
    assert run.step("synthesis").status == "pending"
    assert run.status == "partial"
    if "reports" in steps:
        page = store.files[("s1", "report-1.html")]
        assert "What kinds of tasks they look useful for" not in page
    with pytest.raises(NotFoundError):
        AnalysisResults(store).result("s1", "synthesis")
    again = synthesis_build(store, model).run("s1")
    assert again.status == "done"
    assert again.step("synthesis").status == "done"


def test_rerunning_other_steps_keeps_the_profiles() -> None:
    """Steps that do not feed the profiles leave step i done."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model).run("s1")
    run = synthesis_build(store, model).run("s1", only=["interaction"])
    assert run.step("synthesis").status == "done" and run.status == "done"


def test_a_failing_model_skips_the_person_and_the_budget_fails_the_step(
) -> None:
    """A model error is a reason per person; the call limit fails step i."""
    store = InMemoryAnalysisStore()

    def down(system: str, user: str) -> str:
        """Fail on the profiles only."""
        if "Items you may cite" in user:
            raise ModelError("Claude Code exited with code 1.")
        return reply(system, user)

    run = synthesis_build(store, FakeChatModel(responder=down)).run("s1")
    step = run.step("synthesis")
    assert step.status == "done" and step.counts["people"] == 0
    stored = json.loads(store.files[("s1", "synthesis.json")])
    reasons = {p["reason"] for p in stored["people"]}
    assert "Claude Code exited with code 1." in reasons
    tight = synthesis_build(
        InMemoryAnalysisStore(), FakeChatModel(responder=profile_reply),
        own=0,
    ).run("s1")
    assert tight.step("synthesis").status == "failed"
    assert "max_judge_calls" in tight.step("synthesis").error


def test_a_local_run_without_leave_says_the_local_model_wrote_it() -> None:
    """The note tells that nothing went to Claude."""
    asked: List[bool] = []
    run = synthesis_build(
        InMemoryAnalysisStore(), FakeChatModel(responder=profile_reply),
        mode="local", asked=asked,
    ).run("s1")
    assert asked == [False]
    assert "local model wrote it" in run.step("synthesis").note


def test_a_run_stored_before_step_i_resumes_with_only_it() -> None:
    """Old records read with ``synthesis`` pending."""
    data = new_run("s1", T0).model_dump(mode="json")
    data["steps"] = [s for s in data["steps"] if s["name"] != "synthesis"]
    for step in data["steps"]:
        step["status"] = "done"
    old = AnalysisRun.model_validate(data)
    assert old.step("synthesis").status == "pending"
    assert steps_to_run(old) == ["synthesis"]


def test_step_i_needs_the_assessment() -> None:
    """Without step g's result the step fails with the reason."""
    store = InMemoryAnalysisStore()
    model = FakeChatModel(responder=profile_reply)
    synthesis_build(store, model).run("s1")
    store.files.pop(("s1", "assessment.json"))
    run = synthesis_build(store, model).run("s1", only=["synthesis"])
    assert "assessment of step g is missing" in run.step("synthesis").error


def test_a_shared_budget_spends_one_limit() -> None:
    """``using`` wraps another judge but counts in the same limit."""
    budget = CallBudget(FakeChatModel(model_name="a"), 2)
    shared = budget.using(FakeChatModel(model_name="b"))
    budget.complete("s", "u", 1)
    assert shared.model_name == "b" and shared.calls == 1
    shared.complete("s", "u", 1)
    assert budget.calls == 2 and budget.remaining == 0
    with pytest.raises(LimitError):
        shared.complete("s", "u", 1)
