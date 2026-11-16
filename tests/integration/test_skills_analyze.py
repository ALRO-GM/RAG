"""``skills analyze`` with the real store, rubric and person report.

The container is built with a fake embedder and a fake judge, so the
whole flow runs in-process without a model; the analysis store writes
to a temporary directory.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
)
from src.domain.skills.analysis_run import new_run
from src.skills_bootstrap import SkillsContainer
from tests.fakes import FakeChatModel, FakeEmbedder
from tests.integration.test_act_tagger_adapters import write_card
from tests.integration.test_skills_container import (
    cli,
    settings,
    skills_project,
)
from tests.unit.test_skills_session_analysis import T0, reply

__all__ = ["skills_project"]


def container(**overrides: Any) -> SkillsContainer:
    """Return a container whose fake judge answers both kinds of prompt."""
    return SkillsContainer(
        settings(**overrides), embedder=FakeEmbedder(dim=256),
        judge=FakeChatModel(responder=reply),
    )


def test_store_writes_reads_and_refuses_unsafe_names(tmp_path: Path) -> None:
    """Files go under directory/session; ids and names are checked."""
    store = FileSystemAnalysisStore(str(tmp_path / "analysis"))
    assert store.load("s1") is None
    store.save(new_run("s1", T0))
    loaded = store.load("s1")
    assert loaded is not None and loaded.session_id == "s1"
    path = store.write("s1", "report-1.html", "<p>hola</p>")
    assert Path(path) == tmp_path / "analysis" / "s1" / "report-1.html"
    assert store.read("s1", "report-1.html") == "<p>hola</p>"
    assert not list((tmp_path / "analysis" / "s1").glob(".*"))
    for session, name in (
        ("../x", "run.json"), ("s1", "../run.json"), ("s1", "a/b.json"),
        ("s1", ".hidden.json"), ("s1", "run.exe"), ("s1", "RUN.json"),
    ):
        with pytest.raises(InvalidInputError):
            store.read(session, name)
    with pytest.raises(NotFoundError):
        store.read("s1", "summary.json")
    (tmp_path / "analysis" / "s1" / "run.json").write_text("{}")
    with pytest.raises(CorruptedDataError):
        store.load("s1")


def test_analyze_runs_every_step_and_prints_the_table(
    skills_project: Path, capsys: Any
) -> None:
    """The CLI prints each step, the table and what people must do."""
    app = container(analysis_dir="analysis", max_confirmations=3)
    cli(app).analyze(session="s1", synthetic=True)
    out = capsys.readouterr().out
    assert "Analysing s1" in out
    assert "acts          unavailable" in out
    assert "Not available: no dialogue-act tagger" in out
    assert "Still to be done by people:" in out
    assert "#19" in out and "#84" in out
    folder = skills_project / "analysis" / "s1"
    run = json.loads((folder / "run.json").read_text())
    assert run["status"] == "done" and run["synthetic"] is True
    assert run["steps"][5]["counts"]["sent"] <= 3
    summary = json.loads((folder / "summary.json").read_text())
    assert len(summary["people"]) == 3
    assert (folder / "report-1.html").read_text().startswith("<!")
    assert (folder / "scorecard.md").is_file()


def test_analyze_reports_a_failed_step_and_how_to_resume(
    skills_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing judge ends the command with the step and its reason."""
    app = container(analysis_dir="analysis")

    def broken(system: str, user: str) -> str:
        """A judge that cannot answer."""
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(app._shared_judge, "responder", broken)
    with pytest.raises(InvalidInputError, match="confirmation failed: "
                       "judge exploded. Run the same command again"):
        cli(app).analyze(session="s1")
    run = json.loads(
        (skills_project / "analysis" / "s1" / "run.json").read_text()
    )
    assert [s["status"] for s in run["steps"]][:6] == [
        "done", "done", "done", "unavailable", "unavailable", "failed",
    ]


def test_analyze_unknown_session(skills_project: Path) -> None:
    """No transcript for the id: a clean not-found error."""
    with pytest.raises(NotFoundError):
        cli(container(analysis_dir="analysis")).analyze(session="nope")


DEMO_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples" / "skills" / "simulated" / "reunion_verano.txt"
)


def test_simulate_then_analyze_the_demo_discussion_with_fakes(
    skills_project: Path, capsys: Any
) -> None:
    """The make target's two commands, end to end with a fake judge."""
    commands = cli(container(analysis_dir="analysis", max_confirmations=40))
    commands.simulate(script=str(DEMO_SCRIPT))
    out = capsys.readouterr().out
    assert "sim-reunion-verano: 4 speakers" in out
    written = skills_project / "transcripts" / (
        "sim-reunion-verano.transcript.v1.json"
    )
    assert written.is_file()
    commands.analyze(session="sim-reunion-verano", synthetic=True)
    folder = skills_project / "analysis" / "sim-reunion-verano"
    run = json.loads((folder / "run.json").read_text())
    assert run["status"] == "done"
    counts = {s["name"]: s["counts"] for s in run["steps"]}
    assert counts["sentences"] == {"sentences": 238, "speakers": 4}
    assert counts["confirmation"]["sent"] == 40
    summary = json.loads((folder / "summary.json").read_text())
    assert [p["name"] for p in summary["people"]] == [
        "Lucía", "Andrés", "Nerea", "Joaquín",
    ]


def test_simulate_errors(skills_project: Path) -> None:
    """A missing or broken script is a clean error."""
    commands = cli(container())
    with pytest.raises(NotFoundError):
        commands.simulate(script="missing.txt")
    bad = skills_project / "bad.txt"
    bad.write_text("id: x\nAna: hola\nAna: otra\n", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="in a row"):
        commands.simulate(script=str(bad))


def test_the_tagger_card_lets_acts_and_events_run(
    skills_project: Path,
) -> None:
    """With a tagger card and no weights the rules answer steps d and e."""
    card = write_card(skills_project / "card.json", encoder="no-weights")
    app = container(analysis_dir="analysis", act_card=str(card))
    cli(app).analyze(session="s1")
    folder = skills_project / "analysis" / "s1"
    run = json.loads((folder / "run.json").read_text())
    steps = {s["name"]: s for s in run["steps"]}
    assert steps["acts"]["status"] == "done"
    assert steps["acts"]["model"].startswith("rules (")
    assert "the rules answered" in steps["acts"]["note"]
    assert steps["acts"]["counts"]["sentences"] == 8
    assert "doubtful" in steps["acts"]["counts"]
    assert steps["events"]["status"] == "done"
    acts = json.loads((folder / "acts.json").read_text())
    assert acts["schema"] == "acts.v1"
    assert acts["tagger"]["name"] == "rules"
    events = json.loads((folder / "events.json").read_text())
    assert events["session_id"] == "s1"


def test_without_acts_the_events_step_says_why(skills_project: Path) -> None:
    """The extractor exists but has no acts to read."""
    app = container(analysis_dir="analysis")
    cli(app).analyze(session="s1")
    run = json.loads(
        (skills_project / "analysis" / "s1" / "run.json").read_text()
    )
    events = run["steps"][4]
    assert events["status"] == "unavailable"
    assert "step d" in events["note"]


def test_numeric_looking_ids_stay_text(skills_project: Path) -> None:
    """Fire turns 1e5 into 100000.0: refused; the quoted id works."""
    (skills_project / "transcripts" / "1e5.txt").write_text(
        (skills_project / "transcripts" / "s1.txt").read_text()
    )
    commands = cli(container(analysis_dir="analysis"))
    with pytest.raises(InvalidInputError, match="must be quoted"):
        commands.analyze(session=100000.0)
    commands.analyze(session="1e5")
    assert (skills_project / "analysis" / "1e5" / "run.json").is_file()


def test_the_cli_parses_a_quoted_numeric_id(skills_project: Path) -> None:
    """Through Fire, as the launcher passes it: ``--session="1e5"``."""
    from tests.integration.test_cli import run_cli

    result = run_cli(
        ["skills", "--transcripts_dir", "transcripts", "--analysis_dir",
         "analysis", "analyze", '--session="0x1f"'], skills_project,
    )
    assert "No transcript for session '0x1f'" in result.stderr
    bare = run_cli(
        ["skills", "--analysis_dir", "analysis", "analyze",
         "--session=1_0"], skills_project,
    )
    assert "must be quoted" in bare.stderr


def test_rerunning_assessment_and_reports_adds_the_provisional_tier(
    skills_project: Path,
) -> None:
    """``--steps assessment,reports`` over an old analysis (#103).

    The fake judge rates nothing, so nothing is firm; step f's
    confirmations become provisional estimates and observations, and
    an assessment written before them (format 1) still makes reports.
    """
    commands = cli(container(analysis_dir="analysis", max_confirmations=5))
    commands.analyze(session="s1", synthetic=True)
    folder = skills_project / "analysis" / "s1"
    stored = json.loads((folder / "assessment.json").read_text())
    for card in stored["scorecards"]:
        for item in card["assessments"]:
            for key in ("estimate", "observations"):
                item.pop(key, None)
            item["status"] = "insufficient_evidence"
    stored.pop("format_version")
    (folder / "assessment.json").write_text(json.dumps(stored))
    commands.analyze(session="s1", steps="reports")
    old = json.loads((folder / "summary.json").read_text())
    assert all(p["provisional"] == 0 for p in old["people"])
    commands.analyze(session="s1", steps="assessment,reports")
    fresh = json.loads((folder / "assessment.json").read_text())
    assert fresh["format_version"] == 2
    summary = json.loads((folder / "summary.json").read_text())
    assert sum(p["provisional"] for p in summary["people"]) > 0
    assert sum(p["observations"] for p in summary["people"]) > 0
    assert all(p["rated"] == 0 and p["why"] for p in summary["people"])
    page = (folder / "report-1.html").read_text()
    assert "Provisional estimates and observations" in page
    assert "provisional ~" in (folder / "scorecard.md").read_text()


def test_the_role_reaches_the_synthesis_and_reruns_only_it(
    skills_project: Path, capsys: Any
) -> None:
    """``--role`` text or file; a new one makes the next run redo step i."""
    from tests.unit.test_skills_session_analysis import profile_reply

    app = SkillsContainer(
        settings(analysis_dir="analysis"), embedder=FakeEmbedder(dim=256),
        judge=FakeChatModel(responder=profile_reply),
    )
    role = skills_project / "role.txt"
    role.write_text("Shift lead\nOrganises the morning team.\n")
    cli(app).analyze(session="s1", role=str(role))
    folder = skills_project / "analysis" / "s1"
    run = json.loads((folder / "run.json").read_text())
    assert run["steps"][-1]["name"] == "synthesis"
    assert run["steps"][-1]["status"] == "done"
    synthesis = json.loads((folder / "synthesis.json").read_text())
    assert synthesis["job"] == "Shift lead\nOrganises the morning team."
    assert "Job sought: Shift lead" in (folder / "report-1.html").read_text()
    capsys.readouterr()
    cli(app).analyze(session="s1", role="Another job",
                     claude_synthesis=True)
    out = capsys.readouterr().out
    assert "synthesis     done" in out and "  reports  " not in out
    job = json.loads((folder / "job.json").read_text())
    assert job == {"text": "Another job", "use_claude": True}


def test_role_and_leave_merge_with_the_stored_job(
    skills_project: Path, capsys: Any
) -> None:
    """One option never erases the other one stored (#115)."""
    from tests.unit.test_skills_session_analysis import profile_reply

    app = SkillsContainer(
        settings(analysis_dir="analysis"), embedder=FakeEmbedder(dim=256),
        judge=FakeChatModel(responder=profile_reply),
    )
    folder = skills_project / "analysis" / "s1"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps({"text": "Shift lead", "use_claude": True})
    )
    cli(app).analyze(session="s1", claude_synthesis=False)
    job = json.loads((folder / "job.json").read_text())
    assert job == {"text": "Shift lead", "use_claude": False}
    cli(app).analyze(session="s1", claude_synthesis=True)
    cli(app).analyze(session="s1", role="Store manager")
    job = json.loads((folder / "job.json").read_text())
    assert job == {"text": "Store manager", "use_claude": True}
    cli(app).analyze(session="s1")
    job = json.loads((folder / "job.json").read_text())
    assert job == {"text": "Store manager", "use_claude": True}
    role = skills_project / "long-role.txt"
    role.write_text("x" * 4001)
    with pytest.raises(InvalidInputError, match="at most 4000"):
        cli(app).analyze(session="s1", role=str(role))
    job = json.loads((folder / "job.json").read_text())
    assert job == {"text": "Store manager", "use_claude": True}
    capsys.readouterr()


def test_the_synthesis_judge_is_claude_code_unless_told_otherwise(
    skills_project: Path,
) -> None:
    """Local without leave: the local model; otherwise Claude Code, never
    the paid API; one budget for the whole run."""
    from src.application.skills.judges import CallBudget, FallbackJudge

    def judge_for(judge: str, use_claude: bool) -> Any:
        """The synthesis judge of a container with ``judge``."""
        app = SkillsContainer(settings(judge=judge))
        local = FakeChatModel(model_name="local")
        app.__dict__["local_judge"] = local
        app.__dict__["judge"] = (
            local if judge == "local" else FakeChatModel(model_name="api")
        )
        budget = CallBudget(app.judge, 5)
        chosen = app.synthesis_judge(budget, use_claude)
        assert isinstance(chosen, CallBudget)
        return chosen, budget

    chosen, budget = judge_for("local", False)
    assert chosen is budget and chosen.model_name == "local"
    chosen, budget = judge_for("local", True)
    assert isinstance(chosen.judge, FallbackJudge)
    assert chosen.judge.primary.model_name.startswith("claude-code:")
    chosen, budget = judge_for("api", False)
    assert isinstance(chosen.judge, FallbackJudge)
    chosen, budget = judge_for("claude_code", False)
    assert chosen is budget
    shared = SkillsContainer(settings(), judge=FakeChatModel())
    budget = CallBudget(shared.judge, 1)
    assert shared.synthesis_judge(budget, True) is budget
    setup = shared.judge_setup()
    assert setup.synthesis is not None
    assert setup.synthesis(False).judge is setup.budget
