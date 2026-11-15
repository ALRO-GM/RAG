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
