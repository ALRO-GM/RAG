"""The experiment lab commands run as ``python -m src lab ...``."""

import json
from pathlib import Path
from typing import List

import pytest

from src.domain.experiments import TrialResult
from src.domain.priors import RouterModel
from tests.integration.test_cli import run_cli, source
from tests.samples import (
    MD_PATH,
    PY_PATH,
    TXT_PATH,
    write_corpus,
    write_json,
)

DOCS = "data/datasets/dataset_docs_public.json"
CODE = "data/datasets/dataset_code_public.json"
DATASETS = f"--datasets={DOCS},{CODE}"
LEXICAL = {"semantic": False, "mode": "bm25", "max_chunk_size": 300}
TRIALS = "experiments/results/trials.jsonl"


def question(text: str, path: str, last: int) -> dict[str, object]:
    """Return an answered question as JSON-ready data."""
    return {"question_id": text, "question": text, "answer": "x",
            "sources": [source(path, 0, last)]}


def write_lab_files(project: Path) -> None:
    """Write the corpus, two datasets and lexical-only suites."""
    write_corpus(project)
    write_json(project / DOCS, {"rag_questions": [
        question("How do you load adapters at runtime?", MD_PATH, 120),
        question("How to enable FP8 quantization?", MD_PATH, 200),
        question("What does speculative decoding use?", TXT_PATH, 58),
        question("How does the target model verify tokens?", TXT_PATH, 120),
        {"question_id": "open", "question": "unanswered"},
    ]})
    write_json(project / CODE, {"rag_questions": [
        question("What does load_lora_adapter return?", PY_PATH, 200),
        question("What is the type of n in allocate_blocks?", PY_PATH, 330),
        question("Which class has the allocate_blocks method?", PY_PATH, 330),
        question("What is the name argument of load_lora_adapter?",
                 PY_PATH, 200),
    ]})
    suite = {
        "name": "lex", "description": "search-time priors, lexical only",
        "base": LEXICAL,
        "factors": {"file_boost": [0.5], "test_weight": [0.5]},
    }
    write_json(project / "suites/lex.json", suite)
    write_json(project / "experiments/suites/lex.json", suite)
    write_json(project / "suites/unknown.json", {
        "name": "unknown", "base": {**LEXICAL, "nope": 1},
    })
    write_json(project / "suites/invalid.json", {
        "name": "invalid", "base": {**LEXICAL, "rerank_depth": 0},
    })
    write_json(project / "suites/nofactor.json", {
        "name": "nofactor", "base": LEXICAL,
    })
    write_json(project / "suites/wrong.json", {"strategy": "random"})
    (project / "suites/bad.json").write_text("{not valid json")
    write_json(project / "open.json", {"rag_questions": [
        {"question_id": "open", "question": "unanswered"},
    ]})


def trial_lines(project: Path) -> List[TrialResult]:
    """Return every line of the trial store."""
    text = (project / TRIALS).read_text(encoding="utf-8")
    return [
        TrialResult.model_validate_json(line) for line in text.splitlines()
    ]


@pytest.fixture(scope="module")
def lab_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the lexical suite once on the tiny corpus.

    Shared by the module: the other tests resume, analyse or fail
    without adding any trial to the store.
    """
    project = tmp_path_factory.mktemp("lab")
    write_lab_files(project)
    result = run_cli(
        ["lab", "experiment", "--suite=suites/lex.json", DATASETS], project
    )
    assert result.returncode == 0, result.stderr
    assert "Suite lex: 3 trials" in result.stdout
    for name in ("baseline", "file_boost=0.5", "test_weight=0.5"):
        assert f"  {name} " in result.stdout
    assert "R@5 docs=1.000 code=1.000 (+0.0)" in result.stdout
    assert "s/100q" in result.stdout
    return project


def test_experiment_stores_one_result_per_trial(lab_project: Path) -> None:
    """Results go to experiments/results, indexes to data/experiments."""
    trials = trial_lines(lab_project)
    assert [t.name for t in trials] == [
        "baseline", "file_boost=0.5", "test_weight=0.5",
    ]
    assert {t.suite for t in trials} == {"lex"}
    assert {t.baseline_id for t in trials} == {trials[0].trial_id}
    assert trials[1].params == {**LEXICAL, "file_boost": 0.5}
    assert set(trials[0].ranks) == {"docs", "code"}
    assert [len(r) for r in trials[0].ranks.values()] == [4, 4]
    assert [t.index_cold for t in trials] == [True, False, False]
    indexes = list((lab_project / "data/experiments/indexes").iterdir())
    assert len(indexes) == 1, "search-only factors share one index"
    assert not (lab_project / "data/processed").exists()


def test_experiment_resumes_by_suite_name(lab_project: Path) -> None:
    """A suite found by name reuses the stored trials."""
    before = trial_lines(lab_project)
    result = run_cli(["lab", "experiment", "--suite=lex", DATASETS],
                     lab_project)
    assert result.returncode == 0, result.stderr
    assert "Suite lex: 3 trials" in result.stdout
    assert trial_lines(lab_project) == before


def test_analyze_writes_the_report_and_the_profiles(
    lab_project: Path
) -> None:
    """The Markdown report and one JSON file per profile are written."""
    result = run_cli(["lab", "analyze", "--output=out/report.md"],
                     lab_project)
    assert result.returncode == 0, result.stderr
    assert "Analysed 3 trials → out/report.md" in result.stdout
    report = (lab_project / "out/report.md").read_text(encoding="utf-8")
    assert report.startswith("# Experiment report")
    assert "### lex" in report
    assert "R@5 code | R@5 docs" in report
    for change in ("| baseline |", "`file_boost=0.5`", "`test_weight=0.5`"):
        assert change in report
    for name in ("precision", "balanced", "speed"):
        assert f"  {name}" in result.stdout
        assert f"| **{name}** |" in report
        profile = json.loads(
            (lab_project / f"experiments/profiles/{name}.json").read_text()
        )
        assert profile.items() >= LEXICAL.items()


def test_analyze_default_output_and_profile_reuse(lab_project: Path) -> None:
    """Without --output the report is experiments/REPORT.md; a profile
    written by the lab configures the normal commands."""
    result = run_cli(["lab", "analyze"], lab_project)
    assert result.returncode == 0, result.stderr
    assert "→ experiments/REPORT.md" in result.stdout
    assert (lab_project / "experiments/REPORT.md").is_file()
    index_dir = "--index_dir=data/experiments/main"
    index = run_cli(["index", "--semantic=False", "--max_chunk_size=300",
                     index_dir, "--profile=speed"], lab_project)
    assert index.returncode == 0, index.stderr
    search = run_cli(["search", "load_lora_adapter", "--k=2", index_dir,
                      "--profile=speed"], lab_project)
    assert search.returncode == 0, search.stderr
    out = json.loads(search.stdout)
    assert out["mode"] == "bm25"
    assert out["retrieved_sources"][0]["file_path"] == PY_PATH


@pytest.mark.parametrize("args, message", [
    (["experiment", "--suite=missing", DATASETS], "Suite not found: missing"),
    (["experiment", "--suite=suites/none.json", DATASETS],
     "Suite not found"),
    (["experiment", "--suite=suites/bad.json", DATASETS], "Invalid suite"),
    (["experiment", "--suite=suites/wrong.json", DATASETS], "Invalid suite"),
    (["experiment", "--suite=suites/unknown.json", DATASETS],
     "Unknown parameter(s): nope."),
    (["experiment", "--suite=suites/invalid.json", DATASETS],
     "Invalid parameters"),
    (["experiment", "--suite=lex", "--datasets=missing.json"],
     "File not found"),
    (["experiment", "--suite=lex", "--datasets=suites/bad.json"],
     "Malformed JSON"),
    (["experiment", "--suite=lex", "--datasets=open.json"],
     "no question with reference sources"),
    (["experiment", "--suite=lex", "--datasets=,"], "--datasets must list"),
    (["tune", "--suite=lex", DATASETS, "--folds=1"],
     "folds must be at least 2"),
    (["tune", "--suite=suites/nofactor.json", DATASETS], "no factor to tune"),
    (["tune", "--suite=missing", DATASETS], "Suite not found"),
    (["train_router", f"--datasets={CODE}"], "Router training needs"),
    (["train_router", "--datasets=missing.json"], "File not found"),
])
def test_lab_degenerate_inputs(
    lab_project: Path, args: List[str], message: str
) -> None:
    """Bad lab inputs exit 1 with a clean message and store nothing."""
    before = (lab_project / TRIALS).read_text()
    result = run_cli(["lab", *args], lab_project)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr + result.stdout
    assert "Error: " in result.stderr
    assert message in result.stderr
    assert (lab_project / TRIALS).read_text() == before


def test_analyze_without_results_is_a_clean_error(tmp_path: Path) -> None:
    """Analysing before any experiment tells what to run first."""
    result = run_cli(["lab", "analyze"], tmp_path)
    assert result.returncode == 1
    assert result.stderr.strip() == (
        "Error: No experiment results yet. Run 'lab experiment' first."
    )
    assert not (tmp_path / "experiments").exists()


def test_experiment_without_resume_measures_again(tmp_path: Path) -> None:
    """--resume=False appends new results for the same trials."""
    write_lab_files(tmp_path)
    args = ["lab", "experiment", "--suite=lex", f"--datasets={CODE}"]
    assert run_cli(args, tmp_path).returncode == 0
    assert run_cli(args, tmp_path).returncode == 0
    assert len(trial_lines(tmp_path)) == 3
    again = run_cli([*args, "--resume=False"], tmp_path)
    assert again.returncode == 0, again.stderr
    trials = trial_lines(tmp_path)
    assert len(trials) == 6
    assert [t.trial_id for t in trials[3:]] == [
        t.trial_id for t in trials[:3]
    ]
    assert "R@5 code=1.000" in again.stdout


def test_tune_writes_the_report_and_the_profile(tmp_path: Path) -> None:
    """Tuning stores its trials, a JSON report and the tuned profile."""
    write_lab_files(tmp_path)
    result = run_cli([
        "lab", "tune", "--suite=lex", DATASETS, "--folds=2", "--passes=1",
        "--save_profile=out/tuned.json",
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Tuned lex (" in result.stdout
    assert "objective held-out:" in result.stdout
    assert "profile saved to out/tuned.json" in result.stdout
    tuned = json.loads((tmp_path / "out/tuned.json").read_text())
    assert tuned == {**LEXICAL, "file_boost": 0.5, "test_weight": 0.5}
    report = json.loads(
        (tmp_path / "experiments/results/tuning_lex.json").read_text()
    )
    assert report["best_params"] == tuned
    assert len(report["folds"]) == 2
    assert report["trials_evaluated"] == 1
    assert {t.suite for t in trial_lines(tmp_path)} == {"lex/tune"}


def test_train_router_saves_a_model_used_by_search(tmp_path: Path) -> None:
    """The router is trained on .py and .md/.txt answers and loadable."""
    write_lab_files(tmp_path)
    result = run_cli(["lab", "train_router", DATASETS, "--folds=2"],
                     tmp_path)
    assert result.returncode == 0, result.stderr
    assert ("Router trained on 8 questions → models/query_router.json"
            in result.stdout)
    assert "accuracy: train" in result.stdout
    assert "cross-validated" in result.stdout
    model = RouterModel.model_validate_json(
        (tmp_path / "models/query_router.json").read_text()
    )
    assert model.p_code("What does load_lora_adapter return?") > 0.5
    write_json(tmp_path / "routed.json", {**LEXICAL, "route_strength": 0.5})
    index = run_cli(["index", "--semantic=False", "--max_chunk_size=300"],
                    tmp_path)
    assert index.returncode == 0, index.stderr
    search = run_cli(["search", "What does load_lora_adapter return?",
                      "--profile=routed.json", "--k=3"], tmp_path)
    assert search.returncode == 0, search.stderr
    assert json.loads(search.stdout)["retrieved_sources"][0][
        "file_path"] == PY_PATH
