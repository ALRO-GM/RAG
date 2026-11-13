"""Benchmark CLI helpers: configuration choice, resume, stored rows."""

import json
from pathlib import Path

import pytest

from src.adapters.inbound.cli import (
    completed_runs,
    read_benchmark_rows,
    remove_tree,
    run_key,
    select_configs,
)
from src.domain.errors import InvalidInputError
from tests.integration.test_cli import run_cli

ROW = {
    "dataset": "scifact", "config": "a", "params": {"x": 1},
    "suite": "s", "documents": 1, "chunks": 1, "queries": 1,
    "ndcg10": 0.5, "recall10": 0.5, "recall100": 0.5, "hit": {"1": 0.5},
    "index_seconds": 1.0, "index_cold": False, "search_seconds": 0.1,
}


@pytest.mark.parametrize("wanted, expected", [
    ("", ["hybrid a", "bm25 b", "Hybrid C"]),
    ("hybrid", ["hybrid a", "Hybrid C"]),
    ("bm25, c", ["bm25 b", "Hybrid C"]),
    (("bm25",), ["bm25 b"]),
    (None, ["hybrid a", "bm25 b", "Hybrid C"]),
])
def test_select_configs(wanted: object, expected: list[str]) -> None:
    available = {"hybrid a": {"p": 1}, "bm25 b": {}, "Hybrid C": {}}
    chosen = select_configs(available, wanted)
    assert list(chosen) == expected
    if "hybrid a" in chosen:
        assert chosen["hybrid a"] == {"p": 1}
        assert chosen["hybrid a"] is not available["hybrid a"]


def test_run_key_depends_on_every_part() -> None:
    keys = {
        run_key("d", "s", "c", {"x": 1}),
        run_key("e", "s", "c", {"x": 1}),
        run_key("d", "t", "c", {"x": 1}),
        run_key("d", "s", "k", {"x": 1}),
        run_key("d", "s", "c", {"x": 2}),
    }
    assert len(keys) == 5
    assert run_key("d", "s", "c", {"x": 1}) == run_key("d", "s", "c",
                                                       {"x": 1})


def test_completed_runs_reads_only_the_suite(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    other = {**ROW, "suite": "t", "config": "b"}
    builtin = {**ROW, "suite": "", "config": "c"}
    path.write_text(
        "\n".join([json.dumps(ROW), json.dumps(other), json.dumps(builtin),
                   "not json", "[1]", ""]),
        encoding="utf-8",
    )
    assert completed_runs(path, "s") == {
        run_key("scifact", "s", "a", {"x": 1}),
    }


def test_completed_runs_without_suite_or_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert completed_runs(path, "s") == set()
    path.write_text(json.dumps({**ROW, "suite": ""}), encoding="utf-8")
    assert completed_runs(path, "") == set()


def test_read_benchmark_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert read_benchmark_rows(path) == []
    path.write_text(json.dumps(ROW) + "\n\n", encoding="utf-8")
    rows = read_benchmark_rows(path)
    assert len(rows) == 1 and rows[0].suite == "s"
    assert rows[0].hit == {1: 0.5}


def test_read_benchmark_rows_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(ROW) + "\n{\"dataset\": 1}\n",
                    encoding="utf-8")
    with pytest.raises(InvalidInputError, match="rows.jsonl:2"):
        read_benchmark_rows(path)


def test_remove_tree(tmp_path: Path) -> None:
    target = tmp_path / "work" / "indexes"
    (target / "a").mkdir(parents=True)
    (target / "a" / "f").write_text("x")
    remove_tree(target)
    assert not target.exists()
    remove_tree(target)  # missing: no error


def test_general_command_combines_lab_and_benchmark_rows(
    project: Path,
) -> None:
    """``general`` ranks a suite from stored trials and benchmark rows."""
    suite = {
        "name": "g", "base": {}, "strategy": "list",
        "trials": [{"name": "more semantic",
                    "params": {"semantic_weight": 1.0}}],
    }
    (project / "experiments" / "suites").mkdir(parents=True)
    (project / "experiments" / "suites" / "g.json").write_text(
        json.dumps(suite)
    )

    def trial(name: str, docs: float) -> str:
        return json.dumps({
            "trial_id": name, "suite": "g", "name": name, "params": {},
            "baseline_id": "baseline", "ranks": {},
            "metrics": {"docs": {"recall@5": docs}},
            "search_seconds_per_100": 1.0, "index_seconds": 2.0,
            "index_cold": False, "n_chunks": 3,
        })

    results = project / "experiments" / "results"
    results.mkdir(parents=True)
    (results / "trials.jsonl").write_text(
        trial("baseline", 0.5) + "\n" + trial("more semantic", 0.55) + "\n"
    )
    rows = project / "rows.jsonl"
    rows.write_text("\n".join(json.dumps({
        **ROW, "suite": "g", "config": config, "ndcg10": score,
    }) for config, score in (("baseline", 0.4), ("more semantic", 0.5))))
    done = run_cli(
        ["general", "--suite", "g", "--benchmarks", str(rows)], project
    )
    assert done.returncode == 0, done.stderr
    assert "g: 2 configurations on 2 datasets (42:docs, scifact)" in (
        done.stdout
    )
    assert "Best mean: more semantic; robust: more semantic" in done.stdout
    report = (project / "experiments" / "general" / "g.md").read_text()
    assert "more semantic **(robust)** **(best mean)**" in report
    assert "+25.0%" in report and "+10.0%" in report


def test_general_command_without_results_is_a_clean_error(
    project: Path,
) -> None:
    """A suite that was never run has no baseline to compare with."""
    (project / "experiments" / "suites").mkdir(parents=True)
    (project / "experiments" / "suites" / "g.json").write_text(
        json.dumps({"name": "g", "base": {}, "strategy": "list"})
    )
    done = run_cli(["general", "--suite", "g"], project)
    assert done.returncode == 1
    assert done.stderr.startswith("Error: ")
    assert "no measured baseline" in done.stderr
