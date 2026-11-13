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
