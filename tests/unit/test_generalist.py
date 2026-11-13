"""Generalist study: relative changes, robust choice, leave-one-out."""

from typing import Dict, List

import pytest

from src.application.benchmarking import BenchmarkResult
from src.application.generalist import (
    GeneralistAnalysis,
    lab_scores,
    render_generalist,
)
from src.domain.errors import InvalidInputError
from src.domain.experiments import TrialResult


def trial(
    name: str, docs: float, code: float, suite: str = "s"
) -> TrialResult:
    """A lab trial with recall@5 on the two 42 datasets."""
    return TrialResult(
        trial_id=name, suite=suite, name=name, params={"n": name},
        baseline_id="baseline",
        metrics={"docs": {"recall@5": docs, "mrr": 0.1},
                 "code": {"recall@5": code}},
        ranks={}, search_seconds_per_100=1.0, index_seconds=10.0,
        index_cold=False, n_chunks=5,
    )


def row(config: str, dataset: str, ndcg: float,
        suite: str = "s") -> BenchmarkResult:
    """A benchmark row."""
    return BenchmarkResult(
        dataset=dataset, config=config, params={"n": config}, suite=suite,
        documents=10, chunks=10, queries=5, ndcg10=ndcg, recall10=0.5,
        recall100=0.9, hit={1: 0.1}, index_seconds=2.0, index_cold=False,
        search_seconds=0.5,
    )


def study() -> tuple[List[TrialResult], List[BenchmarkResult]]:
    """Baseline, a generalist, a specialist and a partial config."""
    trials = [
        trial("baseline", 0.50, 0.40),
        trial("general", 0.505, 0.41),    # +1%, +2.5%
        trial("specialist", 0.65, 0.52),  # +30%, +30%
        trial("partial", 0.55, 0.45),
        trial("other-suite", 0.1, 0.1, suite="x"),
    ]
    rows = [
        row("baseline", "fiqa", 0.40),
        row("general", "fiqa", 0.42),     # +5%
        row("specialist", "fiqa", 0.30),  # -25%
        row("baseline", "scifact", 0.70),
        row("general", "scifact", 0.72),  # +2.9%
        row("specialist", "scifact", 0.69),
        row("baseline", "fiqa", 0.99, suite="x"),
    ]
    return trials, rows


class TestAnalysis:
    def test_datasets_and_relative_changes(self) -> None:
        trials, rows = study()
        report = GeneralistAnalysis().analyse("s", trials, rows)
        assert report.datasets == ["42:docs", "42:code", "fiqa", "scifact"]
        general = report.config("general")
        assert general.relative["fiqa"] == pytest.approx(0.05)
        assert general.relative["42:code"] == pytest.approx(0.025)
        assert general.complete
        assert general.wins == 4 and general.losses == 0
        assert general.index_seconds == 14.0  # 10 lab + 2 x 2 benchmark
        base = report.config("baseline")
        assert base.mean_relative == 0.0 and base.wins == 0

    def test_partial_configs_are_not_ranked(self) -> None:
        trials, rows = study()
        report = GeneralistAnalysis().analyse("s", trials, rows)
        partial = report.config("partial")
        assert not partial.complete
        assert [c.name for c in report.configs_complete()] == [
            "baseline", "general", "specialist",
        ]

    def test_best_mean_versus_robust_choice(self) -> None:
        """The specialist wins on average but collapses on FiQA."""
        trials, rows = study()
        report = GeneralistAnalysis(max_loss=0.02).analyse("s", trials, rows)
        specialist = report.config("specialist")
        general = report.config("general")
        assert (specialist.mean_relative or 0) > (general.mean_relative or 0)
        assert report.best == "specialist"
        assert report.robust == "general"

    def test_leave_one_dataset_out(self) -> None:
        trials, rows = study()
        report = GeneralistAnalysis().analyse("s", trials, rows)
        held = {h.dataset: h for h in report.held_out}
        assert set(held) == set(report.datasets)
        # Chosen on the rest (where the specialist shines), it fails FiQA.
        fiqa = held["fiqa"]
        assert fiqa.chosen == "specialist"
        assert fiqa.best == "general"
        assert fiqa.rank == 3
        assert fiqa.regret == pytest.approx(0.05 + 0.25)
        # Held-out 42:docs: chosen on the rest = general? specialist?
        assert held["42:docs"].regret >= 0

    def test_suite_order_is_kept(self) -> None:
        trials, rows = study()
        report = GeneralistAnalysis().analyse(
            "s", trials, rows, order=["specialist", "baseline"]
        )
        assert [c.name for c in report.configs][:2] == [
            "specialist", "baseline",
        ]

    def test_baseline_is_required(self) -> None:
        with pytest.raises(InvalidInputError, match="no measured baseline"):
            GeneralistAnalysis().analyse("s", [trial("a", 0.5, 0.5)], [])

    def test_negative_thresholds_are_rejected(self) -> None:
        with pytest.raises(InvalidInputError):
            GeneralistAnalysis(tolerance=-1)

    def test_zero_baseline_scores_are_skipped(self) -> None:
        trials = [trial("baseline", 0.0, 0.5), trial("a", 0.4, 0.6)]
        report = GeneralistAnalysis().analyse("s", trials, [])
        assert "42:docs" not in report.config("a").relative
        assert not report.config("a").complete

    def test_unknown_config_lookup(self) -> None:
        trials, rows = study()
        report = GeneralistAnalysis().analyse("s", trials, rows)
        with pytest.raises(KeyError):
            report.config("nope")

    def test_single_dataset_has_no_leave_one_out(self) -> None:
        report = GeneralistAnalysis().analyse(
            "s", [], [row("baseline", "fiqa", 0.4), row("a", "fiqa", 0.5)]
        )
        assert report.held_out == []
        assert report.best == "a"


def test_lab_scores_prefix_and_missing_metric() -> None:
    t = trial("baseline", 0.9, 0.8)
    t.metrics["other"] = {"mrr": 0.3}
    scores: Dict[str, float] = lab_scores([t])["baseline"][0]
    assert scores == {"42:docs": 0.9, "42:code": 0.8}


def test_render_marks_choices_and_regret() -> None:
    trials, rows = study()
    text = render_generalist(GeneralistAnalysis().analyse("s", trials, rows))
    assert text.startswith("# Generalist study: `s`")
    assert "general **(robust)**" in text
    assert "specialist **(best mean)**" in text
    assert "partial *(partial)*" in text
    assert "## Leave one dataset out" in text
    assert "Mean regret" in text
    assert "Baseline scores: 42:docs 0.500" in text


def test_render_without_choices() -> None:
    report = GeneralistAnalysis().analyse(
        "s", [], [row("baseline", "fiqa", 0.4)]
    )
    text = render_generalist(report)
    assert "## Leave one dataset out" not in text
    assert "Best mean: **baseline**" in text
