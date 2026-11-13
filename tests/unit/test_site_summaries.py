"""Unit tests: view models of the results site (pure functions)."""

from typing import Any, Dict

import pytest

from results_site.summaries import (
    SiteError,
    changed_params,
    doc_title,
    headline,
    mean,
    parse_jsonl,
    rounded,
    summarize_beir,
    summarize_hardware,
    summarize_rubric,
    summarize_trials,
)


def trial(
    trial_id: str,
    suite: str = "s",
    baseline: str = "base",
    docs: float = 0.5,
    code: float = 0.5,
    **params: Any,
) -> Dict[str, Any]:
    """A lab trial record shaped like ``trials.jsonl``."""
    return {
        "trial_id": trial_id,
        "suite": suite,
        "name": trial_id,
        "params": params,
        "baseline_id": baseline,
        "metrics": {
            "docs": {"recall@5": docs, "mrr": docs / 2},
            "code": {"recall@5": code, "mrr": code / 2},
        },
        "ranks": {"docs": [[1]], "code": [[2]]},
        "search_seconds_per_100": 0.3333333,
        "index_seconds": 1.0,
        "n_chunks": 10,
        "created": "2026-11-12T09:00:00+00:00",
    }


class TestParseJsonl:
    def test_parses_objects_and_skips_blank_lines(self) -> None:
        assert parse_jsonl('{"a": 1}\n\n  \n{"b": 2}\n', "f") == [
            {"a": 1}, {"b": 2},
        ]

    def test_empty_text_gives_no_records(self) -> None:
        assert parse_jsonl("", "f") == []

    def test_invalid_json_names_file_and_line(self) -> None:
        with pytest.raises(SiteError, match=r"f\.jsonl:2: invalid JSON"):
            parse_jsonl('{"a": 1}\n{oops\n', "f.jsonl")

    def test_non_object_line_is_rejected(self) -> None:
        with pytest.raises(SiteError, match=r"f:1: expected a JSON object"):
            parse_jsonl("[1, 2]\n", "f")


class TestRounded:
    def test_rounds_nested_floats_only(self) -> None:
        value = {"a": 0.123456, "b": [1.987654, 2], "c": "x", "d": True}
        assert rounded(value, 2) == {
            "a": 0.12, "b": [1.99, 2], "c": "x", "d": True,
        }

    def test_none_passes_through(self) -> None:
        assert rounded(None) is None


class TestMeanAndHeadline:
    def test_mean_of_nothing_is_none(self) -> None:
        assert mean([]) is None

    def test_mean(self) -> None:
        assert mean([1.0, 2.0, 6.0]) == 3.0

    def test_headline_averages_datasets_with_the_metric(self) -> None:
        metrics = {
            "docs": {"recall@5": 0.8},
            "code": {"recall@5": 0.6},
            "other": {"mrr": 0.1},
        }
        assert headline(metrics) == pytest.approx(0.7)

    def test_headline_without_metric_is_none(self) -> None:
        assert headline({"docs": {"mrr": 0.4}}) is None


class TestChangedParams:
    def test_without_baseline_everything_changed(self) -> None:
        assert changed_params({"a": 1}, None) == {"a": 1}

    def test_only_differences_and_new_keys(self) -> None:
        base = {"a": 1, "b": 2, "c": None}
        params = {"a": 1, "b": 3, "c": None, "d": 0}
        assert changed_params(params, base) == {"b": 3, "d": 0}


class TestSummarizeTrials:
    def test_groups_by_suite_in_order_of_appearance(self) -> None:
        rows = [
            trial("base", suite="b"),
            trial("x", suite="a", baseline="x"),
        ]
        summary = summarize_trials(rows)
        assert [s["suite"] for s in summary["suites"]] == ["b", "a"]
        assert summary["trial_count"] == 2
        assert summary["datasets"] == ["docs", "code"]
        assert summary["metric"] == "recall@5"

    def test_delta_against_baseline_and_best(self) -> None:
        rows = [
            trial("base", docs=0.8, code=0.6, w=1),
            trial("up", docs=0.9, code=0.7, w=2),
            trial("down", docs=0.7, code=0.5, w=0),
        ]
        suite = summarize_trials(rows)["suites"][0]
        by_id = {t["id"]: t for t in suite["trials"]}
        assert suite["baseline"] == pytest.approx(0.7)
        assert by_id["base"]["delta"] == pytest.approx(0.0)
        assert by_id["up"]["delta"] == pytest.approx(0.1)
        assert by_id["down"]["delta"] == pytest.approx(-0.1)
        assert suite["best_id"] == "up"
        assert suite["best_delta"] == pytest.approx(0.1)
        assert by_id["up"]["changed"] == {"w": 2}

    def test_ranks_are_dropped_and_floats_rounded(self) -> None:
        view = summarize_trials([trial("base")])["suites"][0]["trials"][0]
        assert "ranks" not in view
        assert view["search_seconds_per_100"] == 0.3333

    def test_resumed_trial_keeps_last_record(self) -> None:
        rows = [trial("base", docs=0.1), trial("base", docs=0.9)]
        suite = summarize_trials(rows)["suites"][0]
        assert len(suite["trials"]) == 1
        assert suite["trials"][0]["metrics"]["docs"]["recall@5"] == 0.9

    def test_missing_baseline_gives_no_deltas(self) -> None:
        suite = summarize_trials([trial("x", baseline="gone")])["suites"][0]
        assert suite["baseline"] is None
        assert suite["trials"][0]["delta"] is None
        assert suite["trials"][0]["changed"] == suite["trials"][0]["params"]
        assert suite["best_delta"] is None

    def test_trials_without_metrics_are_not_ranked(self) -> None:
        row = trial("base")
        row["metrics"] = {}
        suite = summarize_trials([row])["suites"][0]
        assert suite["best_id"] is None
        assert suite["trials"][0]["headline"] is None

    def test_records_without_suite_or_id_still_summarize(self) -> None:
        summary = summarize_trials([{"metrics": {}}])
        assert summary["suites"][0]["suite"] == "unnamed"
        assert summary["suites"][0]["baseline_id"] is None

    def test_no_rows(self) -> None:
        assert summarize_trials([]) == {
            "metric": "recall@5", "datasets": [], "trial_count": 0,
            "suites": [],
        }


def beir_row(dataset: str, config: str, ndcg: Any) -> Dict[str, Any]:
    """A BEIR benchmark row."""
    return {
        "dataset": dataset, "config": config, "params": {},
        "ndcg10": ndcg, "recall10": 0.5, "recall100": 0.9,
        "hit": {"1": 0.3}, "documents": 100, "queries": 10,
        "search_seconds": 0.123456,
    }


class TestSummarizeBeir:
    def test_groups_sorted_by_ndcg(self) -> None:
        rows = [
            beir_row("d1", "bm25", 0.4),
            beir_row("d1", "hybrid", 0.6),
            beir_row("d2", "bm25", 0.2),
        ]
        summary = summarize_beir(rows)
        d1 = summary["datasets"][0]
        assert d1["dataset"] == "d1"
        assert [r["config"] for r in d1["rows"]] == ["hybrid", "bm25"]
        assert summary["row_count"] == 3

    def test_average_reports_dataset_count_and_sorts_complete_first(
        self,
    ) -> None:
        rows = [
            beir_row("d1", "bm25", 0.4),
            beir_row("d2", "bm25", 0.2),
            beir_row("d1", "only-one", 0.9),
        ]
        averages = summarize_beir(rows)["averages"]
        assert averages[0] == {"config": "bm25", "ndcg10": 0.3,
                               "datasets": 2}
        assert averages[1]["config"] == "only-one"
        assert averages[1]["datasets"] == 1

    def test_rows_without_score_are_kept_but_not_averaged(self) -> None:
        summary = summarize_beir([beir_row("d", "broken", None)])
        assert summary["averages"] == []
        assert summary["datasets"][0]["rows"][0]["ndcg10"] is None


def measured(device: str, name: str, cost: Any, tps: float = 10.0) -> Any:
    """A measured plan, as stored by ``system --benchmark``."""
    return {
        "plan": {"device": device, "threads": 8, "cpus": [0], "name": name},
        "embed_seconds": 0.5, "tokens_per_second": tps, "cost": cost,
    }


class TestSummarizeHardware:
    def test_speedup_of_best_gpu_over_best_cpu(self) -> None:
        run = {
            "created": "2026-11-13T00:00:00+00:00",
            "gpus": [{"name": "RTX"}],
            "results": [
                measured("cpu", "slow", 8.0),
                measured("cuda", "gpu", 0.5),
                measured("cpu", "fast", 4.0),
            ],
        }
        latest = summarize_hardware([run])["latest"]
        assert latest["gpu_speedup"] == 8.0
        assert [r["plan"]["name"] for r in latest["results"]] == [
            "gpu", "fast", "slow",
        ]

    def test_cpu_only_run_has_no_speedup(self) -> None:
        run = {"results": [measured("cpu", "a", 1.0)]}
        assert summarize_hardware([run])["latest"]["gpu_speedup"] is None

    def test_newest_run_first(self) -> None:
        runs = [
            {"created": "2026-01-01", "results": []},
            {"created": "2026-06-01", "results": []},
        ]
        summary = summarize_hardware(runs)
        assert summary["latest"]["created"] == "2026-06-01"
        assert len(summary["runs"]) == 2

    def test_unknown_cost_sorts_last_and_zero_gpu_cost_is_ignored(
        self,
    ) -> None:
        run = {"results": [
            measured("cpu", "unknown", None),
            measured("cpu", "known", 2.0),
            measured("cuda", "zero", 0),
        ]}
        latest = summarize_hardware([run])["latest"]
        assert latest["results"][-1]["plan"]["name"] == "unknown"
        assert latest["gpu_speedup"] is None

    def test_no_runs(self) -> None:
        assert summarize_hardware([]) == {"runs": [], "latest": None}


class TestSummarizeRubric:
    def test_counts(self) -> None:
        data = {
            "name": "Core", "version": "1", "language": "es",
            "competencies": [
                {"indicators": [
                    {"polarity": "positive"}, {"polarity": "negative"},
                ]},
                {"indicators": [{"polarity": "positive"}]},
                {},
            ],
        }
        assert summarize_rubric(data, "r.json") == {
            "path": "r.json", "name": "Core", "version": "1",
            "language": "es", "competencies": 3, "indicators": 3,
            "positive": 2, "negative": 1,
        }

    def test_missing_competencies_is_an_error(self) -> None:
        with pytest.raises(SiteError, match="r.json: no 'competencies'"):
            summarize_rubric({"name": "x"}, "r.json")


class TestDocTitle:
    def test_first_level_one_heading(self) -> None:
        text = "intro\n## Not this\n#  The title \n# Later"
        assert doc_title(text, "fallback") == "The title"

    def test_fallback_without_heading(self) -> None:
        assert doc_title("## sub\ntext", "README") == "README"

    def test_empty_heading_uses_fallback(self) -> None:
        assert doc_title("# \n", "README") == "README"
