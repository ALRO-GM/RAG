"""AnalysisService and the Markdown report on hand-built trials."""

from typing import Any, Dict, Iterable, List, Optional

import pytest

from src.application.analysis import (
    AnalysisReport,
    AnalysisService,
    Interaction,
    TrialRow,
    render_markdown,
)
from src.application.tuning import TuningReport
from src.domain.errors import NotFoundError
from src.domain.experiments import TrialResult
from tests.fakes import InMemoryTrialStore, found_ranks, make_trial

N = 40
BASE_PARAMS: Dict[str, Any] = {"file_boost": 0.0, "mode": "bm25"}


def trial(
    trial_id: str,
    found: Iterable[int],
    seconds: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    baseline: str = "base",
    suite: str = "suite",
    dataset: str = "docs",
) -> TrialResult:
    """Return a trial finding the ``found`` questions out of forty."""
    return make_trial(
        trial_id, {dataset: found_ranks(found, N)},
        params=BASE_PARAMS if params is None else params,
        suite=suite, baseline_id=baseline, seconds=seconds,
    )


BASE = trial("base", range(20))


def analyze(*trials: TrialResult, **rules: Any) -> AnalysisReport:
    """Analyse ``trials`` with the given recommendation rules."""
    return AnalysisService(InMemoryTrialStore(trials), **rules).analyze()


def row(report: AnalysisReport, trial_id: str) -> TrialRow:
    """Return the table line of a trial."""
    [found] = [r for r in report.rows if r.trial_id == trial_id]
    return found


def test_analyze_without_trials_is_not_found() -> None:
    """An empty store tells the user to run an experiment first."""
    with pytest.raises(NotFoundError, match="lab experiment"):
        AnalysisService(InMemoryTrialStore()).analyze()


def test_baseline_row_describes_the_trial_without_delta() -> None:
    """A baseline has its metrics and costs but no comparison."""
    base = make_trial(
        "base", {"docs": found_ranks(range(20), N, rank=2)},
        params=BASE_PARAMS, seconds=1.5, index_seconds=12.0,
        index_cold=True, n_chunks=345, name="baseline",
    )
    report = analyze(base)
    assert report.n_trials == 1
    [line] = report.rows
    assert (line.trial_id, line.suite, line.name) == (
        "base", "suite", "baseline",
    )
    assert line.params == BASE_PARAMS
    assert line.changed == {}
    assert line.verdict == "baseline"
    assert line.recall5 == {"docs": 0.5}
    assert line.mean_recall5 == 0.5
    assert line.mrr == pytest.approx(0.25)
    assert line.objective == pytest.approx(0.5 + 0.1 * 0.25)
    assert (line.delta_points, line.ci_low, line.ci_high) == (0.0, 0.0, 0.0)
    assert (line.search_seconds, line.delta_seconds) == (1.5, 0.0)
    assert line.points_per_second is None
    assert (line.index_seconds, line.index_cold, line.n_chunks) == (
        12.0, True, 345,
    )


def test_rows_macro_average_the_datasets() -> None:
    """Mean recall, MRR and objective weight each dataset equally."""
    result = make_trial("t", {
        "docs": [[1], [1]], "code": [[0], [0], [0], [2]],
    })
    line = row(analyze(result), "t")
    assert line.recall5 == {"docs": 1.0, "code": 0.25}
    assert line.mean_recall5 == pytest.approx(0.625)
    assert line.mrr == pytest.approx((1.0 + 0.125) / 2)
    assert line.objective == pytest.approx(0.625 + 0.1 * 0.5625)
    custom = row(analyze(result, mrr_weight=1.0), "t")
    assert custom.objective == pytest.approx(0.625 + 0.5625)


@pytest.mark.parametrize("found, delta, verdict", [
    (range(40), 50.0, "helps"),
    (range(5), -37.5, "hurts"),
    (range(20), 0.0, "neutral"),
    (range(23), 7.5, "likely helps"),
    (range(17), -7.5, "likely hurts"),
    (range(2, 23), 2.5, "neutral"),
    (range(6, 25), -2.5, "neutral"),
])
def test_verdict_follows_the_bootstrap_interval(
    found: Iterable[int], delta: float, verdict: str
) -> None:
    """The verdict reads the interval first, then the share of wins."""
    report = analyze(BASE, trial("cand", found))
    line = row(report, "cand")
    assert line.delta_points == pytest.approx(delta)
    assert line.verdict == verdict
    assert line.ci_low <= line.delta_points <= line.ci_high
    assert 0.0 <= line.p_gain <= 1.0
    if verdict == "helps":
        assert line.ci_low > 0 and line.p_gain == 1.0
    if verdict == "hurts":
        assert line.ci_high < 0 and line.p_gain == 0.0
    if verdict == "likely helps":
        assert line.ci_low == 0.0 and line.p_gain >= 0.8
    if verdict == "likely hurts":
        assert line.ci_high == 0.0 and line.p_gain <= 0.2
    assert row(report, "base").verdict == "baseline"


def test_delta_is_measured_on_recall_not_on_rank() -> None:
    """Finding the same questions at a better rank is a null delta."""
    better_rank = make_trial(
        "cand", {"docs": found_ranks(range(20), N, rank=1)},
        params=BASE_PARAMS, baseline_id="base",
    )
    base = make_trial(
        "base", {"docs": found_ranks(range(20), N, rank=4)},
        params=BASE_PARAMS,
    )
    line = row(analyze(base, better_rank), "cand")
    assert (line.delta_points, line.verdict) == (0.0, "neutral")
    assert line.objective > row(analyze(base), "base").objective


def test_changed_lists_the_parameters_differing_from_the_baseline() -> None:
    """Modified and added parameters are reported, unchanged ones not."""
    report = analyze(
        BASE,
        trial("a", range(20), params={"file_boost": 0.2, "mode": "bm25"}),
        trial("b", range(20), params={**BASE_PARAMS, "test_weight": 0.5}),
        trial("c", range(20), params={"file_boost": 0.3, "mode": "hybrid"}),
    )
    assert row(report, "a").changed == {"file_boost": 0.2}
    assert row(report, "b").changed == {"test_weight": 0.5}
    assert row(report, "c").changed == {"file_boost": 0.3, "mode": "hybrid"}
    assert row(report, "base").changed == {}


def test_cost_is_reported_as_points_per_extra_second() -> None:
    """A slower trial shows what each extra second buys."""
    report = analyze(BASE, trial("slow", range(40), seconds=3.0))
    line = row(report, "slow")
    assert line.search_seconds == 3.0
    assert line.delta_seconds == pytest.approx(2.0)
    assert line.points_per_second == pytest.approx(25.0)


@pytest.mark.parametrize("seconds", [1.0, 1.04, 0.5])
def test_no_rate_for_trials_that_are_not_slower(seconds: float) -> None:
    """Within 0.05 s of the baseline, or faster, there is no rate."""
    report = analyze(BASE, trial("cand", range(40), seconds=seconds))
    line = row(report, "cand")
    assert line.points_per_second is None
    assert line.delta_seconds == pytest.approx(seconds - 1.0)


def grid(both: Iterable[int] = range(30)) -> List[TrialResult]:
    """Return a baseline, two single changes and their combination."""
    return [
        trial("base", range(20), params={"x": 0, "y": 0}),
        trial("x", range(24), params={"x": 1, "y": 0}),
        trial("y", range(22), params={"x": 0, "y": 1}),
        trial("xy", both, params={"x": 1, "y": 1}),
    ]


def test_interaction_of_two_changes_measured_alone_and_together() -> None:
    """Synergy compares the joint gain with the sum of single gains."""
    report = analyze(*grid())
    [found] = report.interactions
    assert isinstance(found, Interaction)
    assert (found.suite, found.first, found.second) == ("suite", "x=1", "y=1")
    assert found.gain_first == pytest.approx(10.0)
    assert found.gain_second == pytest.approx(5.0)
    assert found.gain_both == pytest.approx(25.0)
    assert found.synergy == pytest.approx(10.0)


@pytest.mark.parametrize("both, synergy", [
    (range(26), 0.0), (range(24), -5.0),
])
def test_interaction_additive_and_overlapping_gains(
    both: Iterable[int], synergy: float
) -> None:
    """Gains that add up have no synergy; overlapping ones a negative."""
    [found] = analyze(*grid(both)).interactions
    assert found.synergy == pytest.approx(synergy)


def test_no_interaction_without_both_single_changes() -> None:
    """A pair is only analysed when each change was also tried alone."""
    base, single_x, single_y, pair = grid()
    assert analyze(base, single_x, pair).interactions == []
    assert analyze(base, single_x, single_y).interactions == []
    triple = trial("xyz", range(30), params={"x": 1, "y": 1, "z": 1})
    assert analyze(base, single_x, single_y, triple).interactions == []


def test_interactions_do_not_mix_baselines() -> None:
    """Single changes against another baseline are not combined."""
    base, single_x, _, pair = grid()
    other = trial("other", range(20), params={"x": 0, "y": 0, "s": 2},
                  suite="second")
    foreign_y = trial("y2", range(22), params={"x": 0, "y": 1, "s": 2},
                      baseline="other", suite="second")
    report = analyze(base, single_x, pair, other, foreign_y)
    assert report.interactions == []
    assert row(report, "y2").changed == {"y": 1}


def front() -> List[TrialResult]:
    """Return four standalone trials: three on the front, one dominated."""
    return [
        trial("mid", range(30), seconds=2.0, baseline="mid"),
        trial("slow", range(40), seconds=5.0, baseline="slow"),
        trial("dominated", range(24), seconds=3.0, baseline="dominated"),
        trial("fast", range(20), seconds=1.0, baseline="fast"),
    ]


def profiles(report: AnalysisReport) -> Dict[str, str]:
    """Return the trial chosen by each profile."""
    return {p.name: p.trial_id for p in report.profiles}


def test_pareto_front_lists_non_dominated_trials_by_cost() -> None:
    """The front goes from the fastest to the most precise trial."""
    assert analyze(*front()).pareto == ["fast", "mid", "slow"]


def test_three_profiles_with_their_configuration() -> None:
    """Precision, balanced and speed are recommended in that order."""
    report = analyze(*front(), max_search_seconds=4.0,
                     min_points_per_second=10.0, required={"docs": 0.4})
    assert [p.name for p in report.profiles] == [
        "precision", "balanced", "speed",
    ]
    assert profiles(report) == {
        "precision": "mid", "balanced": "mid", "speed": "fast",
    }
    precision = report.profiles[0]
    assert precision.params == BASE_PARAMS
    assert precision.mean_recall5 == 0.75
    assert precision.recall5 == {"docs": 0.75}
    assert precision.search_seconds == 2.0
    assert "4 s per 100 questions" in precision.reason
    assert "10 recall points" in report.profiles[1].reason
    assert "5 points above" in report.profiles[2].reason


@pytest.mark.parametrize("budget, chosen", [
    (22.0, "slow"), (4.0, "mid"), (1.5, "fast"), (1.0, "fast"),
])
def test_precision_profile_is_the_best_objective_within_budget(
    budget: float, chosen: str
) -> None:
    """Trials slower than the search budget are not candidates."""
    report = analyze(*front(), max_search_seconds=budget)
    assert profiles(report)["precision"] == chosen


def test_precision_profile_prefers_the_faster_of_equal_trials() -> None:
    """With the same objective the cheaper configuration wins."""
    report = analyze(
        trial("a", range(30), seconds=3.0, baseline="a"),
        trial("b", range(30), seconds=2.0, baseline="b"),
    )
    assert profiles(report)["precision"] == "b"
    assert report.pareto == ["b"]


@pytest.mark.parametrize("rate, chosen", [
    (30.0, "fast"), (10.0, "mid"), (8.0, "slow"), (0.5, "slow"),
])
def test_balanced_profile_stops_when_seconds_buy_too_little(
    rate: float, chosen: str
) -> None:
    """Walking the front, a step is taken only above the minimum rate."""
    report = analyze(*front(), min_points_per_second=rate)
    assert profiles(report)["balanced"] == chosen


@pytest.mark.parametrize("required, margin, chosen", [
    ({"docs": 0.4}, 0.05, "fast"),
    ({"docs": 0.5}, 0.0, "fast"),
    ({"docs": 0.5}, 0.05, "mid"),
    ({"docs": 0.8}, 0.05, "slow"),
    ({"docs": 0.5, "code": 0.99}, 0.0, "fast"),
])
def test_speed_profile_is_the_fastest_safe_configuration(
    required: Dict[str, float], margin: float, chosen: str
) -> None:
    """Safe means a margin above the required recall of known datasets."""
    report = analyze(*front(), required=required, margin=margin)
    assert profiles(report)["speed"] == chosen


def test_profiles_are_omitted_when_no_trial_qualifies() -> None:
    """No affordable or no safe trial means no such recommendation."""
    report = analyze(*front(), max_search_seconds=0.5,
                     required={"docs": 0.99})
    assert profiles(report) == {"balanced": "slow"}


def test_default_rules_ignore_datasets_without_requirement() -> None:
    """Required recalls default to docs/code; other datasets are free."""
    weak = trial("weak", range(2), seconds=0.5, baseline="weak",
                 dataset="other")
    assert profiles(analyze(weak))["speed"] == "weak"
    weak_docs = trial("weak", range(2), seconds=0.5, baseline="weak")
    assert "speed" not in profiles(analyze(weak_docs))


# ------------------------------------------------------------ markdown


def section(text: str, heading: str) -> List[str]:
    """Return the non-empty lines of the block following ``heading``."""
    block = text.split(heading + "\n", 1)[1].lstrip("\n")
    return block.split("\n\n", 1)[0].splitlines()


def test_render_markdown_has_every_section() -> None:
    """The report has profiles, the front, the suites and interactions."""
    text = render_markdown(analyze(*grid()))
    assert text.startswith("# Experiment report\n")
    assert "4 trials." in text
    for heading in ("## Recommended profiles", "## Cost/benefit per suite",
                    "## Pareto front: recall against search time",
                    "### suite", "## Interactions"):
        assert f"\n{heading}\n" in text
    assert "| suite | `x=1` | `y=1` | +10.0 | +5.0 | +25.0 | +10.0 |" in text


def test_render_markdown_has_one_row_per_trial() -> None:
    """Each suite table lists its baseline first, then every trial."""
    report = analyze(
        BASE,
        trial("up", range(40), seconds=3.0,
              params={"file_boost": 0.2, "mode": "bm25"}),
        trial("down", range(5), params={"file_boost": 0.9, "mode": "bm25"}),
        trial("solo", range(10), baseline="solo", suite="second"),
    )
    text = render_markdown(report)
    table = section(text, "### suite")
    assert len(table) == 2 + 3
    assert table[0].startswith("| Change | R@5 docs | Mean R@5 | MRR |")
    assert set(table[1]) == {"|", "-"}
    assert table[2].startswith("| baseline | 0.500 | 0.500 |")
    assert table[2].endswith("| baseline |")
    assert table[3].startswith("| `file_boost=0.2` | 1.000 | 1.000 |")
    assert "| +50.0 (" in table[3]
    assert table[3].endswith("| 3.00 | +2.00 | +25.00 | helps |")
    assert table[4].startswith("| `file_boost=0.9` |")
    assert table[4].endswith("| — | hurts |")
    assert len(section(text, "### second")) == 2 + 1
    assert all(line.count("|") == table[0].count("|") for line in table)


def test_render_markdown_lists_profiles_and_the_front() -> None:
    """Profiles show their numbers and parameters; the front its trials."""
    report = analyze(*front(), max_search_seconds=4.0,
                     min_points_per_second=10.0, required={"docs": 0.4})
    text = render_markdown(report)
    table = section(text, "## Recommended profiles")
    assert table[0] == "| Profile | Mean R@5 | R@5 docs | s/100 q | Why |"
    assert [line.split("|")[1].strip() for line in table[2:]] == [
        "**precision**", "**balanced**", "**speed**",
    ]
    assert table[2].startswith("| **precision** | 0.750 | 0.750 | 2.00 |")
    assert f"`speed`: `{BASE_PARAMS}`" in text
    pareto = text.split("## Pareto front")[1].split("## Cost/benefit")[0]
    rows = [line for line in pareto.splitlines() if line.startswith("| ")]
    assert [r.split("|")[1].strip() for r in rows[1:]] == [
        "1.00", "2.00", "5.00",
    ]
    assert "## Interactions" not in text


def test_render_markdown_has_one_column_per_dataset() -> None:
    """Datasets are columns, sorted by name; missing ones show 0."""
    both = make_trial("both", {"docs": [[1]], "code": [[0]]})
    only = make_trial("only", {"docs": [[1]]}, baseline_id="both")
    table = section(render_markdown(analyze(both, only)), "### suite")
    assert table[0].startswith("| Change | R@5 code | R@5 docs | Mean R@5 |")
    assert table[2].startswith("| baseline | 0.000 | 1.000 | 0.500 |")


def test_render_markdown_reports_cold_index_builds() -> None:
    """Trials that built their index from scratch show its cost."""
    cold = make_trial("cold", {"docs": [[1]]}, index_cold=True,
                      index_seconds=12.4, n_chunks=345)
    warm = make_trial("warm", {"docs": [[1]]}, baseline_id="cold",
                      params={"k1": 2}, index_seconds=0.3)
    text = render_markdown(analyze(cold, warm))
    assert "Cold index builds: baseline → 12 s (345 chunks)\n" in text
    assert "Cold index builds" not in render_markdown(analyze(warm))


def test_time_differences_below_the_resolution_are_ignored() -> None:
    """Sub-resolution timing noise does not decide the profiles."""
    slower_better = trial("better", range(30), seconds=1.3,
                          params={"x": 1})
    faster_worse = trial("worse", range(25), seconds=0.9, params={"x": 2})
    report = analyze(BASE, slower_better, faster_worse,
                     required={"docs": 0.5})
    assert report.pareto == ["better"]
    assert profiles(report)["speed"] == "better"
    exact = analyze(BASE, slower_better, faster_worse,
                    required={"docs": 0.5}, time_resolution=0.0)
    assert profiles(exact)["speed"] == "worse"


def test_a_trial_shared_by_two_suites_is_one_profile_candidate() -> None:
    """Rows are per suite, but the Pareto front has unique trials."""
    again = trial("base", range(20), suite="other")
    report = analyze(BASE, again)
    assert report.n_trials == 2
    assert report.pareto == ["base"]


def test_tuning_reports_are_rendered() -> None:
    """The report shows in-sample and held-out gains and the stability."""
    tuning = TuningReport(
        suite="sweep", best_params={"x": 2}, in_sample=0.95, baseline=0.90,
        cv_estimate=0.92, cv_baseline=0.90, search_seconds_per_100=1.0,
        stability={"x": {"2": 4, "0": 1}}, consensus_params={"x": 2},
        consensus_score=0.94,
    )
    store = InMemoryTrialStore([BASE])
    report = AnalysisService(store).analyze([tuning])
    text = render_markdown(report)
    assert "## Tuning: sweep" in text
    assert "| In-sample | 0.9000 | 0.9500 | +5.0 |" in text
    assert "| 0.9000 | 0.9200 | +2.0 |" in text
    assert "| `x` | 2 ×4, 0 ×1 | `2` |" in text


def test_tuner_trials_are_not_profile_candidates() -> None:
    """Trials explored by the tuner never become a recommendation."""
    lucky = trial("lucky", range(40), suite="sweep/tune", params={"x": 9})
    report = analyze(BASE, lucky, required={"docs": 0.1})
    assert set(profiles(report).values()) == {"base"}
    assert report.pareto == ["base"]
    assert row(report, "lucky").mean_recall5 == 1.0


def cold(trial_id: str, found: Iterable[int], index_seconds: float,
         seconds: float, params: Dict[str, Any]) -> TrialResult:
    """Return a trial measured with a cold index build."""
    return make_trial(
        trial_id, {"docs": found_ranks(found, N)}, params=params,
        baseline_id="base", seconds=seconds, index_seconds=index_seconds,
        index_cold=True,
    )


def test_end_to_end_front_and_speed_profile_use_cold_builds() -> None:
    """Cold trials are compared on indexing + 200 questions."""
    lexical = cold("lexical", range(28), 2.0, 0.1,
                   {"semantic": False, "lab_cold": True})
    hybrid = cold("hybrid", range(30), 60.0, 0.5, {"lab_cold": True})
    slow = cold("slow", range(30), 200.0, 0.5,
                {"embedding_max_tokens": 256, "lab_cold": True})
    report = analyze(BASE, lexical, hybrid, slow, required={"docs": 0.5})
    assert report.pareto_end_to_end == ["lexical", "hybrid"]
    assert row(report, "hybrid").end_to_end_seconds == pytest.approx(61.0)
    assert row(report, "base").end_to_end_seconds is None
    assert profiles(report)["speed"] == "lexical"
    [speed] = [p for p in report.profiles if p.name == "speed"]
    assert speed.params == {"semantic": False}
    text = render_markdown(report)
    assert "cold indexing + 200 questions" in text
    assert "| 2 | 0.2 | 2 | 0.700 | lexical |" in text


def test_tuner_tables_are_truncated_in_the_report() -> None:
    """Only the best tuner trials are listed."""
    explored = [
        trial(f"t{i}", range(i), suite="sweep/tune", params={"x": i})
        for i in range(1, 15)
    ]
    text = render_markdown(analyze(BASE, *explored))
    assert "14 configurations explored by the tuner" in text
    assert "`x=14`" in text and "`x=2`" not in text
