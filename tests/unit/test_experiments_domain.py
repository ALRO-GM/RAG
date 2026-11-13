"""Experiment models and statistics: suites, ranks, bootstrap, Pareto."""

from typing import Dict, List, Tuple

import pytest
from pydantic import ValidationError

from src.domain.experiments import (
    K_VALUES,
    Ranks,
    Suite,
    TrialSpec,
    kfold_indices,
    macro_mean,
    objective_scores,
    paired_bootstrap,
    params_id,
    pareto_front,
    rank_metrics,
    recall_scores,
    reciprocal_ranks,
    source_ranks,
    synergy,
)
from src.domain.models import MinimalSource
from tests.fakes import make_trial


def src(path: str, first: int, last: int) -> MinimalSource:
    """Return a source."""
    return MinimalSource(file_path=path, first_character_index=first,
                         last_character_index=last)


def names_and_params(suite: Suite) -> List[Tuple[str, Dict[str, object]]]:
    """Return the (name, params) pairs of the expanded suite."""
    return [(s.name, s.params) for s in suite.expand()]


# -------------------------------------------------------------- suites


def test_expand_one_at_a_time_changes_one_factor_per_trial() -> None:
    """The baseline comes first, then each value alone on the base."""
    suite = Suite(name="s", base={"a": 1, "b": "x"},
                  factors={"a": [2, 3], "b": ["y"]})
    assert names_and_params(suite) == [
        ("baseline", {"a": 1, "b": "x"}),
        ("a=2", {"a": 2, "b": "x"}),
        ("a=3", {"a": 3, "b": "x"}),
        ("b=y", {"a": 1, "b": "y"}),
    ]


def test_expand_drops_trials_identical_to_an_earlier_one() -> None:
    """A factor value equal to the base is the baseline, not a new trial."""
    suite = Suite(name="s", base={"a": 1},
                  factors={"a": [1, 2, 2], "b": [5]})
    assert names_and_params(suite) == [
        ("baseline", {"a": 1}),
        ("a=2", {"a": 2}),
        ("b=5", {"a": 1, "b": 5}),
    ]


def test_expand_grid_is_the_product_of_the_factors() -> None:
    """Every combination is tried once; the base combination is deduped."""
    suite = Suite(name="s", strategy="grid", base={"a": 1, "b": "x", "c": 0},
                  factors={"a": [1, 2], "b": ["x", "y"]})
    assert names_and_params(suite) == [
        ("baseline", {"a": 1, "b": "x", "c": 0}),
        ("a=1,b=y", {"a": 1, "b": "y", "c": 0}),
        ("a=2,b=x", {"a": 2, "b": "x", "c": 0}),
        ("a=2,b=y", {"a": 2, "b": "y", "c": 0}),
    ]


def test_expand_list_merges_each_trial_into_the_base() -> None:
    """Explicit trials override the base; duplicates are dropped."""
    suite = Suite(name="s", strategy="list", base={"a": 1, "b": 1}, trials=[
        TrialSpec(name="first", params={"a": 2}),
        TrialSpec(name="same-as-first", params={"a": 2, "b": 1}),
        TrialSpec(name="same-as-baseline"),
        TrialSpec(name="second", params={"c": 3}),
    ], factors={"ignored": [1, 2]})
    assert names_and_params(suite) == [
        ("baseline", {"a": 1, "b": 1}),
        ("first", {"a": 2, "b": 1}),
        ("second", {"a": 1, "b": 1, "c": 3}),
    ]


@pytest.mark.parametrize("strategy", ["one_at_a_time", "grid", "list"])
def test_expand_without_factors_is_only_the_baseline(strategy: str) -> None:
    """An empty suite still has its baseline, a copy of the base."""
    suite = Suite.model_validate(
        {"name": "s", "strategy": strategy, "base": {"a": 1}}
    )
    [baseline] = suite.expand()
    assert (baseline.name, baseline.params) == ("baseline", {"a": 1})
    baseline.params["a"] = 2
    assert suite.base == {"a": 1}


def test_suite_rejects_unknown_strategy_and_missing_name() -> None:
    """The strategy is one of three literals and the name is required."""
    with pytest.raises(ValidationError):
        Suite.model_validate({"name": "s", "strategy": "random"})
    with pytest.raises(ValidationError):
        Suite.model_validate({"base": {}})


def test_params_id_is_stable_and_order_independent() -> None:
    """The same parameters give the same short hexadecimal identifier."""
    first = params_id({"a": 1, "b": [1, 2], "c": None})
    assert first == params_id({"c": None, "b": [1, 2], "a": 1})
    assert len(first) == 12
    assert int(first, 16) >= 0


def test_params_id_depends_on_values_and_salt() -> None:
    """Any changed value, key or salt gives another identifier."""
    ids = {
        params_id({"a": 1}), params_id({"a": 2}), params_id({"b": 1}),
        params_id({"a": 1.5}), params_id({"a": "1"}), params_id({}),
        params_id({"a": 1}, "salt"), params_id({"a": 1}, "other"),
    }
    assert len(ids) == 8


def test_trial_result_mean_metric_averages_datasets() -> None:
    """A metric is averaged over the datasets, not over the questions."""
    trial = make_trial("t", {"docs": [[1], [1]], "code": [[0], [0], [0], [1]]})
    assert trial.mean_metric("recall@5") == pytest.approx(0.625)
    assert make_trial("empty", {}).mean_metric("recall@5") == 0.0


# --------------------------------------------------------------- ranks


def test_source_ranks_gives_rank_of_first_overlapping_result() -> None:
    """Each reference gets the 1-based position of its first match."""
    expected = [src("a.py", 0, 100), src("b.md", 0, 100)]
    retrieved = [
        src("c.md", 0, 100), src("b.md", 50, 150), src("a.py", 0, 100),
        src("b.md", 0, 100),
    ]
    assert source_ranks(expected, retrieved) == [3, 2]


def test_source_ranks_uses_the_iou_threshold() -> None:
    """Another file or an overlap below 5% IoU is not a match (rank 0)."""
    reference = [src("a.py", 0, 100)]
    assert source_ranks(reference, [src("b.py", 0, 100)]) == [0]
    assert source_ranks(reference, [src("a.py", 99, 1000)]) == [0]
    assert source_ranks(reference, [src("a.py", 100, 200)]) == [0]
    assert source_ranks(reference, [src("a.py", 0, 2000)]) == [1]
    assert source_ranks(reference, [src("a.py", 0, 2001)]) == [0]


def test_source_ranks_degenerate_inputs() -> None:
    """No result gives zeros; no reference gives no rank."""
    assert source_ranks([src("a.py", 0, 10), src("a.py", 20, 30)], []) == [
        0, 0,
    ]
    assert source_ranks([], [src("a.py", 0, 10)]) == []


RANKS: Ranks = [[1], [6], [0], [2, 0], []]


def test_recall_scores_counts_references_found_within_k() -> None:
    """Recall is the share of references ranked in 1..k, per question."""
    assert recall_scores(RANKS, 5) == [1.0, 0.0, 0.0, 0.5, 0.0]
    assert recall_scores(RANKS, 10) == [1.0, 1.0, 0.0, 0.5, 0.0]
    assert recall_scores(RANKS, 1) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert recall_scores([], 5) == []


def test_reciprocal_ranks_average_the_references() -> None:
    """Missing references count 0; found ones 1 / rank."""
    assert reciprocal_ranks(RANKS) == pytest.approx(
        [1.0, 1 / 6, 0.0, 0.25, 0.0]
    )


def test_rank_metrics_reports_every_k_and_mrr() -> None:
    """Recall@1/3/5/10 and MRR are means over the questions."""
    metrics = rank_metrics([[1], [3], [7], [0]])
    assert set(metrics) == {f"recall@{k}" for k in K_VALUES} | {"mrr"}
    assert metrics["recall@1"] == 0.25
    assert metrics["recall@3"] == 0.5
    assert metrics["recall@5"] == 0.5
    assert metrics["recall@10"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 7) / 4)


def test_rank_metrics_of_no_question_are_zero() -> None:
    """An empty dataset has zero metrics instead of a division error."""
    assert set(rank_metrics([]).values()) == {0.0}


def test_objective_scores_add_a_small_mrr_term() -> None:
    """The objective is recall@k plus mrr_weight times 1 / rank."""
    ranks = {"docs": [[1], [2], [0]], "code": [[7]]}
    assert objective_scores(ranks) == {
        "docs": pytest.approx([1.1, 1.05, 0.0]),
        "code": pytest.approx([0.1 / 7]),
    }
    custom = objective_scores(ranks, k=1, mrr_weight=1.0)
    assert custom["docs"] == pytest.approx([2.0, 0.5, 0.0])


def test_objective_breaks_recall_ties_by_rank() -> None:
    """Same recall@5, better rank: the objective prefers the latter."""
    low = macro_mean(objective_scores({"d": [[5], [5]]}))
    high = macro_mean(objective_scores({"d": [[1], [5]]}))
    assert high > low
    assert high - low < 0.1, "a tie-break, smaller than one recall point"


# ---------------------------------------------------------- statistics


def test_macro_mean_weights_datasets_equally() -> None:
    """Each dataset counts once, whatever its number of questions."""
    scores = {"small": [1.0, 1.0], "large": [0.0, 0.0, 0.0, 1.0]}
    assert macro_mean(scores) == pytest.approx(0.625)


def test_macro_mean_on_a_subset_of_questions() -> None:
    """Only the chosen positions count; unlisted datasets are skipped."""
    scores = {"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]}
    assert macro_mean(scores, {"a": [0], "b": [2]}) == 1.0
    assert macro_mean(scores, {"a": [1, 2], "b": [0, 2]}) == 0.25
    assert macro_mean(scores, {"a": [0, 1]}) == 0.5
    assert macro_mean(scores, {"a": [], "b": [2]}) == 1.0


def test_macro_mean_without_values_is_zero() -> None:
    """Nothing to average gives 0 instead of an error."""
    assert macro_mean({}) == 0.0
    assert macro_mean({"a": []}) == 0.0
    assert macro_mean({"a": [1.0]}, {}) == 0.0


def test_paired_bootstrap_identical_trials_have_zero_delta() -> None:
    """Comparing a trial with itself gives a null, zero-width interval."""
    scores = {"docs": [1.0, 0.0, 1.0, 1.0], "code": [0.0, 1.0]}
    assert paired_bootstrap(scores, scores) == (0.0, 0.0, 0.0, 0.0)


def test_paired_bootstrap_clearly_better_candidate() -> None:
    """A large consistent gain has an interval strictly above zero."""
    baseline = {"docs": [0.0] * 40}
    candidate = {"docs": [1.0] * 20 + [0.0] * 20}
    delta, low, high, p_gain = paired_bootstrap(candidate, baseline)
    assert delta == pytest.approx(0.5)
    assert 0.0 < low < delta < high < 1.0
    assert p_gain == 1.0


def test_paired_bootstrap_clearly_worse_candidate() -> None:
    """A consistent loss has an interval strictly below zero."""
    baseline = {"docs": [1.0] * 30}
    candidate = {"docs": [0.0] * 30}
    assert paired_bootstrap(candidate, baseline) == (-1.0, -1.0, -1.0, 0.0)


def test_paired_bootstrap_small_gain_is_not_significant() -> None:
    """Three more questions out of forty: the interval reaches zero."""
    baseline = {"docs": [0.0] * 40}
    candidate = {"docs": [1.0] * 3 + [0.0] * 37}
    delta, low, high, p_gain = paired_bootstrap(candidate, baseline)
    assert delta == pytest.approx(0.075)
    assert low == 0.0 < high
    assert 0.8 < p_gain < 1.0


def test_paired_bootstrap_is_deterministic_for_a_seed() -> None:
    """The same seed gives the same interval; the delta ignores the seed."""
    baseline = {"docs": [0.0, 1.0] * 15}
    candidate = {"docs": [1.0, 1.0, 0.0] * 10}
    first = paired_bootstrap(candidate, baseline, seed=3)
    assert first == paired_bootstrap(candidate, baseline, seed=3)
    other = paired_bootstrap(candidate, baseline, seed=4, samples=50)
    assert other[0] == first[0]
    assert other[1:] != first[1:]


def test_paired_bootstrap_macro_averages_the_datasets() -> None:
    """The delta is the mean of the per-dataset mean differences."""
    baseline = {"docs": [0.0, 0.0], "code": [1.0, 1.0, 1.0, 1.0]}
    candidate = {"docs": [1.0, 1.0], "code": [1.0, 1.0, 1.0, 0.0]}
    delta, low, high, _ = paired_bootstrap(candidate, baseline)
    assert delta == pytest.approx((1.0 - 0.25) / 2)
    assert low <= delta <= high


@pytest.mark.parametrize("candidate, baseline", [
    ({}, {}),
    ({"docs": [1.0]}, {}),
    ({"docs": [1.0]}, {"code": [0.0]}),
    ({"docs": []}, {"docs": []}),
])
def test_paired_bootstrap_without_common_questions(
    candidate: Dict[str, List[float]], baseline: Dict[str, List[float]]
) -> None:
    """Nothing to compare gives zeros instead of an error."""
    assert paired_bootstrap(candidate, baseline) == (0.0, 0.0, 0.0, 0.0)


def test_paired_bootstrap_ignores_datasets_missing_on_one_side() -> None:
    """Only the datasets present in both trials are compared."""
    delta, _, _, _ = paired_bootstrap(
        {"docs": [1.0, 1.0], "extra": [0.0]}, {"docs": [0.0, 1.0]}
    )
    assert delta == 0.5


def test_pareto_front_keeps_non_dominated_points_by_cost() -> None:
    """A point survives when nothing cheaper is at least as good."""
    points = [(1.0, 0.5), (2.0, 0.7), (3.0, 0.6), (4.0, 0.9), (2.0, 0.4)]
    assert pareto_front(points) == [0, 1, 3]
    assert pareto_front(list(reversed(points))) == [4, 3, 1]


def test_pareto_front_ties() -> None:
    """Equal cost keeps the best value; equal value keeps the cheapest."""
    assert pareto_front([(1.0, 0.5), (1.0, 0.6)]) == [1]
    assert pareto_front([(2.0, 0.5), (1.0, 0.5)]) == [1]
    assert pareto_front([(1.0, 0.5), (1.0, 0.5)]) == [0]


def test_pareto_front_degenerate_inputs() -> None:
    """No point gives no front; one point is its own front."""
    assert pareto_front([]) == []
    assert pareto_front([(3.0, 0.0)]) == [0]


@pytest.mark.parametrize("n, folds", [(10, 5), (7, 3), (5, 5), (3, 2)])
def test_kfold_indices_partition_the_questions(n: int, folds: int) -> None:
    """Every index is in exactly one sorted fold of balanced size."""
    parts = kfold_indices(n, folds)
    assert len(parts) == folds
    assert sorted(i for part in parts for i in part) == list(range(n))
    assert all(part == sorted(part) for part in parts)
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_indices_are_shuffled_deterministically() -> None:
    """The same seed gives the same folds; another seed another split."""
    assert kfold_indices(20, 4, seed=1) == kfold_indices(20, 4, seed=1)
    assert kfold_indices(20, 4, seed=1) != kfold_indices(20, 4, seed=2)
    assert kfold_indices(20, 4) != [
        list(range(f, 20, 4)) for f in range(4)
    ], "the questions are shuffled before being dealt"


def test_kfold_indices_with_fewer_questions_than_folds() -> None:
    """Extra folds are empty rather than an error."""
    parts = kfold_indices(2, 4)
    assert sorted(len(p) for p in parts) == [0, 0, 1, 1]
    assert kfold_indices(0, 3) == [[], [], []]


@pytest.mark.parametrize("base, a, b, both, expected", [
    (0.5, 0.6, 0.7, 0.8, 0.0),
    (0.0, 1.0, 1.0, 3.0, 1.0),
    (0.0, 2.0, 2.0, 2.0, -2.0),
    (0.5, 0.5, 0.5, 0.5, 0.0),
])
def test_synergy(
    base: float, a: float, b: float, both: float, expected: float
) -> None:
    """Synergy is gain(A+B) minus the two individual gains."""
    assert synergy(base, a, b, both) == pytest.approx(expected)
