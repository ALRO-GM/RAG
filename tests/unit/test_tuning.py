"""Tuner and RouterTrainer on synthetic objectives and tiny datasets."""

from typing import Any, List, Mapping, Tuple

import pytest

from src.application.experiments import ExperimentService
from src.application.tuning import (
    RouterTrainer,
    Tuner,
    TuningReport,
    consensus,
)
from src.domain.errors import InvalidInputError
from src.domain.experiments import Suite, TrialSpec
from src.domain.models import RagDataset, UnansweredQuestion
from src.domain.priors import FEATURE_NAMES
from tests.fakes import (
    FakePipelineFactory,
    InMemoryDatasetRepository,
    InMemoryRouterRepository,
    InMemoryTrialStore,
    RankRule,
    dataset_json,
    found_ranks,
    make_chunk,
    make_question,
    make_trial,
)

N = 10
TARGETS = {
    f"question {i}": make_chunk("docs/a.md", 100 * i, f"paragraph {i}")
    for i in range(N)
}
PATHS = {"docs": "docs.json"}
HITS = {0: 4, 1: 6, 2: 9, 3: 6}
"""Value of factor ``x`` → how many of the first questions are found."""
POINT = 1.1
"""Objective of a question found at rank 1: recall 1 + 0.1 · MRR 1."""

Wired = Tuple[ExperimentService, FakePipelineFactory, InMemoryTrialStore]


def by_x(params: Mapping[str, Any], query: str) -> int:
    """Find the first ``HITS[x]`` questions at rank 1; ``y`` is ignored."""
    return 1 if int(query.split()[-1]) < HITS[params.get("x", 0)] else 0


def by_x_and_y(params: Mapping[str, Any], query: str) -> int:
    """Like :func:`by_x`, and ``y == 'b'`` also finds the last question."""
    last = params.get("y") == "b" and query.endswith(str(N - 1))
    return 1 if last else by_x(params, query)


def wire(rule: RankRule = by_x) -> Wired:
    """Return an experiment service over a scripted objective."""
    factory = FakePipelineFactory(TARGETS, rule)
    store = InMemoryTrialStore()
    datasets = InMemoryDatasetRepository({"docs.json": dataset_json(
        [make_question(q, c) for q, c in TARGETS.items()]
    )})
    return ExperimentService(factory, datasets, store), factory, store


SUITE = Suite(name="sweep", base={"x": 0, "y": "a"},
              factors={"x": [0, 1, 2, 3], "y": ["a", "b"]})


def tune(suite: Suite = SUITE, rule: RankRule = by_x) -> TuningReport:
    """Tune ``suite`` on the scripted objective."""
    return Tuner(wire(rule)[0]).tune(suite, PATHS)


def test_tuner_finds_the_best_value_of_a_factor() -> None:
    """Coordinate descent moves ``x`` to the value finding most questions."""
    report = tune()
    assert report.suite == "sweep"
    assert report.best_params == {"x": 2, "y": "a"}
    assert report.baseline == pytest.approx(POINT * 0.4)
    assert report.in_sample == pytest.approx(POINT * 0.9)
    assert report.search_seconds_per_100 >= 0.0


def test_tuner_ties_keep_the_base_value() -> None:
    """A factor without any measurable effect is left where it was."""
    assert tune().best_params["y"] == "a"
    moved = SUITE.model_copy(update={"base": {"x": 0, "y": "b"}})
    assert tune(moved).best_params == {"x": 2, "y": "b"}
    flat = tune(rule=lambda params, query: 1)
    assert flat.best_params == SUITE.base
    assert flat.in_sample == flat.baseline == pytest.approx(POINT)


def test_tuner_keeps_a_base_that_is_already_the_best() -> None:
    """Starting from the optimum, nothing moves."""
    best = SUITE.model_copy(update={"base": {"x": 2, "y": "a"}})
    report = tune(best)
    assert report.best_params == {"x": 2, "y": "a"}
    assert report.in_sample == report.baseline


def test_tuner_tunes_several_factors() -> None:
    """Each factor that helps is moved, one after the other."""
    report = tune(rule=by_x_and_y)
    assert report.best_params == {"x": 2, "y": "b"}
    assert report.in_sample == pytest.approx(POINT)


def test_tuner_starts_factors_missing_from_the_base_at_first_value() -> None:
    """The first listed value is the default; other base keys are kept."""
    suite = Suite(name="s", base={"other": 7}, factors={"x": [3, 2, 0]})
    report = tune(suite)
    assert report.baseline == pytest.approx(POINT * 0.6)
    assert report.best_params == {"other": 7, "x": 2}


def test_tuner_report_has_cross_validated_estimates() -> None:
    """Each fold selects on its training part and is scored held-out."""
    report = tune()
    assert len(report.folds) == 5
    assert report.cv_estimate == pytest.approx(POINT * 0.9)
    assert report.cv_baseline == pytest.approx(POINT * 0.4)
    assert report.cv_estimate == pytest.approx(
        sum(f.test for f in report.folds) / 5
    )
    for fold in report.folds:
        assert fold.params == {"x": 2, "y": "a"}
        assert fold.baseline_test <= fold.test <= POINT
        assert fold.train in (
            pytest.approx(POINT * 7 / 8), pytest.approx(POINT),
        )


def test_tuner_number_of_folds_and_seed() -> None:
    """``folds`` sets the number of held-out parts; the seed the split."""
    tuner = Tuner(wire()[0])
    two = tuner.tune(SUITE, PATHS, folds=2, seed=0)
    assert len(two.folds) == 2
    assert two.cv_estimate == pytest.approx(POINT * 0.9)
    other = tuner.tune(SUITE, PATHS, folds=2, seed=1)
    assert other.best_params == two.best_params
    assert [f.test for f in other.folds] != [f.test for f in two.folds]


def test_tuner_evaluates_each_parameter_set_once() -> None:
    """Folds and passes share trials: one pipeline per distinct set."""
    service, factory, store = wire()
    report = Tuner(service).tune(SUITE, PATHS)
    tried = [(p["x"], p["y"]) for p in factory.created]
    assert sorted(tried) == [(0, "a"), (1, "a"), (2, "a"), (2, "b"), (3, "a")]
    assert report.trials_evaluated == 5
    assert {r.suite for r in store.all()} == {"sweep/tune"}
    assert len(store.all()) == 5


def test_tuner_resumes_trials_from_the_store() -> None:
    """A second tuner over the same store builds no pipeline."""
    service, factory, _ = wire()
    first = Tuner(service).tune(SUITE, PATHS)
    again = Tuner(service).tune(SUITE, PATHS)
    assert len(factory.created) == 5
    assert again == first


def test_tuner_min_gain_and_passes_limit_the_moves() -> None:
    """A gain below min_gain, or no pass at all, keeps the start."""
    service, _, _ = wire()
    assert Tuner(service, min_gain=1.0).tune(
        SUITE, PATHS
    ).best_params == SUITE.base
    assert Tuner(service).tune(
        SUITE, PATHS, passes=0
    ).best_params == SUITE.base


def test_tuner_score_is_recall_plus_mrr_minus_time() -> None:
    """The objective penalises seconds per 100 queries by time_weight."""
    service, _, _ = wire()
    trial = make_trial("t", {"docs": [[1], [0], [2], [0]]}, seconds=2.0)
    assert Tuner(service).score(trial) == pytest.approx(0.5375)
    assert Tuner(service, mrr_weight=0.0).score(trial) == pytest.approx(0.5)
    assert Tuner(service, time_weight=0.1).score(trial) == pytest.approx(
        0.3375
    )
    assert Tuner(service).score(trial, {"docs": [0, 1]}) == pytest.approx(
        0.55
    )


@pytest.mark.parametrize("time_weight, best, seconds", [
    (0.0, 1, 5.0), (0.05, 1, 5.0), (0.2, 0, 1.0),
])
def test_tuner_time_weight_penalises_slow_configurations(
    time_weight: float, best: int, seconds: float
) -> None:
    """A better but slower value is only chosen when the gain pays."""
    service, factory, store = wire()
    questions = service.load_questions(PATHS)
    for x, cost in ((0, 1.0), (1, 5.0)):
        spec = TrialSpec(name="stored", params={"x": x})
        store.append(make_trial(
            service.trial_id(spec, questions),
            {"docs": found_ranks(range(HITS[x + 1]), N)},
            params={"x": x}, seconds=cost,
        ))
    suite = Suite(name="s", base={"x": 0}, factors={"x": [0, 1]})
    report = Tuner(service, time_weight=time_weight).tune(suite, PATHS)
    assert factory.created == [], "both trials come from the store"
    assert report.best_params == {"x": best}
    assert report.search_seconds_per_100 == seconds
    assert report.in_sample == pytest.approx(
        POINT * HITS[best + 1] / N - time_weight * seconds
    )


def test_tuner_needs_factors() -> None:
    """A suite without factor has nothing to tune."""
    service, factory, _ = wire()
    with pytest.raises(InvalidInputError, match="no factor"):
        Tuner(service).tune(Suite(name="s", base={"x": 1}), PATHS)
    assert factory.created == []


@pytest.mark.parametrize("folds", [1, 0, -3])
def test_tuner_needs_two_folds(folds: int) -> None:
    """Cross-validation with fewer than two folds is rejected."""
    service, factory, _ = wire()
    with pytest.raises(InvalidInputError, match="folds must be at least 2"):
        Tuner(service).tune(SUITE, PATHS, folds=folds)
    assert factory.created == []


def test_tuner_reports_dataset_errors() -> None:
    """Dataset problems surface before any trial."""
    service, factory, _ = wire()
    with pytest.raises(InvalidInputError, match="No dataset given"):
        Tuner(service).tune(SUITE, {})
    assert factory.created == []


def test_tuner_rejects_a_factor_without_values() -> None:
    """A factor with no candidate value is a user error."""
    suite = Suite(name="s", factors={"x": []})
    with pytest.raises(InvalidInputError):
        tune(suite)


# -------------------------------------------------------------- router

CODE_QUESTIONS = [
    f"What is the default value of {name} in {cls}?" for name, cls in [
        ("max_tokens", "SamplingParams"), ("block_size", "CacheConfig"),
        ("num_workers", "EngineArgs"), ("top_k", "BeamSearch"),
        ("seed_value", "ModelConfig"), ("lora_rank", "LoraRequest"),
        ("gpu_fraction", "DeviceConfig"), ("swap_space", "SchedulerConfig"),
        ("max_len", "PromptAdapter"), ("log_level", "ServerArgs"),
    ]
]
DOCS_QUESTIONS = [
    f"How do you install {product} with Docker?" for product in [
        "vLLM", "the server", "the chart", "the proxy", "the gateway",
        "the client", "the plugin", "the worker", "the router", "the cache",
    ]
]


def router_datasets() -> InMemoryDatasetRepository:
    """Return a code dataset (.py sources) and a docs one (.md)."""
    code = [
        make_question(q, make_chunk("pkg/config.py", 10 * i, "x = 1"))
        for i, q in enumerate(CODE_QUESTIONS)
    ]
    docs = [
        make_question(q, make_chunk("docs/install.md", 10 * i, "Install"))
        for i, q in enumerate(DOCS_QUESTIONS)
    ]
    mixed = RagDataset(rag_questions=[
        *code[:2], *docs[:2], UnansweredQuestion(question="Not labelled?"),
    ])
    return InMemoryDatasetRepository({
        "code.json": dataset_json(code),
        "docs.json": dataset_json(docs),
        "mixed.json": mixed.model_dump_json(),
        "open.json": RagDataset(rag_questions=[
            UnansweredQuestion(question="Not labelled?"),
        ]).model_dump_json(),
    })


def trainer() -> Tuple[RouterTrainer, InMemoryRouterRepository]:
    """Return a trainer over the fake datasets and its repository."""
    repository = InMemoryRouterRepository()
    return RouterTrainer(router_datasets(), repository), repository


def test_router_trainer_labels_questions_by_python_sources() -> None:
    """Questions answered in .py files are code; the model is saved."""
    service, repository = trainer()
    report = service.train(["code.json", "docs.json"])
    assert report.n_questions == 20
    assert report.train_accuracy == 1.0
    assert report.path == "memory://router"
    assert repository.saves == 1
    model = repository.model
    assert model is not None
    assert all(model.p_code(q) > 0.5 for q in CODE_QUESTIONS)
    assert all(model.p_code(q) < 0.5 for q in DOCS_QUESTIONS)


def test_router_trainer_reports_cross_validated_accuracy() -> None:
    """Held-out questions of a separable problem are routed correctly."""
    service, _ = trainer()
    report = service.train(["code.json", "docs.json"], folds=4)
    assert 0.9 <= report.cv_accuracy <= 1.0
    assert report == trainer()[0].train(["code.json", "docs.json"], folds=4)


def test_router_trainer_report_names_every_weight() -> None:
    """Weights are reported by feature name and vocabulary word."""
    service, repository = trainer()
    report = service.train(["code.json", "docs.json"])
    assert repository.model is not None
    names: List[str] = list(FEATURE_NAMES) + repository.model.vocabulary
    assert set(report.weights) == set(names)
    assert report.weights["snake_case"] > 0 > report.weights["docker"]


def test_router_trainer_skips_questions_without_references() -> None:
    """Unanswered questions have no label and are not counted."""
    service, _ = trainer()
    report = service.train(["mixed.json"], folds=2)
    assert report.n_questions == 4


@pytest.mark.parametrize("paths", [
    ["code.json"], ["docs.json"], ["open.json"], [],
])
def test_router_trainer_needs_both_classes(paths: List[str]) -> None:
    """Training on code only, docs only or nothing is a user error."""
    service, repository = trainer()
    with pytest.raises(InvalidInputError, match="Router training needs"):
        service.train(paths)
    assert repository.saves == 0


def test_router_trainer_rejects_a_single_fold() -> None:
    """Cross-validation with one fold is a user error, like the tuner."""
    service, _ = trainer()
    with pytest.raises(InvalidInputError):
        service.train(["code.json", "docs.json"], folds=1)


def test_consensus_keeps_changes_most_folds_agree_on() -> None:
    """A value chosen by enough folds replaces the starting value."""
    start = {"x": 0, "y": "a", "z": 1.0}
    folds = [
        {"x": 2, "y": "a", "z": 0.5}, {"x": 2, "y": "b", "z": 0.6},
        {"x": 2, "y": "a", "z": 0.7}, {"x": 2, "y": "a", "z": 0.5},
        {"x": 3, "y": "a", "z": 0.9},
    ]
    assert consensus(start, folds) == {"x": 2, "y": "a", "z": 1.0}


def test_consensus_drops_changes_of_a_few_folds() -> None:
    """Changes selected by a minority of folds are treated as noise."""
    start = {"x": 0}
    folds = [{"x": 1}, {"x": 1}, {"x": 1}, {"x": 0}, {"x": 0}]
    assert consensus(start, folds) == {"x": 0}
    assert consensus(start, folds, min_folds=0.6) == {"x": 1}
    assert consensus(start, []) == {"x": 0}


def test_tuner_report_has_stability_and_consensus() -> None:
    """The report counts the value chosen per fold and the consensus."""
    report = tune()
    assert report.stability["x"] == {"2": 5}
    assert report.stability["y"] == {"a": 5}
    assert report.consensus_params == {"x": 2, "y": "a"}
    assert report.consensus_score == pytest.approx(report.in_sample)
