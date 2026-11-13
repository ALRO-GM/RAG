"""ExperimentService with a scripted pipeline factory and fake stores."""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from src.application.experiments import (
    ExperimentService,
    Questions,
    dataset_name,
)
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.experiments import Suite, TrialSpec
from src.domain.models import (
    AnsweredQuestion,
    Chunk,
    RagDataset,
    UnansweredQuestion,
)
from tests.fakes import (
    FakePipelineFactory,
    InMemoryDatasetRepository,
    InMemoryTrialStore,
    RankRule,
    RecordingProgress,
    dataset_json,
    make_chunk,
    make_question,
)

DOCS = [
    make_chunk("docs/a.md", 100 * i, f"documentation paragraph {i}")
    for i in range(3)
]
CODE = [
    make_chunk("pkg/a.py", 100 * i, f"def function_{i}(): pass",
               kind="python")
    for i in range(2)
]
TARGETS: Dict[str, Chunk] = {
    **{f"docs question {i}": c for i, c in enumerate(DOCS)},
    **{f"code question {i}": c for i, c in enumerate(CODE)},
}
PATHS = {"docs": "docs.json", "code": "code.json"}

Wired = Tuple[ExperimentService, FakePipelineFactory, InMemoryTrialStore]


def by_position(params: Mapping[str, Any], query: str) -> int:
    """Rank question ``i`` at ``i + shift`` (0 when beyond ``limit``)."""
    rank = int(query.split()[-1]) + int(params.get("shift", 1))
    return rank if rank <= int(params.get("limit", 99)) else 0


def repository() -> InMemoryDatasetRepository:
    """Return a repository holding the docs and code datasets."""
    return InMemoryDatasetRepository({
        "docs.json": dataset_json([
            make_question(q, c) for q, c in TARGETS.items()
            if q.startswith("docs")
        ]),
        "code.json": dataset_json([
            make_question(q, c) for q, c in TARGETS.items()
            if q.startswith("code")
        ]),
    })


def wire(
    rule: Optional[RankRule] = by_position, k: int = 10, repeats: int = 1
) -> Wired:
    """Return a service over fakes, with its factory and its store."""
    factory = FakePipelineFactory(TARGETS, rule)
    store = InMemoryTrialStore()
    service = ExperimentService(
        factory, repository(), store, RecordingProgress(), k, repeats
    )
    return service, factory, store


def questions(service: ExperimentService) -> Questions:
    """Load both datasets."""
    return service.load_questions(PATHS)


@pytest.mark.parametrize("path, name", [
    ("data/datasets/AnsweredQuestions/dataset_docs_public.json", "docs"),
    ("dataset_code_public.json", "code"),
    ("some/dir/questions.json", "questions"),
    ("dataset_my_set_public.json", "my_set"),
    ("dataset_public.json", "dataset_public"),
    ("dataset.json", "dataset"),
])
def test_dataset_name(path: str, name: str) -> None:
    """The 'dataset' and 'public' parts of the file name are dropped."""
    assert dataset_name(path) == name


def test_load_questions_keeps_questions_with_references() -> None:
    """Unanswered questions and answers without source are skipped."""
    service, _, _ = wire()
    mixed = RagDataset(rag_questions=[
        make_question("kept", DOCS[0]),
        UnansweredQuestion(question="no answer"),
        AnsweredQuestion(question="no source", sources=[], answer="a"),
    ])
    assert isinstance(service.datasets, InMemoryDatasetRepository)
    service.datasets.files["mixed.json"] = mixed.model_dump_json()
    loaded = service.load_questions({"m": "mixed.json", "d": "docs.json"})
    assert list(loaded) == ["m", "d"]
    assert [q.question for q in loaded["m"]] == ["kept"]
    assert len(loaded["d"]) == 3


def test_load_questions_rejects_a_dataset_without_references() -> None:
    """A dataset with nothing to evaluate is a user error naming it."""
    service, _, _ = wire()
    assert isinstance(service.datasets, InMemoryDatasetRepository)
    service.datasets.files["open.json"] = RagDataset(rag_questions=[
        UnansweredQuestion(question="no answer"),
    ]).model_dump_json()
    with pytest.raises(InvalidInputError, match="open.json has no question"):
        service.load_questions({"docs": "docs.json", "open": "open.json"})


def test_load_questions_needs_a_dataset() -> None:
    """No dataset at all is a user error."""
    service, _, _ = wire()
    with pytest.raises(InvalidInputError, match="No dataset given"):
        service.load_questions({})


def test_load_questions_propagates_repository_errors() -> None:
    """A missing dataset file is reported by the repository."""
    service, _, _ = wire()
    with pytest.raises(NotFoundError, match="missing.json"):
        service.load_questions({"docs": "missing.json"})


def test_trial_id_depends_on_params_and_datasets_not_on_the_name() -> None:
    """The same parameters on the same questions are the same trial."""
    service, _, _ = wire()
    loaded = questions(service)
    first = service.trial_id(TrialSpec(name="a", params={"x": 1}), loaded)
    assert first == service.trial_id(
        TrialSpec(name="b", params={"x": 1}), loaded
    )
    assert first != service.trial_id(
        TrialSpec(name="a", params={"x": 2}), loaded
    )
    assert first != service.trial_id(
        TrialSpec(name="a", params={"x": 1}), {"docs": loaded["docs"]}
    )
    assert first != service.trial_id(
        TrialSpec(name="a", params={"x": 1}),
        {"docs": loaded["docs"][:2], "code": loaded["code"]},
    )


def test_run_trial_records_the_rank_of_every_question() -> None:
    """Ranks, metrics and index figures describe the evaluated pipeline."""
    service, factory, store = wire()
    spec = TrialSpec(name="shifted", params={"shift": 3})
    loaded = questions(service)
    result = service.run_trial("my-suite", spec, loaded)
    assert result.ranks == {"docs": [[3], [4], [5]], "code": [[3], [4]]}
    assert result.metrics["docs"]["recall@3"] == pytest.approx(1 / 3)
    assert result.metrics["docs"]["recall@5"] == 1.0
    assert result.metrics["code"]["recall@1"] == 0.0
    assert result.metrics["code"]["mrr"] == pytest.approx((1 / 3 + 1 / 4) / 2)
    assert (result.suite, result.name) == ("my-suite", "shifted")
    assert result.params == {"shift": 3}
    assert result.trial_id == service.trial_id(spec, loaded)
    assert result.baseline_id == result.trial_id
    assert (result.index_seconds, result.index_cold) == (1.5, True)
    assert result.n_chunks == factory.n_chunks
    assert result.search_seconds_per_100 >= 0.0
    assert result.created.endswith("+00:00")
    assert store.all() == [result]
    assert store.get(result.trial_id) == result


def test_run_trial_keeps_the_given_baseline_id() -> None:
    """A trial of a suite points to the baseline of that suite."""
    service, _, _ = wire()
    result = service.run_trial(
        "s", TrialSpec(name="t"), questions(service), baseline_id="base-id"
    )
    assert result.baseline_id == "base-id"


def test_run_trial_ranks_are_limited_to_the_search_depth() -> None:
    """References beyond ``k`` results count as not found."""
    service, factory, _ = wire(k=3)
    result = service.run_trial(
        "s", TrialSpec(name="t", params={"shift": 2}), questions(service)
    )
    assert result.ranks == {"docs": [[2], [3], [0]], "code": [[2], [3]]}


def test_run_trial_ranks_every_reference_of_a_question() -> None:
    """A question with two references gets one rank per reference."""
    service, factory, _ = wire()
    absent = make_chunk("docs/other.md", 0, "never retrieved")
    loaded: Questions = {
        "docs": [make_question("docs question 1", DOCS[1], absent)],
    }
    result = service.run_trial("s", TrialSpec(name="t"), loaded)
    assert result.ranks == {"docs": [[2, 0]]}
    assert result.metrics["docs"]["recall@5"] == 0.5


@pytest.mark.parametrize("repeats, timed", [(1, 1), (3, 3), (0, 1)])
def test_run_trial_builds_one_pipeline_and_times_repeats(
    repeats: int, timed: int
) -> None:
    """One index build, one warm-up query, then the timed passes."""
    service, factory, _ = wire(repeats=repeats)
    service.run_trial("s", TrialSpec(name="t", params={"shift": 1}),
                      questions(service))
    assert factory.created == [{"shift": 1}]
    assert factory.index_calls == 1
    assert factory.search_calls[0] == ["docs question 0"]
    assert len(factory.search_calls) == 1 + 2 * timed
    assert factory.search_calls[1] == [f"docs question {i}" for i in range(3)]
    assert factory.search_calls[2] == [f"code question {i}" for i in range(2)]


def test_run_trial_resume_reuses_the_stored_result() -> None:
    """An already evaluated trial is read back without any pipeline."""
    service, factory, store = wire()
    loaded = questions(service)
    spec = TrialSpec(name="t", params={"shift": 2})
    first = service.run_trial("s", spec, loaded)
    again = service.run_trial("s", spec, loaded)
    assert again == first
    assert len(factory.created) == 1
    assert factory.index_calls == 1
    assert store.appended == 1


def test_run_trial_resume_is_shared_through_the_store() -> None:
    """Another service over the same store resumes the same trials."""
    service, _, store = wire()
    spec = TrialSpec(name="t", params={"shift": 2})
    first = service.run_trial("s", spec, questions(service))
    factory = FakePipelineFactory(TARGETS, by_position)
    other = ExperimentService(factory, repository(), store)
    assert other.run_trial("s", spec, questions(other)) == first
    assert factory.created == []


def test_run_trial_without_resume_measures_again() -> None:
    """resume=False runs the pipeline and stores a fresh result."""
    service, factory, store = wire()
    loaded = questions(service)
    spec = TrialSpec(name="t", params={"shift": 2})
    first = service.run_trial("s", spec, loaded)
    factory.rule = lambda params, query: 1
    again = service.run_trial("s", spec, loaded, resume=False)
    assert len(factory.created) == 2
    assert again.trial_id == first.trial_id
    assert again.ranks["docs"] == [[1], [1], [1]]
    assert store.appended == 2
    assert store.all() == [again], "the latest result replaces the first"


def test_run_trial_propagates_factory_errors() -> None:
    """A parameter set rejected by the factory stores nothing."""
    service, factory, store = wire()

    def reject(params: Mapping[str, Any]) -> Any:
        """Refuse every parameter set."""
        raise InvalidInputError("Unknown parameter(s): nope.")

    factory.create = reject  # type: ignore[method-assign]
    with pytest.raises(InvalidInputError, match="nope"):
        service.run_trial("s", TrialSpec(name="t", params={"nope": 1}),
                          questions(service))
    assert store.all() == []


SUITE = Suite(name="shift-sweep", base={"shift": 1},
              factors={"shift": [1, 2, 6], "limit": [2]})


def test_run_suite_evaluates_the_baseline_and_every_trial() -> None:
    """Results follow Suite.expand and all point to the first trial."""
    service, factory, store = wire()
    results = service.run_suite(SUITE, PATHS)
    assert [r.name for r in results] == [
        "baseline", "shift=2", "shift=6", "limit=2",
    ]
    assert {r.suite for r in results} == {"shift-sweep"}
    assert {r.baseline_id for r in results} == {results[0].trial_id}
    assert len({r.trial_id for r in results}) == 4
    assert factory.created == [
        {"shift": 1}, {"shift": 2}, {"shift": 6}, {"shift": 1, "limit": 2},
    ]
    assert store.all() == results
    recalls: List[float] = [r.mean_metric("recall@5") for r in results]
    assert recalls == pytest.approx([1.0, 1.0, 0.0, 5 / 6])


def test_run_suite_reports_progress_once_per_suite() -> None:
    """The trials are tracked under the name of the suite."""
    service, _, _ = wire()
    service.run_suite(SUITE, PATHS)
    assert isinstance(service.progress, RecordingProgress)
    assert service.progress.loops == [("shift-sweep", 4)]


def test_run_suite_resume_only_runs_the_new_trials() -> None:
    """A second run evaluates nothing; a grown suite only its new trial."""
    service, factory, store = wire()
    first = service.run_suite(SUITE, PATHS)
    assert service.run_suite(SUITE, PATHS) == first
    assert len(factory.created) == 4
    grown = SUITE.model_copy(update={
        "factors": {"shift": [1, 2, 6, 3], "limit": [2]},
    })
    results = service.run_suite(grown, PATHS)
    assert factory.created[4:] == [{"shift": 3}]
    assert len(results) == len(store.all()) == 5


def test_run_suite_without_resume_runs_everything_again() -> None:
    """resume=False re-evaluates the stored trials too."""
    service, factory, store = wire()
    service.run_suite(SUITE, PATHS)
    service.run_suite(SUITE, PATHS, resume=False)
    assert len(factory.created) == 8
    assert len(store.all()) == 4


def test_run_suite_rejects_bad_datasets_before_any_trial() -> None:
    """Dataset problems are reported without building a pipeline."""
    service, factory, _ = wire()
    with pytest.raises(NotFoundError):
        service.run_suite(SUITE, {"docs": "missing.json"})
    with pytest.raises(InvalidInputError):
        service.run_suite(SUITE, {})
    assert factory.created == []


def test_run_suite_sets_baseline_on_trials_resumed_from_another_suite(
) -> None:
    """Every result of a suite belongs to it and points to its baseline."""
    service, _, _ = wire()
    service.run_trial("shift-sweep/tune", TrialSpec(
        name="shift=2", params={"shift": 2}
    ), questions(service))
    results = service.run_suite(SUITE, PATHS)
    assert {r.baseline_id for r in results} == {results[0].trial_id}
    assert {r.suite for r in results} == {"shift-sweep"}
