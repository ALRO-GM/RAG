"""Use cases: tune parameters with cross-validation, train the router."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.application.experiments import ExperimentService, Questions
from src.application.ports import DatasetRepository, RouterRepository
from src.domain.errors import InvalidInputError
from src.domain.experiments import (
    Suite,
    TrialResult,
    TrialSpec,
    kfold_indices,
    macro_mean,
    objective_scores,
)
from src.domain.models import AnsweredQuestion, RagDataset
from src.domain.priors import FEATURE_NAMES, RouterModel, fit_router

Subset = Optional[Dict[str, List[int]]]


class FoldResult(BaseModel):
    """What one cross-validation fold selected and how it generalised."""

    params: Dict[str, Any]
    train: float
    test: float
    baseline_test: float


class TuningReport(BaseModel):
    """Outcome of a tuning run.

    ``cv_estimate`` is the honest expectation on unseen questions: each
    fold selects its values on four fifths of the questions and is
    scored on the remaining fifth.
    """

    suite: str
    best_params: Dict[str, Any]
    in_sample: float
    baseline: float
    cv_estimate: float
    cv_baseline: float
    search_seconds_per_100: float
    folds: List[FoldResult] = Field(default_factory=list)
    trials_evaluated: int = 0
    stability: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    consensus_params: Dict[str, Any] = Field(default_factory=dict)
    consensus_score: float = 0.0


def consensus(
    start: Mapping[str, Any],
    fold_params: Sequence[Mapping[str, Any]],
    min_folds: float = 0.8,
    min_agreement: float = 0.6,
) -> Dict[str, Any]:
    """Keep only the changes most folds agree on.

    A factor leaves its starting value when at least ``min_folds`` of
    the folds changed it *and* at least ``min_agreement`` of them chose
    the same new value. Changes selected by a few folds only are
    treated as noise and dropped.
    """
    chosen: Dict[str, Any] = dict(start)
    n = len(fold_params)
    for factor, base in start.items():
        values = [p.get(factor, base) for p in fold_params]
        moved = [v for v in values if v != base]
        if not moved or len(moved) < min_folds * n:
            continue
        top = max(set(moved), key=moved.count)
        if moved.count(top) >= min_agreement * n:
            chosen[factor] = top
    return chosen


class Tuner:
    """Coordinate descent over the factors of a suite.

    One factor is changed at a time, keeping the value that maximises
    ``recall@5 + mrr_weight·MRR − time_weight·seconds per 100 queries``;
    passes are repeated until nothing changes. Ties keep the current
    value, so a factor only moves for a measurable gain.
    """

    def __init__(
        self,
        experiments: ExperimentService,
        mrr_weight: float = 0.1,
        time_weight: float = 0.0,
        min_gain: float = 1e-9,
    ):
        """Configure the objective."""
        self.experiments = experiments
        self.mrr_weight = mrr_weight
        self.time_weight = time_weight
        self.min_gain = min_gain
        self._memo: Dict[str, TrialResult] = {}
        self._baseline_id = ""

    def _evaluate(
        self, suite: str, params: Mapping[str, Any], questions: Questions
    ) -> TrialResult:
        """Run (or recall) the trial of a parameter set."""
        spec = TrialSpec(
            name=",".join(f"{k}={v}" for k, v in sorted(params.items())),
            params=dict(params),
        )
        key = self.experiments.trial_id(spec, questions)
        if key not in self._memo:
            self._memo[key] = self.experiments.run_trial(
                f"{suite}/tune", spec, questions, self._baseline_id
            )
        return self._memo[key]

    def score(self, result: TrialResult, subset: Subset = None) -> float:
        """Return the objective of a trial on a subset of questions."""
        scores = objective_scores(result.ranks, mrr_weight=self.mrr_weight)
        return (
            macro_mean(scores, subset)
            - self.time_weight * result.search_seconds_per_100
        )

    def _descend(
        self, suite: Suite, questions: Questions, subset: Subset, passes: int
    ) -> Tuple[Dict[str, Any], float]:
        """Return the best parameters found on ``subset``."""
        current: Dict[str, Any] = dict(suite.base)
        for factor, values in suite.factors.items():
            current.setdefault(factor, values[0])
        best = self.score(
            self._evaluate(suite.name, current, questions), subset
        )
        for _ in range(passes):
            changed = False
            for factor, values in suite.factors.items():
                for value in values:
                    if value == current[factor]:
                        continue
                    candidate = {**current, factor: value}
                    score = self.score(
                        self._evaluate(suite.name, candidate, questions),
                        subset,
                    )
                    if score > best + self.min_gain:
                        best, current, changed = score, candidate, True
            if not changed:
                break
        return current, best

    def tune(
        self,
        suite: Suite,
        dataset_paths: Mapping[str, str],
        folds: int = 5,
        passes: int = 3,
        seed: int = 0,
    ) -> TuningReport:
        """Tune the factors of ``suite`` and cross-validate the gain."""
        if not suite.factors:
            raise InvalidInputError("The suite has no factor to tune.")
        empty = [f for f, values in suite.factors.items() if not values]
        if empty:
            raise InvalidInputError(
                f"Factor(s) without values: {', '.join(empty)}."
            )
        if folds < 2:
            raise InvalidInputError("folds must be at least 2.")
        questions = self.experiments.load_questions(dataset_paths)
        start: Dict[str, Any] = dict(suite.base)
        for factor, values in suite.factors.items():
            start.setdefault(factor, values[0])
        self._memo = {}
        self._baseline_id = self.experiments.trial_id(
            TrialSpec(name="baseline", params=start), questions
        )
        baseline = self._evaluate(suite.name, start, questions)
        tests = {
            name: kfold_indices(len(qs), folds, seed)
            for name, qs in questions.items()
        }
        fold_results: List[FoldResult] = []
        for fold in range(folds):
            test = {name: tests[name][fold] for name in questions}
            train = {
                name: sorted(
                    set(range(len(questions[name]))) - set(test[name])
                )
                for name in questions
            }
            params, train_score = self._descend(
                suite, questions, train, passes
            )
            chosen = self._evaluate(suite.name, params, questions)
            fold_results.append(FoldResult(
                params=params,
                train=train_score,
                test=self.score(chosen, test),
                baseline_test=self.score(baseline, test),
            ))
        best_params, in_sample = self._descend(
            suite, questions, None, passes
        )
        final = self._evaluate(suite.name, best_params, questions)
        agreed = consensus(start, [f.params for f in fold_results])
        stability: Dict[str, Dict[str, int]] = {}
        for factor in suite.factors:
            counts: Dict[str, int] = {}
            for fold_result in fold_results:
                value = str(fold_result.params[factor])
                counts[value] = counts.get(value, 0) + 1
            stability[factor] = counts
        return TuningReport(
            stability=stability,
            consensus_params=agreed,
            consensus_score=self.score(
                self._evaluate(suite.name, agreed, questions)
            ),
            suite=suite.name,
            best_params=best_params,
            in_sample=in_sample,
            baseline=self.score(baseline),
            cv_estimate=sum(f.test for f in fold_results) / folds,
            cv_baseline=sum(f.baseline_test for f in fold_results) / folds,
            search_seconds_per_100=final.search_seconds_per_100,
            folds=fold_results,
            trials_evaluated=len(self._memo),
        )


class RouterReport(BaseModel):
    """Accuracy of the trained query router."""

    path: str
    n_questions: int
    train_accuracy: float
    cv_accuracy: float
    weights: Dict[str, float]


def _accuracy(
    model: RouterModel, queries: Sequence[str], labels: Sequence[int]
) -> float:
    """Return the share of queries routed to the right side."""
    hits = sum(
        1 for q, y in zip(queries, labels)
        if (model.p_code(q) >= 0.5) == bool(y)
    )
    return hits / len(queries) if queries else 0.0


class RouterTrainer:
    """Fit the query router on questions with reference sources.

    The label of a question is whether its first reference is a Python
    file. Accuracy is also estimated by cross-validation, to check that
    the model generalises to unseen questions.
    """

    def __init__(
        self, datasets: DatasetRepository, repository: RouterRepository
    ):
        """Wire the ports."""
        self.datasets = datasets
        self.repository = repository

    def train(
        self, dataset_paths: Sequence[str], folds: int = 5, seed: int = 0
    ) -> RouterReport:
        """Train, cross-validate and save the router."""
        if folds < 2:
            raise InvalidInputError("folds must be at least 2.")
        queries: List[str] = []
        labels: List[int] = []
        for path in dataset_paths:
            for q in self.datasets.read(path, RagDataset).rag_questions:
                if isinstance(q, AnsweredQuestion) and q.sources:
                    queries.append(q.question)
                    labels.append(
                        int(q.sources[0].file_path.endswith(".py"))
                    )
        if len(set(labels)) < 2:
            raise InvalidInputError(
                "Router training needs questions answered in code and "
                "questions answered in documentation."
            )
        correct = 0
        for test in kfold_indices(len(queries), folds, seed):
            held = set(test)
            model = fit_router(
                [q for i, q in enumerate(queries) if i not in held],
                [y for i, y in enumerate(labels) if i not in held],
            )
            correct += sum(
                1 for i in test
                if (model.p_code(queries[i]) >= 0.5) == bool(labels[i])
            )
        final = fit_router(queries, labels)
        return RouterReport(
            path=self.repository.save(final),
            n_questions=len(queries),
            train_accuracy=_accuracy(final, queries, labels),
            cv_accuracy=correct / len(queries),
            weights=dict(zip(
                list(FEATURE_NAMES) + final.vocabulary, final.weights
            )),
        )
