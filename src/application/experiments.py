"""Use case: run experiment suites and record per-question results."""

import time
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Dict, List, Mapping, Optional

from src.application.ports import (
    DatasetRepository,
    NoProgress,
    PipelineFactory,
    Progress,
    TrialStore,
)
from src.domain.errors import InvalidInputError
from src.domain.experiments import (
    Ranks,
    Suite,
    TrialResult,
    TrialSpec,
    params_id,
    rank_metrics,
    source_ranks,
)
from src.domain.models import AnsweredQuestion, RagDataset

LAB_VERSION = "3"
"""Bump when pipeline code changes make stored results obsolete."""

Questions = Dict[str, List[AnsweredQuestion]]


def dataset_name(path: str) -> str:
    """Return a short name: ``dataset_docs_public.json`` → ``docs``."""
    stem = PurePath(path).stem
    parts = [p for p in stem.split("_") if p not in ("dataset", "public")]
    return "_".join(parts) or stem


class ExperimentService:
    """Evaluate parameter sets on datasets with reference spans.

    Results are stored with the rank of every question. A trial whose
    parameters were already evaluated is read back from the store, so
    suites, sweeps and the tuner share work.
    """

    def __init__(
        self,
        factory: PipelineFactory,
        datasets: DatasetRepository,
        store: TrialStore,
        progress: Optional[Progress] = None,
        k: int = 10,
        repeats: int = 1,
    ):
        """Wire the ports; ``k`` is the depth of the stored ranks."""
        self.factory = factory
        self.datasets = datasets
        self.store = store
        self.progress = progress or NoProgress()
        self.k = k
        self.repeats = max(1, repeats)

    def load_questions(self, dataset_paths: Mapping[str, str]) -> Questions:
        """Read the answered questions of each dataset."""
        questions: Questions = {}
        for name, path in dataset_paths.items():
            dataset = self.datasets.read(path, RagDataset)
            answered = [
                q for q in dataset.rag_questions
                if isinstance(q, AnsweredQuestion) and q.sources
            ]
            if not answered:
                raise InvalidInputError(
                    f"{path} has no question with reference sources."
                )
            questions[name] = answered
        if not questions:
            raise InvalidInputError("No dataset given.")
        return questions

    def trial_id(self, spec: TrialSpec, questions: Questions) -> str:
        """Identify a trial by its parameters and its datasets."""
        content = {
            name: [
                (q.question, [s.model_dump() for s in q.sources])
                for q in qs
            ]
            for name, qs in sorted(questions.items())
        }
        return params_id(spec.params, LAB_VERSION + params_id(content))

    def run_trial(
        self,
        suite: str,
        spec: TrialSpec,
        questions: Questions,
        baseline_id: str = "",
        resume: bool = True,
    ) -> TrialResult:
        """Evaluate one parameter set (or reuse its stored result)."""
        trial_id = self.trial_id(spec, questions)
        if resume:
            stored = self.store.get(trial_id)
            if stored is not None:
                return self._relabel(
                    stored, suite, spec.name, baseline_id or trial_id
                )
        pipeline = self.factory.create(spec.params)
        report, cold = pipeline.index()
        first = next(iter(questions.values()))[0].question
        pipeline.search([first], self.k)
        ranks: Dict[str, Ranks] = {}
        best = float("inf")
        for _ in range(self.repeats):
            start = time.perf_counter()
            found = {
                name: pipeline.search([q.question for q in qs], self.k)
                for name, qs in questions.items()
            }
            best = min(best, time.perf_counter() - start)
        for name, qs in questions.items():
            ranks[name] = [
                source_ranks(q.sources, [h.chunk.to_source() for h in hits])
                for q, hits in zip(qs, found[name])
            ]
        total = sum(len(qs) for qs in questions.values())
        result = TrialResult(
            trial_id=trial_id,
            suite=suite,
            name=spec.name,
            params=dict(spec.params),
            baseline_id=baseline_id or trial_id,
            metrics={n: rank_metrics(r) for n, r in ranks.items()},
            ranks=ranks,
            search_seconds_per_100=round(best / total * 100, 3),
            index_seconds=report.seconds,
            index_cold=cold,
            n_chunks=report.chunks,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.store.append(result)
        return result

    def _relabel(
        self, stored: TrialResult, suite: str, name: str, baseline_id: str
    ) -> TrialResult:
        """Record a reused measurement under the suite that asked for it.

        The same parameters can belong to several suites; each suite
        keeps its own row, pointing at its own baseline.
        """
        if (stored.suite, stored.name, stored.baseline_id) == (
            suite, name, baseline_id
        ):
            return stored
        copy = stored.model_copy(update={
            "suite": suite, "name": name, "baseline_id": baseline_id,
        })
        self.store.append(copy)
        return copy

    def run_suite(
        self,
        suite: Suite,
        dataset_paths: Mapping[str, str],
        resume: bool = True,
    ) -> List[TrialResult]:
        """Run every trial of ``suite``; the first one is its baseline."""
        questions = self.load_questions(dataset_paths)
        specs = suite.expand()
        baseline_id = self.trial_id(specs[0], questions)
        return [
            self.run_trial(suite.name, spec, questions, baseline_id, resume)
            for spec in self.progress.track(specs, len(specs), suite.name)
        ]
