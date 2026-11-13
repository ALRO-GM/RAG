"""Inbound adapter: experiment lab commands (``python -m src lab ...``)."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from pydantic import ValidationError

from src.application.analysis import AnalysisService, render_markdown
from src.application.experiments import ExperimentService, dataset_name
from src.application.tuning import Tuner, TuningReport
from src.domain.errors import (
    InvalidInputError,
    NotFoundError,
    short_reason,
)
from src.domain.experiments import Suite

DEFAULT_DATASETS = (
    "data/datasets/AnsweredQuestions/dataset_docs_public.json,"
    "data/datasets/AnsweredQuestions/dataset_code_public.json"
)


class LabSettings(Protocol):
    """The settings the lab commands read."""

    @property
    def lab_dir(self) -> str:
        """Directory of suites, results and profiles."""
        ...


class LabApp(Protocol):
    """What a composition root must offer to host the lab commands.

    Both the RAG container and the soft-skills container do, so the
    same commands measure document search and evidence retrieval.
    """

    @property
    def settings(self) -> LabSettings:
        """Settings with the lab directory."""
        ...

    @property
    def experiments(self) -> ExperimentService:
        """Experiment runner."""
        ...

    @property
    def tuner(self) -> Tuner:
        """Parameter tuner."""
        ...

    @property
    def analysis(self) -> AnalysisService:
        """Trial analysis."""
        ...


def parse_paths(value: Any, option: str = "datasets") -> List[str]:
    """Parse a comma-separated list (or a Fire tuple) of paths."""
    raw = value if isinstance(value, (tuple, list)) else str(value).split(",")
    paths = [str(p).strip() for p in raw if str(p).strip()]
    if not paths:
        raise InvalidInputError(f"--{option} must list at least one file.")
    return paths


def load_suite(suite: Any, lab_dir: str) -> Suite:
    """Read a suite by name (``<lab_dir>/suites/<name>.json``) or path."""
    text = str(suite)
    path = Path(text)
    if not path.is_file():
        path = Path(lab_dir) / "suites" / f"{text}.json"
    if not path.is_file():
        raise NotFoundError(f"Suite not found: {text}")
    try:
        return Suite.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise InvalidInputError(
            f"Invalid suite {path}: {short_reason(exc)}"
        )


def write_text(path: str, text: str) -> None:
    """Write a text file, creating its directory."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise InvalidInputError(f"Cannot write {path}: {exc}")


class LabCLI:
    """Experiment lab: measure, tune and analyse retrieval settings."""

    def __init__(
        self,
        app: Callable[[], LabApp],
        guard: Callable[[Callable[..., Any]], Callable[..., Any]],
        default_datasets: str = DEFAULT_DATASETS,
    ):
        """Use the container built by the calling CLI."""
        self._app = app
        self._default_datasets = default_datasets
        self.experiment = guard(self._experiment)
        self.tune = guard(self._tune)
        self.analyze = guard(self._analyze)
        self.train_router = guard(self._train_router)

    def _datasets(self, datasets: Any) -> Dict[str, str]:
        """Map dataset names to their paths."""
        named: Dict[str, str] = {}
        for path in parse_paths(datasets or self._default_datasets):
            name = dataset_name(path)
            if name in named:
                raise InvalidInputError(
                    f"Two datasets are named {name!r}: {named[name]} and "
                    f"{path}. Rename one of the files."
                )
            named[name] = path
        return named

    def _experiment(
        self,
        suite: str,
        datasets: str = "",
        resume: bool = True,
    ) -> None:
        """Run every trial of a suite and store the results.

        Args:
            suite: Suite name (experiments/suites/NAME.json) or path.
            datasets: Comma-separated answered datasets.
            resume: Reuse stored results of identical trials.
        """
        app = self._app()
        spec = load_suite(suite, app.settings.lab_dir)
        results = app.experiments.run_suite(
            spec, self._datasets(datasets), bool(resume)
        )
        base = results[0]
        print(f"Suite {spec.name}: {len(results)} trials")
        for r in results:
            recalls = " ".join(
                f"{n}={m['recall@5']:.3f}" for n, m in r.metrics.items()
            )
            delta = 100 * (
                r.mean_metric("recall@5") - base.mean_metric("recall@5")
            )
            print(
                f"  {r.name:<44} R@5 {recalls} ({delta:+.1f}) "
                f"{r.search_seconds_per_100:.2f} s/100q"
            )

    def _tune(
        self,
        suite: str,
        datasets: str = "",
        folds: int = 5,
        passes: int = 3,
        time_weight: float = 0.0,
        save_profile: str = "",
    ) -> None:
        """Tune the factors of a suite with cross-validation.

        Args:
            suite: Suite name or path; its factors are the search space.
            datasets: Comma-separated answered datasets.
            folds: Number of cross-validation folds.
            passes: Maximum coordinate-descent passes.
            time_weight: Objective penalty per second per 100 questions.
            save_profile: Write the consensus parameters (the changes
                most folds agree on) to this JSON file.
        """
        app = self._app()
        spec = load_suite(suite, app.settings.lab_dir)
        tuner = app.tuner
        tuner.time_weight = float(time_weight)
        report = tuner.tune(
            spec, self._datasets(datasets), int(folds), int(passes)
        )
        write_text(
            f"{app.settings.lab_dir}/results/tuning_{spec.name}.json",
            report.model_dump_json(indent=2),
        )
        print(f"Tuned {spec.name} ({report.trials_evaluated} trials)")
        print(f"  best parameters: {json.dumps(report.best_params)}")
        print(f"  objective in-sample: {report.baseline:.4f} → "
              f"{report.in_sample:.4f}")
        print(f"  objective held-out:  {report.cv_baseline:.4f} → "
              f"{report.cv_estimate:.4f}")
        print(f"  consensus parameters: "
              f"{json.dumps(report.consensus_params)}")
        print(f"  consensus objective in-sample: "
              f"{report.consensus_score:.4f}")
        if save_profile:
            write_text(
                str(save_profile),
                json.dumps(report.consensus_params, indent=2),
            )
            print(f"  consensus profile saved to {save_profile}")

    def _analyze(self, output: str = "") -> None:
        """Analyse every stored trial and write the report.

        Args:
            output: Markdown report path (default:
                experiments/REPORT.md); profiles go next to it.
        """
        app = self._app()
        lab = app.settings.lab_dir
        tunings: List[TuningReport] = []
        for path in sorted(Path(lab, "results").glob("tuning_*.json")):
            try:
                tunings.append(TuningReport.model_validate_json(
                    path.read_text(encoding="utf-8")
                ))
            except (OSError, ValidationError, ValueError):
                raise InvalidInputError(f"Invalid tuning report: {path}")
        report = app.analysis.analyze(tunings)
        target = str(output) if output else f"{lab}/REPORT.md"
        write_text(target, render_markdown(report))
        for profile in report.profiles:
            write_text(
                f"{lab}/profiles/{profile.name}.json",
                json.dumps(profile.params, indent=2),
            )
        print(f"Analysed {report.n_trials} trials → {target}")
        for p in report.profiles:
            print(f"  {p.name:<10} R@5 {p.mean_recall5:.3f} "
                  f"{p.search_seconds:.2f} s/100q  {p.params}")

    def _train_router(
        self, datasets: str = "", folds: int = 5
    ) -> None:
        """Train the code-or-docs query router and report its accuracy."""
        trainer = getattr(self._app(), "router_trainer", None)
        if trainer is None:
            raise InvalidInputError(
                "This lab has no query router to train."
            )
        report = trainer.train(
            parse_paths(datasets or self._default_datasets), int(folds)
        )
        print(f"Router trained on {report.n_questions} questions → "
              f"{report.path}")
        print(f"  accuracy: train {report.train_accuracy:.3f}, "
              f"cross-validated {report.cv_accuracy:.3f}")
        strongest = sorted(
            report.weights.items(), key=lambda kv: -abs(kv[1])
        )[:12]
        for name, weight in strongest:
            print(f"  {name:<22} {weight:+.2f}")
