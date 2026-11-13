"""Use case: choose a configuration that generalises across corpora.

The 42 lab measures recall@5 on the vLLM questions; the public
benchmarks measure nDCG@10 on scientific text, arguments, finance and
code. Each corpus is one *dataset* here, and every metric is turned
into a relative change against the suite's baseline, so a point on
SciDocs and a point on the 42 code questions weigh the same.

A configuration is *generalist* if it improves on average without
losing much anywhere. Choosing it on the same datasets that score it
overfits, so the report also runs leave-one-dataset-out: pick the best
configuration on all datasets but one, then see how that choice does
on the one left out (the "regret" is what it misses against the best
configuration for that dataset).
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from src.application.benchmarking import BenchmarkResult
from src.domain.errors import InvalidInputError
from src.domain.experiments import TrialResult

LAB_METRIC = "recall@5"
BENCHMARK_METRIC = "ndcg10"
BASELINE = "baseline"


class ConfigScore(BaseModel):
    """One configuration across every dataset of the study."""

    name: str
    params: Dict[str, object]
    scores: Dict[str, float]
    relative: Dict[str, float]
    mean_relative: Optional[float]
    worst_relative: Optional[float]
    wins: int
    losses: int
    complete: bool
    index_seconds: float
    search_seconds: float


class HeldOut(BaseModel):
    """Leave-one-dataset-out: what the choice on the rest gives here."""

    dataset: str
    chosen: str
    chosen_relative: float
    best: str
    best_relative: float
    rank: int

    @property
    def regret(self) -> float:
        """How much the choice misses against the best for this dataset."""
        return self.best_relative - self.chosen_relative


class GeneralistReport(BaseModel):
    """Every configuration scored on every dataset, and the choices."""

    suite: str
    datasets: List[str]
    configs: List[ConfigScore]
    best: Optional[str]
    robust: Optional[str]
    max_loss: float
    held_out: List[HeldOut]

    def configs_complete(self) -> List[ConfigScore]:
        """Configurations measured on every dataset."""
        return [c for c in self.configs if c.complete]

    def config(self, name: str) -> ConfigScore:
        """Return a configuration by name."""
        for config in self.configs:
            if config.name == name:
                return config
        raise KeyError(name)


def lab_scores(
    trials: Sequence[TrialResult], prefix: str = "42:"
) -> Dict[str, Tuple[Dict[str, float], Dict[str, object], float, float]]:
    """``{trial name: (scores per dataset, params, index s, search s)}``.

    Search seconds are per 100 questions in the lab; they are kept as
    reported, since the study compares configurations, not corpora.
    """
    out: Dict[
        str, Tuple[Dict[str, float], Dict[str, object], float, float]
    ] = {}
    for trial in trials:
        scores = {
            f"{prefix}{name}": float(metrics[LAB_METRIC])
            for name, metrics in trial.metrics.items()
            if LAB_METRIC in metrics
        }
        out[trial.name] = (
            scores, dict(trial.params), trial.index_seconds,
            trial.search_seconds_per_100,
        )
    return out


class GeneralistAnalysis:
    """Combine lab trials and benchmark rows of one suite.

    Args:
        tolerance: Relative change below which a dataset counts as
            neither a win nor a loss (noise).
        max_loss: The robust choice may lose at most this much
            (relative) on any dataset.
    """

    def __init__(self, tolerance: float = 0.005, max_loss: float = 0.02):
        """Set the thresholds."""
        if tolerance < 0 or max_loss < 0:
            raise InvalidInputError("Thresholds must not be negative.")
        self.tolerance = tolerance
        self.max_loss = max_loss

    def analyse(
        self,
        suite: str,
        trials: Sequence[TrialResult],
        rows: Sequence[BenchmarkResult],
        order: Sequence[str] = (),
    ) -> GeneralistReport:
        """Score every configuration of ``suite`` on every dataset.

        ``order`` lists configuration names in suite order (others
        follow in order of appearance). The suite must have a
        ``baseline`` configuration: every change is measured against it.
        """
        table: Dict[str, Dict[str, float]] = {}
        params: Dict[str, Dict[str, object]] = {}
        index_s: Dict[str, float] = {}
        search_s: Dict[str, float] = {}
        names: List[str] = list(order)
        datasets: List[str] = []

        def add(name: str, dataset: str, value: float) -> None:
            if name not in names:
                names.append(name)
            if dataset not in datasets:
                datasets.append(dataset)
            table.setdefault(name, {})[dataset] = value

        for name, (scores, p, idx, srch) in lab_scores(
            [t for t in trials if t.suite == suite]
        ).items():
            params.setdefault(name, p)
            index_s[name] = index_s.get(name, 0.0) + idx
            search_s[name] = search_s.get(name, 0.0) + srch
            for dataset, value in scores.items():
                add(name, dataset, value)
        for row in rows:
            if row.suite != suite:
                continue
            params.setdefault(row.config, dict(row.params))
            index_s[row.config] = index_s.get(row.config, 0.0) + (
                row.index_seconds
            )
            search_s[row.config] = search_s.get(row.config, 0.0) + (
                row.search_seconds
            )
            add(row.config, row.dataset, row.ndcg10)
        if BASELINE not in table:
            raise InvalidInputError(
                f"Suite {suite!r} has no measured baseline."
            )
        base = table[BASELINE]
        configs = [
            self._score(
                n, table.get(n, {}), base, datasets, params.get(n, {}),
                index_s.get(n, 0.0), search_s.get(n, 0.0),
            )
            for n in names if n in table
        ]
        complete = [c for c in configs if c.complete]
        best = max(
            complete, key=lambda c: c.mean_relative or 0.0, default=None
        )
        safe = [
            c for c in complete
            if c.worst_relative is not None
            and c.worst_relative >= -self.max_loss
        ]
        robust = max(
            safe, key=lambda c: c.mean_relative or 0.0, default=None
        )
        return GeneralistReport(
            suite=suite,
            datasets=datasets,
            configs=configs,
            best=best.name if best else None,
            robust=robust.name if robust else None,
            max_loss=self.max_loss,
            held_out=self._leave_one_out(complete, datasets),
        )

    def _score(
        self,
        name: str,
        scores: Mapping[str, float],
        base: Mapping[str, float],
        datasets: Sequence[str],
        params: Dict[str, object],
        index_seconds: float,
        search_seconds: float,
    ) -> ConfigScore:
        """Relative changes of one configuration against the baseline."""
        relative = {
            d: (scores[d] - base[d]) / base[d]
            for d in datasets
            if d in scores and d in base and base[d] > 0
        }
        values = list(relative.values())
        return ConfigScore(
            name=name,
            params=params,
            scores=dict(scores),
            relative=relative,
            mean_relative=sum(values) / len(values) if values else None,
            worst_relative=min(values) if values else None,
            wins=sum(1 for v in values if v > self.tolerance),
            losses=sum(1 for v in values if v < -self.tolerance),
            complete=len(relative) == len(datasets) and bool(datasets),
            index_seconds=round(index_seconds, 1),
            search_seconds=round(search_seconds, 2),
        )

    @staticmethod
    def _leave_one_out(
        configs: Sequence[ConfigScore], datasets: Sequence[str]
    ) -> List[HeldOut]:
        """Choose on all datasets but one; score the choice on that one."""
        if len(configs) < 2 or len(datasets) < 2:
            return []
        out: List[HeldOut] = []
        for held in datasets:
            rest = [d for d in datasets if d != held]

            def mean_rest(c: ConfigScore) -> float:
                return sum(c.relative[d] for d in rest) / len(rest)

            chosen = max(configs, key=mean_rest)
            ranking = sorted(configs, key=lambda c: -c.relative[held])
            out.append(HeldOut(
                dataset=held,
                chosen=chosen.name,
                chosen_relative=chosen.relative[held],
                best=ranking[0].name,
                best_relative=ranking[0].relative[held],
                rank=1 + [c.name for c in ranking].index(chosen.name),
            ))
        return out


def render_generalist(report: GeneralistReport) -> str:
    """Markdown report of a generalist study."""

    def pct(value: Optional[float]) -> str:
        return "—" if value is None else f"{100 * value:+.1f}%"

    lines = [
        f"# Generalist study: `{report.suite}`",
        "",
        "Relative change against the suite baseline on each dataset "
        f"(42 datasets: {LAB_METRIC}; public ones: nDCG@10). "
        "Configurations sorted by mean relative change; only those "
        "measured on every dataset are ranked.",
        "",
    ]
    header = ["Configuration", "mean", "worst", "W/L", *report.datasets,
              "index s", "search s"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    ordered = sorted(
        report.configs,
        key=lambda c: (not c.complete, -(c.mean_relative or 0.0)),
    )
    for c in ordered:
        mark = ""
        if c.name == report.robust:
            mark = " **(robust)**"
        if c.name == report.best:
            mark += " **(best mean)**"
        cells = [
            f"{c.name}{mark}" + ("" if c.complete else " *(partial)*"),
            pct(c.mean_relative), pct(c.worst_relative),
            f"{c.wins}/{c.losses}",
            *[pct(c.relative.get(d)) for d in report.datasets],
            f"{c.index_seconds:.0f}", f"{c.search_seconds:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    base = next((c for c in report.configs if c.name == BASELINE), None)
    if base is not None:
        lines += [
            "",
            "Baseline scores: " + ", ".join(
                f"{d} {base.scores[d]:.3f}" for d in report.datasets
                if d in base.scores
            ) + ".",
        ]
    lines += [
        "",
        f"- Best mean: **{report.best or '—'}**.",
        f"- Robust choice (best mean losing at most "
        f"{100 * report.max_loss:.0f}% on every dataset): "
        f"**{report.robust or '—'}**.",
    ]
    if report.held_out:
        regrets = [h.regret for h in report.held_out]
        lines += [
            "",
            "## Leave one dataset out",
            "",
            "Chosen on the other datasets, scored on the one left out.",
            "",
            "| Held out | chosen | its change | rank | best there | "
            "regret |",
            "|---|---|---|---|---|---|",
        ]
        for h in report.held_out:
            lines.append(
                f"| {h.dataset} | {h.chosen} | {pct(h.chosen_relative)} "
                f"| {h.rank}/{len(report.configs_complete())} | {h.best} "
                f"({pct(h.best_relative)}) | {100 * h.regret:.1f} pts |"
            )
        lines += [
            "",
            f"Mean regret {100 * sum(regrets) / len(regrets):.1f} points; "
            f"the held-out choice improved on the baseline in "
            f"{sum(1 for h in report.held_out if h.chosen_relative > 0)}"
            f" of {len(report.held_out)} datasets.",
        ]
    return "\n".join(lines) + "\n"
