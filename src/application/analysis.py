"""Use case: turn stored trials into statistics and recommendations.

The analysis answers four questions:

1. What does each change gain, and is the gain real? (paired bootstrap
   against the baseline of its suite.)
2. What does it cost? (search seconds per 100 questions, index seconds,
   recall points per extra second.)
3. Do changes add up? (interaction of pairs of factors in grid suites.)
4. Which configuration to use for precision, for speed and for the best
   compromise? (Pareto front of recall against search time.)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.application.ports import TrialStore
from src.application.tuning import TuningReport
from src.domain.errors import NotFoundError
from src.domain.experiments import (
    TrialResult,
    macro_mean,
    objective_scores,
    paired_bootstrap,
    pareto_front,
    recall_scores,
    synergy,
)


TUNE_SUFFIX = "/tune"
"""Suites of trials explored by the tuner. They are search steps, not
candidates: picking the best of hundreds would reward luck."""
MAX_TUNE_ROWS = 10


class TrialRow(BaseModel):
    """One line of the cost/benefit table."""

    trial_id: str
    suite: str
    name: str
    params: Dict[str, Any]
    changed: Dict[str, Any]
    recall5: Dict[str, float]
    mean_recall5: float
    mrr: float
    objective: float
    delta_points: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_gain: float = 0.0
    verdict: str = "baseline"
    search_seconds: float = 0.0
    delta_seconds: float = 0.0
    points_per_second: Optional[float] = None
    index_seconds: float = 0.0
    index_cold: bool = False
    end_to_end_seconds: Optional[float] = None
    n_chunks: int = 0


class Interaction(BaseModel):
    """Whether two changes add up (points of mean recall@5)."""

    suite: str
    first: str
    second: str
    gain_first: float
    gain_second: float
    gain_both: float
    synergy: float


class Profile(BaseModel):
    """A recommended configuration."""

    name: str
    reason: str
    trial_id: str
    params: Dict[str, Any]
    mean_recall5: float
    recall5: Dict[str, float]
    search_seconds: float


class AnalysisReport(BaseModel):
    """Everything the analysis produces."""

    n_trials: int
    rows: List[TrialRow]
    interactions: List[Interaction] = Field(default_factory=list)
    pareto: List[str] = Field(default_factory=list)
    pareto_end_to_end: List[str] = Field(default_factory=list)
    profiles: List[Profile] = Field(default_factory=list)
    tunings: List[TuningReport] = Field(default_factory=list)


def _recall5(result: TrialResult) -> Dict[str, List[float]]:
    """Return the per-question recall@5 of each dataset."""
    return {n: recall_scores(r, 5) for n, r in result.ranks.items()}


def _changed(
    params: Dict[str, Any], base: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the parameters that differ from the baseline."""
    return {
        k: v for k, v in params.items()
        if k not in base or base[k] != v
    }


class AnalysisService:
    """Compute cost/benefit statistics from the trial store."""

    def __init__(
        self,
        store: TrialStore,
        mrr_weight: float = 0.1,
        max_search_seconds: float = 22.0,
        min_points_per_second: float = 0.5,
        required: Optional[Dict[str, float]] = None,
        margin: float = 0.05,
        time_resolution: float = 1.0,
        questions: int = 200,
    ):
        """Configure the recommendation rules.

        Args:
            store: Where the trials are.
            mrr_weight: Tie-break weight of MRR in the objective.
            max_search_seconds: Budget per 100 questions for the
                precision profile (22 s keeps 200 questions under half
                of the 90 s limit).
            min_points_per_second: The balanced profile only accepts a
                slower configuration when it gains at least this many
                recall points per extra second (per 100 questions).
            required: Minimum recall@5 per dataset.
            margin: Safety margin above ``required`` for the speed
                profile.
            time_resolution: Search times closer than this (seconds per
                100 questions) are measurement noise and count as equal.
            questions: Number of questions of the end-to-end scenario
                (cold indexing + that many searches).
        """
        self.store = store
        self.mrr_weight = mrr_weight
        self.max_search_seconds = max_search_seconds
        self.min_points_per_second = min_points_per_second
        self.required = required or {"docs": 0.80, "code": 0.50}
        self.margin = margin
        self.time_resolution = time_resolution
        self.questions = questions

    def _cost(self, row: TrialRow) -> float:
        """Return the search time rounded to the timing resolution."""
        step = self.time_resolution
        if step <= 0:
            return row.search_seconds
        return round(row.search_seconds / step) * step

    def _row(
        self, result: TrialResult, baseline: Optional[TrialResult]
    ) -> TrialRow:
        """Build the table line of a trial against its baseline."""
        scores = _recall5(result)
        row = TrialRow(
            trial_id=result.trial_id,
            suite=result.suite,
            name=result.name,
            params=result.params,
            changed={},
            recall5={n: m["recall@5"] for n, m in result.metrics.items()},
            mean_recall5=macro_mean(scores),
            mrr=result.mean_metric("mrr"),
            objective=macro_mean(
                objective_scores(result.ranks, mrr_weight=self.mrr_weight)
            ),
            search_seconds=result.search_seconds_per_100,
            index_seconds=result.index_seconds,
            index_cold=result.index_cold,
            n_chunks=result.n_chunks,
        )
        if result.index_cold:
            row.end_to_end_seconds = (
                result.index_seconds
                + result.search_seconds_per_100 * self.questions / 100
            )
        if baseline is None or baseline.trial_id == result.trial_id:
            return row
        delta, low, high, p_gain = paired_bootstrap(
            scores, _recall5(baseline)
        )
        row.changed = _changed(result.params, baseline.params)
        row.delta_points = 100 * delta
        row.ci_low, row.ci_high, row.p_gain = 100 * low, 100 * high, p_gain
        row.delta_seconds = (
            result.search_seconds_per_100 - baseline.search_seconds_per_100
        )
        if row.delta_seconds > 0.05:
            row.points_per_second = row.delta_points / row.delta_seconds
        if low > 0:
            row.verdict = "helps"
        elif high < 0:
            row.verdict = "hurts"
        elif delta > 0:
            row.verdict = "likely helps" if p_gain >= 0.8 else "neutral"
        elif delta < 0:
            row.verdict = "likely hurts" if p_gain <= 0.2 else "neutral"
        else:
            row.verdict = "neutral"
        return row

    def _interactions(
        self, rows: List[TrialRow], by_id: Dict[str, TrialResult]
    ) -> List[Interaction]:
        """Find pairs of changes measured alone and together."""
        singles: Dict[Tuple[str, str], TrialRow] = {}
        for row in rows:
            if len(row.changed) == 1:
                base = by_id[row.trial_id].baseline_id
                (factor, value), = row.changed.items()
                singles[(base, f"{factor}={value}")] = row
        found: List[Interaction] = []
        for row in rows:
            if len(row.changed) != 2:
                continue
            base = by_id[row.trial_id].baseline_id
            keys = [f"{f}={v}" for f, v in row.changed.items()]
            a, b = singles.get((base, keys[0])), singles.get((base, keys[1]))
            if a is None or b is None:
                continue
            found.append(Interaction(
                suite=row.suite,
                first=keys[0],
                second=keys[1],
                gain_first=a.delta_points,
                gain_second=b.delta_points,
                gain_both=row.delta_points,
                synergy=synergy(
                    0.0, a.delta_points, b.delta_points, row.delta_points
                ),
            ))
        return found

    def _profile(self, name: str, reason: str, row: TrialRow) -> Profile:
        """Describe a recommended configuration."""
        return Profile(
            name=name, reason=reason, trial_id=row.trial_id,
            params={
                k: v for k, v in row.params.items()
                if not k.startswith("lab_")
            },
            mean_recall5=row.mean_recall5,
            recall5=row.recall5, search_seconds=row.search_seconds,
        )

    def _profiles(
        self, rows: List[TrialRow], front: List[TrialRow]
    ) -> List[Profile]:
        """Pick the precision, balanced and speed configurations."""
        profiles: List[Profile] = []
        affordable = [
            r for r in rows if r.search_seconds <= self.max_search_seconds
        ]
        if affordable:
            best = max(
                affordable, key=lambda r: (r.objective, -self._cost(r))
            )
            profiles.append(self._profile(
                "precision",
                "highest objective within "
                f"{self.max_search_seconds:g} s per 100 questions",
                best,
            ))
        if front:
            chosen = front[0]
            for row in front[1:]:
                extra = self._cost(row) - self._cost(chosen)
                gain = 100 * (row.mean_recall5 - chosen.mean_recall5)
                if extra <= 0 or gain / extra >= self.min_points_per_second:
                    chosen = row
            profiles.append(self._profile(
                "balanced",
                "last point of the Pareto front that still gains "
                f"{self.min_points_per_second:g} recall points per extra "
                "second",
                chosen,
            ))
        safe = [
            r for r in rows
            if all(
                r.recall5.get(name, 0.0) >= need + self.margin
                for name, need in self.required.items()
                if name in r.recall5
            )
        ]
        timed = [r for r in safe if r.end_to_end_seconds is not None]
        if timed:
            fastest = min(
                timed,
                key=lambda r: (r.end_to_end_seconds or 0.0, -r.objective),
            )
            profiles.append(self._profile(
                "speed",
                f"fastest cold indexing + {self.questions} questions "
                f"({fastest.end_to_end_seconds:.0f} s) at least "
                f"{100 * self.margin:g} points above every required "
                "recall@5",
                fastest,
            ))
        elif safe:
            fastest = min(
                safe, key=lambda r: (self._cost(r), -r.objective)
            )
            profiles.append(self._profile(
                "speed",
                "fastest search at least "
                f"{100 * self.margin:g} points above every required "
                "recall@5",
                fastest,
            ))
        return profiles

    def analyze(
        self, tunings: Sequence[TuningReport] = ()
    ) -> AnalysisReport:
        """Return the full analysis of the stored trials."""
        results = self.store.all()
        if not results:
            raise NotFoundError(
                "No experiment results yet. Run 'lab experiment' first."
            )
        by_id = {r.trial_id: r for r in results}
        rows = [self._row(r, by_id.get(r.baseline_id)) for r in results]
        curated: Dict[str, TrialRow] = {}
        for candidate in rows:
            if candidate.suite.endswith(TUNE_SUFFIX):
                continue
            kept = curated.get(candidate.trial_id)
            if kept is None or (
                candidate.end_to_end_seconds is not None
                and kept.end_to_end_seconds is None
            ):
                curated[candidate.trial_id] = candidate
        unique = list(curated.values()) or rows
        front_ids = pareto_front(
            [(self._cost(r), round(r.mean_recall5, 4)) for r in unique]
        )
        front = [unique[i] for i in front_ids]
        timed = [r for r in unique if r.end_to_end_seconds is not None]
        timed_front = pareto_front(
            [(r.end_to_end_seconds or 0.0, round(r.mean_recall5, 4))
             for r in timed]
        )
        return AnalysisReport(
            n_trials=len(rows),
            rows=rows,
            interactions=self._interactions(rows, by_id),
            pareto=[r.trial_id for r in front],
            pareto_end_to_end=[timed[i].trial_id for i in timed_front],
            profiles=self._profiles(unique, front),
            tunings=list(tunings),
        )


def _fmt_changed(row: TrialRow) -> str:
    """Describe what a trial changes."""
    if not row.changed:
        return "baseline"
    return ", ".join(f"`{k}={v}`" for k, v in row.changed.items())


def _suite_table(rows: List[TrialRow], datasets: List[str]) -> List[str]:
    """Render the cost/benefit table of one suite."""
    head = (
        "| Change | " + " | ".join(f"R@5 {d}" for d in datasets)
        + " | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q "
        "| Δ s | points/s | Verdict |"
    )
    lines = [head, "|" + "---|" * (9 + len(datasets))]
    ordered = [r for r in rows if not r.changed] + sorted(
        (r for r in rows if r.changed), key=lambda r: -r.objective
    )
    for r in ordered:
        recalls = " | ".join(
            f"{r.recall5.get(d, 0.0):.3f}" for d in datasets
        )
        if r.changed:
            delta = (
                f"{r.delta_points:+.1f} "
                f"({r.ci_low:+.1f}, {r.ci_high:+.1f})"
            )
            p_gain = f"{r.p_gain:.2f}"
            d_time = f"{r.delta_seconds:+.2f}"
            rate = (
                "—" if r.points_per_second is None
                else f"{r.points_per_second:+.2f}"
            )
        else:
            delta = p_gain = d_time = rate = "—"
        lines.append(
            f"| {_fmt_changed(r)} | {recalls} | {r.mean_recall5:.3f} "
            f"| {r.mrr:.3f} | {delta} | {p_gain} "
            f"| {r.search_seconds:.2f} | {d_time} | {rate} | {r.verdict} |"
        )
    return lines


def _tuning_section(tuning: TuningReport) -> List[str]:
    """Render the cross-validation summary of a tuning run."""
    gain_in = 100 * (tuning.in_sample - tuning.baseline)
    gain_cv = 100 * (tuning.cv_estimate - tuning.cv_baseline)
    gain_agreed = 100 * (tuning.consensus_score - tuning.baseline)
    lines = [
        f"## Tuning: {tuning.suite}",
        "",
        "Objective = mean recall@5 + 0.1·MRR. *In-sample* tunes and "
        "scores on the same questions (optimistic). *Held-out* tunes on "
        "four fifths of the questions and scores on the rest: it is the "
        "gain to expect on unseen questions.",
        "",
        "| | Baseline | Tuned | Gain (points) |",
        "|---|---|---|---|",
        f"| In-sample | {tuning.baseline:.4f} | {tuning.in_sample:.4f} "
        f"| {gain_in:+.1f} |",
        f"| Held-out (cross-validation) | {tuning.cv_baseline:.4f} "
        f"| {tuning.cv_estimate:.4f} | {gain_cv:+.1f} |",
        f"| Consensus profile, in-sample | {tuning.baseline:.4f} "
        f"| {tuning.consensus_score:.4f} | {gain_agreed:+.1f} |",
        "",
        "Stability: value chosen by each fold. A factor is adopted in "
        "the consensus profile only when most folds agree.",
        "",
        "| Factor | Values chosen (folds) | Consensus |",
        "|---|---|---|",
    ]
    for factor, counts in tuning.stability.items():
        chosen = ", ".join(
            f"{value} ×{count}" for value, count in
            sorted(counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"| `{factor}` | {chosen} "
            f"| `{tuning.consensus_params.get(factor)}` |"
        )
    lines += [
        "",
        f"Tuned (in-sample best): `{tuning.best_params}`",
        "",
        f"Consensus: `{tuning.consensus_params}`",
        "",
    ]
    return lines


def render_markdown(report: AnalysisReport) -> str:
    """Render the analysis as a Markdown document."""
    datasets = sorted({d for r in report.rows for d in r.recall5})
    by_id = {r.trial_id: r for r in report.rows}
    out: List[str] = [
        "# Experiment report",
        "",
        f"{report.n_trials} trials. Recall@5 uses the moulinette rule "
        "(IoU ≥ 0.05). `Δ points` is the change of mean recall@5 against "
        "the baseline of the suite, with a paired bootstrap 95% interval; "
        "`P(gain)` is the share of resamples where the change wins. "
        "`s/100 q` is the warm search time per 100 questions and "
        "`points/s` the recall points gained per extra second.",
        "",
        "Verdicts: **helps/hurts** when the interval excludes 0; "
        "**likely helps/hurts** when P(gain) is beyond 0.8/0.2; "
        "otherwise **neutral** (within noise).",
        "",
        "## Recommended profiles",
        "",
        "| Profile | Mean R@5 | "
        + " | ".join(f"R@5 {d}" for d in datasets)
        + " | s/100 q | Why |",
        "|" + "---|" * (4 + len(datasets)),
    ]
    for p in report.profiles:
        recalls = " | ".join(f"{p.recall5.get(d, 0.0):.3f}" for d in datasets)
        out.append(
            f"| **{p.name}** | {p.mean_recall5:.3f} | {recalls} "
            f"| {p.search_seconds:.2f} | {p.reason} |"
        )
    out.append("")
    for p in report.profiles:
        out += [f"`{p.name}`: `{p.params}`", ""]
    out += [
        "## Pareto front: recall against search time",
        "",
        "Configurations that no other one beats on both axes, from the "
        "fastest to the most precise.",
        "",
        "| s/100 q | Mean R@5 | Suite | Configuration |",
        "|---|---|---|---|",
    ]
    for trial_id in report.pareto:
        r = by_id[trial_id]
        out.append(
            f"| {r.search_seconds:.2f} | {r.mean_recall5:.3f} | {r.suite} "
            f"| {_fmt_changed(r)} |"
        )
    if report.pareto_end_to_end:
        out += [
            "",
            "## Pareto front: recall against cold indexing + 200 questions",
            "",
            "Measured with every cache disabled (`lab_cold`). The limits "
            "of the subject add up to 390 s (300 s indexing + 90 s for "
            "200 questions).",
            "",
            "| Index s | 200 questions s | Total s | Mean R@5 "
            "| Configuration |",
            "|---|---|---|---|---|",
        ]
        rows_by_key = {
            r.trial_id: r for r in report.rows
            if r.end_to_end_seconds is not None
        }
        for trial_id in report.pareto_end_to_end:
            r = rows_by_key[trial_id]
            out.append(
                f"| {r.index_seconds:.0f} | {2 * r.search_seconds:.1f} "
                f"| {r.end_to_end_seconds:.0f} | {r.mean_recall5:.3f} "
                f"| {r.name} |"
            )
    out += ["", "## Cost/benefit per suite", ""]
    suites: Dict[str, List[TrialRow]] = {}
    for r in report.rows:
        suites.setdefault(r.suite, []).append(r)
    for name, rows in suites.items():
        cold = [r for r in rows if r.index_cold]
        shown = rows
        if name.endswith(TUNE_SUFFIX) and len(rows) > MAX_TUNE_ROWS:
            best = sorted(rows, key=lambda r: -r.objective)[:MAX_TUNE_ROWS]
            shown = [r for r in rows if not r.changed][:1] + best
            out += [f"### {name}", "", (
                f"{len(rows)} configurations explored by the tuner; the "
                f"{MAX_TUNE_ROWS} best in-sample are shown. They are not "
                "candidates for the profiles (see the tuning section)."
            ), ""] + _suite_table(shown, datasets) + [""]
            continue
        out += [f"### {name}", ""] + _suite_table(rows, datasets) + [""]
        if cold:
            out.append(
                "Cold index builds: " + "; ".join(
                    f"{_fmt_changed(r)} → {r.index_seconds:.0f} s "
                    f"({r.n_chunks} chunks)" for r in cold
                )
            )
            out.append("")
    for tuning in report.tunings:
        out += _tuning_section(tuning)
    if report.interactions:
        out += [
            "## Interactions",
            "",
            "`synergy = gain(A+B) − gain(A) − gain(B)` in recall points: "
            "about 0 means the gains add up, negative means they overlap "
            "(both fix the same questions), positive means they need each "
            "other.",
            "",
            "| Suite | A | B | Gain A | Gain B | Gain A+B | Synergy |",
            "|---|---|---|---|---|---|---|",
        ]
        for i in report.interactions:
            out.append(
                f"| {i.suite} | `{i.first}` | `{i.second}` "
                f"| {i.gain_first:+.1f} | {i.gain_second:+.1f} "
                f"| {i.gain_both:+.1f} | {i.synergy:+.1f} |"
            )
        out.append("")
    return "\n".join(out)
