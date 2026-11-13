"""Pure view models: raw lab records in, JSON-ready dictionaries out.

Nothing here touches the file system, so every rule is unit-tested
with literal records. Numbers are rounded for display only; the raw
files are published next to the summaries for anyone who wants more.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

Record = Dict[str, Any]

#: Metric used to rank trials and compute deltas against the baseline.
HEADLINE_METRIC = "recall@5"


class SiteError(ValueError):
    """A source file cannot be turned into a page."""


def parse_jsonl(text: str, source: str) -> List[Record]:
    """Parse JSON Lines, ignoring blank lines.

    Raises:
        SiteError: a line is not a JSON object (the message names the
            file and line so the broken record can be fixed).
    """
    records: List[Record] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SiteError(f"{source}:{number}: invalid JSON ({exc.msg}).")
        if not isinstance(value, dict):
            raise SiteError(f"{source}:{number}: expected a JSON object.")
        records.append(value)
    return records


def rounded(value: Any, digits: int = 4) -> Any:
    """Round floats (recursively inside dicts and lists) for display."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    return value


def mean(values: Iterable[float]) -> Optional[float]:
    """Arithmetic mean, or ``None`` for no values."""
    items = list(values)
    return sum(items) / len(items) if items else None


def headline(metrics: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Mean of the headline metric over the datasets that report it."""
    return mean(
        float(m[HEADLINE_METRIC]) for m in metrics.values()
        if isinstance(m, dict) and HEADLINE_METRIC in m
    )


_MISSING = object()


def changed_params(params: Record, base: Optional[Record]) -> Record:
    """Parameters whose value differs from the baseline's.

    Without a baseline every parameter counts as changed. A parameter
    the baseline does not set is a change too.
    """
    if base is None:
        return dict(params)
    return {k: v for k, v in params.items() if base.get(k, _MISSING) != v}


def _trial_view(
    row: Record, baseline: Optional[float], base_params: Optional[Record]
) -> Record:
    """One trial without its per-question ranks."""
    metrics = row.get("metrics") or {}
    params = row.get("params") or {}
    score = headline(metrics)
    delta = (
        None if score is None or baseline is None else score - baseline
    )
    view: Record = rounded({
        "id": row.get("trial_id", ""),
        "name": row.get("name", ""),
        "params": params,
        "changed": changed_params(params, base_params),
        "metrics": metrics,
        "headline": score,
        "delta": delta,
        "search_seconds_per_100": row.get("search_seconds_per_100"),
        "index_seconds": row.get("index_seconds"),
        "n_chunks": row.get("n_chunks"),
        "created": row.get("created", ""),
    })
    return view


def summarize_trials(rows: List[Record]) -> Record:
    """Group lab trials by suite, with deltas against each baseline.

    A trial id seen twice (a resumed run) keeps its last record. The
    baseline of a suite is the trial whose id equals ``baseline_id``;
    when it is missing, deltas are ``None`` rather than invented.
    """
    suites: Dict[str, Dict[str, Record]] = {}
    for row in rows:
        suite = str(row.get("suite", "unnamed"))
        suites.setdefault(suite, {})[str(row.get("trial_id", ""))] = row
    datasets: List[str] = []
    views: List[Record] = []
    for suite, by_id in suites.items():
        trials = list(by_id.values())
        baseline_id = next(
            (t.get("baseline_id") for t in trials if t.get("baseline_id")),
            None,
        )
        base_row = by_id.get(str(baseline_id)) if baseline_id else None
        base = headline(base_row.get("metrics") or {}) if base_row else None
        base_params = (base_row.get("params") or {}) if base_row else None
        items = [_trial_view(t, base, base_params) for t in trials]
        for item in items:
            for name in item["metrics"]:
                if name not in datasets:
                    datasets.append(name)
        ranked = [i for i in items if i["headline"] is not None]
        best = max(ranked, key=lambda i: i["headline"], default=None)
        views.append({
            "suite": suite,
            "baseline_id": baseline_id,
            "baseline": rounded(base),
            "best_id": best["id"] if best else None,
            "best_delta": best["delta"] if best else None,
            "trials": items,
        })
    return {
        "metric": HEADLINE_METRIC,
        "datasets": datasets,
        "trial_count": sum(len(v["trials"]) for v in views),
        "suites": views,
    }


def summarize_beir(rows: List[Record]) -> Record:
    """Group external benchmark rows by dataset and average per config.

    The average of a configuration only covers the datasets where it
    was run; ``datasets`` says how many, so partial runs are visible.
    """
    by_dataset: Dict[str, List[Record]] = {}
    configs: List[str] = []
    scores: Dict[str, List[float]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        config = str(row.get("config", ""))
        by_dataset.setdefault(dataset, []).append(rounded({
            "config": config,
            "params": row.get("params") or {},
            "ndcg10": row.get("ndcg10"),
            "recall10": row.get("recall10"),
            "recall100": row.get("recall100"),
            "hit": row.get("hit") or {},
            "documents": row.get("documents"),
            "queries": row.get("queries"),
            "search_seconds": row.get("search_seconds"),
        }))
        if config not in configs:
            configs.append(config)
        if isinstance(row.get("ndcg10"), (int, float)):
            scores.setdefault(config, []).append(float(row["ndcg10"]))
    averages = [
        {"config": c, "ndcg10": rounded(mean(scores[c])),
         "datasets": len(scores[c])}
        for c in configs if c in scores
    ]
    averages.sort(key=lambda a: (-a["datasets"], -a["ndcg10"]))
    return {
        "datasets": [
            {"dataset": d, "rows": sorted(
                r, key=lambda x: -(x["ndcg10"] or 0.0)
            )}
            for d, r in by_dataset.items()
        ],
        "averages": averages,
        "row_count": len(rows),
    }


def summarize_hardware(rows: List[Record]) -> Record:
    """Runtime measurements, newest run first, with the GPU speed-up.

    ``gpu_speedup`` compares the cost of the best GPU plan with the
    best CPU plan of the same run (cost = seconds for the workload).
    """
    runs: List[Record] = []
    for row in rows:
        results = list(row.get("results") or [])
        cpu = [r for r in results if _device(r) == "cpu"]
        gpu = [r for r in results if _device(r) != "cpu"]
        best_cpu = min(cpu, key=_cost, default=None)
        best_gpu = min(gpu, key=_cost, default=None)
        speedup = None
        if best_cpu and best_gpu and _cost(best_gpu) > 0:
            speedup = _cost(best_cpu) / _cost(best_gpu)
        runs.append(rounded({
            "created": row.get("created", ""),
            "signature": row.get("signature", ""),
            "gpus": row.get("gpus") or [],
            "driver": row.get("driver", ""),
            "torch": row.get("torch", ""),
            "results": sorted(results, key=_cost),
            "gpu_speedup": speedup,
        }, 3))
    runs.sort(key=lambda r: str(r["created"]), reverse=True)
    return {"runs": runs, "latest": runs[0] if runs else None}


def _device(result: Record) -> str:
    """Device of a measured plan."""
    return str((result.get("plan") or {}).get("device", "cpu"))


def _cost(result: Record) -> float:
    """Cost of a measured plan; unknown costs sort last."""
    cost = result.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else float("inf")


def summarize_rubric(data: Record, source: str) -> Record:
    """Index entry of a rubric file: what it covers, not its content."""
    competencies = data.get("competencies")
    if not isinstance(competencies, list):
        raise SiteError(f"{source}: no 'competencies' list.")
    indicators = [
        i for c in competencies for i in (c.get("indicators") or [])
    ]
    return {
        "path": source,
        "name": data.get("name", ""),
        "version": data.get("version", ""),
        "language": data.get("language", ""),
        "competencies": len(competencies),
        "indicators": len(indicators),
        "positive": sum(
            1 for i in indicators if i.get("polarity") == "positive"
        ),
        "negative": sum(
            1 for i in indicators if i.get("polarity") == "negative"
        ),
    }


def doc_title(markdown: str, fallback: str) -> str:
    """First level-1 heading of a Markdown document, else ``fallback``."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback
