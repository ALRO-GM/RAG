"""Summary of a replay: how many calls, and p50/p90 per stage."""

from typing import Any, Dict, List, Sequence

from orchestrator.domain.records import SuggestionRecord


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q`` quantile (0-1) of ``values`` (linear interpolation)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(
    records: List[SuggestionRecord], skips: Dict[str, int]
) -> Dict[str, Any]:
    """Return counts by status and p50/p90 of each measured stage."""
    by_status: Dict[str, int] = {}
    for record in records:
        by_status[record.status] = by_status.get(record.status, 0) + 1
    answered = [r for r in records if r.status in ("ok", "no_suggestion")]
    stages: Dict[str, Dict[str, float]] = {}
    for name in ("knowledge_ms", "reasoner_ms", "total_ms"):
        values = [
            v for r in answered
            if (v := getattr(r.stages, name)) is not None
        ]
        if values:
            stages[name] = {
                "p50": round(percentile(values, 0.5), 1),
                "p90": round(percentile(values, 0.9), 1),
            }
    return {
        "calls": len(records), "by_status": by_status,
        "skipped": dict(skips), "stages": stages,
    }
