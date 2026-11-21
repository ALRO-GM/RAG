"""Summarise the suggestion JSONL files of the stage-1 end-to-end runs."""

import json
import sys
from collections import Counter
from typing import Any, Dict, List


def pct(values: List[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (q in 0..100)."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))
    return round(ordered[int(rank) - 1], 1)


def summarise(path: str) -> Dict[str, Any]:
    """Counts and latency percentiles of one run."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    out: Dict[str, Any] = {
        "calls": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
    }
    for scope, subset in (("all", rows), (
        "ok_only", [r for r in rows if r["status"] == "ok"]
    )):
        out[scope] = {
            key: {"p50": pct([r["stages"][key] for r in subset], 50),
                  "p90": pct([r["stages"][key] for r in subset], 90)}
            for key in ("knowledge_ms", "reasoner_ms", "total_ms")
            if subset
        }
    return out


if __name__ == "__main__":
    print(json.dumps({p: summarise(p) for p in sys.argv[1:]}, indent=1))
