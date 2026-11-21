"""Retrieval eval of the seed domains (PL-9, #146).

Usage (from the repository root, CPU, in process, no server)::

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \\
        .venv/bin/python experiments/platform/seed_eval.py \\
        [--out experiments/platform/seed_eval.json]

Each authored domain under ``knowledge/domains`` is indexed with the
knowledge embedder (``Container.knowledge_registry``) and its 20 labelled
queries (``experiments/platform/queries/<domain>.jsonl``) go through the
``KnowledgeQueryService`` three ways: no filter, ``script_phase`` filter
and ``intent`` plus ``script_phase`` filters. The filters are the labels
of the relevant chunks (an oracle: what a perfect classifier would send),
so the filtered numbers are an upper bound. The content is synthetic.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.domain.knowledge.contract import KnowledgeQuery, KnowledgeResult
from src.domain.knowledge.filters import KnowledgeFilters

DOMAINS = ("interview.generic", "support.demo")
TOP_K = 5
CONDITIONS = ("no filter", "script_phase", "intent + script_phase")

Search = Callable[[KnowledgeQuery], KnowledgeResult]


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Read the labelled queries (JSON lines) of one domain."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall_at_k(relevant: Sequence[str], returned: Sequence[str]) -> float:
    """Return the share of ``relevant`` ids found in ``returned``."""
    if not relevant:
        raise ValueError("a query needs at least one relevant id")
    return len(set(relevant) & set(returned)) / len(set(relevant))


def filters_for(item: Dict[str, Any], condition: str) -> KnowledgeFilters:
    """Return the filters of ``condition`` from a query's labels."""
    if condition == "no filter":
        return KnowledgeFilters()
    if condition == "script_phase":
        return KnowledgeFilters(script_phase=list(item["script_phase"]))
    if condition == "intent + script_phase":
        return KnowledgeFilters(intent=list(item["intent"]),
                                script_phase=list(item["script_phase"]))
    raise ValueError(f"unknown condition {condition!r}")


def evaluate(
    domain: str, queries: List[Dict[str, Any]], search: Search,
    condition: str, top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Return mean recall@k, hit@k and the missed queries of a condition."""
    recalls: List[float] = []
    hits = 0
    missed: List[str] = []
    prefix = f"{domain}:"
    for n, item in enumerate(queries):
        result = search(KnowledgeQuery.model_validate({
            "schema": "knowledge.query.v1", "request_id": f"seed-{n}",
            "domain": domain, "text": item["text"], "top_k": top_k,
            "filters": filters_for(item, condition).model_dump(),
        }))
        ids = [c.id[len(prefix):] for c in result.chunks]
        score = recall_at_k(item["relevant"], ids)
        recalls.append(score)
        if score > 0:
            hits += 1
        if score < 1:
            missed.append(str(item["id"]))
    count = len(queries)
    return {
        "condition": condition, "queries": count, "top_k": top_k,
        "recall": round(sum(recalls) / count, 4),
        "hit": round(hits / count, 4), "not_fully_found": missed,
    }


def run(
    domains: Sequence[str], queries_dir: Path, search: Search,
    chunk_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Evaluate every domain under every condition."""
    report: Dict[str, Any] = {}
    for domain in domains:
        queries = load_queries(queries_dir / f"{domain}.jsonl")
        report[domain] = {
            "chunks": (chunk_counts or {}).get(domain),
            "results": [evaluate(domain, queries, search, c)
                        for c in CONDITIONS],
        }
    return report


def format_report(report: Dict[str, Any]) -> str:
    """Return the report as a Markdown table."""
    lines = ["| domain | chunks | filters | recall@5 | hit@5 |",
             "|---|---|---|---|---|"]
    for domain, body in report.items():
        for row in body["results"]:
            lines.append(f"| {domain} | {body['chunks']} | "
                         f"{row['condition']} | {row['recall']:.3f} | "
                         f"{row['hit']:.3f} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Index the seed domains, evaluate them and save the report."""
    from src.bootstrap import Container
    from src.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/platform/"
                        "seed_eval.json")
    args = parser.parse_args(argv)
    container = Container(Settings(index_dir="no-index"))
    counts: Dict[str, int] = {}
    for domain in DOMAINS:
        counts[domain] = container.knowledge_registry.index(
            domain, force=True).report.chunks
    report = run(DOMAINS, Path("experiments/platform/queries"),
                 container.knowledge_query.run, counts)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(format_report(report))


if __name__ == "__main__":
    main()
