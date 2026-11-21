"""The seed domains of #146 and the script that evaluates them.

The committed domains are checked against their manifests and labelled
queries; the evaluation itself runs on a fake search function.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

from src.adapters.outbound.fs_domain_catalogue import FileSystemDomainCatalogue
from src.domain.knowledge.contract import (
    KnowledgeQuery,
    KnowledgeResult,
    ResultChunk,
    ResultMetadata,
    ResultTimings,
)
from src.domain.models import MinimalSource

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "platform" / "seed_eval.py"
QUERIES = ROOT / "experiments" / "platform" / "queries"
DOMAINS = ("interview.generic", "support.demo")


def load_script() -> ModuleType:
    """Import the eval script, which is not in a package."""
    spec = importlib.util.spec_from_file_location("seed_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed_eval = load_script()


def records(domain: str) -> List[Dict[str, Any]]:
    """Return the JSONL records of a seed domain."""
    path = ROOT / "knowledge" / "domains" / domain / "fragmentos.jsonl"
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_is_valid_and_large_enough(domain: str) -> None:
    """Manifest parses, every record is in the vocabularies, >= 60 chunks."""
    manifest = FileSystemDomainCatalogue(
        str(ROOT / "knowledge" / "domains"), "no-data").manifest(domain)
    rows = records(domain)
    assert len(rows) >= 60
    assert len({r["id"] for r in rows}) == len(rows)
    voc = manifest.vocabularies
    for row in rows:
        assert row["intent"] in voc.intent
        assert row["script_phase"] in voc.script_phase
        assert set(row["tags"]) <= set(voc.tags)


@pytest.mark.parametrize("domain", DOMAINS)
def test_queries_point_at_existing_chunks(domain: str) -> None:
    """20 queries per domain, labels copied from their relevant chunks."""
    by_id = {r["id"]: r for r in records(domain)}
    queries = seed_eval.load_queries(QUERIES / f"{domain}.jsonl")
    assert len(queries) == 20
    for q in queries:
        assert q["relevant"] and set(q["relevant"]) <= set(by_id)
        assert set(q["intent"]) == {by_id[i]["intent"] for i in q["relevant"]}


def test_support_has_the_customer_signal_vocabulary() -> None:
    """ARCHITECTURE §8.1: the six signals are intents of support.demo."""
    manifest = FileSystemDomainCatalogue(
        str(ROOT / "knowledge" / "domains"), "no-data").manifest(
            "support.demo")
    signals = {"complaint", "stated_frustration", "urgency", "churn_risk",
               "confusion", "satisfaction"}
    assert signals <= set(manifest.vocabularies.intent)
    assert signals <= {r["intent"] for r in records("support.demo")}


def test_recall_at_k() -> None:
    """Recall is the share of relevant ids returned."""
    assert seed_eval.recall_at_k(["a", "b"], ["b", "x"]) == 0.5
    with pytest.raises(ValueError):
        seed_eval.recall_at_k([], ["a"])


def test_filters_for() -> None:
    """Each condition builds its filters; an unknown one is refused."""
    item = {"intent": ["x"], "script_phase": ["p"]}
    assert seed_eval.filters_for(item, "no filter").script_phase == []
    assert seed_eval.filters_for(item, "script_phase").intent == []
    both = seed_eval.filters_for(item, "intent + script_phase")
    assert both.intent == ["x"] and both.script_phase == ["p"]
    with pytest.raises(ValueError):
        seed_eval.filters_for(item, "other")


def fake_result(query: KnowledgeQuery, ids: List[str]) -> KnowledgeResult:
    """Build a result whose chunks are ``ids`` of the query's domain."""
    chunks = [ResultChunk(
        id=f"{query.domain}:{i}", text="t", score=1.0,
        source=MinimalSource(file_path="f", first_character_index=0,
                             last_character_index=1),
        metadata=ResultMetadata()) for i in ids]
    return KnowledgeResult(
        request_id=query.request_id, domain=query.domain,
        index_version="v", chunks=chunks,
        timings=ResultTimings(total_ms=0.0))


def test_run_and_report() -> None:
    """Filters change what the fake search returns and the numbers."""
    def search(query: KnowledgeQuery) -> KnowledgeResult:
        """Return the right chunk only when the intent filter is given."""
        return fake_result(query, ["a"] if query.filters.intent else ["z"])

    queries = [{"id": "q1", "text": "hola", "relevant": ["a"],
                "intent": ["x"], "script_phase": ["p"]}]
    out = seed_eval.evaluate("d", queries, search, "no filter")
    assert out["recall"] == 0.0 and out["not_fully_found"] == ["q1"]
    out = seed_eval.evaluate("d", queries, search, "intent + script_phase")
    assert out["recall"] == 1.0 and out["hit"] == 1.0
    folder = Path(__file__).parent / "_seed_tmp"
    folder.mkdir(exist_ok=True)
    try:
        (folder / "d.jsonl").write_text(json.dumps(queries[0]) + "\n",
                                        encoding="utf-8")
        report = seed_eval.run(["d"], folder, search, {"d": 1})
    finally:
        (folder / "d.jsonl").unlink()
        folder.rmdir()
    assert len(report["d"]["results"]) == 3
    assert "| d | 1 | no filter | 0.000 | 0.000 |" \
        in seed_eval.format_report(report)
