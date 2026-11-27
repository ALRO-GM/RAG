"""How long from an item said to its tick, the parts measured here (#286).

Two parts of "said → ticked" that do not need the running services:

* ``orchestrator``: from the first employee final of a session to the
  checklist published with the item said, through
  :class:`MandatoryTracker` with a port that answers at once (inline
  executor), 500 runs;
* ``rag``: the RAG's own check (:class:`MandatoryItems` with the
  knowledge embedding model) on the CPU, in this process: the first
  check (model load and the examples' vectors) and then the 18 interview
  lines of ``experiments/live/lv16/mandatory_lines.json`` five times.

Speech end → final is measured in ``docs/live/LIVE_MEASURES.md``; the
service on the GPU is not measured here. Run::

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. \\
        python experiments/live/lv286/mandatory_timing.py [--rag]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.mandatory import MandatoryTracker
from orchestrator.domain.events import LiveEvent

RUNS = 500
ROOT = Path(__file__).resolve().parents[3]
LINES = ROOT / "experiments" / "live" / "lv16" / "mandatory_lines.json"


class Instant:
    """A mandatory port that says the item at once."""

    def items(self, domain: str) -> Dict[str, Any]:
        """One item."""
        return {"domain": domain, "version": 1, "remind_seconds": 45,
                "items": [{"id": "aviso", "text": "Aviso de notas",
                           "phase": "opening"}]}

    def check(self, domain: str, text: str) -> Dict[str, Any]:
        """Every sentence says it."""
        return {"items": [{"id": "aviso", "score": 0.9, "said": True}]}


def first_final() -> LiveEvent:
    """The employee's first sentence of the session."""
    return LiveEvent.model_validate({
        "schema": "live.event.v1", "session_id": "s", "seq": 0,
        "type": "final", "at": "2026-11-27T10:00:00Z", "stream_time": 0.6,
        "utterance": {"id": "t0.s0", "turn": 0, "index": 0,
                      "speaker": "S1", "start": 0.0, "end": 0.6,
                      "text": "Hola, te aviso de que tomo notas.",
                      "revision": 0}})


def quantiles(values: List[float]) -> Tuple[float, float]:
    """p50 and p90 (nearest rank below)."""
    ordered = sorted(values)
    return (ordered[len(ordered) // 2],
            ordered[int(0.9 * (len(ordered) - 1))])


def orchestrator_part() -> Dict[str, float]:
    """Microseconds from the first final to the tick published."""
    times: List[float] = []
    for _ in range(RUNS):
        got: List[Tuple[float, Dict[str, Any]]] = []
        track = MandatoryTracker(
            Instant(), "d", {"S1": "interviewer"},
            lambda m: got.append((time.perf_counter(), dict(m))),
            InlineExecutor())
        track.start()
        start = time.perf_counter()
        track.on_event(first_final())
        said = [at for at, m in got if m.get("type") == "mandatory"
                and m["checklist"]["items"][0]["state"] == "said"]
        times.append((said[0] - start) * 1e6)
    p50, p90 = quantiles(times)
    return {"runs": RUNS, "p50_us": round(p50, 1), "p90_us": round(p90, 1)}


def rag_part() -> Dict[str, Any]:
    """Milliseconds of the RAG's check on the CPU, in this process."""
    import yaml

    from src.adapters.outbound.minilm_embedder import (
        SentenceTransformerEmbedder,
    )
    from src.application.knowledge.mandatory import MandatoryItems
    from src.config import KnowledgeSettings
    from src.domain.knowledge.domains import DomainManifest

    def manifest(domain_id: str) -> DomainManifest:
        """A seed domain's manifest (as the LV-16 threshold script)."""
        path = ROOT / "knowledge" / "domains" / domain_id / "manifest.yaml"
        return DomainManifest.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8")))

    settings = KnowledgeSettings()
    checker = MandatoryItems(manifest, SentenceTransformerEmbedder(
        settings.embedding_model,
        max_seq_length=settings.embedding_max_tokens))
    lines = json.loads(LINES.read_text("utf-8"))["interview.generic"]
    start = time.perf_counter()
    checker.check("interview.generic", lines[0]["text"])
    first = (time.perf_counter() - start) * 1000
    times: List[float] = []
    for _ in range(5):
        for line in lines:
            start = time.perf_counter()
            checker.check("interview.generic", line["text"])
            times.append((time.perf_counter() - start) * 1000)
    p50, p90 = quantiles(times)
    return {"lines": len(lines), "checks": len(times),
            "first_check_ms": round(first, 1), "p50_ms": round(p50, 1),
            "p90_ms": round(p90, 1)}


def main() -> None:
    """Print the parts as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag", action="store_true",
                        help="also time the RAG's check (loads the model)")
    args = parser.parse_args()
    found: Dict[str, Any] = {"orchestrator": orchestrator_part()}
    if args.rag:
        found["rag_cpu"] = rag_part()
    print(json.dumps(found))


if __name__ == "__main__":
    main()
