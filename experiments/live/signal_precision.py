"""Precision of each customer signal on labelled synthetic snippets (#183).

``experiments/live/signal_snippets.json`` holds 60 two-line support
snippets (agent, customer) with one gold label each, written before any
run. Each snippet is sent once (no retry: one Haiku call per snippet) to
the running RAG's ``POST /reason`` with the product's settings
(``suggester.support@1``, one suggestion, retrieval approximated by word
overlap with the support domain's chunks) and the signals the customer's
sentence *expresses* are read as the live loop reads them
(:func:`orchestrator.domain.signals.expressed`)::

    PYTHONPATH=. python experiments/live/signal_precision.py \\
        --token-file ~/.config/ragputin/rag-token \\
        --out experiments/live/signal_results.json

A rerun reuses the answers already in ``--out`` (``--fresh`` asks again).
Per signal: precision (of the snippets where it was shown, the share
whose gold label is it), recall, and a confusion matrix of gold label
against the signals shown (a snippet with two signals counts in both
columns; none shown counts in ``none``).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

from orchestrator.domain.signals import SIGNALS, expressed

ROOT = Path(__file__).resolve().parents[2]
SNIPPETS = Path(__file__).with_name("signal_snippets.json")
CHUNKS = ROOT / "knowledge/domains/support.demo/fragmentos.jsonl"
LABELS = list(SIGNALS) + ["none"]


def words(text: str) -> Set[str]:
    """Lower-case words of at least 4 letters."""
    return set(re.findall(r"\w{4,}", text.lower()))


def build_request(snippet: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``reasoner.request.v1`` of one snippet."""
    chunks = [json.loads(x) for x in CHUNKS.read_text().splitlines() if x]
    said = words(snippet["customer"])
    chunks.sort(key=lambda c: -len(said & words(c["text"])))
    return {
        "schema": "reasoner.request.v1", "request_id": snippet["id"],
        "role": "suggester", "model": "haiku", "language": "es",
        "template": "suggester.support@1",
        "context": {
            "session": {"id": "signals", "type": "otro",
                        "goal": "atención al cliente"},
            "speaker_roles": {"agente": "agent", "cliente": "customer"},
            "window": [
                {"id": "t0.s0", "speaker": "agente",
                 "text": snippet["agent"]},
                {"id": "t1.s0", "speaker": "cliente",
                 "text": snippet["customer"]},
            ],
            "focus": "t1.s0",
        },
        "retrieved": [{"id": "support.demo:" + c["id"], "text": c["text"]}
                      for c in chunks[:4]],
        "constraints": {"max_suggestions": 1, "deadline_ms": 2800},
    }


def shown(response: Dict[str, Any], snippet: Dict[str, Any]) -> List[str]:
    """Return the signals the live loop would show for ``response``."""
    texts = {"t0.s0": ("agente", snippet["agent"]),
             "t1.s0": ("cliente", snippet["customer"])}
    roles = {"agente": "agent", "cliente": "customer"}
    found: List[str] = []
    for suggestion in response.get("suggestions") or []:
        for item in expressed(suggestion, texts, roles):
            if item["signal"] not in found:
                found.append(item["signal"])
    return found


def score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return per-signal precision, recall and the confusion matrix."""
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for row in rows:
        for predicted in row["shown"] or ["none"]:
            matrix[row["gold"]][predicted] += 1
    per: Dict[str, Any] = {}
    for label in SIGNALS:
        true = matrix[label][label]
        said = sum(matrix[g][label] for g in LABELS)
        gold_n = sum(1 for r in rows if r["gold"] == label)
        per[label] = {
            "shown": said, "correct": true, "gold": gold_n,
            "precision": round(true / said, 3) if said else None,
            "recall": round(
                sum(1 for r in rows
                    if r["gold"] == label and label in r["shown"]) / gold_n,
                3) if gold_n else None,
        }
    none_rows = [r for r in rows if r["gold"] == "none"]
    return {
        "per_signal": per, "confusion": matrix,
        "none_clean": sum(1 for r in none_rows if not r["shown"]),
        "none_total": len(none_rows),
    }


def main() -> None:
    """Ask, score and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    done: Dict[str, Dict[str, Any]] = {}
    if args.out.exists() and not args.fresh:
        done = {r["id"]: r for r in
                json.loads(args.out.read_text("utf-8"))["rows"]}
    snippets = json.loads(SNIPPETS.read_text("utf-8"))["snippets"]
    calls = 0
    rows: List[Dict[str, Any]] = []
    for snippet in snippets[:args.limit]:
        row: Optional[Dict[str, Any]] = done.get(snippet["id"])
        if row is None:
            calls += 1
            reply = httpx.post(
                args.url + "/reason", json=build_request(snippet),
                headers={"Authorization": f"Bearer {token}"}, timeout=30,
            ).json()
            row = {
                "id": snippet["id"], "gold": snippet["label"],
                "customer": snippet["customer"],
                "status": reply.get("status"), "shown": shown(reply, snippet),
                "raw_signals": [s.get("signals")
                                for s in reply.get("suggestions") or []],
            }
        rows.append(row)
    result = {"haiku_calls_this_run": calls, "snippets": len(rows),
              "statuses": {s: sum(1 for r in rows if r["status"] == s)
                           for s in sorted({str(r["status"]) for r in rows})},
              **score(rows), "rows": rows}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("haiku_calls_this_run", "statuses", "per_signal",
                       "none_clean", "none_total")}, indent=1))


if __name__ == "__main__":
    main()
