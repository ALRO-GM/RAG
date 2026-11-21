"""Measure the similarity threshold of the mandatory items (LV-16).

Every sentence of ``mandatory_lines.json`` (synthetic employee lines,
labelled before the run with the item they say or ``null``) is checked
against its domain's mandatory items with the RAG's own use case
(:class:`src.application.knowledge.mandatory.MandatoryItems`) and the
knowledge embedding model on the CPU. For a range of thresholds it
counts, over all sentences, true hits (the labelled item reaches the
threshold and is the sentence's closest item, the rule of
:func:`src.domain.knowledge.mandatory.item_scores`), misses and false
hits (any other item said)::

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. \\
        python experiments/live/lv16/mandatory_threshold.py \\
        --out experiments/live/lv16/mandatory_threshold.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.adapters.outbound.minilm_embedder import SentenceTransformerEmbedder
from src.application.knowledge.mandatory import MandatoryItems
from src.config import KnowledgeSettings
from src.domain.knowledge.domains import DomainManifest

ROOT = Path(__file__).resolve().parents[3]
LINES = Path(__file__).with_name("mandatory_lines.json")
THRESHOLDS = [round(0.40 + 0.025 * i, 3) for i in range(17)]


def manifest(domain_id: str) -> DomainManifest:
    """Read a seed domain's manifest."""
    path = ROOT / "knowledge" / "domains" / domain_id / "manifest.yaml"
    return DomainManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def main() -> None:
    """Score every line and print the table of thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    settings = KnowledgeSettings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        max_seq_length=settings.embedding_max_tokens,
    )
    checker = MandatoryItems(manifest, embedder)
    data = json.loads(LINES.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for domain, lines in data.items():
        if domain == "note":
            continue
        for line in lines:
            found = checker.check(domain, line["text"])
            scores = {i["id"]: i["score"] for i in found["items"]}
            rows.append({"domain": domain, "item": line["item"],
                         "text": line["text"], "scores": scores})
    table = []
    for threshold in THRESHOLDS:
        hits = misses = false = 0
        for row in rows:
            top = max(row["scores"].values())
            for item, score in row["scores"].items():
                said = score >= threshold and score == top
                if item == row["item"]:
                    if said:
                        hits += 1
                    else:
                        misses += 1
                elif said:
                    false += 1
        positives = hits + misses
        table.append({
            "threshold": threshold, "hits": hits, "misses": misses,
            "false_hits": false,
            "recall": round(hits / positives, 3) if positives else None,
            "precision": round(hits / (hits + false), 3)
            if hits + false else None,
        })
    positive = [r["scores"][r["item"]] for r in rows if r["item"]]
    negative = [s for r in rows for i, s in r["scores"].items()
                if i != r["item"]]
    report = {
        "model": settings.embedding_model, "sentences": len(rows),
        "positive_sentences": len(positive),
        "labelled_item_score": {"min": min(positive),
                                "max": max(positive)},
        "other_item_score_max": max(negative),
        "table": table, "rows": rows,
    }
    for line in table:
        print(json.dumps(line))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("table", "rows")}))
    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False,
                                       indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
