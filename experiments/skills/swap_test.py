"""Swap test of the dialogue-act tagger and judges (#86, FAIRNESS.md J1).

The synthetic sentences of ``examples/skills/fairness/swap_sentences.json``
come in four variants per base (``f``: a woman's name and feminine
agreement; ``m``: a man's; ``origin``: a name of another origin with
feminine agreement; ``colloquial``: the ``f`` sentence in a colloquial
register). A model that labels by what is done, not by who does it,
gives every variant of a base the same label. This script labels every
sentence with one approach and counts, per variant, the bases whose
label differs from the ``f`` variant's (Wilson interval)::

    PYTHONPATH=. python experiments/skills/swap_test.py rules out.json
    PYTHONPATH=. python experiments/skills/swap_test.py encoder out.json
    PYTHONPATH=. python experiments/skills/swap_test.py local out.json \\
        --model Qwen/Qwen3-8B --bits 4
    PYTHONPATH=. python experiments/skills/swap_test.py claude out.json \\
        --max-calls 60

``rules`` and ``encoder`` are the tagger of step d (``act_tagger``);
``local`` and ``claude`` are the judges asked for one category per
sentence (``ActClassifier``, the prompt of ``act_prompting``), the
reply cache off. ``claude`` is Claude Code (``claude -p``, the
subscription), never the paid API; ``--max-calls`` stops it.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.application.skills.judges import ActClassifier, ActItem, CallBudget
from src.config import SkillsSettings
from src.domain.skills.act_tagging import ActInput
from src.domain.skills.dialogue_acts import ContextLine
from src.domain.skills.fairness import swap_changes
from src.skills_bootstrap import SkillsContainer

SENTENCES = "examples/skills/fairness/swap_sentences.json"
REFERENCE = "f"


def tagger_labels(
    app: SkillsContainer, kind: str, items: List[Dict[str, Any]]
) -> Tuple[str, List[Optional[str]], List[Dict[str, float]]]:
    """Return the tagger's version, top labels and distributions."""
    tagger, note = app.act_tagger(kind)
    if note:
        raise SystemExit(note)
    inputs = [
        ActInput(id=i["id"], speaker=i["speaker"], text=i["text"],
                 context=[ContextLine(speaker=s, text=t)
                          for s, t in i["context"]])
        for i in items
    ]
    probabilities = [dict(p) for p in tagger.predict(inputs)]
    labels: List[Optional[str]] = [
        max(p, key=lambda k: p[k]) if p else None for p in probabilities
    ]
    return f"{tagger.name}:{tagger.version}", labels, probabilities


def judge_labels(
    app: SkillsContainer, kind: str, items: List[Dict[str, Any]],
    max_calls: int,
) -> Tuple[str, List[Optional[str]], int]:
    """Return the judge's model, its labels and the calls it took."""
    judge = app.claude_judge if kind == "claude" else app.local_judge
    budget = CallBudget(judge, max_calls)
    acts = [
        ActItem(id=i["id"], speaker=i["speaker"], text=i["text"],
                context=[(s, t) for s, t in i["context"]],
                reference=[i["intended"]])
        for i in items
    ]
    verdicts = ActClassifier(budget).classify(acts)
    return judge.model_name, [v.act for v in verdicts], budget.calls


def total_variation(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Return half the L1 distance between two distributions."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def summarise(
    items: List[Dict[str, Any]], labels: List[Optional[str]],
    probabilities: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Return the label changes per variant and the hit rate per variant."""
    keyed = {(i["base"], i["variant"]): label
             for i, label in zip(items, labels)}
    changes = [c.model_dump() for c in swap_changes(keyed, REFERENCE)]
    hits: Dict[str, List[int]] = {}
    for item, label in zip(items, labels):
        cell = hits.setdefault(item["variant"], [0, 0])
        cell[0] += int(label == item["intended"])
        cell[1] += 1
    out: Dict[str, Any] = {
        "changes": changes,
        "changed_total": sum(c["changed"] for c in changes),
        "pairs_total": sum(c["pairs"] for c in changes),
        "intended_hits": {v: {"hits": h, "n": n}
                          for v, (h, n) in sorted(hits.items())},
        "unparsed": sum(1 for label in labels if label is None),
    }
    if probabilities is not None:
        by_key = {(i["base"], i["variant"]): p
                  for i, p in zip(items, probabilities)}
        shift: Dict[str, List[float]] = {}
        for (base, variant), p in by_key.items():
            if variant != REFERENCE and (base, REFERENCE) in by_key:
                shift.setdefault(variant, []).append(
                    total_variation(p, by_key[(base, REFERENCE)]))
        out["mean_total_variation"] = {
            v: round(sum(s) / len(s), 4) for v, s in sorted(shift.items())
        }
        out["max_total_variation"] = {
            v: round(max(s), 4) for v, s in sorted(shift.items())
        }
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """Label the swap sentences with one approach and write the counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("approach",
                        choices=("rules", "encoder", "local", "claude"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--sentences", default=SENTENCES)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8, 16))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args(argv)
    with open(args.sentences, encoding="utf-8") as handle:
        items: List[Dict[str, Any]] = json.load(handle)["items"]
    settings: Dict[str, Any] = {
        "use_cache": False, "progress": False, "device": args.device,
    }
    if args.approach == "local":
        settings.update({"judge": "local", "llm": args.model,
                         "local_bits": args.bits,
                         "local_policy": "largest_fitting"})
    app = SkillsContainer(SkillsSettings.model_validate(settings))
    result: Dict[str, Any] = {"approach": args.approach,
                              "sentences": args.sentences}
    probabilities: Optional[List[Dict[str, float]]] = None
    if args.approach in ("rules", "encoder"):
        model, labels, probabilities = tagger_labels(
            app, args.approach, items)
    else:
        model, labels, calls = judge_labels(
            app, args.approach, items, args.max_calls)
        result["calls"] = calls
    result["model"] = model
    result.update(summarise(items, labels, probabilities))
    result["labels"] = {i["id"]: label for i, label in zip(items, labels)}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "model", "changed_total", "pairs_total", "unparsed")}))
    for change in result["changes"]:
        print(change["variant"], change["changed"], "/", change["pairs"],
              change["bases"])


if __name__ == "__main__":
    main()
