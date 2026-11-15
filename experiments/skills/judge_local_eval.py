"""Measure a local judge: peak GPU memory, speed and quality (#74).

Two synthetic tasks, both from this repository:

* **dialogue acts**: ``--acts`` sentences spread evenly over the Spanish
  set ``examples/skills/dialogue_acts/dialogues.json`` (3 previous
  sentences as context), against their first intended label;
* **tag confirmation**: the candidate tags of the Spanish demo session
  ``examples/skills/transcripts/demo_es.txt`` confirmed or rejected, as
  in ``docs/skills/AUDIO.md`` §4 (how many confirmed, how many are the
  first indicator listed, how many negative).

It loads exactly the model asked for (the preference policy is
bypassed) through the analyzer's own classes
(``SkillsContainer.local_judge``), with the reply cache off: peak GPU
memory, load time, dialogue-act accuracy and the tag-confirmation
counts. It prints one JSON object (``--output`` also saves it), the
figures of ``docs/skills/DESIGN.md`` §4.2. It needs a CUDA GPU;
quantised runs need ``uv sync --group local-judge``.

Usage::

    HF_HUB_OFFLINE=1 PYTHONPATH=. python \\
        experiments/skills/judge_local_eval.py Qwen/Qwen3-8B --bits 4 \\
        --output experiments/skills/results/judge_local_qwen3-8b-4bit.json
"""

import argparse
import json
import time
from typing import Any, Dict

import torch

from src.application.skills.judges import (
    ActClassifier,
    act_accuracy,
    act_items,
    release,
)
from src.application.skills.tagging import TagConfirmer
from src.config import SkillsSettings
from src.skills_bootstrap import SkillsContainer

DIALOGUES = "examples/skills/dialogue_acts/dialogues.json"
DEMO = "examples/skills/transcripts/demo_es.txt"


def gigabytes(value: int) -> float:
    """Return bytes as GB with two decimals."""
    return round(value / 1e9, 2)


def measure(model: str, bits: int, acts: int) -> Dict[str, Any]:
    """Run both tasks with one local model and return the figures."""
    settings = SkillsSettings.model_validate({
        "judge": "local", "llm": model, "local_bits": bits,
        "local_policy": "largest_fitting",
        "use_cache": False, "progress": False, "device": "cuda",
    })
    app = SkillsContainer(settings)
    judge = app.local_judge
    out: Dict[str, Any] = {"model": model, "bits": bits}
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    start = time.perf_counter()
    out["model_name"] = judge.model_name
    judge.complete("Di hola.", "Hola", 4)
    out["load_s"] = round(time.perf_counter() - start, 1)

    with open(DIALOGUES, encoding="utf-8") as handle:
        items = act_items(json.load(handle), 3, acts)
    start = time.perf_counter()
    verdicts = ActClassifier(judge).classify(items)
    spent = time.perf_counter() - start
    predicted: Dict[str, int] = {}
    for verdict in verdicts:
        key = verdict.act or "?"
        predicted[key] = predicted.get(key, 0) + 1
    out.update({
        "acts_items": len(items),
        "acts_accuracy": act_accuracy(items, verdicts),
        "acts_unparsed": predicted.get("?", 0),
        "acts_s_per_item": round(spent / max(len(items), 1), 2),
        "acts_predicted": dict(
            sorted(predicted.items(), key=lambda kv: -kv[1])
        ),
        "peak_vram_judge_gb": gigabytes(
            torch.cuda.max_memory_allocated() - base
        ),
    })

    document = app.transcripts.read(DEMO)
    language = app.retriever.language_of(document)
    rubric = app.rubric(language)
    tagged = app.tagger.tag(document, rubric)
    start = time.perf_counter()
    confirmed = TagConfirmer(judge).confirm(tagged, rubric, language)
    spent = time.perf_counter() - start
    decisions = [t.decision for t in confirmed if t.decision is not None]
    shown = [d for d in decisions if d.status == "confirmed"]
    out.update({
        "tags_sentences": len(decisions),
        "tags_confirmed": len(shown),
        "tags_rejected": sum(
            1 for d in decisions if d.status == "rejected"
        ),
        "tags_unparsed": sum(
            1 for d in decisions if d.status == "unparsed"
        ),
        "tags_first_indicator": sum(
            1 for d in shown if (d.indicator_id or "").endswith(".P1")
        ),
        "tags_negative": sum(1 for d in shown if d.polarity == "negative"),
        "tags_s_per_item": round(spent / max(len(decisions), 1), 2),
    })
    release(judge)
    out["after_release_gb"] = gigabytes(torch.cuda.memory_allocated())
    return out


def main() -> None:
    """Parse the arguments, measure, print the JSON figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hugging Face model id")
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8, 16))
    parser.add_argument("--acts", type=int, default=200,
                        help="dialogue-act sentences (spread over the set)")
    parser.add_argument("--output", default="",
                        help="also write the figures to this JSON file")
    args = parser.parse_args()
    text = json.dumps(measure(args.model, args.bits, args.acts),
                      ensure_ascii=False, indent=1)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
