"""Measure the dialogue-act taggers of #82 and choose the default.

Approaches (``docs/skills/DIALOGUE_ACT_TAGGER.md``):

- ``rules`` (A): rule keys calibrated by an outcome table fitted on
  held-out dialogues (Spanish set: 5-fold cross-fitting by dialogue;
  public corpora: the corpus's own train split);
- ``llm:<model>`` (B): the verdicts of ``skills judge_acts`` saved as
  JSON (``--llm``), calibrated per answered label by 5-fold
  cross-fitting by dialogue over the sentences that were judged;
- ``encoder`` (D): the fine-tuned model, zero-shot on Spanish, with the
  temperature fitted on the English dev split; ``encoder+T`` the same
  logits with the temperature cross-fitted on the Spanish folds;
  ``encoder-es`` the per-fold models trained with 4 Spanish folds added
  (``--es-folds``), each scoring only its held-out fold;
- ``combined``: per category the approach whose claims of it are most
  precise (:func:`~src.domain.skills.act_tagging.choose_routes`), chosen
  on the other folds and applied to the held-out one.

References: the Spanish set (all 1,534 sentences, primary label and
weights), and the ``test`` split (:func:`~src.domain.skills.act_tagging.
split_of`) of AMI, MRDA, Switchboard-DA (non-commercial, evaluation
only) and DIHANA, scored on compatible sets. Writes ``scores.json``,
``margin.json``, ``speed.json`` and ``summary.md`` to ``--output`` and
the default tagger's card to ``--card``.

Usage::

    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python \\
        experiments/skills/act_tagger_eval.py \\
        --encoder data/models/act-tagger/xlmr-base-ami-mrda \\
        --es-folds data/models/act-tagger/xlmr-es-folds \\
        --llm experiments/skills/results/act_tagger/llm_qwen3-4b.json \\
        --llm experiments/skills/results/act_tagger/llm_claude_code.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.adapters.outbound.encoder_act_tagger import (
    EncoderActTagger,
    wait_for_gpu,
    write_json,
)
from src.application.skills.act_tagger import TaggerCard, llm_outcome
from src.domain.errors import ModelError
from src.domain.skills.act_calibration import (
    Array,
    as_probabilities,
    cross_fit_outcomes,
    fit_outcomes,
    fit_temperature,
    fold_of,
    margin_curve,
    propose_margin,
    reference_rates,
    softmax,
)
from src.domain.skills.act_rules import rule_key
from src.domain.skills.act_scoring import TaggerScores, score
from src.domain.skills.act_tagging import (
    ActInput,
    Probabilities,
    Reference,
    choose_routes,
    record_input,
    record_reference,
    route,
    split_of,
)
from src.domain.skills.dialogue_acts import CATEGORIES, ActRecord

FOLDS = 5
PUBLIC = ("ami", "mrda", "swda", "dihana")
MARGINS = [round(0.05 * i, 2) for i in range(0, 11)]


class Corpus:
    """The usable records of one corpus, split for evaluation."""

    def __init__(self, name: str, path: str, test_only: bool):
        """Read ``path``; keep the test split when ``test_only``."""
        self.name = name
        self.train: List[ActRecord] = []
        self.test: List[ActRecord] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = ActRecord.model_validate_json(line)
                if not record.categories:
                    continue
                if not test_only:
                    self.test.append(record)
                elif split_of(record.dialogue) == "test":
                    self.test.append(record)
                elif split_of(record.dialogue) == "train":
                    self.train.append(record)
        self.inputs = [record_input(r) for r in self.test]
        self.references = [_ref(r) for r in self.test]
        self.groups = [r.dialogue for r in self.test]


def _ref(record: ActRecord) -> Reference:
    """Return a usable record's reference."""
    reference = record_reference(record)
    assert reference is not None
    return reference


def rules_predictions(corpus: Corpus) -> Tuple[List[Probabilities],
                                               List[str]]:
    """Return calibrated rule predictions and keys for the test records."""
    keys = [rule_key(i) for i in corpus.inputs]
    if not corpus.train:
        return cross_fit_outcomes(keys, corpus.references, corpus.groups,
                                  FOLDS), keys
    table = fit_outcomes([rule_key(record_input(r)) for r in corpus.train],
                         [_ref(r) for r in corpus.train])
    return [table.probabilities(k) for k in keys], keys


def cached_logits(tagger: EncoderActTagger, corpus: Corpus,
                  cache: Path) -> Array:
    """Return the encoder's logits for a corpus, cached on disk."""
    path = cache / f"{tagger.version}.{corpus.name}.npy"
    if path.is_file():
        logits: Array = np.load(path)
        if len(logits) == len(corpus.inputs):
            return logits
    logits = tagger.logits(corpus.inputs)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, logits)
    return logits


def cross_fit_temperature(logits: Array, corpus: Corpus
                          ) -> Tuple[List[Probabilities], List[float]]:
    """Return probabilities with the temperature fitted on other folds."""
    folds = fold_of(corpus.groups, FOLDS)
    out = np.zeros_like(logits)
    temperatures = []
    for fold in range(FOLDS):
        train = [n for n, f in enumerate(folds) if f != fold]
        test = [n for n, f in enumerate(folds) if f == fold]
        t = fit_temperature(logits[train],
                            [corpus.references[n] for n in train])
        temperatures.append(round(t, 4))
        out[test] = softmax(logits[test], t)
    return as_probabilities(out), temperatures


def es_fold_predictions(directory: str, corpus: Corpus, device: str
                        ) -> List[Probabilities]:
    """Score each Spanish fold with the model that never saw it."""
    folds = fold_of(corpus.groups, FOLDS)
    result: List[Probabilities] = [{} for _ in corpus.inputs]
    for fold in range(FOLDS):
        test = [n for n, f in enumerate(folds) if f == fold]
        tagger = EncoderActTagger(f"{directory}/fold-{fold}", device)
        for n, p in zip(test, tagger.predict(
                [corpus.inputs[n] for n in test])):
            result[n] = p
        tagger.release()
    return result


def llm_predictions(path: str, corpus: Corpus
                    ) -> Tuple[str, List[int], List[Probabilities]]:
    """Return the judge, the judged indices and calibrated predictions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    verdicts = {v["id"]: v for v in data["first"]}
    index = {f"{r.dialogue}/{r.id}": n for n, r in enumerate(corpus.test)}
    judged = sorted(index[i] for i in verdicts if i in index)
    models = sorted({v["model"] for v in verdicts.values()})
    outcomes = [llm_outcome(verdicts[f"{corpus.test[n].dialogue}/"
                                     f"{corpus.test[n].id}"]["act"])
                for n in judged]
    predictions = cross_fit_outcomes(
        outcomes, [corpus.references[n] for n in judged],
        [corpus.groups[n] for n in judged], FOLDS)
    return "+".join(models), judged, predictions


def combined_predictions(corpus: Corpus,
                         outputs: Dict[str, List[Probabilities]],
                         keys: List[str]
                         ) -> Tuple[List[Probabilities], List[str]]:
    """Route each held-out fold with routes chosen on the other folds."""
    folds = fold_of(corpus.groups, FOLDS)
    result: List[Probabilities] = [{} for _ in corpus.inputs]
    sources = [""] * len(corpus.inputs)
    for fold in range(FOLDS):
        train = [n for n, f in enumerate(folds) if f != fold]
        routes, fallback = choose_routes(
            [corpus.references[n].categories for n in train],
            {name: [o[n] for n in train] for name, o in outputs.items()})
        for n, f in enumerate(folds):
            if f == fold:
                sources[n], result[n] = route(
                    routes, fallback,
                    {name: o[n] for name, o in outputs.items()},
                    {"rules": keys[n]})
    return result, sources


def compact(scores: TaggerScores) -> Dict[str, Any]:
    """Return scores rounded for the JSON file."""
    data = scores.model_dump()
    for key in ("top1", "top3", "log_loss", "ece", "macro_f1"):
        if data[key] is not None:
            data[key] = round(data[key], 4)
    for value in data["per_category"].values():
        for key in ("precision", "recall", "f1"):
            if value[key] is not None:
                value[key] = round(value[key], 4)
    return data


def speed(tagger: EncoderActTagger, items: Sequence[ActInput],
          device: str) -> Dict[str, Any]:
    """Return sentences per second of the encoder on ``device``."""
    tagger.device = device
    tagger.release()
    tagger.logits(items[:32])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tagger.logits(items)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    out: Dict[str, Any] = {
        "device": device, "sentences": len(items),
        "seconds": round(seconds, 2),
        "sentences_per_s": round(len(items) / seconds, 1)}
    if device.startswith("cuda"):
        out["peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)
    else:
        out["threads"] = torch.get_num_threads()
    tagger.release()
    return out


def table_row(name: str, s: Dict[str, Any]) -> str:
    """Return one Markdown row of the summary table."""
    f1 = s["macro_f1"]
    return (f"| {name} | {s['sentences']} | {s['top1']:.3f} | "
            f"{s['top3']:.3f} | {'—' if f1 is None else f'{f1:.3f}'} | "
            f"{s['log_loss']:.3f} | {s['ece']:.3f} |")


def summary(scores: Dict[str, Dict[str, Dict[str, Any]]],
            extra: Dict[str, Any]) -> str:
    """Return the Markdown summary of every score."""
    lines = ["# Dialogue-act tagger: measured results (#82)", "",
             "Generated by `experiments/skills/act_tagger_eval.py`; "
             "see `docs/skills/DIALOGUE_ACT_TAGGER.md`.", ""]
    for corpus, approaches in scores.items():
        lines += [f"## {corpus}", "",
                  "| Approach | Sentences | Top-1 | Top-3 | Macro-F1 | "
                  "Log-loss | ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
        lines += [table_row(n, s) for n, s in approaches.items()]
        lines += ["", "Per-category F1 (— : not measurable on this "
                  "corpus or never referenced):", "",
                  "| Approach | " + " | ".join(CATEGORIES) + " |",
                  "|---|" + "---:|" * len(CATEGORIES)]
        for n, s in approaches.items():
            cells = []
            for c in CATEGORIES:
                v = s["per_category"][c]
                cells.append("—" if v["f1"] is None or not v["support"]
                             else f"{v['f1']:.2f}")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Settings and costs", "", "```json",
              json.dumps(extra, ensure_ascii=False, indent=1), "```", ""]
    return "\n".join(lines)


def main() -> None:  # noqa: C901 - one linear experiment
    """Run every approach on every corpus and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/dialogue_acts")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--es-folds", default="")
    parser.add_argument("--llm", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corpora", default="own,ami,mrda,swda,dihana")
    parser.add_argument("--output",
                        default="experiments/skills/results/act_tagger")
    parser.add_argument("--card",
                        default="knowledge/dialogue_acts/act_tagger.json")
    parser.add_argument("--cache", default="data/models/act-tagger/cache")
    parser.add_argument("--need-gb", type=float, default=1.2,
                        help="free GPU memory to wait for")
    parser.add_argument("--wait-seconds", type=float, default=900.0,
                        help="how long to wait for it before the CPU")
    parser.add_argument("--target", type=float, default=0.8,
                        help="precision of clear labels for the margin")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        try:
            wait_for_gpu(args.need_gb, args.wait_seconds)
        except ModelError as exc:
            print(f"{exc} Scoring on the CPU instead.", flush=True)
            args.device = "cpu"
    encoder = EncoderActTagger(args.encoder, args.device)
    scores: Dict[str, Dict[str, Dict[str, Any]]] = {}
    extra: Dict[str, Any] = {"encoder": encoder.card.model_dump(),
                             "folds": FOLDS}
    default: Optional[Tuple[str, List[Probabilities], Corpus]] = None
    card_parts: Dict[str, Any] = {}
    for name in args.corpora.split(","):
        corpus = Corpus(name, f"{args.data}/{name}.jsonl",
                        test_only=name in PUBLIC)
        refs = corpus.references
        result: Dict[str, Dict[str, Any]] = {}
        rules, keys = rules_predictions(corpus)
        result["rules"] = compact(score(refs, rules))
        logits = cached_logits(encoder, corpus, Path(args.cache))
        coded = as_probabilities(softmax(logits, encoder.card.temperature))
        result["encoder"] = compact(score(refs, coded))
        if name == "own":
            spanish_t, temps = cross_fit_temperature(logits, corpus)
            result["encoder+T"] = compact(score(refs, spanish_t))
            extra["spanish_fold_temperatures"] = temps
            outputs: Dict[str, List[Probabilities]] = {
                "rules": rules, "encoder": spanish_t}
            combined, sources = combined_predictions(corpus, outputs, keys)
            result["combined"] = compact(score(refs, combined))
            extra["combined_sources"] = {
                s: sources.count(s) for s in sorted(set(sources))}
            pool = {"rules": rules, "encoder+T": spanish_t,
                    "combined": combined}
            if args.es_folds:
                es = es_fold_predictions(args.es_folds, corpus, args.device)
                result["encoder-es"] = compact(score(refs, es))
                both, _ = combined_predictions(
                    corpus, {"rules": rules, "encoder": es}, keys)
                result["combined-es"] = compact(score(refs, both))
                pool.update({"encoder-es": es, "combined-es": both})
            for path in args.llm:
                model, judged, predictions = llm_predictions(path, corpus)
                sub = [refs[n] for n in judged]
                result[f"llm:{model}"] = compact(score(sub, predictions))
                for other, values in (("rules", rules),
                                      ("encoder+T", spanish_t),
                                      ("combined", combined)):
                    result[f"{other} (same {len(judged)})"] = compact(
                        score(sub, [values[n] for n in judged]))
            # Only what can be shipped from these weights competes for
            # the default: the per-fold Spanish models are a measurement.
            best = max(("rules", "encoder+T", "combined"),
                       key=lambda n: (result[n]["top1"],
                                      -result[n]["log_loss"]))
            default = (best, pool[best], corpus)
            routes, fallback = choose_routes(
                [r.categories for r in refs], outputs)
            card_parts = {
                "rules": fit_outcomes(keys, refs),
                "temperature": fit_temperature(logits, refs),
                "routes": routes, "fallback": fallback,
                "base_rates": reference_rates(refs)}
        scores[name] = result
        print(name, {k: (v["top1"], v["macro_f1"]) for k, v in
                     result.items()}, flush=True)
    out = Path(args.output)
    write_json(str(out / "scores.json"), scores)
    if default is not None:
        chosen, predictions, corpus = default
        rates = reference_rates(corpus.references)
        curve = margin_curve(corpus.references, predictions, rates, MARGINS)
        margin = propose_margin(curve, args.target)
        write_json(str(out / "margin.json"), {
            "approach": chosen, "target_precision": args.target,
            "proposed_margin": margin,
            "curve": [p.model_dump() for p in curve]})
        extra.update({"default": chosen, "proposed_margin": margin})
        card = TaggerCard(
            name=chosen,
            rules=card_parts["rules"], encoder=args.encoder,
            temperature=card_parts["temperature"],
            routes=card_parts["routes"] if chosen == "combined" else [],
            fallback=(card_parts["fallback"] if chosen == "combined"
                      else chosen.split("+")[0]),
            base_rates=card_parts["base_rates"],
            margin=margin if margin is not None else MARGINS[-1],
            source="experiments/skills/act_tagger_eval.py on the Spanish "
                   "set; figures in experiments/skills/results/act_tagger")
        write_json(args.card, card.model_dump())
    own = Corpus("own", f"{args.data}/own.jsonl", test_only=False)
    costs = [speed(encoder, own.inputs, "cpu")]
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        costs.append(speed(encoder, own.inputs, args.device))
    start = time.perf_counter()
    for item in own.inputs:
        rule_key(item)
    costs.append({"device": "cpu", "approach": "rules",
                  "sentences": len(own.inputs),
                  "sentences_per_s": round(
                      len(own.inputs) / (time.perf_counter() - start), 1)})
    extra["speed"] = costs
    write_json(str(out / "speed.json"), costs)
    (out / "summary.md").write_text(summary(scores, extra),
                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in extra.items() if k != "encoder"},
                     indent=1))


if __name__ == "__main__":
    main()
