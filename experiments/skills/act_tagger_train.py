"""Fine-tune the dialogue-act encoder (approach D of #82).

English mode (default): XLM-R base on the AMI and ICSI-MRDA records
imported by ``skills dialogue_acts_import`` (``data/dialogue_acts/
{ami,mrda}.jsonl``), split by meeting with
:func:`~src.domain.skills.act_tagging.split_of` (train / dev / test),
trained with the compatible-set loss; the temperature is fitted on
``--dev`` dev sentences. Switchboard-DA is never used for training
(non-commercial licence).

Spanish folds (``--spanish-folds``): starting from the English model,
continue training on 4 of the 5 dialogue folds of our Spanish set
(``own.jsonl``) and save one model per held-out fold, so every Spanish
sentence can be scored by a model that never saw its dialogue.

Usage (workstation, CUDA build)::

    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python \\
        experiments/skills/act_tagger_train.py \\
        --output data/models/act-tagger/xlmr-base-ami-mrda
    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python \\
        experiments/skills/act_tagger_train.py --spanish-folds \\
        --init data/models/act-tagger/xlmr-base-ami-mrda \\
        --output data/models/act-tagger/xlmr-es-folds

The figures (time, peak VRAM, dev log-loss, temperature) are printed
as JSON and kept in each model's ``act_tagger.json``.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

from src.adapters.outbound.encoder_act_tagger import (
    TrainConfig,
    train_encoder,
    write_json,
)
from src.domain.skills.act_calibration import fold_of
from src.domain.skills.act_tagging import (
    ActInput,
    Reference,
    record_input,
    record_reference,
    split_of,
)
from src.domain.skills.dialogue_acts import ActRecord

Pair = Tuple[ActInput, Reference]


def records(path: str) -> Iterator[ActRecord]:
    """Yield the usable records of one imported corpus."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = ActRecord.model_validate_json(line)
            if record.categories:
                yield record


def pair(record: ActRecord) -> Pair:
    """Return a record as tagger input and reference."""
    reference = record_reference(record)
    assert reference is not None
    return record_input(record), reference


def english(data: str, corpora: List[str], dev_size: int, seed: int
            ) -> Tuple[List[Pair], List[Pair]]:
    """Return the training pairs and a sample of dev pairs."""
    train: List[Pair] = []
    dev: List[Pair] = []
    for corpus in corpora:
        for record in records(f"{data}/{corpus}.jsonl"):
            if record.non_commercial:
                raise SystemExit(f"{corpus} is non-commercial: not for "
                                 "training")
            split = split_of(record.dialogue)
            if split == "train":
                train.append(pair(record))
            elif split == "dev":
                dev.append(pair(record))
    random.Random(seed).shuffle(dev)
    return train, dev[:dev_size]


def main() -> None:
    """Parse the arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/dialogue_acts")
    parser.add_argument("--corpora", default="ami,mrda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--init", default="")
    parser.add_argument("--spanish-folds", action="store_true")
    parser.add_argument("--spanish-all", action="store_true",
                        help="train the shipped model on every Spanish "
                             "dialogue")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dev", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0,
                        help="train on this many sentences only (smoke)")
    parser.add_argument("--config", default="{}",
                        help="JSON overrides of TrainConfig")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(message)s")
    config = TrainConfig.model_validate(json.loads(args.config))
    if not (args.spanish_folds or args.spanish_all):
        train, dev = english(args.data, args.corpora.split(","), args.dev,
                             config.seed)
        if args.limit:
            random.Random(config.seed).shuffle(train)
            train = train[:args.limit]
        report = train_encoder(train, dev, args.output, config,
                               init=args.init)
        out = {"mode": "english", "output": args.output,
               **report.model_dump()}
        print(json.dumps(out, indent=1))
        write_json(f"{args.output}/train_report.json", out)
        return
    own = list(records(f"{args.data}/own.jsonl"))
    if args.spanish_all:
        # The shipped model: every Spanish dialogue, the same recipe as
        # a fold (temperature on the last fifth); no held-out figure.
        train = [pair(r) for r in own]
        calib = train[-max(len(train) // 5, 1):]
        report = train_encoder(
            train[:len(train) - len(calib)], calib, args.output,
            config.model_copy(update={"version": f"{config.version}-es"}),
            init=args.init)
        out = {"mode": "spanish-all", "init": args.init,
               **report.model_dump()}
        print(json.dumps(out, indent=1))
        write_json(f"{args.output}/train_report.json", out)
        return
    folds = fold_of([r.dialogue for r in own], args.folds)
    reports = []
    for fold in range(args.folds):
        train = [pair(r) for r, f in zip(own, folds) if f != fold]
        held = [pair(r) for r, f in zip(own, folds) if f == fold]
        # The temperature is fitted on the last fifth of the training
        # folds; the held-out fold is never seen.
        calib = train[-max(len(train) // 5, 1):]
        fit = train[:len(train) - len(calib)]
        report = train_encoder(
            fit, calib, str(Path(args.output) / f"fold-{fold}"),
            config.model_copy(update={
                "version": f"{config.version}-es-fold{fold}"}),
            init=args.init)
        reports.append({"fold": fold, "held_out": len(held),
                        **report.model_dump()})
        print(json.dumps(reports[-1], indent=1))
    write_json(f"{args.output}/train_report.json",
               {"mode": "spanish-folds", "init": args.init,
                "folds": reports})


if __name__ == "__main__":
    main()
