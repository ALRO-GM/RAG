"""Measure judges on confirming rubric indicators (#85).

The reference is ``examples/skills/judge_indicators/reference.json``:
120 synthetic Spanish sentences, half showing one indicator (almost a third of
those negative), half only resembling one. Every judge answers every
item twice, with the indicators listed forwards and reversed, through
the prompts of ``skills tag --confirm``. Figures: precision and recall
of the confirmations, how often it confirms what is not there, position
bias, and (``combine``) agreement between judges.

A judge is ``opencode:<provider/model>`` (the installed ``opencode run``,
free models, synthetic text only) or a Hugging Face id such as
``Qwen/Qwen3-4B`` (4-bit, needs a CUDA GPU and ``--group local-judge``).

Usage::

    R=experiments/skills/results
    E=experiments/skills/judge_indicators_eval.py
    HF_HUB_OFFLINE=1 PYTHONPATH=. python $E run Qwen/Qwen3-4B \\
        --output $R/judge_indicators_qwen3-4b.json
    PYTHONPATH=. python $E combine $R/judge_indicators_qwen3-*.json \\
        --output $R/judge_indicators_summary.json
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.application.skills.indicator_probe import IndicatorProbe
from src.application.skills.ports import ChatModel
from src.domain.errors import QuotaExhaustedError
from src.domain.skills.judge_reference import (
    Order,
    Outcome,
    ReferenceItem,
    ReferenceSet,
    agreement,
    position_bias,
    score,
)
from src.domain.skills.rubric import RubricCatalogue

REFERENCE = "examples/skills/judge_indicators/reference.json"
RUBRIC = "knowledge/rubrics/core.es.json"
OPENCODE = "/home/alberto/.opencode/bin/opencode"
ORDERS: Tuple[Order, ...] = ("forward", "reversed")


class OpencodeRun:
    """A free opencode model through ``opencode run --standalone``.

    The adapter ``adapters/outbound/opencode_judge.py`` follows the same
    2.x command line since #132 but runs in a sandboxed HOME (anonymous
    free tier only, every tool denied). This experiment predates it and
    calls opencode directly with the owner's own login (and so the
    owner's config) and the system and user text as one message. Only
    the synthetic reference is ever sent.
    """

    def __init__(self, model: str, retries: int = 3, timeout: int = 150,
                 replies: Optional[Dict[str, str]] = None):
        """Choose the model (``provider/model``) and the retry budget.

        ``replies`` are answers kept from an earlier run, by the hash of
        the message; they are served again instead of asking (the free
        tier allows only a few dozen calls a day).
        """
        self.model = model
        self.retries = retries
        self.timeout = timeout
        self.replies: Dict[str, str] = dict(replies or {})

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return f"opencode:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's text reply, retrying transient failures."""
        message = f"{system}\n\n{user}"
        key = hashlib.sha1(message.encode("utf-8")).hexdigest()
        if key in self.replies:
            return self.replies[key]
        error = ""
        for attempt in range(self.retries):
            with tempfile.TemporaryDirectory() as cwd:
                done = subprocess.run(
                    [OPENCODE, "run", "--standalone", "--format", "json",
                     "-m", self.model, message],
                    capture_output=True, text=True, cwd=cwd,
                    timeout=self.timeout, check=False,
                )
            text, error = read_events(done.stdout)
            if text and not error:
                self.replies[key] = text
                return text
            if "usage limit" in error.lower():
                raise QuotaExhaustedError(error[:200])
            time.sleep(2 ** attempt)
        raise RuntimeError(f"opencode gave no answer: {error[:200]}")


def read_events(stdout: str) -> Tuple[str, str]:
    """Return ``(text, error)`` from ``opencode run --format json``."""
    parts: List[str] = []
    error = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            detail = event.get("error") or {}
            error = str(detail.get("message") or detail)
        part = event.get("part")
        if event.get("type") == "text" and isinstance(part, dict):
            parts.append(str(part.get("text") or ""))
    return "".join(parts), error


def local_judge(model: str, bits: int) -> ChatModel:
    """Return exactly this Hugging Face model as the analyzer's judge."""
    from src.config import SkillsSettings
    from src.skills_bootstrap import SkillsContainer

    settings = SkillsSettings.model_validate({
        "judge": "local", "llm": model, "local_bits": bits,
        "local_policy": "largest_fitting", "use_cache": False,
        "progress": False, "device": "cuda", "local_on_cpu": "fail",
    })
    return SkillsContainer(settings).local_judge


def run(judge: ChatModel, threads: int) -> Dict[str, Any]:
    """Ask the judge every item in both orders and score it.

    A judge that runs out of quota (the free opencode tier) stops the
    run: what was answered is scored, ``complete`` says it is partial.
    """
    with open(REFERENCE, encoding="utf-8") as handle:
        reference = ReferenceSet.model_validate(json.load(handle))
    with open(RUBRIC, encoding="utf-8") as handle:
        rubric = RubricCatalogue.model_validate(json.load(handle))
    negatives = {
        i.id for c in rubric.competencies for i in c.indicators
        if i.polarity == "negative"
    }
    probe = IndicatorProbe(judge, rubric)
    outcomes: Dict[str, List[Outcome]] = {o: [] for o in ORDERS}
    seconds: Dict[str, float] = {}
    complete = True
    gone = threading.Event()

    def ask(item: ReferenceItem, order: Order) -> Optional[Outcome]:
        """Answer one item, or ``None`` once the quota is gone."""
        if gone.is_set():
            return None
        try:
            return probe.ask(item, order)
        except QuotaExhaustedError:
            gone.set()
            return None

    for order in ORDERS:
        start = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            answers = list(pool.map(
                lambda item, o=order: ask(item, o), reference.items
            ))
        outcomes[order] = [a for a in answers if a is not None]
        complete = complete and len(outcomes[order]) == len(answers)
        seconds[order] = round(time.perf_counter() - start, 1)
    result: Dict[str, Any] = {
        "judge": judge.model_name,
        "complete": complete,
        "calls": sum(len(v) for v in outcomes.values()),
        "seconds": seconds,
        "forward": score(reference.items, outcomes["forward"], negatives),
        "reversed": score(reference.items, outcomes["reversed"], negatives),
        "position": position_bias(outcomes["forward"], outcomes["reversed"]),
        "outcomes": {
            order: [o.model_dump() for o in outcomes[order]]
            for order in ORDERS
        },
    }
    replies = getattr(judge, "replies", None)
    if replies:
        result["replies"] = replies
    return result


def combine(paths: Sequence[str]) -> Dict[str, Any]:
    """Return the agreement of every pair of saved runs."""
    runs: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))

    def flat(result: Dict[str, Any]) -> List[Outcome]:
        return [
            Outcome.model_validate(o)
            for order in ORDERS for o in result["outcomes"][order]
        ]

    pairs = {}
    for i, first in enumerate(runs):
        for second in runs[i + 1:]:
            pairs[f"{first['judge']} / {second['judge']}"] = agreement(
                flat(first), flat(second)
            )
    return {
        "judges": [r["judge"] for r in runs],
        "agreement": pairs,
    }


def main() -> None:
    """Parse the arguments and run one judge or combine saved runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="measure one judge")
    one.add_argument("judge", help="opencode:<provider/model> or HF id")
    one.add_argument("--bits", type=int, default=4, choices=(4, 8, 16))
    one.add_argument("--threads", type=int, default=1)
    one.add_argument("--replay", default="",
                     help="saved run whose opencode replies are reused")
    both = sub.add_parser("combine", help="agreement of saved runs")
    both.add_argument("files", nargs="+")
    for each in (one, both):
        each.add_argument("--output", default="")
    args = parser.parse_args()
    result: Dict[str, Any]
    if args.command == "run":
        kept: Dict[str, str] = {}
        if args.replay:
            with open(args.replay, encoding="utf-8") as handle:
                kept = json.load(handle).get("replies", {})
        judge: ChatModel = (
            OpencodeRun(args.judge.split(":", 1)[1], replies=kept)
            if args.judge.startswith("opencode:")
            else local_judge(args.judge, args.bits)
        )
        result = run(judge, args.threads)
    else:
        result = combine(args.files)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text if args.command == "combine" else json.dumps(
        {k: v for k, v in result.items() if k not in ("outcomes", "replies")},
        ensure_ascii=False, indent=1))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
