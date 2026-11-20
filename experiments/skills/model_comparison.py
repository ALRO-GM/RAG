"""Run the judge phases with one Claude model and compare two models.

Three phases, all on synthetic data only:

* ``acts``: the 140 dialogue-act sentences of
  ``docs/skills/DIALOGUE_ACT_TAGGER.md`` §2 (``skills judge_acts --sample
  140``), scored against their intended acts;
* ``indicators``: the 120 reference items of
  ``examples/skills/judge_indicators/reference.json`` (branch
  ``exp/85-judge-indicators``; read from there with ``git show`` when the
  file is not in the checkout), forward order by default, scored against
  the author's verdict;
* ``session``: the whole analysis of ``sim-reunion-verano-audio``
  (confirmation, assessment, synthesis).

Every reply goes through the shared reply cache, whose keys include the
judge's name (``claude-code:<model>``): each model has its own namespace
and answers Opus already gave are served for free. ``--cap`` is a hard
limit of real calls over all phases; ``--dry-run`` prints how many calls
would be made and calls nothing. Run it from the checkout that holds
``data/`` (transcripts, models, the cache)::

    PYTHONPATH=. python experiments/skills/model_comparison.py \\
        run --model haiku --out experiments/skills/results/model_comparison
    ... compare --a opus --b haiku --out DIR   # writes comparison.json

Compare writes agreement of ``b`` with ``a`` and the tokens of each.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_analyses import levels, read_run  # noqa: E402

from src.adapters.outbound.claude_code_judge import (  # noqa: E402
    ClaudeCodeJudge,
)
from src.adapters.outbound.sqlite_cache import SqliteCache  # noqa: E402
from src.application.caching import cache_key  # noqa: E402
from src.application.skills.judges import (  # noqa: E402
    CallBudget,
    act_accuracy,
    act_items,
)
from src.application.skills.ports import ChatModel  # noqa: E402
from src.config import SkillsSettings  # noqa: E402
from src.domain.errors import LimitError  # noqa: E402
from src.domain.skills.act_prompting import (  # noqa: E402
    act_system_prompt,
    act_user_prompt,
)
from src.domain.skills.rubric import RubricCatalogue  # noqa: E402
from src.domain.skills.speech import Sentence  # noqa: E402
from src.domain.skills.tag_prompting import (  # noqa: E402
    confirm_system_prompt,
    confirm_user_prompt,
    parse_confirmation,
)
from src.skills_bootstrap import SkillsContainer  # noqa: E402

SESSION = "sim-reunion-verano-audio"
DIALOGUES = "examples/skills/dialogue_acts/dialogues.json"
INDICATORS = "examples/skills/judge_indicators/reference.json"
INDICATORS_BRANCH = "exp/85-judge-indicators"
RUBRIC = "knowledge/rubrics/core.es.json"
STORED = f"data/output/skills/analysis/{SESSION}"
STORED_ACTS = "experiments/skills/results/act_tagger/llm_claude_code.json"
"""The Opus verdicts on the 140 sentences, from ``skills judge_acts``."""
PHASES = ("indicators", "acts", "session")
MODELS = ("haiku", "sonnet", "opus")


def load_indicators(path: str) -> Dict[str, Any]:
    """Return the indicator reference, from ``path`` or its branch."""
    if Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text)  # type: ignore[no-any-return]
    shown = subprocess.run(
        ["git", "show", f"{INDICATORS_BRANCH}:{INDICATORS}"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(shown.stdout)  # type: ignore[no-any-return]


def sentence(position: int, speaker: str, text: str) -> Sentence:
    """Wrap a reference line as a sentence of a made-up turn."""
    return Sentence(id=f"t{position}.s0", turn=position, index=0,
                    speaker=speaker, text=text)


def indicator_prompts(
    item: Dict[str, Any], rubric: RubricCatalogue
) -> Tuple[str, str, List[str]]:
    """Return the system and user prompts of one item and its indicators."""
    offered = [rubric.competency(c) for c in item["offered"]]
    context = []
    for n, line in enumerate(item["context"]):
        who, _, text = line.partition(": ")
        context.append(sentence(n, who, text))
    target = sentence(len(context), item["speaker"], item["text"])
    ids = [i.id for c in offered for i in c.indicators]
    return (confirm_system_prompt(rubric.language),
            confirm_user_prompt(context, target, offered), ids)


def indicator_key(model: str, system: str, user: str) -> str:
    """Return the cache key of one indicator question."""
    return cache_key("indicator", f"claude-code:{model}", system, user)


def phase_indicators(
    judge: ChatModel, cache: Any, items: Sequence[Dict[str, Any]],
    rubric: RubricCatalogue,
) -> Dict[str, Any]:
    """Ask every item (cached); return the named indicator per item."""
    named: Dict[str, Optional[str]] = {}
    unparsed = 0
    for item in items:
        system, user, ids = indicator_prompts(item, rubric)
        key = cache_key("indicator", judge.model_name, system, user)
        reply = cache.get(key)
        if reply is None:
            reply = judge.complete(system, user, 120)
            cache.set(key, reply)
        found, _, ok = parse_confirmation(reply)
        if not ok:
            unparsed += 1
            continue
        named[item["id"]] = found if found in ids else None
    right = sum(1 for i in items if i["id"] in named
                and named[i["id"]] == i["expect"])
    return {"named": named, "unparsed": unparsed,
            "accuracy": round(right / len(items), 4)}


def phase_acts(
    container: SkillsContainer, judge: ChatModel
) -> Dict[str, Any]:
    """Classify the 140 act sentences (cached); return acts and accuracy."""
    documents = json.loads(Path(DIALOGUES).read_text(encoding="utf-8"))
    items = act_items(documents, 3, 140)
    verdicts = container.act_classifier(judge).classify(items)
    return {"acts": {v.id: v.act for v in verdicts},
            "accuracy": act_accuracy(items, verdicts)}


def phase_session(
    container: SkillsContainer, out: Path, cap: int
) -> Dict[str, Any]:
    """Run the whole analysis of the demo session; return where it is."""
    job = Path(STORED) / "job.json"
    target = out / "analysis" / SESSION
    target.mkdir(parents=True, exist_ok=True)
    if job.is_file() and not (target / "job.json").is_file():
        (target / "job.json").write_text(job.read_text(encoding="utf-8"),
                                         encoding="utf-8")
    run = container.session_analysis.run(SESSION, False, True)
    return {"directory": str(target), "status": run.status,
            "steps": {s.name: s.status for s in run.steps},
            "failed": [s.error for s in run.steps if s.status == "failed"],
            "cap": cap}


def count_missing(model: str) -> Dict[str, int]:
    """Return the real calls the indicator and act phases would make."""
    cache = SqliteCache("data/skills_cache")
    rubric = RubricCatalogue.model_validate(
        json.loads(Path(RUBRIC).read_text(encoding="utf-8")))
    reference = load_indicators(INDICATORS)
    missing_indicators = 0
    for item in reference["items"]:
        system, user, _ = indicator_prompts(item, rubric)
        if cache.get(indicator_key(model, system, user)) is None:
            missing_indicators += 1
    documents = json.loads(Path(DIALOGUES).read_text(encoding="utf-8"))
    system = act_system_prompt()
    missing_acts = sum(
        1 for it in act_items(documents, 3, 140)
        if cache.get(cache_key(
            "act", f"claude-code:{model}", system,
            act_user_prompt(it.context, it.speaker, it.text))) is None
    )
    return {"indicators": missing_indicators, "acts": missing_acts}


def stored_session_calls() -> int:
    """Return the judge calls the stored Opus analysis of the session made."""
    path = Path(STORED) / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    return sum(int(s["counts"].get("judge_calls", 0)) for s in run["steps"])


def dry_run(model: str, phases: Sequence[str], cap: int) -> int:
    """Print how many real calls ``model`` would make; return the total."""
    missing = count_missing(model)
    session = stored_session_calls()
    if model == "opus":
        print("opus: the session and the 140 acts have a stored baseline "
              "(see compare); only the indicators are missing")
    wanted = {"indicators": missing["indicators"], "acts": missing["acts"],
              "session": session}
    print(f"model {model}: calls per phase (cache checked for the first "
          f"two; session = calls of the stored Opus run, an upper bound "
          f"for opus, whose cached answers are free)")
    total = 0
    for phase in phases:
        print(f"  {phase:<11} {wanted[phase]}")
        total += wanted[phase]
    print(f"  total       {total} (cap {cap})")
    return total


def run(args: argparse.Namespace) -> None:
    """Run the chosen phases with ``--model`` and write ``<model>.json``."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = SkillsSettings.model_validate({
        "judge": "claude_code", "claude_model": args.model,
        "analysis_dir": str(out / "analysis"),
        "judge_state_path": str(out / "judge.json"),
        "max_judge_calls": max(args.cap, 1), "progress": False,
        "max_confirmations": 150,
    })
    claude = ClaudeCodeJudge(args.model, settings.claude_executable,
                             settings.claude_timeout, settings.claude_effort)
    container = SkillsContainer(settings, judge=claude)
    budget = CallBudget(claude, args.cap, container.usage)
    result: Dict[str, Any] = {"model": args.model, "cap": args.cap,
                              "phases": {}, "usage": {}}
    rubric = RubricCatalogue.model_validate(
        json.loads(Path(RUBRIC).read_text(encoding="utf-8")))
    try:
        for phase in args.phases:
            before = container.usage.snapshot()
            calls = budget.calls
            if phase == "indicators":
                items = load_indicators(args.indicators)["items"]
                data = phase_indicators(budget, container.cache, items, rubric)
            elif phase == "acts":
                data = phase_acts(container, budget)
            else:
                spent = budget.calls
                settings_left = max(args.cap - spent, 1)
                container.settings.max_judge_calls = settings_left
                data = phase_session(container, out, settings_left)
            used = container.usage.snapshot().minus(before)
            result["phases"][phase] = data
            result["usage"][phase] = used.model_dump()
            print(f"{phase}: {budget.calls - calls} calls through the "
                  f"budget, {used.cache_hits} cache hits")
    except LimitError as exc:
        result["stopped"] = str(exc)
        print(f"STOPPED: {exc}")
    result["usage_total"] = container.usage.snapshot().model_dump()
    path = out / f"{args.model}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"Saved {path}")


def same(a: Any, b: Any) -> Optional[float]:
    """Return the share of equal pairs over keys both have."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    return round(sum(1 for k in keys if a[k] == b[k]) / len(keys), 4)


def session_labels(directory: str) -> Dict[str, Any]:
    """Return per-sentence decisions and per-person levels of a run."""
    run = read_run(Path(directory))
    decisions: Dict[str, str] = {}
    for item in run["confirmation"].get("sentences", []):
        decision = item.get("decision")
        if decision:
            label = (decision.get("indicator_id") or "none"
                     if decision["status"] == "confirmed"
                     else decision["status"])
            decisions[item["sentence"]["id"]] = label
    ratings: Dict[str, Any] = {}
    for person in run["summary"].get("people", []):
        for competency, value in levels(person).items():
            ratings[f"{person['name']}/{competency}"] = list(value)
    return {"decisions": decisions, "ratings": ratings}


def compare(args: argparse.Namespace) -> None:
    """Write the agreement of model ``b`` with ``a`` and both usages."""
    out = Path(args.out)
    first = json.loads((out / f"{args.a}.json").read_text(encoding="utf-8"))
    second = json.loads((out / f"{args.b}.json").read_text(encoding="utf-8"))
    report: Dict[str, Any] = {"a": args.a, "b": args.b}
    pa, pb = first["phases"], second["phases"]
    if "acts" not in pa and "acts" in pb and args.a == "opus":
        stored = json.loads(Path(args.stored_acts).read_text(encoding="utf-8"))
        pa["acts"] = {"acts": {v["id"]: v["act"] for v in stored["first"]},
                      "accuracy": stored["accuracy"]}
    if "acts" in pa and "acts" in pb:
        report["acts"] = {
            "agreement": same(pa["acts"]["acts"], pb["acts"]["acts"]),
            f"accuracy_{args.a}": pa["acts"]["accuracy"],
            f"accuracy_{args.b}": pb["acts"]["accuracy"],
        }
    if "indicators" in pa and "indicators" in pb:
        report["indicators"] = {
            "agreement": same(pa["indicators"]["named"],
                              pb["indicators"]["named"]),
            f"accuracy_{args.a}": pa["indicators"]["accuracy"],
            f"accuracy_{args.b}": pb["indicators"]["accuracy"],
        }
    if "session" in pb:
        base = args.stored_a or pa.get("session", {}).get(
            "directory", STORED)
        a = session_labels(base)
        b = session_labels(pb["session"]["directory"])
        firm = [k for k in set(a["ratings"]) & set(b["ratings"])
                if a["ratings"][k][0] == b["ratings"][k][0] == "firm"]
        report["session"] = {
            "confirmation_agreement": same(a["decisions"], b["decisions"]),
            "confirmation_items": len(
                set(a["decisions"]) & set(b["decisions"])),
            "rating_agreement": same(a["ratings"], b["ratings"]),
            "rating_items": len(set(a["ratings"]) & set(b["ratings"])),
            "firm_within_one": (
                round(sum(1 for k in firm if abs(
                    a["ratings"][k][1] - b["ratings"][k][1]) <= 1)
                    / len(firm), 4) if firm else None),
            "firm_items": len(firm),
        }
    report["usage"] = {
        args.a: first.get("usage", {}), args.b: second.get("usage", {})}
    report["usage_total"] = {
        args.a: first.get("usage_total"), args.b: second.get("usage_total")}
    path = out / "comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "usage"},
                     ensure_ascii=False, indent=1))
    print(f"Saved {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the arguments and run, count (``--dry-run``) or compare."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="run the phases with one model")
    one.add_argument("--model", choices=MODELS, required=True)
    one.add_argument("--phases", nargs="+", choices=PHASES,
                     default=list(PHASES))
    one.add_argument("--cap", type=int, default=150,
                     help="hard limit of real calls over all phases")
    one.add_argument("--dry-run", action="store_true",
                     help="print the number of calls; call nothing")
    one.add_argument("--indicators", default=INDICATORS)
    one.add_argument("--out", default="experiments/skills/results/"
                     "model_comparison")
    two = sub.add_parser("compare", help="agreement of two saved runs")
    two.add_argument("--a", choices=MODELS, default="opus")
    two.add_argument("--b", choices=MODELS, required=True)
    two.add_argument("--stored-a", default="",
                     help="analysis directory of the baseline session "
                     "(default: the stored Opus run)")
    two.add_argument("--stored-acts", default=STORED_ACTS,
                     help="Opus verdicts on the 140 sentences")
    two.add_argument("--out", default="experiments/skills/results/"
                     "model_comparison")
    args = parser.parse_args(argv)
    if args.command == "compare":
        compare(args)
    elif args.dry_run:
        dry_run(args.model, args.phases, args.cap)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
