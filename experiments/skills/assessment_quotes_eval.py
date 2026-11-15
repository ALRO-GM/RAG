"""Measure the judges' quotes in the assessment step (#104).

For each synthetic session, judge and prompt style it runs step g
(``AssessmentService``) of ``skills analyze`` with the analyzer's own
classes and counts what happens to the judge's quotes: proposed,
grounded (found verbatim in a turn of the participant), kept by the
similarity filter at each threshold of ``--thresholds``, and the
competencies rated. Every grounded quote is listed with its cosine, so
the hand labels of ``--labels`` (``good``/``bad``: does the turn show
the indicator?) give, per threshold, how many good quotes the filter
keeps and how many bad ones pass.

The replies are kept in ``--replies`` (a JSON object: cache key ->
reply, the keys of the analyzer's cache), so the measurement can be
repeated without a model; ``--seed_cache`` reads missing replies from
an existing reply cache directory (read only; for Claude's replies of
the demo run). Claude Code is called only with ``--claude_calls`` > 0
and at most that many times; local models load with the ``fixed``
policy (they wait for GPU memory, never take a smaller model).

Only synthetic sessions belong here: the replies file and the output
contain the transcripts' words.

Usage::

    S=examples/skills/simulated
    HF_HUB_OFFLINE=1 PYTHONPATH=. python \\
        experiments/skills/assessment_quotes_eval.py \\
        --sessions $S/sim-reunion-verano.transcript.v1.json \\
        --judges claude-code:opus,Qwen/Qwen3-8B --styles standard,guided \\
        --replies experiments/skills/results/assessment_quotes_replies.json \\
        --labels experiments/skills/results/assessment_quotes_labels.json \\
        --output experiments/skills/results/assessment_quotes.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.adapters.outbound.sqlite_cache import SqliteCache
from src.application.skills.assessment import JudgeTuning
from src.application.skills.judges import CallBudget, release
from src.application.skills.ports import ChatModel
from src.config import SkillsSettings
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.prompting import PromptStyle
from src.skills_bootstrap import SkillsContainer

THRESHOLDS = (-1.0, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
Label = Tuple[str, str, int, str]


class JsonReplies:
    """A reply cache kept in a JSON file, with an optional read-only seed."""

    def __init__(self, path: str, seed: Optional[SqliteCache] = None):
        """Load ``path`` when it exists; ``seed`` fills misses."""
        self.path = Path(path)
        self.seed = seed
        self.data: Dict[str, str] = {}
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[str]:
        """Return the stored reply (copied from the seed if needed)."""
        if key in self.data:
            return self.data[key]
        found = self.seed.get(key) if self.seed else None
        if found is not None:
            self.set(key, found)
        return found

    def clear(self) -> int:
        """Forget nothing: the measured replies are kept on purpose."""
        return 0

    def set(self, key: str, value: str) -> None:
        """Store a reply and write the file (not the empty reply of a gap)."""
        if not value:
            return
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=0,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )


class NoCall:
    """A judge that never answers: a missing reply is a measured gap."""

    def __init__(self, name: str):
        """Keep the name the replies are keyed by."""
        self.name = name
        self.misses = 0

    @property
    def model_name(self) -> str:
        """Return the judge's identifier."""
        return self.name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Count the miss and return an unparseable reply."""
        self.misses += 1
        return ""


def container(judge: str, style: PromptStyle) -> SkillsContainer:
    """Return the analyzer's classes for one judge and prompt style."""
    local = not judge.startswith("claude-code:")
    settings = SkillsSettings.model_validate({
        "judge": "local" if local else "claude_code",
        "llm": judge if local else "auto",
        "local_bits": 4, "local_policy": "fixed",
        "local_wait_seconds": 1800.0, "local_prompt": style,
        "claude_model": judge.split(":", 1)[1] if not local else "opus",
        "progress": False, "use_cache": False,
    })
    return SkillsContainer(settings)


def judge_for(
    app: SkillsContainer, judge: str, claude_calls: int, replay: bool,
) -> ChatModel:
    """Return the model behind ``judge`` (Claude only within budget).

    ``replay`` answers from the stored replies only, without loading a
    model (a local judge is keyed by its name at 4 bits).
    """
    if replay:
        local = not judge.startswith("claude-code:")
        return NoCall(f"{judge}@4bit" if local else judge)
    if not judge.startswith("claude-code:"):
        return app.local_judge
    if claude_calls <= 0:
        return NoCall(judge)
    return CallBudget(app.claude_judge, claude_calls)


def load_labels(path: str) -> Dict[Label, str]:
    """Return the hand labels by (session, participant, turn, indicator)."""
    if not path or not Path(path).is_file():
        return {}
    items = json.loads(Path(path).read_text(encoding="utf-8"))["labels"]
    return {
        (i["session"], i["participant"], int(i["turn"]), i["indicator"]):
        i["label"] for i in items
    }


def quotes(result: SessionAssessment) -> List[Dict[str, Any]]:
    """Return every kept quote of an assessment with its cosine."""
    return [
        {
            "participant": card.participant,
            "competency": item.competency_id,
            "turn": e.turn, "indicator": e.indicator_id,
            "similarity": e.similarity, "quote": e.quote,
        }
        for card in result.scorecards
        for item in card.assessments
        for e in item.evidence
    ]


def counts(result: SessionAssessment) -> Dict[str, int]:
    """Return the quote and rating counts of an assessment."""
    items = [a for c in result.scorecards for a in c.assessments]
    return {
        "judgements": len(items),
        "proposed": sum(a.proposed for a in items),
        "not_grounded": sum(a.rejected for a in items),
        "grounded": sum(len(a.evidence) + a.unsupported for a in items),
        "kept": sum(len(a.evidence) for a in items),
        "rated": sum(a.rating is not None for a in items),
        "unparseable": sum(
            "unparseable_reply" in a.flags for a in items
        ),
    }


def auc(good: Sequence[float], bad: Sequence[float]) -> Optional[float]:
    """Return how often a good quote's cosine beats a bad one's (ties 1/2).

    0.5 means the cosine does not tell good quotes from bad ones.
    """
    if not good or not bad:
        return None
    wins = sum(
        1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad
    )
    return round(wins / (len(good) * len(bad)), 3)


def run_one(
    session: str, judge: str, style: PromptStyle, replies: JsonReplies,
    thresholds: Sequence[float], labels: Dict[Label, str],
    claude_calls: int, replay: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Measure one session with one judge; return it and Claude calls."""
    app = container(judge, style)
    model = judge_for(app, judge, claude_calls, replay)
    service = app.assessment_with(model)
    service.cache = replies
    document = app.transcripts.read(session)
    rubric = app.rubric(app.retriever.language_of(document))
    asked: PromptStyle = "standard" if judge.startswith(
        "claude-code:"
    ) else style

    def floor(threshold: float) -> None:
        """Ask with ``asked`` and keep quotes from ``threshold`` up."""
        service.tuning = lambda name: JudgeTuning(asked, threshold)

    floor(-1.0)
    full = service.assess(document, rubric)
    spent = int(getattr(model, "calls", 0))
    out: Dict[str, Any] = {
        "session": document.session_id, "judge": judge,
        "judge_model": full.judge_model,
        "style": asked,
        "seconds": full.seconds, "missing_replies": int(
            getattr(model, "misses", 0)
        ),
        **counts(full), "quotes": quotes(full), "by_threshold": [],
    }
    scored: Dict[str, List[float]] = {"good": [], "bad": []}
    for item in out["quotes"]:
        label = labels.get((document.session_id, item["participant"],
                            item["turn"], item["indicator"]))
        if label in scored and item["similarity"] is not None:
            scored[label].append(item["similarity"])
    out["cosine_auc"] = auc(scored["good"], scored["bad"])
    for threshold in thresholds:
        floor(threshold)
        result = service.assess(document, rubric)
        row: Dict[str, Any] = {"threshold": threshold, **counts(result)}
        good = bad = good_kept = bad_kept = 0
        for item in quotes(full):
            key = (document.session_id, item["participant"], item["turn"],
                   item["indicator"])
            label = labels.get(key)
            passes = (item["similarity"] is None
                      or item["similarity"] >= threshold)
            if label == "good":
                good += 1
                good_kept += passes
            elif label == "bad":
                bad += 1
                bad_kept += passes
        row.update({"labelled_good": good, "good_kept": good_kept,
                    "labelled_bad": bad, "bad_kept": bad_kept})
        out["by_threshold"].append(row)
    release(model)
    return out, spent


def at(run: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Return the row of ``run`` for ``threshold``."""
    return next(
        r for r in run["by_threshold"] if abs(r["threshold"] - threshold)
        < 1e-9
    )


def table(runs: Sequence[Dict[str, Any]], low: float, high: float) -> str:
    """Return the runs as a Markdown table (counts at two thresholds).

    Precision is the share of hand-labelled grounded quotes labelled
    ``good``, over every grounded quote (threshold -1).
    """
    lines = [
        "| Session | Judge | Prompt | Proposed | Grounded | Good / "
        f"labelled | Kept ≥{high:g} | Rated ≥{high:g} | Kept ≥{low:g} | "
        f"Rated ≥{low:g} | Cosine AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        every, strict, loose = at(run, -1.0), at(run, high), at(run, low)
        labelled = every["labelled_good"] + every["labelled_bad"]
        lines.append(
            f"| {run['session']} | {run['judge_model']} | {run['style']} | "
            f"{run['proposed']} | {run['grounded']} | "
            f"{every['labelled_good']} / {labelled} | {strict['kept']} | "
            f"{strict['rated']} | {loose['kept']} | {loose['rated']} | "
            f"{run.get('cosine_auc')} |"
        )
    return "\n".join(lines)


def sweep(runs: Sequence[Dict[str, Any]]) -> str:
    """Return, per run and threshold, the good and bad quotes kept."""
    thresholds = [r["threshold"] for r in runs[0]["by_threshold"]]
    lines = [
        "| Session | Judge | Prompt | " + " | ".join(
            f"≥{t:g} good / bad" for t in thresholds
        ) + " |",
        "|---|---|---|" + "---|" * len(thresholds),
    ]
    for run in runs:
        cells = [
            f"{r['good_kept']}/{r['labelled_good']} · "
            f"{r['bad_kept']}/{r['labelled_bad']}"
            for r in run["by_threshold"]
        ]
        lines.append(
            f"| {run['session']} | {run['judge_model']} | {run['style']} | "
            + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Parse the arguments, run every combination, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", default="",
                        help="comma-separated transcript files")
    parser.add_argument("--judges", default="",
                        help="comma-separated: claude-code:<model> or a "
                        "Hugging Face model of the ladder")
    parser.add_argument("--styles", default="standard,guided",
                        help="prompt styles of the local judges")
    parser.add_argument("--replies", default="")
    parser.add_argument("--seed_cache", default="",
                        help="reply cache directory to copy replies from")
    parser.add_argument("--labels", default="")
    parser.add_argument("--claude_calls", type=int, default=0,
                        help="most Claude Code calls of the whole run")
    parser.add_argument("--thresholds", default=",".join(
        str(t) for t in THRESHOLDS))
    parser.add_argument("--replay", action="store_true",
                        help="only the stored replies; no model is loaded")
    parser.add_argument("--output", required=True)
    parser.add_argument("--table", default="",
                        help="print the table of --output at LOW,HIGH "
                        "thresholds (e.g. 0,0.3) and stop")
    args = parser.parse_args()
    if args.table:
        low, high = (float(t) for t in args.table.split(","))
        runs = json.loads(Path(args.output).read_text(encoding="utf-8"))
        print(table(runs["runs"], low, high))
        print()
        print(sweep(runs["runs"]))
        return
    if not (args.sessions and args.judges and args.replies):
        parser.error("--sessions, --judges and --replies are required")
    seed = SqliteCache(args.seed_cache) if args.seed_cache else None
    replies = JsonReplies(args.replies, seed)
    labels = load_labels(args.labels)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    output = Path(args.output)
    done: List[Dict[str, Any]] = []
    if output.is_file():
        done = json.loads(output.read_text(encoding="utf-8"))["runs"]
    budget = args.claude_calls
    for judge in args.judges.split(","):
        styles = ["standard"] if judge.startswith("claude-code:") else [
            s for s in args.styles.split(",")
        ]
        for style in styles:
            for session in args.sessions.split(","):
                style_name: PromptStyle = (
                    "guided" if style == "guided"
                    else "precise" if style == "precise" else "standard"
                )
                run, spent = run_one(
                    session, judge, style_name, replies, thresholds,
                    labels, budget, args.replay,
                )
                budget -= spent
                done = [r for r in done if (
                    r["session"], r["judge"], r["style"]
                ) != (run["session"], run["judge"], run["style"])]
                done.append(run)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(
                    {"thresholds": thresholds, "runs": done},
                    ensure_ascii=False, indent=1,
                ) + "\n", encoding="utf-8")
                print(json.dumps({
                    k: v for k, v in run.items()
                    if k not in ("quotes", "by_threshold")
                }, ensure_ascii=False))


if __name__ == "__main__":
    main()
