"""``python -m orchestrator rebuild-round``: rebuild an ended round (#264).

When the orchestrator lost a live round (a restart during or after it,
a failed judgement, a session the transcriber recovered from its log
after its own restart), the conversation is still kept: the
transcriber's ``transcript.v1`` (``--transcript``) or its live log
(``--events``, ``<transcriber data>/live/<session>/events.jsonl``).
This command replays it through the round's trackers with
:class:`~orchestrator.application.rebuild_round.RoundRebuild`: the
topics of the role's plan (``--plan``), the product's mandatory items
and the lines to avoid (only with ``--knowledge http``: they ask the
RAG), then judges the candidate's answers (``--candidate``; not with
``--no-judge``) and stores the report as at the end of the round. The
hand-off record (checklist and the judgement's labels, no words) is
written next to the serve's (``--handoff-dir``).

It prints only numbers and states, never the conversation's words.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from orchestrator.adapters.outbound.files import load_transcript
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_core_questions import FileCoreQuestions
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.adapters.outbound.jsonl_sink import JsonHandoff
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.live_judgement import without_words
from orchestrator.application.ports import (
    AvoidPort,
    KnowledgePort,
    MandatoryPort,
    ReasonerPort,
)
from orchestrator.application.rebuild_round import (
    RoundRebuild,
    events_of_log,
    events_of_transcript,
)
from orchestrator.domain.config import InterviewSection, ProductConfig

ROOT = Path(__file__).resolve().parents[3]
RAG_TOKEN_FILE = Path("~/.config/ragputin/rag-token").expanduser()


def add_round_parser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
    config_dir: Path,
) -> None:
    """Add ``rebuild-round``."""
    parser = sub.add_parser(
        "rebuild-round",
        help="rebuild an ended live round from its transcript or log")
    parser.add_argument("session", help="the live session id")
    kept = parser.add_mutually_exclusive_group()
    kept.add_argument("--transcript", type=Path, default=None,
                      help="its transcript.v1 (default: data/transcripts/"
                           "<session>.transcript.v1.json)")
    kept.add_argument("--events", type=Path, default=None,
                      help="its live log (events.jsonl)")
    parser.add_argument("--product", default="live_assist.interview")
    parser.add_argument("--config-dir", type=Path, default=config_dir)
    parser.add_argument("--plan", default=None, help="the role's plan id")
    parser.add_argument("--candidate", default=None,
                        help="the candidate's profile id in that plan")
    parser.add_argument("--no-judge", action="store_true",
                        help="follow the trackers only")
    parser.add_argument("--interview-dir", type=Path,
                        default=ROOT / "data" / "interview")
    parser.add_argument("--handoff-dir", type=Path,
                        default=ROOT / "data" / "live" / "serve")
    parser.add_argument("--reasoner", default="http",
                        help="http | fake | module:factory")
    parser.add_argument("--reasoner-url", default="http://127.0.0.1:8000")
    parser.add_argument("--knowledge", default="http", help="fake | http")
    parser.add_argument("--knowledge-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rag-token-file", type=Path, default=RAG_TOKEN_FILE)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--fake-delay-ms", type=float, default=0.0)


def read_lines(path: Path) -> List[Dict[str, Any]]:
    """The JSON objects of a live log; a line cut by a crash is skipped."""
    found: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def run_round(
    args: argparse.Namespace,
    product: ProductConfig,
    reasoner: ReasonerPort,
    knowledge: KnowledgePort,
    mandatory: Optional[MandatoryPort] = None,
    avoid: Optional[AvoidPort] = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run ``rebuild-round``; print a summary of numbers and states."""
    if args.events is not None:
        events, roles = events_of_log(read_lines(args.events))
    else:
        path = args.transcript or (ROOT / "data" / "transcripts"
                                   / f"{args.session}.transcript.v1.json")
        transcript = load_transcript(path)
        if transcript.session.id != args.session:
            raise ValueError(f"the transcript is of session "
                             f"{transcript.session.id!r}, not "
                             f"{args.session!r}")
        events, roles = events_of_transcript(transcript)
    if not events:
        raise ValueError("nothing to rebuild: no events")
    store = JsonPlanStore(args.interview_dir)
    plan = profile = None
    if args.plan is not None:
        try:
            plan = store.load(args.plan)
        except KeyError as exc:
            raise ValueError(f"no plan {args.plan!r}") from exc
        if args.candidate is not None:
            try:
                profile = JsonCandidateStore(args.interview_dir).load(
                    args.plan, args.candidate)
            except KeyError as exc:
                raise ValueError(f"no candidate {args.candidate!r} in "
                                 "that plan") from exc
    wiring = None
    if plan is not None and not args.no_judge:
        planner = InterviewPlanner(store, knowledge, reasoner, None,
                                   lambda: InterviewSection())
        book = CandidateBook(
            JsonCandidateStore(args.interview_dir),
            FileCoreQuestions(ROOT / "knowledge" / "interview"
                              / "core-questions.json"),
            lambda plan_id: plan is not None and plan_id == plan.id,
            plans=store)
        wiring = InterviewWiring(planner, store, reasoner, candidates=book)
    rebuilt = RoundRebuild(InlineExecutor(), mandatory, avoid, wiring
                           ).rebuild(
        args.session, events, roles, product=product, plan=plan,
        rounds=profile.rounds if profile is not None else (),
        candidate=profile.name if profile is not None else None,
        candidate_id=profile.id if profile is not None else None,
        model=args.model, judge=not args.no_judge)
    record: Dict[str, Any] = {
        "session_id": args.session, "product": product.product,
        "rebuilt": True, "mandatory": rebuilt.mandatory,
        "avoid": len(rebuilt.avoid or []) if rebuilt.avoid is not None
        else None,
    }
    if rebuilt.accuracy is not None:
        record["accuracy"] = without_words(rebuilt.accuracy)
    JsonHandoff(args.handoff_dir / f"{args.session}.{product.product}"
                ".handoff.json").write(record)
    out(json.dumps(rebuilt.summary(), ensure_ascii=False))
    return 0
