"""``python -m orchestrator interview rejudge``: judge a live round again.

For a live-page round whose judgement of the answers (#259) failed or
was lost (a restart while it ran): the round is rebuilt offline from
the stored ``transcript.v1`` of the session (default
``data/transcripts/<session>.transcript.v1.json``; its speakers carry
``role: interviewer | candidate``), replayed through the topic tracker
and judged with the same template; the full report is stored with the
plan's results (``<interview dir>/results/<plan>/<session>.json``), so
it shows in «Comparar candidatos» and in the candidate's file (#262).

The reasoner is the one the orchestrator uses: ``http`` (default) asks
the running RAG on ``--reasoner-url`` with the token of
``--rag-token-file``, which must already have ``interview.report@2``;
``module:factory`` builds another adapter, e.g.
``experiments.live.orchestrator_reasoner:build`` (Haiku in process
through ``claude -p``, the owner's subscription). It works with the
services running and prints only numbers and states, never the
candidate's words. A closed position can be judged too: the report
belongs to the candidate's file.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from orchestrator.adapters.outbound.files import load_transcript
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_core_questions import FileCoreQuestions
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.live_judgement import (
    LIVE_JUDGE_TEMPLATE,
    rejudge,
)
from orchestrator.application.ports import KnowledgePort, ReasonerPort
from orchestrator.domain.config import InterviewSection

ROOT = Path(__file__).resolve().parents[3]
RAG_TOKEN_FILE = Path("~/.config/ragputin/rag-token").expanduser()


def add_interview_parser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add ``interview`` and its ``rejudge`` action."""
    parser = sub.add_parser(
        "interview", help="interview reports kept with the plans (#262)")
    actions = parser.add_subparsers(dest="action", required=True)
    again = actions.add_parser(
        "rejudge", help="judge a live round again from its transcript")
    again.add_argument("session", help="the live session id")
    again.add_argument("--plan", required=True, help="the role's plan id")
    again.add_argument("--candidate", required=True,
                       help="the candidate's profile id in that plan")
    again.add_argument("--transcript", type=Path, default=None,
                       help="the session's transcript.v1 (default: "
                            "data/transcripts/<session>.transcript.v1.json)")
    again.add_argument("--interview-dir", type=Path,
                       default=ROOT / "data" / "interview")
    again.add_argument("--reasoner", default="http",
                       help="http | fake | module:factory")
    again.add_argument("--reasoner-url", default="http://127.0.0.1:8000")
    again.add_argument("--knowledge", default="http", help="fake | http")
    again.add_argument("--knowledge-url", default="http://127.0.0.1:8000")
    again.add_argument("--rag-token-file", type=Path, default=RAG_TOKEN_FILE)
    again.add_argument("--model", default="haiku")
    again.add_argument("--force", action="store_true",
                       help="judge a session that is not one of the "
                            "candidate's rounds, or replace a stored report")
    again.add_argument("--fake-delay-ms", type=float, default=0.0)


def run_interview(
    args: argparse.Namespace,
    reasoner: ReasonerPort,
    knowledge: KnowledgePort,
) -> int:
    """Run ``interview rejudge``; print a summary of numbers and states."""
    path = args.transcript or (ROOT / "data" / "transcripts"
                               / f"{args.session}.transcript.v1.json")
    transcript = load_transcript(path)
    if transcript.session.id != args.session:
        raise ValueError(f"the transcript is of session "
                         f"{transcript.session.id!r}, not {args.session!r}")
    store = JsonPlanStore(args.interview_dir)
    try:
        plan = store.load(args.plan)
    except KeyError as exc:
        raise ValueError(f"no plan {args.plan!r}") from exc
    profiles = JsonCandidateStore(args.interview_dir)
    try:
        profile = profiles.load(args.plan, args.candidate)
    except KeyError as exc:
        raise ValueError(f"no candidate {args.candidate!r} in that plan") \
            from exc
    if not args.force:
        if all(r.session_id != args.session for r in profile.rounds):
            raise ValueError("that session is not one of the candidate's "
                             "rounds (--force to judge it anyway)")
        try:
            store.result(args.plan, args.session)
        except KeyError:
            pass
        else:
            raise ValueError("a report of that session is already stored "
                             "(--force to replace it)")
    planner = InterviewPlanner(store, knowledge, reasoner, None,
                               lambda: InterviewSection())
    book = CandidateBook(
        profiles, FileCoreQuestions(ROOT / "knowledge" / "interview"
                                    / "core-questions.json"),
        lambda plan_id: plan_id == plan.id, plans=store)
    wiring = InterviewWiring(planner, store, reasoner, candidates=book)
    result = rejudge(transcript, plan, wiring, args.model, profile.name,
                     profile.id)
    print(json.dumps(summary(result, plan.state), ensure_ascii=False))
    return 0 if result is not None and result.get("saved") else 1


def summary(result: Any, plan_state: str) -> Dict[str, Any]:
    """What the command prints: states and counts, no words."""
    if result is None:
        return {"state": "nothing_answered", "saved": False,
                "plan_state": plan_state}
    labels: Dict[str, int] = {}
    for slot in result.get("slots") or []:
        for key, n in (slot.get("accuracy") or {}).items():
            labels[key] = labels.get(key, 0) + int(n)
    return {"state": result.get("state"), "saved": bool(result.get("saved")),
            "judge": result.get("judge") or LIVE_JUDGE_TEMPLATE,
            "slots": len(result.get("slots") or []),
            "to_check": len(result.get("to_check") or []),
            "labels": labels, "plan_state": plan_state}
