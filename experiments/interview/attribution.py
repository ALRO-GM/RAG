"""Small measurement of who answered what in a live round (#268).

``attribution_set.json`` holds three invented live-page rounds of one
plan, in spoken Spanish, with the writer's expected attribution: per
point, the candidate utterances that answer it and the concepts treated
or said to be unknown. Each round is attributed three ways:

* ``rules``: the round replayed through the live page's topic tracker
  (the word rules of #258, no model);
* ``live``: the same replay, plus the live page's tick call
  (``interview.ticks@1``, :class:`LiveTicker`) after every candidate
  final (#268). The committed results of this way were measured with
  ``suggester.interview@3``, the first design (commit 27f941c), later
  replaced after a warm-path measurement (``warm_ticks.py``);
* ``after``: the model reading the whole round after the session
  (``interview.attribution@1`` through
  :meth:`~orchestrator.application.live_judgement.LiveJudge.attribute_round`).

The result gives, per way, precision and recall of the (utterance,
point) pairs and of the treated concepts, the unknown concepts found,
and whether the line that tries to instruct the model got anything.

The real model is Haiku through ``claude -p`` (the owner's subscription,
one cold process per call, in process, sequential; never a paid API
key): ``--repeats`` × 3 calls for ``after`` and one call per candidate
final for ``live`` (its latency here is a cold process's, not the warm
one of the live page). ``--fake`` runs the same path on the fake
reasoner::

    python -m experiments.interview.attribution --out R.json --repeats 2

Synthetic: the set was written for this purpose by a model; agreement
with the writer's intent is not accuracy on real interviews.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
)
from orchestrator.application.live_judgement import (
    LiveJudge,
    replayed_tracker,
)
from orchestrator.application.live_ticks import LiveTicker
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.ports import CancelToken, Document
from orchestrator.application.replay import transcript_to_events
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview import InterviewPlan
from orchestrator.domain.interview_attribution import RoundLine
from orchestrator.domain.next_round import topic_states

HERE = Path(__file__).resolve().parent
SET = HERE / "attribution_set.json"
SPEAKERS = {"interviewer": "S1", "candidate": "S2"}
WINDOW = 6
"""Lines of the conversation the suggester sees (the live product's)."""

Pairs = Set[Tuple[str, str]]
Concepts = Dict[Tuple[str, str], str]


class Counting:
    """The real reasoner in process (``claude -p``), counting calls."""

    def __init__(self, timeout: float) -> None:
        """Build the reasoner over Claude Code (one process per call)."""
        from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
        from src.application.reasoning import Reasoner

        self.inner = Reasoner(lambda alias: ClaudeCodeJudge(
            alias or "haiku", timeout=timeout, retries=0))
        self.calls: List[Dict[str, Any]] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Ask the real reasoner and keep what happened (no words)."""
        from src.domain.reasoning.roles import ReasonRequest

        started = time.perf_counter()
        answer: Document = self.inner.reason(
            ReasonRequest.model_validate(request)).to_contract()
        self.calls.append({
            "template": request.get("template"),
            "status": answer.get("status"),
            "wall_ms": round((time.perf_counter() - started) * 1000.0),
            "output_tokens": (answer.get("usage") or {}).get(
                "output_tokens"),
        })
        return answer


class Fake(FakeReasoner):
    """The fake reasoner with the same call log as :class:`Counting`."""

    def __init__(self) -> None:
        """No model."""
        super().__init__()
        self.calls: List[Dict[str, Any]] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer without a model and log the call."""
        answer = super().reason(request, cancel)
        self.calls.append({"template": request.get("template"),
                           "status": answer.get("status"),
                           "wall_ms": 0, "output_tokens": None})
        return answer


def load() -> Tuple[InterviewPlan, List[Dict[str, Any]], str]:
    """The set's plan, rounds and note."""
    data = json.loads(SET.read_text(encoding="utf-8"))
    return (InterviewPlan.model_validate(data["plan"]), data["rounds"],
            data["note"])


def transcript(round_: Dict[str, Any]) -> Transcript:
    """The round as a ``transcript.v1`` (S1 interviews, S2 answers)."""
    return Transcript.model_validate({
        "schema": "transcript.v1",
        "session": {"id": f"set-{round_['id']}", "language": "es",
                    "started_at": "2026-11-25T10:00:00Z"},
        "speakers": [{"id": "S1", "role": "interviewer"},
                     {"id": "S2", "role": "candidate"}],
        "utterances": [
            {"id": line["id"], "turn": n, "index": 0,
             "speaker": SPEAKERS[line["role"]], "start": float(n),
             "end": n + 0.9, "text": line["text"]}
            for n, line in enumerate(round_["lines"])],
    })


def expected(round_: Dict[str, Any]) -> Tuple[Pairs, Concepts]:
    """The writer's (utterance, point) pairs and concept statuses."""
    pairs: Pairs = set()
    concepts: Concepts = {}
    for sid, item in round_["expected"].items():
        pairs |= {(u, sid) for u in item["answers"]}
        concepts.update({(sid, pid): status
                         for pid, status in item["points"].items()})
    return pairs, concepts


def tracked(topics: TopicTracker) -> Tuple[Pairs, Concepts]:
    """What a topic tracker attributed and ticked."""
    pairs = {(u, sid) for sid, said in topics.answers.items()
             for u, _ in said}
    concepts: Concepts = {(sid, p.id): p.status
                          for sid, points in topics.points.items()
                          for p in points if p.status != "pending"}
    return pairs, concepts


def rules_way(plan: InterviewPlan, round_: Dict[str, Any]
              ) -> Tuple[Pairs, Concepts]:
    """The live page's word rules on the replayed round."""
    return tracked(replayed_tracker(transcript(round_), plan))


def live_way(plan: InterviewPlan, round_: Dict[str, Any], reasoner: Any
             ) -> Tuple[Pairs, Concepts]:
    """The rules plus the live page's tick call after each candidate final.

    Exactly as the live page: a :class:`LiveTicker` (``interview.ticks@1``,
    the last :data:`WINDOW` lines, the pending concepts of
    :meth:`TopicTracker.focus_points`), run inline here.
    """
    roles = {"S1": "interviewer", "S2": "candidate"}
    topics = TopicTracker(plan, topic_states(plan, []), roles,
                          lambda message: None)
    # Inline and unspaced: each candidate final is asked about at once
    # (the live page batches them, one call per 30 s at most).
    ticker = LiveTicker(reasoner, topics, roles, lambda job: job(),
                        every_s=0.0)
    for event in transcript_to_events(transcript(round_), roles):
        topics.on_event(event)
        ticker.on_event(event)
    return tracked(topics)


def after_way(plan: InterviewPlan, round_: Dict[str, Any], reasoner: Any
              ) -> Optional[Tuple[Pairs, Concepts]]:
    """The model reading the whole round (``None``: the call failed)."""
    planner = InterviewPlanner(_NoPlans(), FakeKnowledge([]), reasoner, None,
                               lambda: InterviewSection())
    wiring = InterviewWiring(planner, _NoPlans(), reasoner)
    judge = LiveJudge(plan, wiring, f"set-{round_['id']}", lambda m: None)
    lines = [RoundLine(line["id"], line["role"], line["text"])
             for line in round_["lines"]]
    found, _ = judge.attribute_round(lines)
    if found is None:
        return None
    pairs = {(u, sid) for sid, slot in found.items() for u in slot.answers}
    concepts: Concepts = {(sid, pid): p.status
                          for sid, slot in found.items()
                          for pid, p in slot.points.items()}
    return pairs, concepts


class _NoPlans:
    """A plan store that keeps nothing (the measurement stores nothing)."""

    def save(self, plan: InterviewPlan) -> None:
        """Keep nothing."""

    def load(self, plan_id: str) -> InterviewPlan:
        """No plan."""
        raise KeyError(plan_id)

    def list(self) -> List[InterviewPlan]:
        """No plans."""
        return []

    def save_result(self, plan_id: str, sheet: Document) -> None:
        """Keep nothing."""

    def results(self, plan_id: str) -> List[Document]:
        """No results."""
        return []

    def result(self, plan_id: str, session_id: str) -> Document:
        """No result."""
        raise KeyError(session_id)

    def delete_result(self, plan_id: str, session_id: str) -> bool:
        """Nothing to delete."""
        return False


def ratio(hits: int, total: int) -> Optional[float]:
    """``hits / total`` to three decimals (``None`` when nothing counts)."""
    return round(hits / total, 3) if total else None


def score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Precision and recall of pairs and treated concepts, summed."""
    out: Dict[str, Any] = {}
    for key in ("pairs", "treated"):
        tp = sum(r[key]["tp"] for r in rows)
        got = sum(r[key]["got"] for r in rows)
        want = sum(r[key]["want"] for r in rows)
        out[key] = {"tp": tp, "got": got, "want": want,
                    "precision": ratio(tp, got), "recall": ratio(tp, want)}
    out["unknown"] = {k: sum(r["unknown"][k] for r in rows)
                      for k in ("tp", "got", "want")}
    out["injection_line_used"] = sum(r["injection_line_used"] for r in rows)
    out["failed_calls"] = sum(r.get("failed", 0) for r in rows)
    return out


def compare(round_: Dict[str, Any], got: Optional[Tuple[Pairs, Concepts]]
            ) -> Dict[str, Any]:
    """One round's counts against the writer's intent."""
    pairs, concepts = expected(round_)
    if got is None:
        empty = {"tp": 0, "got": 0, "want": 0}
        return {"pairs": {**empty, "want": len(pairs)},
                "treated": {**empty, "want": sum(
                    1 for s in concepts.values() if s == "treated")},
                "unknown": {**empty, "want": sum(
                    1 for s in concepts.values() if s == "unknown")},
                "injection_line_used": 0, "failed": 1}
    found_pairs, found = got
    treated = {k for k, s in concepts.items() if s == "treated"}
    unknown = {k for k, s in concepts.items() if s == "unknown"}
    ticked = {k for k, s in found.items() if s == "treated"}
    said_unknown = {k for k, s in found.items() if s == "unknown"}
    injection = {line["id"] for line in round_["lines"]
                 if "instrucciones" in line["text"].lower()}
    return {
        "pairs": {"tp": len(found_pairs & pairs), "got": len(found_pairs),
                  "want": len(pairs)},
        "treated": {"tp": len(ticked & treated), "got": len(ticked),
                    "want": len(treated)},
        "unknown": {"tp": len(said_unknown & unknown),
                    "got": len(said_unknown), "want": len(unknown)},
        "injection_line_used": sum(1 for u, _ in found_pairs
                                   if u in injection),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the three ways on the set and write the counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--no-live", action="store_true",
                        help="skip the suggester calls (``live``)")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    reasoner: Any = Fake() if args.fake else Counting(args.timeout)
    plan, rounds, note = load()
    started = datetime.now().isoformat(timespec="seconds")
    rows: Dict[str, List[Dict[str, Any]]] = {"rules": [], "live": [],
                                             "after": []}
    for round_ in rounds:
        rows["rules"].append({"round": round_["id"], **compare(
            round_, rules_way(plan, round_))})
        if not args.no_live:
            rows["live"].append({"round": round_["id"], **compare(
                round_, live_way(plan, round_, reasoner))})
        for run in range(1, args.repeats + 1):
            rows["after"].append({"round": round_["id"], "run": run,
                                  **compare(round_, after_way(
                                      plan, round_, reasoner))})
        print(f"round {round_['id']}: {len(reasoner.calls)} calls",
              flush=True)
    result: Dict[str, Any] = {
        "schema": "attribution-run.v1", "synthetic": True,
        "model": "fake" if args.fake else "haiku (claude -p)",
        "started": started, "set": SET.name, "note": note,
        "repeats_after": args.repeats,
        "summary": {way: score(found) for way, found in rows.items()
                    if found},
        "rows": rows, "calls": reasoner.calls,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1)
                        + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
