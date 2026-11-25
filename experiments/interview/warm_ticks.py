"""``suggester.interview@2`` vs ``@3`` on the warm live path (#268 review).

The live page's suggestion call runs on the RAG's warm Claude Code
processes (:class:`~src.adapters.outbound.claude_code_stream.ClaudeCodeStream`,
Haiku, effort ``low``, the default of ``live.suggester_effort``) with a
1.8 s deadline. ``@3`` adds the pending concepts to the prompt and a
``treated`` list to the reply; this measures what that costs there.

For the first ``--n`` candidate finals of the invented set
(``attribution_set.json``), each template is asked once, sequentially,
with the same request the live page would send (last six lines, the
speakers' roles, the pending concepts of :meth:`TopicTracker.focus_points`;
no retrieved chunks, for both). Each template has its own warm
backend (as in the service, where one product uses one template), is
warmed first and gets one call that is not counted. The calls get the
warm backend's own 10 s instead of 1.8 s, so the full reply time is
seen; the share within 1.8 s is computed from it. A reply longer than
the process cap (template cap + 300 thinking tokens) fails as
``invalid_output`` and counts as over the cap.

``suggester.interview@3`` was removed after this measurement (the
decision is in the design doc §6.3); it ran against the templates of
commit 810f51b, which this script needs to run again. Text only, the
owner's subscription (``claude`` CLI), in process; never a paid API
key. Synthetic set: the numbers say what the prompt costs on
the warm path, not how it does on real interviews::

    python -m experiments.interview.warm_ticks --out R.json --n 15
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.application.next_round import TopicTracker
from orchestrator.application.replay import transcript_to_events
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview import InterviewPlan
from orchestrator.domain.next_round import topic_states

SET = Path(__file__).resolve().parent / "attribution_set.json"
SPEAKERS = {"interviewer": "S1", "candidate": "S2"}


def load() -> Tuple[InterviewPlan, List[Dict[str, Any]]]:
    """The set's plan and rounds."""
    data = json.loads(SET.read_text(encoding="utf-8"))
    return InterviewPlan.model_validate(data["plan"]), data["rounds"]


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


DEADLINE_MS = 1800.0
TEMPLATES = ("suggester.interview@2", "suggester.interview@3")
WINDOW = 6


def requests(n: int) -> List[Dict[str, Any]]:
    """The first ``n`` live requests of the set (one per candidate final)."""
    plan, rounds = load()
    roles = {"S1": "interviewer", "S2": "candidate"}
    found: List[Dict[str, Any]] = []
    for round_ in rounds:
        topics = TopicTracker(plan, topic_states(plan, []), roles,
                              lambda message: None)
        window: List[Dict[str, Any]] = []
        for event in transcript_to_events(transcript(round_), roles):
            topics.on_event(event)
            u = event.utterance
            if event.type != "final" or u is None:
                continue
            window = (window + [{"id": u.id, "speaker": u.speaker,
                                 "text": u.text}])[-WINDOW:]
            if roles[u.speaker] != "candidate":
                continue
            found.append({
                "request_id": f"warm-{u.id}", "role": "suggester",
                "model": "haiku", "language": "es",
                "context": {"session": {"id": f"set-{round_['id']}",
                                        "type": "entrevista"},
                            "speaker_roles": roles, "window": list(window),
                            "focus": u.id},
                "retrieved": [], "payload": topics.focus_points(),
                "constraints": {"max_suggestions": 1, "mode": "pincelada",
                                "deadline_ms": 10000},
            })
    return found[:n]


def run_template(template: str, asked: List[Dict[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """Ask one template on its own warm backend; keep numbers only."""
    from src.adapters.outbound.claude_code_stream import ClaudeCodeStream
    from src.application.reasoning import Reasoner
    from src.domain.reasoning.roles import ReasonRequest
    from src.domain.reasoning.templates import get_template
    from src.domain.skills.claude_models import check_claude_model

    stream = ClaudeCodeStream(check_claude_model("haiku"), effort="low")
    reasoner = Reasoner(lambda alias: stream, live_models=lambda a: stream)
    spec = get_template(template, "suggester")
    rows: List[Dict[str, Any]] = []
    try:
        first = ReasonRequest.model_validate({**asked[0],
                                              "template": template})
        stream.warm(spec.system(first), spec.max_output_tokens)
        time.sleep(8.0)
        reasoner.reason(first)  # not counted: the pool settles
        for item in asked:
            request = ReasonRequest.model_validate({**item,
                                                    "template": template})
            started = time.perf_counter()
            answer = reasoner.reason(request).to_contract()
            wall = (time.perf_counter() - started) * 1000.0
            usage = answer.get("usage") or {}
            latency = answer.get("latency") or {}
            error = answer.get("error") or {}
            rows.append({
                "focus": item["context"]["focus"],
                "status": answer.get("status"),
                "error": error.get("kind"),
                "wall_ms": round(wall),
                "total_ms": latency.get("total_ms"),
                "first_token_ms": latency.get("first_token_ms"),
                "output_tokens": usage.get("output_tokens"),
                "suggestions": len(answer.get("suggestions") or []),
                "ticks": len(((answer.get("data") or {}).get("points"))
                             or []),
            })
            print(template, rows[-1]["status"], rows[-1]["wall_ms"],
                  rows[-1]["output_tokens"], flush=True)
    finally:
        stream.close()
    return rows


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shares and spreads of one template's calls."""
    done = [r for r in rows if r["status"] in ("ok", "no_suggestion")]
    times = [r["wall_ms"] for r in done]
    tokens = [r["output_tokens"] for r in done
              if isinstance(r["output_tokens"], int)]

    def spread(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        ordered = sorted(values)
        return {"min": ordered[0], "median": statistics.median(ordered),
                "p90": ordered[max(0, round(0.9 * len(ordered)) - 1)],
                "max": ordered[-1]}

    return {
        "calls": len(rows),
        "answered": len(done),
        "within_1800ms": sum(1 for t in times if t <= DEADLINE_MS),
        "over_cap": sum(1 for r in rows if r["error"] == "invalid_output"),
        "other_failures": len(rows) - len(done) - sum(
            1 for r in rows if r["error"] == "invalid_output"),
        "with_suggestion": sum(1 for r in done if r["suggestions"]),
        "with_ticks": sum(1 for r in done if r["ticks"]),
        "wall_ms": spread([float(t) for t in times]),
        "output_tokens": spread([float(t) for t in tokens]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run both templates and write the numbers (no words)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args(argv)
    asked = requests(args.n)
    started = datetime.now().isoformat(timespec="seconds")
    rows = {template: run_template(template, asked)
            for template in TEMPLATES}
    result = {
        "schema": "warm-ticks-run.v1", "synthetic": True,
        "backend": "ClaudeCodeStream haiku, effort low (warm, in process)",
        "started": started, "deadline_ms": DEADLINE_MS,
        "uncounted_calls": len(TEMPLATES),
        "summary": {t: summary(r) for t, r in rows.items()},
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1)
                        + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
