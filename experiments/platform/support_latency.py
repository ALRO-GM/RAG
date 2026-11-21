"""Latency of the live suggester variants through the warm backend.

Runs real ``claude`` calls (a budget is passed in ``--calls``): one
process per variant, primed, then one call per trigger utterance of the
support transcript. Retrieval is approximated by word overlap with the
seed domain's chunks (latency depends on prompt size, not on which
chunks). Prints and writes one JSON line per call.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

from src.adapters.outbound.claude_code_stream import ClaudeCodeStream
from src.domain.errors import ModelError
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning import templates as T

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT = ROOT / "examples/platform/sim-soporte-lumbre.transcript.v1.json"
CHUNKS = ROOT / "knowledge/domains/support.demo/fragmentos.jsonl"
TRIGGERS = ["t5.s0", "t7.s0", "t11.s0"]


def words(text: str) -> Set[str]:
    """Lower-case words of at least 4 letters."""
    return set(re.findall(r"\w{4,}", text.lower()))


def request(
    trigger: str, template: str, max_suggestions: int
) -> ReasonRequest:
    """Build the request the orchestrator would send for ``trigger``."""
    data = json.loads(TRANSCRIPT.read_text())
    utts = data["utterances"]
    index = next(i for i, u in enumerate(utts) if u["id"] == trigger)
    window = utts[max(0, index - 11):index + 1]
    chunks = [json.loads(x) for x in CHUNKS.read_text().splitlines() if x]
    said = words(utts[index]["text"])
    chunks.sort(key=lambda c: -len(said & words(c["text"])))
    roles = {"agente": "agent", "cliente": "customer"}
    return ReasonRequest.model_validate({
        "schema": "reasoner.request.v1", "request_id": trigger,
        "role": "suggester", "model": "haiku", "language": "es",
        "template": template,
        "context": {
            "session": {"id": "s", "type": data["session"]["type"],
                        "goal": data["session"]["goal"]},
            "speaker_roles": roles,
            "window": [{"id": u["id"], "speaker": u["speaker"],
                        "text": u["text"]} for u in window],
            "focus": trigger,
        },
        "retrieved": [{"id": "support.demo:" + c["id"], "text": c["text"]}
                      for c in chunks[:4]],
        "constraints": {"max_suggestions": max_suggestions,
                        "deadline_ms": 1800},
    })


def main() -> None:
    """Measure one variant."""
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--triggers", default=",".join(TRIGGERS))
    args = ap.parse_args()
    variants: Dict[str, Any] = VARIANTS
    spec = variants[args.variant]
    system_of: Callable[[ReasonRequest], str] = spec["system"]
    user_of: Callable[[ReasonRequest], str] = spec["user"]
    cap: int = spec["cap"]
    backend = ClaudeCodeStream("haiku", timeout=15.0, pool_size=1,
                               effort=args.effort)
    rows: List[Dict[str, Any]] = []
    for trig in args.triggers.split(","):
        req = request(trig, spec["template"], spec["n"])
        system = system_of(req)
        if not rows:
            backend._start((system, cap), prime=True)  # 1 priming call
        try:
            text = backend.complete_within(system, user_of(req), cap, 15.0)
        except ModelError as exc:
            text = f"ERROR {exc}"
        lat = backend.last_latency
        parsed = T.get_template("suggester.support@1", "suggester").parse(
            text, req) if not text.startswith("ERROR") else None
        row = {"variant": args.variant, "effort": args.effort, "trigger": trig,
               "first_token_ms": lat.first_token_ms if lat else None,
               "total_ms": lat.total_ms if lat else None,
               "output_tokens": (backend.last_usage.output_tokens
                                 if backend.last_usage else None),
               "suggestions": len(parsed.suggestions) if parsed else 0,
               "reply": text}
        print(json.dumps(row, ensure_ascii=False))
        rows.append(row)
    with open(args.out, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    backend.close()


SHORT_SYSTEM = (
    "You assist a customer-support agent on a live call. Reply in Spanish "
    "with ONE short line for the agent to say next (under 20 words), "
    "grounded in the knowledge chunks; invent no policies or deadlines."
)


def _line_user(request: ReasonRequest) -> str:
    return (f"Knowledge:\n{T.chunks_text(request)}\n\nConversation:\n"
            f"{T.window_text(request)}\n\nSay:")


VARIANTS: Dict[str, Any] = {
    "support": {"template": "suggester.support@1", "n": 3,
                "system": T._support_system, "user": T._suggester_user,
                "cap": T.SUPPORT_MAX_OUTPUT_TOKENS},
    "interview": {"template": "suggester.interview@1", "n": 3,
                  "system": T._interview_system, "user": T._suggester_user,
                  "cap": T.DEFAULT_MAX_OUTPUT_TOKENS},
    "one": {"template": "suggester.support@1", "n": 1,
            "system": T._support_system, "user": T._suggester_user,
            "cap": T.SUPPORT_MAX_OUTPUT_TOKENS},
    "line": {"template": "suggester.support@1", "n": 1,
             "system": lambda r: SHORT_SYSTEM, "user": _line_user,
             "cap": 60},
}

if __name__ == "__main__":
    main()
    sys.exit(0)
