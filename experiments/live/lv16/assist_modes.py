"""Latency and a short quality check of the two assistance modes (LV-16).

Real Haiku calls through the RAG's own live path: the warm ``claude -p``
stream backend (one process per template, effort ``low``) behind the
:class:`~src.application.reasoning.Reasoner`, so the ``@2`` templates
parse, ground and cut the replies as in production. For each trigger
utterance of the synthetic transcripts in ``examples/platform`` (the
customer's or candidate's sentences), the same request is sent once in
``pincelada`` and once in ``guiado``, with the products' settings
(support: one suggestion, deadline 2.8 s; interview: three, 1.8 s).
Retrieval is approximated by word overlap with the seed domain's chunks,
the window is the 12 previous utterances, as the orchestrator sends.

Every process start costs one priming call; the script counts them and
stops before ``--budget`` real calls (primes included)::

    PYTHONPATH=. python experiments/live/lv16/assist_modes.py \\
        --out experiments/live/lv16/assist_modes.json
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from src.adapters.outbound.claude_code_stream import ClaudeCodeStream
from src.application.reasoning import Reasoner
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning.templates import get_template

ROOT = Path(__file__).resolve().parents[3]
CASES: List[Dict[str, Any]] = [
    {"name": "support", "template": "suggester.support@2",
     "transcript": "sim-soporte-lumbre", "domain": "support.demo",
     "roles": {"agente": "agent", "cliente": "customer"},
     "max_suggestions": 1, "deadline_ms": 2800,
     "triggers": ["t3.s0", "t5.s0", "t7.s0", "t17.s0", "t19.s0", "t21.s0"]},
    {"name": "interview", "template": "suggester.interview@2",
     "transcript": "sim-entrevista-backend", "domain": "interview.generic",
     "roles": {"entrevistador": "interviewer", "candidata": "candidate"},
     "max_suggestions": 3, "deadline_ms": 1800,
     "triggers": ["t1.s0", "t3.s0", "t5.s0", "t9.s0", "t15.s0", "t21.s0"]},
]
MODES = ("pincelada", "guiado")
JUDGING = re.compile(
    r"\b(nota|puntu\w*|valora\w*|nervios\w*|confianza|emoci\w*|estr[eé]s"
    r"|personalidad|excelente|buen[ao]s? respuestas?|inseguri\w*)\b",
    re.IGNORECASE,
)
"""Words of a rating or an emotion about the candidate (a crude flag)."""


def words(text: str) -> Set[str]:
    """Lower-case words of at least 4 letters."""
    return set(re.findall(r"\w{4,}", text.lower()))


def build(case: Dict[str, Any], trigger: str, mode: str) -> ReasonRequest:
    """The request the orchestrator would send for ``trigger``."""
    name = f"{case['transcript']}.transcript.v1.json"
    path = ROOT / "examples" / "platform" / name
    data = json.loads(path.read_text(encoding="utf-8"))
    utts = data["utterances"]
    at = next(i for i, u in enumerate(utts) if u["id"] == trigger)
    window = utts[max(0, at - 11):at + 1]
    folder = ROOT / "knowledge" / "domains" / case["domain"]
    lines = (folder / "fragmentos.jsonl").read_text().splitlines()
    chunks = [json.loads(x) for x in lines if x]
    said = words(utts[at]["text"])
    chunks.sort(key=lambda c: -len(said & words(c["text"])))
    return ReasonRequest.model_validate({
        "request_id": f"{case['name']}-{trigger}-{mode}", "role": "suggester",
        "model": "haiku", "language": "es", "template": case["template"],
        "context": {
            "session": {"id": "lv16", "type": data["session"].get("type")},
            "speaker_roles": case["roles"],
            "window": [{"id": u["id"], "speaker": u["speaker"],
                        "text": u["text"]} for u in window],
            "focus": trigger,
        },
        "retrieved": [
            {"id": f"{case['domain']}:{c['id']}", "text": c["text"]}
            for c in chunks[:4]
        ],
        "constraints": {"max_suggestions": case["max_suggestions"],
                        "deadline_ms": case["deadline_ms"], "mode": mode},
    })


class Counted(ClaudeCodeStream):
    """The warm backend, counting the priming call of every process."""

    primes = 0

    def _prime(self, worker: Any) -> None:
        Counted.primes += 1
        super()._prime(worker)


def wait_warm(backend: Counted, system: str, cap: int) -> None:
    """Wait until a primed process for ``(system, cap)`` is in the pool."""
    for _ in range(300):
        with backend._pool_lock:
            if any(w.key == (system, cap) for w in backend._workers):
                return
        time.sleep(0.1)


def main() -> None:
    """Run the calls and summarise them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    backend = Counted("haiku", effort="low", pool_size=1, timeout=20.0)
    reasoner = Reasoner(lambda alias: backend, live_models=lambda a: backend)
    rows: List[Dict[str, Any]] = []
    calls = 0
    for case in CASES:
        probe = build(case, case["triggers"][0], "pincelada")
        template = get_template(case["template"], "suggester")
        reasoner.warm_up(case["template"], "haiku", "es",
                         case["max_suggestions"])
        wait_warm(backend, template.system(probe),
                  template.max_output_tokens)
        for trigger in case["triggers"]:
            for mode in MODES:
                if calls + Counted.primes >= args.budget:
                    break
                request = build(case, trigger, mode)
                started = time.perf_counter()
                answer = reasoner.reason(request)
                wall = (time.perf_counter() - started) * 1000.0
                calls += 1
                texts = [s.text for s in answer.suggestions]
                rows.append({
                    "case": case["name"], "trigger": trigger, "mode": mode,
                    "status": answer.status, "wall_ms": round(wall, 1),
                    "first_token_ms": answer.latency.first_token_ms,
                    "total_ms": answer.latency.total_ms,
                    "output_tokens": answer.usage.output_tokens,
                    "texts": texts,
                    "words": [len(t.rstrip("…").split()) for t in texts],
                    "cut": [t.endswith("…") for t in texts],
                    "signals": [s.signals for s in answer.suggestions],
                    "judging_words": [bool(JUDGING.search(t))
                                      for t in texts]
                    if case["name"] == "interview" else [],
                })
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    backend.close()
    summary: Dict[str, Any] = {}
    for case in CASES:
        for mode in MODES:
            got = [r for r in rows
                   if r["case"] == case["name"] and r["mode"] == mode]
            if not got:
                continue
            ok = [r for r in got if r["status"] == "ok"]
            totals = [r["total_ms"] for r in got if r["status"] in
                      ("ok", "no_suggestion")]
            firsts = [r["first_token_ms"] for r in got
                      if r["first_token_ms"] is not None]
            all_words = [w for r in ok for w in r["words"]]
            summary[f"{case['name']}/{mode}"] = {
                "calls": len(got),
                "status": {s: sum(1 for r in got if r["status"] == s)
                           for s in sorted({r["status"] for r in got})},
                "within_deadline": sum(
                    1 for r in got if r["status"] in ("ok", "no_suggestion")
                    and r["total_ms"] <= case["deadline_ms"]),
                "deadline_ms": case["deadline_ms"],
                "total_ms_p50": round(statistics.median(totals), 1)
                if totals else None,
                "total_ms_max": round(max(totals), 1) if totals else None,
                "first_token_ms_p50": round(statistics.median(firsts), 1)
                if firsts else None,
                "words_median": statistics.median(all_words)
                if all_words else None,
                "words_max": max(all_words) if all_words else None,
                "cut_to_8_words": sum(c for r in ok for c in r["cut"]),
                "suggestions": len(all_words),
                "judging_words": sum(j for r in ok
                                     for j in r["judging_words"]),
            }
    report = {"model": "haiku", "effort": "low", "calls": calls,
              "priming_calls": Counted.primes,
              "real_calls_total": calls + Counted.primes,
              "summary": summary, "rows": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                        + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
