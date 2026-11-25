"""Small measurement of the answer judge (#259), on an invented set.

``answer_accuracy_set.json`` holds 36 invented answers (3 jobs × 3
questions × 4 labels: solid, correct_basic, wrong, off_topic), each
written to show one label. Each job is one "interview" of 12 slots (one
answer each, in a fixed shuffled order) judged by ``interview.report@2``
exactly as the product sends and reads it (``report_request`` and
``judge_slots``); the run is repeated ``--repeats`` times. The result
says, per expected label, how often the judge gave it, the confusion
table and how stable the label was across repeats.

``--smoke`` also tries the #258 templates once on the real model: the
planner (``interview.planner@2``) on one invented offer and the slot
filler (``interview.slots@2``) on six invented answers.

The real model is Haiku through ``claude -p`` (the owner's
subscription, one cold process per call; never a paid API key).
``--fake`` runs the same path on the fake reasoner (no model)::

    python -m experiments.interview.answer_accuracy --out R.json --smoke

Synthetic: the set was written by a model for this purpose; agreement
with the writer's intent is not accuracy on real interviews.
"""

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.application.interview import (
    InterviewPlanner,
    ReasonerFiller,
    report_request,
)
from orchestrator.application.ports import CancelToken, Document
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.interview import (
    Answer,
    InterviewPlan,
    Point,
    SheetSlot,
    Slot,
)
from orchestrator.domain.interview_judgement import judge_slots
from orchestrator.domain.interview_points import quote

HERE = Path(__file__).resolve().parent
SET = HERE / "answer_accuracy_set.json"
LABELS = ("solid", "correct_basic", "wrong", "off_topic")
TEMPLATE = "interview.report@2"
SEED = 259
OFFER = """Dependiente/a para tienda de running (Valencia, jornada completa)
Funciones: asesorar a corredores en la elección de zapatillas y material,
caja y TPV, reposición y orden de la tienda.
Requisitos: experiencia de un año en atención al cliente en comercio;
conocimiento de zapatillas de running (pisada, amortiguación, tallas);
disponibilidad para turnos rotativos incluidos sábados.
Se valora: inglés básico para atender a turistas.
"""
FILL_ANSWERS: Tuple[Tuple[str, List[str]], ...] = (
    ("Miro la pisada en la cinta y si es pronadora busco estabilidad.",
     ["p1"]),
    ("El drop es la diferencia entre talón y puntera; la amortiguación "
     "depende de la espuma y del peso del corredor.", ["p2"]),
    ("Dejo un dedo de margen delante y que se la pruebe con su calcetín.",
     ["p3"]),
    ("La pisada sí la sé ver, pero lo del drop no lo conozco.",
     ["p1", "p2"]),
    ("Me gusta mucho el ambiente de las tiendas de deporte.", []),
    ("Pues eso, lo normal, ayudar al cliente.", []),
)


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
        """Ask the real reasoner and keep what happened."""
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


def items(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The job's 12 answers in a fixed shuffled order."""
    found = [
        {"job": job["id"], "question": q["id"], "label": q["label"],
         "text": q["question"], "expected": label,
         "answer": q["answers"][label]}
        for q in job["questions"] for label in LABELS
    ]
    random.Random(f"{SEED}-{job['id']}").shuffle(found)
    return found


def judge_job(reasoner: Any, job: Dict[str, Any], run: int
              ) -> List[Dict[str, Any]]:
    """One "interview" of the job, judged as the product does."""
    rows = items(job)
    slots = [Slot(id=f"s{i}", label=r["label"], level="minimo",
                  questions=[r["text"]], evidence=r["text"])
             for i, r in enumerate(rows, start=1)]
    plan = InterviewPlan(id=f"set-{job['id']}", title=job["title"],
                         summary=job["summary"], slots=slots)
    texts = {f"u{i}": r["answer"] for i, r in enumerate(rows, start=1)}
    sheet = [SheetSlot(id=s.id, label=s.label, level="minimo",
                       status="filled",
                       answers=[Answer(utterance=f"u{i}",
                                       quote=quote(texts[f"u{i}"]),
                                       status="complete")])
             for i, s in enumerate(slots, start=1)]
    request = report_request(plan, sheet, texts, {}, "", TEMPLATE, "haiku",
                             f"acc-{job['id']}-{run}")
    try:
        response = reasoner.reason(request, CancelToken())
    except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
        response = {"status": "error", "error": {"message": str(exc)}}
    data = response.get("data") if response.get("status") == "ok" else None
    judged, checks = judge_slots(data or {}, sheet, {}, texts)
    out = []
    for row, slot in zip(rows, judged):
        j = slot.judgement
        out.append({**{k: row[k] for k in ("job", "question", "expected")},
                    "run": run, "slot": slot.id,
                    "got": j.label if j else None,
                    "reason": j.reason if j else "",
                    "checked": any(c.slot == slot.id for c in checks)})
    return out


def summary(rows: List[Dict[str, Any]], repeats: int) -> Dict[str, Any]:
    """Agreement per expected label, confusion and stability."""
    per: Dict[str, Any] = {}
    for label in LABELS:
        mine = [r for r in rows if r["expected"] == label]
        hits = sum(1 for r in mine if r["got"] == label)
        per[label] = {"n": len(mine), "agree": hits,
                      "rate": round(hits / len(mine), 3) if mine else None,
                      "got": dict(Counter(str(r["got"]) for r in mine))}
    stable: Counter[str] = Counter()
    for key in {(r["job"], r["question"], r["expected"]) for r in rows}:
        got = {r["got"] for r in rows
               if (r["job"], r["question"], r["expected"]) == key}
        stable["same" if len(got) == 1 else "varies"] += 1
    total = sum(1 for r in rows if r["got"] == r["expected"])
    return {"answers": len(rows) // max(1, repeats), "repeats": repeats,
            "judgements": len(rows), "agree": total,
            "rate": round(total / len(rows), 3) if rows else None,
            "per_label": per, "stability": dict(stable),
            "wrong_listed_to_check": sum(
                1 for r in rows if r["expected"] == "wrong" and r["checked"])}


def smoke(reasoner: Any) -> Dict[str, Any]:
    """The #258 templates once on the model: planner and slot filler."""

    class Plans:
        def __init__(self) -> None:
            self.plans: Dict[str, InterviewPlan] = {}

        def save(self, plan: InterviewPlan) -> None:
            self.plans[plan.id] = plan

        def load(self, plan_id: str) -> InterviewPlan:
            return self.plans[plan_id]

        def list(self) -> List[InterviewPlan]:
            return list(self.plans.values())

        def save_result(self, plan_id: str, sheet: Document) -> None:
            pass

        def results(self, plan_id: str) -> List[Document]:
            return []

        def result(self, plan_id: str, session_id: str) -> Document:
            raise KeyError(session_id)

        def delete_result(self, plan_id: str, session_id: str) -> bool:
            return False

    planner = InterviewPlanner(
        Plans(), FakeKnowledge([]), reasoner, None,
        lambda: InterviewSection(), new_id=lambda: "smoke-plan",
        now=lambda: datetime(2026, 11, 25, 12, 0, 0))
    plan = planner.create({"offer": OFFER, "minutes": 20})
    target = Slot(
        id="s1", label="Conocimiento de zapatillas de running",
        level="minimo", questions=["¿Qué sabes de zapatillas de running?"],
        points=[Point(id="p1", text="Tipos de pisada"),
                Point(id="p2", text="Amortiguación y drop"),
                Point(id="p3", text="Cómo elegir la talla")])
    filler = ReasonerFiller(reasoner)
    fills = []
    for answer, expected in FILL_ANSWERS:
        found = filler.fill(answer, target, [target], target.questions[0],
                            CancelToken())
        fills.append({
            "answer": answer, "expected_points": expected,
            "fills": [{"slot": f.slot, "status": f.status, "quote": f.quote,
                       "points": [{"id": h.point, "status": h.status,
                                   "quote": h.quote} for h in f.points]}
                      for f in found]})
    return {
        "planner": {"status": plan["planner"]["status"],
                    "error": plan["planner"].get("error"),
                    "slots": [{"label": s["label"], "level": s["level"],
                               "kind": s["kind"],
                               "points": [p["text"] for p in s["points"]]}
                              for s in plan["slots"]]},
        "filler": {"fallbacks": filler.fallbacks, "answers": fills},
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the set (and the smoke) and write the result."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    reasoner: Any = Fake() if args.fake else Counting(args.timeout)
    data = json.loads(SET.read_text(encoding="utf-8"))
    started = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, Any]] = []
    for run in range(1, args.repeats + 1):
        for job in data["jobs"]:
            rows.extend(judge_job(reasoner, job, run))
            print(f"run {run} {job['id']}: {len(reasoner.calls)} calls",
                  flush=True)
    result: Dict[str, Any] = {
        "schema": "answer-accuracy-run.v1", "template": TEMPLATE,
        "model": "fake" if args.fake else "haiku (claude -p)",
        "started": started, "set": SET.name, "note": data["note"],
        "summary": summary(rows, args.repeats), "rows": rows,
    }
    if args.smoke:
        result["smoke"] = smoke(reasoner)
    result["calls"] = reasoner.calls
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1)
                        + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
