"""Smoke test of the AI interviewer with real Haiku calls (#208).

Not a measurement: a check that the four templates work end to end on
the real model (``claude -p``, one cold process per call, so the times
are not the live bot's). It drafts the plans of two synthetic job
offers (shop assistant, backend developer) with ``interview.planner@1``
and runs a short text interview for the shop assistant: the opening,
two scripted candidate answers (each through ``interview.slots@1`` and
``interviewer@1``) and the report (``interview.report@1``). Eight Haiku
calls in all; the knowledge port reads the method corpus of
``interview.generic`` with a word-overlap fake (no index needed)::

    python experiments/voicebot/interviewer_smoke.py --out R.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.adapters.outbound.fakes import FakeKnowledge
from orchestrator.application.interview import (
    InterviewDirector,
    InterviewPlanner,
    ReasonerFiller,
)
from orchestrator.application.ports import CancelToken, Document
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.interview import InterviewPlan
from orchestrator.domain.records import TriggerInfo, TurnRecord
from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
from src.application.reasoning import Reasoner
from src.domain.reasoning.roles import ReasonRequest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "knowledge" / "domains" / "interview.generic"
SHOP = """Dependiente/a para tienda de deportes (Valencia, jornada completa)
Funciones: atención al cliente en sala, caja y TPV, reposición y orden
de la tienda, recepción de mercancía.
Requisitos: experiencia mínima de un año en atención al cliente en
comercio; disponibilidad para turnos rotativos de mañana y tarde,
incluidos sábados; manejo de caja.
Se valora: conocimiento de material de running y de montaña; inglés
básico para atender a turistas.
"""
BACKEND = """Desarrollador/a backend junior (Python)
Equipo de 6 personas que mantiene la API de pedidos de una tienda online.
Requisitos: Python y un framework web (FastAPI, Django o Flask); SQL
(PostgreSQL); control de versiones con git; tests automáticos.
Se valora: Docker, colas de mensajes, experiencia con APIs REST en
producción, inglés técnico leído.
Ofrecemos: formación, mentoría y trabajo híbrido.
"""
ANSWERS = (
    "Estuve un año y medio en una tienda de deportes del centro, en sala y "
    "en caja; atendía a clientes, cobraba con TPV y hacía los cierres.",
    "Pues sí, los sábados puedo, y turnos de mañana o de tarde también, me "
    "adapto sin problema.",
)


class Counting:
    """The real reasoner in process (``claude -p``), counting calls.

    The same glue as ``experiments/live/orchestrator_reasoner.py``.
    """

    def __init__(self, timeout: float) -> None:
        """Build the reasoner over Claude Code (one process per call)."""
        self.inner = Reasoner(lambda alias: ClaudeCodeJudge(
            alias or "haiku", timeout=timeout, retries=0))
        self.calls: List[Dict[str, Any]] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Ask the real reasoner and keep what happened."""
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


class Output:
    """The feed, kept in a list."""

    def __init__(self) -> None:
        """Start empty."""
        self.messages: List[Document] = []

    def publish(self, message: Document) -> None:
        """Keep a message."""
        self.messages.append(message)

    def record(self, record: TurnRecord) -> None:
        """Turn records are not used here."""


class Store:
    """Plans and sheets in memory."""

    def __init__(self) -> None:
        """Start empty."""
        self.plans: Dict[str, InterviewPlan] = {}
        self.sheets: List[Document] = []

    def save(self, plan: InterviewPlan) -> None:
        """Keep a plan."""
        self.plans[plan.id] = plan

    def load(self, plan_id: str) -> InterviewPlan:
        """Return a plan."""
        return self.plans[plan_id]

    def list(self) -> List[InterviewPlan]:
        """Every plan."""
        return list(self.plans.values())

    def save_result(self, plan_id: str, sheet: Document) -> None:
        """Keep a sheet."""
        self.sheets.append(sheet)

    def results(self, plan_id: str) -> List[Document]:
        """Every sheet."""
        return list(self.sheets)

    def result(self, plan_id: str, session_id: str) -> Document:
        raise KeyError(session_id)

    def delete_result(self, plan_id: str, session_id: str) -> bool:
        return False


class Clock:
    """Wall time."""

    def now(self) -> float:
        """Monotonic seconds."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Wait."""
        time.sleep(seconds)


def corpus() -> FakeKnowledge:
    """The method corpus behind a word-overlap fake."""
    chunks = []
    for path in sorted(CORPUS.glob("*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            chunks.append({"id": row["id"], "text": row["text"]})
    return FakeKnowledge(chunks)


def say(director: InterviewDirector, reasoner: Counting, payload: Document,
        turn: int) -> str:
    """Word one step with ``interviewer@1``; tell the director."""
    request: Document = {
        "schema": "reasoner.request.v1", "request_id": f"smoke-b{turn}",
        "role": "conversational", "model": "haiku",
        "template": "interviewer@1", "language": "es",
        "context": {"window": [
            {"id": f"x{i}", "speaker": s.speaker, "text": s.text}
            for i, s in enumerate(director.said[-8:])
        ]},
        "payload": payload, "constraints": {"deadline_ms": 30000},
    }
    answer = reasoner.reason(request, CancelToken())
    text = str((answer.get("reply") or {}).get("text") or "")
    record = TurnRecord(
        session_id="smoke", turn=turn, request_id=f"smoke-b{turn}",
        trigger=TriggerInfo(utterance="", speaker="BOT", role="bot",
                            stream_time=0.0, text=""),
        status="ok", spoken=[text] if text else [], audio_ms=4000.0)
    director.spoken(record)
    return text


def main() -> None:
    """Run the smoke test and write its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reasoner = Counting(timeout=150.0)
    store = Store()
    ids = iter(["plan-tienda", "plan-backend"])
    planner = InterviewPlanner(store, corpus(), reasoner, None,
                               lambda: InterviewSection(), new_id=lambda:
                               next(ids))
    shop = planner.create({"offer": SHOP, "minutes": 15})
    backend = planner.create({"offer": BACKEND, "minutes": 20})
    section = InterviewSection()
    director = InterviewDirector(
        store.load(shop["id"]), "smoke-tienda", section,
        ReasonerFiller(reasoner), reasoner, Output(), Clock(), store,
        None, False, "Candidata sintética",
    )
    spoken = []
    plan = director.plan_turn("", None, CancelToken())
    spoken.append({"step": plan.payload,
                   "text": say(director, reasoner, plan.payload, 1)})
    fills = []
    for turn, answer in enumerate(ANSWERS, start=2):
        director.heard(f"u{turn}", answer, float(turn * 10),
                       float(turn * 10 + 6))
        before = {s["id"]: s["status"]
                  for s in director.coverage()["slots"]}
        plan = director.plan_turn(answer, f"u{turn}", CancelToken())
        after = {s["id"]: s["status"] for s in director.coverage()["slots"]}
        fills.append({"answer": answer, "changed": {
            k: [before[k], v] for k, v in after.items() if before[k] != v}})
        spoken.append({"step": plan.payload,
                       "text": say(director, reasoner, plan.payload, turn)})
    result = director.finish()
    json.dump({
        "plans": [shop, backend], "turns": spoken, "fills": fills,
        "sheet": (result or {}).get("sheet"),
        "calls": reasoner.calls, "haiku_calls": len(reasoner.calls),
    }, args.out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(reasoner.calls)} calls:",
          [(c["template"], c["status"], c["wall_ms"]) for c in
           reasoner.calls])


if __name__ == "__main__":
    main()
