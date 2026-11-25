"""Interview reports kept in the candidate's file; closing a position (#262).

Synthetic data only: invented plans, candidates and answers.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from orchestrator.adapters.inbound.cli import main
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.application.candidates import CandidateBook
from orchestrator.application.interview import (
    InterviewPlanner,
    InterviewWiring,
    PlanError,
    PositionCloser,
)
from orchestrator.application.live_judgement import (
    JUDGE_BATCH,
    LiveJudge,
    merged_answers,
    rejudge,
)
from orchestrator.application.ports import CancelToken, Document
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.events import Transcript
from orchestrator.domain.interview import (
    Answer,
    InterviewPlan,
    Sheet,
    SheetSlot,
    Slot,
    closed_plan,
    edited_plan,
    reopened_plan,
)
from orchestrator.domain.candidate import Round, new_profile
from tests.unit.orchestrator.test_candidate import Core, MemoryCandidates
from tests.unit.orchestrator.test_interview_usecases import Store

NOW = datetime(2026, 11, 25, 12, 0, 0)
SECRET = "Medí cada consulta con trazas inventadas para la prueba"


def plan(n: int = 2) -> InterviewPlan:
    slots = [Slot(id=f"s{i}", label=f"Bases de datos tema{i}",
                  level="minimo",
                  questions=[f"¿Qué sabes de bases de datos tema{i}?"])
             for i in range(1, n + 1)]
    return InterviewPlan(id="dev", title="Backend", slots=slots)


# ---- the plan's state ----------------------------------------------------

def test_a_stored_plan_without_state_loads_open(tmp_path: Path) -> None:
    store = JsonPlanStore(tmp_path)
    document = plan().to_contract()
    document.pop("state")
    document.pop("closures")
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans" / "dev.json").write_text(json.dumps(document))
    loaded = store.load("dev")
    assert loaded.state == "open" and loaded.closures == []


def test_close_and_reopen_keep_a_record_and_edits_keep_the_state() -> None:
    shut = closed_plan(plan(), NOW)
    assert shut.state == "closed" and shut.closures[0].closed_at == NOW
    assert closed_plan(shut, NOW + timedelta(days=1)) == shut
    edited = edited_plan(shut, {**shut.to_contract(), "state": "open",
                                "closures": []}, NOW)
    assert edited.state == "closed" and len(edited.closures) == 1
    again = reopened_plan(shut, NOW + timedelta(days=2))
    assert again.state == "open"
    assert again.closures[0].reopened_at == NOW + timedelta(days=2)
    assert reopened_plan(again, NOW) == again


def test_closing_deletes_nothing() -> None:
    store = Store()
    store.save(plan())
    store.save_result("dev", {"session_id": "r1", "candidate_id": "ana"})
    closer = PositionCloser(store, now=lambda: NOW)
    assert closer.close("dev")["state"] == "closed"
    assert store.result("dev", "r1")["candidate_id"] == "ana"
    assert closer.reopen("dev")["state"] == "open"
    with pytest.raises(PlanError):
        closer.close("nope")


# ---- reports in the candidate's file --------------------------------------

def book(store: Store, profiles: MemoryCandidates,
         days: int = 30, now: datetime = NOW) -> CandidateBook:
    return CandidateBook(profiles, Core(), lambda p: p in store.plans,
                         days, now=lambda: now, plans=store)


def test_the_file_lists_its_reports_and_deleting_it_deletes_them() -> None:
    store, profiles = Store(), MemoryCandidates()
    store.save(plan())
    ana = new_profile("dev", "Ana Ruiz", [], NOW)
    ana = ana.model_copy(update={"rounds": [
        Round(session_id="old-ai", at="2026-11-01T10:00:00", kind="bot")]})
    profiles.save(ana)
    store.save_result("dev", {"session_id": "r1", "kind": "live",
                              "candidate_id": ana.id, "finished_at": "2"})
    store.save_result("dev", {"session_id": "old-ai", "finished_at": "1"})
    store.save_result("dev", {"session_id": "other", "candidate_id": "x"})
    files = book(store, profiles)
    found = files.get("dev", ana.id)["reports"]
    assert [(r["session_id"], r["kind"]) for r in found] == [
        ("r1", "live"), ("old-ai", "ai")]
    store.fail_delete = True
    with pytest.raises(PlanError) as failed:
        files.delete("dev", ana.id)
    assert failed.value.code == "delete_failed"
    assert profiles.list("dev")  # kept: deleting again finishes it
    store.fail_delete = False
    gone = files.delete("dev", ana.id)
    assert gone["reports_deleted"] == 2
    assert [s["session_id"] for s in store.results("dev")] == ["other"]
    assert profiles.list("dev") == []


def test_a_file_with_reports_never_expires_by_days() -> None:
    store, profiles = Store(), MemoryCandidates()
    store.save(plan())
    old = NOW - timedelta(days=90)
    kept = new_profile("dev", "Ana Ruiz", [], old)
    gone = new_profile("dev", "Luis Gil", [kept.id], old)
    profiles.save(kept)
    profiles.save(gone)
    store.save_result("dev", {"session_id": "r1", "candidate_id": kept.id})
    assert book(store, profiles).purge() == 1
    assert [p.id for p in profiles.list("dev")] == [kept.id]


# ---- the live report, kept in full ------------------------------------------

class Judge(FakeReasoner):
    """Counts report calls; can fail them."""

    def __init__(self, status: str = "ok") -> None:
        super().__init__()
        self.reports = 0
        self.answer_status = status

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        if request.get("template") == "interview.report@2":
            self.reports += 1
            if self.answer_status != "ok":
                self.requests.append(request)
                return {"schema": "reasoner.response.v1",
                        "request_id": request["request_id"],
                        "role": "structured", "status": "timeout",
                        "error": {"kind": "timeout",
                                  "message": "deadline 90000 ms"}}
        return super().reason(request, cancel)


def wiring(store: Store, reasoner: Any) -> InterviewWiring:
    planner = InterviewPlanner(store, FakeKnowledge([]), reasoner, None,
                               lambda: InterviewSection())
    return InterviewWiring(planner, store, reasoner)


def answered(
    n: int, per_slot: int,
) -> Tuple[List[SheetSlot], Dict[str, str]]:
    slots, texts = [], {}
    for i in range(1, n + 1):
        answers = []
        for k in range(per_slot):
            uid = f"t{i}.{k}"
            texts[uid] = f"{SECRET} {i}.{k} " + "palabra " * 30
            answers.append(Answer(utterance=uid, quote=texts[uid][:40],
                                  status="partial"))
        slots.append(SheetSlot(id=f"s{i}", label=f"Bases de datos tema{i}",
                               level="minimo", status="partial",
                               answers=answers))
    return slots, texts


def test_a_live_report_is_kept_in_full_with_the_plan() -> None:
    store, reasoner = Store(), Judge()
    store.save(plan())
    published: List[Document] = []
    judge = LiveJudge(plan(), wiring(store, reasoner), "r1",
                      published.append, candidate="Ana Ruiz",
                      candidate_id="ana-ruiz", now=lambda: NOW)
    slots, texts = answered(2, 2)
    result = judge.judge_slots(slots, texts)
    assert result is not None and result["saved"] is True
    kept = Sheet.model_validate(store.result("dev", "r1"))
    assert kept.kind == "live" and kept.candidate_id == "ana-ruiz"
    assert kept.suitability is None and all(s.score is None
                                            for s in kept.slots)
    assert kept.slots[0].judgement is not None
    assert SECRET in kept.slots[0].judgement.quote  # in full, with words
    assert [m["state"] for m in published] == ["reporting", "ready"]


def test_a_long_round_is_judged_in_batches_within_the_limits() -> None:
    store, reasoner = Store(), Judge()
    store.save(plan(10))
    slots, texts = answered(10, 12)  # 120 utterances of ~250 characters
    judge = LiveJudge(plan(10), wiring(store, reasoner), "r1",
                      lambda m: None)
    result = judge.judge_slots(slots, texts)
    assert result is not None and result["state"] == "ready"
    assert reasoner.reports == -(-10 // JUDGE_BATCH) == 3
    for request in reasoner.requests:
        if request.get("template") != "interview.report@2":
            continue
        payload = request["payload"]["slots"]
        assert len(payload) <= JUDGE_BATCH
        for slot in payload:
            assert len(slot["answers"]) <= 6
            assert all(len(a["text"]) <= 700 for a in slot["answers"])
    parts = merged_answers([(f"u{k}", "x") for k in range(4)],
                           {f"u{k}": "palabra " * 40 for k in range(4)})
    assert [u for u, _ in parts] == ["u0", "u2"]  # joined, ≤ 700 each


def test_a_failed_judgement_is_logged_without_words_and_kept_nowhere(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, reasoner = Store(), Judge(status="timeout")
    store.save(plan())
    published: List[Document] = []
    judge = LiveJudge(plan(), wiring(store, reasoner), "r1",
                      published.append)
    slots, texts = answered(2, 2)
    with caplog.at_level(logging.WARNING):
        result = judge.judge_slots(slots, texts)
    assert result == {"state": "failed", "judge": "interview.report@2",
                      "saved": False}
    assert store.results("dev") == []
    logged = caplog.text
    assert "status timeout" in logged and "deadline 90000 ms" in logged
    assert "data NoneType" in logged and SECRET not in logged


# ---- judging a round again from its transcript ------------------------------

def transcript(session: str = "r9") -> Dict[str, Any]:
    lines = [("S1", "¿Qué sabes de bases de datos tema1, por ejemplo?"),
             ("S2", f"{SECRET}: índices, planes y caché."),
             ("S1", "¿Y de bases de datos tema2?"),
             ("S2", "Normalizo las tablas y reviso las claves foráneas.")]
    return {
        "schema": "transcript.v1",
        "session": {"id": session, "language": "es",
                    "started_at": "2026-11-25T10:00:00Z"},
        "speakers": [{"id": "S1", "role": "interviewer"},
                     {"id": "S2", "role": "candidate"}],
        "utterances": [
            {"id": f"t{n}.s0", "turn": n, "index": 0, "speaker": who,
             "start": float(n), "end": n + 0.9, "text": text}
            for n, (who, text) in enumerate(lines)],
    }


def test_a_round_is_judged_again_from_its_transcript() -> None:
    store = Store()
    store.save(closed_plan(plan(), NOW))  # a closed position too
    result = rejudge(Transcript.model_validate(transcript()), plan(),
                     wiring(store, Judge()), candidate="Ana Ruiz",
                     candidate_id="ana-ruiz", now=lambda: NOW)
    assert result is not None and result["saved"] is True
    kept = store.result("dev", "r9")
    assert kept["kind"] == "live" and len(kept["slots"]) == 2
    assert kept["candidate_id"] == "ana-ruiz"


def test_the_rejudge_command_prints_numbers_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    plans = JsonPlanStore(tmp_path / "interview")
    plans.save(plan())
    profiles = JsonCandidateStore(tmp_path / "interview")
    ana = new_profile("dev", "Ana Ruiz", [], NOW)
    profiles.save(ana)
    path = tmp_path / "r9.transcript.v1.json"
    path.write_text(json.dumps(transcript()), "utf-8")
    code = main(["interview", "rejudge", "r9", "--plan", "dev",
                 "--candidate", ana.id, "--transcript", str(path),
                 "--interview-dir", str(tmp_path / "interview"),
                 "--reasoner", "fake", "--knowledge", "fake",
                 "--rag-token-file", str(tmp_path / "none")])
    out = capsys.readouterr().out
    assert code == 0
    printed = json.loads(out)
    assert printed["state"] == "ready" and printed["saved"] is True
    assert printed["slots"] == 2 and printed["plan_state"] == "open"
    assert SECRET not in out and "Ana" not in out
    kept = plans.result("dev", "r9")
    assert kept["candidate"] == "Ana Ruiz" and kept["kind"] == "live"
    assert (tmp_path / "interview" / "results" / "dev" / "r9.json"
            ).stat().st_mode & 0o777 == 0o600
    missing = main(["interview", "rejudge", "r9", "--plan", "dev",
                    "--candidate", "nobody", "--transcript", str(path),
                    "--interview-dir", str(tmp_path / "interview"),
                    "--reasoner", "fake", "--knowledge", "fake",
                    "--rag-token-file", str(tmp_path / "none")])
    assert missing == 2
