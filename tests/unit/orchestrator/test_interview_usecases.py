"""The interview planner and director with fakes (#208)."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.application.interview import (
    HandoffFailed,
    HeuristicFiller,
    InterviewDirector,
    InterviewPlanner,
    OfferUnavailable,
    PlanError,
    ReasonerFiller,
    judge_chunks,
    read_fills,
    read_report,
    status_scores,
)
from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerUnavailable,
)
from orchestrator.domain.config import InterviewSection
from orchestrator.domain.interview import (
    InterviewPlan,
    PlanSource,
    Point,
    SheetSlot,
    Slot,
)
from orchestrator.domain.records import TriggerInfo, TurnRecord

NOW = datetime(2026, 11, 21, 10, 0, 0)
OFFER = (
    "Dependiente/a de tienda de deportes\n"
    "- Experiencia de un año en atención al cliente en tienda\n"
    "- Disponibilidad para turnos rotativos y sábados\n"
    "- Valorable conocimiento de material de running\n"
)


class Store:
    def __init__(self) -> None:
        self.plans: Dict[str, InterviewPlan] = {}
        self.sheets: Dict[str, List[Document]] = {}

    def save(self, plan: InterviewPlan) -> None:
        self.plans[plan.id] = plan

    def load(self, plan_id: str) -> InterviewPlan:
        return self.plans[plan_id]

    def list(self) -> List[InterviewPlan]:
        return list(self.plans.values())

    def save_result(self, plan_id: str, sheet: Document) -> None:
        self.sheets.setdefault(plan_id, []).append(sheet)

    def results(self, plan_id: str) -> List[Document]:
        return list(self.sheets.get(plan_id, []))


class Offers:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def document_text(self, domain: str, document: str) -> Tuple[str, str]:
        if self.fail:
            raise OfferUnavailable("HTTP 409: not ready")
        return "oferta.pdf", OFFER


class Down:
    def reason(self, request: Document, cancel: CancelToken) -> Document:
        raise ReasonerUnavailable("ConnectError")


class Handoff:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.transcripts: List[Document] = []
        self.jobs: List[Tuple[str, str]] = []

    def publish_transcript(self, document: Document) -> None:
        if self.fail:
            raise HandoffFailed("HTTP 422: bad")
        self.transcripts.append(document)

    def start_analysis(self, session_id: str, job: str) -> None:
        self.jobs.append((session_id, job))


class Output:
    def __init__(self) -> None:
        self.messages: List[Document] = []

    def publish(self, message: Document) -> None:
        self.messages.append(message)

    def record(self, record: TurnRecord) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def planner(reasoner: Any = None, offers: Any = None,
            store: Optional[Store] = None) -> InterviewPlanner:
    ids = iter(f"plan-{i}" for i in range(1, 50))
    return InterviewPlanner(
        store or Store(),
        FakeKnowledge([{"id": "m1", "text": "Pregunta STAR de atención"}],
                      everything=True),
        reasoner or FakeReasoner(), offers or Offers(),
        lambda: InterviewSection(minutes=25), now=lambda: NOW,
        new_id=lambda: next(ids),
    )


def test_the_planner_drafts_a_plan_from_pasted_text() -> None:
    reasoner = FakeReasoner()
    made = planner(reasoner).create({"offer": OFFER, "role_type": "comercio"})
    assert made["schema"] == "interview.plan.v1"
    assert made["minutes"] == 25 and made["role_type"] == "comercio"
    assert made["planner"]["status"] == "ok"
    assert made["planner"]["chunks"] == ["m1"]
    labels = [s["label"] for s in made["slots"]]
    assert labels[0].startswith("Experiencia de un año")
    request = reasoner.requests[0]
    assert request["role"] == "structured"
    assert request["template"] == "interview.planner@2"
    assert len(request["payload"]["competencies"]) == 11


def test_the_planner_reads_an_uploaded_offer_and_falls_back() -> None:
    made = planner(Down()).create(
        {"document": {"domain": "ofertas", "id": "abc"}, "minutes": "15",
         "title": "Dependienta"})
    assert made["source"]["kind"] == "document"
    assert made["source"]["name"] == "oferta.pdf"
    assert made["planner"]["status"] == "fallback"
    assert "unavailable" in made["planner"]["error"]
    assert made["title"] == "Dependienta" and made["minutes"] == 15
    with pytest.raises(PlanError) as caught:
        planner(offers=Offers(fail=True)).create(
            {"document": {"domain": "d", "id": "x"}})
    assert caught.value.code == "offer"
    with pytest.raises(PlanError):
        planner().create({"offer": "corto"})
    with pytest.raises(PlanError):
        planner().create({"document": {"domain": "", "id": "x"}})


def test_a_failed_model_answer_also_falls_back() -> None:
    made = planner(FakeReasoner(status="timeout")).create({"offer": OFFER})
    assert made["planner"]["status"] == "fallback"
    assert made["planner"]["error"].startswith("timeout")


def test_edits_lists_and_results() -> None:
    store = Store()
    plans = planner(store=store)
    made = plans.create({"offer": OFFER})
    made["slots"][0]["weight"] = 5
    edited = plans.update(made["id"], made)
    assert edited["revision"] == 1 and edited["slots"][0]["weight"] == 5
    with pytest.raises(PlanError) as caught:
        plans.update(made["id"], made)
    assert caught.value.code == "conflict"
    bad = dict(edited, slots=[])
    with pytest.raises(PlanError) as caught:
        plans.update(made["id"], bad)
    assert caught.value.code == "invalid_request"
    with pytest.raises(PlanError) as caught:
        plans.get("plan-404")
    assert caught.value.code == "not_found"
    store.save_result(made["id"], {"session_id": "a", "suitability": 40.0,
                                   "slots": [{"id": "s1", "score": 40}]})
    store.save_result(made["id"], {"session_id": "b", "suitability": 80.0,
                                   "slots": []})
    store.save_result(made["id"], {"session_id": "c", "suitability": None})
    rows = plans.results(made["id"])["candidates"]
    assert [r["session_id"] for r in rows] == ["b", "a", "c"]
    assert rows[1]["slots"][0]["score"] == 40
    assert plans.list()[0]["interviews"] == 3


def slots() -> List[Slot]:
    return [
        Slot(id="s1", label="Atención al cliente en tienda", level="minimo",
             questions=["¿Dónde has atendido a clientes?"],
             evidence="Dónde y cuánto tiempo"),
        Slot(id="s2", label="Comunicación", level="minimo",
             kind="competency", competency="COMM",
             questions=["Cuéntame una vez..."]),
    ]


def test_read_fills_keeps_known_slots_and_real_quotes() -> None:
    text = "Estuve dos años en una tienda de deportes atendiendo en sala"
    found = read_fills({"slots": [
        {"id": "s1", "status": "complete", "quote": "dos años en una tienda"},
        {"id": "s2", "status": "partial", "quote": "invented words"},
        {"id": "s9", "status": "complete", "quote": "x"},
        {"id": "s1", "status": "great", "quote": "x"},
        "junk",
    ]}, text, {"s1", "s2"})
    assert [(f.slot, f.status) for f in found] == [
        ("s1", "complete"), ("s2", "partial")]
    assert found[0].quote == "dos años en una tienda"
    assert found[1].quote.startswith("Estuve dos años")


def test_the_reasoner_filler_falls_back_to_the_heuristic() -> None:
    answer = "Estuve dos años en caja en un supermercado del barrio"
    filler = ReasonerFiller(FakeReasoner())
    found = filler.fill(answer, slots()[0], slots(), "¿Dónde?",
                        CancelToken())
    assert [(f.slot, f.status) for f in found] == [("s1", "partial")]
    down = ReasonerFiller(Down())
    assert down.fill(answer, slots()[0], slots(), "", CancelToken()) == \
        HeuristicFiller().fill(answer, slots()[0], slots(), "",
                               CancelToken())
    assert down.fallbacks == 1 and down.calls == 1


def test_read_report_merges_scores_and_keeps_unknown_at_zero() -> None:
    sheet = [
        SheetSlot(id="s1", label="a", level="minimo", status="filled"),
        SheetSlot(id="s2", label="b", level="minimo", status="unknown"),
        SheetSlot(id="s3", label="c", level="deseable", status="partial"),
        SheetSlot(id="s4", label="d", level="opcional", status="empty"),
    ]
    merged, others, summary = read_report({
        "slots": [
            {"id": "s1", "score": 130, "above": True, "confidence": 2,
             "notes": "  Más de lo pedido  "},
            {"id": "s2", "score": 60, "above": True},
            {"id": "s3", "score": "alto"},
            {"id": "s4", "score": 0, "confidence": 0.1},
        ],
        "other_skills": [{"label": "Inventario", "quote": "llevaba stock"},
                         {"label": " "}],
        "summary": "Resumen",
    }, sheet)
    assert [(s.score, s.above, s.scored_by) for s in merged] == [
        (100, True, "model"), (0, False, "model"), (40, False, "status"),
        (None, False, None)]
    assert merged[0].confidence == 1.0 and merged[0].notes == \
        "Más de lo pedido"
    assert [o.label for o in others] == ["Inventario"]
    assert summary == "Resumen"
    assert status_scores(sheet)[0].score == 70


def record(text: str, turn: int) -> TurnRecord:
    found = TurnRecord(
        session_id="vb", turn=turn, request_id=f"vb-b{turn}",
        trigger=TriggerInfo(utterance="", speaker="BOT", role="bot",
                            stream_time=0.0, text=""),
        status="ok", spoken=[text], audio_ms=3000.0)
    found.stages.total_ms = 2000.0
    found.stages.first_audio_ms = 900.0
    return found


def director(reasoner: Any = None, handoff: Any = None,
             post_call: bool = True,
             store: Optional[Store] = None) -> InterviewDirector:
    plan = InterviewPlan(id="plan-1", title="Dependiente/a", slots=slots(),
                         summary="Tienda")
    return InterviewDirector(
        plan, "vb-1", InterviewSection(max_probes=1), HeuristicFiller(),
        reasoner or FakeReasoner(), Output(), Clock(), store or Store(),
        handoff if handoff is not None else Handoff(), post_call,
        "Ana", now=lambda: NOW,
    )


def test_a_directed_interview_from_opening_to_the_sheet() -> None:
    store, handoff = Store(), Handoff()
    boss = director(handoff=handoff, store=store)
    out: Output = boss.output  # type: ignore[assignment]
    clock: Clock = boss.clock  # type: ignore[assignment]
    first = boss.plan_turn("", None, CancelToken())
    assert first.payload["action"] == "open"
    assert first.payload["slot"] == "s1"
    assert first.query.startswith("Atención al cliente")
    clock.t = 103.0
    boss.spoken(record("Hola, soy el entrevistador IA. ¿Dónde?", 1))
    answer = "Estuve dos años en caja en un supermercado del barrio"
    boss.heard("u1", answer, 3.0, 7.0)
    second = boss.plan_turn(answer, "u1", CancelToken())
    assert second.payload["action"] == "ask"
    assert second.payload["slot"] == "s2"
    clock.t = 110.0
    boss.spoken(record("Gracias. Cuéntame una vez...", 2))
    boss.heard("u2", "No sé.", 12.0, 13.0)
    boss.heard("u2", "No lo conozco.", 12.0, 13.5)  # a revision
    closing = boss.plan_turn("No lo conozco.", "u2", CancelToken())
    assert closing.payload["action"] == "close" and boss.closed
    clock.t = 116.0
    boss.spoken(record("Muchas gracias por tu tiempo.", 3))
    kinds = [m["type"] for m in out.messages]
    assert kinds.count("coverage") == 3
    assert out.messages[-1] == {"type": "interview", "state": "closed",
                                "closed_by": "covered"}
    result = boss.finish()
    assert result is not None and boss.finish() is result
    sheet = result["sheet"]
    assert sheet["schema"] == "interview.sheet.v1"
    assert sheet["candidate"] == "Ana" and sheet["scorer"] == "haiku"
    assert [s["status"] for s in sheet["slots"]] == ["filled", "unknown"]
    # The fake judge says correct_basic: the score is capped at 75 (#259).
    assert [s["score"] for s in sheet["slots"]] == [75, 0]
    assert sheet["slots"][0]["judgement"]["label"] == "correct_basic"
    assert sheet["judge"] == "interview.report@2"
    assert sheet["suitability"] == 37.5
    assert sheet["minimums_covered"] and sheet["turns_to_minimums"] == 2
    assert result["handoff"] == {"transcript": True, "analysis": "started",
                                 "error": None}
    transcript = handoff.transcripts[0]
    assert [u["speaker"] for u in transcript["utterances"]] == [
        "BOT", "S1", "BOT", "S1", "BOT"]
    assert transcript["session"]["type"] == "entrevista"
    ids = {u["text"]: u["id"] for u in transcript["utterances"]}
    assert sheet["slots"][0]["answers"][0]["utterance"] == ids[answer]
    assert "Atención al cliente en tienda" in handoff.jobs[0][1]
    assert store.sheets["plan-1"][0]["session_id"] == "vb-1"
    assert result["counts"]["unknown"] == 1


def test_finish_without_answers_and_with_a_failing_handoff() -> None:
    quiet = director(handoff=Handoff(fail=True))
    result = quiet.finish()
    assert result is not None
    assert result["sheet"]["scorer"] == "status"
    assert result["sheet"]["state"] == "stopped"
    assert result["handoff"]["error"] == "nothing was said"
    loud = director(handoff=Handoff(fail=True), reasoner=Down(),
                    post_call=False)
    loud.plan_turn("", None, CancelToken())
    loud.spoken(record("Hola", 1))
    result = loud.finish()
    assert result is not None
    assert result["handoff"]["error"] == "HTTP 422: bad"
    assert loud.coverage()["slots"][0]["asked"] == 1


def test_a_turn_dropped_during_the_filler_leaves_the_agenda_alone() -> None:
    boss = director()
    boss.plan_turn("", None, CancelToken())
    token = CancelToken()

    class Dropping(HeuristicFiller):
        def fill(self, *args: Any) -> Any:
            token.cancel()  # the candidate spoke while the filler ran
            return super().fill(*args)

    boss.filler = Dropping()
    target, turns = boss.target, boss.agenda.turns
    boss.plan_turn("Estuve dos años en caja en un supermercado del barrio",
                   "u1", token)
    assert (boss.target, boss.agenda.turns) == (target, turns)
    assert boss.agenda.sheet_slots()[0].answers == []


def test_reask_and_hint_turns_do_not_move_the_agenda() -> None:
    boss = director()
    boss.plan_turn("", None, CancelToken())
    turns = boss.agenda.turns
    again = boss.plan_turn("", None, CancelToken(), "reask")
    assert again.payload["action"] == "reask"
    assert again.payload["slot"] == "s1"
    hint = boss.plan_turn("Pues en mi último trabajo, eh", None,
                          CancelToken(), "hint")
    assert hint.payload["action"] == "hint"
    assert hint.payload["answer_so_far"] == "Pues en mi último trabajo, eh"
    assert boss.agenda.turns == turns and boss.target == "s1"
    out: Output = boss.output  # type: ignore[assignment]
    assert [m["action"] for m in out.messages
            if m["type"] == "coverage"][-2:] == ["reask", "hint"]


# ---- concepts per point (#258) ------------------------------------------

def running() -> Slot:
    return Slot(id="s1", label="Material de running", level="minimo",
                questions=["¿Qué sabes de zapatillas de running?"],
                points=[Point(id="p1", text="Tipos de pisada"),
                        Point(id="p2", text="Amortiguación y drop"),
                        Point(id="p3", text="Cómo elegir la talla")])


def test_read_fills_reads_the_points_of_each_slot() -> None:
    text = "Sé de pisada pronadora, el drop no lo controlo"
    found = read_fills({"slots": [
        {"id": "s1", "status": "partial", "quote": "pisada pronadora",
         "points": [
             {"id": "p1", "status": "treated", "quote": "pisada pronadora"},
             {"id": "p2", "status": "unknown", "quote": "inventado"},
             {"id": "p1", "status": "treated", "quote": "again"},
             {"id": "p9", "status": "treated"},
             {"id": "p3", "status": "pending"}, "junk"]},
        {"id": "s2", "points": [{"id": "p1", "status": "treated"}]},
        {"id": "s3", "points": "nope"},
    ]}, text, {"s1", "s2", "s3"},
        {"s1": {"p1", "p2", "p3"}, "s2": {"p1"}})
    assert [(f.slot, f.status) for f in found] == [("s1", "partial"),
                                                   ("s2", "partial")]
    assert [(h.point, h.status) for h in found[0].points] == [
        ("p1", "treated"), ("p2", "unknown")]
    assert found[0].points[1].quote.startswith("Sé de pisada")
    assert read_fills({"slots": [{"id": "s1", "status": "partial",
                                  "points": [{"id": "p1",
                                              "status": "treated"}]}]},
                      text, {"s1"})[0].points == ()


class Replies:
    """A reasoner that answers the filler with fixed data."""

    def __init__(self, data: Document) -> None:
        self.data = data
        self.requests: List[Document] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        self.requests.append(request)
        return {"status": "ok", "data": self.data}


def test_the_reasoner_filler_sends_the_points_and_fills_the_gap() -> None:
    target = running()
    answer = "Conozco los tipos de pisada y sé elegir bien la talla."
    told = Replies({"slots": [{"id": "s1", "status": "partial",
                               "quote": "tipos de pisada", "points": [
                                   {"id": "p1", "status": "treated"}]}]})
    found = ReasonerFiller(told).fill(answer, target, [target], "",
                                      CancelToken())
    sent = told.requests[0]
    assert sent["template"] == "interview.slots@2"
    assert sent["payload"]["slots"][0]["points"][2] == {
        "id": "p3", "text": "Cómo elegir la talla"}
    assert [h.point for h in found[0].points] == ["p1"]  # the model decides
    silent = Replies({"slots": [{"id": "s1", "status": "partial",
                                 "quote": "tipos de pisada"}]})
    found = ReasonerFiller(silent).fill(answer, target, [target], "",
                                        CancelToken())
    assert [h.point for h in found[0].points] == ["p1", "p3"]  # the rules
    nothing = Replies({"slots": []})  # the model: it covers no slot
    assert ReasonerFiller(nothing).fill(answer, target, [target], "",
                                        CancelToken()) == []
    plain = Replies({"slots": []})
    assert ReasonerFiller(plain).fill("Nada que ver", target, [target], "",
                                      CancelToken()) == []


def test_the_interviewer_is_told_the_points_and_the_sheet_keeps_them() -> None:
    store, handoff = Store(), Handoff()
    plan = InterviewPlan(id="plan-1", title="Dependiente/a",
                         slots=[running()])
    reasoner = FakeReasoner()
    boss = InterviewDirector(
        plan, "vb-1", InterviewSection(max_probes=1), HeuristicFiller(),
        reasoner, Output(), Clock(), store, handoff, False, "Ana",
        now=lambda: NOW)
    boss.plan_turn("", None, CancelToken())
    boss.spoken(record("Hola. ¿Qué sabes de zapatillas de running?", 1))
    answer = "Sé distinguir los tipos de pisada de cada cliente."
    boss.heard("u1", answer, 3.0, 6.0)
    probe = boss.plan_turn(answer, "u1", CancelToken())
    assert probe.payload["action"] == "probe"
    assert probe.payload["pending_points"] == ["Amortiguación y drop",
                                               "Cómo elegir la talla"]
    assert probe.payload["treated_points"] == ["Tipos de pisada"]
    coverage = boss.output.messages[-1]["coverage"]  # type: ignore
    assert [p["status"] for p in coverage["slots"][0]["points"]] == [
        "treated", "pending", "pending"]
    boss.spoken(record("Me has hablado de pisada. ¿Y del drop y la talla?",
                       2))
    more = "El drop no lo conozco."
    boss.heard("u2", more, 9.0, 10.0)
    boss.plan_turn(more, "u2", CancelToken())
    result = boss.finish()
    assert result is not None
    points = result["sheet"]["slots"][0]["points"]
    assert [p["status"] for p in points] == ["treated", "unknown", "pending"]
    ids = {u["text"]: u["id"] for u in handoff.transcripts[0]["utterances"]}
    assert points[0]["utterance"] == ids[answer]
    assert points[1]["utterance"] == ids[more]
    assert points[2]["utterance"] is None


# ---- accuracy of the answers (#259) -------------------------------------

def test_the_judge_reads_the_plan_s_documents_when_it_has_them() -> None:
    plan = InterviewPlan(
        id="plan-1", title="Dependiente/a", slots=[running()],
        source=PlanSource(kind="document", domain="tienda",
                          document="oferta.pdf"))
    knowledge = FakeKnowledge([{"id": "tienda:c1",
                                "text": "El drop es la diferencia de "
                                        "altura entre talón y puntera"}])
    reasoner = FakeReasoner()
    boss = InterviewDirector(
        plan, "vb-1", InterviewSection(max_probes=0), HeuristicFiller(),
        reasoner, Output(), Clock(), Store(), None, False, "Ana",
        now=lambda: NOW, knowledge=knowledge)
    boss.plan_turn("", None, CancelToken())
    answer = "El drop es la diferencia de altura entre talón y puntera."
    boss.heard("u1", answer, 3.0, 6.0)
    boss.plan_turn(answer, "u1", CancelToken())
    result = boss.finish()
    assert result is not None
    request = [r for r in reasoner.requests
               if r.get("template") == "interview.report@2"][0]
    slot = request["payload"]["slots"][0]
    assert slot["chunks"][0]["id"] == "tienda:c1"
    assert slot["answers"] == [{"utterance": "u1", "text": answer}]
    assert knowledge.queries[0]["domain"] == "tienda"
    judged = result["sheet"]["slots"][0]["judgement"]
    assert judged["label"] == "correct_basic" and judged["basis"] == "general"


def test_a_pasted_offer_or_the_plain_report_has_no_chunks() -> None:
    knowledge = FakeKnowledge([{"id": "c1", "text": "drop"}],
                              everything=True)
    plan = InterviewPlan(id="plan-1", title="T", slots=[running()])
    assert judge_chunks(knowledge, plan, [], {}, "r") == {}
    plain = director()
    plain.config = InterviewSection(report_template="interview.report@1")
    assert plain._chunks([]) == {}
    assert knowledge.queries == []


def finished_sheet(monkeypatch: pytest.MonkeyPatch, broken: str) -> Document:
    """A short interview whose report step ``broken`` raises."""
    import orchestrator.application.interview as module

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("odd reply")

    monkeypatch.setattr(module, broken, boom)
    boss = director()
    boss.plan_turn("", None, CancelToken())
    answer = "Estuve dos años en caja en un supermercado del barrio"
    boss.heard("u1", answer, 3.0, 7.0)
    boss.plan_turn(answer, "u1", CancelToken())
    result = boss.finish()
    assert result is not None
    sheet: Document = result["sheet"]
    return sheet


def test_a_judgement_that_fails_keeps_the_plain_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #259: the end of the interview never breaks on it."""
    sheet = finished_sheet(monkeypatch, "judge_slots")
    assert sheet["scorer"] == "haiku" and sheet["judge"] is None
    assert sheet["slots"][0]["score"] == 80  # the model's, not capped
    assert sheet["to_check"] == []


def test_a_report_that_cannot_be_read_gives_status_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sheet = finished_sheet(monkeypatch, "read_report")
    assert sheet["scorer"] == "status" and sheet["judge"] is None
    assert sheet["slots"][0]["scored_by"] == "status"
