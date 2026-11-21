"""The AI interviewer's adapters and routes (#208): files, RAG, API."""

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.adapters.outbound.fs_interview_plans import JsonPlanStore
from orchestrator.adapters.outbound.http_interview import (
    HttpHandoff,
    HttpOffers,
)
from orchestrator.application.interview import (
    HandoffFailed,
    OfferUnavailable,
)
from orchestrator.domain.interview import InterviewPlan
from tests.unit.orchestrator.test_interview_sessions import live
from tests.unit.orchestrator.test_interview_usecases import OFFER

ROOT = Path(__file__).resolve().parents[3]


def example() -> InterviewPlan:
    path = ROOT / "contracts" / "examples" / "interview.plan.v1.json"
    return InterviewPlan.model_validate_json(path.read_text("utf-8"))


def test_the_plan_store_keeps_plans_and_sheets(tmp_path: Path) -> None:
    store = JsonPlanStore(tmp_path)
    plan = example()
    assert store.list() == []
    store.save(plan)
    assert store.load(plan.id) == plan
    newer = plan.model_copy(update={"id": "plan-2", "updated_at": None})
    store.save(newer)
    assert [p.id for p in store.list()] == [plan.id, "plan-2"]
    (tmp_path / "plans" / "broken.json").write_text("{", "utf-8")
    assert len(store.list()) == 2
    with pytest.raises(KeyError):
        store.load("plan-404")
    with pytest.raises(KeyError):
        store.load("../etc")
    store.save_result(plan.id, {"session_id": "s-1", "finished_at": "1"})
    store.save_result(plan.id, {"session_id": "s-2", "finished_at": "2"})
    (tmp_path / "results" / plan.id / "x.json").write_text("[", "utf-8")
    assert [r["session_id"] for r in store.results(plan.id)] == [
        "s-2", "s-1"]
    with pytest.raises(KeyError):
        store.save_result(plan.id, {"session_id": "../x"})
    assert store.results("plan-2") == []


def rag(responses: Dict[str, httpx.Response],
        seen: List[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.get(request.url.path, httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_offers_are_read_from_the_rag() -> None:
    seen: List[httpx.Request] = []
    path = "/knowledge/domains/ofertas/documents/abc/text"
    client = rag({path: httpx.Response(200, json={
        "id": "abc", "name": "oferta.pdf", "text": OFFER})}, seen)
    offers = HttpOffers("http://rag/", "tok", client=client)
    assert offers.document_text("ofertas", "abc") == ("oferta.pdf", OFFER)
    assert seen[0].headers["authorization"] == "Bearer tok"
    with pytest.raises(OfferUnavailable):
        offers.document_text("ofertas", "missing")
    with pytest.raises(OfferUnavailable):
        offers.document_text("../x", "abc")
    bad = HttpOffers("http://rag", client=rag(
        {path: httpx.Response(200, text="nope")}, seen))
    with pytest.raises(OfferUnavailable):
        bad.document_text("ofertas", "abc")
    odd = HttpOffers("http://rag", client=rag(
        {path: httpx.Response(200, json={"text": 3})}, seen))
    with pytest.raises(OfferUnavailable):
        odd.document_text("ofertas", "abc")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    gone = HttpOffers("http://rag", client=httpx.Client(
        transport=httpx.MockTransport(down)))
    with pytest.raises(OfferUnavailable):
        gone.document_text("ofertas", "abc")


def test_the_handoff_posts_the_transcript_and_the_analysis() -> None:
    seen: List[httpx.Request] = []
    client = rag({
        "/transcripts": httpx.Response(201, json={"id": "vb-1"}),
        "/analysis/sessions/vb-1": httpx.Response(202, json={}),
        "/analysis/sessions/vb-2": httpx.Response(
            409, json={"detail": "running"}),
    }, seen)
    handoff = HttpHandoff("http://rag", "tok", client=client)
    handoff.publish_transcript({"schema": "transcript.v1"})
    handoff.start_analysis("vb-1", "Puesto: x")
    assert json.loads(seen[1].content) == {"job": "Puesto: x"}
    with pytest.raises(HandoffFailed) as caught:
        handoff.start_analysis("vb-2", "x")
    assert "409" in str(caught.value) and "running" in str(caught.value)
    with pytest.raises(HandoffFailed):
        handoff.start_analysis("../x", "x")


def test_the_plan_routes(tmp_path: Path) -> None:
    sessions = live([])
    web = TestClient(create_app(sessions))
    made = web.post("/interview/plans", json={"offer": OFFER})
    assert made.status_code == 201
    plan: Dict[str, Any] = made.json()
    assert web.get("/interview/plans").json()["plans"][0]["id"] == \
        plan["id"]
    assert web.get(f"/interview/plans/{plan['id']}").json() == plan
    plan["minutes"] = 30
    saved = web.put(f"/interview/plans/{plan['id']}", json=plan)
    assert saved.status_code == 200 and saved.json()["revision"] == 1
    stale = web.put(f"/interview/plans/{plan['id']}", json=plan)
    assert stale.status_code == 409 and stale.json()["code"] == "conflict"
    assert web.get("/interview/plans/plan-x").status_code == 404
    assert web.post("/interview/plans",
                    json={"offer": "x"}).status_code == 422
    results = web.get(f"/interview/plans/{plan['id']}/results").json()
    assert results["candidates"] == []
    big = web.post("/interview/plans", content=b"x" * (100 * 1024))
    assert big.status_code == 422
    started = web.post("/sessions", json={
        "product": "voicebot.interview", "audio": "mic",
        "plan": plan["id"], "candidate": "Ana"})
    assert started.status_code == 201
    assert started.json()["plan"] == plan["id"]
    wrong = web.post("/sessions", json={
        "product": "voicebot", "audio": "mic", "domain": 3})
    assert wrong.status_code == 422


def test_without_interview_wiring_the_routes_say_unavailable() -> None:
    sessions = live([], wired=False)
    sessions.tts = FakeTts()
    web = TestClient(create_app(sessions))
    assert web.get("/interview/plans").status_code == 503
