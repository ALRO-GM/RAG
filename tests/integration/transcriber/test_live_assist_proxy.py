"""``/api/assist/...``: the orchestrator's live-page API, forwarded."""

from pathlib import Path
from typing import List

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from tests.integration.transcriber.test_analysis_judge import CODE
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.live_assist_proxy import (
    ALLOWED,
    URL_VARIABLE,
    add_live_assist_proxy,
    read_token,
)
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings

SSE = b"id: 1\nevent: message\ndata: {\"type\": \"session\"}\n\n"


def guarded(request: Request) -> None:
    """Stand-in for the access-code dependency of the real router."""
    if request.headers.get("X-Access-Code") != "code":
        raise HTTPException(401)


def app_with(
    handler: "httpx.MockTransport", token_file: "Path | None"
) -> TestClient:
    api = APIRouter(prefix="/api", dependencies=[Depends(guarded)])
    add_live_assist_proxy(api, token_file, "http://orch:8772/",
                          transport=handler)
    app = FastAPI()
    app.include_router(api)
    return TestClient(app)


def test_calls_are_forwarded_with_the_bearer_token(tmp_path: Path) -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "d1", "state": "running"})

    token = tmp_path / "token"
    token.write_text("secret\n", encoding="utf-8")
    web = app_with(httpx.MockTransport(handler), token)
    answer = web.post(
        "/api/assist/sessions/d1/replay",
        json={"product": "live_assist.support", "transcript": "x"},
        headers={"X-Access-Code": "code"},
    )
    assert answer.status_code == 201
    assert answer.json() == {"id": "d1", "state": "running"}
    request = seen[0]
    assert str(request.url) == "http://orch:8772/sessions/d1/replay"
    assert request.headers["authorization"] == "Bearer secret"
    assert b"live_assist.support" in request.content


def test_the_code_is_needed_and_only_known_routes_pass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be reached")

    web = app_with(httpx.MockTransport(handler), None)
    assert web.get("/api/assist/products").status_code == 401
    auth = {"X-Access-Code": "code"}
    for path in ("health", "sessions/a/b", "../x", "sessions/../../etc",
                 "products/x"):
        assert web.get(f"/api/assist/{path}",
                       headers=auth).status_code == 404, path
    assert web.post("/api/assist/sessions/.x/stop",
                    headers=auth).status_code == 404
    assert web.post("/api/assist/sessions", content=b"x" * 20000,
                    headers=auth).status_code == 413


def test_the_stream_is_passed_through_with_resume() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=SSE, headers={"content-type": "text/event-stream"}
        )

    web = app_with(httpx.MockTransport(handler), None)
    answer = web.get(
        "/api/assist/sessions/d1/suggestions/stream?after=3&x=1",
        headers={"X-Access-Code": "code", "Last-Event-ID": "3"},
    )
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("text/event-stream")
    assert answer.content == SSE
    assert seen[0].url.params.get("after") == "3"
    assert "x" not in seen[0].url.params
    assert seen[0].headers["last-event-id"] == "3"
    assert "authorization" not in seen[0].headers


def test_refusals_and_a_missing_orchestrator() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": "not_found"})

    web = app_with(httpx.MockTransport(refuse), None)
    auth = {"X-Access-Code": "code"}
    stream = web.get("/api/assist/sessions/x/suggestions/stream",
                     headers=auth)
    assert stream.status_code == 404
    assert stream.json() == {"code": "not_found"}

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    web = app_with(httpx.MockTransport(down), None)
    for path in ("products", "sessions/x/suggestions/stream"):
        answer = web.get(f"/api/assist/{path}", headers=auth)
        assert answer.status_code == 502
        assert answer.json()["code"] == "orchestrator_unavailable"


def test_tokens_and_routes() -> None:
    assert read_token(None) is None
    assert read_token(Path("/nonexistent/token")) is None
    assert ALLOWED.match("sessions/live-20261121-1010/suggestions/stream")
    assert ALLOWED.match("sessions/vb-1/barge-in")
    assert not ALLOWED.match("sessions/a/suggestions")


def test_the_transcriber_mounts_the_route_behind_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(URL_VARIABLE, "http://127.0.0.1:9")
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
        rag_token_file=tmp_path / "none",
    ))
    web = TestClient(create_app(container, CODE))
    assert web.get("/api/assist/products").status_code == 401
    answer = web.get("/api/assist/products",
                     headers={"X-Access-Code": CODE})
    assert answer.status_code == 502


def test_barge_in_is_forwarded() -> None:
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"stopped": True})

    web = app_with(httpx.MockTransport(handler), None)
    answer = web.post("/api/assist/sessions/vb-1/barge-in",
                      headers={"X-Access-Code": "code"})
    assert answer.json() == {"stopped": True}
    assert seen == ["/sessions/vb-1/barge-in"]


def test_the_lv16_session_routes_are_forwarded() -> None:
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    web = app_with(httpx.MockTransport(handler), None)
    auth = {"X-Access-Code": "code"}
    for path in ("sessions/d1/mode", "sessions/d1/speakers/S2",
                 "sessions/d1/mandatory/aviso-datos"):
        answer = web.post(f"/api/assist/{path}", json={}, headers=auth)
        assert answer.status_code == 200, path
    assert seen == ["/sessions/d1/mode", "/sessions/d1/speakers/S2",
                    "/sessions/d1/mandatory/aviso-datos"]
    assert not ALLOWED.match("sessions/d1/mandatory/Bad_Item")
    assert web.post("/api/assist/sessions/d1/speakers/.x", json={},
                    headers=auth).status_code == 404


def test_interview_plan_routes_are_forwarded() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "plan-1"})

    web = app_with(httpx.MockTransport(handler), None)
    head = {"X-Access-Code": "code"}
    offer = {"offer": "ñ" * 20000}
    assert web.post("/api/assist/interview/plans", json=offer,
                    headers=head).status_code == 200
    assert web.get("/api/assist/interview/plans", headers=head).json() == {
        "id": "plan-1"}
    assert web.put("/api/assist/interview/plans/plan-1",
                   json={"revision": 0}, headers=head).status_code == 200
    web.get("/api/assist/interview/plans/plan-1", headers=head)
    web.get("/api/assist/interview/plans/plan-1/results", headers=head)
    assert [(r.method, r.url.path) for r in seen] == [
        ("POST", "/interview/plans"), ("GET", "/interview/plans"),
        ("PUT", "/interview/plans/plan-1"),
        ("GET", "/interview/plans/plan-1"),
        ("GET", "/interview/plans/plan-1/results"),
    ]
    assert len(seen[0].content) > 40000
    assert seen[2].content == b'{"revision":0}'
    assert ALLOWED.match("interview/plans/plan-1/results")
    assert not ALLOWED.match("interview/plans/../x")
    assert web.get("/api/assist/interview/plans/plan-1").status_code == 401


def test_candidate_routes_are_forwarded() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "ana"})

    web = app_with(httpx.MockTransport(handler), None)
    head = {"X-Access-Code": "code"}
    base = "/api/assist/interview/plans/plan-1/candidates"
    web.get(base, headers=head)
    web.post(base, json={"name": "Ana"}, headers=head)
    web.get(f"{base}/ana", headers=head)
    web.put(f"{base}/ana/answers/work_authorization", json={"text": "Sí"},
            headers=head)
    web.delete(f"{base}/ana/answers/work_authorization", headers=head)
    web.delete(f"{base}/ana", headers=head)
    web.get("/api/assist/interview/candidates/by-session/live-1",
            headers=head)
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/interview/plans/plan-1/candidates"),
        ("POST", "/interview/plans/plan-1/candidates"),
        ("GET", "/interview/plans/plan-1/candidates/ana"),
        ("PUT", "/interview/plans/plan-1/candidates/ana/answers/"
                "work_authorization"),
        ("DELETE", "/interview/plans/plan-1/candidates/ana/answers/"
                   "work_authorization"),
        ("DELETE", "/interview/plans/plan-1/candidates/ana"),
        ("GET", "/interview/candidates/by-session/live-1"),
    ]
    assert seen[3].content == '{"text":"Sí"}'.encode()
    assert not ALLOWED.match("interview/plans/p/candidates/a/answers/../x")
    assert not ALLOWED.match("interview/plans/p/candidates/a/answers/Bad")
    assert web.get(f"{base}/ana").status_code == 401


def test_confirm_and_discard_routes_are_forwarded() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "ana"})

    web = app_with(httpx.MockTransport(handler), None)
    head = {"X-Access-Code": "code"}
    base = "/api/assist/interview/plans/plan-1/candidates/ana"
    web.post(f"{base}/confirm", json={}, headers=head)
    web.delete(f"{base}/proposals/salary_expectation", headers=head)
    assert [(r.method, r.url.path) for r in seen] == [
        ("POST", "/interview/plans/plan-1/candidates/ana/confirm"),
        ("DELETE", "/interview/plans/plan-1/candidates/ana/proposals/"
                   "salary_expectation"),
    ]
    assert not ALLOWED.match("interview/plans/p/candidates/a/proposals/X")


def test_the_demo_report_and_voice_example_are_forwarded() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"demo": True})

    web = app_with(httpx.MockTransport(handler), None)
    auth = {"X-Access-Code": "code"}
    for part in ("report", "voicebot"):
        answer = web.get(f"/api/assist/demos/llamada-lumbre/{part}",
                         headers=auth)
        assert answer.json() == {"demo": True}
    assert [str(r.url) for r in seen] == [
        "http://orch:8772/demos/llamada-lumbre/report",
        "http://orch:8772/demos/llamada-lumbre/voicebot",
    ]
    assert web.get("/api/assist/demos/x/other",
                   headers=auth).status_code == 404
    assert web.get("/api/assist/demos/../report",
                   headers=auth).status_code == 404
    assert ALLOWED.match("demos/a.b-c/report")


def test_the_sample_may_take_longer_than_a_plain_call() -> None:
    """A cloned voice may wait for GPU memory and load its model (#253)."""
    from transcriber.adapters.inbound.live_assist_proxy import (
        SAMPLE_TIMEOUT,
        TIMEOUT,
    )

    waits = {}

    def handler(request: httpx.Request) -> httpx.Response:
        waits[request.url.path] = request.extensions["timeout"]["read"]
        return httpx.Response(200, content=b"RIFF",
                              headers={"content-type": "audio/wav"})

    web = app_with(httpx.MockTransport(handler), None)
    auth = {"X-Access-Code": "code"}
    web.post("/api/assist/voicebot/sample", json={"choices": {}},
             headers=auth)
    web.get("/api/assist/products", headers=auth)
    assert waits["/voicebot/sample"] == SAMPLE_TIMEOUT
    assert SAMPLE_TIMEOUT >= 140.0  # 20 s for memory + 120 s to speak
    assert waits["/products"] == TIMEOUT


def test_the_demo_voices_are_forwarded_with_their_audio() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith(".wav"):
            return httpx.Response(200, content=b"RIFFdata", headers={
                "content-type": "audio/wav",
                "cache-control": "private, no-store"})
        return httpx.Response(200, json={"examples": {}})

    web = app_with(httpx.MockTransport(handler), None)
    auth = {"X-Access-Code": "code"}
    assert web.get("/api/assist/demos/llamada-lumbre/voices",
                   headers=auth).json() == {"examples": {}}
    clip = web.get("/api/assist/demos/llamada-lumbre/voices/vb0.user.wav",
                   headers=auth)
    assert clip.content == b"RIFFdata"
    assert clip.headers["content-type"] == "audio/wav"
    assert clip.headers["cache-control"] == "private, no-store"
    assert [str(r.url) for r in seen] == [
        "http://orch:8772/demos/llamada-lumbre/voices",
        "http://orch:8772/demos/llamada-lumbre/voices/vb0.user.wav",
    ]
    for bad in ("voices/x.mp3", "voices/..%2Fx.wav", "voices/a/b.wav"):
        assert web.get(f"/api/assist/demos/llamada-lumbre/{bad}",
                       headers=auth).status_code == 404
    assert web.get("/api/assist/demos/llamada-lumbre/voices/t0.s0.wav"
                   ).status_code == 401, "behind the access code"
