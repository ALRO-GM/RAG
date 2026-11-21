"""``orchestrator serve``: routes, token, errors and the SSE stream."""

import json
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.sessions import LiveSessions
from tests.unit.orchestrator.fixtures import product, transcript
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    SignalReasoner,
    Transcripts,
)

AUTH = {"Authorization": "Bearer tok"}


def app(token: "str | None" = "tok") -> TestClient:
    sessions = LiveSessions(
        {"live_assist.support": product(),
         "live_assist.interview": product("live_assist.interview")},
        FakeKnowledge(), SignalReasoner(), Transcripts({"t": transcript()}),
        Clock(), InlineRunner(), InlineExecutor,
    )
    return TestClient(create_app(sessions, token, keepalive=0.01))


def sse(text: str) -> List[Dict[str, Any]]:
    """Parse a whole SSE body into ``{"id", "event", "data"}``."""
    out = []
    for block in text.split("\n\n"):
        item: Dict[str, Any] = {}
        for line in block.splitlines():
            key, _, value = line.partition(": ")
            if key in ("id", "event", "data"):
                item[key] = value
        if "data" in item:
            out.append(item)
    return out


def test_the_token_guards_everything_but_health() -> None:
    web = app()
    assert web.get("/health").json() == {"status": "ok"}
    assert web.get("/products").status_code == 401
    assert web.get("/products",
                   headers={"Authorization": "Bearer no"}).status_code == 401
    answer = web.get("/products", headers=AUTH)
    assert answer.status_code == 200
    assert answer.headers["cache-control"] == "no-store"
    names = [p["product"] for p in answer.json()["products"]]
    assert names == ["live_assist.interview", "live_assist.support"]
    assert web.get("/transcripts", headers=AUTH).json() == {
        "transcripts": [{"id": "t"}]}


def test_a_replay_is_streamed_as_server_sent_events() -> None:
    web = app(None)
    started = web.post("/sessions/demo/replay", json={
        "product": "live_assist.support", "transcript": "t", "speed": None})
    assert started.status_code == 201
    assert web.get("/sessions/demo").json()["state"] == "ended"
    assert web.get("/sessions").json()["sessions"][0]["id"] == "demo"
    answer = web.get("/sessions/demo/suggestions/stream")
    assert answer.headers["content-type"].startswith("text/event-stream")
    items = sse(answer.text)
    assert items[-1]["event"] == "end"
    messages = [json.loads(i["data"]) for i in items[:-1]]
    assert [int(i["id"]) for i in items[:-1]] == list(
        range(1, len(messages) + 1))
    kinds = {m["type"] for m in messages}
    assert kinds == {"session", "event", "suggestion", "status"}
    record = next(m["record"] for m in messages if m["type"] == "suggestion")
    assert record["stages"]["knowledge_ms"] is not None
    resumed = sse(web.get("/sessions/demo/suggestions/stream",
                          headers={"Last-Event-ID": "3"}).text)
    assert resumed[0]["id"] == "4"
    after = sse(web.get("/sessions/demo/suggestions/stream?after=5").text)
    assert after[0]["id"] == "6"


def test_refusals_carry_stable_codes() -> None:
    web = app(None)
    assert web.get("/sessions/nope").status_code == 404
    assert web.get("/sessions/nope/suggestions/stream").status_code == 404
    bad = [
        ({"product": "live_assist.support"}, 422),
        ({"product": "live_assist.support", "transcript": "t",
          "speed": "fast"}, 422),
        ({"product": "live_assist.support", "transcript": "t",
          "speed": True}, 422),
        ({"product": "x", "transcript": "t"}, 404),
    ]
    for body, status in bad:
        assert web.post("/sessions/a/replay", json=body
                        ).status_code == status, body
    assert web.post("/sessions/a/replay", content=b"[1]").status_code == 422
    assert web.post("/sessions/a/replay", content=b"{").status_code == 422
    assert web.post("/sessions/a/replay",
                    content=b"{" + b" " * 20000 + b"}").status_code == 422
    live = web.post("/sessions", json={"product": "live_assist.support",
                                       "audio": "mic"})
    assert live.status_code == 503
    assert live.json()["code"] == "unavailable"
    web.post("/sessions/b/replay", json={
        "product": "live_assist.support", "transcript": "t"})
    assert web.post("/sessions/b/stop").json()["state"] == "ended"


def test_barge_in_route() -> None:
    from orchestrator.adapters.outbound.fakes import FakeReasoner, FakeTts

    sessions = LiveSessions(
        {"voicebot": product("voicebot")}, FakeKnowledge(), FakeReasoner(),
        Transcripts({"t": transcript()}), Clock(), InlineRunner(hold=True),
        InlineExecutor, tts=FakeTts(),
    )
    web = TestClient(create_app(sessions, "tok", keepalive=0.01))
    assert web.post("/sessions/x/barge-in", headers=AUTH).status_code == 404
    web.post("/sessions/vb/replay", headers=AUTH,
             json={"product": "voicebot", "transcript": "t"})
    answer = web.post("/sessions/vb/barge-in", headers=AUTH)
    assert answer.status_code == 200 and answer.json() == {"stopped": False}
