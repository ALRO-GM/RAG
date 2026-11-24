"""LV-17 adapters: the avoid check over HTTP, the demo folder and route."""

import json
from typing import Any, List

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.http_avoid import HttpAvoid
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import AvoidUnavailable
from orchestrator.application.sessions import LiveSessions
from tests.unit.orchestrator.fixtures import REPO, product
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sentences_go_to_the_rag_s_avoid_check() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"language": "es", "hints": []})

    port = HttpAvoid("http://rag:8000/", "tok", client=client(handler))
    assert port.check("¿Qué tal?", "es") == {"language": "es", "hints": []}
    assert str(seen[0].url) == "http://rag:8000/knowledge/interview/avoid"
    assert json.loads(seen[0].content) == {"text": "¿Qué tal?",
                                           "language": "es"}
    assert seen[0].headers["authorization"] == "Bearer tok"


@pytest.mark.parametrize("response", [
    httpx.Response(404), httpx.Response(200, text="nope"),
    httpx.Response(200, json=[1]),
])
def test_avoid_failures_are_one_exception(response: httpx.Response) -> None:
    port = HttpAvoid("http://rag:8000", client=client(lambda r: response))
    with pytest.raises(AvoidUnavailable):
        port.check("hola", "es")


def test_the_routes_list_and_play_the_demo() -> None:
    sessions = LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge([]), FakeReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor,
        demos=DirDemos(REPO / "examples" / "demo"),
    )
    web = TestClient(create_app(sessions, None, keepalive=0.01))
    listed = web.get("/transcripts").json()
    assert listed["transcripts"] == []
    assert listed["demos"][0]["id"] == "entrevista-lucia"
    started = web.post("/sessions/demo-a/replay",
                       json={"demo": "entrevista-lucia", "speed": None})
    assert started.status_code == 201
    assert started.json()["demo"]["candidate"] == "Lucía Marín"
    missing = web.post("/sessions/demo-b/replay", json={"demo": "nope"})
    assert missing.status_code == 404
