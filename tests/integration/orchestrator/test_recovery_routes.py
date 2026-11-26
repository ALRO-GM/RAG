"""The orchestrator's side of a recovered round over HTTP (#265).

``POST /sessions/{id}/recovered`` rebuilds a round from the merged
transcript; deleting a candidate's file asks the transcriber to delete
any recovered audio of their rounds; the transcriber client's
``discard_recovery`` and the transcriber's round-rebuilder client.
Fakes and in-process transports only; invented words.
"""

from pathlib import Path
from typing import List

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.sessions import LiveSessions
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    SignalReasoner,
    Transcripts,
)
from orchestrator.adapters.outbound.http_transcriber import HttpTranscriber
from orchestrator.application.ports import TranscriberUnavailable
from orchestrator.adapters.outbound.fs_candidates import JsonCandidateStore
from orchestrator.adapters.outbound.fs_core_questions import FileCoreQuestions
from orchestrator.application.candidates import CandidateBook
from tests.integration.orchestrator.test_candidate_adapters import CORE
from tests.unit.orchestrator.test_interview_usecases import OFFER
from tests.unit.orchestrator.test_interview_sessions import live
from tests.integration.orchestrator.test_http_server import AUTH, app
from tests.unit.orchestrator.test_interview_kept import transcript
from transcriber.adapters.outbound.http_round_rebuilder import (
    HttpRoundRebuilder,
)
from transcriber.domain.errors import PublishError


def test_a_recovered_transcript_rebuilds_its_round() -> None:
    assert app().post("/sessions/r9/recovered",
                      json={"transcript": transcript("r9")},
                      headers=AUTH).status_code == 503, "no executor given"
    sessions = LiveSessions(
        {"live_assist.interview": product("live_assist.interview")},
        FakeKnowledge(), SignalReasoner(), Transcripts({}), Clock(),
        InlineRunner(), InlineExecutor)
    web = TestClient(create_app(sessions, "tok",
                                round_executor=InlineExecutor()))
    answer = web.post("/sessions/r9/recovered",
                      json={"transcript": transcript("r9")}, headers=AUTH)
    assert answer.status_code == 200, answer.text
    assert answer.json()["finals"] == 4
    assert web.post("/sessions/r9/recovered", json={},
                    headers=AUTH).status_code == 422
    assert web.post("/sessions/x/recovered",
                    json={"transcript": transcript("r9")},
                    headers=AUTH).status_code == 422
    assert web.post("/sessions/r9/recovered",
                    json={"transcript": transcript("r9")}
                    ).status_code == 401


def test_deleting_a_file_deletes_the_recovered_audio_too(
    tmp_path: Path,
) -> None:
    sessions = live([])
    wiring = sessions.interview
    assert wiring is not None and sessions.transcriber is not None
    wiring.candidates = CandidateBook(
        JsonCandidateStore(tmp_path), FileCoreQuestions(CORE),
        lambda p: p in wiring.store.plans,  # type: ignore[attr-defined]
        30,
    )
    asked: List[str] = []
    sessions.transcriber.discard_recovery = asked.append  # type: ignore
    web = TestClient(create_app(sessions))
    plan = web.post("/interview/plans", json={"offer": OFFER}).json()
    base = f"/interview/plans/{plan['id']}/candidates"
    web.post(base, json={"name": "Ana Ruiz"})
    started = web.post("/sessions", json={
        "product": "voicebot.interview", "audio": "mic",
        "plan": plan["id"], "candidate_id": "ana-ruiz"})
    assert started.status_code == 201
    assert web.delete(f"{base}/ana-ruiz").json()["deleted"] is True
    assert asked == ["vb-1"]
    assert web.delete(f"{base}/ana-ruiz").status_code == 404


def test_the_transcriber_client_deletes_recovered_audio() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = {"gone": 404, "down": 503}.get(
            request.url.path.split("/")[-2], 204)
        return httpx.Response(status)

    client = HttpTranscriber("http://t:8770", "code", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    client.discard_recovery("iv-1")
    client.discard_recovery("gone")
    with pytest.raises(TranscriberUnavailable):
        client.discard_recovery("down")
    assert seen[0].method == "DELETE"
    assert seen[0].url.path == "/api/live/sessions/iv-1/recovery"
    assert seen[0].headers["X-Access-Code"] == "code"

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no")

    with pytest.raises(TranscriberUnavailable):
        HttpTranscriber("http://t", "c", client=httpx.Client(
            transport=httpx.MockTransport(broken))).discard_recovery("x")


def test_the_round_rebuilder_posts_the_transcript_with_the_token(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text("tok\n")
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500 if "bad" in request.url.path else 200)

    transport = httpx.MockTransport(handler)
    HttpRoundRebuilder("http://o:8772/", token, transport).rebuild(
        "iv-1", {"schema": "transcript.v1"})
    assert seen[0].url.path == "/sessions/iv-1/recovered"
    assert seen[0].headers["Authorization"] == "Bearer tok"
    with pytest.raises(PublishError):
        HttpRoundRebuilder("http://o", token, transport).rebuild("bad", {})
    with pytest.raises(PublishError):
        HttpRoundRebuilder("http://o", None, transport).rebuild("a/b", {})
    HttpRoundRebuilder("http://o", tmp_path / "none", transport).rebuild(
        "iv-2", {})
    assert "Authorization" not in seen[-1].headers

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no")

    with pytest.raises(PublishError):
        HttpRoundRebuilder("http://o", None, httpx.MockTransport(broken)
                           ).rebuild("iv-3", {})
