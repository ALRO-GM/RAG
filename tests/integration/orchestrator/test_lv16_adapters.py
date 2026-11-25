"""LV-16 adapters: mandatory items over HTTP and from the manifests,
hand-off files, naming a speaker on the transcriber and the routes."""

import json
from pathlib import Path
from typing import Any, List

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.adapters.inbound.http_server import create_app
from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.fs_mandatory import FileMandatory
from orchestrator.adapters.outbound.http_mandatory import HttpMandatory
from orchestrator.adapters.outbound.http_transcriber import HttpTranscriber
from orchestrator.adapters.outbound.jsonl_sink import JsonHandoff
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import MandatoryUnavailable
from orchestrator.application.sessions import LiveSessions
from tests.unit.orchestrator.fixtures import REPO, product, transcript
from tests.unit.orchestrator.test_sessions import (
    Clock,
    InlineRunner,
    Transcripts,
)

DOMAINS = REPO / "knowledge" / "domains"


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_mandatory_items_and_checks_go_to_the_rag() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"domain": "d", "items": []})
        return httpx.Response(200, json={"items": [
            {"id": "a", "score": 0.7, "said": True}]})

    port = HttpMandatory("http://rag:8000/", "tok", client=client(handler))
    assert port.items("support.demo")["items"] == []
    assert port.check("support.demo", "hola")["items"][0]["said"] is True
    assert str(seen[0].url) == (
        "http://rag:8000/knowledge/domains/support.demo/mandatory")
    assert str(seen[1].url) == "http://rag:8000/knowledge/mandatory"
    assert json.loads(seen[1].content) == {"domain": "support.demo",
                                           "text": "hola"}
    assert all(r.headers["authorization"] == "Bearer tok" for r in seen)


@pytest.mark.parametrize("response", [
    httpx.Response(404), httpx.Response(200, text="nope"),
    httpx.Response(200, json=[1]),
])
def test_mandatory_failures_are_one_exception(response: httpx.Response
                                              ) -> None:
    port = HttpMandatory("http://x", client=client(lambda r: response))
    with pytest.raises(MandatoryUnavailable):
        port.items("d")
    with pytest.raises(MandatoryUnavailable):
        port.check("d", "t")


def test_the_manifests_give_the_lists_and_a_word_check() -> None:
    port = FileMandatory(DOMAINS)
    support = port.items("support.demo")
    assert support["version"] == 1 and support["threshold"] == 0.575
    assert [i["id"] for i in support["items"]][:2] == [
        "aviso-datos", "condiciones-contrato"]
    interview = port.items("interview.generic")
    assert {i["id"] for i in interview["items"]} == {
        "uso-datos", "aviso-asistente", "siguientes-pasos"}
    example = support["items"][2]["must_say"][1]
    said = port.check("support.demo", example)
    hits = [i["id"] for i in said["items"] if i["said"]]
    assert hits == ["desistimiento"]
    other = port.check("support.demo", "¿Me dice su número de pedido?")
    assert not any(i["said"] for i in other["items"])


def test_the_manifest_adapter_refuses_bad_or_missing_domains(
    tmp_path: Path,
) -> None:
    port = FileMandatory(tmp_path)
    with pytest.raises(MandatoryUnavailable):
        port.items("../etc")
    with pytest.raises(MandatoryUnavailable):
        port.items("missing")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "manifest.yaml").write_text("id: plain\n")
    assert port.items("plain") == {"domain": "plain", "version": 0,
                                   "items": []}
    assert port.check("plain", "hola")["items"] == []


def test_the_hand_off_is_written_as_json(tmp_path: Path) -> None:
    sink = JsonHandoff(tmp_path / "logs" / "s.handoff.json")
    sink.write({"session_id": "s", "names": {"S1": "Ána"}})
    written = json.loads((tmp_path / "logs" / "s.handoff.json").read_text())
    assert written["names"] == {"S1": "Ána"}
    # Only its owner reads it (#259 review), also when rewritten.
    path = tmp_path / "logs" / "s.handoff.json"
    assert path.stat().st_mode & 0o777 == 0o600
    path.chmod(0o644)
    sink.write({"session_id": "s", "accuracy": {"state": "ready"}})
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["accuracy"] == {"state": "ready"}


def test_a_name_is_sent_to_the_transcriber() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"session_id": "s"})

    HttpTranscriber("http://tr:8770", "c0de", client=client(handler)
                    ).rename("s-1", "S2", "Carla")
    assert str(seen[0].url) == (
        "http://tr:8770/api/live/sessions/s-1/speakers/S2")
    assert json.loads(seen[0].content) == {"name": "Carla"}


def test_the_session_routes_switch_mode_name_and_tick() -> None:
    runner = InlineRunner(hold=True)
    sessions = LiveSessions(
        {"live_assist.support": product()}, FakeKnowledge(), FakeReasoner(),
        Transcripts({"t": transcript()}), Clock(), runner, InlineExecutor,
        mandatory=FileMandatory(DOMAINS),
    )
    web = TestClient(create_app(sessions, None, keepalive=0.01))
    started = web.post("/sessions/d-1/replay", json={
        "product": "live_assist.support", "transcript": "t",
        "employee": {"id": "ana", "name": "Ana"}})
    assert started.status_code == 201
    assert started.json()["names"] == {"a": "Ana"}
    assert web.post("/sessions/d-1/mode",
                    json={"mode": "guiado"}).json() == {"mode": "guiado"}
    assert web.post("/sessions/d-1/mode", json={}).status_code == 422
    named = web.post("/sessions/d-1/speakers/c", json={"name": "Carlos"})
    assert named.json()["name"] == "Carlos"
    refused = web.post("/sessions/d-1/mandatory/aviso-datos", json={})
    assert refused.status_code == 422
    ticked = web.post("/sessions/d-1/mandatory/aviso-datos",
                      json={"confirm": True})
    assert ticked.status_code == 200
    assert ticked.json()["ticks"][0]["item"] == "aviso-datos"
    assert web.post("/sessions/nope/mode",
                    json={"mode": "guiado"}).status_code == 404
