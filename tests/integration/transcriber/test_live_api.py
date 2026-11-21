"""The live routes over the real API with the fake runtime (LV-2, LV-7).

A session is started over HTTP, fed PCM over the audio WebSocket and
followed over the events WebSocket; every event must follow
``live.event.v1``, the live log lands in ``<data_dir>/live`` without any
audio, and the transcript is published when the start request asks.
"""

import json
import stat
from pathlib import Path
from typing import Any, Dict, Iterator, List

import jsonschema
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.unit.transcriber.live_fakes import (
    LIVE_SCHEMA,
    bursts,
    chunks_of,
    pcm,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import ApiSettings, TranscriberSettings

CODE = "a-long-enough-access-code"
AUTH = {"X-Access-Code": CODE}
WAIT = 30.0
TONE = 0.5


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A fake-runtime container; its runner is closed afterwards."""
    built = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
        api=ApiSettings(failed_code_limit=3),
    ))
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client that sends the access code."""
    return TestClient(create_app(container, CODE), headers=AUTH)


def speech(channels: int = 1) -> bytes:
    """Speech-like tones the fake models hear, then silence."""
    first = bursts([(0.2, 2.2)], 3.2, TONE)
    if channels == 1:
        return pcm(first)
    return pcm(first, bursts([(2.3, 3.0)], 3.2, TONE))


def start(client: TestClient, **body: Any) -> Dict[str, Any]:
    """Start a live session; return the answer."""
    answer = client.post("/api/live/sessions", json=body)
    assert answer.status_code == 201, answer.text
    return dict(answer.json())


def send_audio(client: TestClient, session: str, data: bytes,
               channels: int = 1) -> Dict[str, Any]:
    """Send ``data`` over the audio socket, then stop; return ready."""
    with client.websocket_connect(f"/api/live/{session}/audio") as ws:
        ready = ws.receive_json()
        for message in chunks_of(data, 3200 * channels):
            ws.send_bytes(message)
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "stopped"}
    return dict(ready)


def follow(client: TestClient, path: str) -> List[Dict[str, Any]]:
    """Read every event of a socket until it closes."""
    events: List[Dict[str, Any]] = []
    with client.websocket_connect(path) as ws:
        try:
            while True:
                events.append(json.loads(ws.receive_text()))
        except WebSocketDisconnect as exc:
            assert exc.code == 1000
    return events


def test_a_live_session_from_start_to_transcript(
    client: TestClient, container: TranscriberContainer
) -> None:
    started = start(
        client, session_id="call-1", channels=2,
        product="live_assist.interview",
        speakers=[
            {"channel": 0, "id": "S1", "role": "interviewer"},
            {"channel": 1, "id": "S2", "role": "candidate"},
        ],
        handoff={"transcript_v1": True},
    )
    assert started["audio"] == "/api/live/call-1/audio"
    assert started["events"] == "/api/live/call-1/events"
    assert started["device"] in (None, "cpu")
    ready = send_audio(client, "call-1", speech(2), channels=2)
    assert ready == {
        "type": "ready", "session_id": "call-1", "channels": 2,
        "sample_rate": 16000, "format": "pcm_s16le",
        "max_message_bytes": 64000,
    }
    events = follow(client, "/api/live/call-1/events")
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    assert events[0]["type"] == "session_started"
    assert [s["role"] for s in events[0]["speakers"]] == [
        "interviewer", "candidate"
    ]
    assert events[-1]["type"] == "session_ended"
    finals = [e["utterance"] for e in events if e["type"] == "final"]
    assert {u["speaker"] for u in finals} == {"S1", "S2"}
    late = follow(client, "/live/call-1/events?from_seq=1")
    assert late == events[1:]
    info = client.get("/api/live/sessions/call-1").json()
    assert info["state"] == "ended" and info["transcript"]["published"]
    folder = container.settings.data_dir / "live" / "call-1"
    assert sorted(p.name for p in folder.iterdir()) == [
        "events.jsonl", "transcript.json"
    ]
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    logged = [
        json.loads(line)
        for line in (folder / "events.jsonl").read_text().splitlines()
    ]
    assert logged == events
    published = json.loads(
        (container.settings.published() / "call-1.transcript.v1.json")
        .read_text()
    )
    assert published["session"]["id"] == "call-1"
    assert [u["speaker"] for u in published["utterances"]] == ["S1", "S2"]
    listed = client.get("/api/live/sessions").json()["sessions"]
    assert [s["session_id"] for s in listed] == ["call-1"]


def test_a_browser_socket_sends_the_code_in_its_first_message(
    container: TranscriberContainer,
) -> None:
    app = create_app(container, CODE)
    TestClient(app, headers=AUTH).post(
        "/api/live/sessions", json={"session_id": "b"}
    )
    browser = TestClient(app)
    with browser.websocket_connect("/api/live/b/audio") as ws:
        ws.send_text(json.dumps({"type": "auth", "access_code": CODE}))
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(pcm(bursts([(0.1, 1.0)], 1.0, TONE)))
    answer = TestClient(app, headers=AUTH).post(
        "/api/live/sessions/b/stop"
    )
    assert answer.json()["state"] == "ended"
    assert answer.json()["finals"] >= 1


def refused(client: TestClient, path: str, first: str = "") -> Any:
    """Connect, maybe send ``first``; return the error and close code."""
    with client.websocket_connect(path) as ws:
        if first:
            ws.send_text(first)
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    return error["code"], closed.value.code


def test_sockets_without_a_valid_code_are_refused_and_throttled(
    container: TranscriberContainer,
) -> None:
    app = create_app(container, CODE)
    TestClient(app, headers=AUTH).post(
        "/api/live/sessions", json={"session_id": "t"}
    )
    plain = TestClient(app)
    wrong = json.dumps({"type": "auth", "access_code": "wrong"})
    assert refused(plain, "/api/live/t/audio", wrong) == (
        "unauthorized", 4401
    )
    assert refused(plain, "/api/live/t/events", "not json") == (
        "unauthorized", 4401
    )
    assert refused(plain, "/api/live/t/events", wrong) == (
        "unauthorized", 4401
    )
    assert refused(plain, "/api/live/t/events", wrong) == (
        "too_many_attempts", 4429
    )
    assert plain.post("/api/live/sessions", json={}).status_code == 429
    TestClient(app, headers=AUTH).post("/api/live/sessions/t/stop")


def test_socket_refusals_name_their_reason(client: TestClient) -> None:
    assert refused(client, "/api/live/nope/audio") == ("not_found", 4404)
    assert refused(client, "/api/live/nope/events") == ("not_found", 4404)
    start(client, session_id="r")
    assert refused(client, "/api/live/r/events?from_seq=x") == (
        "invalid_input", 4400
    )
    assert refused(client, "/api/live/r/events?from_seq=-1") == (
        "invalid_input", 4400
    )
    with client.websocket_connect("/api/live/r/audio") as ws:
        ws.receive_json()
        assert refused(client, "/api/live/r/audio") == ("wrong_state", 4409)
        ws.send_bytes(b"\x00")
        assert ws.receive_json()["code"] == "invalid_input"
    client.post("/api/live/sessions/r/stop")
    assert refused(client, "/api/live/r/audio") == ("wrong_state", 4409)


def test_http_errors_of_the_live_routes(client: TestClient) -> None:
    bad = client.post("/api/live/sessions", json={"session_id": "../x"})
    assert bad.status_code == 422
    start(client, session_id="e")
    again = client.post("/api/live/sessions", json={"session_id": "e"})
    assert again.status_code == 409
    other = client.post("/api/live/sessions", json={"session_id": "f"})
    assert other.json()["code"] == "busy"
    assert client.get("/api/live/sessions/zzz").status_code == 404
    assert client.post("/api/live/sessions/zzz/stop").status_code == 404
    client.post("/api/live/sessions/e/stop")


def test_a_socket_that_closes_lets_another_source_in(
    client: TestClient,
) -> None:
    start(client, session_id="c")
    with client.websocket_connect("/api/live/c/audio") as ws:
        ws.receive_json()
        ws.send_text("hello")
    with client.websocket_connect("/api/live/c/audio") as ws:
        assert ws.receive_json()["type"] == "ready"
    client.post("/api/live/sessions/c/stop")


def test_a_speaker_named_during_the_session_is_named_in_the_transcript(
    client: TestClient, container: TranscriberContainer
) -> None:
    start(
        client, session_id="named-1", channels=2,
        product="live_assist.support",
        speakers=[
            {"channel": 0, "id": "S1", "role": "agent", "name": "Ana"},
            {"channel": 1, "id": "S2", "role": "customer"},
        ],
        handoff={"transcript_v1": True},
    )
    named = client.post("/api/live/sessions/named-1/speakers/S2",
                        json={"name": "  Carla  Ruiz "})
    assert named.status_code == 200
    assert named.json()["speakers"][1]["name"] == "Carla Ruiz"
    for bad in ({"name": ""}, {"name": "x" * 81}, {}, [1]):
        answer = client.post("/api/live/sessions/named-1/speakers/S2",
                             json=bad)
        assert answer.status_code in (400, 422), bad
    assert client.post("/api/live/sessions/named-1/speakers/S2",
                       content=b"{").status_code in (400, 422)
    assert client.post("/api/live/sessions/named-1/speakers/..x",
                       json={"name": "A"}).status_code in (400, 404, 422)
    assert client.post("/api/live/sessions/zzz/speakers/S2",
                       json={"name": "A"}).status_code == 404
    send_audio(client, "named-1", speech(2), channels=2)
    follow(client, "/api/live/named-1/events")
    published = json.loads(
        (container.settings.published() / "named-1.transcript.v1.json")
        .read_text()
    )
    names = {s["id"]: s["name"] for s in published["speakers"]}
    assert names == {"S1": "Ana", "S2": "Carla Ruiz"}
    ended = client.post("/api/live/sessions/named-1/speakers/S2",
                        json={"name": "Otra"})
    assert ended.status_code == 409
