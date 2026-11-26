"""«Compartir pantalla» over the real API and WebSocket (#278).

The protocol end to end with the fake runtime: the access code is
needed, a late joiner gets the whole page before any change, a device
that only watches cannot act (the server drops what its page sends), a
dropped link comes back with its token and resynchronises, and the
refusals close with their codes. Invented names and passwords only.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from transcriber.adapters.inbound import share_ws
from transcriber.adapters.inbound.http_api import create_app
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import ApiSettings, TranscriberSettings

CODE = "a-long-enough-access-code"
WAIT = 30.0


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
def client(container: TranscriberContainer) -> Iterator[TestClient]:
    """One client (one event loop) for every socket of a test."""
    with TestClient(create_app(container, CODE)) as test_client:
        yield test_client


def hello(ws: WebSocketTestSession, **message: Any) -> Dict[str, Any]:
    """Authenticate, say hello, return the first answer."""
    ws.send_text(json.dumps({"type": "auth", "access_code": CODE}))
    ws.send_text(json.dumps(message))
    return dict(ws.receive_json())


def until(ws: WebSocketTestSession, kind: str) -> Dict[str, Any]:
    """Read messages until one of ``kind``; return it."""
    for _ in range(50):
        message = dict(ws.receive_json())
        if message["type"] == kind:
            return message
    raise AssertionError(f"no {kind}")


def closed_with(ws: WebSocketTestSession) -> int:
    """Read until the socket closes; return its close code."""
    with pytest.raises(WebSocketDisconnect) as caught:
        for _ in range(50):
            ws.receive_json()
    return int(caught.value.code)


def test_a_device_needs_the_access_code(client: TestClient) -> None:
    with client.websocket_connect("/api/share") as ws:
        ws.send_text(json.dumps({"type": "auth", "access_code": "wrong"}))
        assert ws.receive_json()["code"] == "unauthorized"
        assert closed_with(ws) == 4401


def test_share_join_and_see_the_same_page(client: TestClient) -> None:
    with client.websocket_connect("/api/share") as host:
        welcome = hello(host, type="host", name="Sala Uno", pin="2468",
                        label="Tablet")
        assert welcome["type"] == "welcome" and welcome["host"] is True
        assert welcome["token"] and welcome["limits"]["pin_digits"] == 4
        until(host, "roster")
        with client.websocket_connect("/api/share") as viewer:
            joined = hello(viewer, type="join", name="sala uno",
                           pin="2468", label="Proyector")
            assert joined["host"] is False
            roster = until(viewer, "roster")
            assert [d["sends"] for d in roster["devices"]] == [True, False]
            asked = until(host, "need_snapshot")
            assert asked["device"] == joined["device"]
            # A change made before its page reached the device is not
            # sent to it: it would apply to nothing.
            host.send_text(json.dumps({"type": "m", "body": {"k": "d"}}))
            host.send_text(json.dumps({
                "type": "m", "snap": True, "to": joined["device"],
                "body": {"k": "s", "html": "<p>x</p>"}}))
            host.send_text(json.dumps({"type": "m", "body": {"k": "d2"}}))
            assert until(viewer, "m")["body"]["k"] == "s"
            assert until(viewer, "m")["body"]["k"] == "d2"

            # Watch only: the server drops the action.
            viewer.send_text(json.dumps({"type": "in", "body": {"k": "x"}}))
            viewer.send_text(json.dumps({"type": "ping"}))
            until(viewer, "pong")
            host.send_text(json.dumps({"type": "ping"}))
            assert host.receive_json()["type"] == "pong"

            host.send_text(json.dumps({
                "type": "set", "device": joined["device"], "sends": True}))
            until(viewer, "roster")
            viewer.send_text(json.dumps({
                "type": "in", "body": {"k": "click", "node": 3}}))
            action = until(host, "in")
            assert action == {"type": "in", "from": joined["device"],
                              "body": {"k": "click", "node": 3}}

            host.send_text(json.dumps({"type": "leave"}))
            assert until(viewer, "ended")["reason"] == "stopped"
            assert closed_with(viewer) == 4410


def test_a_dropped_link_comes_back_without_the_password(
    client: TestClient,
) -> None:
    with client.websocket_connect("/api/share") as host:
        hello(host, type="host", name="Sala", pin="1357", label="Tablet")
        with client.websocket_connect("/api/share") as viewer:
            token = hello(viewer, type="join", name="Sala", pin="1357",
                          label="PC")["token"]
            until(host, "need_snapshot")
        roster = until(host, "roster")
        assert [d["online"] for d in roster["devices"]] == [True, False]
        with client.websocket_connect("/api/share") as viewer:
            back = hello(viewer, type="resume", token=token)
            assert back["type"] == "welcome" and "token" not in back
            until(host, "need_snapshot")


def test_unknown_name_and_wrong_password_close_alike_then_lock(
    client: TestClient,
) -> None:
    with client.websocket_connect("/api/share") as host:
        hello(host, type="host", name="Sala", pin="1357", label="Tablet")
        answers = []
        for name, pin in [("Nadie", "1357"), ("Sala", "0000")]:
            with client.websocket_connect("/api/share") as ws:
                error = hello(ws, type="join", name=name, pin=pin, label="x")
                answers.append((error, closed_with(ws)))
        assert answers[0] == answers[1]
        assert answers[0][0]["code"] == "no_match" and answers[0][1] == 4404
        for pin in ["0001", "0002", "0003"]:
            with client.websocket_connect("/api/share") as ws:
                hello(ws, type="join", name="Sala", pin=pin, label="x")
        with client.websocket_connect("/api/share") as ws:
            error = hello(ws, type="join", name="Sala", pin="1357", label="x")
            assert error["code"] == "too_many_attempts"
            assert error["retry_after"] == 60
            assert closed_with(ws) == 4429


@pytest.mark.parametrize("message,code", [
    ({"type": "join", "name": "Sala", "pin": "12"}, 4400),
    ({"type": "dance"}, 4400),
    ({"type": "resume", "token": "nope"}, 4410),
])
def test_bad_hellos_are_refused(
    client: TestClient, message: Dict[str, Any], code: int
) -> None:
    with client.websocket_connect("/api/share") as ws:
        assert hello(ws, **message)["type"] == "error"
        assert closed_with(ws) == code


def test_a_message_too_large_closes_the_device(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with client.websocket_connect("/api/share") as host:
        hello(host, type="host", name="Sala", pin="1357", label="Tablet")
        with client.websocket_connect("/api/share") as viewer:
            hello(viewer, type="join", name="Sala", pin="1357", label="PC")
            viewer.send_text(json.dumps({
                "type": "in", "body": "x" * (20 * 1024)}))
            assert until(viewer, "error")["code"] == "too_large"
            assert closed_with(viewer) == 4413
    assert share_ws.CLOSE_CODES["too_large"] == 4413
