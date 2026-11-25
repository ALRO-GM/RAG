"""Adapters of ``serve``: transcriber HTTP, event socket, transcript files."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
from websockets.exceptions import ConnectionClosed, InvalidURI
from websockets.frames import Close

from orchestrator.adapters.inbound import cli
from orchestrator.adapters.outbound.fs_transcripts import DirTranscripts
from orchestrator.adapters.outbound.http_transcriber import (
    STOP_TIMEOUT,
    HttpTranscriber,
    read_code,
)
from orchestrator.adapters.outbound.ws_events import WsEvents
from orchestrator.application.ports import (
    CancelToken,
    TranscriberUnavailable,
)
from tests.unit.orchestrator.fixtures import REPO


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_transcriber_is_asked_with_the_access_code() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/stop"):
            return httpx.Response(200, json={"state": "ended"})
        return httpx.Response(201, json={"id": "s", "state": "running"})

    port = HttpTranscriber("http://tr:8770/", "c0de", client=client(handler))
    assert port.start({"session_id": "s"}) == {"id": "s", "state": "running"}
    port.stop("s")
    assert [str(r.url) for r in seen] == [
        "http://tr:8770/api/live/sessions",
        "http://tr:8770/api/live/sessions/s/stop"]
    assert all(r.headers["x-access-code"] == "c0de" for r in seen)
    assert json.loads(seen[0].content) == {"session_id": "s"}


def test_transcriber_refusals_and_failures_are_one_exception() -> None:
    def busy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "busy", "message": "one"})

    with pytest.raises(TranscriberUnavailable, match="409 busy: one"):
        HttpTranscriber("http://tr", "c", client=client(busy)).start({})

    def text(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"oops")

    with pytest.raises(TranscriberUnavailable, match="500 error"):
        HttpTranscriber("http://tr", "c", client=client(text)).stop("s")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no", request=request)

    with pytest.raises(TranscriberUnavailable, match="ConnectError"):
        HttpTranscriber("http://tr", "c", client=client(down)).start({})

    def odd(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1])

    assert HttpTranscriber("http://tr", "c", client=client(odd)
                           ).start({}) == {}


def test_codes_are_read_from_files(tmp_path: Path) -> None:
    path = tmp_path / "code"
    path.write_text(" abc \n", encoding="utf-8")
    assert read_code(path) == "abc"
    path.write_text("", encoding="utf-8")
    assert read_code(path) is None
    assert read_code(tmp_path / "missing") is None


class FakeSocket:
    """A socket that hands out scripted messages, then closes."""

    def __init__(self, script: List[Any], stop: Optional[CancelToken] = None
                 ) -> None:
        self.script = script
        self.stop = stop
        self.closed = False

    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def recv(self, timeout: float) -> str:
        if not self.script:
            if self.stop is not None:
                self.stop.cancel()
            raise TimeoutError()
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return str(item)


def opener(socket: FakeSocket, seen: List[Dict[str, Any]]) -> Any:
    def open_(url: str, **kwargs: Any) -> FakeSocket:
        seen.append({"url": url, **kwargs})
        return socket
    return open_


def closed(code: int) -> ConnectionClosed:
    return ConnectionClosed(Close(code, "bye"), None)


def test_events_are_followed_until_a_normal_close() -> None:
    seen: List[Dict[str, Any]] = []
    socket = FakeSocket([TimeoutError(), '{"seq": 0}', "not json", "[1]",
                         '{"seq": 1}', closed(1000)])
    events = WsEvents("https://host:8770/", "c0de",
                      opener=opener(socket, seen))
    got = list(events.follow("s-1", CancelToken()))
    assert got == [{"seq": 0}, {"seq": 1}]
    assert seen[0]["url"] == "wss://host:8770/api/live/s-1/events"
    again: List[Dict[str, Any]] = []
    resumed = WsEvents("https://host:8770/", "c0de", opener=opener(
        FakeSocket([closed(1000)]), again))
    assert list(resumed.follow("s-1", CancelToken(), from_seq=7)) == []
    assert again[0]["url"] == \
        "wss://host:8770/api/live/s-1/events?from_seq=7"
    assert seen[0]["additional_headers"] == {"X-Access-Code": "c0de"}
    assert socket.closed


def test_event_failures_and_stops() -> None:
    def refuse(url: str, **kwargs: Any) -> Any:
        raise InvalidURI(url, "bad")

    with pytest.raises(TranscriberUnavailable):
        list(WsEvents("http://h", "c", opener=refuse).follow(
            "s", CancelToken()))
    with pytest.raises(TranscriberUnavailable, match="4404"):
        list(WsEvents("http://h", "c", opener=opener(
            FakeSocket([closed(4404)]), [])).follow("s", CancelToken()))
    error = '{"type": "error", "code": "not_found"}'
    with pytest.raises(TranscriberUnavailable, match="not_found"):
        list(WsEvents("http://h", "c", opener=opener(
            FakeSocket([error]), [])).follow("s", CancelToken()))
    stop = CancelToken()
    socket = FakeSocket(['{"seq": 0}'], stop)
    got = list(WsEvents("http://h", "c", opener=opener(socket, [])).follow(
        "s", stop))
    assert got == [{"seq": 0}]


def test_transcripts_are_listed_and_loaded_from_folders(
    tmp_path: Path
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    example = REPO / "examples/platform/sim-soporte-lumbre.transcript.v1.json"
    shutil.copy(example, second / example.name)
    shutil.copy(example, second / "dup.transcript.v1.json")
    (first / "dup.transcript.v1.json").write_text("{}", encoding="utf-8")
    store = DirTranscripts([tmp_path / "missing", first, second])
    listed = {t["id"]: t for t in store.list()}
    assert set(listed) == {"sim-soporte-lumbre"}
    assert listed["sim-soporte-lumbre"]["utterances"] == 27
    assert store.load("sim-soporte-lumbre").session.type == "soporte"
    for bad in ("dup", "../a/dup", "", "missing"):
        with pytest.raises(KeyError):
            store.load(bad)


def test_serve_listens_on_loopback_only_and_wires_the_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.is_loopback("localhost") and cli.is_loopback("::1")
    assert not cli.is_loopback("0.0.0.0") and not cli.is_loopback("x")
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2
    assert cli.main(["serve", "--token-file", str(tmp_path / "no")]) == 2
    code = tmp_path / "code"
    code.write_text("c0de", encoding="utf-8")
    token = tmp_path / "token"
    token.write_text("tok", encoding="utf-8")
    ran: List[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: ran.append(kw))
    status = cli.main([
        "serve", "--port", "18999", "--token-file", str(token),
        "--transcriber-code-file", str(code), "--knowledge", "fake",
        "--reasoner", "fake", "--log-dir", str(tmp_path / "log"),
        "--transcripts", str(REPO / "examples/platform"), "--tts", "none",
    ])
    assert status == 0
    assert ran == [{"host": "127.0.0.1", "port": 18999,
                    "log_level": "warning"}]
    args = cli.build_parser().parse_args([
        "serve", "--transcriber-code-file", str(code), "--knowledge", "fake",
        "--reasoner", "fake", "--log-dir", str(tmp_path / "log"),
        "--transcripts", str(REPO / "examples/platform"), "--tts", "fake",
    ])
    sessions = cli.build_sessions(args)
    assert "voicebot" in sessions.products
    assert sessions.transcriber is not None
    sessions.start_replay("d", "live_assist.support",
                          "sim-soporte-lumbre", 1.0)
    sessions.stop("d")
    found, finished = sessions.feed("d").read(10**6, 5.0)
    if not finished:
        sessions.feed("d").read(10**6, 5.0)
    assert sessions.get("d")["state"] == "ended"


@pytest.mark.parametrize("name", ["live_assist.support",
                                  "live_assist.interview"])
@pytest.mark.parametrize("audio", ["stereo", "mic"])
def test_the_start_body_is_what_the_transcriber_accepts(
    name: str, audio: str
) -> None:
    """The contract between the two services, checked on both sides."""
    from orchestrator.application.sessions import (
        live_speakers,
        transcriber_body,
    )
    from tests.unit.orchestrator.fixtures import product
    from transcriber.domain.live_session import LiveSessionRequest

    config = product(name)
    body = transcriber_body(config, "live-1", live_speakers(config, audio))
    request = LiveSessionRequest.model_validate(body)
    assert request.channels == (2 if audio == "stereo" else 1)
    assert request.product == name


def test_the_voicebot_start_body_is_what_the_transcriber_accepts() -> None:
    from orchestrator.application.sessions import (
        live_speakers,
        transcriber_body,
    )
    from tests.unit.orchestrator.fixtures import product
    from transcriber.domain.live_session import LiveSessionRequest

    config = product("voicebot")
    body = transcriber_body(config, "vb-1", live_speakers(config, "mic"))
    request = LiveSessionRequest.model_validate(body)
    assert request.channels == 1 and request.hang_ms == 400


def test_tts_choices(tmp_path: Path, capsys: pytest.CaptureFixture[str]
                     ) -> None:
    from orchestrator.adapters.inbound import cli
    from orchestrator.adapters.outbound.piper_tts import PiperTts
    from tests.unit.orchestrator.fixtures import product

    products = {"voicebot": product("voicebot")}
    parse = cli.build_parser().parse_args
    args = parse(["serve", "--tts", "piper", "--piper-python",
                  str(tmp_path / "none")])
    assert cli.build_tts(args, products) is None
    assert "voice bot off" in capsys.readouterr().err
    with pytest.raises(ValueError):
        cli.build_tts(parse(["serve", "--tts", "espeak"]), products)
    fake_python = tmp_path / "python"
    fake_python.write_text("", encoding="utf-8")
    args = parse(["serve", "--piper-python", str(fake_python),
                  "--voices-dir", str(tmp_path)])
    tts = cli.build_tts(args, products)
    assert isinstance(tts, PiperTts)
    tts.close()


def test_a_stop_waits_longer_than_the_transcriber_may_take() -> None:
    """#263: the transcriber's stop lets the source send what it holds
    (``stop_drain_seconds``) and then waits for the end (60 s); the
    orchestrator's client must not give up first."""
    import inspect

    from transcriber.application.live_session import LiveSessions
    from transcriber.domain.live_session import LiveServiceSettings

    worst = LiveServiceSettings().stop_drain_seconds + inspect.signature(
        LiveSessions.stop).parameters["timeout"].default
    assert STOP_TIMEOUT > worst
    waited: List[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        waited.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={})

    port = HttpTranscriber("http://tr", "c", client=client(handler))
    port.stop("s")
    port.start({})
    assert waited == [STOP_TIMEOUT, 40.0]
