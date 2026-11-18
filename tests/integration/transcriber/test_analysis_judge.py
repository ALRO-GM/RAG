"""The analysis judge setting: forwarded from the web to the RAG."""

import dataclasses
import http.server
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.http_analysis_judge import (
    HttpAnalysisJudge,
)
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)
from transcriber.application.analysis import AnalysisJudgeSetting
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
    PublishError,
)

CODE = "a-long-enough-access-code"
TOKEN = "r" * 40
STATUS = {"mode": "claude_code", "chosen": False, "claude_quota": "unknown",
          "modes": ["claude_code", "local", "both", "opencode"]}


class StandInRag:
    """A RAG answering ``/analysis/judge`` and recording the requests."""

    def __init__(self) -> None:
        """Answer 200 with :data:`STATUS` until told otherwise."""
        self.requests: List[Dict[str, Any]] = []
        self.status = 200
        self.body: bytes = json.dumps(STATUS).encode()
        self.location = ""

    def handler(self) -> type:
        """Return the request handler class."""
        rag = self

        class Handler(http.server.BaseHTTPRequestHandler):
            """Record and answer."""

            def _answer(self) -> None:
                """Record the request and send the scripted answer."""
                length = int(self.headers.get("Content-Length", "0"))
                rag.requests.append({
                    "method": self.command, "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": self.rfile.read(length),
                })
                self.send_response(rag.status)
                if rag.location:
                    self.send_header("Location", rag.location)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(rag.body)))
                self.end_headers()
                self.wfile.write(rag.body)

            def do_GET(self) -> None:  # noqa: N802 - http.server's name
                """Answer a GET."""
                self._answer()

            def do_PUT(self) -> None:  # noqa: N802 - http.server's name
                """Answer a PUT."""
                self._answer()

            def do_DELETE(self) -> None:  # noqa: N802 - http.server's name
                """Answer a DELETE."""
                self._answer()

            def log_message(self, *args: Any) -> None:
                """Keep the test output clean."""

        return Handler


@pytest.fixture()
def rag() -> Iterator[Tuple[str, StandInRag]]:
    """Serve a stand-in RAG on loopback."""
    stand_in = StandInRag()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), stand_in.handler()
    )
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}/rag", stand_in
    finally:
        server.shutdown()
        server.server_close()


def test_the_adapter_forwards_with_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    judge = HttpAnalysisJudge(url, lambda: TOKEN)
    assert judge.status() == STATUS
    assert judge.choose("local") == STATUS
    get, put = stand_in.requests
    assert (get["method"], get["path"]) == ("GET", "/rag/analysis/judge")
    assert put["method"] == "PUT" and json.loads(put["body"]) == {
        "mode": "local"
    }
    assert get["auth"] == f"Bearer {TOKEN}"


def test_redirects_are_not_followed_with_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    stand_in.status, stand_in.location = 302, "/elsewhere"
    with pytest.raises(AnalysisUnavailableError, match="302"):
        HttpAnalysisJudge(url, lambda: TOKEN).status()
    publisher = HttpTranscriptPublisher(
        url, "RAG_TOKEN", {"RAG_TOKEN": TOKEN}, retries=0,
    )
    with pytest.raises(PublishError, match="302"):
        publisher.retract("s1")
    assert [r["path"] for r in stand_in.requests] == [
        "/rag/analysis/judge", "/rag/transcripts/s1",
    ], "the redirect target was never asked"


def test_the_adapter_reports_every_failure_without_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    judge = HttpAnalysisJudge(url, lambda: TOKEN)
    stand_in.status = 401
    with pytest.raises(AnalysisUnavailableError, match="401") as caught:
        judge.status()
    assert TOKEN not in str(caught.value)
    stand_in.status, stand_in.body = 200, b"not json"
    with pytest.raises(AnalysisUnavailableError, match="JSON"):
        judge.status()
    stand_in.body = b"[1]"
    with pytest.raises(AnalysisUnavailableError, match="JSON"):
        judge.status()

    def missing() -> str:
        """No token file."""
        raise PublishError("no token")

    with pytest.raises(AnalysisUnavailableError, match="no token"):
        HttpAnalysisJudge(url, missing).status()
    with pytest.raises(AnalysisUnavailableError, match="Cannot reach"):
        HttpAnalysisJudge("http://127.0.0.1:9", lambda: TOKEN, 1).status()


class FakeJudge:
    """The port, in memory."""

    def __init__(self) -> None:
        """Start on the default judge."""
        self.mode = "claude_code"

    def status(self) -> Dict[str, Any]:
        """Return the status."""
        return {**STATUS, "mode": self.mode}

    def choose(self, mode: str) -> Dict[str, Any]:
        """Change the mode."""
        self.mode = mode
        return self.status()


def test_the_use_case_checks_the_mode_and_the_connection() -> None:
    port = FakeJudge()
    setting = AnalysisJudgeSetting(port)
    assert setting.choose("both")["mode"] == "both"
    with pytest.raises(InvalidInputError):
        setting.choose("api")
    with pytest.raises(AnalysisUnavailableError):
        AnalysisJudgeSetting(None).status()


def client(tmp_path: Path, judge: Optional[FakeJudge]) -> TestClient:
    """The transcriber API with ``judge`` behind its analysis routes."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    container = dataclasses.replace(
        container, analysis_judge=AnalysisJudgeSetting(judge)
    )
    return TestClient(create_app(container, CODE))


def test_the_routes_need_the_code_and_forward(tmp_path: Path) -> None:
    api = client(tmp_path, FakeJudge())
    assert api.get("/api/analysis/judge").status_code == 401
    assert api.put("/api/analysis/judge",
                   json={"mode": "local"}).status_code == 401
    auth = {"X-Access-Code": CODE}
    assert api.get("/api/analysis/judge", headers=auth).json()["mode"] == (
        "claude_code"
    )
    changed = api.put("/api/analysis/judge", json={"mode": "local"},
                      headers=auth)
    assert changed.status_code == 200 and changed.json()["mode"] == "local"
    wrong = api.put("/api/analysis/judge", json={"mode": "x"}, headers=auth)
    assert wrong.status_code == 400
    extra = api.put("/api/analysis/judge", json={"mode": "local", "x": 1},
                    headers=auth)
    assert extra.status_code == 422
    off = client(tmp_path, None).get("/api/analysis/judge", headers=auth)
    assert off.status_code == 503
    assert off.json()["code"] == "analysis_unavailable"


def test_the_container_forwards_only_with_the_http_publisher(
    tmp_path: Path,
) -> None:
    plain = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    with pytest.raises(AnalysisUnavailableError):
        plain.analysis_judge.status()
    linked = build_container(TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "data"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": "http://127.0.0.1:9/rag"},
    }), environ={})
    judge = linked.analysis_judge._judge
    assert isinstance(judge, HttpAnalysisJudge)
    assert judge.url == "http://127.0.0.1:9/rag/analysis/judge"
    with pytest.raises(AnalysisUnavailableError, match="token"):
        linked.analysis_judge.status()
