"""A session's analysis: forwarded from the web to the RAG (#88)."""

import dataclasses
import http.server
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.test_analysis_judge import (
    CODE,
    TOKEN,
    StandInRag,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.http_session_analysis import (
    HttpSessionAnalysis,
)
from transcriber.application.analysis import SessionAnalysisAccess
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
    NotFoundError,
    PublishError,
    StateError,
)

AUTH = {"X-Access-Code": CODE}
RUN = {"session_id": "s1", "status": "done", "steps": [], "active": False}


@pytest.fixture()
def rag() -> Iterator[Tuple[str, StandInRag]]:
    """Serve a stand-in RAG on loopback that also answers POST."""
    stand_in = StandInRag()
    stand_in.body = json.dumps(RUN).encode()
    base = stand_in.handler()

    class Handler(base):  # type: ignore[valid-type, misc]
        """Answer POST like the other methods."""

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            """Answer a POST."""
            self._answer()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}/rag", stand_in
    finally:
        server.shutdown()
        server.server_close()


def test_the_adapter_forwards_every_call_with_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    port = HttpSessionAnalysis(url, lambda: TOKEN)
    assert port.start("s1", True, False) == RUN
    assert port.status("s1") == RUN
    assert port.result("s1", "sentences") == RUN
    stand_in.body = b"<!doctype html><p>hola</p>"
    assert port.report("s1", 2) == "<!doctype html><p>hola</p>"
    paths = [(r["method"], r["path"]) for r in stand_in.requests]
    assert paths == [
        ("POST", "/rag/analysis/sessions/s1"),
        ("GET", "/rag/analysis/sessions/s1"),
        ("GET", "/rag/analysis/sessions/s1/steps/sentences"),
        ("GET", "/rag/analysis/sessions/s1/reports/2"),
    ]
    assert json.loads(stand_in.requests[0]["body"]) == {
        "restart": True, "synthetic": False,
    }
    assert {r["auth"] for r in stand_in.requests} == {f"Bearer {TOKEN}"}


@pytest.mark.parametrize("status, error", [
    (404, NotFoundError), (409, StateError), (400, InvalidInputError),
    (422, InvalidInputError), (500, AnalysisUnavailableError),
    (302, AnalysisUnavailableError),
])
def test_the_rag_refusals_become_domain_errors(
    rag: Tuple[str, StandInRag], status: int, error: type,
) -> None:
    url, stand_in = rag
    stand_in.status, stand_in.location = status, "/elsewhere"
    with pytest.raises(error) as caught:
        HttpSessionAnalysis(url, lambda: TOKEN).status("s1")
    assert TOKEN not in str(caught.value)
    assert len(stand_in.requests) == 1, "a redirect is never followed"


def test_the_adapter_reports_bad_answers_and_no_rag(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    port = HttpSessionAnalysis(url, lambda: TOKEN)
    stand_in.body = b"[1]"
    with pytest.raises(AnalysisUnavailableError, match="JSON"):
        port.status("s1")
    stand_in.body = b"\xff\xfe"
    with pytest.raises(AnalysisUnavailableError, match="text"):
        port.report("s1", 1)

    def missing() -> str:
        """No token file."""
        raise PublishError("no token")

    with pytest.raises(AnalysisUnavailableError, match="no token"):
        HttpSessionAnalysis(url, missing).status("s1")
    with pytest.raises(AnalysisUnavailableError, match="Cannot reach"):
        HttpSessionAnalysis("http://127.0.0.1:9", lambda: TOKEN, 1).status(
            "s1"
        )


class FakeAnalysis:
    """The port, in memory."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.calls: List[Tuple[Any, ...]] = []
        self.running = False

    def start(
        self, session_id: str, restart: bool, synthetic: bool
    ) -> Dict[str, Any]:
        """Start, or refuse while running."""
        if self.running:
            raise StateError("already running")
        self.running = True
        self.calls.append(("start", session_id, restart, synthetic))
        return {"session_id": session_id, "active": True}

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the run."""
        if session_id == "none":
            raise NotFoundError("No analysis.")
        return {**RUN, "session_id": session_id}

    def result(self, session_id: str, step: str) -> Any:
        """Return a step's data."""
        self.calls.append(("result", session_id, step))
        return {"step": step}

    def report(self, session_id: str, position: int) -> str:
        """Return a page."""
        return f"<!doctype html><p>{position}</p>"


def test_the_use_case_checks_ids_steps_and_numbers() -> None:
    port = FakeAnalysis()
    access = SessionAnalysisAccess(port)
    assert access.result("s1", "acts") == {"step": "acts"}
    for call in (
        lambda: access.status("../x"), lambda: access.status(".x"),
        lambda: access.result("s1", "run"),
        lambda: access.result("s1", "../run"),
        lambda: access.report("s1", 0), lambda: access.report("s1", 100),
    ):
        with pytest.raises(InvalidInputError):
            call()
    with pytest.raises(AnalysisUnavailableError):
        SessionAnalysisAccess(None).status("s1")


def client(tmp_path: Path, port: Optional[FakeAnalysis]) -> TestClient:
    """The transcriber API with ``port`` behind its analysis routes."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    container = dataclasses.replace(
        container, session_analysis=SessionAnalysisAccess(port)
    )
    return TestClient(create_app(container, CODE))


def test_the_routes_need_the_code_and_forward(tmp_path: Path) -> None:
    port = FakeAnalysis()
    api = client(tmp_path, port)
    for method, path in (
        ("POST", "/api/analysis/sessions/s1"),
        ("GET", "/api/analysis/sessions/s1"),
        ("GET", "/api/analysis/sessions/s1/steps/sentences"),
        ("GET", "/api/analysis/sessions/s1/reports/1"),
    ):
        assert api.request(method, path).status_code == 401
    assert port.calls == []
    started = api.post("/api/analysis/sessions/s1", json={"synthetic": True},
                       headers=AUTH)
    assert started.status_code == 202
    assert port.calls == [("start", "s1", False, True)]
    busy = api.post("/api/analysis/sessions/s1", headers=AUTH)
    assert busy.status_code == 409 and busy.json()["code"] == "wrong_state"
    assert api.get("/api/analysis/sessions/s1", headers=AUTH).json()[
        "session_id"] == "s1"
    assert api.get("/api/analysis/sessions/none",
                   headers=AUTH).status_code == 404
    step = api.get("/api/analysis/sessions/s1/steps/candidates",
                   headers=AUTH)
    assert step.json() == {"step": "candidates"}
    assert api.get("/api/analysis/sessions/s1/steps/run",
                   headers=AUTH).status_code == 400
    extra = api.post("/api/analysis/sessions/s2", json={"x": 1},
                     headers=AUTH)
    assert extra.status_code == 422
    report = api.get("/api/analysis/sessions/s1/reports/3", headers=AUTH)
    assert report.status_code == 200
    assert report.text == "<!doctype html><p>3</p>"
    csp = report.headers["content-security-policy"]
    assert csp.startswith("sandbox;") and "script-src" not in csp
    assert report.headers["content-disposition"].startswith("attachment;")
    assert report.headers["x-content-type-options"] == "nosniff"
    off = client(tmp_path, None).get(
        "/api/analysis/sessions/s1", headers=AUTH
    )
    assert off.status_code == 503
    assert off.json()["code"] == "analysis_unavailable"


def test_the_container_links_the_analysis_only_with_the_http_publisher(
    tmp_path: Path,
) -> None:
    plain = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    with pytest.raises(AnalysisUnavailableError):
        plain.session_analysis.status("s1")
    linked = build_container(TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "data"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": "http://127.0.0.1:9/rag"},
    }), environ={})
    port = linked.session_analysis._port_or_none
    assert isinstance(port, HttpSessionAnalysis)
    assert port.url == "http://127.0.0.1:9/rag/analysis/sessions/"
