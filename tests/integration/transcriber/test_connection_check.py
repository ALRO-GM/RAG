"""The connection check's routes and health probe (#247)."""

import dataclasses
import http.server
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.test_analysis_judge import CODE
from transcriber.adapters.inbound.http_api import (
    UPLOAD_PROBE_BYTES,
    create_app,
)
from transcriber.adapters.outbound.http_health_probe import HttpHealthProbe
from transcriber.application.connection import (
    AVAILABLE,
    NOT_SET_UP,
    UNAVAILABLE,
    CheckServices,
    FixedProbe,
)
from transcriber.bootstrap import build_container, build_services
from transcriber.config import TranscriberSettings

AUTH = {"X-Access-Code": CODE}


class Health:
    """What the stand-in service answers on ``/health``."""

    def __init__(self) -> None:
        """Answer 200 at once until told otherwise."""
        self.status = 200
        self.wait = 0.0
        self.location = ""
        self.paths: List[str] = []


@pytest.fixture()
def service() -> Iterator[Tuple[str, Health]]:
    """Serve a stand-in health route on loopback."""
    health = Health()

    class Handler(http.server.BaseHTTPRequestHandler):
        """Answer as :class:`Health` says."""

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            """Answer a GET."""
            health.paths.append(self.path)
            time.sleep(health.wait)
            body = b'{"status": "ok", "fingerprint": "internal"}'
            try:
                self.send_response(health.status)
                if health.location:
                    self.send_header("Location", health.location)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the probe gave up waiting (the timeout case)

        def log_message(self, *args: Any) -> None:
            """Keep the test output clean."""

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}", health
    finally:
        server.shutdown()


def closed_port() -> int:
    """A loopback port nobody listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_the_probe_is_available_only_on_a_timely_200(
    service: Tuple[str, Health],
) -> None:
    url, health = service
    probe = HttpHealthProbe(url + "/health", timeout=0.5)
    assert probe.available()
    assert health.paths == ["/health"]
    health.status = 500
    assert not probe.available()
    health.status = 302
    health.location = url + "/elsewhere"
    assert not probe.available()
    assert "/elsewhere" not in health.paths
    health.status, health.location, health.wait = 200, "", 1.0
    assert not probe.available()


def test_the_probe_is_unavailable_on_a_closed_port() -> None:
    port = closed_port()
    assert not HttpHealthProbe(f"http://127.0.0.1:{port}/health").available()
    assert not HttpHealthProbe("not a url").available()


def test_build_services_asks_the_orchestrator_and_the_rag(
    service: Tuple[str, Health],
) -> None:
    url, health = service
    port = closed_port()
    check = build_services(
        {"TRANSCRIBER_ORCHESTRATOR_URL": url + "/"}, True,
        f"http://127.0.0.1:{port}/",
    )
    assert check() == {
        "transcription": AVAILABLE,
        "assistant": AVAILABLE,
        "knowledge": UNAVAILABLE,
    }
    assert health.paths == ["/health"]
    alone = build_services({}, False, None)
    assert alone.probes["knowledge"] is None
    assert isinstance(alone.probes["assistant"], HttpHealthProbe)
    assert alone.probes["assistant"].url == "http://127.0.0.1:8772/health"
    assert alone()["transcription"] == UNAVAILABLE


def client(tmp_path: Path, check: "CheckServices | None") -> TestClient:
    """The transcriber API with ``check`` as its services check."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    assert isinstance(container.services, CheckServices)
    container = dataclasses.replace(container, services=check)
    return TestClient(create_app(container, CODE))


def test_the_services_route_needs_the_code_and_says_one_word_each(
    tmp_path: Path,
) -> None:
    api = client(tmp_path, CheckServices({
        "transcription": FixedProbe(True), "assistant": FixedProbe(False),
    }))
    assert api.get("/api/connection").status_code == 401
    answer = api.get("/api/connection", headers=AUTH)
    assert answer.status_code == 200
    assert answer.headers["cache-control"] == "no-store"
    body: Dict[str, Any] = answer.json()
    assert body == {"services": {
        "transcription": AVAILABLE, "assistant": UNAVAILABLE,
        "knowledge": NOT_SET_UP,
    }}
    off = client(tmp_path, None).get("/api/connection", headers=AUTH)
    assert set(off.json()["services"].values()) == {NOT_SET_UP}


def test_the_upload_is_read_counted_and_limited(tmp_path: Path) -> None:
    api = client(tmp_path, None)
    body = bytes(64 * 1024)
    assert api.post("/api/connection/upload",
                    content=body).status_code == 401
    answer = api.post("/api/connection/upload", content=body, headers=AUTH)
    assert answer.status_code == 200
    assert answer.json() == {"bytes": len(body)}
    big = api.post("/api/connection/upload",
                   content=bytes(UPLOAD_PROBE_BYTES + 1), headers=AUTH)
    assert big.status_code == 413
    assert big.json()["code"] == "payload_too_large"
