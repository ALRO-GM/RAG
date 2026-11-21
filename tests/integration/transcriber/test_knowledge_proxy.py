"""``/api/knowledge/...``: the RAG's knowledge routes, forwarded."""

import dataclasses
import http.server
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tests.integration.test_knowledge_upload_api import app as rag_app
from tests.integration.test_transcript_ingest import TOKEN as RAG_TOKEN
from tests.integration.transcriber.test_analysis_judge import (
    CODE,
    TOKEN,
    StandInRag,
)
from tests.knowledge_samples import make_docx
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.http_knowledge import (
    HttpKnowledge,
    detail_of,
)
from transcriber.application.knowledge import KnowledgeAccess
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    PublishError,
    RagUnavailableError,
    StateError,
    TooLargeError,
    UnsupportedMediaError,
)

AUTH = {"X-Access-Code": CODE}
__all__ = ["rag_app"]


@pytest.fixture()
def rag() -> Iterator[Tuple[str, StandInRag]]:
    """Serve a stand-in RAG that answers every method."""
    stand_in = StandInRag()
    stand_in.body = b'[{"id": "acme"}]'
    base = stand_in.handler()

    class Handler(base):  # type: ignore[valid-type, misc]
        """Answer POST and PATCH like the other methods."""

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            """Answer a POST."""
            self._answer()

        def do_PATCH(self) -> None:  # noqa: N802 - http.server's name
            """Answer a PATCH."""
            self._answer()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}/rag", stand_in
    finally:
        server.shutdown()
        server.server_close()


def test_the_adapter_forwards_with_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    """Paths, bodies, content types and the token reach the RAG."""
    url, stand_in = rag
    port = HttpKnowledge(url, lambda: TOKEN)
    assert port.call("GET", "domains") == [{"id": "acme"}]
    port.call("POST", "domains/acme/documents", b"--b\r\n", "multipart/x")
    stand_in.status, stand_in.body = 204, b""
    assert port.call("DELETE", "domains/acme") is None
    assert [(r["method"], r["path"]) for r in stand_in.requests] == [
        ("GET", "/rag/knowledge/domains"),
        ("POST", "/rag/knowledge/domains/acme/documents"),
        ("DELETE", "/rag/knowledge/domains/acme"),
    ]
    assert stand_in.requests[1]["body"] == b"--b\r\n"
    assert {r["auth"] for r in stand_in.requests} == {f"Bearer {TOKEN}"}


@pytest.mark.parametrize("status, error", [
    (400, InvalidInputError), (422, InvalidInputError),
    (404, NotFoundError), (409, StateError), (413, TooLargeError),
    (415, UnsupportedMediaError), (401, RagUnavailableError),
    (500, RagUnavailableError), (302, RagUnavailableError),
])
def test_refusals_keep_the_rag_message(
    rag: Tuple[str, StandInRag], status: int, error: type,
) -> None:
    """The RAG's ``detail`` is passed on; the token never is."""
    url, stand_in = rag
    stand_in.status = status
    stand_in.body = json.dumps({"detail": "Tipo no admitido"}).encode()
    stand_in.location = "/elsewhere"
    with pytest.raises(error) as caught:
        HttpKnowledge(url, lambda: TOKEN).call("GET", "domains")
    assert TOKEN not in str(caught.value)
    if error is not RagUnavailableError:
        assert str(caught.value) == "Tipo no admitido"
    assert len(stand_in.requests) == 1


def test_bad_answers_and_no_rag(rag: Tuple[str, StandInRag]) -> None:
    """Non-JSON answers, a missing token and no RAG are reported."""
    url, stand_in = rag
    stand_in.body = b"<html>"
    with pytest.raises(RagUnavailableError, match="JSON"):
        HttpKnowledge(url, lambda: TOKEN).call("GET", "domains")

    def missing() -> str:
        """No token file."""
        raise PublishError("no token")

    with pytest.raises(RagUnavailableError, match="no token"):
        HttpKnowledge(url, missing).call("GET", "domains")
    with pytest.raises(RagUnavailableError, match="Cannot reach"):
        HttpKnowledge("http://127.0.0.1:9", lambda: TOKEN, 1).call(
            "GET", "domains")
    assert detail_of(b'{"detail": [1]}') == ""
    assert detail_of(b"\xff") == ""
    with pytest.raises(RagUnavailableError, match="no knowledge"):
        KnowledgeAccess(None).domains()


class FakeKnowledge:
    """The port, recording the calls."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.calls: List[Tuple[Any, ...]] = []

    def call(
        self, method: str, path: str, body: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> Any:
        """Record and answer like the RAG."""
        self.calls.append((method, path, body, content_type))
        if path == "domains/missing/documents":
            raise NotFoundError("No knowledge domain 'missing'.")
        if method == "DELETE":
            return None
        return {"ok": path}


def client(tmp_path: Path, port: Optional[FakeKnowledge]) -> TestClient:
    """The transcriber API with ``port`` behind its knowledge routes."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    container = dataclasses.replace(container,
                                    knowledge=KnowledgeAccess(port))
    return TestClient(create_app(container, CODE))


def test_routes_need_the_code_check_ids_and_forward(tmp_path: Path) -> None:
    """Every route is guarded; ids are checked before forwarding."""
    port = FakeKnowledge()
    api = client(tmp_path, port)
    routes = [
        ("GET", "/api/knowledge/domains", 200),
        ("POST", "/api/knowledge/domains", 201),
        ("DELETE", "/api/knowledge/domains/acme", 204),
        ("GET", "/api/knowledge/domains/acme/documents", 200),
        ("PATCH", "/api/knowledge/domains/acme/documents/doc-1", 200),
        ("DELETE", "/api/knowledge/domains/acme/documents/doc-1", 204),
        ("POST", "/api/knowledge/query", 200),
    ]
    for method, path, _ in routes:
        assert api.request(method, path).status_code == 401
    assert port.calls == []
    for method, path, status in routes:
        answer = api.request(method, path, headers=AUTH, content=b"{}")
        assert answer.status_code == status, (path, answer.text)
    assert [c[:2] for c in port.calls] == [
        ("GET", "domains"), ("POST", "domains"), ("DELETE", "domains/acme"),
        ("GET", "domains/acme/documents"),
        ("PATCH", "domains/acme/documents/doc-1"),
        ("DELETE", "domains/acme/documents/doc-1"), ("POST", "query"),
    ]
    upload = api.post("/api/knowledge/domains/acme/documents", headers=AUTH,
                      files={"file": ("a.txt", b"hola")})
    assert upload.status_code == 202
    method, path, body, kind = port.calls[-1]
    assert kind is not None and kind.startswith("multipart/form-data")
    assert body is not None and b"hola" in body
    not_multipart = api.post("/api/knowledge/domains/acme/documents",
                             headers=AUTH, json={})
    assert not_multipart.status_code == 400
    for bad in ("/api/knowledge/domains/Bad/documents",
                "/api/knowledge/domains/a%20b/documents"):
        assert api.get(bad, headers=AUTH).json()["code"] == "invalid_input"
    bad_doc = api.delete("/api/knowledge/domains/acme/documents/UP",
                         headers=AUTH)
    assert bad_doc.status_code == 400
    missing = api.get("/api/knowledge/domains/missing/documents",
                      headers=AUTH)
    assert missing.status_code == 404
    huge = api.post("/api/knowledge/query", headers=AUTH,
                    content=b"x" * (64 * 1024 + 1))
    assert huge.status_code == 413
    off = client(tmp_path, None).get("/api/knowledge/domains", headers=AUTH)
    assert (off.status_code, off.json()["code"]) == (503, "rag_unavailable")


def test_the_container_links_knowledge_with_the_http_publisher(
    tmp_path: Path,
) -> None:
    """Only the HTTP publisher gives a knowledge link."""
    linked = build_container(TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "data"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": "http://127.0.0.1:9/rag"},
    }), environ={})
    port = linked.knowledge._port_or_none
    assert isinstance(port, HttpKnowledge)
    assert port.url == "http://127.0.0.1:9/rag/knowledge/"


def free_port() -> int:
    """Return a free loopback port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def test_end_to_end_upload_through_the_transcriber(
    tmp_path: Path, rag_app: Any,
) -> None:
    """Web → transcriber → real RAG app: upload, index, list, query."""
    rag_client, queue = rag_app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(
        rag_client.app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        container = build_container(TranscriberSettings(
            data_dir=tmp_path / "data", runtime="fake",
        ))
        container = dataclasses.replace(container, knowledge=KnowledgeAccess(
            HttpKnowledge(f"http://127.0.0.1:{port}", lambda: RAG_TOKEN)
        ))
        web = TestClient(create_app(container, CODE))
        made = web.post("/api/knowledge/domains", headers=AUTH, json={
            "id": "acme", "title": "Acme", "language": "es"})
        assert made.status_code == 201, made.text
        sent = web.post(
            "/api/knowledge/domains/acme/documents", headers=AUTH,
            files={"file": ("Manual.docx", make_docx())},
            data={"metadata": json.dumps({"intent": "refund"})},
        )
        assert sent.status_code == 202, sent.text
        assert sent.json()["status"] == "queued"
        queue.join()
        [document] = web.get("/api/knowledge/domains/acme/documents",
                             headers=AUTH).json()
        assert document["status"] == "indexed"
        found = web.post("/api/knowledge/query", headers=AUTH, json={
            "schema": "knowledge.query.v1", "request_id": "r",
            "domain": "acme", "text": "devolver",
            "filters": {"intent": ["refund"]},
        })
        assert found.status_code == 200 and found.json()["chunks"]
        refused = web.post(
            "/api/knowledge/domains/acme/documents", headers=AUTH,
            files={"file": ("virus.pdf", b"MZ\x90")},
        )
        assert refused.status_code == 415
        assert refused.json()["code"] == "unsupported_media_type"
        assert "PDF" in refused.json()["message"]
    finally:
        server.should_exit = True
        thread.join(5)
