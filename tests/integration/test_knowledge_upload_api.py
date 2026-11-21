"""Uploading documents through the RAG's ``/knowledge/domains`` routes."""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import KnowledgeRoutes
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.adapters.outbound.fs_document_store import (
    FileSystemDocumentStore,
    check_document_id,
)
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.thread_job_queue import ThreadJobQueue
from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.bootstrap import KnowledgeIndexFactory, utc_now
from src.config import KnowledgeSettings, Settings
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.documents import KnowledgeDocument
from tests.fakes import FakeEmbedder
from tests.integration.test_knowledge_api import knowledge_client
from tests.integration.test_knowledge_domains import write
from tests.integration.test_transcript_ingest import AUTH
from tests.integration.test_layout_extraction import FakeLayout
from tests.knowledge_samples import make_docx, make_image, make_pdf

App = Tuple[TestClient, ThreadJobQueue]


@pytest.fixture()
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> App:
    """Return a client with the document routes and its worker."""
    return make_app(tmp_path, monkeypatch, DocumentExtractors())


def make_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    extractor: DocumentExtractors,
) -> App:
    """Build the routes over ``extractor`` in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    k = KnowledgeSettings(max_upload_bytes=200_000)
    library: KnowledgeLibrary
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
        k.max_chunk_size,
        extra_documents=lambda d: library.pending_documents(d),
    )
    queue = ThreadJobQueue()
    library = KnowledgeLibrary(
        registry, FileSystemDocumentStore(k.data_dir), extractor, queue,
        k.max_upload_bytes, utc_now,
    )
    routes = KnowledgeRoutes(KnowledgeQueryService(registry), library)
    return knowledge_client(tmp_path, registry, routes), queue


def create(client: TestClient, domain: str = "acme") -> Dict[str, Any]:
    """Create a domain and return its row."""
    response = client.post("/knowledge/domains", headers=AUTH, json={
        "id": domain, "title": "Acme soporte", "language": "es",
    })
    assert response.status_code == 201, response.text
    row: Dict[str, Any] = response.json()
    return row


def upload(
    client: TestClient, name: str, data: bytes,
    metadata: Any = None, domain: str = "acme",
) -> Any:
    """Upload ``data`` as ``name`` with optional metadata."""
    form = {} if metadata is None else {"metadata": json.dumps(metadata)}
    return client.post(
        f"/knowledge/domains/{domain}/documents", headers=AUTH,
        files={"file": (name, data, "application/octet-stream")}, data=form,
    )


def documents(client: TestClient, domain: str = "acme") -> List[Any]:
    """Return the domain's documents."""
    response = client.get(f"/knowledge/domains/{domain}/documents",
                          headers=AUTH)
    assert response.status_code == 200
    found: List[Any] = response.json()
    return found


def ask(client: TestClient, text: str, **filters: List[str]) -> List[str]:
    """Return the ids of the chunks a query finds."""
    response = client.post("/knowledge/query", headers=AUTH, json={
        "schema": "knowledge.query.v1", "request_id": "r",
        "domain": "acme", "text": text, "filters": filters,
    })
    assert response.status_code == 200, response.text
    return [c["id"] for c in response.json()["chunks"]]


def test_domain_rows(app: App) -> None:
    """Created domains are listed with the contract's fields."""
    client, _ = app
    row = create(client)
    assert row == {
        "id": "acme", "title": "Acme soporte", "language": "es",
        "documents": 0, "chunks": 0, "index_version": None,
        "updated_at": None,
    }
    assert client.get("/knowledge/domains", headers=AUTH).json() == [row]
    assert client.post("/knowledge/domains", headers=AUTH, json={
        "id": "acme", "title": "x"}).status_code == 409
    for bad in ({"id": "Bad", "title": "x"}, {"id": "a"},
                {"id": "a", "title": "x", "extra": 1}):
        assert client.post("/knowledge/domains", headers=AUTH,
                           json=bad).status_code == 422


def test_upload_index_relabel_and_delete(app: App) -> None:
    """The whole life of a document, from upload to deletion."""
    client, queue = app
    create(client)
    payload = make_docx()
    response = upload(client, "../../Manual devoluciones.docx", payload,
                      {"intent": "refund", "tags": ["faq"]})
    assert response.status_code == 202, response.text
    document = response.json()
    assert document["status"] == "queued"
    assert document["name"] == "Manual devoluciones.docx"
    assert document["type"] == "docx"
    assert set(document) == {"id", "name", "type", "size", "pages", "chunks",
                             "status", "error", "metadata", "uploaded_at"}
    assert set(document["metadata"]) == {"category", "intent",
                                         "script_phase", "tags"}
    queue.join()
    [done] = documents(client)
    assert done["status"] == "indexed" and done["chunks"] > 0
    stored = Path(f"data/knowledge/acme/originals/{done['id']}.docx")
    assert stored.read_bytes() == payload
    assert Path(f"data/knowledge/acme/extracted/{done['id']}.md").is_file()
    assert ask(client, "devolver treinta", intent=["refund"])
    assert ask(client, "devolver treinta", intent=["pricing"]) == []
    row = client.get("/knowledge/domains", headers=AUTH).json()[0]
    assert (row["documents"], row["chunks"]) == (1, done["chunks"])
    assert row["index_version"] and row["updated_at"]

    patched = client.patch(
        f"/knowledge/domains/acme/documents/{done['id']}", headers=AUTH,
        json={"metadata": {"intent": "pricing"}},
    )
    assert patched.status_code == 200 and patched.json()["status"] == \
        "queued"
    queue.join()
    assert ask(client, "devolver treinta", intent=["pricing"])
    assert documents(client)[0]["metadata"]["tags"] == []

    gone = client.delete(f"/knowledge/domains/acme/documents/{done['id']}",
                         headers=AUTH)
    assert gone.status_code == 204
    queue.join()
    assert documents(client) == []
    assert not stored.exists()
    assert client.post("/knowledge/query", headers=AUTH, json={
        "schema": "knowledge.query.v1", "request_id": "r",
        "domain": "acme", "text": "devolver"}).status_code == 404


def test_scanned_pdf_fails_with_a_message(app: App) -> None:
    """A PDF without text ends ``failed`` with the reason."""
    client, queue = app
    create(client)
    assert upload(client, "scan.pdf", make_pdf([""])).status_code == 202
    queue.join()
    [document] = documents(client)
    assert document["status"] == "failed"
    assert "OCR" in document["error"] and document["chunks"] == 0
    good = upload(client, "ok.pdf", make_pdf(["Refunds take five days."]))
    assert good.status_code == 202
    queue.join()
    statuses = {d["name"]: d["status"] for d in documents(client)}
    assert statuses == {"scan.pdf": "failed", "ok.pdf": "indexed"}
    pages = {d["name"]: d["pages"] for d in documents(client)}
    assert pages == {"scan.pdf": None, "ok.pdf": 1}


def test_images_and_scans_are_read_with_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a layout reader a scanned PDF and a PNG are indexed with pages."""
    pages = {1: "Cinturón de seguridad obligatorio", 2: "| Vía | Límite |"}
    client, queue = make_app(
        tmp_path, monkeypatch, DocumentExtractors(FakeLayout(pages)))
    create(client)
    assert upload(client, "scan.pdf", make_pdf(["", ""])).status_code == 202
    assert upload(client, "foto.png", make_image("png")).status_code == 202
    queue.join()
    rows = {d["name"]: d for d in documents(client)}
    assert rows["scan.pdf"]["status"] == "indexed"
    assert rows["scan.pdf"]["pages"] == 2
    assert rows["foto.png"]["status"] == "indexed"
    assert ask(client, "cinturón seguridad")


def test_images_are_refused_without_ocr(app: App) -> None:
    """Without the ``ocr`` group an image upload is a 415 saying why."""
    client, _ = app
    create(client)
    response = upload(client, "foto.png", make_image("png"))
    assert response.status_code == 415
    assert "OCR" in response.json()["detail"]


@pytest.mark.parametrize("name, data, metadata, status", [
    ("virus.pdf", b"MZ\x90\x00", None, 415),
    ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", None, 415),
    ("fake.docx", b"PK\x03\x04 not a zip", None, 415),
    ("a.txt", b"x" * 200_001, None, 413),
    ("a.txt", b"", None, 422),
    ("a.txt", b"hola", {"colour": "red"}, 422),
    ("a.txt", b"hola", {"category": "chisme"}, 422),
])
def test_upload_refusals(
    app: App, name: str, data: bytes, metadata: Any, status: int
) -> None:
    """Bad files and metadata are refused at once, nothing is stored."""
    client, _ = app
    create(client)
    response = upload(client, name, data, metadata)
    assert response.status_code == status, response.text
    assert documents(client) == []


def test_upload_envelope_errors(app: App) -> None:
    """No file, not multipart, unknown domain, duplicates, no token."""
    client, queue = app
    create(client)
    url = "/knowledge/domains/acme/documents"
    assert client.post(url, headers=AUTH, data={"x": "1"}).status_code == 422
    assert client.post(url, headers=AUTH, json={}).status_code == 422
    assert client.post(url, headers=AUTH, files={
        "file": ("a.txt", b"x")}, data={"metadata": "{bad"}
    ).status_code == 422
    assert upload(client, "a.txt", b"x", domain="nope").status_code == 404
    assert upload(client, "a.txt", b"hola").status_code == 202
    assert upload(client, "copia.txt", b"hola").status_code == 202
    assert upload(client, "a.txt", b"hola").status_code == 409
    assert client.post(url, files={"file": ("b.txt", b"x")}).status_code \
        == 401
    big = {**AUTH, "content-length": str(300_000 + 2 * 1024 * 1024)}
    assert client.post(url, headers=big, content=b"x").status_code == 413
    queue.join()


def test_document_routes_refuse_bad_ids(app: App) -> None:
    """Ids that could leave the domain's folders are refused."""
    client, _ = app
    create(client)
    base = "/knowledge/domains/acme/documents"
    for bad in ("%2E%2E", "a%2Fb", "UPPER"):
        assert client.delete(f"{base}/{bad}", headers=AUTH).status_code \
            in (404, 422)
    assert client.patch(f"{base}/missing-1", headers=AUTH, json={
        "metadata": {}}).status_code == 404
    assert client.patch(f"{base}/missing-1", headers=AUTH,
                        json={"intent": "x"}).status_code == 422
    assert client.get("/knowledge/domains/Bad/documents",
                      headers=AUTH).status_code == 422
    with pytest.raises(InvalidInputError):
        check_document_id("../x")


def test_delete_domains(app: App) -> None:
    """Created domains go with their files; authored ones stay."""
    client, queue = app
    create(client)
    upload(client, "a.txt", b"hola")
    queue.join()
    assert client.delete("/knowledge/domains/acme",
                         headers=AUTH).status_code == 204
    assert not Path("data/knowledge/acme").exists()
    assert not Path("data/processed/domains/acme").exists()
    assert client.delete("/knowledge/domains/acme",
                         headers=AUTH).status_code == 404
    write(Path("knowledge/domains/repo/manifest.yaml"),
          "id: repo\ntitle: In git\n")
    refused = client.delete("/knowledge/domains/repo", headers=AUTH)
    assert refused.status_code == 422 and "authored" in refused.text


def test_store_records_and_errors(tmp_path: Path) -> None:
    """Records round-trip; missing and corrupt ones are reported."""
    store = FileSystemDocumentStore(str(tmp_path))
    document = KnowledgeDocument(id="a-1", name="a.txt", type="txt", size=1,
                                 uploaded_at="2026-11-21T10:00:00+00:00")
    assert store.list("d") == []
    store.save("d", document)
    assert store.get("d", "a-1") == document
    with pytest.raises(NotFoundError):
        store.original("d", document)
    (tmp_path / "d/documents/b-2.json").write_text("{")
    with pytest.raises(Exception, match="unreadable"):
        store.list("d")


def test_job_queue_survives_failures() -> None:
    """A failing job is logged and the next one still runs."""
    queue = ThreadJobQueue()
    ran: List[str] = []
    done = threading.Event()

    def boom() -> None:
        """Fail."""
        raise RuntimeError("boom")

    def record() -> None:
        """Note that it ran."""
        ran.append("next")
        done.set()

    queue.submit(boom)
    queue.submit(record)
    assert done.wait(5)
    queue.join()
    assert ran == ["next"]
