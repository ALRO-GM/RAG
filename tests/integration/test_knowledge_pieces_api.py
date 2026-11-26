"""Uploads in pieces over the RAG's HTTP API, on the real file store (#282).

The routes, the upload area and the document store on disk; the limits
of the test app are 200 000 bytes for every kind (see
:func:`tests.integration.test_knowledge_upload_api.make_app`).
"""

import stat
from pathlib import Path
from typing import Any, Dict

import pytest

from src.adapters.outbound.fs_upload_area import (
    FileSystemUploadArea,
    UPLOADS_DIR,
)
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.knowledge.documents import document_id
from src.domain.knowledge.uploads import PIECE_BYTES, PartialUpload
from src.adapters.outbound.document_extractors import DocumentExtractors
from tests.integration.test_knowledge_upload_api import (
    App,
    create,
    documents,
    make_app,
)
from tests.integration.test_transcript_ingest import AUTH
from tests.knowledge_samples import make_docx, make_pdf

BASE = "/knowledge/domains/acme/uploads"


@pytest.fixture()
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> App:
    """Return a client with the upload routes and its worker."""
    return make_app(tmp_path, monkeypatch, DocumentExtractors())


def start(client: Any, name: str, size: int, **extra: Any) -> Any:
    """Announce a file."""
    return client.post(BASE, headers=AUTH,
                       json={"name": name, "size": size, **extra})


def piece(client: Any, upload: str, offset: int, data: bytes) -> Any:
    """Send one piece."""
    return client.put(f"{BASE}/{upload}/pieces/{offset}", headers=AUTH,
                      content=data)


def started(client: Any, name: str, size: int) -> Dict[str, Any]:
    """Announce a file that must be accepted; return the upload."""
    response = start(client, name, size)
    assert response.status_code == 201, response.text
    body: Dict[str, Any] = response.json()
    return body


def test_pieces_resume_and_complete(app: App) -> None:
    """A cut in the middle: ask what arrived, go on, complete."""
    client, queue = app
    create(client)
    data = make_pdf(["Política de devoluciones: 30 días."] * 3)
    upload = started(client, "Política.pdf", len(data))
    assert upload["received"] == 0 and upload["piece_bytes"] == PIECE_BYTES
    assert upload["type"] == "pdf" and upload["document"] is None
    half = len(data) // 2
    assert piece(client, upload["id"], 0, data[:half]).status_code == 200
    # the answer of the next piece was lost: the page asks and goes on
    status = client.get(f"{BASE}/{upload['id']}", headers=AUTH).json()
    assert status["received"] == half
    sent_again = piece(client, upload["id"], 0, data[:half])
    assert sent_again.json()["received"] == half
    gap = piece(client, upload["id"], half + 10, data[half + 10:])
    assert gap.status_code == 409 and str(half) in gap.json()["detail"]
    assert piece(client, upload["id"], half, data[half:]).json()[
        "received"] == len(data)
    done = client.post(f"{BASE}/{upload['id']}/complete", headers=AUTH)
    assert done.status_code == 202, done.text
    document = done.json()
    assert document["id"] == document_id("Política.pdf", data)
    again = client.post(f"{BASE}/{upload['id']}/complete", headers=AUTH)
    assert again.json() == document
    original = Path(
        f"data/knowledge/acme/originals/{document['id']}.pdf"
    )
    assert original.read_bytes() == data
    assert stat.S_IMODE(original.stat().st_mode) == 0o600
    assert not list(Path(f"data/knowledge/{UPLOADS_DIR}").glob("*.part"))
    queue.join()
    row = documents(client)[0]
    assert (row["status"], row["pages"], row["pages_done"]) == \
        ("indexed", 3, None)


@pytest.mark.parametrize("name, size, status", [
    ("a.exe", 10, 415),
    ("a.md", 0, 422),
    ("a.md", 200_001, 413),
    ("a.pdf", 200_001, 413),
])
def test_refused_when_announced(
    app: App, name: str, size: int, status: int,
) -> None:
    """Type by name, size by kind: refused before any byte."""
    client, _ = app
    create(client)
    response = start(client, name, size)
    assert response.status_code == status, response.text
    assert not list(Path(f"data/knowledge/{UPLOADS_DIR}").glob("*.part"))


def test_envelope_and_piece_errors(app: App) -> None:
    """Bad bodies, bad ids, pieces too large or past the end, no token."""
    client, _ = app
    create(client)
    assert client.post(BASE, headers=AUTH, json={"name": "a"}) \
        .status_code == 422
    assert start(client, "a.md", 5, metadata={"intent": 1}).status_code \
        == 422
    assert start(client, "a.md", 5, metadata={"category": "chisme"}) \
        .status_code == 422
    assert client.post("/knowledge/domains/nope/uploads", headers=AUTH,
                       json={"name": "a.md", "size": 5}).status_code == 404
    upload = started(client, "a.md", 5)["id"]
    assert piece(client, upload, 0, b"123456").status_code == 422
    big = {**AUTH, "content-length": str(PIECE_BYTES + 1)}
    assert client.put(f"{BASE}/{upload}/pieces/0", headers=big,
                      content=b"x").status_code == 413
    assert piece(client, upload, -1, b"1").status_code == 409
    assert client.post(f"{BASE}/{upload}/complete", headers=AUTH) \
        .status_code == 409
    for bad in ("nope", "../x", "f" * 32):
        assert client.get(f"{BASE}/{bad}", headers=AUTH).status_code in \
            (404, 422)
    assert client.get(f"{BASE}/{upload}").status_code == 401
    assert client.delete(f"{BASE}/{upload}", headers=AUTH).status_code \
        == 204
    assert client.get(f"{BASE}/{upload}", headers=AUTH).status_code == 404


def test_type_not_matching_the_name(app: App) -> None:
    """A text file called .docx is refused on its first piece (415)."""
    client, _ = app
    create(client)
    data = b"just text, no zip" * 300
    upload = started(client, "a.docx", len(data))
    refused = piece(client, upload["id"], 0, data[:5000])
    assert refused.status_code == 415
    assert client.get(f"{BASE}/{upload['id']}", headers=AUTH) \
        .status_code == 404
    real = make_docx()
    ok = started(client, "b.docx", len(real))
    piece(client, ok["id"], 0, real)
    assert client.post(f"{BASE}/{ok['id']}/complete", headers=AUTH) \
        .status_code == 202


def test_no_room_on_the_disk(
    app: App, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk that would not hold the file answers 507 with a reason."""
    client, _ = app
    create(client)
    monkeypatch.setattr(FileSystemUploadArea, "free_bytes",
                        lambda self: 100)
    refused = start(client, "a.md", 1000)
    assert refused.status_code == 507
    assert "espacio" in refused.json()["detail"]


def test_limits_are_served(app: App) -> None:
    """The page reads the limits of each kind and the piece size."""
    client, _ = app
    answer = client.get("/knowledge/limits", headers=AUTH)
    assert answer.status_code == 200
    assert answer.json() == {
        "pdf": 200_000, "video": 200_000, "audio": 200_000,
        "document": 200_000, "piece_bytes": PIECE_BYTES,
        "max_media_seconds": None,
    }
    assert client.get("/knowledge/limits").status_code == 401


def test_the_area_on_disk(tmp_path: Path) -> None:
    """Mode 600, writes at offsets, missing and stray files, the sweep's
    view of orphans."""
    area = FileSystemUploadArea(str(tmp_path))
    assert area.ids() == [] and area.free_bytes() > 0
    upload = PartialUpload(id="a" * 32, domain="acme", name="a.md",
                           type="md", size=6)
    area.create(upload)
    part = tmp_path / UPLOADS_DIR / f"{upload.id}.part"
    assert stat.S_IMODE(part.stat().st_mode) == 0o600
    area.write(upload.id, 0, b"abc")
    area.write(upload.id, 3, b"def")
    assert area.head(upload.id, 4) == b"abcd"
    assert b"".join(area.read(upload.id)) == b"abcdef"
    assert area.file(upload.id).size == 6
    assert area.load(upload.id) == upload
    (tmp_path / UPLOADS_DIR / "stray.part").write_bytes(b"")
    (tmp_path / UPLOADS_DIR / ("b" * 32 + ".part")).write_bytes(b"")
    assert area.ids() == ["a" * 32, "b" * 32]
    area.drop_file(upload.id)
    for call in (lambda: area.head(upload.id, 1),
                 lambda: list(area.read(upload.id)),
                 lambda: area.file(upload.id),
                 lambda: area.write(upload.id, 0, b"x")):
        with pytest.raises(NotFoundError):
            call()
    area.discard(upload.id)
    with pytest.raises(NotFoundError):
        area.load(upload.id)
    (tmp_path / UPLOADS_DIR / ("c" * 32 + ".json")).write_text("{")
    with pytest.raises(CorruptedDataError):
        area.load("c" * 32)
    missing = FileSystemUploadArea(str(tmp_path / "not" / "yet"))
    assert missing.free_bytes() > 0


def test_the_store_moves_and_copies_across_disks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A received file is renamed in; across disks it is copied, the
    source deleted, the copy mode 600."""
    import os

    from src.adapters.outbound.fs_document_store import (
        FileSystemDocumentStore,
    )
    from src.application.knowledge.ports import StoredFile
    from src.domain.knowledge.documents import KnowledgeDocument

    store = FileSystemDocumentStore(str(tmp_path / "data"))
    document = KnowledgeDocument(id="a-1", name="a.pdf", type="pdf", size=3,
                                 uploaded_at="2026-11-27T10:00:00+00:00")
    source = tmp_path / "received.part"
    source.write_bytes(b"pdf")
    real = os.replace

    def cross_device(a: Any, b: Any) -> None:
        """Refuse to rename the received file, as across disks."""
        if str(a).endswith("received.part"):
            raise OSError(18, "Invalid cross-device link")
        real(a, b)

    monkeypatch.setattr(os, "replace", cross_device)
    store.keep_original("acme", document, StoredFile(str(source), 3))
    kept = store.original("acme", document)
    assert Path(kept.path).read_bytes() == b"pdf" and kept.size == 3
    assert not source.exists()
    assert stat.S_IMODE(Path(kept.path).stat().st_mode) == 0o600
