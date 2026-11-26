"""A recording uploaded to the RAG's library, transcribed by the transcriber.

End to end over HTTP on loopback (KN-3a): the RAG's ``/knowledge``
routes with its real library, file store and indexes, and the
transcriber's app (fake models) served by uvicorn, reached through
:class:`~src.adapters.outbound.http_media_transcriber.
HttpMediaTranscriber`.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
import uvicorn
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import KnowledgeRoutes
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.adapters.outbound.fs_document_store import FileSystemDocumentStore
from src.adapters.outbound.fs_domain_catalogue import (
    FileSystemDomainCatalogue,
)
from src.adapters.outbound.http_media_transcriber import (
    HttpMediaTranscriber,
    job_of,
    message_of,
    refusal,
)
from src.adapters.outbound.fs_upload_area import FileSystemUploadArea
from src.adapters.outbound.thread_job_queue import ThreadJobQueue
from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.media import UNREACHABLE, MediaTranscription
from src.application.knowledge.ports import StoredFile
from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.knowledge.uploads import DocumentUploads
from src.bootstrap import KnowledgeIndexFactory, utc_now
from src.config import KnowledgeSettings, Settings
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    ServiceUnavailableError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.media import media_job_id
from tests.fakes import FakeEmbedder
from tests.integration.test_knowledge_api import knowledge_client
from tests.integration.test_knowledge_upload_api import (
    create,
    documents,
    upload,
)
from tests.integration.test_transcript_ingest import AUTH
from tests.integration.transcriber.audio import scene, sound_file
from tests.knowledge_samples import as_file
from tests.integration.transcriber.test_knowledge_proxy import free_port
from transcriber.adapters.inbound.http_api import create_app
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import TranscriberSettings

CODE = "a-long-enough-access-code"
TOKEN = "t" * 40

Served = Tuple[str, TranscriberContainer]
App = Tuple[TestClient, ThreadJobQueue, ThreadJobQueue, KnowledgeSettings]


@pytest.fixture()
def transcriber(tmp_path: Path) -> Iterator[Served]:
    """Serve a transcriber (fake models) linked to a RAG by ``TOKEN``."""
    container = build_container(TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "transcriber"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": "http://127.0.0.1:9/rag"},
    }), environ={"TRANSCRIBER_PUBLISH_TOKEN": TOKEN})
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(
        create_app(container, CODE), host="127.0.0.1", port=port,
        log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", container
    finally:
        server.should_exit = True
        thread.join(5)
        container.runner.close(30)


def rag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str,
) -> App:
    """The RAG's knowledge routes, its audio and video sent to ``url``."""
    monkeypatch.chdir(tmp_path)
    k = KnowledgeSettings(
        max_audio_bytes=5_000_000, max_video_bytes=5_000_000,
        max_document_bytes=5_000_000, max_media_seconds=60, min_free_bytes=0,
    )
    library: KnowledgeLibrary
    registry = DomainIndexRegistry(
        FileSystemDomainCatalogue(k.domains_dir, k.data_dir),
        KnowledgeIndexFactory(k, Settings(), FakeEmbedder()),
        k.max_chunk_size,
        extra_documents=lambda d: library.pending_documents(d),
    )
    queue, media_queue = ThreadJobQueue(), ThreadJobQueue("media")
    library = KnowledgeLibrary(
        registry, FileSystemDocumentStore(k.data_dir), DocumentExtractors(),
        queue, k.upload_limits, utc_now,
        MediaTranscription(
            HttpMediaTranscriber(url, lambda: TOKEN, timeout=10),
            k.max_media_seconds, poll_seconds=0.05, max_wait_seconds=30,
            unreachable_seconds=0.2,
        ),
        media_queue,
    )
    uploads = DocumentUploads(
        library, FileSystemUploadArea(k.data_dir), k.min_free_bytes
    )
    routes = KnowledgeRoutes(
        KnowledgeQueryService(registry), library, uploads=uploads
    )
    return knowledge_client(tmp_path, registry, routes), queue, media_queue, k


def wait(app: App) -> List[Dict[str, Any]]:
    """Let both workers finish; return the documents."""
    client, queue, media_queue, _ = app
    media_queue.join()
    queue.join()
    return documents(client)


def two_voices(seconds: float = 5.0) -> bytes:
    """A WAV file with two voices."""
    return sound_file(scene([
        ("220", seconds), ("-", 1.6), ("880", seconds), ("-", 1.6),
        ("220", 1.6),
    ]), 16000, "wav")


def test_a_recording_becomes_timed_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transcriber: Served,
) -> None:
    """Upload, transcribe, index, query with times, delete everything."""
    url, container = transcriber
    app = rag(tmp_path, monkeypatch, url)
    client, _, _, k = app
    create(client)
    sent = upload(client, "Formación reembolsos.wav", two_voices(),
                  {"intent": "refund"})
    assert sent.status_code == 202, sent.text
    assert sent.json()["status"] == "queued"
    assert sent.json()["type"] == "wav"
    [document] = wait(app)
    assert document["status"] == "indexed", document
    assert document["chunks"] >= 1 and document["error"] is None
    text = client.get(
        f"/knowledge/domains/acme/documents/{document['id']}/text",
        headers=AUTH,
    ).json()["text"]
    assert text.startswith("[00:00–") and "**Voz 1:**" in text
    assert "**Voz 2:**" in text
    found = client.post("/knowledge/query", headers=AUTH, json={
        "schema": "knowledge.query.v1", "request_id": "r",
        "domain": "acme", "text": "hola vamos", "mode": "lexical",
        "filters": {"intent": ["refund"]},
    })
    assert found.status_code == 200, found.text
    chunks = found.json()["chunks"]
    assert chunks
    meta = chunks[0]["metadata"]
    assert meta["start_s"] == 0.0 and meta["end_s"] > 10
    # The transcriber kept nothing, and it never was a session.
    jobs = container.knowledge_media
    assert jobs is not None and jobs._sessions.ids() == []
    assert container.list_sessions().sessions == []
    # The original is kept like any upload; deleting removes it all.
    data = Path(k.data_dir) / "acme"
    assert (data / "originals" / f"{document['id']}.wav").is_file()
    assert (data / "extracted" / f"{document['id']}.md").is_file()
    gone = client.delete(
        f"/knowledge/domains/acme/documents/{document['id']}", headers=AUTH,
    )
    assert gone.status_code == 204
    assert wait(app) == []
    assert not list((data / "originals").iterdir())
    assert not list((data / "extracted").iterdir())
    assert not list((data / "documents").iterdir())


def test_failures_show_on_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transcriber: Served,
) -> None:
    """Too long, no speech, not audio: refused or failed with a reason."""
    url, _ = transcriber
    app = rag(tmp_path, monkeypatch, url)
    client = app[0]
    create(client)
    long = upload(client, "larga.wav", two_voices(31.0))
    silent = upload(client, "silencio.wav", sound_file(
        scene([("-", 2.0)]), 16000, "wav"))
    assert long.status_code == silent.status_code == 202
    fake = upload(client, "clase.mp3", b"hola, no soy audio")
    assert fake.status_code == 415
    assert "audio or video" in fake.json()["detail"]
    other = upload(client, "clase.aiff", b"FORM\x00\x00AIFF")
    assert other.status_code == 415
    errors = {d["name"]: (d["status"], d["error"]) for d in wait(app)}
    assert errors["larga.wav"][0] == "failed"
    assert "La grabación dura 2 min; el límite es 1 min" in \
        errors["larga.wav"][1]
    assert errors["silencio.wav"][0] == "failed"
    assert "No se oye voz" in errors["silencio.wav"][1]


def test_the_transcriber_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing listens: the document fails with a clear message."""
    app = rag(tmp_path, monkeypatch, f"http://127.0.0.1:{free_port()}")
    client = app[0]
    create(client)
    assert upload(client, "a.wav", two_voices()).status_code == 202
    [document] = wait(app)
    assert (document["status"], document["error"]) == ("failed", UNREACHABLE)


def test_the_adapter_maps_the_answers(transcriber: Served) -> None:
    """Unknown jobs, refused formats, a wrong token and odd answers."""
    url, _ = transcriber
    port = HttpMediaTranscriber(url, lambda: TOKEN)
    job = media_job_id("acme", "x")
    with pytest.raises(NotFoundError):
        port.status(job)
    port.forget(job)  # absent: nothing to do
    with pytest.raises(UnsupportedDocumentError, match="exe"):
        port.submit(job, as_file(b"x"), "exe", "es", 60)
    with pytest.raises(InvalidInputError):
        port.status("not-a-job")
    with pytest.raises(ServiceUnavailableError, match="401"):
        HttpMediaTranscriber(url, lambda: "wrong").status(job)
    with pytest.raises(NotFoundError, match="missing"):
        port.submit(job, StoredFile("/nonexistent/x.wav", 3), "wav", "es",
                    60)
    assert isinstance(refusal(507, "no room"), InvalidInputError)
    assert isinstance(refusal(409, "running"), BusyError)
    assert str(refusal(500, "")) == "The transcriber answered 500."
    assert message_of(b"\xff") == "" and message_of(b'{"message": 1}') == ""
    with pytest.raises(ServiceUnavailableError):
        job_of({"state": 3})
    assert job_of({"state": "queued", "message": 2}).message is None
