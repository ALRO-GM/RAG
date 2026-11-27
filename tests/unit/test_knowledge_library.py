"""Domains and uploaded documents: the library use case."""

from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.ports import (
    ExtractedDocument,
    Progress,
    StoredFile,
)
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.knowledge.uploads import DocumentUploads
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    TooLargeError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import DocumentMetadata, KnowledgeDocument
from src.domain.knowledge.domains import DomainManifest, Vocabularies
from src.domain.knowledge.uploads import UploadLimits
from tests.knowledge_fakes import (
    FakeExtractor,
    FakeUploadArea,
    InMemoryCatalogue,
    InMemoryDocumentStore,
    InMemoryIndexes,
    ManualQueue,
)

Setup = Tuple[KnowledgeLibrary, ManualQueue, InMemoryDocumentStore,
              InMemoryIndexes, DocumentUploads]

LIMITS = UploadLimits(pdf=100, video=100, audio=100, document=100)


@pytest.fixture()
def setup(tmp_path: Path) -> Setup:
    """Return a library over one empty domain and its fakes."""
    indexes = InMemoryIndexes()
    catalogue = InMemoryCatalogue(DomainManifest(
        id="acme", title="Acme",
        vocabularies=Vocabularies(intent=["refund", "pricing"]),
    ))
    store = InMemoryDocumentStore(indexes, tmp_path / "originals")
    queue = ManualQueue()
    library: KnowledgeLibrary
    registry = DomainIndexRegistry(
        catalogue, indexes,
        extra_documents=lambda d: library.pending_documents(d),
    )
    library = KnowledgeLibrary(
        registry, store, FakeExtractor(), queue, LIMITS,
        lambda: "2026-11-21T10:00:00+00:00",
    )
    uploads = DocumentUploads(library, FakeUploadArea(tmp_path / "up"), 0)
    return library, queue, store, indexes, uploads


def test_upload_is_queued_then_indexed(setup: Setup) -> None:
    """The request returns ``queued``; the worker indexes the text."""
    library, queue, store, indexes, uploads = setup
    meta = DocumentMetadata(intent="refund", tags=["faq"])
    document = uploads.upload("acme", "../Guía.md", b"# T\n\nrefund", meta)
    assert document.status == "queued"
    assert document.name == "Guía.md" and document.type == "md"
    assert document.size == 11
    assert library.domains()[0].documents == 1
    assert queue.run_all() == 1
    done = store.get("acme", document.id)
    assert (done.status, done.chunks, done.pages, done.error) == \
        ("indexed", 1, 1, None)
    text = indexes.files["acme"][f"extracted/{document.id}.md"]
    assert "intent: refund" in text and text.endswith("refund\n")
    row = library.domains()[0]
    assert (row.documents, row.chunks) == (1, 1)
    assert library.documents("acme") == [done]


@pytest.mark.parametrize("name, data, meta, error", [
    ("a.md", b"", DocumentMetadata(), InvalidInputError),
    ("a.md", b"x" * 101, DocumentMetadata(), TooLargeError),
    ("a.exe", b"x", DocumentMetadata(), UnsupportedDocumentError),
    ("a.pdf", b"not a pdf", DocumentMetadata(), UnsupportedDocumentError),
    ("a.docx", b"BAD zip", DocumentMetadata(), UnsupportedDocumentError),
    ("a.md", b"x", DocumentMetadata(intent="shipping"), InvalidInputError),
])
def test_upload_refusals(
    setup: Setup, name: str, data: bytes, meta: DocumentMetadata,
    error: type,
) -> None:
    """Nothing is stored or queued when an upload is refused."""
    library, queue, store, _, uploads = setup
    with pytest.raises(error):
        uploads.upload("acme", name, data, meta)
    assert store.list("acme") == [] and queue.jobs == []


def test_duplicate_and_unknown_domain(setup: Setup) -> None:
    """The same file twice is refused; an unknown domain is 404."""
    library, _, _, _, uploads = setup
    uploads.upload("acme", "a.txt", b"same", DocumentMetadata())
    with pytest.raises(BusyError, match="already uploaded as 'a.txt'"):
        uploads.upload("acme", "a.txt", b"same", DocumentMetadata())
    with pytest.raises(NotFoundError):
        uploads.upload("nope", "b.txt", b"x", DocumentMetadata())
    with pytest.raises(NotFoundError):
        library.documents("nope")


def test_failed_extraction_is_recorded(setup: Setup) -> None:
    """The reason is kept and nothing of the file is indexed."""
    library, queue, store, indexes, uploads = setup
    document = uploads.upload("acme", "scan.txt", b"FAIL", DocumentMetadata())
    queue.run_all()
    failed = store.get("acme", document.id)
    assert (failed.status, failed.error, failed.chunks) == \
        ("failed", "no text in it", 0)
    assert indexes.files.get("acme", {}) == {}


def test_indexing_errors_fail_the_document(setup: Setup) -> None:
    """An error while indexing marks the document failed too."""
    library, queue, store, indexes, uploads = setup
    indexes.fail_with = InvalidInputError("other.md:3: intent 'x'")
    document = uploads.upload("acme", "a.txt", b"text", DocumentMetadata())
    queue.run_all()
    assert store.get("acme", document.id).error == "other.md:3: intent 'x'"
    indexes.fail_with = RuntimeError("boom")
    other = uploads.upload("acme", "b.txt", b"more", DocumentMetadata())
    queue.run_all()
    assert store.get("acme", other.id).error == \
        "Error interno / Internal error: RuntimeError."


def test_update_requeues_with_new_metadata(setup: Setup) -> None:
    """New metadata is checked, recorded and indexed again."""
    library, queue, store, indexes, uploads = setup
    document = uploads.upload("acme", "a.txt", b"text", DocumentMetadata())
    queue.run_all()
    with pytest.raises(InvalidInputError):
        library.update("acme", document.id,
                       DocumentMetadata(intent="shipping"))
    changed = library.update("acme", document.id,
                             DocumentMetadata(intent="pricing"))
    assert changed.status == "queued"
    queue.run_all()
    assert store.get("acme", document.id).status == "indexed"
    assert "intent: pricing" in \
        indexes.files["acme"][f"extracted/{document.id}.md"]
    with pytest.raises(NotFoundError):
        library.update("acme", "missing-1", DocumentMetadata())


def test_relabel_while_processing_keeps_queued(setup: Setup) -> None:
    """A job finishing after a relabel does not overwrite ``queued``."""
    library, queue, store, _, uploads = setup
    document = uploads.upload("acme", "a.txt", b"text", DocumentMetadata())
    extract = library.extractor.extract

    def relabel_meanwhile(
        source: StoredFile, kind: str, progress: Optional[Progress] = None,
    ) -> ExtractedDocument:
        """Change the metadata while the first job extracts."""
        library.update("acme", document.id, DocumentMetadata(intent="refund"))
        return extract(source, kind, progress)

    library.extractor.extract = relabel_meanwhile  # type: ignore
    queue.jobs.pop(0)()
    assert store.get("acme", document.id).status == "queued"
    library.extractor.extract = extract  # type: ignore
    queue.run_all()
    done = store.get("acme", document.id)
    assert (done.status, done.metadata.intent) == ("indexed", "refund")


def test_delete_document_reindexes(setup: Setup) -> None:
    """A deleted document leaves the store and then the index."""
    library, queue, store, indexes, uploads = setup
    first = uploads.upload("acme", "a.txt", b"one", DocumentMetadata())
    uploads.upload("acme", "b.txt", b"two", DocumentMetadata())
    queue.run_all()
    library.delete("acme", first.id)
    assert [d.name for d in store.list("acme")] == ["b.txt"]
    queue.run_all()
    assert library.domains()[0].chunks == 1
    with pytest.raises(NotFoundError):
        library.delete("acme", first.id)


def test_deleted_while_queued_is_skipped(setup: Setup) -> None:
    """A job for a document deleted meanwhile does nothing."""
    library, queue, store, _, uploads = setup
    document = uploads.upload("acme", "a.txt", b"one", DocumentMetadata())
    library.delete("acme", document.id)
    queue.run_all()
    assert store.list("acme") == []


def test_recover_requeues_unfinished_documents(setup: Setup) -> None:
    """After a restart queued and processing uploads run again."""
    library, queue, store, _, uploads = setup
    a = uploads.upload("acme", "a.txt", b"one", DocumentMetadata())
    b = uploads.upload("acme", "b.txt", b"two", DocumentMetadata())
    queue.jobs.clear()
    store.save("acme", store.get("acme", b.id).model_copy(
        update={"status": "processing"}))
    assert library.recover() == 2
    queue.run_all()
    assert {store.get("acme", d.id).status for d in (a, b)} == {"indexed"}
    assert library.recover() == 0


def test_domains_create_and_delete(setup: Setup) -> None:
    """Created domains can be deleted; authored ones cannot."""
    library, queue, store, indexes, uploads = setup
    row = library.create_domain(DomainManifest(id="hr", title="RR. HH."))
    assert (row.id, row.documents, row.index_version) == ("hr", 0, None)
    with pytest.raises(BusyError):
        library.create_domain(DomainManifest(id="hr", title="again"))
    uploads.upload("hr", "a.txt", b"x", DocumentMetadata())
    queue.run_all()
    library.delete_domain("hr")
    assert [d.id for d in library.domains()] == ["acme"]
    assert "hr" not in indexes.built
    library.registry.catalogue.authored_ids.add("acme")  # type: ignore
    with pytest.raises(InvalidInputError, match="authored"):
        library.delete_domain("acme")
    assert [d.id for d in library.domains()] == ["acme"]


def test_long_documents_show_the_pages_read(setup: Setup) -> None:
    """While a document is read its row says «N of M pages»; throttled."""
    library, queue, store, _, uploads = setup
    library.extractor = FakeExtractor(pages=4)
    ticks = iter([0.0, 0.5, 1.0, 5.0, 5.1, 9.0, 9.5])
    library.clock = lambda: next(ticks)
    seen: List[Tuple[Optional[int], Optional[int]]] = []
    save = store.save

    def recording(domain_id: str, document: KnowledgeDocument) -> None:
        """Note the pages of every save while processing."""
        if document.status == "processing":
            seen.append((document.pages_done, document.pages))
        save(domain_id, document)

    store.save = recording  # type: ignore[method-assign]
    document = uploads.upload("acme", "long.md", b"text", DocumentMetadata())
    queue.run_all()
    assert seen == [(None, None), (0, 4), (3, 4), (4, 4)]
    done = store.get("acme", document.id)
    assert (done.status, done.pages, done.pages_done) == ("indexed", 4, None)


def test_progress_after_the_end_is_dropped(setup: Setup) -> None:
    """A late report on a document no longer processing changes nothing."""
    library, queue, store, _, uploads = setup
    document = uploads.upload("acme", "a.md", b"text", DocumentMetadata())
    queue.run_all()
    report = library._progress("acme", document.id)
    report(1, 1)
    assert store.get("acme", document.id).pages_done is None
    library._progress("acme", "gone-1")(1, 1)
