"""Domains and uploaded documents: the library use case."""

from typing import Tuple

import pytest

from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.ports import ExtractedDocument
from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    TooLargeError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import DocumentMetadata
from src.domain.knowledge.domains import DomainManifest, Vocabularies
from tests.knowledge_fakes import (
    FakeExtractor,
    InMemoryCatalogue,
    InMemoryDocumentStore,
    InMemoryIndexes,
    ManualQueue,
)

Setup = Tuple[KnowledgeLibrary, ManualQueue, InMemoryDocumentStore,
              InMemoryIndexes]


@pytest.fixture()
def setup() -> Setup:
    """Return a library over one empty domain and its fakes."""
    indexes = InMemoryIndexes()
    catalogue = InMemoryCatalogue(DomainManifest(
        id="acme", title="Acme",
        vocabularies=Vocabularies(intent=["refund", "pricing"]),
    ))
    store = InMemoryDocumentStore(indexes)
    queue = ManualQueue()
    library: KnowledgeLibrary
    registry = DomainIndexRegistry(
        catalogue, indexes,
        extra_documents=lambda d: library.pending_documents(d),
    )
    library = KnowledgeLibrary(
        registry, store, FakeExtractor(), queue, 100,
        lambda: "2026-11-21T10:00:00+00:00",
    )
    return library, queue, store, indexes


def test_upload_is_queued_then_indexed(setup: Setup) -> None:
    """The request returns ``queued``; the worker indexes the text."""
    library, queue, store, indexes = setup
    meta = DocumentMetadata(intent="refund", tags=["faq"])
    document = library.upload("acme", "../Guía.md", b"# T\n\nrefund", meta)
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
    library, queue, store, _ = setup
    with pytest.raises(error):
        library.upload("acme", name, data, meta)
    assert store.list("acme") == [] and queue.jobs == []


def test_duplicate_and_unknown_domain(setup: Setup) -> None:
    """The same file twice is refused; an unknown domain is 404."""
    library, _, _, _ = setup
    library.upload("acme", "a.txt", b"same", DocumentMetadata())
    with pytest.raises(BusyError, match="already uploaded as 'a.txt'"):
        library.upload("acme", "a.txt", b"same", DocumentMetadata())
    with pytest.raises(NotFoundError):
        library.upload("nope", "b.txt", b"x", DocumentMetadata())
    with pytest.raises(NotFoundError):
        library.documents("nope")


def test_failed_extraction_is_recorded(setup: Setup) -> None:
    """The reason is kept and nothing of the file is indexed."""
    library, queue, store, indexes = setup
    document = library.upload("acme", "scan.txt", b"FAIL", DocumentMetadata())
    queue.run_all()
    failed = store.get("acme", document.id)
    assert (failed.status, failed.error, failed.chunks) == \
        ("failed", "no text in it", 0)
    assert indexes.files.get("acme", {}) == {}


def test_indexing_errors_fail_the_document(setup: Setup) -> None:
    """An error while indexing marks the document failed too."""
    library, queue, store, indexes = setup
    indexes.fail_with = InvalidInputError("other.md:3: intent 'x'")
    document = library.upload("acme", "a.txt", b"text", DocumentMetadata())
    queue.run_all()
    assert store.get("acme", document.id).error == "other.md:3: intent 'x'"
    indexes.fail_with = RuntimeError("boom")
    other = library.upload("acme", "b.txt", b"more", DocumentMetadata())
    queue.run_all()
    assert store.get("acme", other.id).error == \
        "Error interno / Internal error: RuntimeError."


def test_update_requeues_with_new_metadata(setup: Setup) -> None:
    """New metadata is checked, recorded and indexed again."""
    library, queue, store, indexes = setup
    document = library.upload("acme", "a.txt", b"text", DocumentMetadata())
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
    library, queue, store, _ = setup
    document = library.upload("acme", "a.txt", b"text", DocumentMetadata())
    extract = library.extractor.extract

    def relabel_meanwhile(data: bytes, kind: str) -> ExtractedDocument:
        """Change the metadata while the first job extracts."""
        library.update("acme", document.id, DocumentMetadata(intent="refund"))
        return extract(data, kind)

    library.extractor.extract = relabel_meanwhile  # type: ignore
    queue.jobs.pop(0)()
    assert store.get("acme", document.id).status == "queued"
    library.extractor.extract = extract  # type: ignore
    queue.run_all()
    done = store.get("acme", document.id)
    assert (done.status, done.metadata.intent) == ("indexed", "refund")


def test_delete_document_reindexes(setup: Setup) -> None:
    """A deleted document leaves the store and then the index."""
    library, queue, store, indexes = setup
    first = library.upload("acme", "a.txt", b"one", DocumentMetadata())
    library.upload("acme", "b.txt", b"two", DocumentMetadata())
    queue.run_all()
    library.delete("acme", first.id)
    assert [d.name for d in store.list("acme")] == ["b.txt"]
    queue.run_all()
    assert library.domains()[0].chunks == 1
    with pytest.raises(NotFoundError):
        library.delete("acme", first.id)


def test_deleted_while_queued_is_skipped(setup: Setup) -> None:
    """A job for a document deleted meanwhile does nothing."""
    library, queue, store, _ = setup
    document = library.upload("acme", "a.txt", b"one", DocumentMetadata())
    library.delete("acme", document.id)
    queue.run_all()
    assert store.list("acme") == []


def test_recover_requeues_unfinished_documents(setup: Setup) -> None:
    """After a restart queued and processing uploads run again."""
    library, queue, store, _ = setup
    a = library.upload("acme", "a.txt", b"one", DocumentMetadata())
    b = library.upload("acme", "b.txt", b"two", DocumentMetadata())
    queue.jobs.clear()
    store.save("acme", store.get("acme", b.id).model_copy(
        update={"status": "processing"}))
    assert library.recover() == 2
    queue.run_all()
    assert {store.get("acme", d.id).status for d in (a, b)} == {"indexed"}
    assert library.recover() == 0


def test_domains_create_and_delete(setup: Setup) -> None:
    """Created domains can be deleted; authored ones cannot."""
    library, queue, store, indexes = setup
    row = library.create_domain(DomainManifest(id="hr", title="RR. HH."))
    assert (row.id, row.documents, row.index_version) == ("hr", 0, None)
    with pytest.raises(BusyError):
        library.create_domain(DomainManifest(id="hr", title="again"))
    library.upload("hr", "a.txt", b"x", DocumentMetadata())
    queue.run_all()
    library.delete_domain("hr")
    assert [d.id for d in library.domains()] == ["acme"]
    assert "hr" not in indexes.built
    library.registry.catalogue.authored_ids.add("acme")  # type: ignore
    with pytest.raises(InvalidInputError, match="authored"):
        library.delete_domain("acme")
    assert [d.id for d in library.domains()] == ["acme"]
