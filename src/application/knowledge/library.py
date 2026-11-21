"""Use case: manage domains and their uploaded documents.

Uploads are checked at once (size, type by content, metadata against
the domain's vocabularies) and stored with status ``queued``; one
background worker then extracts the text, writes it with the
document's metadata as front matter, re-indexes the domain
incrementally and sets ``indexed`` (or ``failed`` with the reason).
Changing a document's metadata or deleting it re-indexes the domain
the same way.
"""

import logging
import threading
from typing import Callable, List

from src.application.knowledge.ports import (
    DocumentExtractor,
    DocumentStore,
    JobQueue,
)
from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    RagError,
    TooLargeError,
)
from src.domain.knowledge.documents import (
    DocumentMetadata,
    KnowledgeDocument,
    detect_type,
    document_id,
    document_text,
    metadata_errors,
    sanitise_filename,
)
from src.domain.knowledge.domains import DomainManifest, DomainSummary

LOG = logging.getLogger(__name__)

SNIFF_BYTES = 4096
"""Bytes of an upload read to decide its type."""


class KnowledgeLibrary:
    """Create and delete domains; upload, relabel and delete documents.

    Args:
        registry: The domain index registry.
        store: Where uploads, records and extracted text are kept.
        extractor: Turns an upload into Markdown.
        queue: Runs the extraction and indexing jobs in order.
        max_upload_bytes: Largest file accepted.
        now: Returns the current time as ISO 8601 text.
    """

    def __init__(
        self,
        registry: DomainIndexRegistry,
        store: DocumentStore,
        extractor: DocumentExtractor,
        queue: JobQueue,
        max_upload_bytes: int,
        now: Callable[[], str],
    ) -> None:
        """Wire the ports."""
        self.registry = registry
        self.store = store
        self.extractor = extractor
        self.queue = queue
        self.max_upload_bytes = max_upload_bytes
        self.now = now
        self._lock = threading.RLock()

    # Domains ------------------------------------------------------------

    def domains(self) -> List[DomainSummary]:
        """Return every domain's row."""
        return self.registry.summaries()

    def create_domain(self, manifest: DomainManifest) -> DomainSummary:
        """Create an empty domain (``BusyError`` if the id is taken)."""
        with self._lock:
            self.registry.catalogue.create(manifest)
        return self.registry.summary(manifest.id)

    def delete_domain(self, domain_id: str) -> None:
        """Delete a created domain, its uploads and its index.

        Domains authored in the repository are refused
        (``InvalidInputError``).
        """
        manifest = self.registry.manifest(domain_id)
        with self._lock:
            if self.registry.catalogue.authored(manifest.id):
                raise InvalidInputError(
                    f"El dominio {manifest.id!r} está en el repositorio "
                    f"(knowledge/domains); bórralo allí. / The domain "
                    f"{manifest.id!r} is authored in the repository; "
                    "delete it there."
                )
            self.registry.forget(manifest.id)
            self.registry.catalogue.remove(manifest.id)
            self.registry.indexes.drop(manifest.id)

    def pending_documents(self, domain_id: str) -> int:
        """Count uploads whose text is not (yet) an indexable source."""
        sources = set(self.registry.indexes.sources(domain_id))
        return sum(
            1 for d in self.store.list(domain_id)
            if self.store.text_path(domain_id, d.id) not in sources
        )

    # Documents ----------------------------------------------------------

    def documents(self, domain_id: str) -> List[KnowledgeDocument]:
        """Return a domain's uploaded documents, oldest first."""
        manifest = self.registry.manifest(domain_id)
        return self.store.list(manifest.id)

    def _check_metadata(
        self, manifest: DomainManifest, metadata: DocumentMetadata
    ) -> None:
        """Refuse metadata outside the domain's vocabularies."""
        errors = metadata_errors(metadata, manifest)
        if errors:
            raise InvalidInputError("; ".join(errors) + ".")

    def upload(
        self,
        domain_id: str,
        filename: str,
        data: bytes,
        metadata: DocumentMetadata,
    ) -> KnowledgeDocument:
        """Accept a file for the domain; it is processed in the background.

        Refuses empty and oversized files (``TooLargeError``), types
        not recognised from the content (``UnsupportedDocumentError``),
        metadata outside the vocabularies and a file already uploaded
        (``BusyError``).
        """
        manifest = self.registry.manifest(domain_id)
        if not data:
            raise InvalidInputError("El archivo está vacío. / Empty file.")
        if len(data) > self.max_upload_bytes:
            raise TooLargeError(
                f"El archivo supera {self.max_upload_bytes} bytes. / The "
                f"file is larger than {self.max_upload_bytes} bytes."
            )
        name = sanitise_filename(filename)
        kind = detect_type(name, data[:SNIFF_BYTES])
        self.extractor.verify(data, kind)
        self._check_metadata(manifest, metadata)
        document = KnowledgeDocument(
            id=document_id(name, data), name=name, type=kind,
            size=len(data), metadata=metadata, uploaded_at=self.now(),
        )
        with self._lock:
            try:
                existing = self.store.get(manifest.id, document.id)
            except NotFoundError:
                existing = None
            if existing is not None:
                raise BusyError(
                    f"Este archivo ya está subido como {existing.name!r}. "
                    f"/ This file is already uploaded as {existing.name!r}."
                )
            self.store.save_original(manifest.id, document, data)
            self.store.save(manifest.id, document)
        self._queue_processing(manifest.id, document.id)
        return document

    def update(
        self, domain_id: str, doc_id: str, metadata: DocumentMetadata
    ) -> KnowledgeDocument:
        """Change a document's metadata; it is re-indexed with it."""
        manifest = self.registry.manifest(domain_id)
        self._check_metadata(manifest, metadata)
        with self._lock:
            document = self.store.get(manifest.id, doc_id)
            document = document.model_copy(update={
                "metadata": metadata, "status": "queued", "error": None,
            })
            self.store.save(manifest.id, document)
        self._queue_processing(manifest.id, document.id)
        return document

    def delete(self, domain_id: str, doc_id: str) -> None:
        """Delete a document; its chunks leave the index in the background.
        """
        manifest = self.registry.manifest(domain_id)
        with self._lock:
            self.store.delete(manifest.id, doc_id)
        self.queue.submit(lambda: self._reindex(manifest.id))

    def recover(self) -> int:
        """Queue again the documents left queued or processing; count them.

        Called when the server starts: a restart must not strand an
        upload.
        """
        count = 0
        for domain_id in self.registry.catalogue.ids():
            try:
                documents = self.store.list(domain_id)
            except RagError as exc:
                LOG.warning("Cannot list documents of %s: %s", domain_id, exc)
                continue
            for document in documents:
                if document.status in ("queued", "processing"):
                    self._queue_processing(domain_id, document.id)
                    count += 1
        return count

    # Background jobs ----------------------------------------------------

    def _queue_processing(self, domain_id: str, doc_id: str) -> None:
        """Queue the extraction and indexing of one document."""
        self.queue.submit(lambda: self.process(domain_id, doc_id))

    def _reindex(self, domain_id: str) -> None:
        """Re-index a domain after a deletion (errors are logged)."""
        try:
            self.registry.index(domain_id)
        except RagError as exc:
            LOG.warning("Re-indexing %s failed: %s", domain_id, exc)

    def process(self, domain_id: str, doc_id: str) -> None:
        """Extract, write and index one document; record the outcome."""
        try:
            with self._lock:
                document = self.store.get(domain_id, doc_id).model_copy(
                    update={"status": "processing", "error": None}
                )
                self.store.save(domain_id, document)
            data = self.store.original(domain_id, document)
            extracted = self.extractor.extract(data, document.type)
            with self._lock:
                current = self.store.get(domain_id, doc_id)
                path = self.store.write_text(
                    domain_id, doc_id, document_text(current, extracted.text)
                )
            self.registry.index(domain_id)
            searcher, _ = self.registry.searcher(domain_id, "lexical")
            chunks = sum(
                1 for c in searcher.snapshot.chunks if c.file_path == path
            )
            self._finish(domain_id, doc_id, "indexed", None, chunks,
                         extracted.pages)
        except RagError as exc:
            self._fail(domain_id, doc_id, str(exc))
        except Exception as exc:  # never leave a document "processing"
            LOG.exception("Processing %s/%s failed", domain_id, doc_id)
            self._fail(domain_id, doc_id,
                       f"Error interno / Internal error: "
                       f"{type(exc).__name__}.")

    def _fail(self, domain_id: str, doc_id: str, error: str) -> None:
        """Mark a document failed and keep its text out of the index."""
        try:
            self.store.remove_text(domain_id, doc_id)
        except (RagError, OSError):
            pass
        self._finish(domain_id, doc_id, "failed", error, 0, None)

    def _finish(
        self, domain_id: str, doc_id: str, status: str,
        error: object, chunks: int, pages: object,
    ) -> None:
        """Record the outcome unless the document was deleted or changed.

        A document queued again meanwhile (new metadata) keeps
        ``queued``: its own job records the outcome.
        """
        with self._lock:
            try:
                current = self.store.get(domain_id, doc_id)
            except (RagError, OSError):
                return
            if current.status == "queued":
                return
            update = {"status": status, "error": error, "chunks": chunks}
            if pages is not None or status == "failed":
                update["pages"] = pages
            self.store.save(domain_id, current.model_copy(update=update))
