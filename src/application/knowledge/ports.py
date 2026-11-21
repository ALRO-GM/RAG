"""Ports of the knowledge-domain use cases.

The file-system catalogue and document store, the extractors and the
background worker live in ``src/adapters/outbound``; the per-domain
indexing and search services are built by the composition root
(``src/bootstrap.py``). Unit tests use in-memory fakes.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from src.application.ports import IndexSnapshot
from src.domain.knowledge.documents import KnowledgeDocument
from src.domain.knowledge.domains import DomainManifest
from src.domain.models import Chunk, IndexReport, ScoredChunk


class DomainCatalogue(Protocol):
    """Where the domains' manifests live."""

    def ids(self) -> List[str]:
        """Return the ids of every domain, sorted."""
        ...

    def manifest(self, domain_id: str) -> DomainManifest:
        """Return a domain's manifest (``NotFoundError`` if none)."""
        ...

    def authored(self, domain_id: str) -> bool:
        """Tell whether the domain is authored in the repository."""
        ...

    def create(self, manifest: DomainManifest) -> None:
        """Create a domain (``BusyError`` if the id is taken)."""
        ...

    def remove(self, domain_id: str) -> None:
        """Delete a created domain's folder, uploads included."""
        ...


class DomainIndexer(Protocol):
    """Build or incrementally update one domain's index."""

    def run(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: Optional[int] = None,
        force: bool = False,
    ) -> IndexReport:
        """Index the domain's sources; reuse unchanged files."""
        ...


class DomainSearcher(Protocol):
    """Search one domain's index."""

    @property
    def snapshot(self) -> IndexSnapshot:
        """Return the loaded index."""
        ...

    def search_where(
        self, query: str, k: int, keep: Callable[[Chunk], bool]
    ) -> List[ScoredChunk]:
        """Return the top-``k`` chunks that ``keep`` accepts."""
        ...


@dataclass(frozen=True)
class IndexState:
    """What is on disk of a domain's index."""

    fingerprint: str
    chunks: int
    files: int
    built_at: str


class DomainIndexes(Protocol):
    """Build the indexing and search services of each domain."""

    def indexer(self, manifest: DomainManifest) -> DomainIndexer:
        """Return the indexing service of a domain."""
        ...

    def searcher(self, manifest: DomainManifest, mode: str) -> DomainSearcher:
        """Return a search service of a domain (``mode``: bm25,
        semantic or hybrid)."""
        ...

    def state(self, domain_id: str) -> Optional[IndexState]:
        """Return the state of a domain's index (``None``: not built)."""
        ...

    def sources(self, domain_id: str) -> List[str]:
        """Return the paths of the domain's indexable files."""
        ...

    def drop(self, domain_id: str) -> None:
        """Delete a domain's index."""
        ...


class DocumentStore(Protocol):
    """Uploaded documents of a domain: records, originals, text."""

    def list(self, domain_id: str) -> List[KnowledgeDocument]:
        """Return the domain's documents, oldest first."""
        ...

    def get(self, domain_id: str, document_id: str) -> KnowledgeDocument:
        """Return one document (``NotFoundError`` if none)."""
        ...

    def save(self, domain_id: str, document: KnowledgeDocument) -> None:
        """Create or replace a document's record."""
        ...

    def save_original(
        self, domain_id: str, document: KnowledgeDocument, data: bytes
    ) -> None:
        """Keep the uploaded bytes of a document."""
        ...

    def original(self, domain_id: str, document: KnowledgeDocument) -> bytes:
        """Return the uploaded bytes of a document."""
        ...

    def write_text(self, domain_id: str, document_id: str, text: str) -> str:
        """Write the extracted text to index; return its path."""
        ...

    def text_path(self, domain_id: str, document_id: str) -> str:
        """Return the path the document's extracted text is indexed as."""
        ...

    def remove_text(self, domain_id: str, document_id: str) -> None:
        """Delete the extracted text (it leaves the index)."""
        ...

    def delete(self, domain_id: str, document_id: str) -> None:
        """Delete a document: record, original and text."""
        ...


@dataclass(frozen=True)
class ExtractedDocument:
    """Text extracted from an uploaded file, as Markdown."""

    text: str
    pages: Optional[int] = None


class DocumentExtractor(Protocol):
    """Turn an uploaded file into Markdown text."""

    def verify(self, data: bytes, document_type: str) -> None:
        """Check the file's inside (``UnsupportedDocumentError``)."""
        ...

    def extract(self, data: bytes, document_type: str) -> ExtractedDocument:
        """Return the text of ``data`` (``InvalidInputError`` if none)."""
        ...


class GpuGate(Protocol):
    """Tell whether a batch job may use the GPU right now.

    The card belongs to the live mode while a session runs; ingestion
    is a batch job that yields (``docs/KNOWLEDGE.md`` §6).
    """

    def free(self) -> bool:
        """Return ``True`` when no live session holds the GPU."""
        ...


class PageLayoutReader(Protocol):
    """Read pages of a PDF or image as Markdown, with OCR and tables."""

    def read_pages(
        self, data: bytes, kind: str, pages: Optional[Sequence[int]],
        ocr: bool,
    ) -> Dict[int, str]:
        """Return the Markdown of each page (1-based) of ``data``.

        ``pages`` limits a PDF to those pages (all when ``None``, as for
        images); ``ocr`` turns text recognition on for bitmaps.
        """
        ...


class JobQueue(Protocol):
    """Run jobs in the background, one at a time, in order."""

    def submit(self, job: Callable[[], None]) -> None:
        """Queue ``job``; return at once."""
        ...
