"""Ports of the knowledge-domain use cases.

The file-system catalogue and document store, the extractors and the
background worker live in ``src/adapters/outbound``; the per-domain
indexing and search services are built by the composition root
(``src/bootstrap.py``). Unit tests use in-memory fakes.
"""

from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple,
)

from src.application.ports import IndexSnapshot
from src.domain.knowledge.documents import KnowledgeDocument
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.uploads import PartialUpload
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


@dataclass(frozen=True)
class StoredFile:
    """A file an adapter keeps on disk: where it is and its size.

    The use cases pass it from one port to another without opening it;
    only adapters read it, in pieces (a 1 GB video is never held in
    memory).
    """

    path: str
    size: int


Progress = Callable[[int, int], None]
"""Told ``(done, total)`` pages while a long document is read."""


class UploadArea(Protocol):
    """Files being received in pieces, until they become documents.

    Raises ``NotFoundError`` for an upload it does not have.
    """

    def create(self, upload: PartialUpload) -> None:
        """Keep a new upload's record and an empty file for its bytes."""
        ...

    def load(self, upload_id: str) -> PartialUpload:
        """Return an upload's record."""
        ...

    def save(self, upload: PartialUpload) -> None:
        """Replace an upload's record."""
        ...

    def write(self, upload_id: str, offset: int, data: bytes) -> None:
        """Write ``data`` at ``offset`` of the upload's file."""
        ...

    def head(self, upload_id: str, size: int) -> bytes:
        """Return the first ``size`` bytes of the upload's file."""
        ...

    def read(self, upload_id: str) -> Iterator[bytes]:
        """Yield the upload's bytes in pieces, from the start."""
        ...

    def file(self, upload_id: str) -> StoredFile:
        """Return the upload's file (to verify it or keep it)."""
        ...

    def discard(self, upload_id: str) -> None:
        """Delete an upload's file and record (absent: nothing)."""
        ...

    def drop_file(self, upload_id: str) -> None:
        """Delete only the upload's file (the record stays)."""
        ...

    def ids(self) -> List[str]:
        """Return the ids of every upload kept."""
        ...

    def free_bytes(self) -> int:
        """Return the free space of the disk the uploads are on."""
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

    def keep_original(
        self, domain_id: str, document: KnowledgeDocument, source: StoredFile
    ) -> None:
        """Keep a received file as the document's original.

        The file is moved (renamed on the same disk), not copied; it is
        no longer where it was.
        """
        ...

    def original(
        self, domain_id: str, document: KnowledgeDocument
    ) -> StoredFile:
        """Return the document's original (``NotFoundError`` if gone)."""
        ...

    def write_text(self, domain_id: str, document_id: str, text: str) -> str:
        """Write the extracted text to index; return its path."""
        ...

    def text_path(self, domain_id: str, document_id: str) -> str:
        """Return the path the document's extracted text is indexed as."""
        ...

    def read_text(self, domain_id: str, document_id: str) -> str:
        """Return the extracted text (``NotFoundError`` when there is none)."""
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

    def verify(self, source: StoredFile, document_type: str) -> None:
        """Check the file's inside (``UnsupportedDocumentError``)."""
        ...

    def extract(
        self, source: StoredFile, document_type: str,
        progress: Optional[Progress] = None,
    ) -> ExtractedDocument:
        """Return the text of ``source`` (``InvalidInputError`` if none).

        ``progress`` is told the pages read so far of a long document.
        """
        ...


class GpuGate(Protocol):
    """Tell whether a batch job may use the GPU right now.

    The card belongs to the live mode while a session runs; ingestion
    is a batch job that yields (``docs/KNOWLEDGE.md`` §6).
    """

    def free(self) -> bool:
        """Return ``True`` when no live session holds the GPU."""
        ...


@dataclass(frozen=True)
class PageContent:
    """One page read with layout: its Markdown and the figures on it.

    ``figures`` are the pictures of the page as PNG bytes, in reading
    order (only when the reader was asked for them).
    """

    markdown: str
    figures: Tuple[bytes, ...] = ()


class PageLayoutReader(Protocol):
    """Read pages of a PDF or image as Markdown, with OCR and tables."""

    def read_pages(
        self, source: StoredFile, kind: str, pages: Optional[Sequence[int]],
        ocr: bool, figures: bool = False,
    ) -> Dict[int, PageContent]:
        """Return each page (1-based) of the file ``source``.

        ``pages`` limits a PDF to those pages (all when ``None``, as for
        images); ``ocr`` turns text recognition on for bitmaps;
        ``figures`` also returns the pictures of each page.
        """
        ...


@dataclass(frozen=True)
class ImageDescription:
    """What a captioner says about one image."""

    caption: str
    text: str = ""


class ImageCaptioner(Protocol):
    """Describe an image: a caption and the text written in it."""

    def describe(self, image: bytes) -> ImageDescription:
        """Return the caption and the text of ``image`` (any raster format).

        Raises an error when the model cannot run; the caller goes on
        without captions.
        """
        ...


@dataclass(frozen=True)
class MediaJob:
    """Where the transcription of a recording is (KN-3a).

    ``state``: ``queued``, ``running``, ``done`` or ``failed``;
    ``message`` the reason of a failure; ``transcript`` the
    ``transcript.v1`` document once ``done``.
    """

    state: str
    message: Optional[str] = None
    transcript: Optional[Dict[str, Any]] = None


class MediaTranscriber(Protocol):
    """The transcriber service, transcribing a library's recordings.

    The RAG loads no speech model: the recording is sent to the
    transcriber, which runs its own pipeline and GPU rules. Every method
    raises ``ServiceUnavailableError`` when the service cannot be
    reached, ``NotFoundError`` for a job it does not have (``status``)
    and ``UnsupportedDocumentError`` for a format it refuses.
    """

    def submit(
        self, job_id: str, source: StoredFile, file_format: str,
        language: str, max_seconds: float,
    ) -> MediaJob:
        """Start transcribing the file ``source`` as job ``job_id``.

        The file is sent in a stream, never read whole into memory.
        """
        ...

    def status(self, job_id: str) -> MediaJob:
        """Return the job (with its transcript once done)."""
        ...

    def forget(self, job_id: str) -> None:
        """Delete the job and what the transcriber kept of it.

        Raises ``BusyError`` while it is running.
        """
        ...


class JobQueue(Protocol):
    """Run jobs in the background, one at a time, in order."""

    def submit(self, job: Callable[[], None]) -> None:
        """Queue ``job``; return at once."""
        ...
