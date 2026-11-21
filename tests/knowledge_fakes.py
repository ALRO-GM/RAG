"""In-memory fakes of the knowledge-domain ports."""

from typing import Callable, Dict, List, Optional

from src.application.knowledge.ports import ExtractedDocument, IndexState
from src.application.ports import IndexSnapshot
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import KnowledgeDocument
from src.domain.knowledge.domains import DomainManifest
from src.domain.models import Chunk, IndexReport, ScoredChunk
from tests.fakes import make_snapshot


class InMemoryCatalogue:
    """Manifests in a dict; ``authored`` ids cannot be removed."""

    def __init__(self, *manifests: DomainManifest) -> None:
        """Start with ``manifests``."""
        self.manifests = {m.id: m for m in manifests}
        self.authored_ids: set[str] = set()

    def ids(self) -> List[str]:
        """Return the ids, sorted."""
        return sorted(self.manifests)

    def manifest(self, domain_id: str) -> DomainManifest:
        """Return a manifest or raise ``NotFoundError``."""
        if domain_id not in self.manifests:
            raise NotFoundError(f"No knowledge domain {domain_id!r}.")
        return self.manifests[domain_id]

    def authored(self, domain_id: str) -> bool:
        """Tell whether ``domain_id`` is authored."""
        self.manifest(domain_id)
        return domain_id in self.authored_ids

    def create(self, manifest: DomainManifest) -> None:
        """Add a manifest (``BusyError`` if taken)."""
        if manifest.id in self.manifests:
            raise BusyError("exists")
        self.manifests[manifest.id] = manifest

    def remove(self, domain_id: str) -> None:
        """Forget a manifest."""
        del self.manifests[domain_id]


class FakeSearcher:
    """Return the chunks it holds whose text contains a query word."""

    def __init__(self, chunks: List[Chunk], mode: str) -> None:
        """Keep the chunks."""
        self.chunks = chunks
        self.mode = mode

    @property
    def snapshot(self) -> IndexSnapshot:
        """Return an in-memory snapshot of the chunks."""
        return make_snapshot(self.chunks)

    def search_where(
        self, query: str, k: int, keep: Callable[[Chunk], bool]
    ) -> List[ScoredChunk]:
        """Return matching chunks, best (most query words) first."""
        words = query.lower().split()
        scored = [
            ScoredChunk(chunk=c, score=float(
                sum(w in c.text.lower() for w in words)
            ))
            for c in self.chunks if keep(c)
        ]
        scored = [s for s in scored if s.score > 0]
        return sorted(scored, key=lambda s: -s.score)[:k]


class FakeIndexer:
    """Index the domain's in-memory sources."""

    def __init__(self, owner: "InMemoryIndexes", domain_id: str) -> None:
        """Index into ``owner``."""
        self.owner = owner
        self.domain_id = domain_id

    def run(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: Optional[int] = None,
        force: bool = False,
    ) -> IndexReport:
        """Make one chunk per source file."""
        files = self.owner.files.get(self.domain_id, {})
        if self.owner.fail_with is not None:
            raise self.owner.fail_with
        chunks = [
            Chunk(file_path=p, first_character_index=0,
                  last_character_index=len(t), kind="text", text=t)
            for p, t in files.items()
        ]
        self.owner.built[self.domain_id] = chunks
        self.owner.runs.append((self.domain_id, max_chunk_size, force))
        return IndexReport(
            files_total=len(files), files_reindexed=len(files),
            files_reused=0, files_removed=0, chunks=len(chunks),
            seconds=0.0, incremental=False,
        )


class InMemoryIndexes:
    """Domain sources and indexes in memory."""

    def __init__(self) -> None:
        """Start without sources or indexes."""
        self.files: Dict[str, Dict[str, str]] = {}
        self.built: Dict[str, List[Chunk]] = {}
        self.runs: List[tuple[str, int, bool]] = []
        self.searchers_made = 0
        self.fail_with: Optional[Exception] = None

    def indexer(self, manifest: DomainManifest) -> FakeIndexer:
        """Return an indexer of the domain."""
        return FakeIndexer(self, manifest.id)

    def searcher(self, manifest: DomainManifest, mode: str) -> FakeSearcher:
        """Return a searcher over the built chunks."""
        self.searchers_made += 1
        return FakeSearcher(self.built[manifest.id], mode)

    def state(self, domain_id: str) -> Optional[IndexState]:
        """Fingerprint = the texts indexed."""
        if domain_id not in self.built:
            return None
        chunks = self.built[domain_id]
        return IndexState(
            "|".join(c.text for c in chunks), len(chunks),
            len({c.file_path for c in chunks}), "2026-11-21T10:00:00+00:00",
        )

    def sources(self, domain_id: str) -> List[str]:
        """Return the domain's file names."""
        return sorted(self.files.get(domain_id, {}))

    def drop(self, domain_id: str) -> None:
        """Forget the domain's index."""
        self.built.pop(domain_id, None)


class InMemoryDocumentStore:
    """Documents in dicts; extracted texts become the domain's files."""

    def __init__(self, indexes: InMemoryIndexes) -> None:
        """Write extracted texts into ``indexes.files``."""
        self.indexes = indexes
        self.records: Dict[str, Dict[str, KnowledgeDocument]] = {}
        self.originals: Dict[tuple[str, str], bytes] = {}

    def list(self, domain_id: str) -> List[KnowledgeDocument]:
        """Return the documents, oldest first."""
        found = self.records.get(domain_id, {}).values()
        return sorted(found, key=lambda d: (d.uploaded_at, d.id))

    def get(self, domain_id: str, document_id: str) -> KnowledgeDocument:
        """Return one document."""
        try:
            return self.records[domain_id][document_id]
        except KeyError:
            raise NotFoundError(f"No document {document_id!r}.")

    def save(self, domain_id: str, document: KnowledgeDocument) -> None:
        """Store the record."""
        self.records.setdefault(domain_id, {})[document.id] = document

    def save_original(
        self, domain_id: str, document: KnowledgeDocument, data: bytes
    ) -> None:
        """Keep the bytes."""
        self.originals[(domain_id, document.id)] = data

    def original(self, domain_id: str, document: KnowledgeDocument) -> bytes:
        """Return the bytes."""
        return self.originals[(domain_id, document.id)]

    def text_path(self, domain_id: str, document_id: str) -> str:
        """Return the indexed name of the text."""
        return f"extracted/{document_id}.md"

    def read_text(self, domain_id: str, document_id: str) -> str:
        """Return the text written for the document."""
        path = self.text_path(domain_id, document_id)
        text = self.indexes.files.get(domain_id, {}).get(path)
        if text is None:
            raise NotFoundError(f"No text for {document_id!r}.")
        return text

    def write_text(self, domain_id: str, document_id: str, text: str) -> str:
        """Add the text to the domain's sources."""
        path = self.text_path(domain_id, document_id)
        self.indexes.files.setdefault(domain_id, {})[path] = text
        return path

    def remove_text(self, domain_id: str, document_id: str) -> None:
        """Remove the text from the sources."""
        self.indexes.files.get(domain_id, {}).pop(
            self.text_path(domain_id, document_id), None
        )

    def delete(self, domain_id: str, document_id: str) -> None:
        """Delete everything of a document."""
        self.get(domain_id, document_id)
        self.remove_text(domain_id, document_id)
        self.originals.pop((domain_id, document_id), None)
        del self.records[domain_id][document_id]


class FakeExtractor:
    """Decode the bytes as text; ``b"FAIL"`` fails, ``b"BAD"`` is refused."""

    def __init__(self) -> None:
        """Count the extractions."""
        self.extracted = 0

    def verify(self, data: bytes, document_type: str) -> None:
        """Refuse ``b"BAD..."``."""
        if data.startswith(b"BAD"):
            raise UnsupportedDocumentError("not really a docx")

    def extract(self, data: bytes, document_type: str) -> ExtractedDocument:
        """Return the text, or fail on ``b"FAIL..."``."""
        self.extracted += 1
        if data.startswith(b"FAIL"):
            raise InvalidInputError("no text in it")
        return ExtractedDocument(data.decode("utf-8"), pages=1)


class ManualQueue:
    """Keep submitted jobs until :meth:`run_all`."""

    def __init__(self) -> None:
        """No jobs yet."""
        self.jobs: List[Callable[[], None]] = []

    def submit(self, job: Callable[[], None]) -> None:
        """Keep ``job``."""
        self.jobs.append(job)

    def run_all(self) -> int:
        """Run every pending job (and those they submit); count them."""
        count = 0
        while self.jobs:
            self.jobs.pop(0)()
            count += 1
        return count
