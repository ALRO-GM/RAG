"""In-memory fakes of the knowledge-domain ports.

Files (originals, partial uploads) are real files in a folder the test
gives, since the ports hand them over by path.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.application.knowledge.ports import (
    ExtractedDocument,
    IndexState,
    MediaJob,
    Progress,
    StoredFile,
)
from src.application.ports import IndexSnapshot
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    ServiceUnavailableError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import KnowledgeDocument
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.uploads import PartialUpload
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
    """Documents in dicts; extracted texts become the domain's files.

    Originals are moved into ``root`` (``originals`` maps them to their
    file).
    """

    def __init__(self, indexes: InMemoryIndexes, root: Path) -> None:
        """Write extracted texts into ``indexes.files``."""
        self.indexes = indexes
        self.root = root
        self.records: Dict[str, Dict[str, KnowledgeDocument]] = {}
        self.originals: Dict[tuple[str, str], StoredFile] = {}

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

    def keep_original(
        self, domain_id: str, document: KnowledgeDocument, source: StoredFile
    ) -> None:
        """Move the file into ``root``."""
        target = self.root / f"{domain_id}--{document.id}"
        self.root.mkdir(parents=True, exist_ok=True)
        os.replace(source.path, target)
        self.originals[(domain_id, document.id)] = StoredFile(
            str(target), source.size
        )

    def original(
        self, domain_id: str, document: KnowledgeDocument
    ) -> StoredFile:
        """Return the kept file (``NotFoundError`` if none)."""
        try:
            return self.originals[(domain_id, document.id)]
        except KeyError:
            raise NotFoundError("no original") from None

    def bytes_of(self, domain_id: str, document_id: str) -> bytes:
        """Return the bytes of a kept original."""
        return Path(self.originals[(domain_id, document_id)].path).read_bytes()

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
    """Decode the file as text; ``b"FAIL"`` fails, ``b"BAD"`` is refused.

    ``pages`` > 1 reports that many pages, one at a time, to the
    progress callback.
    """

    def __init__(self, pages: int = 1) -> None:
        """Count the extractions."""
        self.extracted = 0
        self.pages = pages

    def verify(self, source: StoredFile, document_type: str) -> None:
        """Refuse ``b"BAD..."``."""
        if Path(source.path).read_bytes().startswith(b"BAD"):
            raise UnsupportedDocumentError("not really a docx")

    def extract(
        self, source: StoredFile, document_type: str,
        progress: Optional[Progress] = None,
    ) -> ExtractedDocument:
        """Return the text, or fail on ``b"FAIL..."``."""
        self.extracted += 1
        data = Path(source.path).read_bytes()
        if data.startswith(b"FAIL"):
            raise InvalidInputError("no text in it")
        for done in range(self.pages + 1):
            if progress is not None:
                progress(done, self.pages)
        return ExtractedDocument(data.decode("utf-8"), pages=self.pages)


class FakeUploadArea:
    """Partial uploads as files in ``root``; ``free`` is the free disk."""

    def __init__(self, root: Path, free: int = 10 ** 12) -> None:
        """Keep the files in ``root``."""
        self.root = root
        self.free = free
        self.records: Dict[str, PartialUpload] = {}

    def _part(self, upload_id: str) -> Path:
        """Return the file of an upload."""
        return self.root / f"{upload_id}.part"

    def create(self, upload: PartialUpload) -> None:
        """Keep the record and an empty file."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._part(upload.id).write_bytes(b"")
        self.records[upload.id] = upload

    def load(self, upload_id: str) -> PartialUpload:
        """Return the record."""
        try:
            return self.records[upload_id]
        except KeyError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None

    def save(self, upload: PartialUpload) -> None:
        """Replace the record."""
        self.records[upload.id] = upload

    def write(self, upload_id: str, offset: int, data: bytes) -> None:
        """Write at ``offset``."""
        self.load(upload_id)
        with open(self._part(upload_id), "r+b") as handle:
            handle.seek(offset)
            handle.write(data)

    def head(self, upload_id: str, size: int) -> bytes:
        """Return the first bytes."""
        return self._part(upload_id).read_bytes()[:size]

    def read(self, upload_id: str) -> Iterator[bytes]:
        """Yield the bytes in pieces of 3 (small, to test the hashing)."""
        data = self._part(upload_id).read_bytes()
        for start in range(0, len(data), 3):
            yield data[start:start + 3]

    def file(self, upload_id: str) -> StoredFile:
        """Return the file."""
        part = self._part(upload_id)
        return StoredFile(str(part), part.stat().st_size)

    def discard(self, upload_id: str) -> None:
        """Delete file and record."""
        self.drop_file(upload_id)
        self.records.pop(upload_id, None)

    def drop_file(self, upload_id: str) -> None:
        """Delete the file."""
        self._part(upload_id).unlink(missing_ok=True)

    def ids(self) -> List[str]:
        """Return the ids kept."""
        return sorted(self.records)

    def free_bytes(self) -> int:
        """Return :attr:`free`."""
        return self.free


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


def contract_transcript(
    *said: tuple[str, float, float, str],
    names: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """A valid ``transcript.v1`` of ``(speaker, start, end, text)`` turns."""
    names = names or {}
    speakers: List[str] = list(dict.fromkeys(s[0] for s in said))
    utterances: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for turn, (who, start, end, text) in enumerate(said):
        utterance = {
            "id": f"t{turn}.s0", "turn": turn, "index": 0, "speaker": who,
            "start": start, "end": end, "text": text,
            "previous_speaker": previous["speaker"] if previous else None,
            "previous_utterance": previous["id"] if previous else None,
            "confidence": None, "speaker_confidence": None,
        }
        utterances.append(utterance)
        previous = utterance
    return {
        "schema": "transcript.v1",
        "session": {"id": "kn-job", "title": None, "language": "es",
                    "started_at": None, "duration": said[-1][2] if said
                    else 0.0, "audio_sha256": None},
        "speakers": [{"id": s, "name": names.get(s), "enrolled": False}
                     for s in speakers],
        "utterances": utterances,
        "pipeline": {"recogniser": "fake", "voice_model": "fake",
                     "transcriber": "0.1.0"},
    }


class FakeMediaTranscriber:
    """The transcriber service, scripted.

    ``answers`` are what successive ``status`` calls return (a
    :class:`MediaJob`, or an exception to raise); when they run out the
    job is done with ``transcript``. ``submit_error`` is raised by the
    next ``submit``; ``busy`` ``forget`` calls are refused first.
    """

    def __init__(self, transcript: Optional[Dict[str, Any]] = None) -> None:
        """Nothing sent yet."""
        self.transcript = transcript or contract_transcript(
            ("SPEAKER_01", 0.0, 4.0, "Hola, bienvenidos al curso."),
            ("SPEAKER_02", 4.5, 9.0, "Hoy hablamos de reembolsos."),
        )
        self.answers: List[Any] = []
        self.submitted: List[tuple[str, int, str, str, float]] = []
        self.sent: List[bytes] = []
        self.forgotten: List[str] = []
        self.submit_error: Optional[Exception] = None
        self.busy = 0

    def submit(
        self, job_id: str, source: StoredFile, file_format: str,
        language: str, max_seconds: float,
    ) -> MediaJob:
        """Record the file; the job is queued."""
        if self.submit_error is not None:
            error, self.submit_error = self.submit_error, None
            raise error
        self.sent.append(Path(source.path).read_bytes())
        self.submitted.append(
            (job_id, source.size, file_format, language, max_seconds)
        )
        return MediaJob("queued")

    def status(self, job_id: str) -> MediaJob:
        """Return the next scripted answer, then done."""
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            assert isinstance(answer, MediaJob)
            return answer
        return MediaJob("done", transcript=self.transcript)

    def forget(self, job_id: str) -> None:
        """Record the deletion (refused while ``busy``)."""
        if self.busy:
            self.busy -= 1
            raise BusyError("running")
        self.forgotten.append(job_id)


UNREACHABLE = ServiceUnavailableError("down")
"""What a fake transcriber that cannot be reached raises."""
