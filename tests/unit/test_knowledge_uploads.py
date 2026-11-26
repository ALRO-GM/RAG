"""Uploads in pieces (#282): limits by kind, resume, disk, sweep."""

import hashlib
import itertools
from pathlib import Path
from typing import List, Optional

import pytest

from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.media import MediaTranscription
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.knowledge.uploads import DocumentUploads
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NoSpaceError,
    NotFoundError,
    TooLargeError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import (
    DocumentMetadata,
    KnowledgeDocument,
    document_id,
    document_id_of,
)
from src.domain.knowledge.domains import DomainManifest, Vocabularies
from src.domain.knowledge.uploads import (
    PIECE_BYTES,
    PartialUpload,
    UploadLimits,
    check_upload_id,
    limit_kind,
    megabytes,
    no_space,
    space_needed,
    too_large,
)
from tests.knowledge_fakes import (
    FakeExtractor,
    FakeMediaTranscriber,
    FakeUploadArea,
    InMemoryCatalogue,
    InMemoryDocumentStore,
    InMemoryIndexes,
    ManualQueue,
)

LIMITS = UploadLimits(pdf=500, video=1000, audio=200, document=100)
PDF = b"%PDF-1.4\n" + b"p" * 300
MP4 = b"\x00\x00\x00\x20ftypisom" + b"v" * 700
MP3_AS_MP4 = b"ID3\x04\x00\x00" + b"a" * 300
TEXT = b"# Guide\n\nrefund policy " + b"t" * 40


class Clock:
    """Seconds since the epoch that the test moves."""

    def __init__(self) -> None:
        """Start at a fixed time."""
        self.now = 1_000_000.0

    def __call__(self) -> float:
        """Return the time."""
        return self.now


class Setup:
    """A library with uploads over fakes, in a temporary folder."""

    def __init__(
        self, root: Path, linked: bool = True, free: int = 10 ** 9,
        min_free: int = 1000,
    ) -> None:
        """Wire everything."""
        self.indexes = InMemoryIndexes()
        catalogue = InMemoryCatalogue(DomainManifest(
            id="acme", title="Acme",
            vocabularies=Vocabularies(intent=["refund"]),
        ))
        self.store = InMemoryDocumentStore(self.indexes, root / "originals")
        self.queue = ManualQueue()
        self.media_queue = ManualQueue()
        self.extractor = FakeExtractor()
        library: KnowledgeLibrary
        registry = DomainIndexRegistry(
            catalogue, self.indexes,
            extra_documents=lambda d: library.pending_documents(d),
        )
        media = MediaTranscription(
            FakeMediaTranscriber(), 7200, sleep=lambda s: None,
        ) if linked else None
        library = KnowledgeLibrary(
            registry, self.store, self.extractor, self.queue, LIMITS,
            lambda: "2026-11-27T10:00:00+00:00", media,
            self.media_queue if linked else None,
        )
        self.library = library
        self.area = FakeUploadArea(root / "uploads", free=free)
        self.clock = Clock()
        counter = itertools.count(1)
        self.uploads = DocumentUploads(
            library, self.area, min_free, ttl_seconds=3600,
            clock=self.clock, new_id=lambda: f"{next(counter):032x}",
        )

    def start(
        self, name: str, size: int,
        metadata: Optional[DocumentMetadata] = None,
    ) -> PartialUpload:
        """Announce a file."""
        return self.uploads.start(
            "acme", name, size, metadata or DocumentMetadata()
        )

    def send(self, upload: PartialUpload, data: bytes, piece: int) -> None:
        """Send ``data`` in pieces of ``piece`` bytes."""
        for offset in range(0, len(data), piece):
            self.uploads.put("acme", upload.id, offset,
                             data[offset:offset + piece])


@pytest.fixture()
def setup(tmp_path: Path) -> Setup:
    """A linked library."""
    return Setup(tmp_path)


# Domain -------------------------------------------------------------------


def test_limits_by_kind_come_from_the_content() -> None:
    """A video type gets the video limit only with video content."""
    assert limit_kind("pdf") == "pdf"
    assert limit_kind("mp4") == "video" and limit_kind("mp4", MP4) == "video"
    assert limit_kind("webm", b"\x1a\x45\xdf\xa3....") == "video"
    assert limit_kind("mp4", MP3_AS_MP4) == "audio"
    assert limit_kind("mp3") == limit_kind("wav", b"RIFF") == "audio"
    assert limit_kind("docx") == limit_kind("png") == "document"
    assert LIMITS.of("video") == 1000 and LIMITS.largest == 1000
    defaults = UploadLimits()
    assert (defaults.pdf, defaults.video, defaults.audio,
            defaults.document) == (500 << 20, 1024 << 20, 200 << 20,
                                   100 << 20)


def test_messages_say_the_real_limit() -> None:
    """The kind's limit, in megabytes, in Spanish and English."""
    text = too_large("pdf", 500 << 20, 612 << 20)
    assert "Los PDF pueden ocupar hasta 500 MB; este ocupa 612 MB" in text
    assert "PDF files may be up to 500 MB; this one is 612 MB" in text
    assert "Los vídeos" in too_large("video", 1 << 30, 2 << 30)
    assert "1024 MB" in no_space(1 << 30) and megabytes(-1) == "0 MB"
    assert space_needed("mp4", 10) == 20 and space_needed("pdf", 10) == 10


def test_streamed_hash_gives_the_old_id() -> None:
    """Existing documents keep their ids; duplicates are still found."""
    data = b"same bytes" * 1000
    digest = hashlib.sha256()
    for start in range(0, len(data), 7):
        digest.update(data[start:start + 7])
    assert document_id_of("Guía.pdf", digest.hexdigest()) == \
        document_id("Guía.pdf", data)
    with pytest.raises(InvalidInputError):
        check_upload_id("../../x")


# Pieces -------------------------------------------------------------------


def test_a_file_in_pieces_becomes_a_document(setup: Setup) -> None:
    """Announced, sent in pieces, completed: the same id as one request."""
    meta = DocumentMetadata(intent="refund")
    upload = setup.start("Guía.md", len(TEXT), meta)
    assert (upload.received, upload.type, upload.checked) == (0, "md", False)
    setup.send(upload, TEXT, 7)
    status = setup.uploads.status("acme", upload.id)
    assert status.received == len(TEXT) and status.checked
    document = setup.uploads.complete("acme", upload.id)
    assert document.id == document_id("Guía.md", TEXT)
    assert document.metadata == meta and document.status == "queued"
    assert setup.store.bytes_of("acme", document.id) == TEXT
    assert not (setup.area.root / f"{upload.id}.part").exists(), "moved"
    assert setup.uploads.complete("acme", upload.id) == document, \
        "completing again answers the same"
    assert setup.uploads.status("acme", upload.id).view()["document"][
        "id"] == document.id
    setup.queue.run_all()
    assert setup.store.get("acme", document.id).status == "indexed"


def test_resume_duplicate_and_out_of_order_pieces(setup: Setup) -> None:
    """A piece sent again is accepted; a gap is refused with the offset."""
    upload = setup.start("a.md", len(TEXT))
    setup.uploads.put("acme", upload.id, 0, TEXT[:10])
    with pytest.raises(BusyError, match="offset 10"):
        setup.uploads.put("acme", upload.id, 20, TEXT[20:30])
    again = setup.uploads.put("acme", upload.id, 0, TEXT[:10])
    assert again.received == 10
    overlap = setup.uploads.put("acme", upload.id, 5, TEXT[5:25])
    assert overlap.received == 25
    with pytest.raises(BusyError, match="missing"):
        setup.uploads.complete("acme", upload.id)
    setup.uploads.put("acme", upload.id, 25, TEXT[25:])
    document = setup.uploads.complete("acme", upload.id)
    assert setup.store.bytes_of("acme", document.id) == TEXT


@pytest.mark.parametrize("offset, data, error", [
    (-1, b"x", BusyError),
    (0, TEXT + b"more", InvalidInputError),
    (0, b"x" * (PIECE_BYTES + 1), TooLargeError),
])
def test_bad_pieces(
    setup: Setup, offset: int, data: bytes, error: type,
) -> None:
    """Negative offsets, pieces past the size or too big are refused."""
    upload = setup.start("a.md", len(TEXT))
    with pytest.raises(error):
        setup.uploads.put("acme", upload.id, offset, data)
    assert setup.uploads.status("acme", upload.id).received == 0


@pytest.mark.parametrize("name, size, error", [
    ("a.exe", 10, UnsupportedDocumentError),
    ("a.md", 0, InvalidInputError),
    ("a.md", 101, TooLargeError),
    ("a.pdf", 501, TooLargeError),
    ("a.mp3", 201, TooLargeError),
    ("a.mp4", 1001, TooLargeError),
])
def test_refused_when_announced(
    setup: Setup, name: str, size: int, error: type,
) -> None:
    """The name's type and the size are checked before any byte."""
    with pytest.raises(error):
        setup.start(name, size)
    assert setup.area.ids() == []


def test_each_kind_has_its_own_limit(setup: Setup) -> None:
    """A 300-byte PDF and a 700-byte video pass; a document would not."""
    assert setup.start("a.pdf", 500).size == 500
    assert setup.start("a.mp4", 1000).size == 1000
    with pytest.raises(TooLargeError, match="Estos documentos"):
        setup.start("a.docx", 300)


def test_the_content_decides_the_limit(setup: Setup) -> None:
    """An MP3 named .mp4 gets the audio limit as soon as it starts."""
    upload = setup.start("trick.mp4", len(MP3_AS_MP4))
    with pytest.raises(TooLargeError, match="Los audios"):
        setup.send(upload, MP3_AS_MP4, 100)
    assert setup.area.ids() == [], "refused: deleted at once"
    ok = setup.start("clase.mp4", len(MP4))
    setup.send(ok, MP4, 100)
    document = setup.uploads.complete("acme", ok.id)
    assert document.type == "mp4" and document.size == len(MP4)


def test_content_not_matching_the_name_is_refused(setup: Setup) -> None:
    """A text file named .pdf is refused on its first piece."""
    upload = setup.start("fake.pdf", 300)
    with pytest.raises(UnsupportedDocumentError):
        setup.uploads.put("acme", upload.id, 0, b"plain text" * 30)
    with pytest.raises(NotFoundError):
        setup.uploads.status("acme", upload.id)


def test_refused_at_completion_deletes_the_file(setup: Setup) -> None:
    """A duplicate, or a file the reader refuses, leaves nothing."""
    first = setup.start("a.md", len(TEXT))
    setup.send(first, TEXT, 50)
    setup.uploads.complete("acme", first.id)
    again = setup.start("a.md", len(TEXT))
    setup.send(again, TEXT, 50)
    with pytest.raises(BusyError, match="already uploaded"):
        setup.uploads.complete("acme", again.id)
    bad = setup.start("bad.txt", 8)
    setup.send(bad, b"BAD text", 50)
    with pytest.raises(UnsupportedDocumentError):
        setup.uploads.complete("acme", bad.id)
    assert setup.area.ids() == [first.id], "only the finished record"


def test_uploads_belong_to_their_domain(setup: Setup) -> None:
    """Another domain's id, an unknown id or a bad one are not found."""
    setup.library.create_domain(DomainManifest(id="hr", title="RR. HH."))
    upload = setup.start("a.md", len(TEXT))
    with pytest.raises(NotFoundError):
        setup.uploads.status("hr", upload.id)
    with pytest.raises(NotFoundError):
        setup.uploads.status("acme", "f" * 32)
    with pytest.raises(InvalidInputError):
        setup.uploads.status("acme", "nope")
    with pytest.raises(NotFoundError):
        setup.uploads.start("nope", "a.md", 3, DocumentMetadata())
    with pytest.raises(InvalidInputError):
        setup.start("a.md", 3, DocumentMetadata(intent="shipping"))
    setup.uploads.cancel("acme", upload.id)
    assert setup.area.ids() == []


def test_media_needs_the_transcriber(tmp_path: Path) -> None:
    """Without a transcriber a recording is refused when announced."""
    plain = Setup(tmp_path, linked=False)
    with pytest.raises(UnsupportedDocumentError, match="transcripción"):
        plain.start("clase.mp4", 100)


# Disk ---------------------------------------------------------------------


def test_disk_is_checked_before_the_first_byte(tmp_path: Path) -> None:
    """The file, the transcriber's copy of a recording and the margin."""
    setup = Setup(tmp_path, free=1800, min_free=1000)
    assert setup.start("a.pdf", 500).size == 500
    with pytest.raises(NoSpaceError, match="espacio"):
        setup.start("b.pdf", 400)  # the first upload's 500 are promised
    with pytest.raises(NoSpaceError):
        setup.start("clase.mp4", 200)  # a recording needs two copies
    assert setup.start("c.md", 100).size == 100


def test_a_piece_never_fills_the_disk(tmp_path: Path) -> None:
    """Other programs filled the disk meanwhile: the piece is refused."""
    setup = Setup(tmp_path, free=10_000, min_free=1000)
    upload = setup.start("a.md", len(TEXT))
    setup.area.free = 520
    with pytest.raises(NoSpaceError):
        setup.uploads.put("acme", upload.id, 0, TEXT[:30])
    assert setup.uploads.status("acme", upload.id).received == 0


# One request and the sweep -----------------------------------------------


def test_one_request_reads_in_pieces(setup: Setup) -> None:
    """``receive`` reads the stream a piece at a time; nothing is kept."""
    asked: List[int] = []
    chunks = [TEXT[:20], TEXT[20:]]

    def read(size: int) -> bytes:
        """Return the next chunk."""
        asked.append(size)
        return chunks.pop(0) if chunks else b""

    document = setup.uploads.receive(
        "acme", "a.md", len(TEXT), read, DocumentMetadata()
    )
    assert isinstance(document, KnowledgeDocument)
    assert asked == [PIECE_BYTES] * 3 and setup.area.ids() == []
    with pytest.raises(InvalidInputError, match="vacío"):
        setup.uploads.upload("acme", "b.md", b"", DocumentMetadata())
    short = iter([b"abc", b""])
    with pytest.raises(InvalidInputError, match="3 of 10"):
        setup.uploads.receive("acme", "c.md", 10, lambda n: next(short),
                              DocumentMetadata())
    assert setup.area.ids() == []


def test_sweep_deletes_abandoned_uploads(setup: Setup) -> None:
    """Untouched for the TTL: deleted; recent ones stay."""
    old = setup.start("old.md", len(TEXT))
    setup.clock.now += 3000
    recent = setup.start("new.md", len(TEXT))
    setup.uploads.put("acme", recent.id, 0, TEXT[:5])
    setup.clock.now += 1000
    assert setup.uploads.sweep() == 1
    assert setup.area.ids() == [recent.id]
    setup.clock.now += 5000
    assert setup.uploads.sweep() == 1 and setup.area.ids() == []
    assert setup.uploads.sweep() == 0
    assert not list(setup.area.root.glob("*.part"))
    assert old.id not in setup.area.ids()
