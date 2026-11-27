"""Audio and video uploads in the library: transcribed, then indexed."""

import tempfile
from pathlib import Path
from typing import List

import pytest

from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.media import (
    LOST,
    NO_SPEECH,
    UNREACHABLE,
    MediaTranscription,
)
from src.application.knowledge.ports import MediaJob, StoredFile
from src.application.knowledge.registry import DomainIndexRegistry
from src.application.knowledge.uploads import DocumentUploads
from src.domain.errors import (
    InvalidInputError,
    NotFoundError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import DocumentMetadata
from src.domain.knowledge.domains import DomainManifest, Vocabularies
from src.domain.knowledge.media import media_job_id
from src.domain.knowledge.uploads import UploadLimits
from tests.knowledge_fakes import (
    UNREACHABLE as DOWN,
    FakeExtractor,
    FakeUploadArea,
    FakeMediaTranscriber,
    InMemoryCatalogue,
    InMemoryDocumentStore,
    InMemoryIndexes,
    ManualQueue,
    contract_transcript,
)

MP3 = b"ID3\x04\x00\x00" + b"\x00" * 64


class Clock:
    """A clock that moves only when the use case sleeps."""

    def __init__(self) -> None:
        """Start at zero; no sleep yet."""
        self.now = 0.0
        self.slept: List[float] = []

    def sleep(self, seconds: float) -> None:
        """Move the clock."""
        self.slept.append(seconds)
        self.now += seconds


class Setup:
    """A library over one domain, with a scripted transcriber."""

    def __init__(self, linked: bool = True) -> None:
        """Wire the fakes (files in a temporary folder)."""
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.indexes = InMemoryIndexes()
        self.catalogue = InMemoryCatalogue(DomainManifest(
            id="acme", title="Acme", language="es",
            vocabularies=Vocabularies(intent=["refund", "pricing"]),
        ))
        self.store = InMemoryDocumentStore(self.indexes, root / "originals")
        self.queue = ManualQueue()
        self.media_queue = ManualQueue()
        self.clock = Clock()
        self.transcriber = FakeMediaTranscriber()
        self.media = MediaTranscription(
            self.transcriber, 7200, poll_seconds=5, max_wait_seconds=600,
            unreachable_seconds=60, sleep=self.clock.sleep,
            now=lambda: self.clock.now,
        )
        library: KnowledgeLibrary
        registry = DomainIndexRegistry(
            self.catalogue, self.indexes,
            extra_documents=lambda d: library.pending_documents(d),
        )
        library = KnowledgeLibrary(
            registry, self.store, FakeExtractor(), self.queue,
            UploadLimits(pdf=10_000, video=10_000, audio=10_000,
                         document=10_000),
            lambda: "2026-11-24T10:00:00+00:00",
            self.media if linked else None,
            self.media_queue if linked else None,
        )
        self.library = library
        self.uploads = DocumentUploads(library, FakeUploadArea(root / "up"), 0)

    def upload(self, name: str = "Clase 1.mp3") -> str:
        """Upload a recording; return its id."""
        return self.uploads.upload(
            "acme", name, MP3, DocumentMetadata(intent="refund")
        ).id

    def run(self) -> None:
        """Run both workers until nothing is left."""
        while self.media_queue.jobs or self.queue.jobs:
            self.media_queue.run_all()
            self.queue.run_all()

    def text(self, doc_id: str) -> str:
        """The written text of a document."""
        return self.store.read_text("acme", doc_id)


@pytest.fixture()
def setup() -> Setup:
    """A linked library."""
    return Setup()


def test_a_recording_is_transcribed_then_indexed(setup: Setup) -> None:
    """Queued, transcribed on its own worker, indexed by the usual one."""
    doc_id = setup.upload()
    document = setup.store.get("acme", doc_id)
    assert (document.type, document.status) == ("mp3", "queued")
    assert setup.queue.jobs == [] and len(setup.media_queue.jobs) == 1
    setup.transcriber.answers = [MediaJob("queued"), MediaJob("running")]
    setup.media_queue.run_all()
    assert setup.store.get("acme", doc_id).status == "processing"
    job = media_job_id("acme", doc_id)
    assert setup.transcriber.submitted == [(job, len(MP3), "mp3", "es",
                                            7200)]
    assert setup.transcriber.forgotten == [job]
    assert setup.clock.slept == [5, 5, 5]
    text = setup.text(doc_id)
    assert "timed: true" in text and "intent: refund" in text
    assert "[00:00–00:04] **Voz 1:** Hola, bienvenidos al curso." in text
    assert "[00:04–00:09] **Voz 2:** Hoy hablamos de reembolsos." in text
    assert len(setup.queue.jobs) == 1
    setup.queue.run_all()
    done = setup.store.get("acme", doc_id)
    assert (done.status, done.chunks, done.error, done.pages) == \
        ("indexed", 1, None, None)
    shown, body = setup.library.text("acme", doc_id)
    assert body.startswith("[00:00–00:04]")


def test_new_metadata_does_not_transcribe_again(setup: Setup) -> None:
    """The transcript is kept; only the front matter changes."""
    doc_id = setup.upload()
    setup.run()
    setup.library.update("acme", doc_id, DocumentMetadata(intent="pricing"))
    assert setup.media_queue.jobs == []
    setup.run()
    assert len(setup.transcriber.submitted) == 1
    text = setup.text(doc_id)
    assert "intent: pricing" in text and "intent: refund" not in text
    assert text.count("timed: true") == 1
    assert text.count("Hola, bienvenidos") == 1
    assert setup.store.get("acme", doc_id).status == "indexed"


@pytest.mark.parametrize("answers, submit_error, message", [
    ([MediaJob("failed", "recognise: No speech recognised.")], None,
     "No se oye voz en el archivo"),
    ([MediaJob("failed", "decode: The recording lasts 2 h 10 min; the "
               "limit is 2 h 00 min.")], None, "La grabación dura 2 h 10"),
    ([MediaJob("failed", "decode: Invalid data")], None,
     "No se pudo leer el sonido"),
    ([], DOWN, UNREACHABLE),
    ([DOWN] * 20, None, UNREACHABLE),
    ([NotFoundError("gone"), NotFoundError("gone")], None, LOST),
    ([MediaJob("running")] * 200, None, "no terminó en 10:00"),
    ([MediaJob("done", transcript={"schema": "nope"})], None,
     "La transcripción falló"),
    ([MediaJob("done", transcript=contract_transcript(
        ("S1", 0.0, 1.0, " ")))], None, NO_SPEECH),
])
def test_failures_end_failed_with_a_message(
    setup: Setup, answers: List[object], submit_error: object,
    message: str,
) -> None:
    """The transcriber down, no speech, too long, unreadable: ``failed``."""
    setup.transcriber.answers = list(answers)
    if isinstance(submit_error, Exception):
        setup.transcriber.submit_error = submit_error
    doc_id = setup.upload()
    setup.run()
    failed = setup.store.get("acme", doc_id)
    assert failed.status == "failed"
    assert failed.error is not None and message in failed.error
    assert setup.indexes.files.get("acme", {}) == {}
    assert setup.library.pending_documents("acme") == 1


def test_a_failed_recording_is_tried_again_on_relabel(setup: Setup) -> None:
    """Saving its metadata again sends a failed recording once more."""
    setup.transcriber.submit_error = DOWN
    doc_id = setup.upload()
    setup.run()
    assert setup.store.get("acme", doc_id).status == "failed"
    setup.library.update("acme", doc_id, DocumentMetadata())
    assert len(setup.media_queue.jobs) == 1
    setup.run()
    assert setup.store.get("acme", doc_id).status == "indexed"


def test_a_lost_job_is_sent_once_more(setup: Setup) -> None:
    """The transcriber restarted: the file goes again and is transcribed."""
    setup.transcriber.answers = [NotFoundError("gone"), MediaJob("running")]
    doc_id = setup.upload()
    setup.run()
    assert len(setup.transcriber.submitted) == 2
    assert setup.store.get("acme", doc_id).status == "indexed"


def test_a_short_outage_is_waited_out(setup: Setup) -> None:
    """A few unanswered questions do not fail the document."""
    setup.transcriber.answers = [DOWN, DOWN, MediaJob("running")]
    doc_id = setup.upload()
    setup.run()
    assert setup.store.get("acme", doc_id).status == "indexed"


def test_deleted_while_transcribed(setup: Setup) -> None:
    """The job is deleted at the transcriber once it ends; nothing comes
    back."""
    doc_id = setup.upload()
    calls: List[str] = []

    def running(job_id: str) -> MediaJob:
        """Still running; the person deletes the document meanwhile."""
        calls.append(job_id)
        if len(calls) == 1:
            setup.library.delete("acme", doc_id)
        return MediaJob("running")

    setup.transcriber.status = running  # type: ignore[method-assign]
    setup.transcriber.busy = 2
    setup.run()
    assert len(calls) == 1
    assert setup.transcriber.forgotten == [media_job_id("acme", doc_id)]
    assert setup.clock.slept == [5, 5, 5]
    assert setup.store.list("acme") == []
    assert setup.indexes.files.get("acme", {}) == {}


def test_deleted_just_as_the_job_ends(setup: Setup) -> None:
    """A document deleted during the last question is not written."""
    doc_id = setup.upload()
    seen: List[str] = []

    def last_status(job_id: str) -> MediaJob:
        """Done, but the document goes meanwhile."""
        seen.append(job_id)
        setup.store.delete("acme", doc_id)
        return MediaJob("done", transcript=setup.transcriber.transcript)

    setup.transcriber.status = last_status  # type: ignore[method-assign]
    setup.run()
    assert seen and setup.store.list("acme") == []
    assert setup.indexes.files.get("acme", {}) == {}


def test_without_a_transcriber_audio_is_refused() -> None:
    """No link to the transcriber: the upload is refused at once."""
    plain = Setup(linked=False)
    with pytest.raises(UnsupportedDocumentError, match="transcripción"):
        plain.upload()
    assert plain.store.list("acme") == []
    with pytest.raises(ValueError):
        KnowledgeLibrary(
            plain.library.registry, plain.store, FakeExtractor(),
            plain.queue, UploadLimits(), lambda: "", plain.media, None,
        )


def test_a_recording_is_not_extracted_and_text_is_needed(
    setup: Setup,
) -> None:
    """Indexing a recording without its transcript fails plainly."""
    doc_id = setup.upload()
    setup.media_queue.jobs.clear()
    setup.library.process("acme", doc_id)
    failed = setup.store.get("acme", doc_id)
    assert failed.status == "failed"
    assert failed.error is not None and "not transcribed" in failed.error


def test_recover_sends_unfinished_recordings_again(setup: Setup) -> None:
    """After a restart a queued recording goes to the transcriber."""
    doc_id = setup.upload()
    setup.media_queue.jobs.clear()
    assert setup.library.recover() == 1
    assert len(setup.media_queue.jobs) == 1
    setup.run()
    assert setup.store.get("acme", doc_id).status == "indexed"


def test_unexpected_errors_never_leave_it_processing(setup: Setup) -> None:
    """A bug in a port marks the document failed."""
    def broken(*_: object) -> MediaJob:
        """A transcriber adapter with a bug."""
        raise RuntimeError("bug")

    setup.transcriber.submit = broken  # type: ignore[method-assign,assignment]
    doc_id = setup.upload()
    setup.run()
    failed = setup.store.get("acme", doc_id)
    assert (failed.status, failed.error) == (
        "failed", "Error interno / Internal error: RuntimeError."
    )


def test_a_document_gone_before_its_turn(setup: Setup) -> None:
    """Deleted before the media worker reaches it: nothing happens."""
    doc_id = setup.upload()
    setup.store.delete("acme", doc_id)
    setup.run()
    assert setup.transcriber.submitted == []


def test_media_transcription_refuses_cleanly() -> None:
    """The use case on its own: a refused format is an input error."""
    transcriber = FakeMediaTranscriber()
    transcriber.submit_error = UnsupportedDocumentError("no")
    media = MediaTranscription(transcriber, 60, sleep=lambda s: None)
    with pytest.raises(InvalidInputError, match="no"):
        media.transcribe("kn-x", StoredFile("x.wav", 1), "wav", "es")
