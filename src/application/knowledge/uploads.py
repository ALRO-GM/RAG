"""Use case: receive a document in pieces, with resume (#282).

A large file (a 500 MB PDF, a 1 GB video) does not travel in one
request: a cut of the connection would start it over, and every hop
would hold it whole in memory. The page announces it (:meth:`start`:
name, size, metadata; refused at once when the name's type is not
supported, the size is over the limit of that type or the disk would
not hold it), sends pieces of at most
:data:`~src.domain.knowledge.uploads.PIECE_BYTES` at increasing offsets
(:meth:`put`), may ask how much arrived (:meth:`status`) to go on after
a cut, and completes it (:meth:`complete`): the file is hashed in
pieces for its id, checked like any upload and moved (not copied) into
the document store by :meth:`~src.application.knowledge.library.
KnowledgeLibrary.accept`.

The content's type is checked as soon as its first bytes arrive: a
file whose content does not match its name, or whose detected kind has
a smaller limit than its size (an audio file named ``.mp4``), is
refused there, not after the whole file was sent.

An upload not touched for ``ttl_seconds`` is deleted by :meth:`sweep`
(at start and every hour), and so is the record of a completed one.
Nothing uploaded is logged.
"""

import hashlib
import io
import logging
import threading
import time
import uuid
from typing import Callable, Optional

from src.application.knowledge.library import (
    SNIFF_BYTES,
    KnowledgeLibrary,
    ReceivedFile,
)
from src.application.knowledge.ports import UploadArea
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NoSpaceError,
    NotFoundError,
    RagError,
    TooLargeError,
    UnsupportedDocumentError,
)
from src.domain.knowledge.documents import (
    DocumentMetadata,
    KnowledgeDocument,
    detect_type,
    sanitise_filename,
    supported_extensions,
    type_from_name,
)
from src.domain.knowledge.uploads import (
    PIECE_BYTES,
    PartialUpload,
    check_upload_id,
    limit_kind,
    no_space,
    space_needed,
    too_large,
)

LOG = logging.getLogger(__name__)

UPLOAD_TTL_SECONDS = 24 * 3600.0
"""How long an upload nobody sends to is kept (a night's pause)."""
STALE_SECONDS = 30 * 60.0
"""After this long without a piece an upload stops reserving disk (a
later piece reserves it again), and one that never got a byte (an answer
to ``start`` lost, a page closed) is swept."""


class DocumentUploads:
    """Receive documents in pieces for :class:`KnowledgeLibrary`.

    Args:
        library: Accepts the completed files.
        area: Where the pieces are written.
        min_free_bytes: Disk always left free (the margin).
        ttl_seconds: How long an untouched upload is kept.
        stale_seconds: See :data:`STALE_SECONDS`.
        clock: Seconds since the epoch (tests pass a fake).
        new_id: Returns a new upload id (tests pass a fake).
    """

    def __init__(
        self,
        library: KnowledgeLibrary,
        area: UploadArea,
        min_free_bytes: int,
        ttl_seconds: float = UPLOAD_TTL_SECONDS,
        stale_seconds: float = STALE_SECONDS,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        """Wire the library and the area."""
        self.library = library
        self.area = area
        self.min_free_bytes = min_free_bytes
        self.ttl_seconds = ttl_seconds
        self.stale_seconds = stale_seconds
        self.clock = clock
        self.new_id = new_id
        self._lock = threading.RLock()

    # Disk ---------------------------------------------------------------

    def _reserved(self) -> int:
        """Return the disk still promised to the uploads in progress.

        An upload without a piece for ``stale_seconds`` promises nothing.
        """
        total = 0
        stale = self.clock() - self.stale_seconds
        for upload_id in self.area.ids():
            try:
                upload = self.area.load(upload_id)
            except RagError:
                continue
            if upload.document is None and upload.touched >= stale:
                total += space_needed(upload.type, upload.size) - \
                    upload.received
        return total

    def _check_space(self, needed: int) -> None:
        """Refuse when ``needed`` bytes more would eat into the margin."""
        free = self.area.free_bytes()
        if free - self._reserved() - needed < self.min_free_bytes:
            raise NoSpaceError(no_space(needed + self.min_free_bytes))

    # Requests -----------------------------------------------------------

    def start(
        self,
        domain_id: str,
        filename: str,
        size: int,
        metadata: DocumentMetadata,
    ) -> PartialUpload:
        """Announce a file; return the upload to send its pieces to.

        Refused at once: an empty file, a name of no supported type, a
        size over that type's limit, audio or video with no transcriber
        linked, metadata outside the vocabularies, a disk that would
        not hold it with the margin (``NoSpaceError``).
        """
        self.library.check_metadata(domain_id, metadata)
        if size <= 0:
            raise InvalidInputError("El archivo está vacío. / Empty file.")
        name = sanitise_filename(filename)
        kind = type_from_name(name)
        if kind is None:
            raise UnsupportedDocumentError(
                f"Tipo de archivo no admitido / Unsupported file type: "
                f"{name!r}. Use {supported_extensions()}."
            )
        self.library.check_kind(kind)
        which = limit_kind(kind)
        limit = self.library.limits.of(which)
        if size > limit:
            raise TooLargeError(too_large(which, limit, size))
        upload = PartialUpload(
            id=check_upload_id(self.new_id()), domain=domain_id, name=name,
            type=kind, size=size, metadata=metadata, touched=self.clock(),
        )
        with self._lock:
            self._check_space(space_needed(kind, size))
            self.area.create(upload)
        return upload

    def _upload(self, domain_id: str, upload_id: str) -> PartialUpload:
        """Return an upload of ``domain_id`` (``NotFoundError`` if none)."""
        upload = self.area.load(check_upload_id(upload_id))
        if upload.domain != domain_id:
            raise NotFoundError(f"No upload {upload_id!r} in {domain_id!r}.")
        return upload

    def status(self, domain_id: str, upload_id: str) -> PartialUpload:
        """Return an upload: how much arrived, the document it became."""
        return self._upload(domain_id, upload_id)

    def put(
        self, domain_id: str, upload_id: str, offset: int, data: bytes
    ) -> PartialUpload:
        """Write one piece at ``offset``; return the upload.

        A piece already received (sent again after a lost answer) is
        accepted without writing; one that starts past what arrived
        would leave a gap and is refused (``BusyError``, with the
        offset expected: ask :meth:`status` and go on from there). A
        piece over :data:`PIECE_BYTES`, or past the announced size, is
        refused. Once the first bytes are there the content's type is
        checked: a mismatch deletes the upload.
        """
        if len(data) > PIECE_BYTES:
            raise TooLargeError(
                f"Una parte no puede pasar de {PIECE_BYTES} bytes. / A "
                f"piece may not be larger than {PIECE_BYTES} bytes."
            )
        with self._lock:
            upload = self._upload(domain_id, upload_id)
            if upload.document is not None:
                return upload
            if offset < 0 or offset > upload.received:
                raise BusyError(
                    f"Se esperaba la parte en {upload.received}. / "
                    f"Expected the piece at offset {upload.received}."
                )
            end = offset + len(data)
            if end > upload.size:
                raise InvalidInputError(
                    f"La parte pasa del tamaño anunciado ({upload.size} "
                    f"bytes). / The piece goes past the announced size "
                    f"({upload.size} bytes)."
                )
            if end <= upload.received:
                return upload  # already here: sent again after a cut
            fresh = data[upload.received - offset:]
            if self.area.free_bytes() - len(fresh) < \
                    self.min_free_bytes // 2:
                raise NoSpaceError(no_space(upload.size - upload.received))
            self.area.write(upload.id, upload.received, fresh)
            upload = upload.model_copy(update={
                "received": end, "touched": self.clock(),
            })
            if not upload.checked and \
                    upload.received >= min(upload.size, SNIFF_BYTES):
                upload = self._checked(upload)
            self.area.save(upload)
            return upload

    def _checked(self, upload: PartialUpload) -> PartialUpload:
        """Check the content's type and its limit; delete it if refused."""
        head = self.area.head(upload.id, SNIFF_BYTES)
        try:
            kind = detect_type(upload.name, head)
            which = limit_kind(kind, head)
            limit = self.library.limits.of(which)
            if upload.size > limit:
                raise TooLargeError(too_large(which, limit, upload.size))
        except RagError:
            self.area.discard(upload.id)
            raise
        return upload.model_copy(update={"checked": True})

    def cancel(self, domain_id: str, upload_id: str) -> None:
        """Delete an upload and what arrived of it."""
        with self._lock:
            upload = self._upload(domain_id, upload_id)
            self.area.discard(upload.id)

    def complete(self, domain_id: str, upload_id: str) -> KnowledgeDocument:
        """Turn a fully received upload into a document; return it.

        Completing it again returns the same document (the answer may
        have been lost). Refused while bytes are missing (``BusyError``).
        A file the library refuses (type, duplicate…) is deleted.
        """
        with self._lock:
            upload = self._upload(domain_id, upload_id)
            if upload.document is not None:
                return upload.document
            if not upload.complete:
                raise BusyError(
                    f"Faltan {upload.size - upload.received} bytes. / "
                    f"{upload.size - upload.received} bytes are missing."
                )
            if upload.sha256 is not None:
                kept = self._accepted(domain_id, upload)
                if kept is not None:
                    return kept
            digest = hashlib.sha256()
            for piece in self.area.read(upload.id):
                digest.update(piece)
            # The digest is recorded first: if the process stops after the
            # file was moved into the store, completing again finds it.
            upload = upload.model_copy(update={"sha256": digest.hexdigest()})
            self.area.save(upload)
            received = ReceivedFile(
                upload.name, self.area.file(upload.id),
                self.area.head(upload.id, SNIFF_BYTES), digest.hexdigest(),
            )
            try:
                document = self.library.accept(
                    domain_id, received, upload.metadata
                )
            except RagError:
                self.area.discard(upload.id)
                raise
            return self._done(upload, document)

    def _done(
        self, upload: PartialUpload, document: KnowledgeDocument
    ) -> KnowledgeDocument:
        """Record the document an upload became; drop its file."""
        self.area.drop_file(upload.id)
        self.area.save(upload.model_copy(update={
            "document": document, "touched": self.clock(),
        }))
        return document

    def _accepted(
        self, domain_id: str, upload: PartialUpload
    ) -> Optional[KnowledgeDocument]:
        """Return the document an interrupted completion already made.

        Only when the upload's file is gone (moved into the store) and
        the document of its name and digest exists.
        """
        try:
            self.area.file(upload.id)
            return None  # the file is still here: complete it normally
        except NotFoundError:
            pass
        assert upload.sha256 is not None
        document = self.library.find(domain_id, upload.name, upload.sha256)
        if document is None:
            raise NotFoundError(
                "La subida se perdió; vuelve a subir el archivo. / The "
                "upload was lost; upload the file again."
            )
        return self._done(upload, document)

    def receive(
        self,
        domain_id: str,
        filename: str,
        size: int,
        read: Callable[[int], bytes],
        metadata: DocumentMetadata,
    ) -> KnowledgeDocument:
        """Receive a whole file of ``size`` bytes in one go.

        ``read(n)`` returns the next bytes (at most ``n``; empty at the
        end): the single request of the API's other callers, read a
        piece at a time. The same checks as the pieces; nothing is kept
        to resume.
        """
        upload = self.start(domain_id, filename, size, metadata)
        try:
            offset = 0
            while True:
                piece = read(PIECE_BYTES)
                if not piece:
                    break
                self.put(domain_id, upload.id, offset, piece)
                offset += len(piece)
            if offset != size:
                raise InvalidInputError(
                    f"Llegaron {offset} bytes de {size}. / {offset} of "
                    f"{size} bytes arrived."
                )
            return self.complete(domain_id, upload.id)
        finally:
            self.area.discard(upload.id)  # nobody completes it again

    def upload(
        self,
        domain_id: str,
        filename: str,
        data: bytes,
        metadata: DocumentMetadata,
    ) -> KnowledgeDocument:
        """Receive a file held in memory (small files, tests)."""
        return self.receive(
            domain_id, filename, len(data), io.BytesIO(data).read, metadata
        )

    # Housekeeping -------------------------------------------------------

    def sweep(self) -> int:
        """Delete the uploads untouched for ``ttl_seconds``, and those
        that never got a byte in ``stale_seconds``; count them."""
        deadline = self.clock() - self.ttl_seconds
        stale = self.clock() - self.stale_seconds
        count = 0
        with self._lock:
            for upload_id in self.area.ids():
                try:
                    upload: Optional[PartialUpload] = self.area.load(
                        upload_id
                    )
                except RagError:
                    upload = None
                if upload is None or upload.touched < deadline or (
                    upload.received == 0 and upload.document is None
                    and upload.touched < stale
                ):
                    self.area.discard(upload_id)
                    count += 1
        if count:
            LOG.info("Swept %d abandoned upload(s)", count)
        return count
