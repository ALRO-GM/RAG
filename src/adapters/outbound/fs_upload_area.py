"""Upload area adapter: files being received in pieces, on disk.

Layout under ``<data_root>/.uploads/`` (a folder no domain id can name,
on the same disk as the originals, so a completed file is renamed into
place, never copied)::

    <id>.json   the PartialUpload record
    <id>.part   the bytes received so far (mode 600)

Pieces are written at their offset and flushed to disk before the
record says they arrived, so a crash never claims bytes that are not
there. The bytes are read back in pieces of :data:`READ_BYTES`.
"""

import os
import shutil
from pathlib import Path
from typing import Iterator, List

from pydantic import ValidationError

from src.adapters.outbound.fs_document_store import write_atomically
from src.application.knowledge.ports import StoredFile
from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
)
from src.domain.knowledge.uploads import PartialUpload, check_upload_id

UPLOADS_DIR = ".uploads"
READ_BYTES = 1024 * 1024
"""Bytes read at a time when the upload is hashed."""


class FileSystemUploadArea:
    """Keep the uploads in progress under ``<data_root>/.uploads``."""

    def __init__(self, data_root: str) -> None:
        """Keep under ``data_root`` (``data/knowledge``)."""
        self.root = Path(data_root) / UPLOADS_DIR
        self.data_root = Path(data_root)

    def _record(self, upload_id: str) -> Path:
        """Return the record file of an upload."""
        return self.root / f"{check_upload_id(upload_id)}.json"

    def _part(self, upload_id: str) -> Path:
        """Return the file of an upload's bytes."""
        return self.root / f"{check_upload_id(upload_id)}.part"

    def create(self, upload: PartialUpload) -> None:
        """Write the record and an empty file only the owner can read."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        part = self._part(upload.id)
        descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                             0o600)
        os.close(descriptor)
        self.save(upload)

    def load(self, upload_id: str) -> PartialUpload:
        """Return an upload's record."""
        path = self._record(upload_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None
        try:
            return PartialUpload.model_validate_json(raw)
        except ValidationError as exc:
            raise CorruptedDataError(f"{path}: unreadable record ({exc}).")

    def save(self, upload: PartialUpload) -> None:
        """Replace an upload's record."""
        write_atomically(
            self._record(upload.id),
            upload.model_dump_json().encode("utf-8"),
        )
        os.chmod(self._record(upload.id), 0o600)

    def write(self, upload_id: str, offset: int, data: bytes) -> None:
        """Write ``data`` at ``offset`` and flush it to disk."""
        part = self._part(upload_id)
        try:
            descriptor = os.open(part, os.O_WRONLY)
        except FileNotFoundError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None
        try:
            view = memoryview(data)
            while view:
                written = os.pwrite(descriptor, view, offset)
                view, offset = view[written:], offset + written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def head(self, upload_id: str, size: int) -> bytes:
        """Return the first ``size`` bytes."""
        try:
            with open(self._part(upload_id), "rb") as handle:
                return handle.read(size)
        except FileNotFoundError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None

    def read(self, upload_id: str) -> Iterator[bytes]:
        """Yield the bytes, :data:`READ_BYTES` at a time."""
        try:
            handle = open(self._part(upload_id), "rb")
        except FileNotFoundError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None
        with handle:
            while True:
                piece = handle.read(READ_BYTES)
                if not piece:
                    return
                yield piece

    def file(self, upload_id: str) -> StoredFile:
        """Return the file's path and size."""
        part = self._part(upload_id)
        try:
            return StoredFile(str(part), part.stat().st_size)
        except FileNotFoundError:
            raise NotFoundError(f"No upload {upload_id!r}.") from None

    def discard(self, upload_id: str) -> None:
        """Delete the file and the record."""
        self.drop_file(upload_id)
        self._record(upload_id).unlink(missing_ok=True)

    def drop_file(self, upload_id: str) -> None:
        """Delete the file only (it may already have been moved)."""
        self._part(upload_id).unlink(missing_ok=True)

    def ids(self) -> List[str]:
        """Return the ids of every record or file there."""
        if not self.root.is_dir():
            return []
        found = {
            p.stem for p in self.root.iterdir()
            if p.suffix in (".json", ".part")
        }
        valid = []
        for stem in sorted(found):
            try:
                valid.append(check_upload_id(stem))
            except InvalidInputError:  # a stray file: not ours, left
                continue
        return valid

    def free_bytes(self) -> int:
        """Return the free space of the disk of the data folder."""
        folder = self.root if self.root.is_dir() else self.data_root
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        return shutil.disk_usage(folder).free
