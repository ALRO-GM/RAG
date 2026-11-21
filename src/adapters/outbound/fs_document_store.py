"""Document store adapter: a domain's uploads on disk.

Layout under ``<data_root>/<domain>/``::

    documents/<id>.json   the KnowledgeDocument record
    originals/<id><ext>   the uploaded bytes, as received
    extracted/<id>.md     front matter + extracted Markdown (indexed)

Ids are checked against their patterns before any path is built, so
no request can name a file outside these folders.
"""

import os
import re
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
)
from src.domain.knowledge.documents import (
    DOCUMENT_ID_PATTERN,
    EXTENSIONS,
    KnowledgeDocument,
)
from src.domain.knowledge.domains import check_domain_id


def check_document_id(value: str) -> str:
    """Return ``value`` if it is a valid document id, else raise."""
    if not re.fullmatch(DOCUMENT_ID_PATTERN, value or ""):
        raise InvalidInputError(f"Invalid document id {value!r}.")
    return value


def write_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


class FileSystemDocumentStore:
    """Keep each domain's uploaded documents under ``data_root``."""

    def __init__(self, data_root: str) -> None:
        """Store under ``data_root`` (``data/knowledge``)."""
        self.root = Path(data_root)

    def _folder(self, domain_id: str, kind: str) -> Path:
        """Return ``<root>/<domain>/<kind>``."""
        return self.root / check_domain_id(domain_id) / kind

    def _record(self, domain_id: str, document_id: str) -> Path:
        """Return the record file of a document."""
        return self._folder(domain_id, "documents") / \
            f"{check_document_id(document_id)}.json"

    def list(self, domain_id: str) -> List[KnowledgeDocument]:
        """Return the domain's documents, oldest upload first."""
        folder = self._folder(domain_id, "documents")
        if not folder.is_dir():
            return []
        found = [
            self._read(path) for path in sorted(folder.glob("*.json"))
        ]
        return sorted(found, key=lambda d: (d.uploaded_at, d.id))

    def _read(self, path: Path) -> KnowledgeDocument:
        """Read one record."""
        try:
            return KnowledgeDocument.model_validate_json(path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise CorruptedDataError(f"{path}: unreadable record ({exc}).")

    def get(self, domain_id: str, document_id: str) -> KnowledgeDocument:
        """Return one document's record."""
        path = self._record(domain_id, document_id)
        if not path.is_file():
            raise NotFoundError(f"No document {document_id!r}.")
        return self._read(path)

    def save(self, domain_id: str, document: KnowledgeDocument) -> None:
        """Create or replace a document's record."""
        write_atomically(
            self._record(domain_id, document.id),
            document.model_dump_json(indent=2).encode("utf-8"),
        )

    def _original(self, domain_id: str, document: KnowledgeDocument) -> Path:
        """Return where a document's uploaded bytes are kept."""
        extension = EXTENSIONS.get(document.type, ("",))[0]
        return self._folder(domain_id, "originals") / \
            f"{check_document_id(document.id)}{extension}"

    def save_original(
        self, domain_id: str, document: KnowledgeDocument, data: bytes
    ) -> None:
        """Keep the uploaded bytes."""
        write_atomically(self._original(domain_id, document), data)

    def original(self, domain_id: str, document: KnowledgeDocument) -> bytes:
        """Return the uploaded bytes."""
        try:
            return self._original(domain_id, document).read_bytes()
        except OSError:
            raise NotFoundError(
                f"The upload of {document.id!r} is missing."
            )

    def text_path(self, domain_id: str, document_id: str) -> str:
        """Return the indexed path of a document's extracted text."""
        path = self._folder(domain_id, "extracted") / \
            f"{check_document_id(document_id)}.md"
        return path.as_posix()

    def read_text(self, domain_id: str, document_id: str) -> str:
        """Return the extracted text (``NotFoundError`` when missing)."""
        path = Path(self.text_path(domain_id, document_id))
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise NotFoundError(
                f"No hay texto extraído de {document_id!r}. / No extracted "
                f"text for {document_id!r}."
            ) from exc

    def write_text(self, domain_id: str, document_id: str, text: str) -> str:
        """Write the extracted text; return its indexed path."""
        path = self.text_path(domain_id, document_id)
        write_atomically(Path(path), text.encode("utf-8"))
        return path

    def remove_text(self, domain_id: str, document_id: str) -> None:
        """Delete the extracted text, if any."""
        Path(self.text_path(domain_id, document_id)).unlink(missing_ok=True)

    def delete(self, domain_id: str, document_id: str) -> None:
        """Delete a document's record, upload and extracted text."""
        document = self.get(domain_id, document_id)
        self.remove_text(domain_id, document_id)
        self._original(domain_id, document).unlink(missing_ok=True)
        self._record(domain_id, document_id).unlink(missing_ok=True)
