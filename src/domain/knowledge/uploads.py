"""Large uploads of the knowledge library: limits by kind, pieces, disk.

A document reaches a domain either in one request (small files, and the
API's other callers) or in pieces (``docs/KNOWLEDGE.md`` §5): the page
announces the file (name, size, metadata), sends it in pieces of at
most :data:`PIECE_BYTES` at increasing offsets, can ask how much
arrived and go on from there after a cut, and finally completes it.
Until then the bytes are a :class:`PartialUpload`, kept on disk and
swept when abandoned.

The largest file accepted depends on its kind (:class:`UploadLimits`):
a PDF, a video, an audio file or any other document. The kind is the
one detected from the file's first bytes
(:func:`~src.domain.knowledge.documents.detect_type`), so a file cannot
pass for a video by its name alone: only a ``.mp4``/``.webm``/``.mov``/
``.3gp`` whose content starts like a video container (ISO media or
Matroska/WebM) gets the video limit.
"""

import re
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.domain.errors import InvalidInputError
from src.domain.knowledge.documents import DocumentMetadata, KnowledgeDocument
from src.domain.knowledge.media import ISO_BOXES, MEDIA_TYPES

MiB = 1024 * 1024

PIECE_BYTES = 8 * MiB
"""Largest piece of a large upload: well under what the tunnel's
container holds in memory (64 MiB, ``docs/DEPLOYMENT.md``)."""

UPLOAD_ID_PATTERN = r"[0-9a-f]{32}"
"""Ids of partial uploads (random, 128 bits)."""

VIDEO_TYPES = ("mp4", "webm", "mov", "3gp")
"""Media types that may hold video (the others are audio only)."""

EBML = b"\x1a\x45\xdf\xa3"
"""Start of a Matroska or WebM file."""

LimitKind = Literal["pdf", "video", "audio", "document"]
"""The kinds the upload limits are set for."""

KIND_NAMES = {
    "pdf": ("Los PDF", "PDF files"),
    "video": ("Los vídeos", "Videos"),
    "audio": ("Los audios", "Audio files"),
    "document": ("Estos documentos", "These documents"),
}
"""How the messages name each kind (Spanish, English)."""


class UploadLimits(BaseModel):
    """The largest file accepted of each kind, in bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pdf: int = Field(default=500 * MiB, ge=1)
    video: int = Field(default=1024 * MiB, ge=1)
    audio: int = Field(default=200 * MiB, ge=1)
    document: int = Field(default=100 * MiB, ge=1)

    def of(self, kind: LimitKind) -> int:
        """Return the limit of ``kind``."""
        value: int = getattr(self, kind)
        return value

    @property
    def largest(self) -> int:
        """Return the largest limit of any kind."""
        return max(self.pdf, self.video, self.audio, self.document)


def is_video_container(head: bytes) -> bool:
    """Tell whether ``head`` starts like a container that holds video."""
    return head.startswith(EBML) or head[4:8] in ISO_BOXES


def limit_kind(document_type: str, head: Optional[bytes] = None) -> LimitKind:
    """Return which limit applies to a file of ``document_type``.

    ``head`` is the start of the file: a video type gets the video
    limit only when the content is a video container. Without ``head``
    (the file is only announced) the type alone decides.
    """
    if document_type == "pdf":
        return "pdf"
    if document_type in VIDEO_TYPES:
        if head is None or is_video_container(head):
            return "video"
        return "audio"
    if document_type in MEDIA_TYPES:
        return "audio"
    return "document"


def megabytes(size: int) -> str:
    """Return ``size`` bytes as whole megabytes (``"500 MB"``)."""
    return f"{max(size, 0) // MiB} MB"


def too_large(kind: LimitKind, limit: int, size: int) -> str:
    """Return the bilingual message of a file over its kind's limit."""
    spanish, english = KIND_NAMES[kind]
    return (
        f"{spanish} pueden ocupar hasta {megabytes(limit)}; este ocupa "
        f"{megabytes(size)}. / {english} may be up to {megabytes(limit)}; "
        f"this one is {megabytes(size)}."
    )


def no_space(needed: int) -> str:
    """Return the bilingual message of a disk that would not hold a file."""
    return (
        f"No hay espacio en el disco del servidor para este archivo "
        f"(necesita unos {megabytes(needed)} libres); borra documentos o "
        f"inténtalo más tarde. / The server's disk has no room for this "
        f"file (it needs about {megabytes(needed)} free); delete documents "
        "or try again later."
    )


def space_needed(document_type: str, size: int) -> int:
    """Return the disk an upload of ``size`` bytes takes at its peak.

    The partial file becomes the stored original (renamed, not copied);
    a recording is also copied to the transcriber while it is
    transcribed (deleted afterwards).
    """
    copies = 2 if document_type in MEDIA_TYPES else 1
    return copies * size


def check_upload_id(value: str) -> str:
    """Return ``value`` if it is a partial upload's id, else raise."""
    if not re.fullmatch(UPLOAD_ID_PATTERN, value or ""):
        raise InvalidInputError(f"Invalid upload id {value!r}.")
    return value


class PartialUpload(BaseModel):
    """A file being received in pieces.

    ``received`` bytes are on disk, from the start; ``checked`` says
    the content's type was checked against the name (once its first
    bytes arrived); ``touched`` is when a piece last arrived (seconds
    since the epoch), for the sweep; ``document`` is the document it
    became once completed (kept so a repeated completion answers the
    same).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str
    name: str
    type: str
    size: int = Field(ge=1)
    received: int = Field(default=0, ge=0)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    checked: bool = False
    touched: float = 0.0
    document: Optional[KnowledgeDocument] = None

    @property
    def complete(self) -> bool:
        """Tell whether every byte arrived."""
        return self.received >= self.size

    def view(self) -> Dict[str, Any]:
        """Return what the API answers about it."""
        return {
            "id": self.id, "domain": self.domain, "name": self.name,
            "type": self.type, "size": self.size,
            "received": self.received, "piece_bytes": PIECE_BYTES,
            "document": (
                self.document.model_dump(mode="json")
                if self.document is not None else None
            ),
        }
