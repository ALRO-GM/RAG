"""Documents uploaded to a knowledge domain: records, names and types.

An upload is kept as it came (``originals/``), its text is extracted
in the background into Markdown whose front matter carries the
document's metadata, and that file is indexed like any authored
source. The type is decided by the file's first bytes, not only by its
name; legacy binary Office files and anything unknown are refused.
Audio and video files (:mod:`src.domain.knowledge.media`) are
transcribed instead of extracted.
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.domain.errors import UnsupportedDocumentError
from src.domain.knowledge.domains import DomainManifest, metadata_problems
from src.domain.knowledge.media import (
    MEDIA_EXTENSIONS,
    MEDIA_TYPES,
    TIMED_KEY,
    looks_like_media,
)
from src.domain.models import ChunkMetadata

DocumentStatus = Literal["queued", "processing", "indexed", "failed"]

DOCUMENT_ID_PATTERN = r"[a-z0-9][a-z0-9-]{0,79}"
"""Ids of uploaded documents (a slug of the name plus a content hash)."""

EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "pdf": (".pdf",),
    "docx": (".docx",),
    "odt": (".odt",),
    "xlsx": (".xlsx",),
    "csv": (".csv",),
    "html": (".html", ".htm"),
    "md": (".md", ".markdown"),
    "txt": (".txt",),
    "png": (".png",),
    "jpg": (".jpg", ".jpeg"),
    "tiff": (".tif", ".tiff"),
    "webp": (".webp",),
    **MEDIA_EXTENSIONS,
}
"""Supported types and the file extensions that announce them."""

IMAGE_SIGNATURES: Dict[str, Tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),
    "webp": (b"RIFF",),
}
"""First bytes of each image type (WebP also has ``WEBP`` at byte 8)."""
IMAGE_TYPES = tuple(IMAGE_SIGNATURES)

ZIP_TYPES = ("docx", "odt", "xlsx")
TEXT_TYPES = ("csv", "html", "md", "txt")
MAX_NAME = 120
"""Longest stored file name (characters)."""


class DocumentMetadata(BaseModel):
    """Labels given to every chunk of an uploaded document."""

    model_config = ConfigDict(extra="forbid")

    category: Optional[str] = Field(default=None, max_length=200)
    intent: Optional[str] = Field(default=None, max_length=200)
    script_phase: Optional[str] = Field(default=None, max_length=200)
    tags: List[str] = Field(default_factory=list, max_length=64)

    def chunk_metadata(self, title: Optional[str] = None) -> ChunkMetadata:
        """Return these labels as chunk metadata with ``title``."""
        return ChunkMetadata(
            title=title, intent=self.intent or None,
            category=self.category or None,
            script_phase=self.script_phase or None,
            tags=[t for t in dict.fromkeys(self.tags) if t],
        )


class KnowledgeDocument(BaseModel):
    """An uploaded document and where its processing is."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str
    size: int
    pages: Optional[int] = None
    chunks: int = 0
    status: DocumentStatus = "queued"
    error: Optional[str] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    uploaded_at: str


def metadata_errors(
    metadata: DocumentMetadata, manifest: DomainManifest
) -> List[str]:
    """Return why ``manifest`` refuses ``metadata`` (empty: accepted)."""
    return [
        message for _, message in
        metadata_problems(metadata.chunk_metadata(), manifest)
    ]


def sanitise_filename(name: str) -> str:
    """Return a safe file name: no folders, controls or odd characters.

    Only the last path component is kept (``../../x.pdf`` → ``x.pdf``);
    characters other than letters, digits, spaces, ``.``, ``-``, ``_``
    and parentheses become ``_``; leading dots are dropped and the
    stem is shortened so the whole name fits :data:`MAX_NAME`.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    base = unicodedata.normalize("NFC", base)
    cleaned = "".join(
        c if (c.isalnum() or c in " .-_()") and c.isprintable() else "_"
        for c in base
    ).strip(" .")
    stem, dot, ext = cleaned.rpartition(".")
    if not dot:
        stem, ext = cleaned, ""
    ext = ext[:10]
    stem = stem[:MAX_NAME - len(ext) - 1].strip(" .") or "document"
    return f"{stem}.{ext}" if ext else stem


def type_from_name(name: str) -> Optional[str]:
    """Return the type announced by a file name's extension, if any."""
    lower = name.lower()
    for kind, extensions in EXTENSIONS.items():
        if lower.endswith(extensions):
            return kind
    return None


def supported_extensions() -> str:
    """Return the supported extensions as text, for messages."""
    return ", ".join(e for exts in EXTENSIONS.values() for e in exts)


def detect_type(name: str, head: bytes) -> str:
    """Return the document type, checked against the file's first bytes.

    ``head`` is the start of the file (at least 1 KiB when the file is
    that long). The extension chooses among the types whose signature
    matches: ``%PDF-`` for PDF, a ZIP header for DOCX/ODT/XLSX (their
    inside is checked at extraction) and, for text types, no NUL byte
    and no binary signature; for audio and video, the start of any
    audio or video container.
    """
    kind = type_from_name(name)
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise UnsupportedDocumentError(
            "Formato antiguo de Office (.doc/.xls/.ppt) no admitido: "
            "guárdalo como .docx, .xlsx o PDF. / Legacy Office format "
            "(.doc/.xls/.ppt) is not supported: save it as .docx, .xlsx "
            "or PDF."
        )
    if kind is None:
        raise UnsupportedDocumentError(
            f"Tipo de archivo no admitido / Unsupported file type: "
            f"{name!r}. Use {supported_extensions()}."
        )
    is_pdf = b"%PDF-" in head[:1024]
    is_zip = head.startswith(b"PK\x03\x04")
    if kind in MEDIA_TYPES:
        if not looks_like_media(head):
            raise UnsupportedDocumentError(
                f"El contenido de {name!r} no es un audio o vídeo "
                f"reconocible. / The content of {name!r} is not a "
                "recognisable audio or video file."
            )
        return kind
    if kind in IMAGE_TYPES:
        starts = head.startswith(IMAGE_SIGNATURES[kind])
        if kind == "webp":
            starts = starts and head[8:12] == b"WEBP"
        if not starts:
            raise UnsupportedDocumentError(
                f"El contenido de {name!r} no es una imagen "
                f"{kind.upper()} válida. / The content of {name!r} is not "
                f"a valid {kind.upper()} image."
            )
        return kind
    ok = {
        "pdf": is_pdf,
        "docx": is_zip, "odt": is_zip, "xlsx": is_zip,
    }.get(kind, not (is_pdf or is_zip or b"\x00" in head))
    if not ok:
        raise UnsupportedDocumentError(
            f"El contenido de {name!r} no es un {kind.upper()} válido. / "
            f"The content of {name!r} is not a valid {kind.upper()}."
        )
    return kind


def slug(text: str) -> str:
    """Return ``text`` as lowercase ASCII words joined by ``-``."""
    folded = unicodedata.normalize("NFKD", text)
    ascii_text = folded.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")


def document_id(name: str, data: bytes) -> str:
    """Return a document's id: its name as a slug plus a content hash.

    The same file uploaded twice gets the same id, so a duplicate is
    noticed; the slug keeps ids readable in chunk ids and paths.
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    digest = hashlib.sha256(data).hexdigest()[:8]
    return f"{(slug(stem)[:64].strip('-') or 'doc')}-{digest}"


def is_media(document: KnowledgeDocument) -> bool:
    """Tell whether a document is an audio or video file (transcribed)."""
    return document.type in MEDIA_TYPES


def document_text(document: KnowledgeDocument, body: str) -> str:
    """Return the indexed text of a document: front matter + Markdown.

    The front matter gives every chunk the document's metadata and its
    name as title (for chunks before any heading); a transcribed
    recording also says ``timed: true`` (its chunks carry their times).
    """
    m = document.metadata
    fields: Dict[str, object] = {"title": document.name}
    for key, value in (
        ("intent", m.intent), ("category", m.category),
        ("script_phase", m.script_phase),
    ):
        if value:
            fields[key] = value
    if m.tags:
        fields["tags"] = list(m.tags)
    if is_media(document):
        fields[TIMED_KEY] = True
    head = yaml.safe_dump(fields, allow_unicode=True, sort_keys=False)
    return f"---\n{head}---\n{body.strip()}\n"
