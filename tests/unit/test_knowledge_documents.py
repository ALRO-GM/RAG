"""Uploaded documents: names, types, ids and indexed text."""

import pytest
from pydantic import ValidationError

from src.domain.errors import UnsupportedDocumentError
from src.domain.knowledge.documents import (
    DocumentMetadata,
    KnowledgeDocument,
    detect_type,
    document_id,
    document_text,
    metadata_errors,
    sanitise_filename,
    slug,
    type_from_name,
)
from src.domain.knowledge.domains import DomainManifest, Vocabularies
from src.domain.knowledge.sources import read_metadata, split_front_matter


@pytest.mark.parametrize("name, expected", [
    ("Manual de ventas.pdf", "Manual de ventas.pdf"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\evil.docx", "evil.docx"),
    ("/abs/path/informe.odt", "informe.odt"),
    (".hidden.txt", "hidden.txt"),
    ("a\x00b\nc.md", "a_b_c.md"),
    ("Política <v2>?.html", "Política _v2__.html"),
    ("", "document"),
    ("...", "document"),
    ("x" * 300 + ".xlsx", "x" * 115 + ".xlsx"),
])
def test_sanitise_filename(name: str, expected: str) -> None:
    """Folders, controls and odd characters never reach the disk."""
    assert sanitise_filename(name) == expected
    assert "/" not in sanitise_filename(name)
    assert len(sanitise_filename(name)) <= 120


@pytest.mark.parametrize("name, head, kind", [
    ("a.pdf", b"%PDF-1.7\n...", "pdf"),
    ("a.PDF", b"\n\n%PDF-1.4", "pdf"),
    ("a.docx", b"PK\x03\x04rest", "docx"),
    ("a.odt", b"PK\x03\x04rest", "odt"),
    ("a.xlsx", b"PK\x03\x04rest", "xlsx"),
    ("a.csv", b"a;b\n1;2\n", "csv"),
    ("a.htm", b"<html><p>x</p>", "html"),
    ("a.markdown", "# Título\n".encode(), "md"),
    ("a.txt", b"\xef\xbb\xbfplain", "txt"),
])
def test_detect_type(name: str, head: bytes, kind: str) -> None:
    """The extension chooses among types whose signature matches."""
    assert detect_type(name, head) == kind


@pytest.mark.parametrize("name, head, message", [
    ("a.pdf", b"MZ\x90\x00 an exe", "not a valid PDF"),
    ("a.docx", b"%PDF-1.7", "not a valid DOCX"),
    ("a.txt", b"PK\x03\x04", "not a valid TXT"),
    ("a.txt", b"text\x00with nul", "not a valid TXT"),
    ("a.md", b"%PDF-1.4", "not a valid MD"),
    ("a.doc", b"\xd0\xcf\x11\xe0\xa1\xb1", "Legacy Office"),
    ("a.xls", b"\xd0\xcf\x11\xe0\xa1\xb1", "Legacy Office"),
    ("a.exe", b"MZ", "Unsupported file type"),
    ("noextension", b"hello", "Unsupported file type"),
])
def test_detect_type_refuses(name: str, head: bytes, message: str) -> None:
    """A wrong signature, legacy Office or an unknown type is refused."""
    with pytest.raises(UnsupportedDocumentError, match=message):
        detect_type(name, head)


def test_type_from_name() -> None:
    """Extensions are case-insensitive."""
    assert type_from_name("X.HTML") == "html"
    assert type_from_name("x.pptx") is None


def test_document_ids_are_readable_and_stable() -> None:
    """Same name and bytes, same id; another content, another id."""
    a = document_id("Política de Devoluciones.pdf", b"one")
    assert a.startswith("politica-de-devoluciones-")
    assert a == document_id("Política de Devoluciones.pdf", b"one")
    assert a != document_id("Política de Devoluciones.pdf", b"two")
    assert document_id("¿¿.txt", b"x").startswith("doc-")
    assert slug("Ñandú  Größe") == "nandu-groe"


def manifest() -> DomainManifest:
    """Return a manifest with vocabularies."""
    return DomainManifest(id="d", title="D", vocabularies=Vocabularies(
        intent=["refund"], tags=["faq"],
    ))


def test_document_text_carries_the_metadata() -> None:
    """The front matter gives every chunk the document's labels."""
    document = KnowledgeDocument(
        id="x-1", name="Guía: devoluciones.docx", type="docx", size=3,
        uploaded_at="2026-11-21T10:00:00+00:00",
        metadata=DocumentMetadata(intent="refund", tags=["faq", "faq"],
                                  category="informacion"),
    )
    text = document_text(document, "\n# Plazos\n\nTreinta días.\n\n")
    front = split_front_matter("x.md", text)
    meta = read_metadata(front.fields, manifest(), lambda k: "x")
    assert meta.title == "Guía: devoluciones.docx"
    assert (meta.intent, meta.category, meta.tags) == \
        ("refund", "informacion", ["faq"])
    assert text[front.body_start:] == "# Plazos\n\nTreinta días.\n"
    bare = document.model_copy(update={"metadata": DocumentMetadata()})
    assert "intent" not in document_text(bare, "x")


def test_metadata_errors_and_validation() -> None:
    """Vocabularies apply to uploads; unknown keys are refused."""
    assert metadata_errors(DocumentMetadata(intent="refund"), manifest()) \
        == []
    errors = metadata_errors(
        DocumentMetadata(intent="other", tags=["vip"], category="x"),
        manifest(),
    )
    assert len(errors) == 3
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate({"colour": "red"})
    assert DocumentMetadata(intent="", tags=["a", "", "a"]) \
        .chunk_metadata().model_dump() == {
            "title": None, "ref": None, "intent": None, "category": None,
            "script_phase": None, "tags": ["a"],
        }
