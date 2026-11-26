"""Text extraction from every uploadable type (generated files)."""

import pytest

from src.adapters.outbound import document_extractors
from src.adapters.outbound.document_extractors import (
    NO_PDF_TEXT,
    DocumentExtractors,
    decode_text,
    markdown_table,
)
from src.domain.errors import InvalidInputError, UnsupportedDocumentError
from src.domain.knowledge.documents import detect_type
from tests.knowledge_samples import (
    as_file,
    HTML,
    make_docx,
    make_odt,
    make_pdf,
    make_xlsx,
    make_zip,
)

EXTRACT = DocumentExtractors()


def test_pdf_pages_become_sections() -> None:
    """Each page with text is a ``## p. N`` section."""
    data = make_pdf(["Refund policy: thirty days.", "", "Contact support."])
    assert detect_type("a.pdf", data[:4096]) == "pdf"
    found = EXTRACT.extract(as_file(data), "pdf")
    assert found.pages == 3
    assert found.text.startswith("## p. 1\n\nRefund policy: thirty days.")
    assert "## p. 3\n\nContact support." in found.text
    assert "## p. 2" not in found.text


def test_pdf_without_text_fails_clearly() -> None:
    """A scanned PDF (no text layer) is refused in Spanish and English."""
    with pytest.raises(InvalidInputError) as caught:
        EXTRACT.extract(as_file(make_pdf(["", ""])), "pdf")
    assert str(caught.value) == NO_PDF_TEXT
    assert "OCR" in NO_PDF_TEXT and "escaneado" in NO_PDF_TEXT


def test_broken_pdf() -> None:
    """A truncated PDF is reported, not raised as a crash."""
    with pytest.raises(InvalidInputError):
        EXTRACT.extract(as_file(b"%PDF-1.4\n1 0 obj << broken"), "pdf")


def test_docx_headings_lists_and_tables() -> None:
    """Heading styles become Markdown headings; tables become tables."""
    data = make_docx()
    assert detect_type("a.docx", data[:4096]) == "docx"
    text = EXTRACT.extract(as_file(data), "docx").text
    assert text.startswith("# Manual de devoluciones\n\n# Plazos")
    assert "El cliente tiene treinta días" in text
    assert "- Ticket de compra" in text
    assert "| Producto | Plazo |\n| --- | --- |\n| Portátil | 15 días |" \
        in text


def test_odt_headings_and_tables() -> None:
    """Outline levels become heading levels."""
    data = make_odt()
    assert detect_type("a.odt", data[:4096]) == "odt"
    text = EXTRACT.extract(as_file(data), "odt").text
    assert text.startswith("# Política de envíos\n\n## Plazos\n\n")
    assert "Los envíos tardan 48 horas." in text
    assert "| Zona | Precio |" in text and "| Península | 5 EUR |" in text


def test_xlsx_sheets_become_sections() -> None:
    """Each non-empty sheet is a heading and a table."""
    found = EXTRACT.extract(as_file(make_xlsx()), "xlsx")
    assert found.pages == 2
    assert found.text.startswith("## Precios\n\n| Plan | Precio | Notas |")
    assert "| Pro | 25.5 | soporte \\| 24h |" in found.text
    assert "Vacía" not in found.text


@pytest.mark.parametrize("data", [
    "Plan;Precio\nBásico;10\n".encode("utf-8"),
    "Plan;Precio\nBásico;10\n".encode("cp1252"),
    "Plan\tPrecio\nBásico\t10\n".encode("utf-16"),
])
def test_csv_any_delimiter_and_encoding(data: bytes) -> None:
    """The delimiter is sniffed; UTF-8, UTF-16 and Windows-1252 decode."""
    text = EXTRACT.extract(as_file(data), "csv").text
    assert text == "| Plan | Precio |\n| --- | --- |\n| Básico | 10 |"


def test_html_visible_text_only() -> None:
    """Scripts and styles go; headings, lists and tables stay."""
    text = EXTRACT.extract(as_file(HTML.encode("utf-8")), "html").text
    assert text.startswith("# Centro de ayuda\n\nEscríbenos a soporte.")
    assert "## Horario" in text and "Lunes a viernes" in text
    assert "Canal | Tiempo" in text and "Email | 24 h" in text
    assert "alert" not in text and "color" not in text
    titled = EXTRACT.extract(
        as_file(b"<title>Solo</title><p>texto</p>"), "html"
    )
    assert titled.text == "# Solo\n\ntexto"


def test_markdown_and_text_as_they_are() -> None:
    """Markdown and text are only decoded."""
    md = "# Título\n\nCuerpo.\n".encode("utf-8")
    assert EXTRACT.extract(as_file(md), "md").text == "# Título\n\nCuerpo.\n"
    assert EXTRACT.extract(as_file(b"\xef\xbb\xbfhola"), "txt").text == "hola"


@pytest.mark.parametrize("data, kind", [
    (make_zip({"other.xml": b"x"}), "docx"),
    (make_zip({"word/document.xml": b"x"}), "xlsx"),
    (make_zip({"content.xml": b"x", "mimetype": b"application/zip"}),
     "odt"),
    (b"PK\x03\x04 not really a zip", "docx"),
])
def test_wrong_containers_are_refused(data: bytes, kind: str) -> None:
    """A ZIP that is not the announced Office file is refused."""
    with pytest.raises(UnsupportedDocumentError):
        EXTRACT.verify(as_file(data), kind)


def test_zip_bombs_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Archives that unpack beyond the limit are refused."""
    monkeypatch.setattr(document_extractors, "MAX_UNZIPPED", 100)
    with pytest.raises(UnsupportedDocumentError, match="too large"):
        EXTRACT.verify(
            as_file(make_zip({"word/document.xml": b"x" * 200})), "docx"
        )


def test_empty_and_huge_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """No text, too much text or an unknown type are refused."""
    with pytest.raises(InvalidInputError, match="no text"):
        EXTRACT.extract(as_file(b"   \n"), "txt")
    monkeypatch.setattr(document_extractors, "MAX_TEXT", 5)
    with pytest.raises(InvalidInputError, match="more than 5"):
        EXTRACT.extract(as_file(b"too long text"), "txt")
    with pytest.raises(UnsupportedDocumentError):
        EXTRACT.extract(as_file(b"x"), "pptx")
    with pytest.raises(InvalidInputError, match="Cannot read"):
        EXTRACT.extract(
            as_file(make_zip({"word/document.xml": b"<bad"})), "docx"
        )


def test_helpers() -> None:
    """Tables drop empty trailing cells and rows; text decodes."""
    assert markdown_table([["a", None, ""], [], [None]]) == \
        "| a |\n| --- |"
    assert decode_text("é".encode("cp1252")) == "é"
