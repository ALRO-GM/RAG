"""Document extraction adapter: uploaded files → Markdown text.

Pure-Python libraries with permissive licences: ``pypdf`` (BSD-3) for
PDF, ``python-docx`` (MIT) for DOCX, ``odfpy`` (Apache-2.0) for ODT,
``openpyxl`` (MIT) for XLSX and the standard library for CSV, HTML,
Markdown and text. Headings become Markdown headings (they become the
chunks' titles), tables become Markdown tables. There is no OCR: a PDF
without a text layer fails with a clear message.
"""

import csv
import io
import re
import zipfile
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.application.knowledge.ports import ExtractedDocument
from src.domain.errors import InvalidInputError, UnsupportedDocumentError

MAX_TEXT = 20_000_000
"""Most characters of extracted text kept from one document."""
MAX_ROWS = 50_000
"""Most table rows read from one spreadsheet or CSV."""
MAX_UNZIPPED = 300 * 1024 * 1024
"""Largest total uncompressed size of a DOCX/ODT/XLSX (zip bombs)."""
MIN_PDF_TEXT = 20
"""Fewest non-blank characters for a PDF to count as having text."""

ZIP_MEMBERS = {
    "docx": "word/document.xml",
    "xlsx": "xl/workbook.xml",
    "odt": "content.xml",
}
ODT_MIMETYPE = b"application/vnd.oasis.opendocument.text"

NO_PDF_TEXT = (
    "El PDF no tiene texto extraíble (¿está escaneado?); no se hace OCR. "
    "/ The PDF has no extractable text (scanned?); OCR is not supported."
)


def decode_text(data: bytes) -> str:
    """Decode text bytes: UTF-8 (with or without BOM), else Windows-1252."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def markdown_table(rows: Iterable[Sequence[Any]]) -> str:
    """Return ``rows`` as a Markdown table (first row as header)."""
    lines: List[str] = []
    for index, row in enumerate(rows):
        cells = [
            re.sub(r"\s+", " ", "" if v is None else str(v)).replace(
                "|", "\\|"
            ).strip()
            for v in row
        ]
        while cells and not cells[-1]:
            cells.pop()
        if not cells:
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if len(lines) == 1:
            lines.append("|" + " --- |" * len(cells))
        if index >= MAX_ROWS:
            break
    return "\n".join(lines)


def check_zip(data: bytes, kind: str) -> None:
    """Check that a DOCX/ODT/XLSX is the ZIP container it claims to be."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UnsupportedDocumentError(
            f"El archivo no es un {kind.upper()} válido. / The file is not "
            f"a valid {kind.upper()}."
        )
    with archive:
        names = set(archive.namelist())
        total = sum(i.file_size for i in archive.infolist())
        if total > MAX_UNZIPPED:
            raise UnsupportedDocumentError(
                "El documento descomprimido es demasiado grande. / The "
                "document is too large once uncompressed."
            )
        valid = ZIP_MEMBERS[kind] in names
        if kind == "odt":
            valid = valid and "mimetype" in names and \
                archive.read("mimetype").strip() == ODT_MIMETYPE
    if not valid:
        raise UnsupportedDocumentError(
            f"El archivo no es un {kind.upper()} válido. / The file is not "
            f"a valid {kind.upper()}."
        )


def pdf_text(data: bytes) -> ExtractedDocument:
    """Return the text layer of a PDF, one ``## p. N`` section a page."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise InvalidInputError(
                "El PDF está protegido con contraseña. / The PDF is "
                "password-protected."
            )
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise InvalidInputError(
            f"No se pudo leer el PDF / Cannot read the PDF ({exc})."
        )
    if sum(len(p.strip()) for p in pages) < MIN_PDF_TEXT:
        raise InvalidInputError(NO_PDF_TEXT)
    sections = [
        f"## p. {n}\n\n{text.strip()}"
        for n, text in enumerate(pages, start=1) if text.strip()
    ]
    return ExtractedDocument("\n\n".join(sections), len(pages))


HEADING_STYLE = re.compile(r"^(heading|título|titulo|encabezado)\s*(\d)",
                           re.IGNORECASE)


def docx_text(data: bytes) -> ExtractedDocument:
    """Return a DOCX's paragraphs and tables in document order."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    blocks: List[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = item.style.name if item.style is not None else ""
            match = HEADING_STYLE.match(style or "")
            if match:
                text = "#" * min(int(match.group(2)), 6) + " " + text
            elif (style or "").lower() == "title":
                text = "# " + text
            elif "list" in (style or "").lower():
                text = "- " + text
            blocks.append(text)
        elif isinstance(item, Table):
            blocks.append(markdown_table(
                [cell.text for cell in row.cells] for row in item.rows
            ))
    return ExtractedDocument("\n\n".join(b for b in blocks if b))


def odt_text(data: bytes) -> ExtractedDocument:
    """Return an ODT's headings, paragraphs, lists and tables."""
    from odf import teletype
    from odf.namespaces import TABLENS, TEXTNS
    from odf.opendocument import load

    document = load(io.BytesIO(data))
    blocks: List[str] = []

    def walk(node: Any) -> None:
        """Append the blocks below ``node`` in document order."""
        for child in node.childNodes:
            qname = getattr(child, "qname", None)
            if qname == (TEXTNS, "h"):
                level = int(child.getAttribute("outlinelevel") or 1)
                text = teletype.extractText(child).strip()
                if text:
                    blocks.append("#" * min(level, 6) + " " + text)
            elif qname == (TEXTNS, "p"):
                text = teletype.extractText(child).strip()
                if text:
                    blocks.append(text)
            elif qname == (TABLENS, "table"):
                rows = [
                    [teletype.extractText(c).strip()
                     for c in r.childNodes
                     if getattr(c, "qname", None) == (TABLENS, "table-cell")]
                    for r in child.getElementsByType(_table_row())
                ]
                blocks.append(markdown_table(rows))
            elif hasattr(child, "childNodes"):
                walk(child)

    walk(document.text)
    return ExtractedDocument("\n\n".join(b for b in blocks if b))


def _table_row() -> Any:
    """Return odfpy's table-row element class."""
    from odf.table import TableRow

    return TableRow


def xlsx_text(data: bytes) -> ExtractedDocument:
    """Return each sheet of a workbook as a section with a table."""
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True,
                             data_only=True)
    sections: List[str] = []
    try:
        for sheet in workbook.worksheets:
            table = markdown_table(sheet.iter_rows(values_only=True))
            if table:
                sections.append(f"## {sheet.title}\n\n{table}")
    finally:
        workbook.close()
    return ExtractedDocument("\n\n".join(sections),
                             len(workbook.worksheets))


def csv_text(data: bytes) -> ExtractedDocument:
    """Return a CSV (any common delimiter) as a Markdown table."""
    text = decode_text(data)
    try:
        dialect: Any = csv.Sniffer().sniff(text[:8192], ",;\t|")
    except csv.Error:
        dialect = csv.excel
    return ExtractedDocument(
        markdown_table(csv.reader(io.StringIO(text), dialect))
    )


class _HtmlText(HTMLParser):
    """Collect the visible text of an HTML page as Markdown blocks."""

    BLOCKS = {"p", "div", "li", "tr", "br", "section", "article",
              "blockquote", "pre", "table", "ul", "ol", "dd", "dt"}
    SKIP = {"script", "style", "noscript", "template", "head", "svg"}

    def __init__(self) -> None:
        """Start empty."""
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skipping = 0
        self.heading = 0
        self.title = ""
        self.in_title = False

    def handle_starttag(
        self, tag: str, attrs: List[tuple[str, Optional[str]]]
    ) -> None:
        """Open a block, a heading or a skipped element."""
        if tag in self.SKIP:
            self.skipping += 1
        if tag == "title":
            self.in_title = True
        if re.fullmatch(r"h[1-6]", tag):
            self.heading = int(tag[1])
            self.parts.append("\n\n" + "#" * self.heading + " ")
        elif tag in self.BLOCKS:
            self.parts.append("\n\n" if tag != "br" else "\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        """Close a heading or a skipped element."""
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        if tag == "title":
            self.in_title = False
        if re.fullmatch(r"h[1-6]", tag):
            self.heading = 0
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        """Keep visible text (headings on one line)."""
        if self.in_title:
            self.title += data
            return
        if self.skipping:
            return
        text = re.sub(r"\s+", " ", data)
        if text.strip() or (self.parts and not self.parts[-1].endswith(" ")):
            self.parts.append(text)


def html_text(data: bytes) -> ExtractedDocument:
    """Return the visible text of an HTML page, headings kept."""
    parser = _HtmlText()
    parser.feed(decode_text(data))
    parser.close()
    body = "".join(parser.parts)
    blocks = [re.sub(r"[ \t]+", " ", b).strip(" |")
              for b in re.split(r"\n{2,}", body)]
    text = "\n\n".join(b.strip() for b in blocks if b.strip())
    title = parser.title.strip()
    if title and not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    return ExtractedDocument(text)


def plain_text(data: bytes) -> ExtractedDocument:
    """Return Markdown or text as it is."""
    return ExtractedDocument(decode_text(data))


EXTRACTORS: Dict[str, Callable[[bytes], ExtractedDocument]] = {
    "pdf": pdf_text,
    "docx": docx_text,
    "odt": odt_text,
    "xlsx": xlsx_text,
    "csv": csv_text,
    "html": html_text,
    "md": plain_text,
    "txt": plain_text,
}


class DocumentExtractors:
    """Extract text from every supported document type."""

    def verify(self, data: bytes, document_type: str) -> None:
        """Check the container of a DOCX/ODT/XLSX before accepting it."""
        if document_type in ZIP_MEMBERS:
            check_zip(data, document_type)

    def extract(self, data: bytes, document_type: str) -> ExtractedDocument:
        """Return the text of ``data``; ``InvalidInputError`` when none."""
        extractor = EXTRACTORS.get(document_type)
        if extractor is None:
            raise UnsupportedDocumentError(
                f"Unsupported document type {document_type!r}."
            )
        self.verify(data, document_type)
        try:
            found = extractor(data)
        except InvalidInputError:
            raise
        except Exception as exc:  # a broken file of any library
            raise InvalidInputError(
                f"No se pudo leer el documento / Cannot read the document "
                f"({type(exc).__name__}: {exc})."
            )
        if not found.text.strip():
            raise InvalidInputError(
                "El documento no contiene texto. / The document has no text."
            )
        if len(found.text) > MAX_TEXT:
            raise InvalidInputError(
                f"El documento tiene más de {MAX_TEXT} caracteres de texto. "
                f"/ The document has more than {MAX_TEXT} characters."
            )
        return found
