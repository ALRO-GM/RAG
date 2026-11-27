"""Document extraction adapter: uploaded files → Markdown text.

Pure-Python libraries with permissive licences: ``pypdf`` (BSD-3) for
PDF, ``python-docx`` (MIT) for DOCX, ``odfpy`` (Apache-2.0) for ODT,
``openpyxl`` (MIT) for XLSX and the standard library for CSV, HTML,
Markdown and text. Headings become Markdown headings (they become the
chunks' titles), tables become Markdown tables.

With a :class:`~src.application.knowledge.ports.PageLayoutReader`
(Docling, see :mod:`src.adapters.outbound.docling_pages`) PDF pages are
read with layout and tables, pages without a text layer are OCR'd, and
PNG/JPEG/TIFF/WebP images are accepted. Without one the PDF path is the
plain text layer and a scanned PDF or an image fails with a clear
message.

Files are read from their path, never loaded whole: a 500 MB PDF is
opened once for its text layer (``pypdf`` on an open file, which seeks)
and handed to the layout reader by path, a batch of pages at a time;
the figures of a batch are captioned and dropped before the next one,
and the caller is told the pages read so far.
"""

import csv
import io
import logging
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.application.knowledge.ports import (
    ExtractedDocument,
    ImageCaptioner,
    PageContent,
    PageLayoutReader,
    Progress,
    StoredFile,
)
from src.domain.errors import InvalidInputError, UnsupportedDocumentError
from src.domain.knowledge.documents import IMAGE_TYPES

LOG = logging.getLogger(__name__)

MAX_TEXT = 20_000_000
"""Most characters of extracted text kept from one document."""
MAX_ROWS = 50_000
"""Most table rows read from one spreadsheet or CSV."""
MAX_UNZIPPED = 300 * 1024 * 1024
"""Largest total uncompressed size of a DOCX/ODT/XLSX (zip bombs)."""
MIN_PDF_TEXT = 20
"""Fewest non-blank characters for a PDF to count as having text."""
MIN_PAGE_TEXT = 8
"""Fewest non-blank characters for a page to count as having a text layer."""
MAX_PIXELS = 100_000_000
"""Most pixels in one uploaded image (decompression bombs)."""
MAX_FRAMES = 500
"""Most frames in one multi-page TIFF."""
MIN_LAYOUT_RATIO = 0.6
"""Fewest characters, against the text layer, a layout page must keep.

Below it the layout model lost text (seen on some generated PDFs), so the
page's text layer is used instead."""
MAX_FIGURES = 200
"""Most figures captioned in one document (each costs seconds)."""
LAYOUT_BATCH_PAGES = 20
"""Pages handed to the layout reader at a time (memory, progress)."""

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
NO_OCR_IMAGE = (
    "Las imágenes necesitan OCR y no está instalado (grupo ocr). / "
    "Images need OCR, which is not installed (ocr group)."
)
NO_IMAGE_TEXT = (
    "No se encontró texto en la imagen. / No text found in the image."
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


def check_zip(path: str, kind: str) -> None:
    """Check that a DOCX/ODT/XLSX is the ZIP container it claims to be."""
    try:
        archive = zipfile.ZipFile(path)
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


def check_image(path: str, kind: str) -> None:
    """Check that an image decodes its header and is not a pixel bomb."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            width, height = image.size
            frames = int(getattr(image, "n_frames", 1))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise UnsupportedDocumentError(
            f"El archivo no es una imagen {kind.upper()} válida. / The "
            f"file is not a valid {kind.upper()} image."
        )
    if width * height > MAX_PIXELS or frames > MAX_FRAMES:
        raise UnsupportedDocumentError(
            "La imagen es demasiado grande. / The image is too large."
        )


def read_pdf_pages(path: str) -> List[str]:
    """Return the text layer of each page of a PDF (``""`` when none).

    The file is opened, not loaded: ``pypdf`` given a path would read it
    whole into memory, given an open file it seeks.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        with open(path, "rb") as handle:
            reader = PdfReader(handle)
            if reader.is_encrypted and not reader.decrypt(""):
                raise InvalidInputError(
                    "El PDF está protegido con contraseña. / The PDF is "
                    "password-protected."
                )
            return [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise InvalidInputError(
            f"No se pudo leer el PDF / Cannot read the PDF ({exc})."
        )


def letters(text: str) -> int:
    """Count the non-blank characters of ``text``."""
    return len("".join(text.split()))


def figure_notes(
    captioner: ImageCaptioner, figures: Sequence[bytes]
) -> str:
    """Return the figures as Markdown quotes: caption and text in them."""
    notes = []
    for index, image in enumerate(figures, 1):
        found = captioner.describe(image)
        if not (found.caption or found.text):
            continue
        lines = [f"> **Figura {index}:** {found.caption}".rstrip()]
        if found.text:
            lines.append(f"> Texto en la figura: {found.text}")
        notes.append("\n".join(lines))
    return "\n\n".join(notes)


def page_sections(pages: Dict[int, str]) -> str:
    """Join pages as ``## p. N`` sections; headings inside go one level down.

    The page number lands in the heading path of every chunk of the
    page (``p. 12 > Artículo 5``), which is how results cite it.
    """
    sections = []
    for number in sorted(pages):
        body = re.sub(r"(?m)^(#{1,5}) ", r"\1# ", pages[number].strip())
        body = body.replace("<!-- image -->", "").strip()
        if body:
            sections.append(f"## p. {number}\n\n{body}")
    return "\n\n".join(sections)


def pdf_text(path: str) -> ExtractedDocument:
    """Return the text layer of a PDF, one ``## p. N`` section a page."""
    pages = read_pdf_pages(path)
    if sum(len(p.strip()) for p in pages) < MIN_PDF_TEXT:
        raise InvalidInputError(NO_PDF_TEXT)
    return ExtractedDocument(
        page_sections({n: t for n, t in enumerate(pages, 1)}), len(pages)
    )


def page_runs(numbers: Sequence[int], size: int) -> List[List[int]]:
    """Split page numbers into runs of consecutive pages, ``size`` at most."""
    runs: List[List[int]] = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1 and len(runs[-1]) < size:
            runs[-1].append(number)
        else:
            runs.append([number])
    return runs


HEADING_STYLE = re.compile(r"^(heading|título|titulo|encabezado)\s*(\d)",
                           re.IGNORECASE)


def docx_text(path: str) -> ExtractedDocument:
    """Return a DOCX's paragraphs and tables in document order."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
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


def odt_text(path: str) -> ExtractedDocument:
    """Return an ODT's headings, paragraphs, lists and tables."""
    from odf import teletype
    from odf.namespaces import TABLENS, TEXTNS
    from odf.opendocument import load

    document = load(path)
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


def xlsx_text(path: str) -> ExtractedDocument:
    """Return each sheet of a workbook as a section with a table."""
    from openpyxl import load_workbook

    sections: List[str] = []
    with open(path, "rb") as handle:  # by name it wants an .xlsx suffix
        workbook = load_workbook(handle, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                table = markdown_table(sheet.iter_rows(values_only=True))
                if table:
                    sections.append(f"## {sheet.title}\n\n{table}")
        finally:
            workbook.close()
    return ExtractedDocument("\n\n".join(sections),
                             len(workbook.worksheets))


def csv_text(path: str) -> ExtractedDocument:
    """Return a CSV (any common delimiter) as a Markdown table."""
    text = decode_text(Path(path).read_bytes())
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


def html_text(path: str) -> ExtractedDocument:
    """Return the visible text of an HTML page, headings kept."""
    parser = _HtmlText()
    parser.feed(decode_text(Path(path).read_bytes()))
    parser.close()
    body = "".join(parser.parts)
    blocks = [re.sub(r"[ \t]+", " ", b).strip(" |")
              for b in re.split(r"\n{2,}", body)]
    text = "\n\n".join(b.strip() for b in blocks if b.strip())
    title = parser.title.strip()
    if title and not text.startswith("# "):
        text = f"# {title}\n\n{text}"
    return ExtractedDocument(text)


def plain_text(path: str) -> ExtractedDocument:
    """Return Markdown or text as it is."""
    return ExtractedDocument(decode_text(Path(path).read_bytes()))


EXTRACTORS: Dict[str, Callable[[str], ExtractedDocument]] = {
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
    """Extract text from every supported document type.

    Args:
        layout: Reads pages with layout, tables and OCR (Docling); without
            it PDFs use their text layer and images are refused.
        layout_text_pages: Also send pages that have a text layer to the
            layout reader (tables, reading order); ``False`` keeps the
            fast text layer for them and OCRs only the scanned pages.
        batch_pages: Pages per call to the layout reader.
        captioner: Describes the figures of PDFs and the images uploaded
            (a caption and their text become part of the page); a
            captioner that fails is dropped for the rest of the document.
        max_figures: Most figures captioned per document.
    """

    def __init__(
        self,
        layout: Optional[PageLayoutReader] = None,
        layout_text_pages: bool = True,
        batch_pages: int = LAYOUT_BATCH_PAGES,
        captioner: Optional[ImageCaptioner] = None,
        max_figures: int = MAX_FIGURES,
    ) -> None:
        """Keep the layout reader and its settings."""
        self.layout = layout
        self.layout_text_pages = layout_text_pages
        self.batch_pages = max(1, batch_pages)
        self.captioner = captioner
        self.max_figures = max_figures

    def verify(self, source: StoredFile, document_type: str) -> None:
        """Check the container of a DOCX/ODT/XLSX or an image's header."""
        if document_type in ZIP_MEMBERS:
            check_zip(source.path, document_type)
        elif document_type in IMAGE_TYPES:
            if self.layout is None:
                raise UnsupportedDocumentError(NO_OCR_IMAGE)
            check_image(source.path, document_type)

    def extract(
        self, source: StoredFile, document_type: str,
        progress: Optional[Progress] = None,
    ) -> ExtractedDocument:
        """Return the text of ``source``; ``InvalidInputError`` when none.

        ``progress`` is told the pages read so far of a PDF read with
        layout (after each batch).
        """
        self.verify(source, document_type)
        try:
            if self.layout is not None and document_type == "pdf":
                found = self._pdf_with_layout(source, progress)
            elif document_type in IMAGE_TYPES:
                found = self._image(source, document_type)
            elif document_type in EXTRACTORS:
                found = EXTRACTORS[document_type](source.path)
            else:
                raise UnsupportedDocumentError(
                    f"Unsupported document type {document_type!r}."
                )
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

    def _read(
        self, source: StoredFile, kind: str, numbers: Optional[List[int]],
        ocr: bool,
    ) -> Dict[int, PageContent]:
        """Ask the layout reader for pages; a failure gives no pages."""
        assert self.layout is not None
        try:
            got = self.layout.read_pages(
                source, kind, numbers, ocr, self.captioner is not None
            )
        except InvalidInputError:
            raise
        except Exception as exc:  # the reader's library broke on this batch
            LOG.warning("Layout reader failed on pages %s: %s: %s",
                        numbers, type(exc).__name__, exc)
            return {}
        return {n: c for n, c in got.items()
                if c.markdown.strip() or c.figures}

    def _captions(self, content: PageContent, budget: List[int]) -> str:
        """Return the page's figures as caption notes (``""`` if none).

        ``budget`` holds the figures still allowed in the document; a
        captioner error is logged once and ends captioning (the text of
        the document is kept).
        """
        if self.captioner is None or not content.figures or budget[0] <= 0:
            return ""
        figures = content.figures[:budget[0]]
        try:
            notes = figure_notes(self.captioner, figures)
        except Exception as exc:  # model missing, out of memory, no network
            LOG.warning("Captioning stopped: %s: %s", type(exc).__name__, exc)
            self.captioner, budget[0] = None, 0
            return ""
        budget[0] -= len(figures)
        return notes

    def _pdf_with_layout(
        self, source: StoredFile, progress: Optional[Progress] = None,
    ) -> ExtractedDocument:
        """Read a PDF page by page: layout for text pages, OCR for scans.

        A batch the reader fails on falls back to the text layer of its
        pages (scanned pages of it stay empty). Each batch's figures are
        captioned as soon as it is read and then dropped, so a PDF full
        of pictures never holds them all; ``progress`` is told the pages
        done after each batch.
        """
        texts = read_pdf_pages(source.path)
        total = len(texts)
        has_text = [len(t.strip()) >= MIN_PAGE_TEXT for t in texts]
        bodies: Dict[int, str] = {}
        budget = [self.max_figures]
        plans = (
            (False, [n for n, ok in enumerate(has_text, 1) if ok]),
            (True, [n for n, ok in enumerate(has_text, 1) if not ok]),
        )
        runs = [
            (ocr, run) for ocr, numbers in plans
            if ocr or self.layout_text_pages or self.captioner is not None
            for run in page_runs(numbers, self.batch_pages)
        ]
        planned = sum(len(run) for _, run in runs)
        done = total - planned
        if progress is not None:
            progress(done, total)
        for ocr, run in runs:
            found = self._read(source, "pdf", run, ocr)
            for number in run:
                content = found.get(number)
                if content is None:
                    continue
                body = content.markdown
                text = texts[number - 1]
                if has_text[number - 1] and (
                    not self.layout_text_pages
                    or letters(body) < MIN_LAYOUT_RATIO * letters(text)
                ):
                    body = text  # the layout model dropped text: the layer
                body += "\n\n" + self._captions(content, budget)
                bodies[number] = body
            done += len(run)
            if progress is not None:
                progress(done, total)
        pages: Dict[int, str] = {}
        for number, text in enumerate(texts, 1):
            read = bodies.get(number)
            if read is None:
                read = text if has_text[number - 1] else ""
            pages[number] = read.strip()
        if not any(pages.values()):
            raise InvalidInputError(
                "No se encontró texto en el PDF, ni con OCR. / No text "
                "found in the PDF, not even with OCR."
            )
        return ExtractedDocument(page_sections(pages), total)

    def _image(self, source: StoredFile, kind: str) -> ExtractedDocument:
        """OCR an image (every frame of a TIFF) and caption the picture."""
        if self.layout is None:
            raise InvalidInputError(NO_OCR_IMAGE)
        found = self._read(source, kind, None, True)
        pages = {n: c.markdown for n, c in found.items()}
        if self.captioner is not None:
            picture = Path(source.path).read_bytes()
            note = self._captions(PageContent("", (picture,)), [1])
            if note:
                first = min(pages) if pages else 1
                pages[first] = (pages.get(first, "") + "\n\n" + note).strip()
        if not any(t.strip() for t in pages.values()):
            raise InvalidInputError(NO_IMAGE_TEXT)
        return ExtractedDocument(page_sections(pages), max(pages))
