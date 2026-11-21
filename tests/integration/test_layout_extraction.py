"""PDF and image extraction through a layout reader (a fake one)."""

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from src.adapters.outbound.document_extractors import (
    NO_IMAGE_TEXT,
    NO_OCR_IMAGE,
    DocumentExtractors,
    page_runs,
    page_sections,
)
from src.domain.errors import InvalidInputError, UnsupportedDocumentError
from tests.knowledge_samples import make_image, make_pdf

Call = Tuple[str, Optional[List[int]], bool]


class FakeLayout:
    """A layout reader that answers from a table and records its calls."""

    def __init__(self, pages: Dict[int, str], fail: bool = False) -> None:
        """Answer ``pages``; raise on every call when ``fail``."""
        self.pages = pages
        self.fail = fail
        self.calls: List[Call] = []

    def read_pages(
        self, data: bytes, kind: str, pages: Optional[Sequence[int]],
        ocr: bool,
    ) -> Dict[int, str]:
        """Record the call and return the requested pages."""
        self.calls.append((kind, None if pages is None else list(pages), ocr))
        if self.fail:
            raise RuntimeError("model crashed")
        wanted = pages if pages is not None else sorted(self.pages)
        return {n: self.pages[n] for n in wanted if n in self.pages}


def test_scanned_pages_are_read_with_ocr_and_text_pages_without() -> None:
    """Text pages go to the layout reader without OCR, scans with it."""
    layout = FakeLayout({
        1: "## Título\n\nTexto con tabla\n\n| a | b |\n| --- | --- |",
        2: "Página escaneada",
    })
    extractor = DocumentExtractors(layout)
    found = extractor.extract(make_pdf(["Text layer here", ""]), "pdf")
    assert layout.calls == [("pdf", [1], False), ("pdf", [2], True)]
    assert found.pages == 2
    assert found.text.startswith("## p. 1\n\n### Título")
    assert "| a | b |" in found.text
    assert "## p. 2\n\nPágina escaneada" in found.text


def test_text_pages_can_stay_on_the_fast_text_layer() -> None:
    """Without ``layout_text_pages`` only scanned pages reach the reader."""
    layout = FakeLayout({3: "OCR page"})
    extractor = DocumentExtractors(layout, layout_text_pages=False)
    found = extractor.extract(
        make_pdf(["First text", "Second text", ""]), "pdf"
    )
    assert layout.calls == [("pdf", [3], True)]
    assert "## p. 1\n\nFirst text" in found.text
    assert "## p. 3\n\nOCR page" in found.text


def test_batches_split_runs_of_consecutive_pages() -> None:
    """A long scan goes to the reader ``batch_pages`` at a time."""
    layout = FakeLayout({n: f"page {n}" for n in range(1, 6)})
    extractor = DocumentExtractors(layout, batch_pages=2)
    found = extractor.extract(make_pdf([""] * 5), "pdf")
    assert [c[1] for c in layout.calls] == [[1, 2], [3, 4], [5]]
    assert found.pages == 5
    assert "## p. 5\n\npage 5" in found.text


def test_page_runs() -> None:
    """Runs break at gaps and at the size limit."""
    assert page_runs([1, 2, 4, 5, 6, 9], 2) == [[1, 2], [4, 5], [6], [9]]
    assert page_runs([], 3) == []


def test_page_sections_skip_empty_pages_and_demote_headings() -> None:
    """Empty pages vanish; a heading inside a page goes one level down."""
    text = page_sections({2: "# A\n\ntext", 1: "  ", 3: "###### F"})
    assert text == "## p. 2\n\n## A\n\ntext\n\n## p. 3\n\n###### F"


def test_failing_reader_falls_back_to_the_text_layer() -> None:
    """A crashed batch keeps the pages that have a text layer."""
    layout = FakeLayout({}, fail=True)
    found = DocumentExtractors(layout).extract(
        make_pdf(["Still readable", ""]), "pdf"
    )
    assert found.text == "## p. 1\n\nStill readable"
    assert found.pages == 2


def test_all_scanned_and_unreadable_fails() -> None:
    """No text from the layer or the reader is an error."""
    with pytest.raises(InvalidInputError, match="ni con OCR"):
        DocumentExtractors(FakeLayout({})).extract(make_pdf(["", ""]), "pdf")


def test_without_layout_a_scanned_pdf_still_says_no_ocr() -> None:
    """The old behaviour is kept when the ``ocr`` group is missing."""
    with pytest.raises(InvalidInputError, match="OCR"):
        DocumentExtractors().extract(make_pdf([""]), "pdf")


@pytest.mark.parametrize("kind", ["png", "jpg", "tiff", "webp"])
def test_images_are_read_with_ocr(kind: str) -> None:
    """Every image type goes through the layout reader with OCR on."""
    layout = FakeLayout({1: "Texto de la imagen"})
    extractor = DocumentExtractors(layout)
    data = make_image(kind)
    extractor.verify(data, kind)
    found = extractor.extract(data, kind)
    assert layout.calls == [(kind, None, True)]
    assert found.text == "## p. 1\n\nTexto de la imagen"
    assert found.pages == 1


def test_image_without_text_fails() -> None:
    """An image the reader finds nothing in is refused with a message."""
    with pytest.raises(InvalidInputError, match="No se encontró"):
        DocumentExtractors(FakeLayout({})).extract(make_image(), "png")
    assert "image" in NO_IMAGE_TEXT


def test_images_without_layout_reader_are_refused() -> None:
    """Without the ``ocr`` group an image upload is refused at once."""
    with pytest.raises(UnsupportedDocumentError, match="OCR"):
        DocumentExtractors().verify(make_image(), "png")
    with pytest.raises(InvalidInputError, match="OCR"):
        DocumentExtractors().extract(make_image(), "png")
    assert "ocr" in NO_OCR_IMAGE


def test_broken_and_huge_images_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated image and a pixel bomb fail ``verify``."""
    extractor = DocumentExtractors(FakeLayout({}))
    with pytest.raises(UnsupportedDocumentError, match="válida"):
        extractor.verify(b"\x89PNG\r\n\x1a\nnot really", "png")
    from src.adapters.outbound import document_extractors
    monkeypatch.setattr(document_extractors, "MAX_PIXELS", 10)
    with pytest.raises(UnsupportedDocumentError, match="demasiado grande"):
        extractor.verify(make_image(), "png")
