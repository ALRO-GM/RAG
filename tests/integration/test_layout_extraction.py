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
from src.application.knowledge.ports import (
    ImageDescription,
    PageContent,
    StoredFile,
)
from src.domain.errors import InvalidInputError, UnsupportedDocumentError
from tests.knowledge_samples import as_file, make_image, make_pdf

Call = Tuple[str, Optional[List[int]], bool]


class FakeCaptioner:
    """A captioner that answers from a list and records the images."""

    def __init__(self, fail: bool = False) -> None:
        """Raise on every call when ``fail``."""
        self.fail = fail
        self.images: List[bytes] = []

    def describe(self, image: bytes) -> ImageDescription:
        """Return a caption naming how many images were seen."""
        if self.fail:
            raise RuntimeError("no model")
        self.images.append(image)
        return ImageDescription(
            f"figura {len(self.images)}", "texto" * (len(self.images) % 2)
        )


class FakeLayout:
    """A layout reader that answers from a table and records its calls."""

    def __init__(
        self, pages: Dict[int, str], fail: bool = False,
        figures: Optional[Dict[int, Tuple[bytes, ...]]] = None,
    ) -> None:
        """Answer ``pages``; raise on every call when ``fail``."""
        self.pages = pages
        self.fail = fail
        self.figures = figures or {}
        self.calls: List[Call] = []
        self.asked_figures: List[bool] = []

    def read_pages(
        self, source: StoredFile, kind: str, pages: Optional[Sequence[int]],
        ocr: bool, figures: bool = False,
    ) -> Dict[int, PageContent]:
        """Record the call and return the requested pages."""
        self.calls.append((kind, None if pages is None else list(pages), ocr))
        self.asked_figures.append(figures)
        if self.fail:
            raise RuntimeError("model crashed")
        wanted = pages if pages is not None else sorted(self.pages)
        return {
            n: PageContent(self.pages[n], self.figures.get(n, ())
                           if figures else ())
            for n in wanted if n in self.pages
        }


def test_scanned_pages_are_read_with_ocr_and_text_pages_without() -> None:
    """Text pages go to the layout reader without OCR, scans with it."""
    layout = FakeLayout({
        1: "## Título\n\nTexto con tabla\n\n| a | b |\n| --- | --- |",
        2: "Página escaneada",
    })
    extractor = DocumentExtractors(layout)
    found = extractor.extract(
        as_file(make_pdf(["Text layer here", ""])), "pdf"
    )
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
        as_file(make_pdf(["First text", "Second text", ""])), "pdf"
    )
    assert layout.calls == [("pdf", [3], True)]
    assert "## p. 1\n\nFirst text" in found.text
    assert "## p. 3\n\nOCR page" in found.text


def test_batches_split_runs_of_consecutive_pages() -> None:
    """A long scan goes to the reader ``batch_pages`` at a time."""
    layout = FakeLayout({n: f"page {n}" for n in range(1, 6)})
    extractor = DocumentExtractors(layout, batch_pages=2)
    found = extractor.extract(as_file(make_pdf([""] * 5)), "pdf")
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
        as_file(make_pdf(["Still readable", ""])), "pdf"
    )
    assert found.text == "## p. 1\n\nStill readable"
    assert found.pages == 2


def test_all_scanned_and_unreadable_fails() -> None:
    """No text from the layer or the reader is an error."""
    with pytest.raises(InvalidInputError, match="ni con OCR"):
        DocumentExtractors(FakeLayout({})).extract(
            as_file(make_pdf(["", ""])), "pdf"
        )


def test_without_layout_a_scanned_pdf_still_says_no_ocr() -> None:
    """The old behaviour is kept when the ``ocr`` group is missing."""
    with pytest.raises(InvalidInputError, match="OCR"):
        DocumentExtractors().extract(as_file(make_pdf([""])), "pdf")


@pytest.mark.parametrize("kind", ["png", "jpg", "tiff", "webp"])
def test_images_are_read_with_ocr(kind: str) -> None:
    """Every image type goes through the layout reader with OCR on."""
    layout = FakeLayout({1: "Texto de la imagen"})
    extractor = DocumentExtractors(layout)
    data = make_image(kind)
    extractor.verify(as_file(data), kind)
    found = extractor.extract(as_file(data), kind)
    assert layout.calls == [(kind, None, True)]
    assert found.text == "## p. 1\n\nTexto de la imagen"
    assert found.pages == 1


def test_image_without_text_fails() -> None:
    """An image the reader finds nothing in is refused with a message."""
    with pytest.raises(InvalidInputError, match="No se encontró"):
        DocumentExtractors(FakeLayout({})).extract(
            as_file(make_image()), "png"
        )
    assert "image" in NO_IMAGE_TEXT


def test_images_without_layout_reader_are_refused() -> None:
    """Without the ``ocr`` group an image upload is refused at once."""
    with pytest.raises(UnsupportedDocumentError, match="OCR"):
        DocumentExtractors().verify(as_file(make_image()), "png")
    with pytest.raises(InvalidInputError, match="OCR"):
        DocumentExtractors().extract(as_file(make_image()), "png")
    assert "ocr" in NO_OCR_IMAGE


def test_broken_and_huge_images_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated image and a pixel bomb fail ``verify``."""
    extractor = DocumentExtractors(FakeLayout({}))
    with pytest.raises(UnsupportedDocumentError, match="válida"):
        extractor.verify(as_file(b"\x89PNG\r\n\x1a\nnot really"), "png")
    from src.adapters.outbound import document_extractors
    monkeypatch.setattr(document_extractors, "MAX_PIXELS", 10)
    with pytest.raises(UnsupportedDocumentError, match="demasiado grande"):
        extractor.verify(as_file(make_image()), "png")


def test_figures_are_captioned_into_their_page() -> None:
    """A figure's caption and text follow its page's Markdown."""
    layout = FakeLayout({1: "Texto", 2: ""}, figures={
        1: (b"a", b"b"), 2: (b"c",)})
    captioner = FakeCaptioner()
    found = DocumentExtractors(layout, captioner=captioner).extract(
        as_file(make_pdf(["Has text", "Has text too"])), "pdf")
    assert layout.asked_figures == [True]
    assert captioner.images == [b"a", b"b", b"c"]
    assert "## p. 1\n\nTexto\n\n> **Figura 1:** figura 1\n> Texto en la " \
        "figura: texto\n\n> **Figura 2:** figura 2" in found.text
    assert "## p. 2\n\nHas text too\n\n> **Figura 1:** figura 3" in found.text


def test_figure_budget_and_failing_captioner() -> None:
    """At most ``max_figures`` are captioned; a failure ends captioning."""
    layout = FakeLayout({1: "x text one", 2: "y text two"}, figures={
        1: (b"a", b"b"), 2: (b"c",)})
    captioner = FakeCaptioner()
    found = DocumentExtractors(
        layout, captioner=captioner, max_figures=1
    ).extract(as_file(make_pdf(["x text one", "y text two"])), "pdf")
    assert captioner.images == [b"a"]
    assert "figura 1" in found.text and "figura 2" not in found.text
    broken = DocumentExtractors(layout, captioner=FakeCaptioner(fail=True))
    found = broken.extract(
        as_file(make_pdf(["x text one", "y text two"])), "pdf"
    )
    assert found.text == "## p. 1\n\nx text one\n\n## p. 2\n\ny text two"
    assert broken.captioner is None


def test_a_captioned_pdf_keeps_fast_text_pages_for_the_text() -> None:
    """With ``layout_text_pages=False`` the text layer stays the text."""
    layout = FakeLayout({1: "Docling text"}, figures={1: (b"a",)})
    found = DocumentExtractors(
        layout, layout_text_pages=False, captioner=FakeCaptioner()
    ).extract(as_file(make_pdf(["Layer text here"])), "pdf")
    assert found.text.startswith("## p. 1\n\nLayer text here")
    assert "figura 1" in found.text


def test_an_image_gets_a_caption_and_its_text() -> None:
    """The picture itself is captioned besides being OCR'd."""
    captioner = FakeCaptioner()
    data = make_image("png")
    found = DocumentExtractors(
        FakeLayout({1: "Texto OCR"}), captioner=captioner
    ).extract(as_file(data), "png")
    assert captioner.images == [data]
    assert found.text == (
        "## p. 1\n\nTexto OCR\n\n> **Figura 1:** figura 1\n> Texto en la "
        "figura: texto"
    )
    only = DocumentExtractors(FakeLayout({}), captioner=FakeCaptioner())
    assert "figura 1" in only.extract(as_file(data), "png").text


def test_image_markers_from_the_reader_are_dropped() -> None:
    """Docling's ``<!-- image -->`` placeholder never reaches the index."""
    assert page_sections({1: "A\n\n<!-- image -->\n\nB"}) == \
        "## p. 1\n\nA\n\n\n\nB"


def test_text_dropped_by_the_layout_model_comes_back() -> None:
    """A page whose layout Markdown keeps too little uses its text layer."""
    layout = FakeLayout({1: "short", 2: ""})
    found = DocumentExtractors(layout).extract(
        as_file(make_pdf([
            "A long sentence on the first page", "Second page text",
        ])),
        "pdf",
    )
    assert "A long sentence on the first page" in found.text
    assert "## p. 2\n\nSecond page text" in found.text


def test_batches_are_captioned_as_they_come_and_reported() -> None:
    """Each batch's figures are captioned before the next batch is read
    (never all held at once) and the pages done are reported."""
    events: List[str] = []

    class Ordered(FakeLayout):
        """Notes each read."""

        def read_pages(
            self, source: StoredFile, kind: str,
            pages: Optional[Sequence[int]], ocr: bool, figures: bool = False,
        ) -> Dict[int, PageContent]:
            """Note the batch, then read it."""
            events.append(f"read {list(pages or [])}")
            return super().read_pages(source, kind, pages, ocr, figures)

    class Noting(FakeCaptioner):
        """Notes each caption."""

        def describe(self, image: bytes) -> ImageDescription:
            """Note the figure, then caption it."""
            events.append(f"caption {image.decode()}")
            return super().describe(image)

    layout = Ordered({n: f"page {n} text" for n in range(1, 6)},
                     figures={1: (b"a",), 4: (b"b",)})
    reports: List[Tuple[int, int]] = []
    found = DocumentExtractors(
        layout, batch_pages=2, captioner=Noting(),
    ).extract(as_file(make_pdf([f"page {n} text" for n in range(1, 6)])),
              "pdf", lambda done, total: reports.append((done, total)))
    assert events == ["read [1, 2]", "caption a", "read [3, 4]",
                      "caption b", "read [5]"]
    assert reports == [(0, 5), (2, 5), (4, 5), (5, 5)]
    assert found.pages == 5


def test_pages_left_on_the_text_layer_count_as_done() -> None:
    """Without layout for text pages only the scans are batches."""
    layout = FakeLayout({2: "scanned text"})
    reports: List[Tuple[int, int]] = []
    DocumentExtractors(layout, layout_text_pages=False).extract(
        as_file(make_pdf(["Text layer here", ""])), "pdf",
        lambda done, total: reports.append((done, total)),
    )
    assert reports == [(1, 2), (2, 2)]
