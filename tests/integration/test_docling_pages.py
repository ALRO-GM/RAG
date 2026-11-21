"""The Docling adapter, against a stand-in for the library (no models)."""

import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

from src.adapters.outbound import docling_pages
from src.adapters.outbound.docling_pages import DoclingPages

BUILT: List[Tuple[bool, str]] = []
CONVERTED: List[Any] = []


class Options:
    """Records the keyword arguments a Docling option object got."""

    def __init__(self, **kwargs: Any) -> None:
        """Keep the arguments as attributes."""
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        """Reject what was never given."""
        raise AttributeError(name)


class FakeDocument:
    """A converted document with three pages."""

    pages = {1: None, 2: None, 3: None}

    def export_to_markdown(self, page_no: int) -> str:
        """Return a Markdown line naming the page."""
        return f"text of page {page_no}"


class FakeConverter:
    """A ``DocumentConverter`` that records how it was built and called."""

    def __init__(self, format_options: Dict[str, Options]) -> None:
        """Remember the OCR flag and device of the PDF options."""
        pdf: Any = format_options["pdf"].pipeline_options
        BUILT.append((pdf.do_ocr, pdf.accelerator_options.device))

    def convert(self, stream: Any, **kwargs: Any) -> Any:
        """Record the call and return the fake document."""
        CONVERTED.append((stream.name, kwargs))
        return types.SimpleNamespace(document=FakeDocument())


class Gate:
    """A GPU gate whose answer the test changes."""

    def __init__(self, free: bool) -> None:
        """Start with the card ``free`` or not."""
        self.is_free = free

    def free(self) -> bool:
        """Return the current answer."""
        return self.is_free


@pytest.fixture(autouse=True)
def fake_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stand-in ``docling`` package and a CUDA device."""
    BUILT.clear()
    CONVERTED.clear()

    def module(name: str, **attrs: Any) -> None:
        monkeypatch.setitem(
            sys.modules, name, types.SimpleNamespace(**attrs)
        )

    module("docling")
    module("docling.datamodel")
    module("docling.datamodel.base_models",
           DocumentStream=Options, InputFormat=types.SimpleNamespace(
               PDF="pdf", IMAGE="image"))
    module("docling.datamodel.accelerator_options", AcceleratorOptions=Options)
    module("docling.datamodel.pipeline_options",
           PdfPipelineOptions=Options, RapidOcrOptions=Options)
    module("docling.document_converter", DocumentConverter=FakeConverter,
           PdfFormatOption=lambda pipeline_options: Options(
               pipeline_options=pipeline_options),
           ImageFormatOption=lambda pipeline_options: Options(
               pipeline_options=pipeline_options))
    monkeypatch.setattr(docling_pages, "cuda_available", lambda: True)


def test_requested_pages_come_back_as_markdown() -> None:
    """A page range is converted once and only the wanted pages kept."""
    reader = DoclingPages()
    found = reader.read_pages(b"%PDF", "pdf", [2, 3], True)
    assert found == {2: "text of page 2", 3: "text of page 3"}
    assert CONVERTED[0][0] == "document.pdf"
    assert CONVERTED[0][1] == {"page_range": (2, 3)}


def test_images_and_whole_documents_have_no_page_range() -> None:
    """Without a page list everything is converted."""
    found = DoclingPages().read_pages(b"x", "png", None, True)
    assert sorted(found) == [1, 2, 3]
    assert CONVERTED[0] == ("document.png", {})


def test_converters_are_built_once_per_mode() -> None:
    """OCR on and off are two converters, each built a single time."""
    reader = DoclingPages()
    reader.read_pages(b"x", "pdf", [1], False)
    reader.read_pages(b"x", "pdf", [2], False)
    reader.read_pages(b"x", "pdf", [3], True)
    assert BUILT == [(False, "cpu"), (True, "cpu")]


def test_gpu_only_while_the_gate_is_free() -> None:
    """Free card: CUDA; lease taken: the CPU converter, GPU one dropped."""
    gate = Gate(True)
    reader = DoclingPages(gate)
    reader.read_pages(b"x", "pdf", [1], False)
    gate.is_free = False
    reader.read_pages(b"x", "pdf", [2], False)
    gate.is_free = True
    reader.read_pages(b"x", "pdf", [3], False)
    assert BUILT == [(False, "cuda"), (False, "cpu"), (False, "cuda")]


def test_cpu_without_gate_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """No gate means CPU; no CUDA device means CPU even if the gate is free."""
    assert DoclingPages().pick_device() == "cpu"
    monkeypatch.setattr(docling_pages, "cuda_available", lambda: False)
    assert DoclingPages(Gate(True)).pick_device() == "cpu"


def test_availability_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The group is detected by its modules; torch is optional."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name != "rapidocr" else None,
    )
    assert not docling_pages.docling_available()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert docling_pages.docling_available()
    monkeypatch.undo()
    assert isinstance(docling_pages.cuda_available(), bool)
