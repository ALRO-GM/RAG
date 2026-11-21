"""Page layout reader on Docling (MIT) with RapidOCR (Apache-2.0).

Implements :class:`~src.application.knowledge.ports.PageLayoutReader`:
the layout model (``docling-layout-heron``) finds text, headings and
tables, TableFormer turns tables into Markdown and RapidOCR (ONNX
runtime, the ``latin`` recognition model: Spanish accents and ``ñ``)
reads bitmaps. Docling and its models come with the optional
dependency group ``ocr``; this module imports them only when first
used, so the rest of the service runs without them.

Ingestion is a batch job: before each batch the GPU gate is asked and
the models run on the GPU only when no live session holds it, else on
the CPU. The converter of the other device is dropped (its memory is
returned) when the device changes.
"""

import importlib
import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.application.knowledge.ports import GpuGate, PageContent

LOG = logging.getLogger(__name__)

OCR_LANGUAGES = ("latin",)
"""RapidOCR recognition model: Latin script (Spanish, Catalan, English...)."""
MIN_FIGURE_SIDE = 48
"""Pictures with a shorter side (pixels) are icons or rules, not figures."""


def docling_available() -> bool:
    """Tell whether Docling and RapidOCR are installed (group ``ocr``)."""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("docling", "rapidocr", "PIL")
    )


def cuda_available() -> bool:
    """Tell whether torch sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


class DoclingPages:
    """Read pages with Docling; one converter per (OCR, device) in use.

    Args:
        gpu: Tells whether the GPU is free; ``None`` means CPU only.
        languages: RapidOCR language codes.
        document_timeout: Seconds after which Docling stops a batch.
    """

    def __init__(
        self,
        gpu: Optional[GpuGate] = None,
        languages: Sequence[str] = OCR_LANGUAGES,
        document_timeout: float = 900.0,
    ) -> None:
        """Keep the settings; nothing is loaded yet."""
        self.gpu = gpu
        self.languages = list(languages)
        self.document_timeout = document_timeout
        self._converters: Dict[Tuple[bool, bool, str], Any] = {}
        self.device = "cpu"

    def pick_device(self) -> str:
        """Return ``cuda`` when the card is free and present, else ``cpu``."""
        if self.gpu is not None and self.gpu.free() and cuda_available():
            return "cuda"
        return "cpu"

    def read_pages(
        self, data: bytes, kind: str, pages: Optional[Sequence[int]],
        ocr: bool, figures: bool = False,
    ) -> Dict[int, PageContent]:
        """Return each requested page (1-based): Markdown, figures if asked.

        Figures are the pictures Docling found, as PNG, without the tiny
        ones (icons, rules).
        """
        from docling.datamodel.base_models import DocumentStream

        device = self.pick_device()
        converter = self._converter(ocr, figures, device)
        name = "document." + ("pdf" if kind == "pdf" else kind)
        stream = DocumentStream(name=name, stream=io.BytesIO(data))
        if pages:
            result = converter.convert(
                stream, page_range=(min(pages), max(pages))
            )
        else:
            result = converter.convert(stream)
        wanted = set(pages) if pages else None
        found: Dict[int, PageContent] = {}
        document = result.document
        for number in sorted(document.pages):
            if wanted is None or number in wanted:
                found[number] = PageContent(
                    document.export_to_markdown(page_no=number),
                    tuple(self._figures(document, number)) if figures else (),
                )
        return found

    def _figures(self, document: Any, number: int) -> List[bytes]:
        """Return the pictures of one page as PNG bytes, in reading order."""
        doc_types = importlib.import_module("docling_core.types.doc")
        picture_item: Any = getattr(doc_types, "PictureItem")
        found: List[bytes] = []
        for item, _ in document.iterate_items(page_no=number):
            if not isinstance(item, picture_item):
                continue
            image = item.get_image(document)
            if image is None or min(image.size) < MIN_FIGURE_SIDE:
                continue
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "PNG")
            found.append(buffer.getvalue())
        return found

    def _converter(self, ocr: bool, figures: bool, device: str) -> Any:
        """Return the converter for ``ocr`` on ``device`` (built once)."""
        if device != self.device:
            LOG.info("Document layout moves from %s to %s",
                     self.device, device)
            self._converters.clear()
            self.device = device
            if cuda_available():
                import torch
                import gc

                gc.collect()
                torch.cuda.empty_cache()
        key = (ocr, figures, device)
        if key not in self._converters:
            self._converters[key] = self._build(ocr, figures, device)
        return self._converters[key]

    def _build(self, ocr: bool, figures: bool, device: str) -> Any:
        """Build a Docling converter for PDF and images."""
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        options = PdfPipelineOptions(
            do_ocr=ocr, do_table_structure=True,
            document_timeout=self.document_timeout,
            generate_picture_images=figures,
        )
        options.ocr_options = RapidOcrOptions(lang=self.languages)
        options.accelerator_options = AcceleratorOptions(device=device)
        return DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
        })
