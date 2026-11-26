"""Ingest a ~50 MB mixed PDF (the owner's case: a long Spanish regulation).

Builds a PDF of scanned noisy 300 dpi pages followed by text pages,
checks its size, runs the real upload extractor on it and reports time,
pages per minute and the peak memory of the process.

    python experiments/knowledge/big_pdf_case.py WORK_DIR SCANNED_PAGES
"""

import io
import json
import random
import resource
import sys
import time
from pathlib import Path

from make_scans import PARAGRAPHS, render_page
from ocr_bench import Gate, split_pages, text_pdf
from src.adapters.outbound.docling_pages import DoclingPages
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.application.knowledge.ports import StoredFile

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # pragma: no cover
    raise SystemExit("pypdf is required")


def build(work: Path, scanned: int, text_pages: int) -> Path:
    """Write ``big.pdf``: ``scanned`` noisy 300 dpi scans + text pages."""
    rng = random.Random(7)
    images = []
    for n in range(scanned):
        paragraphs = [PARAGRAPHS[(n + k) % len(PARAGRAPHS)] for k in range(3)]
        images.append(render_page(paragraphs, 300, True, rng))
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:],
                   resolution=300)
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(buffer.getvalue())).pages:
        writer.add_page(page)
    words = " ".join(PARAGRAPHS).split()
    for page in PdfReader(io.BytesIO(text_pdf(text_pages, 48, words))).pages:
        writer.add_page(page)
    path = work / "big.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def main() -> None:
    """Build the PDF, ingest it and print the numbers."""
    work, scanned = Path(sys.argv[1]), int(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    path = build(work, scanned, 200)
    size = path.stat().st_size
    extractor = DocumentExtractors(DoclingPages(Gate(False)))
    started = time.perf_counter()
    found = extractor.extract(StoredFile(str(path), size), "pdf")
    seconds = time.perf_counter() - started
    pages = split_pages(found.text)
    print(json.dumps({
        "pdf_mb": round(size / 1e6, 1), "pages": found.pages,
        "scanned_pages": scanned, "pages_with_text": len(pages),
        "seconds": round(seconds, 1),
        "pages_per_minute": round((found.pages or 0) / seconds * 60, 1),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }))


if __name__ == "__main__":
    main()
