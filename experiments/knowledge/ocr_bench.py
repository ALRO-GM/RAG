"""Measure OCR quality (CER, WER) and speed (pages/minute) of the upload path.

Runs the real ``DocumentExtractors`` + ``DoclingPages`` on the two
synthetic scanned PDFs of ``make_scans.py`` (reference text known) and
on a generated text PDF, on CPU and on the GPU, and prints a JSON
report. Needs the ``ocr`` group and the models in the Hugging Face
cache; run it only while no live session holds the GPU.

    python experiments/knowledge/make_scans.py OUT/scans
    python experiments/knowledge/ocr_bench.py OUT/scans RESULTS.json
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.adapters.outbound.docling_pages import DoclingPages
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.adapters.outbound.gpu_lease_file import FileGpuGate, lease_directory

TEXT_PAGES = 120
LINES_PER_PAGE = 48


class Gate:
    """A GPU gate with a fixed answer (the benchmark chooses the device)."""

    def __init__(self, free: bool) -> None:
        """Say always ``free``."""
        self._free = free

    def free(self) -> bool:
        """Return the fixed answer."""
        return self._free


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Return the Levenshtein distance between two sequences."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> Dict[str, float]:
    """Return CER and WER of ``hypothesis`` against ``reference``."""
    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    return {
        "cer": edit_distance(ref, hyp) / max(1, len(ref)),
        "wer": edit_distance(ref.split(), hyp.split())
        / max(1, len(ref.split())),
    }


def page_body(markdown: str) -> str:
    """Return a page's paragraphs: no headings, no table rows."""
    lines = [ln for ln in markdown.splitlines()
             if ln.strip() and not ln.startswith(("#", "|"))]
    return " ".join(lines)


def split_pages(text: str) -> Dict[int, str]:
    """Split the extractor's ``## p. N`` sections into a dict."""
    parts = re.split(r"(?m)^## p\. (\d+)\n", text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def table_cells(markdown: str) -> List[str]:
    """Return the cells of the Markdown tables of a page."""
    cells: List[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and not set(line) <= set("|- "):
            cells += [c.strip() for c in line.strip("|").split("|")]
    return cells


def text_pdf(pages: int, lines: int, words: List[str]) -> bytes:
    """Return a PDF of ``pages`` pages of ``lines`` Helvetica text lines."""
    objects: Dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids [%s] /Count %d >>" % (
            " ".join(f"{4 + 2 * i} 0 R" for i in range(pages)), pages
        )).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
           b"/Encoding /WinAnsiEncoding >>",
    }
    for p in range(pages):
        rows = []
        for n in range(lines):
            start = (p * lines + n) * 9
            line = " ".join(words[(start + k) % len(words)] for k in range(9))
            raw = line.encode("cp1252")
            esc = b"".join(
                b"\\%03o" % c if c > 127 or c in b"()\\" else bytes([c])
                for c in raw
            )
            rows.append(b"BT /F1 10 Tf 60 %d Td (%s) Tj ET" % (
                760 - n * 15, esc))
        content = b"\n".join(rows)
        objects[4 + 2 * p] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
            "<< /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (5 + 2 * p)
        ).encode()
        objects[5 + 2 * p] = (b"<< /Length %d >>\nstream\n" % len(content)
                              + content + b"\nendstream")
    out = b"%PDF-1.4\n"
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    xref = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for number in range(1, size):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        size, xref)
    return out


def run(scans: Path, device: str, text_pdf_data: bytes) -> Dict[str, Any]:
    """Measure everything on ``device`` (``cpu`` or ``cuda``)."""
    reader = DoclingPages(Gate(device == "cuda"))
    layout = DocumentExtractors(reader)
    fast = DocumentExtractors(reader, layout_text_pages=False)
    result: Dict[str, Any] = {"device": device}
    layout.extract((scans / "clean.pdf").read_bytes(), "pdf")  # warm-up
    table = json.loads((scans / "clean.table.json").read_text())
    expected_cells = [c for row in table for c in row]
    total_pages, total_seconds = 0, 0.0
    for name in ("clean", "noisy"):
        data = (scans / f"{name}.pdf").read_bytes()
        started = time.perf_counter()
        found = layout.extract(data, "pdf")
        seconds = time.perf_counter() - started
        total_pages += found.pages or 0
        total_seconds += seconds
        pages = split_pages(found.text)
        reference = " ".join(
            (scans / f"{name}.p{n}.txt").read_text(encoding="utf-8")
            for n in sorted(pages))
        hypothesis = " ".join(page_body(pages[n]) for n in sorted(pages))
        cells = table_cells("\n".join(pages.values()))
        result[name] = {
            **error_rates(reference, hypothesis),
            "pages": found.pages, "seconds": round(seconds, 2),
            "table_cells_found": sum(
                1 for c in expected_cells for _ in [0]
                if cells.count(c) >= 1) / len(expected_cells),
            "pages_with_table": sum(
                1 for p in pages.values() if "| Vía" in p),
        }
    result["scanned_pages_per_minute"] = round(
        total_pages / total_seconds * 60, 1)
    for label, extractor in (("docling_text_pages", layout),
                             ("pypdf_text_layer", fast)):
        started = time.perf_counter()
        found = extractor.extract(text_pdf_data, "pdf")
        seconds = time.perf_counter() - started
        result[label] = {
            "pages": found.pages, "seconds": round(seconds, 2),
            "pages_per_minute": round((found.pages or 0) / seconds * 60, 1),
        }
    return result


def main(argv: Optional[List[str]] = None) -> None:
    """Run the benchmark and write the report."""
    args = argv if argv is not None else sys.argv[1:]
    scans, out = Path(args[0]), Path(args[1])
    words = " ".join(
        (scans / f"clean.p{n}.txt").read_text(encoding="utf-8")
        for n in (1, 2, 3)).split()
    data = text_pdf(TEXT_PAGES, LINES_PER_PAGE, words)
    lease_free = FileGpuGate(lease_directory(__import__("os").environ)).free()
    report: Dict[str, Any] = {
        "gpu_lease_free": lease_free, "text_pdf_pages": TEXT_PAGES,
        "runs": [],
    }
    for device in ("cpu", "cuda") if lease_free else ("cpu",):
        report["runs"].append(run(scans, device, data))
        print(json.dumps(report["runs"][-1], ensure_ascii=False))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
