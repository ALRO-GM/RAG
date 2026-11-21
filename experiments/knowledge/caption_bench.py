"""Caption synthetic figures (Florence-2, optionally Claude) and a PDF.

Draws five figures with Pillow (bar chart, pie chart, flow diagram, a
stop-sign shape with text, a table image), captions each with the
local Florence-2 captioner (and, with ``--claude``, with ``claude -p``),
then builds a two-page text PDF that embeds two of them and ingests it
through the real upload extractor (Docling finds the pictures). Writes
a JSON report.

    python experiments/knowledge/caption_bench.py OUT.json [--claude] [--gpu]

Without ``--gpu`` the models stay on the CPU (use it while a live
session holds the GPU lease).
"""

import io
import json
import sys
import time
import zlib
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from src.adapters.outbound.claude_captioner import ClaudeCodeCaptioner
from src.adapters.outbound.docling_pages import DoclingPages
from src.adapters.outbound.document_extractors import DocumentExtractors
from src.adapters.outbound.florence_captioner import FlorenceCaptioner
from src.adapters.outbound.gpu_lease_file import FileGpuGate, lease_directory

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return a DejaVu font."""
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def bar_chart() -> Image.Image:
    """Bars of accidents per year."""
    im = Image.new("RGB", (480, 320), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 10), "Víctimas por año", fill="black", font=font(20, True))
    for i, (year, v) in enumerate([("2021", 120), ("2022", 180),
                                   ("2023", 90), ("2024", 150)]):
        x = 50 + i * 100
        d.rectangle([x, 290 - v, x + 60, 290], fill=(40, 90, 200))
        d.text((x + 5, 295), year, fill="black", font=font(14))
    d.line([40, 290, 460, 290], fill="black", width=2)
    return im


def pie_chart() -> Image.Image:
    """A three-slice pie with a legend."""
    im = Image.new("RGB", (400, 320), "white")
    d = ImageDraw.Draw(im)
    box = [40, 40, 240, 240]
    d.pieslice(box, 0, 140, fill=(220, 60, 60))
    d.pieslice(box, 140, 250, fill=(60, 180, 90))
    d.pieslice(box, 250, 360, fill=(240, 200, 50))
    for i, (label, colour) in enumerate([("Coche", (220, 60, 60)),
                                         ("Moto", (60, 180, 90)),
                                         ("Bici", (240, 200, 50))]):
        d.rectangle([270, 80 + i * 40, 290, 100 + i * 40], fill=colour)
        d.text((300, 80 + i * 40), label, fill="black", font=font(16))
    return im


def flow_diagram() -> Image.Image:
    """Three boxes joined by arrows."""
    im = Image.new("RGB", (480, 160), "white")
    d = ImageDraw.Draw(im)
    for i, label in enumerate(["Denuncia", "Alegaciones", "Resolución"]):
        x = 15 + i * 160
        d.rectangle([x, 50, x + 120, 110], outline="black", width=3)
        d.text((x + 10, 70), label, fill="black", font=font(16))
        if i < 2:
            d.line([x + 120, 80, x + 155, 80], fill="black", width=3)
            d.polygon([(x + 155, 72), (x + 155, 88), (x + 165, 80)],
                      fill="black")
    return im


def stop_sign() -> Image.Image:
    """A red octagon with STOP."""
    im = Image.new("RGB", (300, 300), "white")
    d = ImageDraw.Draw(im)
    d.regular_polygon((150, 150, 130), 8, rotation=22.5, fill=(200, 20, 20),
                      outline="white")
    d.text((70, 120), "STOP", fill="white", font=font(60, True))
    return im


def table_image() -> Image.Image:
    """A speed-limit table as a picture."""
    im = Image.new("RGB", (420, 180), "white")
    d = ImageDraw.Draw(im)
    rows = [["Vía", "Turismos", "Camiones"], ["Autopista", "120", "90"],
            ["Carretera", "90", "80"], ["Poblado", "50", "50"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            x, y = 10 + c * 135, 10 + r * 40
            d.rectangle([x, y, x + 135, y + 40], outline="black", width=2)
            d.text((x + 8, y + 10), cell, fill="black",
                   font=font(16, r == 0))
    return im


FIGURES = {
    "bar_chart": bar_chart, "pie_chart": pie_chart,
    "flow_diagram": flow_diagram, "stop_sign": stop_sign,
    "table_image": table_image,
}


def png(image: Image.Image) -> bytes:
    """Return ``image`` as PNG bytes."""
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def pdf_with_figures(texts: List[str], images: List[Image.Image]) -> bytes:
    """Return a PDF: page ``i`` has ``texts[i]`` and image ``i`` below it."""
    objects: Dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
           b"/Encoding /WinAnsiEncoding >>",
    }
    kids = []
    number = 4
    for text, image in zip(texts, images):
        rgb = image.convert("RGB")
        raw = zlib.compress(rgb.tobytes())
        w, h = rgb.size
        lines = b"\n".join(
            b"BT /F1 11 Tf 60 %d Td (%s) Tj ET" % (
                750 - 16 * n, line.encode("cp1252").replace(
                    b"(", b"\\(").replace(b")", b"\\)"))
            for n, line in enumerate(text.split("\n")))
        lines += b"\nq %d 0 0 %d 60 320 cm /Im1 Do Q" % (w, h)
        objects[number + 2] = (b"<< /Length %d >>\nstream\n" % len(lines)
                               + lines + b"\nendstream")
        objects[number + 1] = (
            b"<< /Type /XObject /Subtype /Image /Width %d /Height %d "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter "
            b"/FlateDecode /Length %d >>\nstream\n" % (w, h, len(raw))
            + raw + b"\nendstream")
        objects[number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> /XObject << /Im1 %d 0 R "
            b">> >> /Contents %d 0 R >>" % (number + 1, number + 2))
        kids.append(number)
        number += 3
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out = b"%PDF-1.4\n"
    offsets = {}
    for n in sorted(objects):
        offsets[n] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (n, objects[n])
    xref = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for n in range(1, size):
        out += b"%010d 00000 n \n" % offsets[n]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        size, xref)
    return out


class Gate:
    """A GPU gate with a fixed answer."""

    def __init__(self, free: bool) -> None:
        """Say always ``free``."""
        self._free = free

    def free(self) -> bool:
        """Return the fixed answer."""
        return self._free


def timed(captioner: Any, images: Dict[str, bytes]) -> Dict[str, Any]:
    """Caption every image; return the answers and seconds each."""
    result: Dict[str, Any] = {}
    captioner.describe(next(iter(images.values())))  # warm-up
    for name, data in images.items():
        started = time.perf_counter()
        found = captioner.describe(data)
        result[name] = {
            "caption": found.caption, "text": found.text,
            "seconds": round(time.perf_counter() - started, 2),
        }
    return result


def main() -> None:
    """Run the benchmark and write the report."""
    out = sys.argv[1]
    use_gpu = "--gpu" in sys.argv
    free = FileGpuGate(lease_directory(__import__("os").environ)).free()
    gate = Gate(use_gpu and free)
    images = {name: png(make()) for name, make in FIGURES.items()}
    report: Dict[str, Any] = {
        "gpu_lease_free": free, "device": "cuda" if gate.free() else "cpu",
    }
    florence = FlorenceCaptioner(gate)
    report["florence_base"] = timed(florence, images)
    if "--claude" in sys.argv:
        report["claude_haiku"] = timed(ClaudeCodeCaptioner(), images)
    texts = ["Informe de siniestralidad\nLa figura muestra las víctimas.",
             "Procedimiento sancionador\nEl esquema resume los pasos."]
    pdf = pdf_with_figures(texts, [bar_chart(), flow_diagram()])
    extractor = DocumentExtractors(DoclingPages(gate), captioner=florence)
    started = time.perf_counter()
    found = extractor.extract(pdf, "pdf")
    report["pdf_with_two_figures"] = {
        "seconds": round(time.perf_counter() - started, 2),
        "text": found.text,
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    for name, row in report["florence_base"].items():
        print(name, row)
    print(found.text)


if __name__ == "__main__":
    main()
