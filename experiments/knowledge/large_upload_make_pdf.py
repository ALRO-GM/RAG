"""Write a synthetic large PDF for the large-upload run (#282).

Each page has a text layer (Spanish sentences, Helvetica) over a large
noise JPEG, so the file is big and incompressible. Written to disk a
page at a time; nothing of it is committed.

    python experiments/knowledge/large_upload_make_pdf.py OUT PAGES SIDE

``PAGES=107 SIDE=2000`` gave 502 273 040 bytes (docs/KNOWLEDGE.md §4.2).
"""

import io
import sys
from typing import BinaryIO, List, Tuple

import numpy as np
from PIL import Image

WORDS = (
    "La política de devoluciones permite reembolsar el importe en un "
    "plazo de treinta días naturales desde la entrega si el producto "
    "está en buen estado"
).split()


def write_object(
    handle: BinaryIO, offsets: List[Tuple[int, int]], number: int,
    body: bytes,
) -> None:
    """Write object ``number`` and remember where it starts."""
    offsets.append((number, handle.tell()))
    handle.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")


def noise_jpeg(rng: np.random.Generator, side: int) -> bytes:
    """Return a ``side``×``side`` noise picture as JPEG."""
    buffer = io.BytesIO()
    pixels = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def page_text(page: int) -> bytes:
    """Return the content stream of one page: the picture and 20 lines."""
    lines = ["q 400 0 0 300 100 50 cm /Im0 Do Q"]
    for n in range(20):
        text = " ".join(WORDS[(page + n + k) % len(WORDS)] for k in range(10))
        lines.append(
            f"BT /F1 11 Tf 60 {740 - 30 * n} Td "
            f"(Pagina {page + 1}: {text}) Tj ET"
        )
    return "\n".join(lines).encode("latin-1")


def build(out: str, pages: int, side: int) -> None:
    """Write the PDF to ``out``."""
    rng = np.random.default_rng(282)
    offsets: List[Tuple[int, int]] = []
    with open(out, "wb") as handle:
        handle.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        kids = " ".join(f"{4 + 3 * i} 0 R" for i in range(pages))
        write_object(handle, offsets, 1, b"<< /Type /Catalog /Pages 2 0 R >>")
        write_object(handle, offsets, 2, (
            f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>"
        ).encode())
        write_object(handle, offsets, 3, (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        ))
        for i in range(pages):
            page, content, image = 4 + 3 * i, 5 + 3 * i, 6 + 3 * i
            jpeg = noise_jpeg(rng, side)
            stream = page_text(i)
            write_object(handle, offsets, page, (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content} 0 R /Resources << /Font "
                f"<< /F1 3 0 R >> /XObject << /Im0 {image} 0 R >> >> >>"
            ).encode())
            write_object(handle, offsets, content, (
                f"<< /Length {len(stream)} >>\nstream\n".encode()
                + stream + b"\nendstream"
            ))
            write_object(handle, offsets, image, (
                f"<< /Type /XObject /Subtype /Image /Width {side} "
                f"/Height {side} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /DCTDecode "
                f"/Length {len(jpeg)} >>\nstream\n".encode()
                + jpeg + b"\nendstream"
            ))
        xref = handle.tell()
        total = 4 + 3 * pages
        handle.write(f"xref\n0 {total}\n0000000000 65535 f \n".encode())
        for _, offset in sorted(offsets):
            handle.write(f"{offset:010d} 00000 n \n".encode())
        handle.write((
            f"trailer\n<< /Size {total} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode())


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
