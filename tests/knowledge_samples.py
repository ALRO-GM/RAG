"""Small documents of every uploadable type, generated in memory.

No binary fixture is committed: each test builds the file it needs.
"""

import io
import zipfile
from typing import Sequence


def make_pdf(pages: Sequence[str]) -> bytes:
    """Return a PDF whose pages show ``pages`` (ASCII; '' = no text)."""
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages))),
            len(pages),
        ),
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(pages):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(") \
            .replace(")", "\\)")
        content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET" if text else ""
        objects[4 + 2 * i] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * i} 0 R >>"
        )
        objects[5 + 2 * i] = (
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream"
        )
    out = b"%PDF-1.4\n"
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode()
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    return out


def make_image(kind: str = "png", size: int = 16) -> bytes:
    """Return a tiny image of type ``kind`` (png, jpg, tiff or webp)."""
    from PIL import Image

    out = io.BytesIO()
    format_name = {"jpg": "JPEG", "tiff": "TIFF"}.get(kind, kind.upper())
    Image.new("RGB", (size, size), (255, 255, 255)).save(out, format_name)
    return out.getvalue()


def make_docx() -> bytes:
    """Return a DOCX with a title, headings, a list and a table."""
    import docx

    document = docx.Document()
    document.add_heading("Manual de devoluciones", level=0)
    document.add_heading("Plazos", level=1)
    document.add_paragraph("El cliente tiene treinta días para devolver.")
    document.add_paragraph("Ticket de compra", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Producto"
    table.cell(0, 1).text = "Plazo"
    table.cell(1, 0).text = "Portátil"
    table.cell(1, 1).text = "15 días"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_odt() -> bytes:
    """Return an ODT with headings, a paragraph and a table."""
    from odf.opendocument import OpenDocumentText
    from odf.table import Table, TableCell, TableColumn, TableRow
    from odf.text import H, P

    document = OpenDocumentText()
    document.text.addElement(H(outlinelevel=1, text="Política de envíos"))
    document.text.addElement(H(outlinelevel=2, text="Plazos"))
    document.text.addElement(P(text="Los envíos tardan 48 horas."))
    table = Table(name="Tarifas")
    table.addElement(TableColumn(numbercolumnsrepeated=2))
    for row in (("Zona", "Precio"), ("Península", "5 EUR")):
        tr = TableRow()
        for value in row:
            cell = TableCell()
            cell.addElement(P(text=value))
            tr.addElement(cell)
        table.addElement(tr)
    document.text.addElement(table)
    buffer = io.BytesIO()
    document.write(buffer)
    return buffer.getvalue()


def make_xlsx() -> bytes:
    """Return a workbook with two sheets."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Precios"
    sheet.append(["Plan", "Precio", "Notas"])
    sheet.append(["Básico", 10, "sin soporte"])
    sheet.append(["Pro", 25.5, "soporte | 24h"])
    other = workbook.create_sheet("Vacía")
    other.append([None])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def make_zip(members: dict[str, bytes]) -> bytes:
    """Return a ZIP archive holding ``members``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


HTML = """<!doctype html><html><head><title>Ayuda</title>
<style>p { color: red }</style><script>alert('x')</script></head>
<body><h1>Centro de ayuda</h1><p>Escríbenos a <b>soporte</b>.</p>
<h2>Horario</h2><ul><li>Lunes a viernes</li><li>9 a 18 h</li></ul>
<table><tr><th>Canal</th><th>Tiempo</th></tr>
<tr><td>Email</td><td>24 h</td></tr></table></body></html>"""
