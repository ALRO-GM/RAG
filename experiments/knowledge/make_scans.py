"""Generate two synthetic scanned PDFs with a known text (ground truth).

Pages of Spanish traffic-rules style text are rendered to images with
Pillow and saved as image-only PDFs (no text layer). ``clean`` is a
crisp 200 dpi render; ``noisy`` adds a slight rotation, blur, noise and
lower contrast at 150 dpi. Every page also holds a small table whose
rows are written to ``<name>.table.json``. The paragraphs of each page
go to ``<name>.p<N>.txt`` and are the reference for CER and WER.

    python experiments/knowledge/make_scans.py OUT_DIR
"""

import json
import random
import sys
import textwrap
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFilter, ImageFont

PARAGRAPHS = [
    "Artículo 21. Límites de velocidad en vías fuera de poblado. La "
    "velocidad máxima en autopistas y autovías es de 120 kilómetros por "
    "hora, y en el resto de las carreteras de 90 kilómetros por hora.",
    "Los conductores deberán respetar los límites de velocidad "
    "establecidos y tendrán en cuenta, además, sus propias condiciones "
    "físicas y psíquicas, las características y el estado de la vía, "
    "del vehículo y de su carga, así como las condiciones "
    "meteorológicas, ambientales y de circulación.",
    "Artículo 22. Adelantamiento. El conductor que se disponga a "
    "adelantar deberá cerciorarse de que puede hacerlo sin riesgo, "
    "señalizará su maniobra con antelación suficiente y volverá a su "
    "carril cuando haya recuperado la distancia de seguridad.",
    "Queda prohibido conducir utilizando el teléfono móvil, cascos o "
    "auriculares conectados a aparatos receptores o reproductores de "
    "sonido. La sanción será de 200 euros y la pérdida de 3 puntos.",
    "Artículo 23. Señales de los agentes. Los usuarios de la vía están "
    "obligados a obedecer las señales de la autoridad encargada de la "
    "regulación del tráfico, aunque parezcan contradictorias con las "
    "normas o con otras señales.",
    "El niño menor de 135 centímetros de estatura deberá utilizar un "
    "sistema de retención infantil homologado adaptado a su talla y "
    "peso, y ocupará siempre los asientos traseros del vehículo.",
]
TABLE = [
    ["Vía", "Turismos", "Camiones"],
    ["Autopista", "120", "90"],
    ["Carretera", "90", "80"],
    ["Poblado", "50", "50"],
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def render_page(paragraphs: List[str], dpi: int, noisy: bool,
                rng: random.Random) -> Image.Image:
    """Render one A4 page: heading, paragraphs and the table."""
    w, h = int(8.27 * dpi), int(11.69 * dpi)
    page = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(page)
    body = ImageFont.truetype(FONT, int(0.16 * dpi))
    bold = ImageFont.truetype(FONT_BOLD, int(0.22 * dpi))
    margin = int(0.9 * dpi)
    y = margin
    draw.text((margin, y), "REGLAMENTO GENERAL DE CIRCULACIÓN", font=bold,
              fill=0)
    y += int(0.5 * dpi)
    chars = int((w - 2 * margin) / (0.16 * dpi * 0.62))
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, chars):
            draw.text((margin, y), line, font=body, fill=0)
            y += int(0.26 * dpi)
        y += int(0.18 * dpi)
    y += int(0.2 * dpi)
    col = int(1.6 * dpi)
    row = int(0.36 * dpi)
    for r, cells in enumerate(TABLE):
        for c, cell in enumerate(cells):
            x0 = margin + c * col
            draw.rectangle([x0, y, x0 + col, y + row], outline=0, width=2)
            draw.text((x0 + 10, y + 8), cell,
                      font=bold if r == 0 else body, fill=0)
        y += row
    if noisy:
        page = page.rotate(rng.uniform(-0.6, 0.6), fillcolor=255,
                           resample=Image.Resampling.BICUBIC)
        page = page.filter(ImageFilter.GaussianBlur(0.9))
        px = page.load()
        assert px is not None
        for _ in range(w * h // 40):
            x, yy = rng.randrange(w), rng.randrange(h)
            value = px[x, yy]
            assert isinstance(value, int)
            px[x, yy] = max(0, min(255, value + rng.randint(-60, 60)))
        page = page.point(lambda v: int(60 + v * 0.7))
    return page


def build(out: Path, name: str, dpi: int, noisy: bool, seed: int) -> None:
    """Write ``name.pdf`` (3 pages), the page texts and the table."""
    rng = random.Random(seed)
    pages = []
    for n in range(3):
        paragraphs = [PARAGRAPHS[(2 * n + k) % len(PARAGRAPHS)]
                      for k in range(3)]
        (out / f"{name}.p{n + 1}.txt").write_text(
            "\n".join(paragraphs), encoding="utf-8")
        pages.append(render_page(paragraphs, dpi, noisy, rng))
    pages[0].save(out / f"{name}.pdf", save_all=True,
                  append_images=pages[1:], resolution=dpi)
    (out / f"{name}.table.json").write_text(json.dumps(TABLE))


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    build(target, "clean", 200, False, 1)
    build(target, "noisy", 150, True, 2)
