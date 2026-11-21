# OCR, layout and tables of uploaded documents (KN-1, #191)

Measured on 2026-11-21 on the workstation (i7-12700KF, CPU run; the
RTX 3070 Ti was **not** used: the live mode held `gpu-lease` for the whole
session, so the GPU gate kept ingestion on the CPU. No GPU number is
claimed). Code: `experiments/knowledge/make_scans.py`, `ocr_bench.py`,
`big_pdf_case.py`; raw numbers `ocr_report.json`, `big_pdf_report.json`.
Stack: Docling 2.151.1 (layout `docling-layout-heron`, TableFormer),
RapidOCR 3.10 `latin`, pypdf 7.

## Quality: two synthetic scanned PDFs

Three A4 pages each, Spanish regulation-style text rendered with Pillow and
saved as an image-only PDF (no text layer); `clean` 200 dpi, `noisy` 150 dpi
with rotation (±0.6°), blur, noise and lower contrast. Each page also has a
4×3 table. The reference is the text that was drawn. CER and WER are
character and word edit distance over whitespace-normalised paragraphs
(headings and table rows excluded; the table is checked separately).

| PDF | pages | CER | WER | table cells found | pages with a Markdown table | seconds |
|---|---|---|---|---|---|---|
| clean (200 dpi) | 3 | 0.116 | 0.126 | 12/12 | 3/3 | 6.85 |
| noisy (150 dpi) | 3 | 0.000 | 0.000 | 12/12 | 3/3 | 5.44 |

The clean file is worse than the noisy one because of **reading order, not
recognition**: on pages 1 and 2 the layout model split the last paragraph
into two blocks and Docling interleaved their lines
("…el conductor que se disponga a señalizará su maniobra…"), so the
characters are right and the order is not (CER 0.14 / 0.19 on those pages,
0.003 on page 3). Two files, three pages each, one font: this is a smoke
test of the pipeline, not a general accuracy figure. Real regulations
have not been measured.

## Speed

CPU, 12 cores, models already loaded (one warm-up call), whole upload path:

| Input | pages | pages/minute |
|---|---|---|
| scanned (the two PDFs above, OCR on) | 6 | 29.3 |
| text PDF through Docling (layout + tables, no OCR) | 120 | 119.9 |
| text PDF with `layout_text_pages=False` (pypdf text layer only) | 120 | 24 447 |
| mixed 300-page PDF: 100 noisy scans at 300 dpi + 200 text pages, 48.4 MB | 300 | 63.8 (282 s) |

The 48.4 MB file is a stand-in for the owner's 53 MB regulation (generated,
not the real file). Peak memory of the process: 4.4 GB (4.1 GB at 218 pages
and 9.5 MB: it is the models and Docling's working set, not the file size,
so a 53 MB file fits the same envelope). A 600-page text PDF would take
about 5 minutes with tables, 2 seconds without; the default keeps tables.

## Decisions

- Pages with a text layer still go through Docling (tables as Markdown,
  reading order); `layout_text_pages=False` is the fast path that loses
  tables. Chosen because the owner's case has tables of fines and limits.
- `latin` RapidOCR model: it reads Spanish accents and `ñ`; `en` also did on
  the clean page but glued words ("Artículo21", "120kilómetros") on the
  first try.
- Not measured: GPU speed, real scans of real documents, handwriting,
  skewed 2-column layouts, languages other than Spanish.
