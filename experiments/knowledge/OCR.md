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

A caveat on the text PDF numbers above: the generated text PDFs use
hand-written, non-embedded fonts and Docling dropped the text of some of their
pages (a page of 50 characters came back with 12). Real papers do not:
on the 26-page Dialnet paper of the owner's downloads folder the
Markdown of pages 1-12 kept 100-108 % of the characters of the text
layer on 11 pages and 86 % on one. That is why the extractor compares each
layout page with its text layer and keeps the layer when the layout kept
less than 60 % of it (`MIN_LAYOUT_RATIO`).

# Image captions (KN-2, #192)

`caption_bench.py`, raw numbers in `caption_report.json`. Five synthetic
figures drawn with Pillow (bar chart, pie chart, flow diagram, stop sign,
table picture), CPU only again (live lease held; `gpu_lease_free: false`).
Florence-2-base (`florence-community/Florence-2-base`, MIT, 0.23 B
parameters, 447 MiB on disk, run in fp32 on the CPU).

| Figure | Florence `<DETAILED_CAPTION>` | text (`<OCR_WITH_REGION>`) | s |
|---|---|---|---|
| bar chart | "blue bar chart depicting the number of victims in **Brazil** from 2021 to 2024" | `Victimas por año 2021 2022 2023 2024` | 3.7 |
| pie chart | "pie chart … people who have been diagnosed with **cancer**" | `Coche Moto Bici` | 3.8 |
| flow diagram | "block diagram with the words Denuncia, Alegaciones and Resolución" | `Denuncia Alegaciones Resolución` | 3.7 |
| stop sign | "red stop sign with the word STOP" | `STOP` | 2.9 |
| table picture | "a table with … turismo, camiones, autopista…" | all 12 cells | 4.8 |

- Text in the figure is read correctly in 5/5 (the accent of `Víctimas` is lost
  in the first). Captions describe the kind and the look of the figure
  and are in **English**; two of five invent a subject (Brazil, cancer).
  They help find a figure by what it shows; they are not data. The other
  tasks were worse: `<MORE_DETAILED_CAPTION>` hallucinated more and ran on
  (a paragraph with invented "cancer in the United States"), `<OCR>` glued
  the words of a chart (`Victimas por año2021202220232024`).
- Time: 3-5 s per figure on the CPU (two generations of ≤160 tokens, beam 3).
  A document with 200 figures (the cap) is 10-15 minutes of batch work.
- Claude through `claude -p` (haiku, same isolated mode as the judge, image
  as a base64 block in `stream-json` input): **works**, 2.0-2.2 s per
  image, Spanish, and noticeably better ("Tabla con tres columnas (Vía,
  Turismos, Camiones) … Autopista (120 turismos, 90 camiones)", pie chart
  with the three vehicle labels and the legend, no invented subject). It
  sends the image out of the machine, so it is opt-in
  (`KnowledgeSettings.caption_backend = "claude"`), not the default.
- End to end on a generated two-page PDF with two embedded figures
  (6.5 s): the page's text and the first figure's caption and text came
  out under `## p. 1`; **the second figure was not found** by Docling on this
  synthetic file. One real-world sample of two; no real documents tried.
- Not measured: Florence-2-large (1.55 GB; not downloaded), GPU speed,
  photos, scanned figures inside scanned pages (those are read as page text,
  not as figures).
