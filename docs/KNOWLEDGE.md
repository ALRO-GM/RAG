# Knowledge domains and document upload

Status: built (PL-5 to PL-8, #142–#145, plus document upload; OCR and image
captions KN-1/KN-2, #191 and #192, optional group `ocr`; audio and video
files, spoken content only, KN-3a, #244, §8). Design:
`docs/platform/ARCHITECTURE.md` (Knowledge) and `docs/platform/CONTRACTS.md`
§2–3. The 42 corpus and the benchmarks are not affected: their chunks
carry no metadata and their index files are byte-identical.

## 1. What a domain is

A domain is a folder with a `manifest.yaml`:

```yaml
id: support.acme            # ^[a-z0-9][a-z0-9._-]{0,63}$
title: Soporte Acme
language: es
description: ""             # optional
vocabularies:               # optional; an empty list accepts any value
  intent: [refund, pricing]
  script_phase: [opening, closing]
  tags: [faq, vip]
```

| Where | What |
|---|---|
| `knowledge/domains/<id>/` | authored domains (in git): `manifest.yaml`, `*.md`, `*.markdown`, `*.txt`, `*.jsonl` |
| `data/knowledge/<id>/manifest.yaml` | domains created through the API |
| `data/knowledge/<id>/originals/<doc><ext>` | uploaded files, as received (audio and video too, §8) |
| `data/knowledge/<id>/extracted/<doc>.md` | their text (front matter + Markdown), indexed |
| `data/knowledge/<id>/documents/<doc>.json` | one record per upload (status, metadata…) |
| `data/processed/domains/<id>/` | the domain's index (same format as the 42 index) |

`index_version` hashes the manifest and the index fingerprint (itself a
hash of the indexing parameters and every indexed file).

Commands: `python -m src domain list` and `python -m src domain index <id>
[--force]` (incremental by file hash unless `--force`).

## 2. Metadata

Every chunk may carry `title`, `intent`, `category`, `script_phase` and
`tags`. Sources give them as

- **Markdown front matter** (labels every chunk of the file; other keys
  such as `author` are ignored):
  ```markdown
  ---
  intent: refund
  script_phase: closing
  tags: [faq]
  ---
  # Devoluciones
  ```
- **JSONL**, one record per line: `{"id": "q-7", "text": "...", "intent":
  "refund", "category": "pregunta", "tags": ["faq"]}` (only these keys
  plus `title`, `script_phase`). The record id names the chunk in results.
- **Upload metadata** (§4), written as the front matter of the extracted
  text.

`category` must be one of the 13 dialogue-act categories
(`src/domain/skills/dialogue_acts.CATEGORIES`). `intent`, `script_phase`
and `tags` must be in the manifest's vocabulary when it declares one.
A value outside is refused with `file:line` (indexing stops and the old
index stays) or, for uploads, with a 422. Headings become
`metadata.title` (the heading path, `A > B`).

Filters (`knowledge.query.v1`): a chunk passes when, for every filter
given, its value is in the list; for `tags` when any tag is. Empty or
missing filters do not restrict.

## 3. Settings (unmeasured choices)

`KnowledgeSettings` (`src/config.py`), separate from the 42 `Settings`:
multilingual embedder `sentence-transformers/paraphrase-multilingual-
MiniLM-L12-v2` at 128 tokens on the chunk body (no path), 1000-character
chunks, accent folding for the lexical index (`migración` →
`migracion`), BM25 constants and fusion of the 42 settings without its
priors (routing, test files, file boost) and without the query cache.
The two seed domains (`interview.generic`, `support.demo`) have 20
labelled queries each (`experiments/platform/SEED_DOMAINS.md`); nothing
was tuned on them.
Latency: `experiments/knowledge/QUERY_LATENCY.md`.

## 4. HTTP API

The RAG (`python -m src serve`, port 8000, bearer token; with
`--token_file` it listens on loopback only). The transcriber (port
8770) forwards the same paths under `/api/knowledge/...` behind its
access code (`X-Access-Code`) and adds the RAG's token; the web uses
those.

| Method and path | Body | Answer |
|---|---|---|
| `GET /knowledge/domains` | — | `200` `[{"id","title","language","documents","chunks","index_version","updated_at"}]` |
| `POST /knowledge/domains` | `{"id","title","language"}` | `201` the domain row; `409` id taken; `422` invalid |
| `DELETE /knowledge/domains/{id}` | — | `204`; `404`; `422` for an authored domain |
| `GET /knowledge/domains/{id}/documents` | — | `200` `[document]` (oldest first) |
| `POST /knowledge/domains/{id}/documents` | multipart: `file`, optional `metadata` (JSON text `{"category","intent","script_phase","tags"}`) | `202` document with `status: "queued"` |
| `PATCH /knowledge/domains/{id}/documents/{doc}` | `{"metadata": {...}}` (replaces all four keys) | `200` document, `status: "queued"` |
| `DELETE /knowledge/domains/{id}/documents/{doc}` | — | `204` (chunks leave the index in the background) |
| `GET /knowledge/domains/{id}/documents/{doc}/text` | — | `200` `{"id","name","text","chars"}` (extracted text, front matter cut; the interview planner reads job offers this way, #208); `409` not indexed yet; `404` |
| `POST /knowledge/query` | `knowledge.query.v1` | `200` `knowledge.result.v1`; `404` unknown domain or no index yet |

Document: `{"id","name","type","size","pages","chunks","status","error",
"metadata":{"category","intent","script_phase","tags"},"uploaded_at"}`;
`status` is `queued`, `processing`, `indexed` or `failed` (then `error`
says why, in Spanish and English). Poll the list to follow it. One
background worker extracts, chunks and embeds one document at a time
(a second one waits for the transcriptions of audio and video, §8);
documents left `queued`/`processing` by a restart are queued again
when the server starts.

Upload checks, before anything is stored: at most 200 MiB
(`max_upload_bytes`, `413`); type decided by the first bytes and the
extension (`415`: wrong signature, legacy `.doc`/`.xls`, unknown
extension, a ZIP that is not the announced DOCX/ODT/XLSX or unpacks
beyond 300 MB); metadata against the vocabularies (`422`); the same
file twice (`409`). The stored name is sanitised (no folders, control
or odd characters, at most 120 characters) and the file is saved as
`originals/<doc id><ext>`: ids are checked against fixed patterns
before any path is built. RAG errors are `{"detail": "..."}`; the
transcriber answers `{"code", "message"}` with the RAG's message and
the same status (`payload_too_large` 413, `unsupported_media_type`
415, `invalid_input` 400 for the RAG's 400/422, `not_found`,
`wrong_state` 409, `rag_unavailable` 503).

## 5. Formats and libraries

| Type | Extensions | Extraction | Licence |
|---|---|---|---|
| PDF | `.pdf` | with the `ocr` group (§6): Docling layout + tables + OCR; without it `pypdf` text layer. Either way one `## p. N` section per page; no text → `failed` "El PDF no tiene texto extraíble (¿está escaneado?); no se hace OCR. / … OCR is not supported." Password-protected → `failed` | BSD-3-Clause (pypdf), MIT (Docling) |
| Image | `.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff`, `.webp` | OCR (§6); without the `ocr` group a `415` "Las imágenes necesitan OCR…" | Pillow (HPND) |
| Word | `.docx` | `python-docx` (needs `lxml`, BSD-3): heading styles → `#`, lists → `-`, tables → Markdown tables, in document order | MIT |
| OpenDocument text | `.odt` | `odfpy` (needs `defusedxml`, PSF): outline levels → `#`, tables | Apache-2.0 |
| Excel | `.xlsx` | `openpyxl` read-only (needs `et-xmlfile`, MIT): one `## sheet` section with a table each, at most 50 000 rows | MIT |
| CSV | `.csv` | standard library, delimiter sniffed (`, ; tab |`), UTF-8/UTF-16/Windows-1252 | — |
| HTML | `.html`, `.htm` | standard library `html.parser`: visible text, `h1`–`h6` → `#`, scripts and styles dropped | — |
| Markdown, text | `.md`, `.markdown`, `.txt` | as is | — |
| Audio, video | `.wav`, `.flac`, `.ogg`, `.opus`, `.mp3`, `.m4a`, `.mp4`, `.aac`, `.webm`, `.mov`, `.3gp`, `.amr`, `.wma` | transcribed by the transcriber service (§8); what is on screen is not read (#193) | the transcriber's models |

Multipart parsing uses `python-multipart` (Apache-2.0); manifests
`PyYAML` (MIT). No legacy Office formats, no PowerPoint.

## 6. OCR, layout and tables (KN-1, #191)

Optional dependency group `ocr` (`pyproject.toml`): Docling (MIT) with
RapidOCR (Apache-2.0, ONNX runtime). Without it the service runs as
before (text-layer PDFs; images refused). Setting `KnowledgeSettings.ocr`
(default `True`) turns it on when the group is installed.

| Piece | What | Licence | Disk |
|---|---|---|---|
| `docling` 2.151.1 + `docling-ibm-models`, `docling-parse`… | PDF parsing, reading order | MIT | packages ≈ 0.6 GB in the shared venv |
| `docling-project/docling-layout-heron` | layout model (text, headings, tables, pictures) | Apache-2.0 | 164 MB (HF cache) |
| `docling-project/docling-models` (TableFormer) | table structure → Markdown | MIT | 342 MB (HF cache) |
| RapidOCR 3.10 `latin` PP-OCR models | text recognition, Spanish accents and `ñ` | Apache-2.0 | 43 MB (in `site-packages`) |

How a PDF is read (`DocumentExtractors`, `DoclingPages`):

1. `pypdf` reads the text layer of every page (cheap) and finds the pages
   with fewer than 8 non-blank characters: the **scanned** pages.
2. Pages with text go to Docling without OCR (layout, reading order,
   **tables as Markdown**; `layout_text_pages=False` keeps the fast
   `pypdf` text instead and loses the tables). Scanned pages go to
   Docling with OCR. Both in batches of 20 consecutive pages
   (`layout_batch_pages`) so a 50 MB file never sits in the layout model at
   once, and a batch Docling fails on falls back to the text layer of its
   pages (the failure is logged).
3. Every page is a `## p. N` section and the headings Docling found go one
   level down (`### Artículo 5`), so **the page number is in the heading
   path of every chunk**: `p. 12 > Artículo 5` is the chunk's `title`,
   shown in results and matched by the lexical index. No new index field:
   the 42 index and the domain indexes keep their format.

Images (`png`, `jpg`, `tiff`, `webp`) are checked on upload (header must
decode, at most 100 megapixels and 500 frames) and read with OCR; they
become `## p. 1` (TIFF frames `## p. N`). A broken or empty image is
`failed` with a reason.

**GPU.** Ingestion is a batch job and yields to the live mode: before every
batch `DoclingPages` asks `FileGpuGate`, which reads the transcriber's
lease file (`$XDG_RUNTIME_DIR/gpu-lease/lease.json`, `gpu-lease.v1`:
unexpired and its pid alive = held; no code is shared with the transcriber).
Free and CUDA present → the models run on the GPU; held → CPU, and the
GPU converter is dropped so its VRAM returns. Nothing ever signals or
kills another process.

Install on the workstation (never plain `uv sync`/`uv run`: they swap torch
to the CPU build under the running services):

```sh
uv pip install --dry-run --python .venv/bin/python docling rapidocr   # torch must not change
uv pip install --python .venv/bin/python docling rapidocr
uv pip install --no-deps --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 "torchvision==0.29.1+cu126"
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 2.14.1+cu126 True
```

The dry run left `torch` alone but pulled `torchvision` from PyPI (0.29.2,
built for another CUDA: `operator torchvision::nms does not exist`), hence
the third command (torchvision for `cu126`, matching torch 2.14.1). Docling
also needs `huggingface-hub < 2`: the venv went from 2.4.0 to 1.33.0, which
`transformers` 5.20.2 (`>=1.31,<3`) accepts.

If Docling returns far less text than a page's own text layer (under 60 %),
the layer is used for that page. Measurements: `experiments/knowledge/OCR.md`.

## 7. Image captions (KN-2, #192)

Figures and images get a caption and the text written in them, indexed as
part of their page. Needs the `ocr` group (the layout reader finds the
figures). `KnowledgeSettings.caption_backend`:

| Backend | Where it runs | Licence / size | Notes |
|---|---|---|---|
| `florence` (default) | local; GPU while the lease is free, else CPU | Florence-2-base `florence-community/Florence-2-base`, MIT, 447 MiB on disk (`caption_model` can name `…-large`, MIT, 1.55 GB, not tried) | `<DETAILED_CAPTION>` + `<OCR_WITH_REGION>`; captions in **English** and sometimes invented; 3-5 s per figure on the CPU |
| `claude` | `claude -p` in isolated mode (no tools, no settings, subscription, never the paid API) | — | tested: accepts a base64 image in `stream-json` input; Spanish, better, 2 s per image; **the image leaves the machine**, so it is opt-in |
| `off` | — | — | text and tables only |

In the extracted Markdown a figure is

```markdown
## p. 4

…text of the page…

> **Figura 1:** The image shows a red stop sign with the word "STOP"…
> Texto en la figura: STOP
```

so a chunk's title carries the page (`p. 4`) and the quote is found by the
words of the caption and of the text in the figure. An uploaded image is
OCR'd by the layout reader *and* captioned as a whole (page 1). At most
`max_figures` (200) are captioned per document; pictures under 48 px are
ignored. If the captioner cannot run (model not downloaded, out of memory)
it is switched off for that document, logged, and the text is indexed
without captions. Measurements and the samples: `experiments/knowledge/OCR.md`.

## 8. Audio and video: what is said (KN-3a, #244)

An audio or video file uploaded to a domain is transcribed and its
transcript is indexed like any document. Only what is **said** counts;
keyframes and text on screen are #193.

**Who transcribes.** The transcriber service, with its own pipeline
(the one of «Sesiones»: Whisper, the voice grouping or the lab
diarizer, sentences) in its own job runner. The RAG loads no speech
model, so the GPU rules hold as for any batch job of the transcriber: it
waits while a live session holds the lease (`gpu-lease.v1`), frees the
GPU after every job and never touches the IMG worker.

**How the two services talk.** The RAG's media worker sends the file to
the transcriber on this machine and polls the job:

| Route (transcriber, port 8770) | Answer |
|---|---|
| `PUT /internal/knowledge-media/{job}?format=&language=&max_seconds=` (raw body: the file) | `202` the job: `{"id","state","stage","fraction","message","duration","transcript"}` |
| `GET /internal/knowledge-media/{job}` | `200` the job; `transcript` is a `transcript.v1` document once `state` is `done`; `404` unknown |
| `DELETE /internal/knowledge-media/{job}` | `204`; `409 wrong_state` while it runs |

`{job}` is `kn-` and 32 hex digits of SHA-256 over `<domain>/<document>`,
so a file sent again after a restart is the same job (one queued,
running or done is left as it is; a failed one starts again). The routes
are outside `/api` (the server's Caddy forwards only `/api/*` through the
tunnel), refuse a request carrying the proxy's `X-Client-IP`, and want
the RAG's own bearer token (the file `make rag-token` writes, which the
transcriber already publishes with); no new secret. They exist only when
the transcriber posts to a RAG over HTTP (`TRANSCRIBER_LOCAL_RAG=1`).

**Never a session.** The jobs live in their own repository,
`<transcriber data>/knowledge-media/sessions/`, which the session list,
the player, `publish` and the analysis never read: a knowledge
recording does not appear in «Sesiones», is never posted to
`POST /transcripts` and is never analysed. Its voices are not compared
with the enrolled voiceprints (anonymous `Voz 1`, `Voz 2`…). The RAG
deletes the job as soon as it has the transcript; jobs a stopped
transcriber left are deleted when it starts (the RAG, finding its job
gone, sends the file once more).

**The text.** One paragraph per stretch of one voice (at most 600
characters, longer stretches are split), opened by its times and, when
more than one voice speaks, by the speaker:

```markdown
---
title: Formación reembolsos.mp4
intent: refund
timed: true
---
[00:00–00:42] **Voz 1:** Bienvenidos. Hoy vemos cómo se tramita un reembolso…

[00:43–01:10] **Voz 2:** ¿Y si el cliente no tiene el ticket?
```

`timed: true` makes every chunk of the file carry `metadata.start_s` and
`metadata.end_s`: the start of the paragraph the chunk begins in and the
end of the last paragraph it reaches. `knowledge.result.v1` returns them
(written only when set), so a retrieved chunk can be cited by time; the
page shows «min 03:12–03:40». The text is kept: changing a recording's
metadata re-indexes it without transcribing again.

**States and failures.** The upload answers `202` `queued` at once; the
media worker sets `processing` while the transcriber works, then the
usual worker indexes the text (`indexed`). A failure ends in `failed`
with a message the page shows:

| Cause | `error` |
|---|---|
| transcriber not running, or unreachable for 120 s during a job | "No se pudo contactar con el servicio de transcripción…" |
| no speech | "No se oye voz en el archivo. / No speech was found in the file." |
| unreadable file or no audio track | "No se pudo leer el sonido del archivo…" |
| longer than the limit | "La grabación dura 2 h 10 min; el límite es 2 h 00 min. / …" |
| not done in 6 h (a long live session holds the GPU) | "La transcripción no terminó en 6:00:00…" |
| the job lost twice (transcriber restarted twice) | "…vuelve a subir el archivo." |

Saving the document's metadata again retries a failed recording.

**Limits** (`KnowledgeSettings`, explicit, not measured):

| Setting | Default | Checked |
|---|---|---|
| `max_upload_bytes` | 200 MiB (as every upload; the page sends audio and video up to 200 MiB, other documents up to 100 MB) | at upload, `413` |
| `max_media_seconds` | 7200 (2 h) | by the transcriber after decoding, before any model runs: `failed` |
| `media_wait_seconds` | 21600 (6 h, queue included) | `failed` |
| `media_poll_seconds` | 5 | — |
| `transcriber_url` | `http://127.0.0.1:8770` (loopback only; `null` refuses audio and video with `415`) | — |

A file whose content is not an audio or video container (WAV, FLAC,
Ogg, MP3/ID3, MPEG/ADTS frames, ISO media `ftyp`/`moov`…, WebM/Matroska,
AMR, ASF) is refused at upload with `415`, as an unknown extension is.
The container does not have to match the extension (the transcriber
reads the content).

**The original** is kept in `originals/` like every upload, so a failed
or interrupted transcription can be retried without uploading again.
Deleting the document deletes the record, the original and the text
(and, through the re-index, its chunks); the transcriber's copy is
deleted when the transcript is read, or, for a job still running when
the document is deleted, as soon as it ends.

**Deploy.** No reindex: existing chunks have no times and their index
lines are unchanged. Order: restart the transcriber first (it gains the
routes; nothing it writes changes), then the RAG. `knowledge.result.v1`
gains `start_s`/`end_s` compatibly; its readers (orchestrator, web) read
chunks as plain objects and the reasoner's `retrieved[].metadata` is a
free object, so no reader refuses them.

**Not measured.** Transcription speed and quality of long training
videos: the tests use the transcriber's fake models
(`tests/integration/transcriber/test_knowledge_media*.py`); nothing
here was run with the real models or the GPU.
