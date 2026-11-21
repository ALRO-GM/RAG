# Knowledge domains and document upload

Status: built (PL-5 to PL-8, #142–#145, plus document upload). Design:
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
| `data/knowledge/<id>/originals/<doc><ext>` | uploaded files, as received |
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
No domain has a labelled test set yet, so none of these was tuned.
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
| `POST /knowledge/query` | `knowledge.query.v1` | `200` `knowledge.result.v1`; `404` unknown domain or no index yet |

Document: `{"id","name","type","size","pages","chunks","status","error",
"metadata":{"category","intent","script_phase","tags"},"uploaded_at"}`;
`status` is `queued`, `processing`, `indexed` or `failed` (then `error`
says why, in Spanish and English). Poll the list to follow it. One
background worker extracts, chunks and embeds one document at a time;
documents left `queued`/`processing` by a restart are queued again
when the server starts.

Upload checks, before anything is stored: at most 50 MiB
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
| PDF | `.pdf` | `pypdf` text layer, one `## p. N` section per page; no text → `failed` "El PDF no tiene texto extraíble (¿está escaneado?); no se hace OCR. / … OCR is not supported." Password-protected → `failed` | BSD-3-Clause |
| Word | `.docx` | `python-docx` (needs `lxml`, BSD-3): heading styles → `#`, lists → `-`, tables → Markdown tables, in document order | MIT |
| OpenDocument text | `.odt` | `odfpy` (needs `defusedxml`, PSF): outline levels → `#`, tables | Apache-2.0 |
| Excel | `.xlsx` | `openpyxl` read-only (needs `et-xmlfile`, MIT): one `## sheet` section with a table each, at most 50 000 rows | MIT |
| CSV | `.csv` | standard library, delimiter sniffed (`, ; tab |`), UTF-8/UTF-16/Windows-1252 | — |
| HTML | `.html`, `.htm` | standard library `html.parser`: visible text, `h1`–`h6` → `#`, scripts and styles dropped | — |
| Markdown, text | `.md`, `.markdown`, `.txt` | as is | — |

Multipart parsing uses `python-multipart` (Apache-2.0); manifests
`PyYAML` (MIT). No OCR, no legacy Office formats, no PowerPoint.
