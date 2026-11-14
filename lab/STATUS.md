# Lab web tab (issue #30, "Pruebas") — status

Branch `feat/model-lab`. Updated as work goes; a new session resumes from here.

## Done (committed)
- Backend: `GET …/runs/{run}/fragment-spans` and `?unlabelled=true` on
  `…/fragments` (use case, API, tests, LAB.md §7).
- Pure logic `web/js/lib/lab.js` + `tests/web/lab.test.mjs` (grouping
  answers, colours, scales/zoom, sorting, matrix, label placement,
  LabelQueue save/retry/local copy, keyboard map, export payload).
- Views: `js/views/lab.js` (recordings, catalogue by stage with
  badges, plan counts, start, live progress, cancel/resume, runs list),
  `lab-run.js` (sections, overview: sortable pipelines table, heat map +
  table, recognisers WER, denoiser effect), `lab-timeline.js` (rows per
  diarizer, consensus, labels, zoom/scroll, click to play, cursor, audio
  choice), `lab-review.js` (fragments with grouped answers, text
  variants, labelling buttons + keys, debounced saving, progress,
  unlabelled filter; scores table, best per stage, JSON export via Blob).
- Nav entry, route `#/lab[/<session>[/<run>[/<section>]]]`, es/en texts, CSS.

- Tried in headless Chrome (CDP script, scratch only) against a dev
  instance (fake runtime, sandbox lab, port 8791, own data dir) under the
  production CSP header: no console errors or CSP violations on every
  section; keyboard labelling (1, 2, N, U, K, X, B) saved 4 labels on the
  server; timeline click started playback and moved the cursor.
- Tests extended (test_web_pages, test_bundle); TUTORIAL.md §6, LAB.md tab section.

## Left / known gaps
- Not checked against the real 168-pipeline run (private data; only fake
  data used). Heat map is per denoiser or per diarizer by design.
- Scores table does not show the NC flag (score rows carry no licence).
- Not checked: screen readers, Firefox/Safari, real audio playback with
  the denoised WAV of real models.

## How to continue
Gate: `uv run --no-sync flake8 .`, `uv run --no-sync mypy . --strict`,
`HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= uv run --no-sync pytest -q --cov -p no:cacheprovider`,
`node --test tests/web/*.test.mjs`. Never plain `uv run`.
