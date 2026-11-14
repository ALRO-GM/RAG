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

## Left
- Try it in headless Chrome under the production CSP with fake data
  (dev instance from tests/integration/transcriber/lab_sandbox.py,
  runtime fake, other port and data dir); fix what breaks.
- Extend tests/integration/transcriber/test_web_pages.py / test_bundle.py
  for the new files.
- TUTORIAL.md section (Spanish), LAB.md section on the tab.
- Full gate.

## How to continue
Gate: `uv run --no-sync flake8 .`, `uv run --no-sync mypy . --strict`,
`HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= uv run --no-sync pytest -q --cov -p no:cacheprovider`,
`node --test tests/web/*.test.mjs`. Never plain `uv run`.
