# Lab web tab (issue #30, "Pruebas") — status

Branch `feat/model-lab`. Updated as work goes; a new session resumes from here.

## Done
- Backend: `GET …/runs/{run}/fragment-spans` and `?unlabelled=true` on
  `…/fragments` (use case, API, tests, LAB.md §7).

## Left
- Pure logic `js/lib/lab.js` + `tests/web/lab.test.mjs`.
- Views: choose/run, overview, timeline, fragments (labelling), scores.
- Headless Chrome check under the production CSP with fake data.
- TUTORIAL.md section (Spanish), LAB.md section on the tab.

## How to continue
Gate: flake8, mypy --strict, pytest (HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=),
`node --test tests/web/*.test.mjs`. Always `uv run --no-sync`.
