# Model lab and its web tab ("Pruebas", issue #30) - status

Updated 2026-11-14. The lab (PR #32, fixes #35, #36, #38) is merged in
main and the tab is deployed. Design and results: `docs/transcriber/LAB.md`
(§11.7 and §12 for the full run and how to resume).

## Done
- Backend, CLI and API of the lab (LAB.md §7, §8) and the tab: recordings,
  catalogue by stage with badges, plan counts, start, progress, cancel and
  resume, run overview (sortable pipelines, heat map and table, recognisers'
  WER, denoiser effect), timeline, fragments to label (keys, debounced
  saving), scores with JSON export. Pure logic in `web/js/lib/lab.js`
  (`tests/web/lab.test.mjs`).
- Every model of the catalogue ran on R1: full run `e7a37b38`, 224
  pipelines (7 denoisers x 8 diarizers x 4 recognisers), state `done`
  after one resume (CUDA out of memory on one step: another process held the
  GPU). The 168 earlier pipelines are unchanged in it.
- Tried earlier in headless Chrome against a dev instance with fake data,
  under the production CSP: no console errors or violations; keyboard
  labelling saved on the server; timeline click played.

## Left / known gaps
- Owner labels on the disputed fragments (LAB.md §11.6): until then every
  figure is agreement between models, not correctness.
- Optional: DiariZen v1 / base variants (not installed).
- Not checked: screen readers, Firefox/Safari, real audio playback of the
  denoised WAVs of real models in the tab; the tab against the real run in
  a browser was not re-checked in this update.
- Heat map is per denoiser or per diarizer by design; the scores table does
  not show the non-commercial flag (score rows carry no licence).
- `whisper-large-v3` took 313 s on the original but 29 to 41 s on the
  denoised tracks; the cause was not investigated.

## How to continue
Resume commands: LAB.md §12. Gate: `uv run --no-sync flake8 .`,
`uv run --no-sync mypy . --strict`,
`HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= uv run --no-sync pytest -q --cov -p no:cacheprovider`,
`node --test tests/web/*.test.mjs`. Never plain `uv run`.
