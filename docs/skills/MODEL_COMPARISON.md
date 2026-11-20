# Haiku against Opus as the judge

`experiments/skills/model_comparison.py` runs the judge phases with one
Claude model and compares two saved runs. Synthetic data only.

## How to run

Run from the checkout that holds `data/` (transcripts, models, the reply
cache), with `HF_HUB_OFFLINE=1`:

```bash
S=experiments/skills/model_comparison.py
O=experiments/skills/results/model_comparison
PYTHONPATH=. python $S run --model haiku --dry-run          # calls it would make
PYTHONPATH=. python $S run --model haiku --cap 450 --out $O
PYTHONPATH=. python $S run --model opus --phases indicators --cap 150 --out $O
PYTHONPATH=. python $S compare --b haiku --out $O           # writes comparison.json
```

- Phases: `indicators` (the 120 items of
  `examples/skills/judge_indicators/reference.json`, forward order; the
  script reads it from branch `exp/85-judge-indicators` with `git show`
  when the file is not in the checkout), `acts` (the 140 sentences of
  `DIALOGUE_ACT_TAGGER.md` §2) and `session` (the whole analysis of
  `sim-reunion-verano-audio`: confirmation, assessment, synthesis).
- Every reply goes through the shared cache; its keys include the judge's
  name (`claude-code:<model>`), so each model has its own namespace and a
  second run of the same model costs nothing.
- `--cap` is a hard limit of real calls over all phases; `--dry-run` calls
  nothing and prints the calls per phase (indicators and acts are counted
  against the cache; the session figure is the calls of the stored Opus
  run, an upper bound).
- The Opus baseline of the 140 acts is
  `experiments/skills/results/act_tagger/llm_claude_code.json` and that of
  the session the stored analysis in `data/output/skills/analysis/`;
  only the indicators had no Opus answer, so they were run (120 calls).
- Dry-run calls: haiku 395 (120 + 140 + 135), sonnet the same, opus 120
  (indicators only).

## Result (run 2026-11-20)

Haiku made 410 real calls (120 indicators, 140 acts, 150 session: 102
confirmation, 44 assessment, 4 synthesis); Opus 120. Files:
`experiments/skills/results/model_comparison/{opus,haiku,comparison}.json`
and the Haiku analysis next to them.

| Phase | Items | Agreement with Opus | Accuracy Opus | Accuracy Haiku |
|---|---|---|---|---|
| acts (13 categories, vs intended act) | 140 | 78.6 % | 76.4 % | 72.1 % |
| indicators (vs author's verdict) | 120 | 78.3 % | 91.7 % | 77.5 % |
| session, confirmations | 102 | 67.6 % | no truth | no truth |
| session, ratings (person x competency, level and kind) | 44 | 59.1 % | no truth | no truth |
| session, firm levels within one step | 20 | 100 % | no truth | no truth |

Tokens (from `claude -p --output-format json`; `input` is only the few
uncached tokens, the prompt shows as cache creation or read; cost is the
equivalent API price Claude Code reports, not charged to a subscription):

| Model, phase | Calls | Input | Output | Cache write | Cache read | USD |
|---|---|---|---|---|---|---|
| haiku, indicators | 120 | 240 | 12,050 | 201,629 | 0 | 0.046 |
| haiku, acts | 140 | 280 | 20,314 | 92,383 | 87,431 | 0.030 |
| haiku, session | 150 | 300 | 88,498 | 287,729 | 4,585 | 0.102 |
| haiku, total | 410 | 820 | 120,862 | 581,741 | 92,016 | 0.178 |
| opus, indicators | 120 | 240 | 11,249 | 198,642 | 0 | 1.815 |

Caveats: the Opus acts and session figures come from earlier runs (acts
76.4 % here is the stored file; §2 reports 74.3 % for Opus and this
document does not explain the difference), whose tokens were not recorded, so only the indicator phase has
tokens for both models. The session has no ground truth: agreement is not
accuracy. One run each, no repeats, so differences of a few points are not
established. In the Haiku `run.json` the confirmation step's usage was
corrected by subtracting the earlier phases' calls of the same process
(a bug fixed in `SessionAnalysis`; the per-phase figures in `haiku.json`
were measured directly).

## Choosing the model per step

The Claude model of each judged step of `skills analyze` can be chosen
(default: `claude_model`, `opus`): from the web «Análisis» page when an
analysis is started or resumed, with `skills --claude_confirmation_model
haiku --claude_assessment_model … --claude_synthesis_model … analyze`, or
with the same settings in a profile. Accepted names are the aliases
`haiku`, `sonnet`, `opus` or a full id such as `claude-haiku-4-5`
(allow-list pattern in `src/domain/skills/claude_models.py`; checked again
by the transcriber and the RAG). The choice only applies when the judge is
Claude Code; each model has its own cache namespace and its own row in the
per-step token usage of `run.json`.
