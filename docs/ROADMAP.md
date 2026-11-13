# Retrieval improvement roadmap

Goal: maximise recall@5 on both datasets while staying inside the time
limits (index ≤ 300 s, 200 questions ≤ 90 s), and measure every step so
that each decision is backed by a cost/benefit number.

## 1. Where the errors come from (baseline diagnosis)

Baseline: hybrid `bm25/max + 0.4·cos`, chunks ≤ 2000 chars, MiniLM 128
tokens. 199 public questions, one reference span each.

| Level of success | docs k=5 | code k=5 | docs k=50 | code k=50 |
|---|---|---|---|---|
| Right **file** retrieved | 0.970 | 0.939 | 1.000 | 1.000 |
| Right file, span overlaps | 0.930 | 0.889 | 0.980 | 0.980 |
| Counted by the moulinette (IoU ≥ 0.05) | **0.910** | **0.879** | 0.970 | 0.970 |

Reading:

- **Intra-file ranking** is the largest gap: 6 points on each dataset
  have the right file in the top 5 but the wrong chunk of it.
- **Inter-file ranking**: 3 points (docs) and 6 points (code) do not
  have the right file in the top 5, but all of them have it in the
  top 50. No question is lost by candidate generation.
- **Wrong corpus area**: docs questions spend 32% of their top-5 slots
  on non-documentation files; code questions spend 8% on `tests/`. The
  reference is in `docs/` for 95% of docs questions and in non-test
  Python for 100% of code questions.
- **IoU / boundaries**: 1–2 points overlap the reference but fail the
  IoU rule (reference far smaller than the chunk or cut by a boundary).
- **Ceiling**: ~0.97. About 3% of the references are not the only place
  where the answer lives (label noise).

Statistical caution: with 100 questions the standard error of a recall
of 0.9 is 3 points. A change of 1–2 questions is noise. Every comparison
therefore uses per-question paired results, a bootstrap confidence
interval and cross-validation for tuned values.

## 2. Hypotheses and strategies

| ID | Strategy | Attacks | Stage | Expected gain | Extra cost |
|---|---|---|---|---|---|
| S1 | **Query routing**: a small model predicts whether the answer lives in code or docs, and scales chunk scores by kind | wrong area | search | high (docs) | ~0 |
| S2 | **Path prior**: down-weight test files | wrong area | search | medium (code) | ~0 |
| S3 | **File aggregation**: add a share of the best score of the file to each of its chunks | inter-file | search | medium | ~0 |
| S4 | **Field weights**: count path and heading/definition terms more than body terms | inter/intra-file | index (lexical) | medium | ~0 |
| S5 | **BM25 `k1`, `b`** sweep | both | index (lexical) | low–medium | 0 |
| S6 | **Chunk size** (target 600–2000) | intra-file, IoU | index | medium | embeddings ∝ chunks |
| S7 | **Table-aware Markdown chunks**: table header kept as context of every row group | intra-file | index | low (3 questions) | ~0 |
| S8 | **Embedding model / input length / query prefix** | paraphrases | index | medium | index time |
| S9 | **Hybrid weight and fusion** re-tuned after the above | both | search | low | 0 |
| S10 | **Cross-encoder reranker** depth/weight (already built) | rank 6–10 | search | low–medium | ×4.5 search |

Rejected earlier, with numbers (see `docs/TASKS.md`): narrowing returned
spans (T-21), heavier rerankers.

## 3. Measurement protocol

- **Metrics per trial**: recall@1/3/5/10 and MRR per dataset, from the
  rank of the first chunk with IoU ≥ 0.05 (same rule as the moulinette).
  Objective: mean recall@5 of both datasets, ties broken by MRR.
- **Cost per trial**: warm search seconds per 100 questions, index
  seconds, number of chunks.
- **Stored per trial**: full parameters, metrics, timings and the rank
  obtained by every question, in `experiments/results/trials.jsonl`.
  Per-question ranks allow paired comparisons between any two trials.
- **Separately**: one factor changed at a time against the baseline
  (`ablation` suites).
- **Together**: factorial grids of the factors that helped, to measure
  interactions: `synergy = gain(A+B) − gain(A) − gain(B)`.
- **Fixed values**: each numeric constant has a sweep suite; the result
  is a cost/benefit table (gain in points, extra seconds, points/second).
- **Overfitting control**: tuned values are selected by 5-fold
  cross-validation; the report gives the held-out estimate next to the
  in-sample one.
- **Final check**: the chosen profile is run through the real CLI and
  the official moulinette.

## 4. Ordered tasks

| Order | Task | Deliverable | How it is tested |
|---|---|---|---|
| 1 | T-22 Experiment lab | `ExperimentService`, trial store, suites as JSON, `experiment` command | unit (fakes) + integration (tiny corpus) |
| 2 | T-23 Analysis module | `AnalysisService`: tables, paired bootstrap, Pareto front, interactions, recommended profiles, `analyze` command → `experiments/REPORT.md` | unit on synthetic trials |
| 3 | T-24 Search-time strategies S1–S3 | `ScoreAdjuster` port, router model, priors | unit + ablation suite |
| 4 | T-25 Lexical index strategies S4, S5, S7 | field weights, table chunks | unit + sweep suites |
| 5 | T-26 Chunk size and embeddings S6, S8 | embedding cache, model options | sweep suites (index time measured cold) |
| 6 | T-27 Tuner | cross-validated coordinate descent over the winning factors, `tune` command → profiles | unit on synthetic objective |
| 7 | T-28 Combination and interactions S9, S10 | factorial suite, Pareto front precision/time | analysis report |
| 8 | T-29 Adopt the best profile | new defaults, moulinette run, docs | full test suite + moulinette |

## 5. Results

All tasks are done. Final state, checked with the official moulinette:

| | Before | After | Limit |
|---|---|---|---|
| docs recall@5 | 0.910 | **0.950** | ≥ 0.80 |
| code recall@5 | 0.879 | **0.919** | ≥ 0.50 |
| docs recall@10 | 0.94 | 0.97 | |
| code recall@10 | 0.92 | 0.94 | |
| Full indexing | 92 s | **60 s** | 300 s |
| 100 questions | 4 s | 4 s | 45 s |

| Strategy | Outcome |
|---|---|
| S1 routing | Adopted at 0.1. +1.0 alone; chosen by 5 of 5 folds (0.1 or 0.2). |
| S2 test prior | Adopted at 0.6. No recall@5 change alone, better MRR; 5 of 5 folds. |
| S3 file aggregation | Adopted at 0.2. +0.5 alone; 4 of 5 folds. |
| S4 field weights | Adopted, path ×3 and context ×3. The largest single effects (+1.5 to +2.0); 5 of 5 folds. Removing either field costs 2.5–3.5 points. |
| S5 BM25 constants | `b` 0.75 → 0.5 adopted (4 of 5 folds). `k1` unstable, kept at 1.2. |
| S6 chunk size | Not adopted. Target sizes 600–1500: −1.0 to 0.0. `max_chunk_size` 1500: +1.5 alone but only 3 of 5 folds; 1000: −2.5. |
| S7 table context | Adopted (5 of 5 folds), +0.5 alone. |
| S8 embeddings | 64 input tokens adopted for speed (95 s → 60 s, recall +0.5 to +1.0, within noise). 256 tokens −1.0 and 205 s; `bge-small` −1.5 and 228 s; `multi-qa-MiniLM` +0.5; body-only text −0.5. |
| S9 hybrid weight | Unstable across folds (0.3, 0.4, 0.5, 0.6 all chosen). Kept at 0.4. RRF −4.0. |
| S10 reranker | +1.5 on the untuned pipeline, 0.0 to +0.5 on the tuned one, for +14 s per 100 questions. Kept optional. |
| S11 word bigrams (added during the work) | −1.0 alone, +1.5 combined against +2.0 without. Not adopted. |

What the numbers say about the method:

- The diagnosis predicted up to 6 points from intra-file ranking. The
  adopted changes recovered about 4 of the 10.5 points missing on
  average; the rest is still "right file, wrong chunk" plus label noise.
- Single changes are within noise; the cross-validated estimate of the
  whole tuning is +2.8 objective points against +6.0 in-sample. The
  consensus rule (a change must be made by ≥ 4 folds and agreed by ≥ 3)
  is what decides the defaults.
- Ideas not tried, by expected value: a chunk-level learned ranker
  trained on the per-question results (risky with 199 questions), query
  expansion with pseudo-relevance feedback, and a second reference
  dataset to reduce the noise of every decision.

## 6. Reuse for the soft-skills platform

The lab is independent of the corpus: a suite names a dataset with
reference spans and a set of parameters. For transcripts, the same
commands will tune chunking by speaker turn, rubric-to-evidence
retrieval and the reranker against human-annotated evidence spans.
