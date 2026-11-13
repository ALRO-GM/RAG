# Experiment report

347 trials. Recall@5 uses the moulinette rule (IoU ≥ 0.05). `Δ points` is the change of mean recall@5 against the baseline of the suite, with a paired bootstrap 95% interval; `P(gain)` is the share of resamples where the change wins. `s/100 q` is the warm search time per 100 questions and `points/s` the recall points gained per extra second.

Verdicts: **helps/hurts** when the interval excludes 0; **likely helps/hurts** when P(gain) is beyond 0.8/0.2; otherwise **neutral** (within noise).

## Recommended profiles

| Profile | Mean R@5 | R@5 code | R@5 docs | s/100 q | Why |
|---|---|---|---|---|---|
| **precision** | 0.935 | 0.919 | 0.950 | 0.34 | highest objective within 22 s per 100 questions |
| **balanced** | 0.935 | 0.919 | 0.950 | 0.34 | last point of the Pareto front that still gains 0.5 recall points per extra second |
| **speed** | 0.925 | 0.919 | 0.930 | 0.03 | fastest cold indexing + 200 questions (2 s) at least 5 points above every required recall@5 |

`precision`: `{'semantic': True, 'mode': 'auto', 'rerank': False, 'embedding_max_tokens': 64}`

`balanced`: `{'semantic': True, 'mode': 'auto', 'rerank': False, 'embedding_max_tokens': 64}`

`speed`: `{'semantic': False, 'mode': 'bm25', 'rerank': False}`

## Pareto front: recall against search time

Configurations that no other one beats on both axes, from the fastest to the most precise.

| s/100 q | Mean R@5 | Suite | Configuration |
|---|---|---|---|
| 0.34 | 0.935 | end_to_end | `embedding_max_tokens=64` |

## Pareto front: recall against cold indexing + 200 questions

Measured with every cache disabled (`lab_cold`). The limits of the subject add up to 390 s (300 s indexing + 90 s for 200 questions).

| Index s | 200 questions s | Total s | Mean R@5 | Configuration |
|---|---|---|---|---|
| 2 | 0.1 | 2 | 0.925 | lexical only (no embeddings) |
| 63 | 0.7 | 64 | 0.935 | hybrid, 64-token embeddings |

## Cost/benefit per suite

### search_priors

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.33 | — | — | baseline |
| `route_strength=0.2` | 0.889 | 0.920 | 0.904 | 0.778 | +1.0 (+0.0, +2.5) | 0.85 | 0.34 | +0.01 | — | likely helps |
| `route_strength=0.1` | 0.889 | 0.920 | 0.904 | 0.777 | +1.0 (+0.0, +2.5) | 0.85 | 0.34 | +0.01 | — | likely helps |
| `semantic_weight=0.5` | 0.899 | 0.910 | 0.904 | 0.757 | +1.0 (+0.0, +2.5) | 0.88 | 0.36 | +0.03 | — | likely helps |
| `semantic_weight=0.6` | 0.899 | 0.910 | 0.904 | 0.755 | +1.0 (+0.0, +2.5) | 0.88 | 0.29 | -0.04 | — | likely helps |
| `semantic_weight=0.8` | 0.899 | 0.910 | 0.904 | 0.752 | +1.0 (+0.0, +2.5) | 0.88 | 0.30 | -0.03 | — | likely helps |
| `route_strength=0.3` | 0.879 | 0.920 | 0.899 | 0.778 | +0.5 (-1.0, +2.5) | 0.61 | 0.27 | -0.06 | — | neutral |
| `route_strength=0.5` | 0.879 | 0.920 | 0.899 | 0.777 | +0.5 (-1.0, +2.5) | 0.61 | 0.59 | +0.26 | +1.94 | neutral |
| `route_strength=0.8` | 0.879 | 0.920 | 0.899 | 0.777 | +0.5 (-1.0, +2.5) | 0.61 | 0.32 | -0.02 | — | neutral |
| `semantic_weight=0.3` | 0.889 | 0.910 | 0.899 | 0.763 | +0.5 (-1.0, +2.0) | 0.72 | 0.30 | -0.03 | — | neutral |
| `file_boost=0.3` | 0.879 | 0.920 | 0.899 | 0.763 | +0.5 (+0.0, +1.5) | 0.64 | 0.29 | -0.05 | — | neutral |
| `file_boost=0.5` | 0.879 | 0.920 | 0.899 | 0.763 | +0.5 (+0.0, +1.5) | 0.64 | 0.28 | -0.05 | — | neutral |
| `file_boost=0.2` | 0.879 | 0.920 | 0.899 | 0.762 | +0.5 (+0.0, +1.5) | 0.64 | 0.51 | +0.18 | +2.78 | neutral |
| `file_boost=0.1` | 0.879 | 0.920 | 0.899 | 0.760 | +0.5 (+0.0, +1.5) | 0.64 | 0.33 | -0.00 | — | neutral |
| `file_boost=0.05` | 0.879 | 0.920 | 0.899 | 0.760 | +0.5 (+0.0, +1.5) | 0.64 | 0.28 | -0.05 | — | neutral |
| `test_weight=0.8` | 0.879 | 0.910 | 0.894 | 0.763 | +0.0 (+0.0, +0.0) | 0.00 | 0.28 | -0.05 | — | neutral |
| `test_weight=0.6` | 0.879 | 0.910 | 0.894 | 0.763 | +0.0 (+0.0, +0.0) | 0.00 | 0.30 | -0.03 | — | neutral |
| `test_weight=0.4` | 0.879 | 0.910 | 0.894 | 0.763 | +0.0 (+0.0, +0.0) | 0.00 | 0.30 | -0.04 | — | neutral |
| `test_weight=0.2` | 0.879 | 0.910 | 0.894 | 0.763 | +0.0 (+0.0, +0.0) | 0.00 | 0.29 | -0.04 | — | neutral |
| `test_weight=0.9` | 0.879 | 0.910 | 0.894 | 0.762 | +0.0 (+0.0, +0.0) | 0.00 | 0.36 | +0.02 | — | neutral |
| `semantic_weight=0.2` | 0.889 | 0.890 | 0.889 | 0.759 | -0.5 (-2.5, +1.5) | 0.37 | 0.31 | -0.03 | — | neutral |
| `semantic_weight=0.0` | 0.869 | 0.890 | 0.879 | 0.757 | -1.5 (-3.5, +0.5) | 0.05 | 0.32 | -0.01 | — | likely hurts |
| `fusion=rrf` | 0.838 | 0.870 | 0.854 | 0.663 | -4.0 (-8.0, -0.5) | 0.01 | 0.29 | -0.04 | — | hurts |

### lexical_index

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.29 | — | — | baseline |
| `path_weight=2` | 0.909 | 0.920 | 0.915 | 0.757 | +2.0 (+0.5, +4.0) | 0.98 | 0.33 | +0.05 | — | helps |
| `bm25_k1=0.9` | 0.909 | 0.910 | 0.910 | 0.761 | +1.5 (-0.5, +3.5) | 0.94 | 0.31 | +0.02 | — | likely helps |
| `context_weight=4` | 0.909 | 0.910 | 0.910 | 0.759 | +1.5 (-0.5, +3.5) | 0.94 | 0.29 | +0.01 | — | likely helps |
| `context_weight=3` | 0.909 | 0.910 | 0.910 | 0.756 | +1.5 (-0.5, +3.5) | 0.94 | 0.34 | +0.05 | +29.14 | likely helps |
| `path_weight=3` | 0.909 | 0.910 | 0.910 | 0.754 | +1.5 (-0.5, +3.5) | 0.94 | 0.29 | +0.00 | — | likely helps |
| `context_weight=2` | 0.909 | 0.910 | 0.910 | 0.754 | +1.5 (-0.5, +3.5) | 0.94 | 0.29 | -0.00 | — | likely helps |
| `bm25_b=0.3` | 0.899 | 0.910 | 0.904 | 0.747 | +1.0 (-1.5, +3.5) | 0.82 | 0.29 | +0.01 | — | likely helps |
| `bm25_b=0.9` | 0.889 | 0.910 | 0.899 | 0.773 | +0.5 (+0.0, +1.5) | 0.65 | 0.33 | +0.04 | — | neutral |
| `bm25_b=0.6` | 0.879 | 0.920 | 0.899 | 0.764 | +0.5 (+0.0, +1.5) | 0.64 | 0.36 | +0.08 | +6.58 | neutral |
| `table_context=True` | 0.879 | 0.920 | 0.899 | 0.761 | +0.5 (+0.0, +1.5) | 0.63 | 0.28 | -0.01 | — | neutral |
| `bm25_k1=0.6` | 0.879 | 0.910 | 0.894 | 0.763 | +0.0 (-2.5, +2.5) | 0.46 | 0.28 | -0.01 | — | neutral |
| `bm25_b=0.5` | 0.889 | 0.900 | 0.894 | 0.756 | +0.0 (-2.0, +2.0) | 0.52 | 0.30 | +0.01 | — | neutral |
| `bm25_k1=1.5` | 0.879 | 0.910 | 0.894 | 0.756 | +0.0 (+0.0, +0.0) | 0.00 | 0.32 | +0.03 | — | neutral |
| `bm25_k1=2.0` | 0.879 | 0.900 | 0.889 | 0.756 | -0.5 (-1.5, +0.0) | 0.00 | 0.34 | +0.05 | — | likely hurts |
| `context_weight=0` | 0.848 | 0.890 | 0.869 | 0.752 | -2.5 (-5.0, -0.5) | 0.00 | 0.33 | +0.04 | — | hurts |
| `path_weight=0` | 0.818 | 0.900 | 0.859 | 0.744 | -3.5 (-6.6, -1.0) | 0.00 | 0.29 | -0.00 | — | hurts |

### rerank

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.31 | — | — | baseline |
| `rerank=True` | 0.909 | 0.910 | 0.910 | 0.758 | +1.5 (+0.0, +3.5) | 0.96 | 13.32 | +13.01 | +0.12 | likely helps |
| `rerank=True`, `rerank_depth=20` | 0.909 | 0.910 | 0.910 | 0.756 | +1.5 (+0.0, +3.5) | 0.96 | 25.09 | +24.78 | +0.06 | likely helps |
| `rerank=True`, `rerank_weight=0.2` | 0.909 | 0.910 | 0.910 | 0.753 | +1.5 (+0.0, +3.5) | 0.96 | 12.60 | +12.29 | +0.12 | likely helps |
| `rerank=True`, `rerank_weight=0.5` | 0.909 | 0.910 | 0.910 | 0.738 | +1.5 (-0.5, +4.0) | 0.94 | 12.26 | +11.95 | +0.13 | likely helps |
| `rerank=True`, `rerank_depth=5` | 0.879 | 0.910 | 0.894 | 0.754 | +0.0 (+0.0, +0.0) | 0.00 | 6.84 | +6.53 | +0.00 | neutral |

### chunking

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.29 | — | — | baseline |
| `max_chunk_size=1500` | 0.909 | 0.910 | 0.910 | 0.771 | +1.5 (-1.5, +4.5) | 0.86 | 0.28 | -0.01 | — | likely helps |
| `target_chunk_size=900` | 0.889 | 0.900 | 0.894 | 0.775 | +0.0 (-3.5, +3.0) | 0.52 | 0.29 | +0.01 | — | neutral |
| `target_chunk_size=600` | 0.869 | 0.920 | 0.894 | 0.761 | -0.0 (-3.5, +3.5) | 0.45 | 0.35 | +0.06 | -0.09 | neutral |
| `target_chunk_size=1200` | 0.889 | 0.900 | 0.894 | 0.757 | +0.0 (-3.0, +3.0) | 0.52 | 0.36 | +0.07 | +0.07 | neutral |
| `target_chunk_size=1500` | 0.869 | 0.900 | 0.884 | 0.774 | -1.0 (-3.5, +1.0) | 0.15 | 0.29 | -0.00 | — | likely hurts |
| `max_chunk_size=1000` | 0.869 | 0.870 | 0.869 | 0.743 | -2.5 (-6.5, +1.0) | 0.11 | 0.27 | -0.02 | — | likely hurts |

### embeddings

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.33 | — | — | baseline |
| `embedding_max_tokens=64` | 0.889 | 0.920 | 0.904 | 0.752 | +1.0 (-1.0, +3.0) | 0.78 | 0.30 | -0.03 | — | neutral |
| `embedding_model=sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 0.889 | 0.910 | 0.899 | 0.764 | +0.5 (+0.0, +1.5) | 0.62 | 0.32 | -0.01 | — | neutral |
| `embed_style=body` | 0.879 | 0.900 | 0.889 | 0.774 | -0.5 (-2.5, +1.5) | 0.31 | 0.31 | -0.02 | — | neutral |
| `embedding_max_tokens=256` | 0.889 | 0.880 | 0.884 | 0.753 | -1.0 (-3.0, +1.0) | 0.22 | 0.28 | -0.04 | — | neutral |
| `embedding_model=BAAI/bge-small-en-v1.5`, `query_prefix=Represent this sentence for searching relevant passages: ` | 0.869 | 0.890 | 0.879 | 0.766 | -1.5 (-3.5, +0.5) | 0.05 | 0.47 | +0.14 | -10.45 | likely hurts |

### interactions

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.30 | — | — | baseline |
| `context_weight=2`, `bm25_k1=0.9`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.780 | +2.0 (+0.5, +4.0) | 0.98 | 0.28 | -0.02 | — | helps |
| `path_weight=2`, `context_weight=2`, `bm25_k1=0.9`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.779 | +2.0 (+0.5, +4.0) | 0.98 | 0.31 | +0.01 | — | helps |
| `path_weight=2`, `context_weight=2`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.777 | +2.0 (+0.5, +4.0) | 0.98 | 0.28 | -0.02 | — | helps |
| `path_weight=2`, `bm25_k1=0.9`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.776 | +2.0 (+0.5, +4.0) | 0.98 | 0.28 | -0.02 | — | helps |
| `bm25_k1=0.9`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.775 | +2.0 (+0.5, +4.0) | 0.98 | 0.29 | -0.01 | — | helps |
| `context_weight=2`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.775 | +2.0 (+0.5, +4.0) | 0.98 | 0.29 | -0.01 | — | helps |
| `path_weight=2`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.774 | +2.0 (+0.5, +4.0) | 0.98 | 0.28 | -0.02 | — | helps |
| `path_weight=2` | 0.909 | 0.920 | 0.915 | 0.757 | +2.0 (+0.5, +4.0) | 0.98 | 0.29 | -0.01 | — | helps |
| `bm25_k1=0.9` | 0.909 | 0.910 | 0.910 | 0.761 | +1.5 (-0.5, +3.5) | 0.94 | 0.30 | -0.00 | — | likely helps |
| `path_weight=2`, `bm25_k1=0.9` | 0.909 | 0.910 | 0.910 | 0.760 | +1.5 (-0.5, +3.5) | 0.94 | 0.33 | +0.03 | — | likely helps |
| `context_weight=2`, `bm25_k1=0.9` | 0.909 | 0.910 | 0.910 | 0.759 | +1.5 (-0.5, +3.5) | 0.94 | 0.30 | +0.00 | — | likely helps |
| `path_weight=2`, `context_weight=2` | 0.909 | 0.910 | 0.910 | 0.756 | +1.5 (-0.5, +3.5) | 0.94 | 0.29 | -0.01 | — | likely helps |
| `path_weight=2`, `context_weight=2`, `bm25_k1=0.9` | 0.909 | 0.910 | 0.910 | 0.756 | +1.5 (-0.5, +3.5) | 0.94 | 0.32 | +0.02 | — | likely helps |
| `context_weight=2` | 0.909 | 0.910 | 0.910 | 0.754 | +1.5 (-0.5, +3.5) | 0.94 | 0.33 | +0.03 | — | likely helps |
| `route_strength=0.2` | 0.889 | 0.920 | 0.904 | 0.778 | +1.0 (+0.0, +2.5) | 0.85 | 0.27 | -0.03 | — | likely helps |

### tune_all/tune

252 configurations explored by the tuner; the 10 best in-sample are shown. They are not candidates for the profiles (see the tuning section).

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 0.40 | — | — | baseline |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.2`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.806 | +5.5 (+2.0, +9.1) | 1.00 | 0.30 | -0.11 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.2`, `file_boost=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.806 | +5.5 (+2.0, +9.1) | 1.00 | 0.28 | -0.12 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.2`, `file_boost=0.2`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.806 | +5.5 (+2.0, +9.1) | 1.00 | 0.28 | -0.12 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.804 | +5.5 (+2.0, +9.1) | 1.00 | 0.27 | -0.14 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.1`, `file_boost=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.804 | +5.5 (+2.0, +9.1) | 1.00 | 0.27 | -0.14 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.1`, `file_boost=0.2`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.803 | +5.5 (+2.0, +9.1) | 1.00 | 0.28 | -0.12 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.9`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.5`, `route_strength=0.1`, `file_boost=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.803 | +5.5 (+2.0, +9.1) | 1.00 | 0.27 | -0.13 | — | helps |
| `path_weight=2`, `context_weight=3`, `bm25_k1=0.9`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.5`, `route_strength=0.1`, `file_boost=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.802 | +5.5 (+2.0, +9.1) | 1.00 | 0.29 | -0.12 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.3`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.802 | +5.5 (+2.0, +9.1) | 1.00 | 0.28 | -0.12 | — | helps |
| `path_weight=3`, `context_weight=3`, `bm25_k1=0.6`, `bm25_b=0.5`, `table_context=True`, `max_chunk_size=1500`, `semantic_weight=0.3`, `route_strength=0.3`, `file_boost=0.1`, `test_weight=0.6` | 0.949 | 0.950 | 0.950 | 0.802 | +5.5 (+2.0, +9.1) | 1.00 | 0.28 | -0.12 | — | helps |

### bigrams

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.879 | 0.910 | 0.894 | 0.760 | — | — | 2.26 | — | — | baseline |
| `path_weight=2`, `route_strength=0.2` | 0.909 | 0.920 | 0.915 | 0.774 | +2.0 (+0.5, +4.0) | 0.98 | 2.45 | +0.20 | +10.23 | helps |
| `bigrams=True`, `path_weight=2`, `route_strength=0.2` | 0.889 | 0.930 | 0.909 | 0.786 | +1.5 (-1.0, +4.5) | 0.82 | 2.56 | +0.31 | +4.93 | likely helps |
| `bigrams=True`, `path_weight=2`, `route_strength=0.2`, `bm25_k1=0.9` | 0.889 | 0.930 | 0.909 | 0.783 | +1.5 (-1.0, +4.0) | 0.82 | 2.54 | +0.28 | +5.36 | likely helps |
| `bigrams=True` | 0.869 | 0.900 | 0.884 | 0.758 | -1.0 (-4.5, +2.0) | 0.23 | 1.34 | -0.92 | — | neutral |

### final

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.919 | 0.940 | 0.930 | 0.781 | — | — | 0.64 | — | — | baseline |
| `rerank=True` | 0.929 | 0.940 | 0.935 | 0.779 | +0.5 (+0.0, +1.5) | 0.65 | 13.95 | +13.31 | +0.04 | neutral |
| `rerank=True`, `rerank_depth=5` | 0.919 | 0.940 | 0.930 | 0.789 | +0.0 (+0.0, +0.0) | 0.00 | 7.08 | +6.44 | +0.00 | neutral |
| `rerank=True`, `rerank_depth=20` | 0.919 | 0.930 | 0.925 | 0.779 | -0.5 (-2.5, +1.0) | 0.27 | 25.95 | +25.31 | -0.02 | neutral |
| `mode=bm25`, `semantic=False` | 0.919 | 0.930 | 0.925 | 0.774 | -0.5 (-1.5, +0.0) | 0.00 | 0.03 | -0.61 | — | likely hurts |
| `rerank=True`, `rerank_weight=0.5` | 0.919 | 0.930 | 0.925 | 0.759 | -0.5 (-3.0, +2.0) | 0.34 | 13.87 | +13.23 | -0.04 | neutral |
| `path_weight=1`, `context_weight=1`, `bm25_b=0.75`, `table_context=False`, `route_strength=0.0`, `file_boost=0.0`, `test_weight=1.0` | 0.879 | 0.910 | 0.894 | 0.760 | -3.5 (-6.5, -0.5) | 0.01 | 0.36 | -0.28 | — | hurts |

Cold index builds: `mode=bm25`, `semantic=False` → 2 s (13287 chunks)

### end_to_end

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.919 | 0.940 | 0.930 | 0.781 | — | — | 0.35 | — | — | baseline |
| `embedding_max_tokens=64` | 0.919 | 0.950 | 0.935 | 0.790 | +0.5 (+0.0, +1.5) | 0.63 | 0.34 | -0.02 | — | neutral |
| `rerank=True` | 0.929 | 0.940 | 0.935 | 0.779 | +0.5 (+0.0, +1.5) | 0.65 | 13.29 | +12.93 | +0.04 | neutral |
| `semantic=False`, `mode=bm25` | 0.919 | 0.930 | 0.925 | 0.774 | -0.5 (-1.5, +0.0) | 0.00 | 0.03 | -0.33 | — | likely hurts |

Cold index builds: baseline → 95 s (13287 chunks); `semantic=False`, `mode=bm25` → 2 s (13287 chunks); `rerank=True` → 89 s (13287 chunks); `embedding_max_tokens=64` → 63 s (13287 chunks)

### modes

| Change | R@5 code | R@5 docs | Mean R@5 | MRR | Δ points (95% CI) | P(gain) | s/100 q | Δ s | points/s | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.919 | 0.950 | 0.935 | 0.790 | — | — | 3.27 | — | — | baseline |
| `rerank=True` | 0.929 | 0.940 | 0.935 | 0.784 | +0.0 (-1.5, +1.5) | 0.53 | 48.73 | +45.46 | +0.00 | neutral |
| `mode=bm25` | 0.919 | 0.930 | 0.925 | 0.774 | -1.0 (-2.5, +0.0) | 0.00 | 0.06 | -3.21 | — | likely hurts |
| `mode=semantic` | 0.556 | 0.660 | 0.608 | 0.444 | -32.7 (-39.7, -25.7) | 0.00 | 3.10 | -0.17 | — | hurts |

## Tuning: tune_all

Objective = mean recall@5 + 0.1·MRR. *In-sample* tunes and scores on the same questions (optimistic). *Held-out* tunes on four fifths of the questions and scores on the rest: it is the gain to expect on unseen questions.

| | Baseline | Tuned | Gain (points) |
|---|---|---|---|
| In-sample | 0.9704 | 1.0300 | +6.0 |
| Held-out (cross-validation) | 0.9707 | 0.9982 | +2.8 |
| Consensus profile, in-sample | 0.9704 | 1.0077 | +3.7 |

Stability: value chosen by each fold. A factor is adopted in the consensus profile only when most folds agree.

| Factor | Values chosen (folds) | Consensus |
|---|---|---|
| `path_weight` | 3 ×4, 2 ×1 | `3` |
| `context_weight` | 3 ×3, 2 ×2 | `3` |
| `bm25_k1` | 1.2 ×3, 0.6 ×2 | `1.2` |
| `bm25_b` | 0.5 ×3, 0.75 ×1, 0.3 ×1 | `0.5` |
| `table_context` | True ×5 | `True` |
| `max_chunk_size` | 1500 ×3, 2000 ×2 | `2000` |
| `semantic_weight` | 0.3 ×2, 0.4 ×1, 0.6 ×1, 0.5 ×1 | `0.4` |
| `route_strength` | 0.1 ×3, 0.2 ×2 | `0.1` |
| `file_boost` | 0.2 ×3, 0.1 ×1, 0.0 ×1 | `0.2` |
| `test_weight` | 0.6 ×5 | `0.6` |

Tuned (in-sample best): `{'path_weight': 3, 'context_weight': 3, 'bm25_k1': 0.9, 'bm25_b': 0.5, 'table_context': True, 'max_chunk_size': 1500, 'semantic_weight': 0.5, 'route_strength': 0.1, 'file_boost': 0.1, 'test_weight': 0.6}`

Consensus: `{'path_weight': 3, 'context_weight': 3, 'bm25_k1': 1.2, 'bm25_b': 0.5, 'table_context': True, 'max_chunk_size': 2000, 'semantic_weight': 0.4, 'route_strength': 0.1, 'file_boost': 0.2, 'test_weight': 0.6}`

## Interactions

`synergy = gain(A+B) − gain(A) − gain(B)` in recall points: about 0 means the gains add up, negative means they overlap (both fix the same questions), positive means they need each other.

| Suite | A | B | Gain A | Gain B | Gain A+B | Synergy |
|---|---|---|---|---|---|---|
| interactions | `bm25_k1=0.9` | `route_strength=0.2` | +1.5 | +1.0 | +2.0 | -0.5 |
| interactions | `context_weight=2` | `route_strength=0.2` | +1.5 | +1.0 | +2.0 | -0.5 |
| interactions | `context_weight=2` | `bm25_k1=0.9` | +1.5 | +1.5 | +1.5 | -1.5 |
| interactions | `path_weight=2` | `route_strength=0.2` | +2.0 | +1.0 | +2.0 | -1.0 |
| interactions | `path_weight=2` | `bm25_k1=0.9` | +2.0 | +1.5 | +1.5 | -2.0 |
| interactions | `path_weight=2` | `context_weight=2` | +2.0 | +1.5 | +1.5 | -2.0 |
