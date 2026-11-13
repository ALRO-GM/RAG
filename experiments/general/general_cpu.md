# Generalist study: `general_cpu`

Relative change against the suite baseline on each dataset (42 datasets: recall@5; public ones: nDCG@10). Configurations sorted by mean relative change; only those measured on every dataset are ranked.

| Configuration | mean | worst | W/L | 42:docs | 42:code | scifact | nfcorpus | arguana | scidocs | fiqa | coir:cosqa | coir:stackoverflow-qa | index s | search s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minilm-256 convex 4.0 adaptive->1.0 sat 0.1 **(best mean)** | +11.7% | -6.3% | 6/3 | -6.3% | -5.5% | +3.7% | -2.4% | +6.5% | +20.9% | +38.1% | +34.6% | +15.4% | 17 | 22.9 |
| minilm-256 convex 4.0 adaptive->1.0 sat 0.2 | +11.6% | -8.4% | 6/3 | -8.4% | -5.5% | +3.6% | -2.4% | +6.5% | +21.5% | +38.3% | +34.6% | +16.1% | 17 | 22.8 |
| minilm-256 linear 2.0 adaptive->0.4 sat 0.2 **(robust)** | +10.8% | -1.1% | 7/1 | -1.1% | +0.0% | +5.7% | +4.9% | +6.9% | +14.5% | +33.4% | +22.0% | +11.1% | 17 | 20.4 |
| minilm-256 linear 2.0 adaptive->0.4 sat 0.1 | +10.4% | -1.1% | 7/1 | +0.0% | -1.1% | +5.1% | +4.9% | +6.8% | +13.7% | +32.6% | +22.0% | +9.4% | 28 | 21.0 |
| minilm-64 linear 2.0 adaptive->0.4 sat 0.1 | +6.0% | -5.3% | 7/1 | +1.1% | +0.0% | +2.9% | +4.2% | -5.3% | +12.2% | +16.8% | +19.6% | +2.9% | 18 | 19.1 |
| minilm-64 linear 2.0 adaptive->0.4 sat 0.2 | +5.7% | -5.2% | 6/3 | -2.1% | -1.1% | +2.8% | +4.2% | -5.2% | +12.9% | +17.4% | +19.6% | +3.4% | 17 | 19.5 |
| minilm-64 convex 4.0 adaptive->1.0 sat 0.1 | +1.2% | -13.0% | 3/6 | -6.3% | -7.7% | -3.6% | -3.7% | -13.0% | +17.4% | +6.6% | +27.5% | -6.3% | 16 | 21.7 |
| minilm-64 convex 4.0 adaptive->1.0 sat 0.2 | +0.3% | -13.0% | 3/6 | -10.5% | -7.7% | -4.3% | -3.7% | -13.0% | +18.1% | +6.9% | +27.5% | -10.2% | 17 | 21.7 |
| baseline | +0.0% | +0.0% | 0/0 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 36 | 19.1 |

Baseline scores: 42:docs 0.950, 42:code 0.919, scifact 0.691, nfcorpus 0.339, arguana 0.503, scidocs 0.173, fiqa 0.281, coir:cosqa 0.220, coir:stackoverflow-qa 0.716.

- Best mean: **minilm-256 convex 4.0 adaptive->1.0 sat 0.1**.
- Robust choice (best mean losing at most 2% on every dataset): **minilm-256 linear 2.0 adaptive->0.4 sat 0.2**.

## Leave one dataset out

Chosen on the other datasets, scored on the one left out.

| Held out | chosen | its change | rank | best there | regret |
|---|---|---|---|---|---|
| 42:docs | minilm-256 convex 4.0 adaptive->1.0 sat 0.2 | -8.4% | 8/9 | minilm-64 linear 2.0 adaptive->0.4 sat 0.1 (+1.1%) | 9.5 pts |
| 42:code | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | -5.5% | 6/9 | baseline (+0.0%) | 5.5 pts |
| scifact | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | +3.7% | 3/9 | minilm-256 linear 2.0 adaptive->0.4 sat 0.2 (+5.7%) | 2.0 pts |
| nfcorpus | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | -2.4% | 6/9 | minilm-256 linear 2.0 adaptive->0.4 sat 0.1 (+4.9%) | 7.3 pts |
| arguana | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | +6.5% | 3/9 | minilm-256 linear 2.0 adaptive->0.4 sat 0.2 (+6.9%) | 0.4 pts |
| scidocs | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | +20.9% | 2/9 | minilm-256 convex 4.0 adaptive->1.0 sat 0.2 (+21.5%) | 0.6 pts |
| fiqa | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | +38.1% | 2/9 | minilm-256 convex 4.0 adaptive->1.0 sat 0.2 (+38.3%) | 0.2 pts |
| coir:cosqa | minilm-256 linear 2.0 adaptive->0.4 sat 0.2 | +22.0% | 6/9 | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 (+34.6%) | 12.6 pts |
| coir:stackoverflow-qa | minilm-256 convex 4.0 adaptive->1.0 sat 0.1 | +15.4% | 2/9 | minilm-256 convex 4.0 adaptive->1.0 sat 0.2 (+16.1%) | 0.7 pts |

Mean regret 4.3 points; the held-out choice improved on the baseline in 6 of 9 datasets.
