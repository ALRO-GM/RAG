# Generalist study: `general_models`

Relative change against the suite baseline on each dataset (42 datasets: recall@5; public ones: nDCG@10). Configurations sorted by mean relative change; only those measured on every dataset are ranked.

| Configuration | mean | worst | W/L | 42:docs | 42:code | scifact | nfcorpus | arguana | scidocs | fiqa | coir:cosqa | coir:stackoverflow-qa | index s | search s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-large-512 convex 1.0 **(robust)** **(best mean)** | +14.0% | +0.0% | 8/0 | +0.0% | +1.1% | +8.1% | +11.3% | +23.8% | +14.2% | +37.2% | +18.9% | +11.0% | 1088 | 59.1 |
| bge-base-512 convex 1.0 | +13.8% | -2.2% | 7/1 | +0.0% | -2.2% | +5.6% | +10.8% | +21.7% | +12.7% | +29.3% | +36.3% | +9.8% | 374 | 33.2 |
| arctic-m-512 convex 1.0 | +11.6% | -2.2% | 7/1 | +0.0% | -2.2% | +6.6% | +6.2% | +11.6% | +10.7% | +30.1% | +27.8% | +13.5% | 388 | 33.2 |
| minilm-256 linear 2.0 | +10.9% | -2.2% | 7/2 | -2.1% | -2.2% | +5.9% | +4.9% | +7.0% | +15.4% | +34.3% | +22.0% | +12.7% | 18 | 19.2 |
| bge-small-512 convex 1.0 | +10.1% | -3.3% | 7/2 | -1.1% | -3.3% | +4.8% | +7.2% | +17.2% | +10.1% | +28.8% | +18.3% | +8.8% | 200 | 25.9 |
| minilm-256 dbsf 1.0 | +9.7% | -3.3% | 7/2 | -2.1% | -3.3% | +5.7% | +3.9% | +4.6% | +13.2% | +31.6% | +22.4% | +11.7% | 16 | 22.4 |
| minilm-256 convex 1.0 | +9.1% | -3.3% | 7/2 | -1.1% | -3.3% | +4.6% | +3.3% | +7.1% | +12.4% | +29.2% | +19.0% | +10.9% | 17 | 21.8 |
| e5-base-512 convex 1.0 | +6.8% | -2.1% | 6/2 | -1.1% | +0.0% | +5.7% | +5.8% | +6.1% | +5.0% | +28.8% | -2.1% | +13.2% | 377 | 32.4 |
| minilm-256 rrf | +5.3% | -14.3% | 6/2 | -6.3% | -14.3% | +0.0% | +2.0% | +3.1% | +12.9% | +26.0% | +16.9% | +7.3% | 16 | 21.5 |
| minilm-256 linear 0.4 | +0.6% | -1.5% | 4/1 | +0.0% | +0.0% | +0.7% | +0.3% | +1.8% | -1.5% | +3.1% | -0.5% | +1.6% | 120 | 19.2 |
| baseline | +0.0% | +0.0% | 0/0 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 93 | 18.6 |

Baseline scores: 42:docs 0.950, 42:code 0.919, scifact 0.691, nfcorpus 0.339, arguana 0.503, scidocs 0.173, fiqa 0.281, coir:cosqa 0.220, coir:stackoverflow-qa 0.716.

- Best mean: **bge-large-512 convex 1.0**.
- Robust choice (best mean losing at most 2% on every dataset): **bge-large-512 convex 1.0**.

## Leave one dataset out

Chosen on the other datasets, scored on the one left out.

| Held out | chosen | its change | rank | best there | regret |
|---|---|---|---|---|---|
| 42:docs | bge-large-512 convex 1.0 | +0.0% | 5/11 | baseline (+0.0%) | 0.0 pts |
| 42:code | bge-base-512 convex 1.0 | -2.2% | 6/11 | bge-large-512 convex 1.0 (+1.1%) | 3.3 pts |
| scifact | bge-base-512 convex 1.0 | +5.6% | 6/11 | bge-large-512 convex 1.0 (+8.1%) | 2.5 pts |
| nfcorpus | bge-large-512 convex 1.0 | +11.3% | 1/11 | bge-large-512 convex 1.0 (+11.3%) | 0.0 pts |
| arguana | bge-base-512 convex 1.0 | +21.7% | 2/11 | bge-large-512 convex 1.0 (+23.8%) | 2.1 pts |
| scidocs | bge-large-512 convex 1.0 | +14.2% | 2/11 | minilm-256 linear 2.0 (+15.4%) | 1.2 pts |
| fiqa | bge-base-512 convex 1.0 | +29.3% | 5/11 | bge-large-512 convex 1.0 (+37.2%) | 7.9 pts |
| coir:cosqa | bge-large-512 convex 1.0 | +18.9% | 6/11 | bge-base-512 convex 1.0 (+36.3%) | 17.4 pts |
| coir:stackoverflow-qa | bge-large-512 convex 1.0 | +11.0% | 5/11 | arctic-m-512 convex 1.0 (+13.5%) | 2.5 pts |

Mean regret 4.1 points; the held-out choice improved on the baseline in 7 of 9 datasets.
