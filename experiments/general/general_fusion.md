# Generalist study: `general_fusion`

Relative change against the suite baseline on each dataset (42 datasets: recall@5; public ones: nDCG@10). Configurations sorted by mean relative change; only those measured on every dataset are ranked.

| Configuration | mean | worst | W/L | 42:docs | 42:code | scifact | nfcorpus | arguana | scidocs | fiqa | coir:cosqa | coir:stackoverflow-qa | index s | search s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-large convex 4.0 **(best mean)** | +19.8% | -9.9% | 7/2 | -1.1% | -9.9% | +9.9% | +13.7% | +29.2% | +29.2% | +60.4% | +29.5% | +17.0% | 19 | 58.7 |
| bge-large dbsf 2.0 | +19.6% | -3.3% | 7/1 | +0.0% | -3.3% | +11.0% | +14.3% | +25.7% | +25.0% | +57.0% | +30.6% | +16.2% | 23 | 59.2 |
| bge-base convex 4.0 | +19.1% | -14.3% | 7/2 | -2.1% | -14.3% | +8.1% | +12.7% | +26.5% | +26.6% | +46.5% | +52.7% | +14.8% | 18 | 33.9 |
| bge-large convex 2.0 | +18.7% | -2.2% | 7/2 | -1.1% | -2.2% | +10.2% | +13.4% | +28.0% | +23.2% | +53.9% | +27.4% | +15.6% | 20 | 58.4 |
| bge-base dbsf 2.0 | +18.5% | -7.7% | 7/2 | -1.1% | -7.7% | +8.9% | +12.8% | +23.9% | +22.4% | +44.3% | +49.4% | +13.9% | 18 | 34.5 |
| bge-base convex 2.0 | +18.1% | -6.6% | 7/2 | -1.1% | -6.6% | +8.6% | +12.4% | +25.8% | +20.2% | +42.0% | +48.7% | +13.1% | 18 | 34.2 |
| bge-large embeddings only | +17.0% | -14.3% | 7/2 | -5.3% | -14.3% | +8.4% | +12.3% | +30.0% | +26.7% | +57.8% | +23.9% | +13.2% | 22 | 54.3 |
| bge-base embeddings only | +16.1% | -20.9% | 7/2 | -8.4% | -20.9% | +5.7% | +10.2% | +26.4% | +23.9% | +40.4% | +56.6% | +10.6% | 18 | 29.5 |
| bge-large dbsf 1.0 **(robust)** | +14.9% | +0.0% | 7/0 | +0.0% | +0.0% | +8.7% | +12.3% | +19.7% | +16.4% | +43.2% | +21.4% | +12.2% | 20 | 58.6 |
| bge-base dbsf 1.0 | +14.6% | -3.3% | 7/1 | +0.0% | -3.3% | +6.5% | +11.1% | +18.2% | +14.8% | +33.8% | +39.4% | +10.6% | 18 | 34.2 |
| bge-large convex 1.0 | +14.0% | +0.0% | 8/0 | +0.0% | +1.1% | +8.1% | +11.3% | +23.8% | +14.2% | +37.2% | +18.9% | +11.0% | 186 | 60.0 |
| bge-base convex 1.0 | +13.8% | -2.2% | 7/1 | +0.0% | -2.2% | +5.6% | +10.8% | +21.7% | +12.7% | +29.3% | +36.3% | +9.8% | 64 | 34.2 |
| bge-base convex 0.5 | +7.8% | -1.1% | 8/1 | +1.1% | -1.1% | +3.4% | +7.8% | +14.7% | +4.6% | +14.2% | +20.5% | +4.7% | 28 | 34.2 |
| bge-large convex 0.5 | +6.9% | -1.1% | 7/1 | +0.0% | -1.1% | +4.6% | +8.5% | +16.0% | +4.6% | +18.3% | +5.5% | +5.3% | 29 | 59.6 |
| baseline | +0.0% | +0.0% | 0/0 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 37 | 19.8 |
| bm25 only | -6.4% | -15.6% | 0/8 | -2.1% | +0.0% | -3.5% | -5.8% | -12.1% | -11.1% | -15.6% | -2.3% | -4.9% | 21 | 8.2 |

Baseline scores: 42:docs 0.950, 42:code 0.919, scifact 0.691, nfcorpus 0.339, arguana 0.503, scidocs 0.173, fiqa 0.281, coir:cosqa 0.220, coir:stackoverflow-qa 0.716.

- Best mean: **bge-large convex 4.0**.
- Robust choice (best mean losing at most 2% on every dataset): **bge-large dbsf 1.0**.

## Leave one dataset out

Chosen on the other datasets, scored on the one left out.

| Held out | chosen | its change | rank | best there | regret |
|---|---|---|---|---|---|
| 42:docs | bge-large convex 4.0 | -1.1% | 12/16 | bge-base convex 0.5 (+1.1%) | 2.1 pts |
| 42:code | bge-large convex 4.0 | -9.9% | 13/16 | bge-large convex 1.0 (+1.1%) | 11.0 pts |
| scifact | bge-large convex 4.0 | +9.9% | 3/16 | bge-large dbsf 2.0 (+11.0%) | 1.1 pts |
| nfcorpus | bge-large convex 4.0 | +13.7% | 2/16 | bge-large dbsf 2.0 (+14.3%) | 0.6 pts |
| arguana | bge-large dbsf 2.0 | +25.7% | 7/16 | bge-large embeddings only (+30.0%) | 4.3 pts |
| scidocs | bge-large dbsf 2.0 | +25.0% | 4/16 | bge-large convex 4.0 (+29.2%) | 4.3 pts |
| fiqa | bge-base convex 4.0 | +46.5% | 5/16 | bge-large convex 4.0 (+60.4%) | 13.9 pts |
| coir:cosqa | bge-large convex 4.0 | +29.5% | 8/16 | bge-base embeddings only (+56.6%) | 27.1 pts |
| coir:stackoverflow-qa | bge-large convex 4.0 | +17.0% | 1/16 | bge-large convex 4.0 (+17.0%) | 0.0 pts |

Mean regret 7.1 points; the held-out choice improved on the baseline in 7 of 9 datasets.
