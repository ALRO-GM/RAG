# Generalist study: `general_adaptive`

Relative change against the suite baseline on each dataset (42 datasets: recall@5; public ones: nDCG@10). Configurations sorted by mean relative change; only those measured on every dataset are ranked.

| Configuration | mean | worst | W/L | 42:docs | 42:code | scifact | nfcorpus | arguana | scidocs | fiqa | coir:cosqa | coir:stackoverflow-qa | index s | search s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-large convex 4.0 adaptive->1.0 sat 0.2 **(robust)** **(best mean)** | +20.9% | +0.0% | 7/0 | +0.0% | +0.0% | +10.0% | +13.7% | +29.2% | +29.1% | +59.9% | +29.5% | +16.6% | 19 | 59.4 |
| bge-large convex 4.0 adaptive->0.5 sat 0.2 | +20.8% | +0.0% | 7/0 | +0.0% | +0.0% | +9.9% | +13.7% | +29.2% | +28.8% | +59.4% | +29.5% | +16.4% | 19 | 58.8 |
| bge-large convex 4.0 adaptive->1.0 sat 0.1 | +20.6% | +0.0% | 7/0 | +0.0% | +0.0% | +9.4% | +13.7% | +29.2% | +28.4% | +59.2% | +29.5% | +15.6% | 20 | 58.5 |
| bge-large convex 4.0 adaptive->0.5 sat 0.1 | +20.2% | +0.0% | 7/0 | +0.0% | +0.0% | +9.3% | +13.7% | +29.2% | +27.5% | +58.3% | +29.5% | +14.4% | 22 | 58.7 |
| bge-base convex 4.0 adaptive->1.0 sat 0.1 | +20.2% | -2.2% | 7/1 | +0.0% | -2.2% | +7.8% | +12.7% | +26.5% | +25.3% | +45.7% | +52.7% | +13.3% | 23 | 35.3 |
| bge-base convex 4.0 adaptive->0.5 sat 0.2 | +20.1% | -3.3% | 7/2 | -1.1% | -3.3% | +7.9% | +12.7% | +26.5% | +25.9% | +46.0% | +52.7% | +13.8% | 19 | 35.1 |
| bge-base convex 4.0 adaptive->0.5 sat 0.1 | +20.0% | -2.2% | 8/1 | +1.1% | -2.2% | +7.8% | +12.7% | +26.5% | +24.7% | +45.0% | +52.7% | +12.2% | 19 | 37.8 |
| bge-base convex 4.0 adaptive->1.0 sat 0.2 | +19.8% | -5.5% | 7/2 | -2.1% | -5.5% | +8.0% | +12.7% | +26.5% | +26.2% | +46.2% | +52.7% | +14.0% | 20 | 35.8 |
| bge-large convex 4.0 fixed | +19.8% | -9.9% | 7/2 | -1.1% | -9.9% | +9.9% | +13.7% | +29.2% | +29.2% | +60.4% | +29.5% | +17.0% | 18 | 59.1 |
| bge-large dbsf 2.0 adaptive->1.0 sat 0.2 | +19.6% | -1.1% | 7/1 | +0.0% | -1.1% | +10.7% | +14.3% | +25.7% | +24.6% | +56.5% | +30.6% | +15.6% | 18 | 59.7 |
| bge-large dbsf 2.0 fixed | +19.6% | -3.3% | 7/1 | +0.0% | -3.3% | +11.0% | +14.3% | +25.7% | +25.0% | +57.0% | +30.6% | +16.2% | 29 | 63.5 |
| bge-large dbsf 2.0 adaptive->1.0 sat 0.1 | +19.6% | +0.0% | 7/0 | +0.0% | +0.0% | +10.4% | +14.3% | +25.7% | +24.6% | +56.1% | +30.6% | +14.8% | 19 | 59.0 |
| bge-large dbsf 2.0 adaptive->0.5 sat 0.2 | +19.6% | +0.0% | 7/0 | +0.0% | +0.0% | +10.5% | +14.3% | +25.7% | +24.4% | +56.0% | +30.6% | +15.0% | 18 | 59.1 |
| bge-large dbsf 2.0 adaptive->0.5 sat 0.1 | +19.2% | +0.0% | 7/0 | +0.0% | +0.0% | +10.6% | +14.3% | +25.6% | +23.6% | +55.1% | +30.6% | +13.5% | 18 | 59.2 |
| bge-base convex 4.0 fixed | +19.1% | -14.3% | 7/2 | -2.1% | -14.3% | +8.1% | +12.7% | +26.5% | +26.6% | +46.5% | +52.7% | +14.8% | 20 | 33.1 |
| bge-base dbsf 2.0 adaptive->1.0 sat 0.1 | +18.8% | -3.3% | 7/1 | +0.0% | -3.3% | +8.7% | +12.8% | +23.7% | +21.4% | +43.6% | +49.4% | +12.9% | 19 | 33.7 |
| bge-base dbsf 2.0 adaptive->0.5 sat 0.2 | +18.7% | -3.3% | 7/2 | -1.1% | -3.3% | +8.7% | +12.8% | +23.8% | +21.3% | +43.5% | +49.4% | +12.8% | 18 | 34.4 |
| bge-base dbsf 2.0 adaptive->1.0 sat 0.2 | +18.6% | -4.4% | 7/2 | -1.1% | -4.4% | +8.7% | +12.8% | +23.8% | +21.4% | +43.8% | +49.4% | +13.1% | 19 | 33.4 |
| bge-base dbsf 2.0 fixed | +18.5% | -7.7% | 7/2 | -1.1% | -7.7% | +8.9% | +12.8% | +23.9% | +22.4% | +44.3% | +49.4% | +13.9% | 29 | 33.6 |
| bge-base dbsf 2.0 adaptive->0.5 sat 0.1 | +18.5% | -3.3% | 7/1 | +0.0% | -3.3% | +8.5% | +12.8% | +23.7% | +20.9% | +43.0% | +49.4% | +11.7% | 18 | 33.5 |
| bge-large dbsf 1.0 fixed | +14.9% | +0.0% | 7/0 | +0.0% | +0.0% | +8.7% | +12.3% | +19.7% | +16.4% | +43.2% | +21.4% | +12.2% | 19 | 59.2 |
| baseline | +0.0% | +0.0% | 0/0 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 37 | 18.5 |

Baseline scores: 42:docs 0.950, 42:code 0.919, scifact 0.691, nfcorpus 0.339, arguana 0.503, scidocs 0.173, fiqa 0.281, coir:cosqa 0.220, coir:stackoverflow-qa 0.716.

- Best mean: **bge-large convex 4.0 adaptive->1.0 sat 0.2**.
- Robust choice (best mean losing at most 2% on every dataset): **bge-large convex 4.0 adaptive->1.0 sat 0.2**.

## Leave one dataset out

Chosen on the other datasets, scored on the one left out.

| Held out | chosen | its change | rank | best there | regret |
|---|---|---|---|---|---|
| 42:docs | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +0.0% | 10/22 | bge-base convex 4.0 adaptive->0.5 sat 0.1 (+1.1%) | 1.1 pts |
| 42:code | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +0.0% | 7/22 | baseline (+0.0%) | 0.0 pts |
| scifact | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +10.0% | 6/22 | bge-large dbsf 2.0 fixed (+11.0%) | 1.0 pts |
| nfcorpus | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +13.7% | 9/22 | bge-large dbsf 2.0 fixed (+14.3%) | 0.6 pts |
| arguana | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +29.2% | 3/22 | bge-large convex 4.0 fixed (+29.2%) | 0.0 pts |
| scidocs | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +29.1% | 2/22 | bge-large convex 4.0 fixed (+29.2%) | 0.2 pts |
| fiqa | bge-base convex 4.0 adaptive->1.0 sat 0.1 | +45.7% | 14/22 | bge-large convex 4.0 fixed (+60.4%) | 14.7 pts |
| coir:cosqa | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +29.5% | 19/22 | bge-base convex 4.0 fixed (+52.7%) | 23.1 pts |
| coir:stackoverflow-qa | bge-large convex 4.0 adaptive->1.0 sat 0.2 | +16.6% | 2/22 | bge-large convex 4.0 fixed (+17.0%) | 0.3 pts |

Mean regret 4.5 points; the held-out choice improved on the baseline in 7 of 9 datasets.
