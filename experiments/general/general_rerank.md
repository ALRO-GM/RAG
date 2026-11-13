# Generalist study: `general_rerank`

Relative change against the suite baseline on each dataset (42 datasets: recall@5; public ones: nDCG@10). Configurations sorted by mean relative change; only those measured on every dataset are ranked.

| Configuration | mean | worst | W/L | 42:docs | 42:code | scifact | nfcorpus | arguana | scidocs | fiqa | coir:cosqa | coir:stackoverflow-qa | index s | search s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| general (bge-large adaptive), no reranker **(robust)** **(best mean)** | +14.0% | +0.0% | 7/0 | +0.0% | +0.0% | +7.0% | +9.2% | +36.5% | +15.0% | +37.0% | +8.3% | +13.3% | 996 | 58.7 |
| general + bge-reranker-v2-m3 depth 30 weight 0.5 | +13.4% | -6.6% | 7/1 | +0.0% | -6.6% | +8.1% | +9.1% | +33.4% | +10.8% | +40.9% | +13.5% | +11.8% | 18 | 893.6 |
| general + MiniLM-L6 depth 30 weight 0.5 | +11.0% | -3.3% | 7/2 | -2.1% | -3.3% | +5.8% | +9.5% | +28.6% | +9.3% | +34.4% | +7.7% | +9.0% | 186 | 250.1 |
| general + bge-reranker-base depth 30 weight 0.5 | +9.8% | -5.5% | 7/2 | -3.2% | -5.5% | +5.2% | +6.3% | +22.8% | +8.2% | +33.2% | +12.9% | +8.0% | 18 | 360.1 |
| shipped + MiniLM-L6 depth 30 weight 0.5 | +1.1% | -2.1% | 4/3 | -2.1% | -1.1% | +0.1% | +0.6% | +4.8% | +0.1% | +7.3% | +1.2% | -0.8% | 19 | 219.4 |
| baseline | +0.0% | +0.0% | 0/0 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | 102 | 19.6 |
| general + bge-reranker-v2-m3 depth 30 weight 1.0 | -0.8% | -38.5% | 5/3 | -11.6% | -38.5% | +3.4% | +1.5% | +14.5% | -0.2% | +31.3% | +12.5% | -20.0% | 22 | 893.8 |
| general + MiniLM-L6 depth 30 weight 1.0 | -7.0% | -29.7% | 3/6 | -11.6% | -29.7% | -1.3% | +2.8% | +1.0% | -6.3% | +15.9% | -6.3% | -27.9% | 18 | 252.3 |
| general + bge-reranker-base depth 30 weight 1.0 | -9.2% | -30.9% | 2/7 | -9.5% | -30.8% | -2.4% | -5.4% | -12.7% | -10.5% | +14.6% | +4.6% | -30.9% | 19 | 359.4 |

Baseline scores: 42:docs 0.950, 42:code 0.919, scifact 0.710, nfcorpus 0.353, arguana 0.476, scidocs 0.194, fiqa 0.328, coir:cosqa 0.263, coir:stackoverflow-qa 0.737.

- Best mean: **general (bge-large adaptive), no reranker**.
- Robust choice (best mean losing at most 2% on every dataset): **general (bge-large adaptive), no reranker**.

## Leave one dataset out

Chosen on the other datasets, scored on the one left out.

| Held out | chosen | its change | rank | best there | regret |
|---|---|---|---|---|---|
| 42:docs | general (bge-large adaptive), no reranker | +0.0% | 2/9 | baseline (+0.0%) | 0.0 pts |
| 42:code | general + bge-reranker-v2-m3 depth 30 weight 0.5 | -6.6% | 6/9 | baseline (+0.0%) | 6.6 pts |
| scifact | general (bge-large adaptive), no reranker | +7.0% | 2/9 | general + bge-reranker-v2-m3 depth 30 weight 0.5 (+8.1%) | 1.1 pts |
| nfcorpus | general (bge-large adaptive), no reranker | +9.2% | 2/9 | general + MiniLM-L6 depth 30 weight 0.5 (+9.5%) | 0.3 pts |
| arguana | general (bge-large adaptive), no reranker | +36.5% | 1/9 | general (bge-large adaptive), no reranker (+36.5%) | 0.0 pts |
| scidocs | general (bge-large adaptive), no reranker | +15.0% | 1/9 | general (bge-large adaptive), no reranker (+15.0%) | 0.0 pts |
| fiqa | general (bge-large adaptive), no reranker | +37.0% | 2/9 | general + bge-reranker-v2-m3 depth 30 weight 0.5 (+40.9%) | 3.9 pts |
| coir:cosqa | general (bge-large adaptive), no reranker | +8.3% | 4/9 | general + bge-reranker-v2-m3 depth 30 weight 0.5 (+13.5%) | 5.2 pts |
| coir:stackoverflow-qa | general (bge-large adaptive), no reranker | +13.3% | 1/9 | general (bge-large adaptive), no reranker (+13.3%) | 0.0 pts |

Mean regret 1.9 points; the held-out choice improved on the baseline in 7 of 9 datasets.
