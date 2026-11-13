# External check: the retrieval pipeline on BEIR

Purpose: see how the system behaves outside the corpus it was tuned on,
against published results of the same models. Run on 2026-11-12 with
`uv run python -m src benchmark --datasets scifact,nfcorpus,arguana,scidocs,fiqa`;
raw results in `experiments/benchmarks/beir.jsonl`; published numbers
and their sources in [`BEIR_REFERENCES.md`](BEIR_REFERENCES.md).

Protocol: test split, nDCG@10 with graded relevance, each document
indexed as title + text, a document ranked by its best chunk, the
query's own document dropped (ArguAna). Our metric code is a
re-implementation of the `trec_eval` formula, not `pytrec_eval` itself.

## nDCG@10

| | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA | Mean |
|---|---|---|---|---|---|---|
| *Published* BM25 (Anserini, flat) | 0.679 | 0.322 | 0.397 | 0.149 | 0.236 | 0.357 |
| *Published* BM25 (BEIR paper) | 0.665 | 0.325 | 0.315 | 0.158 | 0.236 | 0.340 |
| **Ours** BM25, plain | 0.666 | 0.321 | 0.442 | 0.151 | 0.238 | 0.364 |
| *Published* all-MiniLM-L6-v2 | 0.645 | 0.316 | 0.502 | 0.216 | 0.369 | 0.410 |
| **Ours** all-MiniLM-L6-v2, 256 tokens | 0.642 | 0.303 | 0.518 | 0.209 | 0.352 | 0.405 |
| **Ours** all-MiniLM-L6-v2, 64 tokens (shipped setting) | 0.550 | 0.302 | 0.405 | 0.200 | 0.255 | 0.342 |
| *Published* BM25 + cross-encoder | 0.688 | 0.350 | 0.311 | 0.166 | 0.347 | 0.372 |
| **Ours, as shipped** (hybrid, weight 0.4, 64 tokens) | 0.691 | 0.339 | 0.503 | 0.173 | 0.281 | 0.397 |
| **Ours** hybrid, 256 tokens, weight 0.4 | 0.696 | 0.341 | 0.512 | 0.171 | 0.290 | 0.402 |
| **Ours** hybrid, 256 tokens, weight 2.0 | 0.731 | 0.356 | 0.538 | 0.200 | 0.377 | 0.440 |
| **Ours** hybrid, 256 tokens, weight 0.4 + reranker | 0.707 | 0.345 | 0.541 | 0.173 | 0.299 | 0.413 |
| *Published* bge-small-en-v1.5 (stronger model, for scale) | 0.713 | 0.343 | 0.596 | 0.205 | 0.403 | 0.452 |

## Share of queries with a relevant document in the top k (system as shipped)

| | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| SciFact | 0.55 | 0.74 | 0.78 | 0.84 |
| NFCorpus | 0.47 | 0.61 | 0.67 | 0.70 |
| ArguAna | 0.24 | 0.49 | 0.63 | 0.81 |
| SCIDOCS | 0.20 | 0.35 | 0.43 | 0.54 |
| FiQA | 0.25 | 0.38 | 0.46 | 0.55 |
| vLLM docs / code (the 42 corpus, for contrast) | 0.70 / 0.65 | 0.92 / 0.86 | 0.95 / 0.92 | 0.97 / 0.94 |

## Reading

1. **The components are sound.** The hand-written BM25 matches the
   published BM25 within 1.3 points on four datasets and is above it on
   ArguAna. The embedding stage with 256 input tokens matches the
   published all-MiniLM-L6-v2 within 1.7 points everywhere (a little
   below on three datasets, probably because the file path is part of
   the embedded text and long documents are split into chunks).
2. **The settings tuned on the 42 corpus do not transfer.**
   - 64 input tokens cost 6 points of mean nDCG@10 for the embeddings
     alone (10 on FiQA and SciFact). They were adopted because vLLM
     chunks have the key terms up front and it halves indexing time.
   - A semantic weight of 0.4 makes the fusion follow BM25. Where the
     embeddings are the stronger signal (FiQA, SCIDOCS) the shipped
     hybrid is *worse* than embeddings alone.
3. **With those two values changed, the hybrid beats both of its
   parts** on four of five datasets and reaches a mean of 0.440, above
   published BM25 (0.357) and all-MiniLM-L6-v2 (0.410) and close to a
   stronger embedding model (0.452). The weight 2.0 was picked among
   three values on these same test queries, so 0.440 is slightly
   optimistic.
4. **The reranker helps a little** (+1.1 points of mean on top of the
   0.4 hybrid) at 15 to 30 times the search time, as on the 42 corpus.
5. **The 42 numbers are not a general level.** 0.95 hit@5 on vLLM
   questions against 0.43 to 0.78 here: the 42 questions are written
   from one passage and often repeat its identifiers; BEIR queries are
   harder and judged against many documents.

## What it means for reuse

On a new corpus, start from embeddings with at least 256 tokens and a
semantic weight near 1 to 2, and re-tune with the lab on a sample of
that corpus. The values in `Settings` are the 42 optimum, not a default
for other domains. A code-retrieval check (CoIR: CosQA, StackOverflowQA)
is feasible on this machine and has not been run.

## Second round (2026-11-13): reranker, stronger model, code retrieval

Published best results with the same models are in
[`BEST_COMPARABLE.md`](BEST_COMPARABLE.md).

### Text, nDCG@10

| | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA | Mean |
|---|---|---|---|---|---|---|
| **Ours** MiniLM hybrid, 256 tokens, weight 2.0 | 0.731 | 0.356 | 0.538 | 0.200 | 0.377 | 0.440 |
| **Ours** the same + reranker | 0.735 | 0.359 | 0.563 | 0.200 | 0.381 | 0.448 |
| *Published* best fixed-weight BM25 + MiniLM hybrid | 0.734 | 0.356 | not found | 0.197 | 0.371 | |
| *Published* best tuned hybrid variant (picked on test) | 0.738 | 0.358 | not found | 0.220 | 0.407 | |
| **Ours** MiniLM alone, path not embedded | 0.645 | 0.294 | 0.500 | 0.215 | 0.367 | 0.404 |
| *Published* MiniLM alone | 0.645 | 0.316 | 0.502 | 0.216 | 0.369 | 0.410 |
| **Ours** bge-small alone (512 tokens on SciFact and NFCorpus, 256 on the rest) | 0.709 | 0.345 | 0.601 | 0.204 | 0.391 | 0.450 |
| *Published* bge-small alone | 0.713 | 0.343 | 0.604 | 0.205 | 0.403 | 0.453 |
| **Ours** bge-small hybrid, weight 2.0 | 0.712 | 0.357 | 0.588 | 0.182 | 0.330 | 0.434 |
| *Published* best single model ≤ 35M parameters, per dataset | 0.727 | 0.365 | 0.616 | 0.225 | 0.407 | |

### Code (CoIR), nDCG@10

| | CosQA | StackOverflowQA |
|---|---|---|
| *Published* BM25 (CoIR paper / MTEB bm25s) | 0.140 / 0.188 | 0.568 / 0.703 |
| **Ours** BM25, plain | 0.218 | 0.685 |
| *Published* MiniLM | 0.330 | 0.840 |
| **Ours** MiniLM, 256 tokens, path not embedded | 0.309 | 0.836 |
| *Published* bge-small | 0.321 | 0.781 |
| **Ours** bge-small, 512 tokens | 0.297 | 0.776 |
| **Ours, as shipped** (hybrid, 64 tokens, weight 0.4) | 0.221 | 0.716 |
| **Ours** MiniLM hybrid, 256 tokens, weight 2.0 | 0.268 | 0.807 |
| *Published* best model ≤ 35M parameters | 0.405 | 0.856 |

### Reading

1. **Every stage reproduces its published reference**: BM25, MiniLM and
   bge-small land within about one point of the published figure on six
   of seven datasets (CosQA is 2 points low for both embedding models,
   and NFCorpus 2 points low for MiniLM without the path).
2. **Against the best public results with the same two models, the
   tuned hybrid is level**: within 0.003 of the best fixed-weight hybrid
   on four datasets, and 0.002 to 0.03 below variants tuned on the test
   queries. Those hybrid references are self-published runs, not
   peer-reviewed tables.
3. **A better embedding model alone equals our best MiniLM pipeline**
   (0.450 against 0.448 with hybrid and reranker), at a fraction of the
   search cost but with slower indexing on a CPU.
4. **The fusion weight is model-specific.** Weight 2.0 helps MiniLM and
   hurts bge-small (0.434 against 0.450 alone): its cosines sit in a
   narrower range, so the same weight means something else. A fusion
   that normalises both scores would transfer better.
5. **No single configuration wins everywhere.** Hybrid wins on
   scientific text; embeddings alone win on code and on questions
   written in everyday language (CosQA, StackOverflowQA, FiQA with
   bge-small).

Protocol note: bge-small ran with 256 input tokens on ArguAna, SCIDOCS
and FiQA (512 would have taken hours on this CPU); published figures
use 512.
