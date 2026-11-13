# BEIR reference numbers (published) for comparison

Compiled 2026-11-12. Datasets: SciFact, NFCorpus, FiQA-2018, ArguAna, SCIDOCS, **test** splits.
Every number is tagged `[verified]` (the source was opened and the number read from it on that date)
or `[unverified]` (memory / secondary source). Nothing here is estimated; missing values say "not found".

Sources opened:

- BEIR paper: Thakur et al. 2021, NeurIPS Datasets & Benchmarks, [arXiv 2104.08663](https://arxiv.org/abs/2104.08663) - Tables 1, 2, 9 and Section 4 (read through the ar5iv HTML rendering, https://ar5iv.labs.arxiv.org/html/2104.08663).
- Pyserini two-click reproductions for BEIR: https://castorini.github.io/pyserini/2cr/beir.html (same Lucene indexes/runs as the Anserini BEIR regressions).
- MTEB results repository (per-task JSON, mteb 1.12.75, model revision 8b3219a): https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a
- bge-small-en-v1.5 model card `model-index` metadata: https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md
- BEIR source code: https://github.com/beir-cellar/beir (`beir/retrieval/evaluation.py`, `beir/retrieval/models/util.py`, `beir/reranking/rerank.py`).

## 0. Dataset sizes (BEIR paper, Table 1)

| Dataset | BEIR name / zip | Corpus docs | Test queries | Titles | Relevance | Avg rel. docs/query | Avg words query / doc |
|---|---|---|---|---|---|---|---|
| SciFact | `scifact` | 5,183 [verified] | 300 [verified] | yes | binary | 1.1 | 12.37 / 213.63 |
| NFCorpus | `nfcorpus` | 3,633 [verified] | 323 [verified] | yes | 3-level | 38.2 | 3.30 / 232.26 |
| FiQA-2018 | `fiqa` | 57,638 [verified] | 648 [verified] | no | binary | 2.6 | 10.77 / 132.32 |
| ArguAna | `arguana` | 8,674 [verified] | 1,406 [verified] | yes | binary | 1.0 | 192.98 / 166.80 |
| SCIDOCS | `scidocs` | 25,657 [verified] | 1,000 [verified] | yes | binary | 4.9 | 9.38 / 176.19 |

Source for all cells: [BEIR paper Table 1](https://arxiv.org/abs/2104.08663).

## 1a. BM25 - BEIR paper (Table 2 nDCG@10, Table 9 Recall@100)

Anserini BM25, default Lucene parameters k1=0.9, b=0.4, title and passage indexed as separate fields (paper, Section 4) [verified].

| Dataset | nDCG@10 | Recall@100 |
|---|---|---|
| SciFact | 0.665 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.908 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| NFCorpus | 0.325 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.250 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| FiQA-2018 | 0.236 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.539 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| ArguAna | 0.315 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.942 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| SCIDOCS | 0.158 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.356 [verified] ([src](https://arxiv.org/abs/2104.08663)) |

## 1b. BM25 - Anserini / Pyserini BEIR regressions

`python -m pyserini.search.lucene --index beir-v1.0.0-<name>.flat --topics beir-v1.0.0-<name>-test --hits 1000 --bm25 --remove-query`
(multifield adds `--fields contents=1.0 title=1.0` on the `.multifield` index) [verified]. Flat = title and text concatenated in one field.
The page reports 3 decimals. The per-dataset Anserini regression docs (4 decimals) could not be opened (GitHub paths returned 404): not found.

| Dataset | Flat nDCG@10 | Flat R@100 | Multifield nDCG@10 | Multifield R@100 |
|---|---|---|---|---|
| SciFact | 0.679 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.925 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.665 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.908 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) |
| NFCorpus | 0.322 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.246 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.325 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.250 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) |
| FiQA-2018 | 0.236 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.539 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.236 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.539 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) |
| ArguAna | 0.397 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.932 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.414 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.943 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) |
| SCIDOCS | 0.149 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.348 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.158 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) | 0.356 [verified] ([src](https://castorini.github.io/pyserini/2cr/beir.html)) |

Note: multifield equals the BEIR paper on four datasets; on ArguAna the paper says 0.315 and the regressions say 0.414 (flat 0.397),
while Recall@100 agrees (0.942 vs 0.943). The cause of the paper's lower ArguAna nDCG@10 was not confirmed in a source: not found.

## 2. sentence-transformers/all-MiniLM-L6-v2 (dense only) - MTEB

From the MTEB results repository JSON files (`ArguAna.json`, `FiQA2018.json`, `NFCorpus.json`, `SCIDOCS.json`, `SciFact.json`), `test` split, mteb 1.12.75.
The Hugging Face model card of all-MiniLM-L6-v2 has no `model-index` MTEB metadata (checked: the README contains no ndcg entries) [verified].
SBERT pretrained-models page: not checked (it only gives an aggregate, not per-dataset values).

| Dataset | nDCG@10 | Recall@100 |
|---|---|---|
| SciFact | 0.64508 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) | 0.92500 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) |
| NFCorpus | 0.31594 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) | 0.31151 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) |
| FiQA-2018 | 0.36867 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) | 0.70606 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) |
| ArguAna | 0.50167 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) | 0.97724 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) |
| SCIDOCS | 0.21641 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) | 0.51012 [verified] ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)) |

## 3. BM25 + cross-encoder/ms-marco-MiniLM-L-6-v2 reranking - BEIR paper "BM25+CE"

Reranks the **top-100** hits of Anserini BM25 (multifield); the checkpoint listed in the paper's Table 6 is `cross-encoder/ms-marco-MiniLM-L-6-v2` [verified].
Recall@100 is by construction identical to BM25 (Table 9).

| Dataset | nDCG@10 | Recall@100 |
|---|---|---|
| SciFact | 0.688 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.908 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| NFCorpus | 0.350 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.250 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| FiQA-2018 | 0.347 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.539 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| ArguAna | 0.311 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.942 [verified] ([src](https://arxiv.org/abs/2104.08663)) |
| SCIDOCS | 0.166 [verified] ([src](https://arxiv.org/abs/2104.08663)) | 0.356 [verified] ([src](https://arxiv.org/abs/2104.08663)) |

Note: on ArguAna the reranker is *worse* than BM25 in the paper (0.311 vs 0.315) and far below the Anserini BM25 (0.397-0.414).

## 4a. BAAI/bge-small-en-v1.5 (context, stronger dense model) - model card MTEB metadata

Self-reported `model-index` values, `test` split (reported on a 0-100 scale, divided by 100 here). bge v1.5 results use the query instruction
"Represent this sentence for searching relevant passages: " for retrieval queries [unverified - from memory of the model card text].

| Dataset | nDCG@10 | Recall@100 |
|---|---|---|
| SciFact | 0.71275 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) | 0.94167 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) |
| NFCorpus | 0.34305 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) | 0.31135 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) |
| FiQA-2018 | 0.40343 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) | 0.69552 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) |
| ArguAna | 0.59550 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) | 0.98435 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) |
| SCIDOCS | 0.20522 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) | 0.45905 [verified] ([src](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/README.md)) |

## 4b. Published hybrid (BM25 + dense) - context only

**No hybrid nDCG@10 table for these five datasets was found** in the Pyserini BEIR 2CR page (it has BM25 flat/multifield, SPLADE-v3, BGE-base, no fusion): not found.

Closest credible source, same model as ours: Bruch, Gai, Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval" ([arXiv 2210.11934](https://arxiv.org/abs/2210.11934)), Table 2.
Lexical = BM25 (PISA, k1=0.9, b=0.4), semantic = all-MiniLM-L6-v2 (cosine, FAISS exact), fusion = RRF (eta=60) or TM2C2 convex combination (alpha=0.8).
**The metric is NOT nDCG@10**: it is NDCG@100 for SciFact and NFCorpus and NDCG@1000 for FiQA, so use it only for the *direction and size of the hybrid gain*. ArguAna and SCIDOCS are not in that paper: not found.

| Dataset | Metric | BM25 | all-MiniLM-L6-v2 | Hybrid RRF(60) | Hybrid TM2C2 (alpha=0.8) |
|---|---|---|---|---|---|
| SciFact | NDCG@100 | 0.698 [verified] | 0.681 [verified] | 0.730 [verified] | 0.753 [verified] |
| NFCorpus | NDCG@100 | 0.268 [verified] | 0.296 [verified] | 0.312 [verified] | 0.327 [verified] |
| FiQA | NDCG@1000 | 0.315 [verified] | 0.467 [verified] | 0.464 [verified] | 0.496 [verified] |
| ArguAna | - | not found | not found | not found | not found |
| SCIDOCS | - | not found | not found | not found | not found |

Source for all cells: [Bruch et al., Table 2](https://arxiv.org/abs/2210.11934) (read through https://ar5iv.labs.arxiv.org/html/2210.11934).

Other strong single-model references on the same Pyserini page ([src](https://castorini.github.io/pyserini/2cr/beir.html)), nDCG@10 / R@100, all [verified]:
BGE-base-en-v1.5: SciFact 0.741 / 0.967, NFCorpus 0.373 / 0.337, FiQA 0.406 / 0.742, ArguAna 0.636 / 0.992, SCIDOCS 0.217 / 0.496.
SPLADE-v3: SciFact 0.714 / 0.936, NFCorpus 0.363 / 0.298, FiQA 0.380 / 0.667, ArguAna 0.487 / 0.982, SCIDOCS 0.157 / 0.367.

## 5. Code retrieval: is there a comparable public benchmark?

Yes: **CoIR** (Li et al., ACL 2025, [arXiv 2407.02883](https://arxiv.org/abs/2407.02883)), 10 datasets in BEIR format, main metric nDCG@10. The paper's Table 3 has a BM25 row, and the MTEB results
repository has all-MiniLM-L6-v2 on the same task names ([src](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a)), so both baselines exist. Sizes (CoIR Table 2, test queries / corpus) and nDCG@10 x100 (BM25 from CoIR Table 3 / all-MiniLM-L6-v2 from MTEB JSON, test split), all [verified]:

| CoIR dataset | Test queries | Corpus | BM25 | all-MiniLM-L6-v2 (MTEB task) |
|---|---|---|---|---|
| CosQA | 500 | 21K | 13.96 | 33.02 (`CosQA`) |
| CodeTransOcean-DL | 180 | 816 | 8.69 | 27.94 (`CodeTransOceanDL`) |
| CodeTransOcean-Contest | 446 | 1K | 50.13 | 61.78 (`CodeTransOceanContest`) |
| StackOverflow QA | 2K | 20K | 56.80 | 83.96 (`StackOverflowQA`) |
| APPS | 3.8K | 9K | 0.95 | 6.60 (`AppsRetrieval`) |
| CodeFeedBack-MT | 13K | 66K | 34.73 | 35.62 (`CodeFeedbackMT`) |
| Synthetic Text2SQL | 6K | 106K | 16.92 | 44.36 (`SyntheticText2SQL`) |
| CodeFeedBack-ST | 31K | 156K | 54.32 | 65.99 (`CodeFeedbackST`) |
| CodeSearchNet (code-to-text) | 53K | 1M | 26.75 | not found (`COIRCodeSearchNetRetrieval.json` absent) |
| CodeSearchNet-CCR | 53K | 1M | 34.69 | see caveat below |

Caveats: (a) the MTEB JSON for `CodeSearchNetRetrieval` (python 79.28, javascript 63.62, go 79.84, ruby 72.84, java 51.65, php 70.82 [verified]) and `CodeSearchNetCCRetrieval`
(python 71.29, javascript 66.75, go 60.00, ruby 69.32, java 66.32, php 58.42 [verified]) are MTEB's own task definitions; whether they use the full CoIR 1M corpus or a subsample was not checked,
so they are not guaranteed comparable with the CoIR BM25 column [unverified]. (b) The MiniLM numbers come from the MTEB runner and the BM25 numbers from the CoIR paper: dataset versions should match but that was not verified file by file [unverified].
CPU feasibility: CosQA (500 queries / 21K short functions), the two CodeTransOcean sets and StackOverflow QA are the same order of magnitude as SciFact / SCIDOCS and are feasible on CPU;
the CodeSearchNet variants (1M documents, 53K queries) are not, short of subsampling (which breaks comparability). The original CodeSearchNet and CoSQA papers report MRR / accuracy, not nDCG@10 with BM25 and MiniLM baselines [unverified - not opened].

## 6. Evaluation protocol (what to replicate)

- **Metric**: BEIR `EvaluateRetrieval.evaluate` uses `pytrec_eval.RelevanceEvaluator` with `ndcg_cut`, `map_cut`, `recall`, `P` at k = 1, 3, 5, 10, 100, 1000, averaged over the queries of the **test** qrels [verified] ([evaluation.py](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)). nDCG uses the graded relevance of the qrels (matters for NFCorpus, 3-level). Queries without test qrels are not evaluated.
- **Title concatenation**: dense models encode `(title + " " + text).strip()` (separator = one space) [verified] ([models/util.py](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/util.py), `sentence_bert.py` default `sep=" "`); the cross-encoder reranker also scores `(title + " " + text).strip()` [verified] ([rerank.py](https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py)). MTEB 1.12.75 does the same [verified]. FiQA has no titles.
- **BM25**: paper = two fields (title, text) with equal weight; Anserini "flat" = one concatenated field. k1=0.9, b=0.4 (not the 1.2/0.75 defaults), Lucene analyzer (Porter stemming, English stopwords) [analyzer detail unverified - from memory].
- **ArguAna**: queries are themselves corpus arguments (avg. 193 words). BEIR `evaluate(..., ignore_identical_ids=True)` is the default and removes any retrieved document whose id equals the query id before scoring [verified]; Anserini/Pyserini do the same with `--remove-query` [verified]; MTEB sets `ignore_identical_ids = True` for the ArguAna task (it is False by default for other tasks) [verified]. Not doing this collapses nDCG@10, since the query's own document takes rank 1.
- **Reranking depth**: BM25+CE reranks top-100 [verified]; its Recall@100 therefore equals BM25's.
- **Truncation**: all-MiniLM-L6-v2 truncates at 256 word pieces by default [unverified - from memory of the model card], which matters for ArguAna queries and long documents.
- **Download**: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip` still resolves - HTTP 200 on 2026-11-12 for `scifact` (2.8 MB), `nfcorpus` (2.4 MB), `fiqa` (17.9 MB), `arguana` (3.8 MB), `scidocs` (142.5 MB) [verified]. Each zip contains `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` [layout unverified - not unzipped]. Alternative: Hugging Face `BeIR/<name>` (corpus/queries) - `BeIR/scifact`, `BeIR/nfcorpus`, `BeIR/fiqa`, `BeIR/arguana`, `BeIR/scidocs` all exist (API returned 200) [verified]; qrels live in the companion `BeIR/<name>-qrels` repos [unverified].

## 7. Summary - nDCG@10, test split (all [verified])

| Dataset | BM25 (BEIR paper) | BM25 Anserini flat | BM25 Anserini multifield | all-MiniLM-L6-v2 | BM25+CE (BEIR paper) | bge-small-en-v1.5 |
|---|---|---|---|---|---|---|
| SciFact | 0.665 | 0.679 | 0.665 | 0.645 | 0.688 | 0.713 |
| NFCorpus | 0.325 | 0.322 | 0.325 | 0.316 | 0.350 | 0.343 |
| FiQA-2018 | 0.236 | 0.236 | 0.236 | 0.369 | 0.347 | 0.403 |
| ArguAna | 0.315 | 0.397 | 0.414 | 0.502 | 0.311 | 0.596 |
| SCIDOCS | 0.158 | 0.149 | 0.158 | 0.216 | 0.166 | 0.205 |
