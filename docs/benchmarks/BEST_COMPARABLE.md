# Best comparable public results (BEIR x5 + CoIR x2)

Compiled 2026-11-12. Complements `BEIR_REFERENCES.md` (not repeated here). Metric is nDCG@10 on the **test** split unless stated.
Tags: `[verified]` = the source was opened on that date and the number read from it; `[unverified]` = not read from a source.
"not found" = searched for and no comparable published number located.

Our system (hybrid BM25 + all-MiniLM-L6-v2, 256 tokens): SciFact 0.731, NFCorpus 0.356, ArguAna 0.538, SCIDOCS 0.200, FiQA 0.377 (mean 0.440).

## Sources

| Key | Source | Kind |
|---|---|---|
| S1 | MTEB results repository, https://github.com/embeddings-benchmark/results , commit `f8fa68f43a0d3868334c764cec9d21ce39c7ecbc` (2026-11-12). Files `results/<org>__<model>/<revision>/<Task>.json`, field `scores.test[].ndcg_at_10`. All 410 model/revision folders having any of the 7 tasks were scanned locally. This repository is the data behind the MTEB leaderboard. `external` revision = numbers imported from the model card (self-reported). | Primary data |
| S2 | BEIR paper, https://arxiv.org/abs/2104.08663 (read via https://ar5iv.labs.arxiv.org/html/2104.08663) | Peer-reviewed |
| S3 | `bayesian-bm25` package page, https://pypi.org/project/bayesian-bm25/ | Self-published, not peer-reviewed |
| S4 | Lucene fork PR "Add BEIR benchmark for Bayesian BM25 hybrid search", https://github.com/jaepil/lucene/pull/1 | Self-published |
| S5 | https://github.com/NikosMav/scifact-hybrid-retrieval (README) | Hobby repo |
| S6 | https://github.com/DiyarSari/rag-retrieval-eval (README) | Hobby repo |
| S7 | https://github.com/Zelros02/hybrid-rag-benchmark (README) | Hobby repo |
| S8 | https://github.com/Ebrahim-KJ/doc-intel (README) | Hobby repo |
| S9 | https://github.com/EllE961/sift-rag (README) | Hobby repo |
| S10 | https://github.com/mishraashmit0025-lgtm/Hybrid-RAG-Eval (README) | Hobby repo |
| S11 | vstash preprint, https://arxiv.org/abs/2604.15484 (v1 HTML) | Preprint |
| S12 | answerai-colbert-small-v1 model card, https://huggingface.co/answerdotai/answerai-colbert-small-v1 | Self-reported |
| S13 | CoIR paper, https://arxiv.org/abs/2407.02883 (via ar5iv), Tables 2 and 3 | Peer-reviewed (ACL 2025) |
| S14 | Hugging Face dataset API + datasets-server for `CoIR-Retrieval/*` and `mteb/*` repos; CoIR loader https://github.com/CoIR-team/coir/blob/main/coir/data_loader.py ; MTEB task files https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/code/cos_qa_retrieval.py and `.../stack_overflow_qa_retrieval.py` | Primary |

**Important:** no peer-reviewed paper and no vendor (Elastic, Weaviate, Qdrant, Vespa, OpenSearch, Pinecone, Milvus) page was found that reports per-dataset
nDCG@10 for hybrid BM25 + all-MiniLM-L6-v2 on these datasets: not found. Bruch et al. (arXiv 2210.11934) uses the same two models but reports NDCG@100/@1000 only (see `BEIR_REFERENCES.md` section 4b).
The Vespa blog https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa-part-two/ is hybrid BM25 + ColBERT, not MiniLM [unverified - seen in search snippet only].
Pyserini 2CR has no fusion rows for these datasets (already established). Everything in section 1 is therefore **self-published, single-run, non-reviewed**.

## 1. Hybrid BM25 + all-MiniLM-L6-v2 (same two models as ours)

All numbers [verified] in the cited source. Dense-only rows are given because they show whether the source's protocol matches MTEB (0.645 / 0.316 / 0.502 / 0.216 / 0.369).

| Source | Fusion | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA | Notes |
|---|---|---|---|---|---|---|---|
| S3 | BM25 only (bm25s, Lucene variant, k1=1.2, b=0.75, Snowball) | 0.6791 | 0.3185 | 0.3616 | 0.1565 | 0.2532 | |
| S3 | Dense only | 0.6451 | 0.3159 | 0.3698 (*) | 0.2164 | 0.3687 | (*) see ArguAna caveat |
| S3 | RRF k=60 | 0.7143 | 0.3443 | 0.3961 (*) | 0.2009 | 0.3685 | zero-shot |
| S3 | Convex, w=0.5 on min-max scores | 0.7338 | 0.3561 | 0.4003 (*) | 0.1965 | 0.3710 | zero-shot |
| S3 | Best of ~20 "Bayesian" fusion variants, per dataset | 0.7381 | 0.3576 | 0.3963 (*) | 0.2197 | 0.4066 | max over variants picked on the test set = optimistic; variants: LogOdds-Local / Balanced-Elbow / LogOdds-Local / Attn-Norm-CV / Balanced-Mix |
| S4 | BM25 only (Lucene; k1, b not stated) | 0.6826 | 0.3236 | not found | not found | 0.2536 | |
| S4 | RRF | 0.7198 | 0.3554 | not found | not found | 0.3659 | KNN = all-MiniLM-L6-v2 |
| S4 | Best variant per dataset | 0.7250 | 0.3602 | not found | not found | 0.3876 | LogOddsW / Attention / Attention; whether "Attention" uses labels is not stated |
| S5 | RRF k=60 (bm25s k1=0.9, b=0.4) | 0.7194 | not found | not found | not found | not found | BM25 0.6762, dense 0.6451 |
| S6 | RRF | 0.694 | not found | not found | not found | not found | BM25 0.664, dense 0.645 |
| S7 | Weaviate hybrid alpha=0.5, 512-token chunks, best chunk per doc | 0.7082 | 0.3442 | not found | not found | 0.3650 | BM25 0.6582 / 0.3097 / 0.2177; dense 0.6470 / 0.3146 / 0.3697 |
| S8 | RRF | 0.681 | 0.342 | not found | not found | 0.344 | BM25 0.654 / 0.305 / 0.221 |

**Best published same-model hybrid per dataset vs ours**

| | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA |
|---|---|---|---|---|---|
| Best zero-shot simple fusion (RRF or fixed convex) | 0.7338 (S3 convex) | 0.3561 (S3 convex) | not found (comparable) | 0.2009 (S3 RRF) | 0.3710 (S3 convex) |
| Best any fusion variant | 0.7381 (S3) | 0.3602 (S4) | not found (comparable) | 0.2197 (S3) | 0.4066 (S3) |
| Ours | 0.731 | 0.356 | 0.538 | 0.200 | 0.377 |
| Ours minus best simple fusion | -0.003 | 0.000 | n/a | -0.001 | +0.006 |
| Ours minus best any variant | -0.007 | -0.004 | n/a | -0.020 | -0.030 |

ArguAna caveat: S3 reports dense-only all-MiniLM-L6-v2 = 0.3698 on ArguAna, while MTEB (S1) reports 0.5017 for the same model, and S3's BM25 0.3616 is below both Anserini (0.397) and MTEB's bm25s baseline (0.4928).
S3's ArguAna protocol therefore differs from the standard one (the cause is not stated on the page; the usual suspect is the query's own document not being removed) and its ArguAna rows are **not comparable**.
For ArguAna the only comparable same-model references are single-model: all-MiniLM-L6-v2 0.5017 and bm25s 0.4928 (both S1) - ours (0.538) is above both.

## 2. Reranking with cross-encoder/ms-marco-MiniLM-L-6-v2

**Which cross-encoder is the BEIR paper's "BM25+CE"?** `cross-encoder/ms-marco-MiniLM-L-6-v2` [verified]. Evidence in S2: Section 4 says a "6-layer, 384-h MiniLM cross-encoder model offers the best
performance on MS MARCO", trained by knowledge distillation from an ensemble of three teachers, and "BM25 + CE reranks the top-100 retrieved hits from a first-stage BM25 (Anserini) model";
the checkpoint table (Table 6) lists "MiniLM-L6 (CE)" with the link `https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2`. "ELECTRA" does not appear in the paper text that was read.
So `BEIR_REFERENCES.md` section 3 attributes it correctly. BM25+CE numbers (already in that file): SciFact 0.688, NFCorpus 0.350, ArguAna 0.311, SCIDOCS 0.166, FiQA 0.347.

Other published results with the same reranker (all [verified] in the cited README; hobby repos):

| Source | First stage -> rerank depth | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA |
|---|---|---|---|---|---|---|
| S5 | hybrid RRF (BM25 + MiniLM-L6) -> CE top-50 | 0.6903 (hybrid alone 0.7194) | not found | not found | not found | not found |
| S6 | hybrid RRF (BM25 + MiniLM-L6) -> CE | 0.689 (hybrid alone 0.694) | not found | not found | not found | not found |
| S9 | BM25 -> CE | 0.688 | 0.350 | not found | not found | not found |
| S9 | hybrid RRF (BM25 + **bge-small**) -> CE top-50 | 0.688 (hybrid alone 0.722) | 0.352 (hybrid alone 0.362) | not found | not found | not found |
| S9 | same, CE score blended with first-stage score, weight tuned on train | 0.729 | 0.367 | not found | not found | not found |
| S10 | hybrid RRF (BM25 + **bge-small**, 200-word chunks) -> CE top-30 | 0.705 (hybrid alone 0.714) | 0.359 (hybrid alone 0.355) | not found | not found | not found |

- all-MiniLM-L6-v2 (dense only) -> ms-marco-MiniLM-L-6-v2: not found on any of the five datasets.
- Any first stage + this cross-encoder on ArguAna, SCIDOCS, FiQA other than the BEIR paper's BM25+CE: not found.
- Consistent finding across S5, S6, S9, S10 and S11 (which reports -0.3% to -3.1% nDCG from cross-encoders, no per-dataset table read): the pure MS MARCO MiniLM reranker *lowers* SciFact nDCG@10 on top of a hybrid first stage and is roughly neutral on NFCorpus.

## 3. Ceiling of the same size class (<= ~35M parameters, single model, no fusion)

From S1 (parameter counts from each folder's `model_meta.json`). All [verified]. `(card)` = `external` revision, i.e. self-reported on the model card and imported into the MTEB repo.

| Model (params) | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA |
|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 (22.7M) | 0.6451 | 0.3159 | 0.5017 | 0.2164 | 0.3687 |
| sentence-transformers/all-MiniLM-L12-v2 (33.4M) | 0.6264 | 0.3225 | 0.4713 | 0.2182 | 0.3727 |
| BAAI/bge-small-en-v1.5 (33.4M) | 0.7127 | 0.3426 | 0.6035 (mteb 1.20.0; card says 0.5955) | 0.2052 | 0.4034 (card) |
| intfloat/e5-small-v2 (33M) | 0.6885 (card) | 0.3245 (card) | 0.4179 | 0.1771 | 0.3743 (card) |
| thenlper/gte-small (33.4M) | 0.7270 (card) | 0.3477 (card) | 0.5544 (card) | 0.2138 (card) | 0.3935 (card) |
| Snowflake/snowflake-arctic-embed-s (33.2M) | 0.6992 (card) | 0.3254 (card) | 0.5687 (card) | 0.1942 (card) | 0.3968 (card) |
| Snowflake/snowflake-arctic-embed-xs (22.6M) | 0.6451 (card) | 0.3089 (card) | 0.5208 (card) | 0.1836 (card) | 0.3452 (card) |
| ibm-granite/granite-embedding-30m-english (30.3M) | 0.7127 | 0.3374 | 0.5636 | 0.2254 | 0.3685 |
| MongoDB/mdbr-leaf-ir (22.9M) | 0.7056 | 0.3653 | 0.5938 | 0.1977 | 0.3917 |
| MongoDB/mdbr-leaf-mt (22.7M) | 0.7111 | 0.3626 | 0.6123 | 0.1869 | 0.3846 |
| avsolatorio/NoInstruct-small-Embedding-v0 (33.4M) | 0.7222 (card) | 0.3492 (card) | 0.5759 (card) | 0.2182 (card) | 0.4065 (card) |
| abhinand/MedEmbed-small-v0.1 (33.4M) | 0.7069 | 0.3462 | 0.6159 (card) | 0.2028 (card) | 0.3850 (card) |
| mixedbread-ai/mxbai-embed-xsmall-v1 (24.1M) | 0.6581 (card) | 0.3205 (card) | 0.4958 (card) | 0.2150 (card) | 0.3710 (card) |
| answerdotai/answerai-colbert-small-v1 (33M, **late interaction**, S12, self-reported) | 0.7477 | 0.373 | 0.5009 | 0.1842 | 0.4115 |
| mteb/baseline-bm25s (lexical, rev 0_1_10) | 0.6874 | 0.3208 | 0.4928 | 0.1578 | 0.2514 |

**Best <= 35M per dataset (S1 scan, dense single-vector)**

| | SciFact | NFCorpus | ArguAna | SCIDOCS | FiQA |
|---|---|---|---|---|---|
| Best <= 35M dense | 0.7270 gte-small (card) | 0.3653 mdbr-leaf-ir | 0.6159 MedEmbed-small-v0.1 (card) | 0.2254 granite-embedding-30m-english | 0.4065 NoInstruct-small-Embedding-v0 (card) |
| Best <= 35M incl. late interaction | 0.7477 answerai-colbert-small-v1 | 0.373 answerai-colbert-small-v1 | 0.6159 (same) | 0.2254 (same) | 0.4115 answerai-colbert-small-v1 |
| Ours (hybrid, MiniLM-L6) | 0.731 | 0.356 | 0.538 | 0.200 | 0.377 |
| Ours minus best dense <= 35M | +0.004 | -0.009 | -0.078 | -0.025 | -0.030 |

Same-size-class **hybrid** (BM25 + bge-small-en-v1.5, 33M), hobby/preprint sources, all [verified] in source:
S9 convex (alpha tuned on train) SciFact 0.736, NFCorpus 0.366; S9 RRF 0.722 / 0.362; S10 min-max fusion SciFact 0.730, NFCorpus 0.358; S5 RRF SciFact 0.7189.
S11 (SQLite FTS5 + bge-small, RRF, 1024-token chunks) Table 6 "BGE-small base" row: SciFact 0.6464, NFCorpus 0.3304, FiQA 0.3281, SCIDOCS 0.1778, ArguAna 0.4188 - these are *below* bge-small alone, and the same preprint gives 0.7263 for SciFact elsewhere (internal inconsistency), so treat S11 as weak evidence.
ArguAna / SCIDOCS / FiQA hybrid with a <= 35M model from a reliable source: not found.

Slightly above the budget, for orientation: ibm-granite/granite-embedding-small-english-r2 (47M, S1): SciFact 0.7549, NFCorpus 0.3714, ArguAna 0.5440, SCIDOCS 0.2406, FiQA 0.4081 [verified].
MiniLM-based SPLADE variants (<= 35M) on these datasets: not found in S1.

Protocol notes for this section: bge, e5, arctic and mdbr-leaf models use a query prefix/instruction (model-specific) while all-MiniLM does not [unverified - from memory of model cards];
`(card)` numbers were produced by the model authors with older mteb versions and may differ by a few thousandths from a re-run (bge-small ArguAna 0.5955 card vs 0.6035 re-run is an example).

## 4. Scale only: best nDCG@10 of any model size in the MTEB results repository (S1)

Top entry per dataset, plus the next ones because several top entries are outliers. All [verified] in S1. This is the repository behind the leaderboard, not a screenshot of the leaderboard UI;
the UI may hide some entries. Many of these large models were trained on the train splits of these datasets (or on data overlapping them), so they are **not zero-shot** - model-by-model status not checked [unverified].

| Dataset | Best | Next entries |
|---|---|---|
| SciFact | 0.8679 papr-ai/papr-embed-code-1-4b (4.0B) | 0.8660 openbmb/MiniCPM-Embedding (card); 0.8591 Salesforce/SFR-Embedding-2_R (card); 0.8540 infly/inf-retriever-v1 |
| NFCorpus | 0.5575 TencentBAC/Conan-embedding-v2 (1.5B) - outlier | 0.4699 voyage-3-m-exp; 0.4496 nvidia/NV-Embed-v2; 0.4366 infly/inf-retriever-v1 |
| ArguAna | 0.8979 voyage-3-m-exp (6.9B) | 0.8695 LGAI-Embedding-Preview; 0.8644 google/gemini-embedding-001; 0.8634 geevec-embeddings-1.0 |
| SCIDOCS | 0.5986 IEITYuan/Yuan-embedding-2.0-en (0.6B) - outlier | 0.3453 voyage-3-m-exp; 0.3274 Qwen/Qwen3-Embedding-8B; 0.3181 Octen-Embedding-8B |
| FiQA-2018 | 0.8206 ai-sage/Giga-Embeddings-instruct (3.2B) - outlier | 0.7801 voyage-3-m-exp; 0.7186 Conan-embedding-v2; 0.6573 nvidia/NV-Embed-v2 (card) |

"Outlier" = far above the next entries (e.g. SCIDOCS 0.60 vs 0.35), which usually indicates in-domain training data; reason not confirmed [unverified].

## 5. Code retrieval: CoIR (CosQA, StackOverflow QA)

### 5a. Published nDCG@10 (x100)

| System | CosQA | StackOverflow QA | Source |
|---|---|---|---|
| BM25 (CoIR paper) | 13.96 | 56.80 | S13 Table 3 [verified] |
| BM25 (`mteb/baseline-bm25s`, rev 0_1_10) | 18.75 | 70.26 | S1 [verified] |
| all-MiniLM-L6-v2 (22.7M) | 33.02 | 83.96 | S1 [verified]; not in the CoIR paper |
| all-MiniLM-L12-v2 (33.4M) | 32.09 | 80.63 | S1 [verified] |
| bge-small-en-v1.5 (33.4M) | 32.05 | 78.05 | S1 [verified]; not in the CoIR paper |
| e5-small-v2 / gte-small / arctic-embed-s | 29.68 / 32.95 / 33.72 | 83.53 / 78.12 / 83.05 | S1 [verified] |
| granite-embedding-30m-english (30.3M) | 35.45 | 83.93 | S1 [verified] |
| Best <= 35M in S1: lightonai/LateOn-Code-edge (16.8M, late interaction) | 40.54 | 85.63 | S1 [verified] |
| CoIR paper, best per column | 32.76 BGE-Base (110M) | 91.54 E5-Mistral (7B) | S13 Table 3 [verified] |
| CoIR paper, best average (Voyage-Code-002, avg 56.26) | 29.79 | 87.68 | S13 Table 3 [verified] |
| Other CoIR paper rows: E5-base / Contriever / OpenAI-Ada-002 | 32.59 / 14.21 / 28.88 | 86.86 / 66.05 / 72.40 | S13 Table 3 [verified] |
| Best in S1 today (any size) | 52.18 google/text-embedding-005 | 97.49 codefuse-ai/F2LLM-v2-14B | S1 [verified] |

Caveat: the two BM25 rows disagree a lot (13.96 vs 18.75; 56.80 vs 70.26). Different BM25 implementations/tokenizers (the CoIR paper's BM25 configuration was not read: [unverified]);
report your own BM25 against both. all-MiniLM-L6-v2 alone already beats every 110M model of the CoIR paper on CosQA.

### 5b. Hugging Face layout (S14, all [verified] through the HF API on 2026-11-12)

CoIR's own loader (`coir/data_loader.py`, `load_data_from_hf`) reads two repos per task:
`load_dataset("CoIR-Retrieval/{task}-queries-corpus")` -> splits `corpus` and `queries`; `load_dataset("CoIR-Retrieval/{task}-qrels")["test"]`. Task names: `cosqa`, `stackoverflow-qa`.

| Repo | Files / splits | Rows |
|---|---|---|
| `CoIR-Retrieval/cosqa-queries-corpus` | `data/corpus-*.parquet`, `data/queries-*.parquet` (config `default`, splits `corpus`, `queries`) | corpus 20,604; queries 20,604 |
| `CoIR-Retrieval/cosqa-qrels` | `data/train-*.parquet`, `data/valid-*.parquet`, `data/test-*.parquet` | train 19,604; valid 500; **test 500** |
| `CoIR-Retrieval/stackoverflow-qa-queries-corpus` | `data/corpus-*.parquet`, `data/queries-*.parquet` | corpus 19,931; queries 19,931 |
| `CoIR-Retrieval/stackoverflow-qa-qrels` | `data/train-*.parquet`, `data/test-*.parquet` (no valid) | train 13,951; **test 1,994** |
| `CoIR-Retrieval/cosqa` (used by MTEB task `CosQA`, pinned revision `bc5efb7e9d437246ce393ed19d772e08e4a79535`) | configs `default` (qrels: train/valid/test), `corpus`, `queries` | 19,604 / 500 / 500; 20,604; 20,604 |
| `mteb/StackOverflowQA` (used by MTEB task `StackOverflowQA`, revision `7b46f5ebd2365278c1bb025986dda2fee893bbf5`) | `corpus/test`, `queries/test`, `qrels/test` | 19,931; 1,994; 1,994 |

- Columns: corpus/queries rows have `_id`, `partition`, `text`, `title`, `language` (+ `meta_information` for CosQA); qrels rows have `query_id`, `corpus_id`, `score` (score "1"). Ids look like `q20105` / `d20105`.
- **Test split** = the `test` split of the qrels repo. Queries are filtered to the ids present in the test qrels (CoIR loader: `self.queries = {qid: ... for qid in self.qrels}`; MTEB loader does the same). The **corpus is NOT filtered**: retrieval runs against the whole corpus (train + valid + test partitions).
- Evaluation sizes: CosQA 500 queries / 20,604 documents; StackOverflow QA 1,994 queries / 19,931 documents (paper Table 2 rounds to 500 / 21K and 2K / 20K). One relevant document per query in both (qrels rows = queries).
- Text: query = the `text` field (CosQA: a short natural-language web query, e.g. "python code to write bool value 1"; StackOverflow QA: full question title + body, avg ~1.4K words per the paper). Document = `text` (CosQA: a Python function with docstring; StackOverflow QA: the answer, text mixed with code). `title` is an empty string in both, so the BEIR-style `title + " " + text` reduces to `text` [title emptiness verified on the first rows only; concatenation in CoIR's vendored BEIR code not opened: unverified].
- Metric: both runners use the BEIR/pytrec_eval nDCG@10 (CoIR: `EvaluateRetrieval(..., score_function="cos_sim")`; MTEB `main_score="ndcg_at_10"`, `eval_splits=["test"]`) [verified].
- StackOverflow QA documents and queries are long (paper: ~1.4K / ~1.2K words on average) so a 256-token model sees only the beginning of each; this applies equally to the published all-MiniLM-L6-v2 number.

## 6. Caveats (read before quoting)

1. Same-model hybrid numbers (section 1) come only from self-published pages and hobby repositories, single runs, no review. Peer-reviewed or vendor per-dataset nDCG@10 for BM25 + all-MiniLM-L6-v2: not found.
2. BM25 differs between sources (Anserini k1=0.9/b=0.4; bm25s k1=1.2/b=0.75; Weaviate; SQLite FTS5; own implementations) by up to ~0.03 nDCG@10 on SciFact, which propagates to the hybrid. Compare gains over each source's own BM25, not only absolutes.
3. "Best variant" rows in S3/S4 select the maximum over many fusion variants on the test queries; that is an optimistic upper bound, not a protocol. The zero-shot RRF / fixed-weight convex rows are the fair comparison for a fixed-weight system. If our fusion weight or chunk length was chosen on the same test queries, the same optimism applies to our numbers.
4. S3's ArguAna numbers use a non-standard protocol (dense 0.370 vs MTEB 0.502) and are excluded from the comparison.
5. Section 3 `(card)` values are self-reported by model authors; several models use query instructions; answerai-colbert-small-v1 is a multi-vector model (much larger index), not a like-for-like dense model.
6. Section 4 is not zero-shot and is taken from the results repository, not from the leaderboard UI; the three flagged outliers should not be quoted as "state of the art" without checking their training data.
7. Numbers read through an HTML-to-text fetch (S2, S3, S4, S11, S13) were transcribed by a summarising reader; S1, S5-S10, S12 and S14 were read directly from raw files / API responses.
