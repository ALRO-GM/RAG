# Generalist study: retrieval beyond the 42 corpus (GPU)

Run on 2026-11-13 on the workstation's RTX 3070 Ti. Question: with a
GPU making larger models and longer inputs affordable, which retrieval
configuration works best **across corpora**, not only on the vLLM
questions it was tuned on?

Reports with every number: `experiments/general/general_models.md`
(G1), `general_fusion.md` (G2), `general_adaptive.md` (G3),
`general_cpu.md` (G4). Raw rows: `experiments/general/benchmarks.jsonl`
and `experiments/results/trials.jsonl`.

## Method

- **Nine datasets**, each weighing the same: the two 42 question sets
  (docs, code; recall@5) and seven public ones (BEIR: SciFact,
  NFCorpus, ArguAna, SCIDOCS, FiQA; CoIR: CosQA, StackOverflowQA;
  nDCG@10). 6 171 public queries over 141 320 documents.
- Every score is a **relative change against the shipped
  configuration**, so a point on FiQA and a point on the 42 code
  questions count alike.
- A configuration is ranked by its **mean** change; the **robust**
  choice is the best mean that loses at most 2% on any dataset.
- **Leave one dataset out**: the configuration is chosen on eight
  datasets and scored on the ninth. This estimates how the choice
  behaves on a corpus it has not seen, which is the actual goal.
- `make study SUITE=<suite> TORCH=cuda` reruns a suite; it resumes and
  frees each dataset's indexes when scored.

## What was added to make it possible

| Change | Why |
|---|---|
| `convex` and `dbsf` fusions | The old fusion adds max-scaled BM25 to a *raw* cosine, so its weight means something different for each embedding model (2.0 helped MiniLM and hurt bge-small). These normalise both scores over the candidates first. |
| `passage_prefix` | Models such as E5 expect a prefix on indexed text. |
| float16 on CUDA | 3× faster than float32 (394 vs 133 texts/s, bge-base, 512 tokens); document vectors within cosine 0.9995, same top-1 on 300/300 queries, 99.7% of the top-10 shared. |
| Length-sorted embedding blocks | Less padding per batch. |
| **Query-adaptive weight** | See G3. |

## Findings

**G1, embedding model** (fusion `convex`, weight 1): the bge family
wins. bge-large +14.0% mean with no loss anywhere; bge-base +13.8% at
a third of the indexing cost; E5-base +6.8%; RRF is the worst fusion
(+5.3% mean, −14.3% on the 42 code questions).

**G2, fusion weight**: more semantic weight keeps helping on public
data (bge-large `convex` 4.0: +19.8% mean, FiQA +60%) but costs the 42
code questions (−9.9%); embeddings alone lose 14% there, BM25 alone
is below the baseline on eight datasets and level on one (−6.4% mean). One fixed weight cannot win on
both sides.

**G3, the weight should depend on the query, not on the corpus.** The
42 questions name identifiers: 98–99% of them contain at least one
(10–15% of their words), against 3–21% of the queries of the three
BEIR sets checked (SciFact, FiQA, ArguAna; under 1% of their words). An identifier is a token the reader expects to match
exactly, so lexical evidence is strong for it; prose is where
embeddings help. With the semantic weight sliding from 4.0 (prose) to
1.0 (identifiers at 20% of the words):

| bge-large, `convex` | mean | worst | 42 code | FiQA |
|---|---|---|---|---|
| fixed weight 1.0 | +14.0% | 0.0% | +1.1% | +37.2% |
| fixed weight 4.0 | +19.8% | −9.9% | −9.9% | +60.4% |
| **adaptive 4.0 → 1.0** | **+20.9%** | **0.0%** | **0.0%** | +59.9% |

Held out, the adaptive choice is within 0–1 point of the best
configuration on seven of nine datasets. The two exceptions are FiQA
(14.7 points behind the best) and CosQA (23.1 behind: bge-base is much
better than bge-large there).

**G4, the same idea on the light CPU model** (all-MiniLM-L6-v2):

| MiniLM | mean | worst | 42 docs / code | cold index on this CPU |
|---|---|---|---|---|
| shipped until now (64 tokens, fixed 0.4) | 0.0% | 0.0% | 0.950 / 0.919 | 34 s |
| **64 tokens, adaptive 2.0 → 0.4** | +6.0% | −5.3% (ArguAna) | 0.960 / 0.919 | 34 s (same index) |
| 256 tokens, adaptive 2.0 → 0.4 | +10.4% | −1.1% | 0.950 / 0.909 | 133 s |

## Decisions

1. **Default** (`Settings`): adaptive weight on the existing index
   (`semantic_weight` 2.0, `identifier_weight` 0.4,
   `identifier_saturation` 0.1). Same index, same indexing time. For a
   query with identifiers at 10% or more of its words (most 42
   questions) the ranking is exactly the old one. On the public 42
   questions: docs recall@5 0.95 → 0.96, code 0.919 unchanged; recall@1
   moves 0.70 → 0.67 on docs and 0.646 → 0.657 on code. That is neutral
   within the noise the lab measured; the gain is outside the 42 corpus
   (7 public datasets up, 1 down).
2. **256 tokens is not the default**: 133 s of cold indexing on this
   CPU is inside the 300 s limit of the 42 evaluation but too close on
   a slower machine.
3. **Profiles** (`--profile NAME`, in `experiments/profiles/`):

   | Profile | Model | For | Mean vs old default |
   |---|---|---|---|
   | `general` | bge-large, 512 tokens, adaptive 4.0 → 1.0 | a GPU; best and robust | +20.9% |
   | `general-fast` | bge-base, 512 tokens, adaptive 4.0 → 1.0 | a GPU, a third of the indexing | +20.2% (−2.2% on 42 code) |
   | `general-cpu` | MiniLM, 256 tokens, adaptive 2.0 → 0.4 | CPU only | +10.4% |
   | `tuned` | MiniLM, 64 tokens, fixed 0.4 | the previous 42 optimum | 0.0% |

## Limits

- The weights were picked on the same test queries they are reported
  on; the leave-one-out table is the honest estimate (mean regret 4.5
  points for the adaptive suite).
- Nine datasets, all in English, two of them written from the same
  codebase. FiQA and CosQA show that the best model still depends on
  the corpus.
- Rerankers were not rerun on the GPU. On CPU the cross-encoder gave
  +0.8 points of mean nDCG@10 for 15–30× the search time; larger
  rerankers are the natural next suite.
- The identifier detector is a set of regular expressions for code
  conventions. A corpus with other exact-match tokens (part numbers,
  legal references) needs its own patterns.
