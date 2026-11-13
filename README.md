*This project has been created as part of the 42 curriculum by <login>.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase. It:

1. **indexes** the repository (Python, Markdown, text) into structure-aware
   chunks, a BM25 lexical index and a MiniLM semantic index;
2. **retrieves** the most relevant source locations for a question with a
   hybrid lexical + semantic ranking;
3. **generates** a grounded answer with `Qwen/Qwen3-0.6B` running locally on
   CPU, citing the excerpts it used;
4. **evaluates** retrieval with recall@k against reference datasets.

The five bonuses are implemented: semantic embeddings, hybrid retrieval,
incremental indexing, caching and a local HTTP API.

An **experiment lab** (`python -m src lab ...`) measures every retrieval
setting on the reference datasets, tunes them with cross-validation and
writes a cost/benefit report ([`experiments/REPORT.md`](experiments/REPORT.md)).
The defaults of the project are the result of that process.

## Instructions

Requirements: Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), ~3 GB of
disk for the CPU deep-learning stack and the models (downloaded from the
Hugging Face Hub on first use).

```bash
make install            # uv sync (CPU-only PyTorch wheels)
make run                # index + search both public datasets
make lint               # flake8 + mypy (lint-strict: mypy --strict)
make test               # unit + integration tests (make test-slow: real models)
make debug ARGS='search "how to load a LoRA adapter"'
make clean
```

Expected layout: the vLLM sources in `data/raw/vllm-0.10.1/`, the datasets in
`data/datasets/{AnsweredQuestions,UnansweredQuestions}/`.

### Commands

Every command runs as `uv run python -m src <command> [options]`.

| Command | Purpose |
|---|---|
| `index --max_chunk_size 2000` | Build/update the index under `data/processed/` |
| `search "<query>" --k 10` | Top-k sources for one query (JSON) |
| `search_dataset --dataset_path P --k 10 --save_directory D` | Batch search → `StudentSearchResults` |
| `answer "<query>" --k 10` | Answer one question, with sources |
| `answer_dataset --student_search_results_path P --save_directory D` | Batch answers → `StudentSearchResultsAndAnswer` |
| `evaluate --student_search_results_path P --dataset_path D` | Our own recall@1/3/5/10 |
| `serve --host 127.0.0.1 --port 8000` | Local HTTP API |
| `clear_cache` | Empty the query/answer cache |
| `system [--benchmark]` | Show the hardware found and the runtime plan; `--benchmark` measures the candidate CPU configurations and remembers the fastest |

Global options (valid for every command): `--index_dir`, `--mode`
(`auto`|`bm25`|`semantic`|`hybrid`), `--semantic_weight`, `--fusion`
(`linear`|`rrf`), `--llm`, `--context_tokens`, `--max_new_tokens`,
`--no_cache`, `--rerank` (cross-encoder second stage, off by default),
`--rerank_depth` (10), `--rerank_weight` (0.3) and `--profile` (a JSON file
of settings, or the name of one in `experiments/profiles/`; an option given
on the command line wins over the profile). `index` also accepts `--semantic=False` (lexical only, ~2 s),
`--force` (full rebuild) and `--extensions .py,.md,.txt,.rst`.

Experiment lab commands (see [Experiment lab](#experiment-lab)):

| Command | Purpose |
|---|---|
| `lab experiment --suite NAME` | Run the trials of `experiments/suites/NAME.json` and store per-question results |
| `lab tune --suite NAME` | Coordinate descent over the factors of the suite, with 5-fold cross-validation |
| `lab analyze` | Statistics, Pareto fronts and recommended profiles → `experiments/REPORT.md` |
| `lab train_router` | Train the code-or-docs query router → `models/query_router.json` |

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000

uv run python -m src search "How do I load a LoRA adapter at runtime?" --k 5

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

uv run python -m src answer "What does --enable-prefix-caching do?"

uv run python -m src serve --port 8000
curl "localhost:8000/search?q=paged%20attention&k=3"
curl -X POST localhost:8000/answer -H 'Content-Type: application/json' \
     -d '{"query": "Which hardware does vLLM support?", "k": 5}'
```

The API documents itself at `http://localhost:8000/docs` (OpenAPI).

## System architecture

The code follows a **hexagonal (ports & adapters) architecture**. The
rationale and the alternatives considered are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
        inbound adapters                core                  outbound adapters
   ┌──────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
   │ Fire CLI  (cli.py)   │─►│ application (use cases)  │─►│ FileSystemCorpus         │
   │ Lab CLI (lab_cli.py) │  │  IndexingService         │  │ ParallelDocumentProcessor│
   │ FastAPI (http_api.py)│  │  SearchService           │  │ BM25Index / BM25Builder  │
   └──────────────────────┘  │  AnswerService           │  │ SentenceTransformerEmbed.│
                             │  DatasetService          │  │ CachingEmbedder          │
                             │  ExperimentService       │  │ CrossEncoderReranker     │
                             │  Tuner, RouterTrainer    │  │ TransformersGenerator    │
                             │  AnalysisService         │  │ FileSystemIndexRepository│
                             │  ports.py (Protocols) ◄───┼──│ JsonlTrialStore, router, │
                             ├──────────────────────────┤  │ SqliteCache, JSON repos, │
                             │ domain (pure, no I/O)    │  │ SourceReader, TqdmProgr. │
                             │  chunking, tokenizer,    │  └──────────────────────────┘
                             │  ranking/fusion, priors, │
                             │  prompt, evaluation,     │      bootstrap.py wires
                             │  experiment statistics   │      adapters into ports
                             └──────────────────────────┘
```

| Layer | Contents | Depends on |
|---|---|---|
| `src/domain/` | Pydantic models, error hierarchy, validation, chunkers (`PythonChunker`, `TextChunker`, `ChunkerRegistry`), tokenizer, top-k / linear / RRF fusion, query router and chunk priors, prompt building, recall@k, experiment models and statistics (paired bootstrap, Pareto front, k-fold) | nothing else in the project |
| `src/application/` | `ports.py` (Protocols for corpus, processor, lexical index, embedder, index repository, reranker, generator, cache, dataset repository, source reader, progress, pipeline factory, trial store, router repository) and the use cases, including the experiment runner, the tuner and the analysis | `domain` |
| `src/adapters/outbound/` | Implementations of the ports: filesystem, process pool, BM25 on scipy, MiniLM, Qwen via transformers, SQLite, JSON, tqdm | `application.ports`, `domain` |
| `src/adapters/inbound/` | Fire CLI and FastAPI app: parse input, call use cases, format output | `application`, `bootstrap` |
| `src/bootstrap.py`, `src/config.py` | Composition root (`Container`) and `Settings` | everything |

The data flow of a query is CLI/HTTP → `SearchService` (mode selection,
batching, cache) → lexical/semantic/hybrid retriever → `AnswerService` →
`Generator` port → Qwen3-0.6B. Indexing is CLI → `IndexingService` →
corpus, processor, BM25 builder and embedder ports → index repository.

Swapping a component means writing a new adapter and changing one line
in `bootstrap.py`. Examples: an LLM served by an API, a vector database,
or a chunker for a new document type registered in `ChunkerRegistry`.

### Tests

| Suite | Count | Scope |
|---|---|---|
| `tests/unit/` | 482 | every domain module and use case, with in-memory fakes of all ports (`tests/fakes.py`); no disk, no models |
| `tests/integration/` | 242 | real adapters on a temporary corpus, the composition root, the CLI as a subprocess (degenerate inputs never print a traceback), the HTTP API via `TestClient` |
| `tests/slow/` (`-m slow`) | 7 | real MiniLM, cross-encoder and Qwen3-0.6B smoke tests |

`make test` runs the unit and integration suites (~30 s), and
`make test-slow` runs the model tests.

## Chunking strategy

Chunks are exact slices of the file: `text == file[first:last]`, with an
exclusive end and a length always ≤ `--max_chunk_size` (default 2000, the
moulinette maximum).

**Python (`PythonChunker`)**: the file is parsed with `ast`. Top-level
functions and classes (with their decorators) become segments, and so does
the code between them (imports, constants). A class larger than the limit
is split into its header and its methods. Any segment that is still too
long is cut on line boundaries. Consecutive small segments are then packed
greedily up to the size limit, so a chunk never starts or ends in the
middle of a definition that would fit. Each chunk records the definitions
it contains (`class Scheduler | def Scheduler.schedule`) as its *context*.
Files that do not parse fall back to the text chunker.

**Markdown/text (`TextChunker`)**: the document is split at headings
(ignoring `#` lines inside fenced code blocks). A section that is too large
is split on blank lines outside code fences, and the paragraphs are packed
back up to the limit. Small sections are merged with their neighbours. The
context is the heading breadcrumb (`LoRA Adapters > Dynamically serving
LoRA Adapters`). Plain-text files without headings (e.g.
`CMakeLists.txt`) degrade to paragraph packing.

What gets indexed for each chunk is `file path + context + text`. Questions
often name a file or a class that only appears in the path or in a
definition far above the matched lines, and this keeps those words
searchable.

Effect of the chunk size (BM25 only, public datasets, recall@5):

| target size | chunks | code | docs |
|---|---|---|---|
| 600 | 23,324 | 0.828 | 0.890 |
| 1000 | 19,178 | 0.869 | 0.860 |
| 1500 | 15,713 | 0.869 | 0.900 |
| **2000 (default)** | **13,287** | **0.879** | **0.890** |

The differences are within noise: one question is worth 1%. 2000 is kept
because fewer chunks make embedding faster and give the LLM more
surrounding code.

## Retrieval method

**Tokenizer**: identifiers are kept whole *and* split into their snake_case
and camelCase parts (`get_kv_cache` → `get_kv_cache, get, kv, cache`).
English stopwords are removed and a light stemmer folds plurals and
`-ing`/`-ed` forms. This lets a question match both a verbatim identifier
and a paraphrase of it.

**BM25** (mandatory, `k1=1.2`, `b=0.75`): at indexing time the weight
`idf(t)·tf·(k1+1)/(tf+k1·(1−b+b·|d|/avgdl))` of every (chunk, term) pair is
precomputed into a sparse CSC matrix. Scoring a query is then a sparse
matrix–vector product over the query's columns, which takes about 1 ms per
query.

**Field weights**: a term found in the file path or in the context of a
chunk (heading breadcrumb, definition names) counts 3 times as much as a
term of the body (`path_weight`, `context_weight`). BM25 uses `k1=1.2`
and `b=0.5`.

**Semantic** (bonus 1): `all-MiniLM-L6-v2` embeddings (384 dimensions,
L2-normalised, 64 input tokens), ranked by cosine similarity (a dot
product).

**Hybrid** (bonus 2, the default when embeddings exist):
`score = bm25 / max(bm25) + 0.4 · cosine`. BM25 is scaled per query to
[0, 1], the range of the cosine. Reciprocal Rank Fusion is also available
(`--fusion rrf`).

**Priors** (applied to the fused score, before ranking):

- **Query routing**: a logistic regression on surface features of the
  question (an identifier, "default value", "how do you"...) and its
  frequent words estimates the probability `p` that the answer is in
  source code. Code chunks are scaled by `1 + 0.1·(2p − 1)` and text
  chunks by `1 − 0.1·(2p − 1)`. The model is 93.5% accurate under
  cross-validation and lives in `models/query_router.json`; without the
  file, routing is simply off.
- **Test files** (`tests/`, `test_*.py`) are scaled by 0.6: they repeat
  the identifiers of the code they test but are never the reference.
- **File aggregation**: each chunk receives 0.2 × the best score of its
  file, so a file with strong evidence ranks above isolated matches while
  the order inside the file is kept.

**Reranking** (optional, `--rerank`): the 10 best first-stage candidates
are re-scored by the cross-encoder `ms-marco-MiniLM-L-6-v2` (256 tokens).
Unlike the embedding model, it reads the query and the passage together.
Its min-max normalised score is blended with the first-stage score:
`0.7 · first + 0.3 · cross-encoder`. A cross-encoder alone is worse
(code R@5 0.71–0.85): it was trained on web passages and misses the exact
identifier matches that BM25 rewards, so it is only used to break ties
near the top.

## Performance analysis

Measured on the public datasets with the official moulinette (k=10, 20-core
CPU, no GPU):

| Dataset | Mode | R@1 | R@3 | R@5 | R@10 | Required R@5 |
|---|---|---|---|---|---|---|
| docs | hybrid (default) | 0.70 | 0.92 | **0.95** | 0.97 | 0.80 |
| code | hybrid (default) | 0.65 | 0.86 | **0.92** | 0.94 | 0.50 |
| docs | bm25 | 0.70 | 0.93 | 0.93 | 0.95 | |
| code | bm25 | 0.61 | 0.86 | 0.92 | 0.95 | |
| docs | semantic | 0.39 | 0.56 | 0.66 | 0.77 | |
| code | semantic | 0.24 | 0.52 | 0.56 | 0.63 | |
| docs | hybrid + rerank | 0.75 | 0.89 | 0.94 | 0.97 | |
| code | hybrid + rerank | 0.60 | 0.88 | 0.93 | 0.94 | |
| docs | hybrid, before tuning | 0.69 | 0.86 | 0.91 | 0.94 | |
| code | hybrid, before tuning | 0.63 | 0.85 | 0.88 | 0.92 | |

| Operation | Time | Limit |
|---|---|---|
| Full indexing, with embeddings | ~35 s (~60 s before the runtime plan) | 300 s |
| Full indexing, `--semantic=False` | ~2 s | |
| Incremental re-index, nothing changed | ~1.5 s | |
| Incremental re-index, one file changed | ~4 s | |
| `search_dataset`, 100 questions (hybrid, cold) | ~4 s | 90 s / 200 q |
| `search_dataset`, 100 questions (hybrid + rerank, cold) | ~17 s | 90 s / 200 q |
| Same single query, cold → cached | 5.4 s → 0.6 s | |
| Answer generation (Qwen3-0.6B, CPU) | ~7–8 s / question | |

Semantic search alone is clearly weaker than BM25: the questions quote
identifiers and flags verbatim. With the tuned lexical index the
embeddings add about one point of mean recall@5 (2 points on docs, none
on code), so `index --semantic=False` (2 s instead of 60 s) is a valid
fast profile (`--profile speed`).

**Why reranking is off by default**: on top of the tuned pipeline the
cross-encoder gains one code question and loses one docs question at
recall@5 (it does improve docs R@1, 0.70 → 0.75), for about 14 s more per
100 questions.

## Experiment lab

Every number above comes from `python -m src lab`, which is independent of
the corpus: a *suite* (`experiments/suites/*.json`) names a base
configuration and the factors to vary; a *trial* is one configuration
evaluated on datasets with reference spans.

- **One at a time, grid or list**: `one_at_a_time` suites measure each
  change alone (ablation, sweep of a constant); `grid` suites measure all
  combinations to detect interactions.
- **What is stored** (`experiments/results/trials.jsonl`): parameters,
  recall@1/3/5/10, MRR, warm search time, index time and the rank obtained
  by *every question*. Any two trials can then be compared question by
  question (paired bootstrap, 95% interval).
- **Cheap trials**: each set of index parameters has its own index
  directory, models are loaded once, and a vector cache keyed by text hash
  avoids re-embedding unchanged chunks. A search-only trial takes ~1 s.
  `"lab_cold": true` disables all reuse to measure real indexing time.
- **Tuning** (`lab tune`): coordinate descent on
  `recall@5 + 0.1·MRR`, repeated inside 5-fold cross-validation. The
  report gives the in-sample gain, the held-out gain and how many folds
  chose each value. The *consensus profile* keeps only the changes that at
  least 4 folds make and 3 folds agree on.
- **Analysis** (`lab analyze`): cost/benefit table per suite (Δ recall
  points with interval, Δ seconds, points per second, verdict),
  interactions (`gain(A+B) − gain(A) − gain(B)`), Pareto fronts of recall
  against search time and against cold indexing + 200 questions, and three
  profiles (`precision`, `balanced`, `speed`) written to
  `experiments/profiles/`.

### What the experiments showed

347 trials on the 199 public questions (details in
[`experiments/REPORT.md`](experiments/REPORT.md), plan and diagnosis in
[`docs/ROADMAP.md`](docs/ROADMAP.md)):

| Change | Alone, Δ mean R@5 | Verdict | Adopted |
|---|---|---|---|
| Path terms ×2–3 | +1.5 to +2.0 | likely helps | ×3 |
| Context terms ×2–4 | +1.5 | likely helps | ×3 |
| Query routing 0.1–0.2 | +1.0 | likely helps | 0.1 |
| BM25 `b` 0.3–0.6 | +0.5 to +1.0 | neutral | 0.5 (chosen by 4 of 5 folds) |
| File aggregation | +0.5 | neutral | 0.2 (4 of 5 folds) |
| Table header as context | +0.5 | neutral | yes (5 of 5 folds) |
| Test files ×0.6 | 0.0 (MRR only) | neutral | yes (5 of 5 folds) |
| 64-token embeddings | +0.5 to +1.0 | neutral | yes: indexing 95 s → 60 s |
| BM25 `k1` 0.6–0.9, `max_chunk_size` 1500, hybrid weight 0.3–0.6 | −0.5 to +1.5 | unstable across folds | no |
| Cross-encoder reranker | +1.5 before tuning, +0.0 after | not worth 14 s | optional |
| Smaller target chunks, 256-token embeddings, `bge-small`, `multi-qa-MiniLM`, word bigrams, RRF | −4.0 to +0.5 | neutral or hurts | no |

Three lessons:

1. **No single change is statistically significant.** With 199 questions
   one question is half a point, and every interval includes zero. Only
   the combination is clearly positive: 0.894 → 0.935 mean recall@5.
2. **Gains do not add up.** In the 2⁴ grid, changes worth +1.0 to +2.0
   alone give +2.0 together: they fix the same questions (negative
   synergy in the report).
3. **Tuning overfits.** The best configuration found in-sample gains 6.0
   points of the objective, but only 2.8 on held-out questions. The
   adopted defaults are therefore the consensus profile (+3.7 in-sample),
   not the in-sample winner.

Remaining errors: 5 docs and 8 code questions. Most have the right file
in the top 5 but another chunk of it, and about 4 are answered in more
than one place of the corpus while the reference lists only one.

## Soft-skills analyzer (business reuse)

The same core powers a second application: from a diarized transcript
of a group session it produces, per participant, a competency scorecard
in which every rating rests on verbatim quotes. It is independent of
the 42 commands and does not change them.

```bash
uv run python -m src skills catalogue --language es
uv run python -m src skills interaction \
    --transcript examples/skills/transcripts/demo_es.txt
uv run python -m src skills evidence \
    --transcript examples/skills/transcripts/demo_es.txt \
    --participant Pablo --competency LISTEN
uv run python -m src skills assess \
    --transcript examples/skills/transcripts/demo_es.txt
```

- **Rubric**: 11 competencies, 110 behavioural indicators, in English
  and Spanish (`knowledge/rubrics/`), derived from published HR and
  teamwork frameworks; sources and their appraisal in
  `docs/skills/THEORY.md`.
- **Verification**: quotes must exist in the participant's own turns;
  a rating needs three incidents from two exchanges, otherwise the
  result is "insufficient evidence" (`docs/skills/DESIGN.md`).
- **Reference data and evaluation**: annotation forms, inter-annotator
  agreement (Krippendorff's alpha, ICC), consensus, system-versus-human
  scores, and the experiment lab on evidence retrieval
  (`docs/skills/ANNOTATION_GUIDE.md`).

The rubric has not been validated with human assessors yet and the
sample sessions are synthetic: the output is a draft for a human
assessor.

## Beyond the 42 corpus (GPU study)

With the GPU, the retrieval was re-measured on nine datasets at once:
the two 42 question sets and seven public ones (BEIR and CoIR), each
configuration scored as a relative change against the shipped one and
checked by leaving one dataset out (`make study SUITE=... TORCH=cuda`,
`docs/benchmarks/GENERALIST.md`). The main result is that the fusion
weight should depend on the **query**: prose leans on the embeddings,
a query that names code identifiers leans on BM25. With that rule
bge-large reaches +20.9% on average without losing on any dataset,
where the best fixed weight gave +19.8% and lost 9.9% on the 42 code
questions. The default now applies the rule on the existing index
(42 docs recall@5 0.95 → 0.96, code unchanged at 0.919, seven public
datasets up and one down); `--profile general`, `general-fast` and
`general-cpu` select the larger configurations.

## Results site and CI/CD

Every result of the project is published at **https://rag.peer4u.app**:
the lab suites, the BEIR/CoIR benchmarks, the soft-skills lab and
rubrics, the hardware runs and all the documents. The experiments run
on the workstation GPU and their result files are committed; GitHub
Actions lints (flake8, `mypy --strict`), runs the unit and integration
tests with a branch-coverage gate, builds the static site
(`python -m results_site build`) and, on every merge to `main`,
deploys it. Workflow and conventions: `CONTRIBUTING.md`; architecture
and operations: `docs/DEPLOYMENT.md`.

```sh
make ci           # what CI runs: lint-strict, coverage, site
make site-serve   # preview the site at http://localhost:8765
```

### From a recording

`skills transcribe --audio FILE --speakers N` turns a recording into
that transcript on this machine: Whisper for the words and their
times, voice embeddings for who spoke when. `skills sentences` and
`skills tag` then split turns into sentences that keep speaker and
time, and propose rubric indicators per sentence. On synthetic
four-speaker meetings in Spanish: 5.4% word error rate and 91.5% of
the words attributed to the right speaker, at about 22× real time on
the GPU. Those figures are from read speech without overlap and are
optimistic; see `docs/skills/AUDIO.md`, which also says what must be
settled before recording real people.

## Design decisions

- **Runtime plan chosen from the hardware.** Before a model loads, the
  machine is probed (physical cores, performance and efficiency cores,
  memory, GPUs usable by the installed PyTorch) and a plan is applied:
  the GPU when there is one, otherwise one thread per performance core
  with the process pinned to those cores. On the development machine
  (8 performance + 4 efficiency cores, no GPU) this was measured, not
  assumed: 8 pinned threads generate 14.7 tokens/s against 10.6 with
  all 20 threads, and full indexing went from ~60 s to ~35 s with
  identical search results. `system --benchmark` repeats the
  measurement on any machine and stores the winner in
  `data/runtime_plan.json`; `--device`, `--torch_threads` and
  `--cpu_cores` override it. A usable GPU is one more measured
  candidate, chosen only if it wins. PyTorch is installed as the CPU
  build by default; `make install-cuda` installs the CUDA 12.6 build
  and `make <target> TORCH=cuda` runs with it. When a graphics card is
  present but unusable, `system` says which piece is missing (kernel
  driver, PyTorch build). On the development machine's RTX 3070 Ti
  the GPU wins: 93 tokens/s against 14.7 on the best CPU plan, and
  embeddings in 0.08 s instead of 0.50 s (6.4× overall). Every
  `system --benchmark` run is appended to
  `experiments/hardware/runtime.jsonl`.

- **BM25 implemented from scratch on scipy sparse matrices** rather than a
  library: precomputing the weights makes queries trivially fast, and the
  whole algorithm stays explainable.
- **Linear fusion over RRF**: on these datasets a weighted sum of normalised
  scores was as good as or better than RRF for every weight tested. RRF
  discards the large BM25 margins that identify verbatim matches.
- **64-token embeddings**: indexing takes 60 s instead of 205 s with 256
  tokens (or 95 s with 128) with no measurable recall loss: the path and
  the heading or definition name come first in the embedded text. This is
  a safety margin for the 5-minute limit on slower machines.
- **Defaults chosen by cross-validated consensus**, not by the best score
  on the public questions, to limit overfitting to them.
- **Incremental indexing by content hash** (bonus 3): each file's chunks,
  term counts and embeddings are cached in `records.pkl`. Only new or
  modified files are re-chunked and re-embedded. Deleted files are dropped.
  The BM25 statistics are global (IDF, average length), so the sparse
  matrix is reassembled from the cached term counts in under a second.
  Changing any setting that affects the index (chunk sizes, field weights,
  BM25 constants, embedding model or input length) invalidates the cache.
- **Caching** (bonus 4): (a) the index is stored ready to query (a
  precomputed BM25 matrix, embeddings loaded with `mmap`, cached in-process
  per fingerprint), and (b) search results and generated answers are stored
  in SQLite, keyed by the index fingerprint. Re-indexing therefore
  invalidates stale entries automatically.
- **Generation**: Qwen3's thinking mode is disabled and decoding is greedy,
  which is faster, deterministic and less prone to drifting. Retrieved
  chunks are added in rank order up to a 2000-token budget, and the
  overflowing chunk is truncated. The system prompt forbids facts that are
  not in the context, asks for citations `[n]` and defines an explicit
  "not enough information" answer.
- **Robustness**: every command is wrapped so that errors (missing files,
  malformed JSON, k ≤ 0, empty query, missing index, unknown mode…) print
  one message and exit with code 1, never a traceback. An empty or
  meaningless question produces an empty result list instead of failing the
  whole dataset.
- **HTTP API** (bonus 5): FastAPI with `GET /health`, `GET|POST /search`
  and `POST /answer`. Requests are validated by pydantic, and access to the
  models is serialised with a lock.

## Challenges faced

- **Character offsets**: `ast` reports byte columns, not characters, so
  Python chunk boundaries are computed only from line numbers mapped to
  character offsets. A test checks `text == file[first:last]` for every
  chunk.
- **Vocabulary mismatch**: questions paraphrase code (`"KV cache blocks"`
  vs `kv_cache_manager.allocate_slots`). Identifier splitting and indexing
  the file path and definition names fixed most cases.
- **Semantic retrieval underperforming**: an equal-weight RRF hybrid was 6–11
  points *below* BM25. Tuning the fusion weight and switching to linear
  fusion made the hybrid match or beat BM25.
- **Indexing time on CPU**: embedding dominated (over 95% of the time). Reducing
  MiniLM's input to 64 tokens and parallelising chunking across processes
  brought full indexing to ~60 s.
- **CPU generation latency**: prompt processing dominates. A 2000-token
  context budget keeps answers at ~7 s with no visible loss of quality.

- **Measuring small gains honestly**: after the first tuning, every
  remaining idea was worth one or two questions, less than the noise of a
  199-question benchmark. The experiment lab (paired per-question
  comparison, cross-validation, consensus across folds) was built to tell
  real gains from luck; it also showed that the cross-encoder and several
  plausible ideas (smaller chunks, bigger embedding models, bigrams) do
  not pay for their cost here.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009).
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods* (SIGIR 2009).
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (NeurIPS 2020).
- Reimers & Gurevych, *Sentence-BERT* (2019); the
  [sentence-transformers](https://www.sbert.net/) documentation.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B), Hugging Face
  [transformers](https://huggingface.co/docs/transformers) documentation.
- Python [`ast`](https://docs.python.org/3/library/ast.html),
  [SciPy sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html),
  [FastAPI](https://fastapi.tiangolo.com/),
  [Python Fire](https://github.com/google/python-fire).

### Use of AI

An AI assistant (Claude) was used to scaffold the project structure,
draft parts of the implementation and the tests, and run the parameter
sweeps (chunk size, BM25 `k1`/`b`, fusion weights, embedding length). Every
design choice was checked against measurements on the public datasets, and
all code was reviewed, tested and is understood by the author(s).
