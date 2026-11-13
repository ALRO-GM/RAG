# Architecture decision: hexagonal (ports & adapters)

## Constraints

- **42 subject**: `python -m src <cmd>`, Fire CLI, pydantic models, flake8 +
  mypy, Qwen3-0.6B. During the defense we may have to change "a few lines"
  live, so the code must stay easy to navigate.
- **Business reuse**: the same core must serve the soft-skills platform:
  transcripts instead of code, ChromaDB/an LLM API instead of
  BM25-on-disk/Qwen, FastAPI instead of a CLI.

## Options considered

| Style | Pros | Cons for this project |
|---|---|---|
| Layered (controller → service → repository) | Simple, familiar | Services depend on concrete infrastructure (torch, scipy, files). Swapping the LLM or the vector store touches the business logic. |
| Clean / Onion (entities, use cases, interface adapters, frameworks) | Strict dependency rule | 4+ layers and request/response objects for each use case: too much ceremony for ~3k lines and a live recode. |
| **Hexagonal (ports & adapters)** | The core depends only on ports (`Protocol`s). Infrastructure is plugged in at the edges. Unit tests use in-memory fakes. | One extra indirection (the composition root). |
| Microservices | Independent scaling | Out of scope: one process, CPU-only, evaluated locally. |

**Decision: hexagonal**, kept pragmatic: three layers plus a composition
root, with no DI framework.

## Dependency rule

```
adapters/inbound  ──►  application  ──►  domain
adapters/outbound ──►  application.ports, domain
bootstrap         ──►  everything (wires adapters into ports)
```

`domain` imports nothing from the project except itself. `application`
imports `domain` and its own `ports`. Adapters implement the ports. Only
`bootstrap.py` knows the concrete classes.

## Layout

```
src/
├── __main__.py              entry point → adapters.inbound.cli
├── bootstrap.py             composition root: Settings → wired services
├── config.py                Settings (all tunables, one place)
├── domain/                  pure logic, no I/O, no ML frameworks
│   ├── models.py            subject models + Chunk, ScoredChunk, reports
│   ├── errors.py            error hierarchy (user-facing)
│   ├── validation.py        k / query validation
│   ├── chunking/            span helpers, Python AST + Markdown chunkers, registry
│   ├── text_analysis.py     tokenizer (identifier splitting, stemming)
│   ├── ranking.py           top-k, linear fusion, RRF, rerank fusion, priors
│   ├── priors.py            query router (logistic regression), test paths
│   ├── experiments.py       Suite, TrialResult, rank metrics, paired
│   │                        bootstrap, Pareto front, k-fold
│   ├── prompting.py         grounded prompt construction
│   └── evaluation.py        IoU, recall@k
├── application/             use cases, orchestrating ports
│   ├── ports.py             Protocols: corpus, processor, lexical index,
│   │                        embedder, reranker, index repository,
│   │                        generator, cache, dataset repository,
│   │                        source reader, pipeline factory, trial
│   │                        store, router repository
│   ├── indexing.py          IndexingService (incremental)
│   ├── retrieval.py         Lexical / Semantic / Hybrid retrievers,
│   │                        RerankingRetriever (wraps any of them),
│   │                        PriorAdjuster (routing, test prior, file boost)
│   ├── search.py            SearchService (mode selection + cache)
│   ├── answering.py         AnswerService (retrieve → generate, cache)
│   ├── datasets.py          batch use cases: search/answer/evaluate dataset
│   ├── experiments.py       ExperimentService: run suites, store trials
│   ├── tuning.py            Tuner (cross-validated coordinate descent,
│   │                        consensus), RouterTrainer
│   └── analysis.py          AnalysisService: cost/benefit, interactions,
│                            Pareto fronts, profiles, Markdown report
└── adapters/
    ├── inbound/             cli.py (Fire), lab_cli.py (experiment lab),
    │                        http_api.py (FastAPI)
    └── outbound/            fs_corpus, parallel_processor, bm25_index,
                             minilm_embedder, caching_embedder,
                             cross_encoder_reranker, transformers_generator,
                             jsonl_trial_store, json_router_repository,
                             fs_index_repository, sqlite_cache,
                             json_dataset_repository, fs_source_reader
```

## The experiment lab in the hexagon

The lab is a set of use cases, not a script on the side. It only talks
to ports:

```
lab_cli ──► ExperimentService ──► PipelineFactory ──► Pipeline (index, search)
            Tuner ─┘        └───► TrialStore, DatasetRepository
            AnalysisService ────► TrialStore
```

`bootstrap.LabPipelineFactory` implements `PipelineFactory` by building
one `Container` per parameter set, so a trial exercises exactly the same
code as the CLI. In unit tests the factory is a fake returning scripted
rankings, which lets the tuner and the analysis be tested in
milliseconds. Suites are data (`experiments/suites/*.json`); adding an
experiment needs no code.

A new tunable follows one path: a field in `Settings` (listed in
`INDEX_FIELDS` if it changes the index), its use in an adapter or use
case, and a suite that sweeps it.

## Runtime planning

`domain/runtime.py` holds the decision (hardware profile → device, CPUs,
threads, model) as pure functions; `adapters/outbound/system_probe.py`
reads the machine, `torch_runtime.py` applies the plan when the first
model loads, and `application/runtime_tuning.py` ranks the candidate
plans measured by `subprocess_benchmark.py` (one process per plan).
Both composition roots resolve the plan once and hand it to every model
adapter.

## The soft-skills analyzer on the same hexagon

The analyzer is a second application on the same core, with its own
composition root (`src/skills_bootstrap.py`) and settings
(`SkillsSettings`). Full description: `docs/skills/DESIGN.md`.

```
src/
├── skills_bootstrap.py      SkillsContainer, SkillsLabFactory
├── domain/
│   ├── languages.py         Spanish analyzer, language detection
│   └── skills/              transcript, rubric, assessment, interaction,
│                            grounding, agreement, prompting
├── application/skills/      ports (ChatModel, TranscriptRepository,
│                            CatalogueRepository), evidence, assessment,
│                            gold, evaluation, lab, report
└── adapters/
    ├── inbound/skills_cli.py
    └── outbound/            fs_transcript_repository,
                             json_catalogue_repository,
                             transformers_chat_model, anthropic_chat_model
```

| Reused as is | How |
|---|---|
| `Embedder`, `LexicalIndexBuilder`, `Cache`, `Progress`, `TrialStore` ports and their adapters | Wired by `SkillsContainer` |
| `domain/ranking.py` | Same `bm25/max + w·cos` fusion ranks turns against indicators |
| `ExperimentService`, `Tuner`, `AnalysisService` | `SkillsLabPipeline` implements the `Pipeline` port: a question is a (session, participant, competency), a source is a turn |
| `LabCLI` | Hosted by any container offering `experiments`, `tuner` and `analysis` (`LabApp` protocol) |

Not reused: the file-corpus indexing service (a session has a few
hundred turns and is indexed in memory) and the answer generator (the
judge needs a structured, verifiable reply, not free text).

Still to build, each behind an existing or new port: audio capture and
diarization (an adapter producing the transcript JSON), a database of
sessions and scorecards, the web dashboard (inbound adapter), and
access control.

## Test strategy

- **Unit** (`tests/unit/`): domain functions and application services with
  in-memory fakes of every port. Fast, no disk, no models.
- **Integration** (`tests/integration/`): real adapters on a tiny corpus:
  filesystem, BM25, index round-trip, incremental indexing, SQLite, JSON
  repositories, Fire CLI as a subprocess, FastAPI via TestClient.
- **Slow** (`-m slow`): real MiniLM / Qwen3 smoke tests, excluded by default.
