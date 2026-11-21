# `POST /knowledge/query` latency (PL-8)

Date: 2026-11-21. Command (from an empty folder, CPU only):

```sh
PYTHONPATH=<repo> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python <repo>/experiments/knowledge/query_latency.py <folder> 45 \
  <repo>/contracts/examples/knowledge.result.v1.json
```

Synthetic domain `interview.backend`: 45 Markdown guides + 40 JSONL
questions, **1030 chunks**, embedded with the knowledge embedder
(`paraphrase-multilingual-MiniLM-L12-v2`, 128 tokens) on the CPU
(8 performance cores, no GPU visible). Queries of six random words,
`top_k` 4, hybrid mode, through the FastAPI app in process (bearer
token, JSON in and out; no network socket). The workstation was busy
with other work (load average 7.9 before the run, 9.1 after).

| Queries | n | p50 | p90 | max |
|---|---|---|---|---|
| no filter | 200 | 10.6 ms | 80.0 ms | 751.1 ms |
| filter `script_phase: [technical]` | 200 | 11.3 ms | 14.0 ms | 182.9 ms |

Indexing the 1030 chunks took 21.8 s on the CPU.

The p50 meets the < 50 ms target. The tail comes from the shared CPU:
an earlier exploratory run of the same script with 90 guides (2020
chunks) while the machine was more loaded (load average about 10)
gave p50 122.1 ms, p90 342.6 ms; a direct call of the use case (no
HTTP) on the 1030-chunk domain gave p50 6.5 ms, of which 5.4 ms is
embedding the query. The running RAG uses the GPU for that embedding.
The text is synthetic, so these numbers say nothing about retrieval
quality. `contracts/examples/knowledge.result.v1.json` is the answer
of the run above to the contract's example query.
