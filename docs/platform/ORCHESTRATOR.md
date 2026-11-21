# Orchestrator (PL-10, PL-11)

Package `orchestrator/`, hexagonal like `src/` and `transcriber/`; no torch.
Dependency group `orchestrator` (pyyaml, httpx). Design: `ARCHITECTURE.md` §3, §4, §10.

```
python -m orchestrator check-config [--dir config/products]
python -m orchestrator replay <transcript.v1.json> --product live_assist.support \
    [--speed 1|N|max] [--out F.jsonl] [--events-out E.jsonl] [--roles a=customer,b=agent] \
    [--knowledge fake|http] [--reasoner fake|http|module:factory]
```

- `domain/config.py` is the pydantic model of the product YAML (`config/products/*.yaml`). Rules checked: live products need vad, trigger and a streaming ASR; one call in flight; no audio kept live; expressed customer signals only in `live_assist.support`.
- `domain/triggers.py` decides which event asks for a suggestion: speaker role, min words, debounce (stream time), `final` or `turn_end`.
- `application/live_assist.py`: one call in flight; a newer trigger cancels the stale call (its answer is dropped, status `cancelled`) or, with `cancel_stale: false`, is skipped. Each call writes a JSONL record with `stages.knowledge_ms`, `reasoner_ms`, `total_ms` and the reasoner's own latency.
- `application/replay.py`: transcript.v1 to `live.event.v1` (finals stamped at sentence end, `turn_end` on speaker change). `--speed 1` is real time; `max` never waits and runs each call to the end.
- Ports: knowledge and reasoner exchange the contract JSON documents. Adapters: in-process fakes, `http_knowledge.py` (`POST /knowledge/query`, on branch feat/knowledge-domains), `http_reasoner.py` (`POST /reason`, issue #157, not built yet).

## In-process reasoner and the import ban

`orchestrator` may not import `src` (the ban test covers the three packages pairwise), so there is no `--inprocess` adapter inside the package. `--reasoner module:factory` loads a factory by name instead; `experiments/live/orchestrator_reasoner.py:build` wraps the RAG's `Reasoner` over `claude -p` and lives outside both packages. It goes away once `POST /reason` exists.

## Roles in a replay

A transcript has no roles. Speakers take the roles of the product's `source.channels` by order (the first speaker the first role, any further speaker the last one), or `--roles`.

## Smoke run (measured, 2026-11-21)

8 real `claude -p --model haiku` calls (cold one-shot path, fake knowledge with no chunks) on the first 25 utterances of `sim-reunion-verano`, `--product live_assist.support --speed max`: all 8 `ok`; reasoner p50 3802 ms, p90 4406 ms (the warm stream-json backend of `ARCHITECTURE.md` §6 is not used here, so this is not the live budget). Skipped triggers: 13 by role, 4 by min_words.
