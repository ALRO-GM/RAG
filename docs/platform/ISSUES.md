==================== C) GitHub issues ====================
Labels to create first: `type:feature`, `area:platform`, `area:live` (or map to `type:core`/`type:api` and `area:rag`). Sizes: S ≤1 d, M 1–2 d.

Stage 1: contracts, reasoner, knowledge per domain, orchestrator config

| # | Title | Labels | Size | Acceptance criteria |
|---|---|---|---|---|
| PL-1 | Platform contracts: live.event, knowledge, reasoner, tts, gpu-lease schemas + examples | type:feature, area:platform | S | 7 schemas in `contracts/` + one valid example each; contract tests validate the examples and reject one invalid case each; `contracts/README.md` table updated; `docs/platform/*` merged |
| PL-2 | Reasoner port with roles (judge, suggester, conversational) in src | type:feature, area:platform, area:skills | M | `src/domain/reasoning` models mirror the request/response schemas; `Reason` use case wraps any `ChatModel`; judge role delegates to the existing confirm path with identical output (regression test); unit tests with a fake ChatModel |
| PL-3 | Claude Code backend: haiku alias, latency breakdown, usage per role | type:feature, area:platform | S | `ClaudeCodeJudge` reports `process_start_ms`/`total_ms`; usage recorded per role in `UsageLedger`; quota/auth map to `status`; no transcript text in logs (test) |
| PL-4 | Suggester templates v1 (interview, support) + grounding | type:feature, area:platform | M | `suggester.interview@1`, `suggester.support@1`; grounding drops suggestions citing unknown ids; template forbids ratings/emotions, and a test with an adversarial window shows no rating text; ≤120 output tokens |
| PL-5 | Optional metadata on Chunk + metadata filters in search | type:feature, area:platform | M | `Chunk.metadata` defaults to empty; filters per §2 semantics; 42 recall@5 and BEIR nDCG@10 unchanged (re-run numbers in PR) |
| PL-6 | Domain index registry | type:feature, area:platform | M | `knowledge/domains/<d>/manifest.yaml`; index under `data/processed/domains/<d>`; `rag domain index <d>` / `rag domain list`; `index_version` hash; two domains load side by side |
| PL-7 | Ingestion with metadata (Markdown front matter + JSONL) | type:feature, area:platform | M | values outside the manifest vocabularies are rejected with file:line; incremental re-index by hash; tests |
| PL-8 | HTTP `POST /knowledge/query` and `POST /reason` | type:feature, area:platform | S | bearer guard; loopback only; contract-validated in/out; p50 of `/knowledge/query` <50 ms on a 1k-chunk domain (measured in PR) |
| PL-9 | Seed domains interview.generic and support.demo + retrieval eval | type:experiment, area:platform | M | ≥60 chunks each with phases/intents; 20 labelled queries per domain; recall@5 with and without filters reported in `experiments/platform/` |
| PL-10 | orchestrator package skeleton + product config schema | type:feature, area:platform | M | `orchestrator/` hexagonal, no torch; pydantic model of §3; `orchestrator check-config` validates the 4 YAMLs; import-ban test covers 3 packages pairwise |
| PL-11 | Replay mode: transcript.v1 → live.event.v1 stream → suggestions | type:feature, area:platform | M | `orchestrator replay <transcript> --product …` emits events with real timing; calls knowledge + reasoner; writes suggestions JSONL with per-stage latency; works on the Piper simulation sessions |
| PL-12 | GPU lease adapter in transcriber (acquire, heartbeat, release, status CLI) | type:feature, area:transcriber | S | file per gpu-lease.v1, atomic writes; expires after crash (test with killed holder); `transcriber gpu-lease status`; JobRunner defers batch jobs while a live lease is held |
| IMG-1 (IMG repo) | img-worker honours gpu-lease.v1 | type:feature | S | no claim while a lease is valid; mid-job: interrupt, requeue, stop ComfyUI child, write ack; resume 30 s after release; the systemd unit is never stopped |
| PL-13 | Reasoner latency decision (consume the separate Haiku measurement) | type:experiment, area:platform | S | p50/p90 of cold `claude -p` vs warm stream-json on the PL-11 replay; decision recorded: is a warm backend needed to meet 1.8 s. ANSWERED by PR #136 (`docs/live/HAIKU_LATENCY.md`): warm stream-json first token median 0.49 s, done median 0.83 s, p90 1.03 s, max 1.41 s vs cold json median 1.93 s, p90 5.56 s; warm process required, cap output tokens, restart/window as context grows |
| PL-14 | Owner decisions: interview use, processor terms, retention | type:docs, area:platform, needs:owner | S | written decisions on: legal basis for candidates; subscription vs API terms for third-party data; retention days; links T-48 |
| PL-15 | Docs: ESTADO + ARCHITECTURE link the platform layer | type:docs, area:platform | S | ESTADO gets rows for P2/P3 with certainty "sin medir" |

Stage 2: live assist (headlines)
- LV-1 PipeWire stereo source (mic ch0, call monitor ch1). area:live, M
- LV-2 Browser mic WebSocket source. area:live, S
- LV-3 Silero VAD + endpointing in `transcriber/domain/live.py`. area:live, S
- LV-4 faster-whisper streaming ASR (partials + finals, LocalAgreement). area:live, M
- LV-5 Channel attribution + speaker roles. area:live, S
- LV-6 Streaming Sortformer v2 + voiceprint naming (mono). area:live, M
- LV-7 live.event.v1 WebSocket emitter + live log. area:live, S
- LV-8 Live session takes the GPU lease; CPU degraded mode. area:live, S
- LV-9 Orchestrator live loop: triggers, debounce, cancel-stale, one call in flight. area:live, M
- LV-10 Live page: suggestions, phase coverage, AI notice banner. type:ui, area:live, M
- LV-11 Session end → transcript.v1 (from finals or batch re-run) → optional P1. area:live, S
- LV-12 Experiment: end-to-end ≤3 s p50/p90 on Piper-rendered calls. type:experiment, M
- LV-13 Experiment: streaming WER/DER vs batch (simulation, CALLHOME). type:experiment, M

Stage 3: voice chatbot (headlines)
- VB-1 Piper TTS adapter + tts.request.v1 (streaming per sentence). S
- VB-2 Conversational template + sentence-chunked reasoner streaming. M
- VB-3 Barge-in and echo handling. M
- VB-4 Voicebot loop in orchestrator (config `voicebot.yaml`). S
