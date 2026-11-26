==================== FILE A: docs/platform/ARCHITECTURE.md ====================
# Platform: pieces, contracts and three products

Date: 2026-11-21. Status: design, nothing below is built unless marked **exists**.
Rule: extend, do not rewrite. `transcript.v1` remains the post-call contract.

## 1. Goal

One set of pieces, each behind a port, talking only through versioned
contracts (`contracts/`, `docs/platform/CONTRACTS.md`). Three products
are compositions of those pieces, chosen by a config file:

| Product | Input | Output | State |
|---|---|---|---|
| P1 Post-call analysis | recorded audio | HR page, assessment, synthesis | **exists** (`skills analyze`, DESIGN.md §12) |
| P2 Live assist (support calls, job interviews) | live audio, stereo or mono | suggestions on screen ≤3 s after a sentence ends | new |
| P3 Voice chatbot | live mic | spoken answer (ASR → RAG → LLM → TTS) | new |

## 2. Pieces as ports

| Port | Responsibility | Existing code that becomes the adapter | New adapters |
|---|---|---|---|
| AudioSource | frames of PCM 16 kHz, 1–2 channels, with timestamps | browser chunk upload (`transcriber/adapters/inbound/http_api.py`, web recorder), `av_audio_decoder.py` (batch) | `pipewire_source.py` (mic = ch0, call monitor = ch1), `ws_audio_source.py` (browser mic over WebSocket) |
| Vad | speech/non-speech per 32 ms frame, endpoint after N ms silence | none (Whisper only) | `silero_vad.py` (CPU) |
| AsrBatch | words with times for a whole recording | **exists** `SpeechRecogniser` + `whisper_recogniser.py` | — |
| AsrStream | partial and final text of the current segment | — | `faster_whisper_stream.py` (CTranslate2 int8, LocalAgreement-2 for partials) |
| SpeakerAttribution (batch) | speaker per word | **exists** `Diarizer` + `lab_runner_diarizer.py` (DiariZen / Nemotron), voiceprints (`wavlm_voice_encoder.py`) | — |
| SpeakerAttribution (stream) | speaker per final sentence | — | `channel_attribution.py` (channel → speaker, no model); `sortformer_stream_diarizer.py` (mono, ≤4 speakers) + voiceprint naming |
| Segmenter / Acts | sentences, turns, dialogue acts | **exists** `transcriber/domain/sentences.py`, `src/domain/skills/act_*`, `encoder_act_tagger.py` | streaming sentence assembler in `transcriber/domain/live.py` |
| Knowledge | per-domain retrieval with metadata filters | **exists** hybrid BM25+MiniLM (`src/application/search.py`, `bm25_index.py`, `minilm_embedder.py`, `fs_index_repository.py`) | `DomainIndexRegistry`, metadata on `Chunk`, `POST /knowledge/query` |
| Reasoner (roles: judge, suggester, conversational) | prompt by role + context + retrieved chunks → structured answer | **exists** `ChatModel` port with `claude_code_judge.py` (claude -p), `local_judge.py`, `opencode_judge.py`, `anthropic_chat_model.py`; `UsageLedger` | `Reasoner` port wrapping `ChatModel`, role templates, `POST /reason`; optional warm `claude -p --input-format stream-json` backend |
| Tts | text → PCM | Piper in `experiments/skills/render_simulation.py` (own venv) | `piper_tts.py` (subprocess, streaming per sentence) |
| Store | transcripts, analyses, live logs | **exists** `fs_session_repository.py`, `fs_transcript_store.py`, `fs_analysis_store.py` | `jsonl_live_log.py` (events + suggestions, no audio); `fs_recovery_store.py` (audio lost in an outage, until transcribed, #265) |
| UI | pages | **exists** transcriber web (`transcriber/adapters/inbound/web`), HR page | live page (suggestions, script phase coverage, privacy banner), SSE |
| GpuLease | cooperative GPU priority | **exists** `GpuMemory` (`torch_gpu_memory.py`, never kills) | `fs_gpu_lease.py` (§7) |

The reasoner's backend decides where text goes: `claude-code` (subscription),
`anthropic-api`, `local` (Qwen3 on the GPU, only when no live lease is held).

## 3. Products as compositions (config, not code)

`config/products/*.yaml`, validated by `orchestrator check-config`.

| Key | P1 post_call | P2 live_assist.interview | P2 live_assist.support | P3 voicebot |
|---|---|---|---|---|
| source | upload | pipewire stereo (ch0 interviewer, ch1 candidate) | pipewire stereo or ws mono | ws mono (user) |
| vad | — | silero, hang 500 ms | silero, hang 500 ms | silero, hang 400 ms |
| asr | batch whisper-large-v3-turbo | stream faster-whisper turbo int8 | same | same |
| attribution | DiariZen/Nemotron + voiceprints | channels | channels / sortformer-stream | single speaker |
| knowledge.domains | rubric.core | interview.<role> | support.<product> | support.<product> |
| reasoner | judge (model per step, #134) | suggester, haiku | suggester, haiku | conversational, haiku |
| trigger | session end | final from candidate ending a question or turn | final from customer | end of user turn |
| tts | — | — | — | piper es_ES voice |
| output | transcript.v1 → skills analyze → HR page | live page; transcript.v1 at end (optional P1) | live page; transcript.v1 at end | speaker audio; log |
| store_audio | yes (consented) | **no** by default | no by default | no |
| buffer_audio_on_outage (#265) | — | yes, at most 48 h, deleted once transcribed | yes, same | no |

Example:
```yaml
product: live_assist.interview
version: 1
language: es
source: {kind: pipewire, channels: {0: {speaker: S1, role: interviewer}, 1: {speaker: S2, role: candidate}}}
vad: {model: silero-v5, hang_ms: 500}
asr: {kind: stream, model: faster-whisper-large-v3-turbo-int8, partial_every_ms: 800}
attribution: {kind: channels}
knowledge: {domains: [interview.backend], top_k: 4, filters_from: [script_phase]}
reasoner: {role: suggester, model: haiku, template: suggester.interview@1, deadline_ms: 1800, max_inflight: 1}
trigger: {on: final, speaker_roles: [candidate], min_words: 4, debounce_ms: 300, cancel_stale: true}
ui: {kind: live_page}
privacy: {store_audio: false, store_text_days: 30, buffer_audio_on_outage: true, buffer_max_hours: 48, banner: interview_ai_notice@3}
handoff: {transcript_v1: true, run_post_call: false}
```

## 4. Deployment: which process runs what

| Process | Port | Pieces | GPU | New? |
|---|---|---|---|---|
| `python -m transcriber serve` | 127.0.0.1:8770 (+ chisel tunnel) | AudioSource, VAD, ASR batch+stream, attribution, sentence assembly, emits `live.event.v1` (WebSocket `/live/{id}/events`), publishes `transcript.v1` | yes, the only GPU user of the platform | extended |
| `make rag-serve` (src) | 127.0.0.1:8000 | Knowledge per domain, Reasoner (all roles), post-call analysis, HR page | MiniLM on CPU for queries; local judge only without a live lease | extended |
| `python -m orchestrator serve` | 127.0.0.1:8772 | reads product config, subscribes to `live.event.v1`, calls `/knowledge/query` and `/reason`, pushes suggestions (SSE) to the UI, calls Piper | none (no torch) | **new package** |
| Piper | subprocess of orchestrator | TTS | CPU | new |
| `img-worker.service` (IMG project) | ComfyUI on 127.0.0.1:8189 | image/video jobs | most of the card | honours the lease (§7) |

Every call between processes uses loopback HTTP, WebSocket or SSE with the
existing bearer-token pattern. The live page is local first (PC in the
call). Remote use adds one more reverse port to the chisel tunnel later.

## 5. VRAM budget (RTX 3070 Ti, 8192 MiB)

| Piece | Model | MiB | Source |
|---|---|---|---|
| ASR stream | faster-whisper-large-v3-turbo, int8 | 1362–1394 peak | LAB.md catalogue (R1) |
| VAD | Silero v5 | 0 (CPU) | — |
| Stream diarization (mono only) | sortformer-streaming-v2, low-latency setting, worker process | 1140 peak | `docs/live/MONO_DIARIZATION.md` (measured) |
| Voiceprint naming (mono only) | wavlm-base-plus-sv | 0 (CPU, in a thread; on the GPU it measured +1348 with torch's context) | `docs/live/MONO_DIARIZATION.md` |
| Knowledge query | MiniLM | 0 (CPU, ~10 ms per query) | — |
| Reasoner | Claude Haiku via claude -p | 0 (remote) | — |
| TTS | Piper | 0 (CPU) | — |
| CUDA context, transcriber process | — | ~300–500 (estimate) | — |
| **Live, stereo** | | **~2.2 GB** | |
| **Live, mono** | | **~2.5 GB** (1394 + 1140 measured) | |
| Batch post-call (unchanged, sequential stages) | whisper turbo fp16; DiariZen 3742 peak | ≤ ~4 GB per stage | LAB.md |
| IMG ComfyUI | — | up to the whole card (`--reserve-vram 0.6`) | IMG worker.py |

Live mode and IMG cannot share the card, so live takes the lease. Batch
post-call jobs wait while a live lease is held: live > batch.

## 6. Live latency budget (target ≤3 s from end of sentence to suggestion)

"End of sentence" means the last voiced frame of the speaker.

| Stage | Budget ms | How |
|---|---|---|
| Endpoint (VAD silence hang) | 500 | Silero, 32 ms frames; tune 300–700 |
| Final ASR of the segment (≤15 s audio) | 350 | int8 turbo, beam 1, already warm; partials run during speech |
| Attribution | 20 stereo / 150 mono | channel lookup; Sortformer chunk output (speaker may be revised later by `speaker_update`); measured mono 151 ms p90 (`docs/live/MONO_DIARIZATION.md`) |
| Event transport (WS loopback) | 10 | — |
| Knowledge query | 50 | domain index in memory, filters by `script_phase` |
| Reasoner (Haiku via claude -p) | 1800 | measured: stream-json persistent process first token median 0.49 s, done median 0.83 s, p90 1.03 s, max 1.41 s; cold one-shot json median 1.93 s, p90 5.56 s (`docs/live/HAIKU_LATENCY.md`, PR #136). Decision: the warm persistent `--input-format stream-json` process is required; cap output tokens (outliers of 384-797 tokens caused 4-6 s tails); restart or window the process as its context grows |
| Push to UI (SSE) | 50 | — |
| Margin | 220 | — |
| **Total** | **3000** | report p50/p90 per stage in `reasoner.response.v1.latency` + `live.event.v1.timings` |

Policies: at most one reasoner call in flight per session; drop a stale
call when a newer trigger arrives; optional speculative call after 250 ms
of silence on the latest partial. Quota or auth errors show a banner and
stop suggesting. There is no silent fallback to the local model during
live mode (it would compete for the GPU).
P3 target: end of user turn → first audio ≤2.5 s. Reasoner output streams
sentence by sentence into Piper (first sentence p50 93 ms, p90 155 ms on the
CPU). VAD on the mic interrupts playback (barge-in). Use a headset to avoid
echo. Measured on 15 synthetic turns (VB-5, `docs/live/VOICEBOT.md`): end of
the user's voice → first audio at the page p50 1584 ms, p90 1731 ms.

## 7. GPU priority: cooperative lease (`gpu-lease.v1`)

Nothing in this platform ever signals or kills another project's process.

| Step | Live side (transcriber `fs_gpu_lease.py`) | IMG side (change in `IMG/worker/worker.py`) |
|---|---|---|
| 1 | Writes `$XDG_RUNTIME_DIR/gpu-lease/lease.json` atomically (tmp + rename): `{schema, holder: "ragputin-live", priority: "live", pid, acquired_at, expires_at: now+15 s, need_mib, starts_at}` | Before `/api/worker/claim`: if a valid lease exists (not expired, pid alive), skip claiming and sleep 5 s |
| 2 | Heartbeat every 5 s pushes `expires_at`; deletes the file on stop. A crash expires it within 15 s | During `generate` (it already polls every 3 s): if a valid lease appears → `POST /interrupt`, return the job to the server as retryable (new IMG server endpoint or `fail` with `retry: true`), `comfy.stop()` (stops only its own ComfyUI child; the systemd unit keeps running) |
| 3 | Waits ≤20 s for `lease.ack.json` from `img-worker` **and** `nvidia-smi` free ≥ `need_mib` | Writes `lease.ack.json` `{holder: "img-worker", freed: true, at}` |
| 4 | If memory is still short: starts in degraded CPU mode (faster-whisper small int8) and says so in the UI | Resumes 30 s after the lease file disappears |
| Scheduled interview | `starts_at` in the future: IMG may finish a job whose ETA is before `starts_at`, then stops claiming | Compares its job ETA with `starts_at` |

The transcriber's `JobRunner` also honours the lease: no batch job starts
while a live lease is held. Without the IMG change the protocol still
works, only in degraded mode. The IMG change is issue IMG-1 in that
repository.

## 8. Privacy and AI Act (interviews, support calls)

| Topic | Design rule |
|---|---|
| AI Act, recruitment = high-risk (Annex III 4(a)) | The suggester helps the interviewer: next question from the script, uncovered phases, facts from the domain knowledge. It never scores, ranks or labels the candidate live. Post-call assessment of candidates (P1) stays an aid for a trained human (ESTADO §1) and is high-risk: it needs the owner's decision (T-48) before any real use. The same holds for the draft judgement of how accurate the answers were (#259: AI interviewer report and the end of a live-page round with a role's plan), which judges answers, never the person, and is never shown to the candidate; the candidate notice is `interview_ai_notice@3` (`@2` with one sentence changed, #265), which says an analysis may be drafted afterwards and reviewed by a person. Check the current application dates: the Digital Omnibus proposal would delay Annex III obligations |
| Ordering candidates (#271, «Ofertas») | The offer page orders the people interviewed for one position by a draft figure computed from the stored reports (the AI interviewer's capped scores, or fixed values of the live judgement labels; `docs/live/VOICEBOT_INTERVIEWER.md` §6.4). This is ranking candidates with an AI system, a high-risk use: **on the list the lawyer must review** before any real use, together with the legal basis and retention (#151). Safeguards in the product: the order is labelled a draft for a person to review before any decision, the formula is shown on the page («Cómo se ordena»), no cut-off and no verdict words, nobody hidden by default, never shown to candidates. Not legal advice |
| Emotion recognition in the workplace is prohibited (Art. 5(1)(f)); covers candidates and the agents themselves | No prosody, emotion or stress inference from voice for anyone. Interviews: no emotion or state signals about the candidate at all. Support: only *expressed customer signals* from text (§8.1) |
| Transparency | Notice to the candidate or customer before the call (versioned like the existing consent notice); banner on the live page |
| Biometrics (GDPR Art. 9) | Stereo mode needs no voiceprint of the other party. Mono mode uses voiceprints only of enrolled, consenting people |
| Minimisation | Live mode keeps no recording (`store_audio: false`); text kept N days (config); deleting a session deletes the live log and the RAG copy. **Outage buffer (#265)**, only with `privacy.buffer_audio_on_outage` (on for `live_assist.interview` and `live_assist.support`; the transcriber refuses uploads for other products): (1) the browser's IndexedDB keeps audio only while the link is down or more than 3 s wait for their acknowledgement (a healthy session writes nothing), each chunk deleted at its acknowledgement or once uploaded; past `buffer_max_hours` (48 h) after the session started it is deleted by a sweep that runs only while a page of the app is open in that browser (load, return, every 10 min) — if the app is never opened again there, it stays until the browser clears the site's data; (2) on the server, `<data_dir>/live-recovery/<session>/audio/` (files 600, folder 700, git-ignored; never «Sesiones», never published) and one WAV per part in `<data_dir>/live-recovery-batch/` while transcribed: deleted as soon as the transcription succeeds (after the texts and the next state are written), when the job ends or fails, at its expiry (the sooner of upload start + 24 h and session start + `buffer_max_hours`; swept at start and hourly), at once on «Descartar» and when the candidate's file is deleted. Deletions logged as counts. See `docs/live/LIVE_SESSION.md` «Long outages» |
| Notice wording | The sentence about the audio in the notices read to the other person is the owner's decision (2026-11-26): «El audio se borra en cuanto se transcribe.» / "The audio is deleted as soon as it is transcribed." (`interview_ai_notice@3`, `support_ai_notice@2`, `voicebot.notice`), replacing «No se guarda el audio», which was not exact once audio is buffered during an outage. No sentence about outages is added. Legal basis and retention stay the owner's decision (#151); this is not legal advice |
| Third-party processor | `claude -p` sends transcript text to Anthropic under the owner's consumer subscription. Fine for the owner's own prototyping. Using it for other people's calls or in a product probably needs Commercial/API terms and a DPA: **owner decision** (PL-14). The reasoner port makes it a config switch (`anthropic-api`, `local`) |
| Licences | DiariZen CC BY-NC: excluded from P2/P3 commercial configs. Piper (piper1-gpl) GPL-3.0: separate process only. Voices have their own licences. Sortformer CC BY 4.0, faster-whisper MIT, Silero MIT |
| Employment context | Consent of candidates or employees is rarely "freely given": the legal basis must be decided by the owner/legal, not assumed |

### 8.1 Expressed customer signals (support only)

Owner decision 2026-11-21. In `live_assist.support` the suggester also labels
what the **customer says** in the text, never what they feel inside and never
from the voice: `complaint`, `stated_frustration`, `urgency`, `churn_risk`,
`confusion`, `satisfaction`. Each signal must cite the utterance ids it comes
from (`about`), is shown as "el cliente dice…" with the quote, and drives the
suggestion (e.g. third call → offer escalation). The support knowledge domain
gets an `intent`/`tags` vocabulary for these signals so retrieval can filter
by them. Disabled by config in every interview product (test). Voice-based
emotion models are out of scope.

## 9. Open-source stack

| Piece | Choice | Alternative kept in the lab |
|---|---|---|
| VAD | Silero VAD v5 (MIT, CPU) | WebRTC VAD |
| Stream ASR | faster-whisper large-v3-turbo int8 + LocalAgreement (whisper_streaming style) | parakeet-tdt-0.6b-v3 (6.3 GB, too big for live) |
| Batch ASR | whisper-large-v3-turbo (exists) | — |
| Stream diarization | NVIDIA streaming Sortformer v2 (≤4 speakers) | diart (pyannote, gated models) |
| Batch diarization | DiariZen / Nemotron (exists) | pyannote-community-1 |
| Retrieval | existing BM25 + MiniLM hybrid | multilingual-e5-small if Spanish recall is low |
| Reasoner | Claude Haiku via claude -p | Anthropic API; local Qwen3-4B 4-bit |
| TTS | Piper 1.8 (own venv) | — |
| Transport | WebSocket (audio, events), SSE (UI) | — |

## 10. Folder additions

```
contracts/  live.event.v1, knowledge.query.v1, knowledge.result.v1,
            reasoner.request.v1, reasoner.response.v1, tts.request.v1,
            gpu-lease.v1 (.schema.json) + examples/
config/products/  post_call.yaml, live_assist.interview.yaml, live_assist.support.yaml, voicebot.yaml
knowledge/domains/<domain>/  manifest.yaml (vocabularies: intent, script_phase) + *.md (front matter) / *.jsonl
src/domain/reasoning/  roles.py, templates.py, suggestion_grounding.py
src/domain/knowledge/  domains.py (manifest, filters)
src/application/reasoning.py, src/application/knowledge.py
src/adapters/outbound/domain_index_registry.py, claude_code_stream.py (warm process, optional)
transcriber/domain/live.py           endpointing, sentence assembly, channel mapping
transcriber/application/live.py      live session use case
transcriber/adapters/outbound/silero_vad.py, faster_whisper_stream.py, sortformer_stream_diarizer.py, pipewire_source.py, fs_gpu_lease.py
transcriber/adapters/inbound/live_ws.py
orchestrator/  domain/ (composition, triggers) application/ (live_assist.py, voicebot.py, replay.py)
               adapters/outbound/ (http_knowledge.py, http_reasoner.py, ws_live_events.py, piper_tts.py, jsonl_live_log.py)
               adapters/inbound/ (cli.py, http_api.py, web/)
docs/platform/  ARCHITECTURE.md, CONTRACTS.md
```
The import-ban test covers `src`, `transcriber` and `orchestrator` pairwise.
Dependency group `orchestrator`: no torch.

## 11. What is NOT done (and not planned now)

- No rewrite of the batch transcriber or the post-call analyzer. Live mode only adds paths.
- No live scoring, ranking or emotion inference of any person.
- No telephony integration (SIP, contact-centre APIs). Audio comes from the PC (PipeWire) or the browser.
- No more than 4 speakers in mono live mode (Sortformer limit). No overlapping-speech separation.
- No paid API by default. No local reasoner during live mode.
- No remote access to the live page in stage 2 (local first).
- No measured latency or streaming accuracy yet: every budget above is a target until LV-12/LV-13.
- No `transcript.v2`: channel and partial data stay out of the post-call contract.
