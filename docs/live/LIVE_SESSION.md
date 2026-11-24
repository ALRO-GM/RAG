# Live sessions: socket audio, roles, events, GPU lease, hand-off (LV-2, LV-5, LV-7, LV-8, LV-11)

Design: `docs/platform/ARCHITECTURE.md` §2–§8; events: `live.event.v1`
(`docs/platform/CONTRACTS.md` §1); the recogniser and endpointing are
those of `docs/live/LIVE_ASR.md`.

## API (transcriber, `python -m transcriber serve`)

Every route needs the access code. HTTP routes take it in the
`X-Access-Code` header. A WebSocket takes it in that header or, from a
browser (which cannot set headers), in its first message
`{"type": "auth", "access_code": "..."}` within 5 s. Wrong codes are
throttled like every other route.

| Route | Body / messages | Answer |
|---|---|---|
| `POST /api/live/sessions` | `{"session_id", "channels": 1\|2, "speakers": [{"channel", "id", "role", "name"?}], "product"?, "handoff": {"transcript_v1", "run_post_call"}, "language": "es"\|null, "hang_ms": 300–700, "partial_every_ms", "diarize"?, "enrolled"?, "parties"?}` | 201 session (`state`, `device`, `model`, `speakers`, `audio`, `events`); 409 `wrong_state` (id in use) or `busy` (one session at a time); 422 invalid body |
| `GET /api/live/sessions`, `GET /api/live/sessions/{id}` | — | the sessions / one session, with `finals`, `rejected_events`, `transcript` (`utterances`, `published`, `analysis`, `error`) |
| `POST /api/live/sessions/{id}/stop` | — | the session after its end |
| `POST /api/live/sessions/{id}/speakers/{speaker}` | `{"name"}` (1–80 characters) | the session; the hand-off's `transcript.v1` lists the name (LV-16); 409 once the session ended |
| `POST /api/live/sessions/{id}/voice` | `{"action": "skip"}` or `{"action": "mine" \| "not_mine", "utterance": "t3.s0"}` (#249) | the session; 409 once ended or for a session without `parties`; 400/422 for an unknown action or a missing utterance |
| `WS /api/live/{id}/audio` | once the recogniser is loaded (GPU placement and model warm-up: about 6 s, `experiments/live/start_delay.py`) the server sends `{"type": "ready", "channels", "sample_rate": 16000, "format": "pcm_s16le", "max_message_bytes"}`; client sends binary PCM s16le 16 kHz, channels interleaved, ≤1 s per message; text `{"type": "stop"}` ends the session (`{"type": "stopped"}` back) | one source per session at a time |
| `WS /api/live/{id}/events` (alias `/live/{id}/events`), `?from_seq=n` | server sends every `live.event.v1` event from `seq` n as one JSON text message, the past ones first; closes with 1000 after the last | — |

Socket refusals: an `{"type": "error", "code", "message"}` message, then
close code 4400 (invalid input), 4401 (no valid code), 4404 (unknown
session), 4409 (wrong state: ended, or a second audio source) or 4429
(too many wrong codes).

## Behaviour

- **Roles (LV-5).** Channel `n` is `speakers[n]`: id and role come from
  the product config (`source.channels` of `config/products/*.yaml`),
  which the orchestrator reads and sends; the transcriber never reads
  that config. Roles are the contract's enum. Without `speakers`:
  `S1`/`S2`, role `unknown`.
- **Backpressure (LV-2).** At most `live.queue_seconds` (10 s) of audio
  wait for the recogniser; a full queue blocks the socket's reader, so
  the browser's `bufferedAmount` grows. 3 s waiting gives an
  `asr_behind` warning; a queue full for 30 s refuses the audio. A
  session with no audio for 120 s stops by itself.
- **Events and log (LV-7).** Every event is checked against
  `contracts/live.event.v1.schema.json` (jsonschema) before it is sent;
  one that fails is logged and dropped (`rejected_events`). Events are
  written to `<data_dir>/live/<id>/events.jsonl` (mode 600, directory
  700). No audio is ever written.
- **GPU (LV-8).** On the real runtime the session writes the GPU lease
  (`need_mib` 2200, §5 stereo), then polls every 0.5 s: GPU when the
  image worker's ack is there and `nvidia-smi` reports ≥ `need_mib`
  free; without an ack, enough free memory after 4 s also counts (no
  worker running). After 20 s, with a lease held by another process, or
  with no GPU reading, it releases the lease, runs faster-whisper
  `small` int8 on the CPU and emits a `warning` `gpu_degraded`. The
  lease is held until the session ends.
- **Mono diarization (LV-6).** With `channels: 1` and `diarize: true`
  the voices of the one channel are told apart by streaming Sortformer
  v2 and named by the voiceprints of `enrolled` people (every enrolled
  person when absent); `speakers` (all on channel 0, at most four) give
  the ids and roles of the voices in the order they are first heard.
  Source kind `mono_diarized`; finals may be `speaker_provisional` and
  are revised by `speaker_update` events. The lease asks for
  `need_mib` + `diarize_need_mib` (1500). See `MONO_DIARIZATION.md`.
- **Two parties (#249).** `parties: {"employee", "other", "voice_check"}`
  (diarized only; both ids among `speakers`) makes every sentence one of
  the two: the roster lists only them, the first sentences are the
  employee's voice check (`voice_check` events: `listening` with the
  seconds heard, then `recognised`, `short`, `skipped` or `failed`),
  sentences whose side is not known from their voice yet carry
  `speaker_unsure: true`, and `POST .../voice` skips the check or
  corrects a voice. The check's audio and its reference stay in memory
  only. The attribution's numbers (ids, times, slots, cosines,
  decisions; no text) are written to `<data_dir>/live/<id>/attribution.jsonl`
  (mode 600) at the end. See `MONO_DIARIZATION.md` §"Voice check".
- **Hand-off (LV-11).** At the end the last revision of each final
  becomes a `transcript.v1` (empty sentences dropped, turns regrouped,
  ids renumbered without gaps, no audio; each speaker keeps its role
  when it is not `unknown`, #235), kept as
  `<data_dir>/live/<id>/transcript.json`. With `handoff.transcript_v1`
  it is published like a batch transcript; with `run_post_call` the
  RAG's post-call analysis is then started. Failures are recorded in
  the session's `transcript.error`, not raised.

Not done here: keeping the audio for a batch
re-run, the live page itself.
