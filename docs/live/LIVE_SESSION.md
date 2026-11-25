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
| `WS /api/live/{id}/audio` | once the recogniser is loaded (GPU placement and model warm-up: about 6 s, `experiments/live/start_delay.py`) the server sends `{"type": "ready", "channels", "sample_rate": 16000, "format": "pcm_s16le", "max_message_bytes", "received_samples"}`; client sends binary PCM s16le 16 kHz, channels interleaved, ≤1 s per message; text `{"type": "stop"}` ends the session (`{"type": "stopped"}` back). Resume (#263): `{"type": "ack", "samples": N}` about every second; `{"type": "gap", "samples": k}` and `{"type": "align", "held": n}` from the client (see «Connection loss» below) | one source per session at a time: a newer socket takes it over |
| `WS /api/live/{id}/events` (alias `/live/{id}/events`), `?from_seq=n` | server sends every `live.event.v1` event from `seq` n as one JSON text message, the past ones first; closes with 1000 after the last | — |

Socket refusals: an `{"type": "error", "code", "message"}` message, then
close code 4400 (invalid input), 4401 (no valid code), 4404 (unknown
session), 4409 (wrong state: ended; or code `replaced`, a newer socket
took the audio source over) or 4429 (too many wrong codes).

## Connection loss (#263)

The public tunnel resets for 1–3 s about once an hour; every socket
through it dies. A short outage must cost nothing:

| What | Where | How |
|---|---|---|
| Offsets | `LiveSession.received` | samples (per channel) the session took, silence included; sent as `received_samples` in `ready` and in `GET /api/live/sessions/{id}` |
| Ack | `live_ws.ACK_SECONDS` = 1 s | after a message, when new samples were taken and 1 s passed: `{"type": "ack", "samples": N}`; the page forgets everything before N |
| Outbox | `js/lib/live-link.js` `Outbox` | every 100 ms piece with its absolute offset, in the page's memory only, bounded to 30 min (`MAX_OUTBOX_SECONDS`, about 58 MB); the oldest goes first when full |
| Reconnect | `js/live-audio.js` | backoff 0.5, 1, 2, 4, 8, 8… s (±20 % jitter), reset once `ready`; until stopped, or the server says the session ended (`not_found`, `wrong_state`), was taken over (`replaced`) or refuses the code; a socket that never says ready is closed after 30 s, one that sends nothing for 20 s while audio waits for its ack is closed and retried |
| Resend | `ready.received_samples` | the page sends from that sample on (the first piece cut at the exact sample): nothing lost, nothing twice |
| Behind | `bufferedAmount` > 10 s | sending pauses (checked again every 250 ms); nothing is dropped |
| Gap | `{"type": "gap", "samples": k}` | audio the full outbox let go becomes `k` samples of silence on the server, so stream time stays on the wall clock (at most 4 h per message) |
| Reattach | `{"type": "align", "held": n}` | a page that reloaded (first `ready` with `received_samples` > 0) has no outbox: the server fills the time nobody sent with silence, leaving room for the `n` samples the page holds |
| Takeover | `claim_source` | a newer socket takes the source; the older one's audio is refused (`replaced`, 4409), so the server never takes audio twice even if it has not noticed the old socket died |
| Grace | `live.source_grace_seconds` = 600 | a source that goes away without `stop` leaves the session waiting 10 min (instead of the 120 s idle stop) with a `warning` `source_lost`; a returning source gives `source_back` with `warning.seconds`, the time away; past the grace the session ends (`end_reason: source_lost`) |
| Late audio | `live.catch_up_seconds` = 3 | resent audio is read as fast as the recogniser goes (the live engine is not paced); while it is more than 3 s late no partials are made; finals carry the real `timings.lag_ms` |
| Suggestions | `orchestrator/domain/triggers.py` `LATE_MS` = 10 s | a final heard more than 10 s late asks for no suggestion (skip reason `late`); topics, concepts, mandatory items and avoid hints still take it |

The page (`#/live`) shows one line under the header (`role="status"`,
text, not colour): «Sin conexión: la entrevista se sigue grabando en
este navegador (N s pendientes)» while the audio socket is down, then
«Conexión recuperada: enviando lo pendiente» while more than 2 s wait
for their ack, then nothing. The stream of suggestions (SSE) says
«Se cortó la conexión; reconectando…» only until it answers again; a
404 means the session ended (no endless retry); no byte for 40 s
(keepalives come every 15 s) reconnects it. The running session's id
is kept in `sessionStorage`: a reload, or coming back to `#/live`,
reattaches to it (stream from the start, audio reconnects) instead of
the start form.

The audio held during an outage lives only in the page's memory and is
dropped as soon as it is acknowledged, so the notice «No se guarda el
audio» stays true. A closed tab, a crashed browser or an outage longer
than the grace still lose that audio (persistent local recording is a
follow-up).

The transcriber's own log (`transcriber serve`'s output) now has one
line per live session start, end with its reason (`stop`, `idle`,
`source_lost`, `failed`) and seconds, source connected / lost / back,
audio socket closed with its code, and silence put in: ids, codes and
seconds, never words or audio.

Not measured (the GPU was busy; everything above was tested with the
fake runtime, fake sockets and fake timers): the catch-up speed after a
60 s or 5 min outage on the real recogniser, the CPU-degraded mode
under catch-up, and the behaviour through the real tunnel.

## Behaviour

- **Roles (LV-5).** Channel `n` is `speakers[n]`: id and role come from
  the product config (`source.channels` of `config/products/*.yaml`),
  which the orchestrator reads and sends; the transcriber never reads
  that config. Roles are the contract's enum. Without `speakers`:
  `S1`/`S2`, role `unknown`.
- **Backpressure (LV-2).** At most `live.queue_seconds` (10 s) of audio
  wait for the recogniser; a full queue blocks the socket's reader, so
  the browser's `bufferedAmount` grows. 3 s waiting gives an
  `asr_behind` warning. A connected source is never refused for a full
  queue (#263; before, 30 s full closed it with 4409): the browser
  keeps the audio and pauses. A session with no audio for 120 s
  (`live.idle_seconds`) stops by itself, unless its source was lost
  (below).
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
