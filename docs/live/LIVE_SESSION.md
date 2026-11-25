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
| Ack | `live_ws.ACK_SECONDS` = 1 s | `{"type": "ack", "samples": N}` when new samples were taken and 1 s passed, after a message or, when the page sends nothing, from a timer (the last audio is acknowledged too); the page forgets everything before N |
| Outbox | `js/lib/live-link.js` `Outbox` | every 100 ms piece with its absolute offset, in the page's memory only, bounded to 30 min (`MAX_OUTBOX_SECONDS`, about 58 MB); the oldest goes first when full |
| Reconnect | `js/live-audio.js` | backoff 0.5, 1, 2, 4, 8, 8… s (±20 % jitter), reset once `ready`; until stopped, or the server says the session ended (`not_found`, `wrong_state`), was taken over (`replaced`) or refuses the code; a socket that never says ready is closed after 30 s, one that sends nothing for 20 s while audio waits for its ack is closed and retried; `stop` closes a socket still connecting and nothing re-arms |
| Resend | `ready.received_samples` | the page sends from that sample on (the first piece cut at the exact sample): nothing lost, nothing twice, whatever was in flight when the link dropped |
| Behind | `bufferedAmount` > 10 s | sending pauses (checked again every 250 ms); nothing is dropped |
| Gap | `{"type": "gap", "samples": k}` | audio the full outbox let go, or time the microphone produced nothing for (over 2 s: a suspended tab, a laptop asleep), becomes `k` samples of silence on the server, so stream time stays on the wall clock. The page counts in total: the time due since its first piece on the monotonic clock (`performance.now()`), plus the time it stood still while the computer slept (the wall clock ran more than 2 s past it between two pieces), against every sample captured or filled. A deficit over 2 s is only suspected at first: a busy page gets the microphone's queued pieces late, in a burst that makes it up, so the pieces from there wait up to 0.5 s; what is still missing then goes as silence before them, and nothing when the burst made it up. The server puts in at most what keeps the stream within 120 s of its own clock (`gap_room`) and, when it cut a gap short, answers `{"type": "gap_clamped", "at", "asked", "samples"}`: the page moves what follows back by the difference and goes on (never fatal) |
| Caught up | `{"type": "caught_up"}` | sent once per connection when everything held has been sent; until then a returned source's audio is a resend |
| Reattach | `{"type": "align", "held": n}` | a page that reloaded (first `ready` with `received_samples` > 0) has no outbox: the server fills the time nobody sent with silence, leaving room for the `n` samples the page holds |
| Takeover | `claim_source` | a newer socket takes the source; the older one's audio is refused (`replaced`, 4409), so the server never takes audio twice even if it has not noticed the old socket died. The same holds for a second tab on the same session: the newest tab's microphone wins and the other tab stops sending and says «Esta sesión está enviando el audio desde otra pestaña o ventana.» (it does not retry). The page obtains its microphone before it opens the socket, so a refused microphone never takes another browser's place; `GET /api/live/sessions/{id}` says `source`: `connected`, `lost` or `none` |
| Grace | `live.source_grace_seconds` = 600, `live.source_gone_seconds` = 120 | a source that goes away without `stop` leaves the session waiting 10 min (instead of the 120 s idle stop) with a `warning` `source_lost`; 2 min when its socket closed with 1001 «going away» (a closed tab). Which close codes the public tunnel really delivers on a reset was not checked: an abnormal close (1006) gets the long grace. A returning source gives `source_back` with `warning.seconds`; past the grace the session ends (`end_reason: source_lost`) |
| Catch-up | worker, `engine.skip_partials` | after `source_back` the resent audio is read as fast as the recogniser goes (the live engine is not paced) with no partials until the queue is empty; its finals come with the real `timings.lag_ms`. Lateness alone never turns partials off (a CPU-degraded session keeps them) |
| Re-anchor | `PaceWatch`, `LiveEngine.realign` | when 30 pieces in a row arrive at real-time pace (each 0.9–1.1 times its own length after the one before) with nothing queued, and the timings think the audio is more than 1 s older than it is (a microphone pause the page did not report), the origin of the timings moves forward: `lag_ms` is true again, so partials and suggestions come back. Never between a `source_back` and the page's `caught_up`: a resend of held audio, even one that happens to come at about real-time pace, keeps its true lateness (and asks for no suggestion) |
| Suggestions | `orchestrator/domain/triggers.py` `LATE_MS` = 10 s | a final heard more than 10 s late asks for no suggestion (skip reason `late`); topics, concepts, mandatory items and avoid hints still take it |
| Stop | `live.stop_drain_seconds` = 15 | a stop asked over HTTP lets a connected source send what it holds for up to 15 s (until its own `stop` or its socket closes) instead of refusing that audio; then it waits up to 60 s for the end, so the orchestrator's client waits 90 s for a stop (`STOP_TIMEOUT`; other calls keep 40 s) |

The page (`#/live`) shows one line under the header, text not colour:
a sentence in a `role="status"` region that changes only with the
state (a screen reader hears it once) and the seconds beside it,
outside the region. «Sin conexión: la entrevista se sigue grabando en
este navegador» «(N s pendientes)» while the audio socket is down, then
«Conexión recuperada: enviando lo pendiente» while more than 2 s wait
for their ack, then nothing. While it shows, the connection watch's
own lost / back line steps aside (one line only). The stream of
suggestions (SSE) says «Se cortó la conexión; reconectando…» only
until it answers again; a 404 means the session ended (no endless
retry); no byte for 40 s (keepalives come every 15 s) reconnects it.
The running session's id, audio kind and page are kept in
`sessionStorage`: a reload, or coming back to `#/live` (or to the
voice assistant), reattaches to it (stream from the start, audio
reconnects) instead of the start form.

«Terminar» first stops the microphone and waits until the server has
acknowledged everything still held: «Enviando lo pendiente antes de
terminar…» «(N s pendientes)», or with the socket down «Sin conexión:
lo pendiente se enviará en cuanto vuelva la conexión. Puedes esperar o
terminar ya.», with a button «Terminar ya: se pierden N s». The wait
counts at most 60 s of connected time (the time offline is not
counted: the page waits for the link to come back). Then the page
says stop, waits for the server's «stopped» (at most 5 s) and stops the
session; a toast says how many seconds never arrived, if any. The voice
assistant's «Terminar» still stops at once.

When the one session slot is taken (a tab closed during a session
leaves it running for its grace, and a new tab has nothing to
reattach to), the start forms of «En directo» and of the voice
assistant say «Ya hay otra sesión en directo en marcha» with
«Volver a la sesión» (reattach; on the other page when the session
belongs to it) and «Terminarla» (asks first; then the new one can
start). When the running session's microphone is still connected from
another browser, «Volver a la sesión» asks first: «Esta sesión recibe
audio desde otro navegador; si vuelves aquí, este micrófono lo
sustituye.» A page that goes back to a session it did not start never
ends it when its own microphone fails («No se pudo usar el micrófono en
esta página; la sesión sigue en marcha.»).

The audio held during an outage lives only in the page's memory and is
dropped as soon as it is acknowledged, so the notice «No se guarda el
audio» stays true. Still lost: a closed tab, a crashed browser or a
reload (the page's memory goes with it; the reattached page fills that
time with silence), an outage longer than the grace, audio the 30 min
outbox had to let go, what «Terminar ya» or the 60 s cap gives up, and
whatever the voice assistant held at its «Terminar» (persistent local
recording is a follow-up).

The transcriber's own log (`transcriber serve`'s output) has one line
per live session start, end with its reason (`stop`, `idle`,
`source_lost`, `failed`) and seconds, source connected / lost (with
the close code) / back, audio socket closed, silence put in, and
timings re-anchored: ids, codes and seconds, never words or audio.

Not measured (the GPU was busy; everything above was tested with the
fake runtime, fake sockets and fake timers): the catch-up speed after a
60 s or 5 min outage on the real recogniser, the CPU-degraded mode
under catch-up, the close codes and behaviour through the real tunnel,
and a real browser's capture during a suspended tab or a sleep (whether
`performance.now()` stands still). The thresholds (2 s pause, 30 pieces
at 0.9–1.1× for re-anchoring, 120 s gap slack) are chosen, not measured.

## Nothing lost at the end or on a crash (#264)

| What happens | Now | Recovered | Still lost |
|---|---|---|---|
| The live worker dies with an exception (CUDA out of memory, any error) | the readers get a `warning` `other` («The live session stopped on an error (…); the transcript of what was heard until then is kept.»); the transcript of every final so far is kept and published as the start request asked; the session's `transcript` summary and its `session.json` say `partial: true`; state `failed`, `end_reason: failed`. The published `transcript.v1` itself does not say it is partial: the contract has no field for it and the RAG validates it strictly, so the post-call analysis cannot tell (left for a contract change) | everything heard until the error | the audio after the error (the page's socket is refused; it reconnects only to a new session) |
| The transcriber stops mid-session (crash, kill, restart) | each session keeps `session.json` (request, speakers, names, model, state, and at the end the reason and the publish's result). At the next start, sessions with that record, newest first and at most 20 per start (the rest are counted in the log and wait), are handed over: one without `transcript.json` is rebuilt from `events.jsonl`, kept, published as asked and listed as `recovered` (`end_reason: restart`); one whose transcript was kept but not published (the process died while the RAG got it, or the RAG refused it) is published again (the RAG replaces a transcript of the same session). Publishing runs on a thread, so the start is not delayed. A session whose log or record cannot be used is logged (its id and the error's type) and left as it is; recovery never stops a start. Sessions logged before #264 (no record) are never touched | the transcript up to the last event written (events are flushed one by one); a publish that did not complete | the audio after the crash; the orchestrator's live view of that round (see below) |
| The orchestrator stops mid-session | the page sees a 404 on its stream (#263) and ends; the transcriber's session goes on until its audio stops, then hands its transcript over as usual | the round's topics, mandatory items, lines to avoid and judgement with `python -m orchestrator rebuild-round` (below) | the suggestions shown during the round (they were advice of the moment) |
| The event stream between transcriber and orchestrator drops (a close code other than 1000, or no close frame at all: a reset, a ping time-out) | `ResumingEvents` reopens it with `?from_seq=` the next event, backoff 0.5 → 5 s, no event handed on twice; a refusal that cannot change (unknown session, wrong code) ends it at once. A stream that stayed open 5 s counts as working, with or without events (a session waiting for its source sends none) | every event | nothing, unless the stream cannot be kept for 660 s in a row (the source grace plus a minute): then the orchestrator ends the round as failed (rebuild it afterwards) |
| «Terminar» with finals still on their way | the transcriber's stop returns once its session has ended; the orchestrator then waits up to 15 s (`STOP_DRAIN_S`) for its follower to read the last events (finals, `session_ended`) before it cancels it | the last finals reach the trackers and the judgement | anything the follower has not read after 15 s |

Publishing to the RAG (and starting the post-call analysis) now happens
after the session has ended, so a stop never waits for it; until it is
done the session stays in the live marker (state `handing_over`), so
the services guard does not let a restart cut it, and its record is
written again with the result.

### Timeouts, from the page down

| Step | Limit | Where |
|---|---|---|
| «Terminar» waits for the server to acknowledge the held audio (#263) | 60 s of connected time; «Terminar ya» ends at once | `js/live-audio.js` `FINISH_CAP_MS` |
| …then for the transcriber's `stopped` | 5 s | `STOPPED_WAIT_MS` |
| The page's stop request, through the transcriber's proxy to the orchestrator | 120 s (other calls 45 s); the page itself sets no limit | `live_assist_proxy.STOP_TIMEOUT` |
| Transcriber: a stop over HTTP lets a connected source finish | 15 s | `live.stop_drain_seconds` |
| Transcriber: then waits for the session's end | 60 s | `LiveSessions.stop(timeout)` |
| Orchestrator → transcriber stop request | 90 s (other calls 40 s) | `http_transcriber.STOP_TIMEOUT` |
| Orchestrator: last events after the transcriber's stop | 15 s | `sessions.STOP_DRAIN_S` |
| Orchestrator: a dropped event stream is retried | until it cannot be kept for 660 s in a row | `ResumingEvents.give_up_s` |
| Transcriber: publish to the RAG (after the end, not in the stop) | 30 s per try, 3 retries, 1 → 2 → 4 s apart | `publish.http` |
| A lost audio source is waited for (#263) | 600 s; 120 s after a close 1001 | `live.source_grace_seconds`, `live.source_gone_seconds` |
| A session without audio stops | 120 s | `live.idle_seconds` |
| The GPU lease of a dead holder | 15 s | `gpu_lease.LEASE_TTL_SECONDS` |
| Tunnel probe failures before the client is restarted | 3 in a row (10–45 s); never while a live audio socket is connected | `chisel_tunnel.PROBE_FAILURES` |
| Caddy keeps upgraded connections after a reload | 10 min | `infra/Caddyfile` `stream_close_delay` |

### Rebuilding a round

`python -m orchestrator rebuild-round <session> [--transcript FILE |
--events FILE] [--plan PLAN --candidate ID] [--no-judge]
[--knowledge http|fake] [--reasoner http|fake|module:factory]` replays
the kept conversation (the transcript defaults to
`data/transcripts/<session>.transcript.v1.json`; `--events` takes the
transcriber's `data/.../live/<session>/events.jsonl`) through the
round's trackers with `RoundRebuild`
(`orchestrator/application/rebuild_round.py`): the plan's topics and
concepts, the product's mandatory items and the lines to avoid (these
two ask the RAG, so only with `--knowledge http`), without any
suggestion; then it judges the candidate's answers and stores the
report with the plan's results and in the candidate's file, as at the
end of the round. It writes the hand-off record next to the serve's
(`--handoff-dir`, default `data/live/serve`; checklist and the
judgement's labels, no words) and prints only numbers and states. The
service takes the events, the roles and optional parts
(`RoundRebuild(executor, mandatory=, avoid=, wiring=).rebuild(session,
events, roles, product=, plan=, rounds=, candidate=, candidate_id=,
model=, judge=, publish=)` → `RebuiltRound`), so other round-level work
plugs into it. `interview rejudge` (#262) stays as it was.

### The services guard

`~/.local/bin/ragputin-servicios` is the owner's file; its reviewable
copy is `infra/ragputin-servicios`. With it:

- `stop` and `restart` refuse (exit 3, one line saying which session)
  while a live interview runs, unless `--force`. «Runs» is either the
  GPU lease held by `ragputin-live` (valid, its process alive) or the
  transcriber's marker `$XDG_RUNTIME_DIR/ragputin/live-active.json`
  (written while any live session is starting, running or waiting for
  its audio source, which may hold no lease; a marker whose process is
  gone is ignored). The check is `infra/live_guard.py` (system
  `python3`, standard library; exit 0 free, 3 running, 2 cannot tell,
  which refuses too).
- `status` adds `Entrevista en directo:` with the same line.
- The units are started with `Restart=on-failure` and `RestartSec=2`,
  at most 5 starts in 5 minutes (`StartLimitBurst=5`,
  `StartLimitIntervalSec=300`): a start that cannot work stops trying
  instead of reloading the models in a loop. `start` first stops and
  clears a unit that is still in its own restart loop.
- A session that has ended but is still handing its transcript over is
  «running» for the guard too.

Install it (when no interview is running), from the main checkout after
the merge:

```sh
install -m 755 ~/Desktop/ideas/RAGPutin/infra/ragputin-servicios ~/.local/bin/ragputin-servicios
```

It reads `infra/live_guard.py` from the same checkout
(`RAGPUTIN_DIR`, default `~/Desktop/ideas/RAGPutin`).

### Caddy

The repository's `infra/Caddyfile` now has `stream_close_delay 10m` on
the tunnel's upstream and on the API's. Honestly, that does little: the
server's internal Caddy runs with `admin off` and is never reloaded;
every site deploy (`infra/install-remote.sh`) recreates the `web`
container and restarts `tunnel`, which cuts the tunnel and every live
socket at once whatever that setting says. A deploy during an interview
therefore still costs a tunnel reset, which the reconnect of #263 now
absorbs (seconds of held audio, resent). The setting would only matter
if that Caddy were ever reloaded in place. The block that is reloaded
in place is the Peer4U server's (`infra/ragputin.caddy`, installed by
`make infra-proxy`, changed by the owner by hand), and there the line
helps: replace its `reverse_proxy ragputin-web:80` with:

```
reverse_proxy ragputin-web:80 {
	stream_close_delay 10m
}
```

### Not measured

Nothing above ran on the GPU or through the real tunnel: a real CUDA
out-of-memory error in the live worker, a kill of the real services
mid-interview, Restart=on-failure on the real units, a Caddy reload
with a live socket open, and the restart guard on the real lease were
tested only with fakes, temporary directories and stand-in commands.
Both Caddy changes (the repository's file, and the Peer4U block's
line in a test site) pass `caddy validate` (caddy 2.10, the local
image); a reload was not tried.

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
