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
| `POST /api/live/sessions` | `{"session_id", "channels": 1\|2, "speakers": [{"channel", "id", "role", "name"?}], "product"?, "handoff": {"transcript_v1", "run_post_call"}, "language": "es"\|null, "hang_ms": 300–700, "partial_every_ms", "diarize"?, "enrolled"?, "parties"?, "buffer_audio"?, "buffer_max_hours"?}` (the last two: #265, the product's leave to keep unsent audio) | 201 session (`state`, `device`, `model`, `speakers`, `audio`, `events`); 409 `wrong_state` (id in use) or `busy` (one session at a time); 422 invalid body |
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
| Grace | `live.source_grace_seconds` = 600, `live.source_gone_seconds` = 120 | a source that goes away without `stop` leaves the session waiting 10 min (instead of the 120 s idle stop) with a `warning` `source_lost`; 2 min when its socket closed with 1001 «going away» (a closed tab). An abrupt loss (no close frame: a dropped connection, a reset of the public tunnel) reaches the server as close code 1005 («no status»), locally and through the tunnel (measured: `experiments/live/reconnect_smoke.py`, 2026-11-25; the client side sees 1006); every code but 1001 gets the long grace. A returning source gives `source_back` with `warning.seconds`; past the grace the session ends (`end_reason: source_lost`) |
| Catch-up | worker, `engine.skip_partials` | after `source_back` the resent audio is read as fast as the recogniser goes (the live engine is not paced) with no partials until the queue is empty; its finals come with the real `timings.lag_ms`. Lateness alone never turns partials off (a CPU-degraded session keeps them) |
| Re-anchor | `PaceWatch`, `LiveEngine.realign` | when 30 pieces in a row arrive at real-time pace (each 0.9–1.1 times its own length after the one before) with nothing queued, and the timings think the audio is more than 1 s older than it is (a microphone pause the page did not report), the origin of the timings moves forward: `lag_ms` is true again, so partials and suggestions come back. Never between a `source_back` and the page's `caught_up`: a resend of held audio, even one that happens to come at about real-time pace, keeps its true lateness (and asks for no suggestion) |
| Suggestions | `orchestrator/domain/triggers.py` `LATE_MS` = 10 s | a final heard more than 10 s late asks for no suggestion (skip reason `late`); topics, concepts, mandatory items and avoid hints still take it |
| Stop | `live.stop_drain_seconds` = 15 | a stop asked over HTTP lets a connected source send what it holds for up to 15 s (until its own `stop` or its socket closes) instead of refusing that audio; then it waits up to 60 s for the end, so the orchestrator's client waits 90 s for a stop (`STOP_TIMEOUT`; other calls keep 40 s) |

«Compartir pantalla» (#278, `docs/web/SCREEN_SHARE.md`) never touches
this: only the sharing device runs the live page and owns the audio
socket; a joined device shows a mirror of it and runs no live code, so it
can never reattach or take the source over.

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

The audio held during an outage lives in the page's memory and is
dropped as soon as it is acknowledged. Without the outage buffer of
#265 (below) still lost: a closed tab, a crashed browser or a reload
(the page's memory goes with it; the reattached page fills that time
with silence), an outage longer than the grace, audio the 30 min outbox
had to let go, and what «Terminar ya» or the 60 s cap gives up. With it
(products `live_assist.interview` and `live_assist.support`) all of
these are kept in the browser and recovered after the session; the
voice assistant never buffers (its «Terminar» still drops what it
held).

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
| The transcriber stops mid-session (crash, kill, restart) | each session keeps `session.json` (request, speakers, names, model, state, and at the end the reason and the publish's result). A session whose hand-over is done (published, nothing asked, or given up) gets an empty `handed` mark and is never opened again, so a start does not grow with history. At the next start, the others with a record, latest end first (`ended_at`, not the file's time) and at most 20 per start (the rest are counted in the log and wait), are handed over: one without `transcript.json` is rebuilt from `events.jsonl`, kept, published as asked and listed as `recovered` (`end_reason: restart`); one whose transcript was kept but not published (the process died while the RAG got it, or the RAG refused it) is published again (the RAG replaces a transcript of the same session), 3 tries in all, then left unpublished for good (`publish_state: given_up`, said in its summary). The record says `publishing` before each publish and the result after: a session found `publishing` (the process died in between) is published once more at most, and its post-call analysis is not started again when the RAG already has one. The publisher cannot ask the RAG whether it has the transcript, so a transcript the owner deleted from the RAG in that window can come back once (rare; not handled). Publishing runs on a thread, at most 300 s per start (the rest wait for the next start), so the start is not delayed and the services guard is not held longer. A session whose log or record cannot be used is logged (its id and the error's type) and left as it is; recovery never stops a start. Sessions logged before #264 (no record) are never touched | the transcript up to the last event written (events are flushed one by one); a publish that did not complete | the audio after the crash; the orchestrator's live view of that round (see below) |
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
  700). No audio is ever written there (the outage buffer of #265 has
  its own folder, below).
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
re-run (only an outage's lost audio is kept, until transcribed, #265),
the live page itself.

## Long outages: kept in the browser, recovered afterwards (#265)

A short outage costs nothing (#263). For a long one, a closed or
crashed tab, or a connection that only returns after the interview, the
page keeps what the server has not acknowledged in the browser and
sends it after the session; the server transcribes it with the batch
pipeline, puts it in the transcript in place of the gap and computes
every indicator again.

### Where audio can remain, and for how long (exact)

Only for products with `privacy.buffer_audio_on_outage` (on for
`live_assist.interview` and `live_assist.support`, `buffer_max_hours:
48`; off for every other product; a deployment turns it off in the
product file, and the transcriber's own switch is
`live.recovery.enabled`). The orchestrator passes the product's leave
in the start request (`buffer_audio`, `buffer_max_hours`), the
transcriber keeps it in the session's record and refuses any upload for
a session without it (409). The page reads it from `page.buffer`.

1. **The page's memory** (#263): every 100 ms piece not acknowledged,
   at most 30 min. Gone at the acknowledgement or when the tab closes.
2. **The browser's disk** (IndexedDB `ragputin-live-audio`). A
   connected session whose backlog is 3 s or less writes **nothing**.
   Audio is written only while the audio link is down, while more than
   3 s wait for their acknowledgement, and at «Terminar» or when the tab
   is hidden or closed (`pagehide`) if something is still unacknowledged
   then. Each chunk is deleted as soon as the server acknowledges all of
   it (unless part of it became silence on the server: it is kept for
   the recovery), all of a session once the server accepted the closed
   upload (only the chunks the server holds), at once on «Descartar».
   Past the expiry (session start + `buffer_max_hours`, 48 h) it is
   deleted by a sweep that runs **only while a page of the app is open
   in that browser**: when it loads, each time it becomes visible again,
   and every 10 minutes. **If nobody opens the app in that browser
   again, the expired audio stays on that browser's disk until the
   browser evicts the site's data or someone clears it**: the web gives
   a page no way to run when it is not open. A tab that crashes keeps
   up to the last 5 s written; what was not yet written is lost.
3. **The server** (`<data_dir>/live-recovery/<session>/audio/*.pcm`,
   folder 700, files 600, git-ignored data dir) from the upload until
   the transcription succeeds (then deleted at once), the job ends or
   fails, or its expiry: the sooner of the upload's start plus
   `live.recovery.ttl_hours` (24 h), the session's start plus
   `buffer_max_hours`, and the page's own expiry; swept at start and
   every hour while the transcriber runs (a stopped transcriber deletes
   it at its next start). Also deleted at once on «Descartar» and when
   the candidate's file is deleted. While transcribed, one WAV per part
   also lives in `<data_dir>/live-recovery-batch/` and goes with it;
   batch parts of no job are swept too.
4. **Text, not audio**: the recovered parts' transcripts and the live
   transcript before the merge, in `<data_dir>/live-recovery/<session>/
   text/`, until the job ends (a process that stopped in between leaves
   them for the next sweep, which deletes the texts of every ended
   job); `recovery.json` (no audio, no words) stays as the page's status
   until «Descartar».

Nothing is put in «Sesiones», nothing is published as audio, and
deletions are logged as counts.

### What happens in each case

| Case | What the page does | What is recovered |
|---|---|---|
| The connection returns during the interview | resends from the server's count (#263); while it was down the audio was also written to IndexedDB; what its 30 min memory let go was sent as a gap and stays in IndexedDB | after the session, the gaps |
| The connection returns only after «Terminar» | «Terminar» waits as before; with the buffer the line says «Sin conexión: lo pendiente está guardado en este navegador y se enviará después de la entrevista. Puedes esperar o terminar ya.» and the button «Terminar ya: se enviará después». Ending then writes everything held, marks the session ended here, tries the stop (offline it fails: the page ends the session itself) and says «La entrevista ha terminado. {n} s de audio que no llegaron a enviarse quedan guardados en este navegador: envíalos desde «En directo» para completar la entrevista.» The server's session ends by itself after its grace (`source_lost`) | everything after the last acknowledged sample |
| The tab was closed or crashed and reopened | a reattached page goes on (its new chunks numbered after the old ones); the old page's chunks were never sent (the server filled that time with silence on `align`) | after the session, that silence |
| The connection never returns | the audio waits in the browser; see 2 above for when it goes | nothing |

On the next visit to «En directo» the start form shows «Audio pendiente
de entrevistas» with one line per session: «Hay {min} min de la
entrevista del {fecha} sin enviar.», «Enviar» and «Descartar» (asks
first: «¿Descartar el audio sin enviar?» «Se borra ahora de este
navegador y del servidor. Lo que no se envió no se podrá añadir a la
entrevista.»). A session that is still on is not offered for sending:
«Esta entrevista sigue en marcha: vuelve a ella y el audio pendiente se
enviará desde allí.» with «Volver a la sesión» (the server refuses an
upload or a close for a session still on, 409). The end of a session
that kept audio shows the same card. Sending says «Enviando el audio
pendiente ({n} %)», then follows the job in a `role="status"` line
(text, not colour): «Pendiente de analizar» → «Analizando ({n} %)» →
«Completado ({n} min recuperados)», «Completado ({n} min recuperados,
sin identificar quién habla)», «Completado: el audio enviado no tenía
conversación que añadir», or «No se pudo recuperar». {n} is the speech
the merged sentences cover, not the length of the gap. A full server
answers 413: «El servidor no tiene sitio para este audio ahora. Sigue
guardado en este navegador; inténtalo más tarde.» A closed upload that
lacks some of the chunks still here (they arrived late) deletes here
only what the server holds and says «Esta entrevista ya se completó con
el audio enviado antes; lo que queda en este navegador llegó tarde y no
se puede añadir. Puedes descartarlo.» Recoveries moving on the server
(or ended in the last 2 h) are listed on the same card; the candidate's
file lists its rounds' recoveries («Audio recuperado de las rondas»).

### Upload protocol (transcriber, behind the access code)

| Route | Body | Answer |
|---|---|---|
| `PUT /api/live/sessions/{id}/recovery/chunks/{n}?offset=&channels=` | raw PCM s16le 16 kHz (`channels` interleaved, 1 from the browser), from stream sample `offset` | the job; a chunk sent again replaces itself (resumable, idempotent). 404 for a session this service never ran; 409 `session_running` while the session is still on, `not_buffered` for a product without the leave, `wrong_state` once closed; 400 for audio past the product's limit counted from the session's start (`buffer_max_hours`, else `live.recovery.ttl_hours`: the page goes on recording offline long after the server ended the session, which waits for a lost source only 10 min), odd bytes, negative numbers, other channels than the first chunk's; 413 past 1 MiB per chunk, 512 MiB per session, 2 GiB in all, 8192 chunks |
| `POST /api/live/sessions/{id}/recovery/close` | `{"count", "expires_at"?}` | the job, `queued`; 409 while chunks are missing or the session is still on; `expires_at` may only bring the expiry closer |
| `GET /api/live/sessions/{id}/recovery` | — | `{session_id, state, progress, seconds, minutes, chunks, added, unidentified, failed_step, error, expires_at, audio_deleted, updated_at}`; the page resumes from `chunks` |
| `DELETE /api/live/sessions/{id}/recovery` | — | 204; everything kept for the session is deleted now, also while a step runs |
| `GET /api/live/recoveries` | — | every job, newest first |

Ids are checked against the session id pattern before any path is
built; one session's chunks live in its own folder, so an id cannot
write into another's. The page sends a chunk three times at most (1, 2,
4 s apart). A chunk refused as such (400) does not strand the rest: the
others are sent, the upload is closed with what the server accepted,
and the page says «{n} s del audio guardado no se pudieron añadir (fuera
del plazo en que se puede guardar) y se han borrado de este navegador.»
A 413 stops the upload (the audio stays; «El servidor no tiene sitio…»),
`not_buffered` says «Este producto no guarda audio durante un corte:
este audio no se puede enviar. Puedes descartarlo.», and
`session_running` for a session this browser ended offline (the link
came back within the server's 10 min grace) first stops that session,
then sends (the stop is retried while its hand-over finishes), but only
when nobody sends its audio now (`GET /api/live/sessions/{id}` says
`source` is not `connected`; unknown counts as connected); a page that
goes back to the session clears the «ended here» mark, so «Enviar» in
another tab never stops an interview that goes on. For any other
session still on, «Volver a la sesión».

### The job (`transcriber/application/live_recovery.py`)

`uploading → queued → transcribing → merging → publishing → analysing →
rejudging → done | failed`, written to `recovery.json` by a thread of
`transcriber serve` (`RecoveryWorker`, every 5 s).

- **Crash safety.** State is written after every step and always
  before anything is deleted: when the transcription is done the parts'
  transcripts are saved and the job is written as `merging` (audio
  marked deleted) before the audio goes. A crash anywhere either redoes
  the step from what is on disk or, finding the audio gone, goes on
  from the saved texts; with neither it fails as `audio_lost` and never
  says «Completado». Tested with a crash injected after each write of
  the whole state machine (store, log and batch writes: 18 distinct
  crash points reached), each followed by a restart.
- **Retries.** A real failure waits 30 s, then 120 s, and fails the job
  at the third try (`failed_step`, `error`: an exception's type,
  `MemoryError` included, never words). A batch job that a restart lost
  (no runner job for it) is submitted again **without** counting a try,
  at most 20 times. A failed job never blocks a live session.
- `queued` waits while the live session is still on or handing over.
  The spans to recover are the uploaded audio inside the silence the
  server put in (`silences`, written to `session.json` as it happens, so
  a crash keeps the gaps before it; one record write at a time, and a
  write for a silence that comes after the end is skipped, so the end's
  record is never overwritten) or after its last sample
  (`received_samples`, written at the end; for a session cut by a crash,
  the last event's stream time). Spans less than 30 s apart share one
  part and a contiguous span longer than 30 min is cut into 30 min
  parts (a sentence said across a cut is transcribed in two pieces);
  each part is one WAV per channel, never one array across a long gap.
  Building one part takes at most about 86 MB (16-bit samples and a
  one-byte mask per sample of 30 min); at most 200 parts. Nothing left: `done`, with
  «no conversation to add».
- `transcribing`: the batch pipeline of the recovery's own repository,
  on the service's job runner, so the batch GPU lease is honoured: it
  waits while a live session holds the GPU. Mono not diarized: one
  voice; diarized: the session's `enrolled` people as expected speakers
  and two voices for a two-party session; stereo: one voice per
  channel. **Services guard**: while one of its parts is running (holds
  the GPU) the session is in the live marker as `recovering`, so
  `ragputin-servicios stop|restart` refuses (it is the one moment a
  restart costs real work: the transcription is redone; the guard says
  «Recuperando el audio de una entrevista: transcribiendo (<id>).»).
  Uploading,
  queued, merging and the later steps do not count: a restart there
  costs nothing (state and audio are on disk). The marker is updated on
  each 5 s round, so a part that starts running is announced within 5 s.
- `merging` (`transcriber/domain/live_recovery.py`, deterministic, from
  the live transcript as it was before the first merge): a live
  sentence whose midpoint falls in a recovered span is dropped (it was
  heard over silence); a recovered sentence, moved by its part's
  offset, is kept only when its midpoint falls in a recovered span (what
  the server heard counts once); both in time order (live first on a
  tie), regrouped into turns, ids renumbered. Speakers: a stereo part
  takes its channel's speaker; mono without diarization the one voice;
  diarized, an enrolled person the batch pipeline recognised keeps their
  id, and with two parties every other voice is the other party once
  the employee is recognised. Otherwise the voices are `R1`, `R2`…
  (role unknown; `R1-1`, `R2-1`… when the job has several parts, since
  each part is transcribed on its own), never guessed: **their sentences enter the transcript
  but feed no indicator** (no topic, concept, mandatory item, line to
  avoid or judged answer), and the page says «sin identificar quién
  habla». The live session's own voice reference (the voice check) is
  kept in memory only and is gone by then, so it cannot be used.
- `publishing` / `analysing`: as the start request asked (the RAG
  replaces a transcript of the same session; the post-call analysis is
  started again with `restart`).
- `rejudging`: `POST <orchestrator>/sessions/{id}/recovered` with the
  merged transcript; the orchestrator replays it through the topics and
  concepts, mandatory items and lines to avoid, attributes and judges
  the answers again (`RecoveredRound` over `RoundRebuild`), replacing the
  stored report in the plan's results and the candidate's file; a
  session it still knows gets the judgement in its view and hand-off
  record.

External work (the batch submission, the RAG, the orchestrator) runs
outside the service's lock: an upload, «Descartar» or a sweep never
waits for it, and a discard during a step wins: the discard mark is
checked before each external call (submitting, saving the merged
transcript, publishing, the analysis, the rebuild), so a step already
running never goes on with a discarded session.

### Notice

The owner decided (2026-11-26) the wording: in the notices read to the
other person, «No se guarda el audio» / "No audio is kept" becomes «El
audio se borra en cuanto se transcribe.» / "The audio is deleted as soon
as it is transcribed." and nothing else changes: `interview_ai_notice@3`
(products `live_assist.interview`, `voicebot.interview`),
`support_ai_notice@2` (`live_assist.support`) and `voicebot.notice`.
The voice bots never buffer (their pages keep nothing beyond memory),
and the new sentence is as true for them as the old one, so they use the
same wording; `voicebot.notice` has no version id in the repository (it
is not a banner id), so its text changed in place.
`interview_ai_notice@1`, `@2` and `support_ai_notice@1` stay in the
dictionaries for old records. The sentence holds on every path except
one, said above: audio in a browser where the app is never opened again
after the session waits for that browser to clear it. Legal basis and
retention remain the owner's decision (#151); this is not legal advice.

### Not measured

Everything above was tested with the fake runtime, a fake IndexedDB,
fake sockets and temporary directories; no GPU run was made. Not
measured: the quality of the re-transcription of recovered audio and of
the speaker mapping (mono two-party sessions in particular), how long
the batch pipeline takes per recovered minute, a real browser's
IndexedDB quota and write speed during an hour-long outage, whether
`pagehide` writes complete when a real tab closes (the last few seconds
may be lost), and the orchestrator's rebuild time with the real
reasoner. The thresholds (3 s backlog, 5 s chunks, 10 min sweep, 30 s
join, 30 min parts) are chosen, not measured.
