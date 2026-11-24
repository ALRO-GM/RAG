# Connection check before and during a live session (LV-19, #247)

Before starting a live interview (on «En directo», `#/live`) or a talk
with the voice assistant (`#/voicebot`), the interviewer sees a card
«Calidad de la conexión» that answers one question: is this connection
good enough to hold the interview now? While a live session runs, a
small chip in its header keeps saying so. The check never blocks: a
poor result asks once before a live session starts («Empezar
igualmente»), and the decision is the interviewer's.

## 1. The path it measures

```
browser ──► public server (rag.peer4u.app, Caddy)  ──► tunnel ──► processing server
            /healthz answered here                    /api/* goes on: transcriber,
                                                      which asks the orchestrator
                                                      and the RAG on loopback
```

The page only ever asks its own origin (CSP `connect-src 'self'`); no
third-party speed test.

| Request | Answered by | Why |
|---|---|---|
| `GET /healthz` ×3 | the public server's Caddy (`infra/Caddyfile`, works with the processing server off) | the device's own network: if this is slow too, the problem is the interviewer's network, not the link. Any HTTP answer counts as reachable (a development server without `/healthz` answers 404, which is still an answer) |
| `GET /api/health` ×6, the first left out | the transcriber, through the tunnel (no access code; reveals nothing) | latency, its variation and lost requests over the whole path the live audio and suggestions take. The first request only opens the connection (TLS, HTTP/2), so it is not counted. A 502–504 comes from the public server when the tunnel is down: the processing server is not reachable |
| `GET /api/connection` (access code) | the transcriber, new | one word per service the browser cannot ask itself: `transcription`, `assistant` (orchestrator `GET /health`), `knowledge` (RAG `GET /health`) |
| `POST /api/connection/upload` (access code), 64 KiB | the transcriber, new | how fast this browser sends what its microphone would: two seconds of the live audio (16 kHz 16-bit mono PCM, 256 kbit/s; `js/live-audio.js`) |

Each request gives up after 2 s (the upload after 4 s); two time-outs in
a row end a series early. A whole check is at most 11 small requests;
against the running services from this workstation, the public
`/api/health` round trip took 40–46 ms (`curl`, 5 requests, 2026-11-24),
so a whole check should take well under two seconds when all is well
(an estimate from that figure, not timed in a browser on that path). A check is
reused for 60 s when the start form is shown again.

### The two new routes

Both are under `/api`, behind the access code like every route but
`/api/health` (`transcriber/adapters/inbound/http_api.py`,
`mount_connection`).

- `GET /api/connection` → `{"services": {"transcription", "assistant",
  "knowledge"}}`, each `available`, `unavailable` or `not_set_up`. The
  use case `transcriber/application/connection.py` (`CheckServices`)
  asks its `ServiceProbe`s at the same time; the adapter
  `adapters/outbound/http_health_probe.py` counts only a `200` within
  1.5 s, never follows a redirect and never passes the answer's body on
  (the RAG's health route lists index statistics). `transcription` is
  `available` when live sessions are set up in this transcriber (it is
  the process answering). The orchestrator is the one the live proxy
  uses (`TRANSCRIBER_ORCHESTRATOR_URL`, default `127.0.0.1:8772`); the
  RAG is the transcripts' HTTP publisher (none: `not_set_up`). No
  address, version or error text reaches the page.
- `POST /api/connection/upload` reads at most 128 KiB (413 above),
  drops it and answers `{"bytes": n}`. Nothing is stored.

No new secret: the orchestrator's and the RAG's health routes need no
token on loopback.

## 2. Thresholds (chosen, not measured on calls)

`js/lib/connection.js` (`THRESHOLDS`). A figure up to «good» is good, up
to «fair» is fair, above is poor; the verdict is the worst of them.

| Figure | Good | Fair | Reasoning |
|---|---|---|---|
| Round trip, median of 5 (`/api/health`) | ≤ 150 ms | ≤ 400 ms | The live path's budget is 3 s from the end of a sentence to its suggestion, with 220 ms of margin (`docs/platform/ARCHITECTURE.md` §6). Remote use adds about one round trip (audio up, suggestion down): 150 ms fits in the margin; 400 ms makes suggestions noticeably later. |
| Variation (mean change between consecutive round trips) | ≤ 50 ms | ≤ 150 ms | Partial lines arrive every 800 ms (`partial_every_ms`); up to 50 ms is not seen, above 150 ms (about a fifth of that step) lines and suggestions come in bursts. |
| Lost requests (failed or over 2 s) | 0 | ≤ 20 % (one in five) | One lost request is a stall of 2 s, most of the 3 s budget. |
| Upload (only when this browser's microphone is the audio) | ≥ 1024 kbit/s | ≥ 384 kbit/s | The microphone is sent raw, 256 kbit/s. Four times that leaves room for the rest of the page; 1.5 times is the least that keeps up. The estimate is a lower bound: 64 KiB never leaves TCP slow start, and one round trip is taken off the time. It is not a speed test. |

Services: the transcription or the live assistant down makes the
verdict poor (no session is possible); the knowledge search down makes
it fair (suggestions are more general). A scripted demo needs only the
assistant, a recorded replay the assistant and the knowledge search.
Services that could not be asked (an older transcriber without the
route) count as neither and read «Sin comprobar».

The «Transcripción» row and the start form share one source of truth:
when the chosen product says live audio cannot start (the
orchestrator's `live`, false when it has no link to the transcriber),
the row reads «No disponible», the verdict is poor and the advice says
live audio is not available now and the demo can be watched. The
transcriber's own word (`/api/connection`) is still asked; either one
being off makes the row unavailable.

The upload is «not measured» when the route is missing or refused for
the code (401, 403, 404, 405: an older server) and then has no effect;
when it times out, fails on the network or is refused on the way (5xx,
413) it is «failed»: a poor upload with its advice, shown as «No llegó
a tiempo» among the figures.

## 3. What the page says

Summary first (`docs/web/DESIGN_SYSTEM.md`): the verdict with its icon
(check, or a circled «!»; never colour alone), one sentence on what to
do when something is wrong, the service rows, and the figures with how
they are measured behind «Más información».

| Verdict (es) | en |
|---|---|
| Conexión adecuada para la entrevista | Connection suitable for the interview |
| Conexión justa: puede haber retrasos en las sugerencias | Fair connection: suggestions may be delayed |
| Conexión insuficiente para la entrevista | Connection not good enough for the interview |

One piece of advice, by priority: no answer at all (`offline`: check the
Wi-Fi or the cable); the public server answers but not the processing
server (`server`: «El servidor de procesamiento no responde: avisa a
administración.»); transcription or assistant down; a slow or lossy
link, attributed to the device's network when `/healthz` is slow too
(`local`: closer to the router, cable, close downloads) and to the link
otherwise (`link`: try again in a minute, then tell the administrator);
a tight upload; the knowledge search down. Rows: «Servicio web»,
«Enlace con el servidor de procesamiento», «Transcripción», «Asistente
en directo (sugerencias)», «Búsqueda en el conocimiento»: «Disponible»,
«No disponible», «No configurado», «Sin comprobar». The term is
always «servidor de procesamiento» / «processing server».

On `#/live` the card sits on the right of «Nueva sesión en directo» on
wide screens and above it on a phone; on `#/voicebot` above the start
card. If «Empezar en directo» is pressed while a check is running, the
button reads «Comprobando la conexión…» and waits for it for at most
5 s (`CHECK.settleMs`), then asks only if the result is poor; a check
that fails or takes longer never stops the start.

## 4. During a session

`ConnectionWatch` (`js/lib/connection.js`), drawn as a chip after the
talk-time bar in the live header (`.lv-conn`: icon and a word, its
longer sentence as the title): «Conexión buena», «Conexión justa»,
«Conexión insuficiente», «Sin conexión».

- One `GET /api/health` every 15 s (5 s after a failure), the services
  every fourth ping: four requests of a few hundred bytes a minute.
- It gets worse only when 3 pings in a row are worse (about 45 s), and
  better after 2 in a row are better; one slow or lost ping changes
  nothing. More than one lost ping in the last 8 (two minutes) caps it
  at «justa». A service counts as down after two checks in a row.
- Two failed pings in a row are a lost connection: up to about 28 s
  after the link drops (a 15 s pause, a 4 s time-out, the 5 s retry and
  another 4 s time-out); about 5 s when the requests fail at once. The line under the header
  (one persistent live region, `role="status"`, whose text changes only
  when the state changes, so a screen reader hears it once) says «Se ha perdido la conexión con el servidor de
  procesamiento. Reintentando…» (it replaces the stream's own
  «reconectando»), and at the first answer «La conexión ha vuelto a las
  HH:MM.» for 30 s.
- It starts from the link part of the start screen's check (round trip,
  variation, losses: what it can keep measuring itself, not the upload
  or the services) when that check is younger than 60 s, else it says
  «Comprobando conexión» until the first ping.
- The chip is hidden once the session ends; the watch stops with it.

The scripted demo (`#/live/demo`) shows «Conexión buena» on a healthy
stack; the demo recorder's stand-in RAG answers `GET /health` for this.

## 5. Not done

- No measurement on real calls or networks: the thresholds are chosen
  from the budget above. Mobile networks were not tried.
- The in-session watch does not read the audio socket's own backlog
  (`bufferedAmount`); the transcriber's `asr_behind` warning covers it.
- The voice assistant's conversation page has no chip (only its start
  screen has the card), and it does not ask before starting with a poor
  result.
- Wording: every page's dictionary now says «servidor de procesamiento»
  / «processing server» where it meant the machine that does the work
  (tested over all of `es.js` and `en.js`). The consent notice the
  transcriber serves (`transcriber/domain/consent.py` texts, versioned)
  was not changed here.
