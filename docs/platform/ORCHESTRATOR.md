# Orchestrator (PL-10, PL-11)

Package `orchestrator/`, hexagonal like `src/` and `transcriber/`; no torch.
Dependency group `orchestrator` (pyyaml, httpx). Design: `ARCHITECTURE.md` §3, §4, §10.

```
python -m orchestrator check-config [--dir config/products]
python -m orchestrator replay <transcript.v1.json> --product live_assist.support \
    [--speed 1|N|max] [--out F.jsonl] [--events-out E.jsonl] [--roles a=customer,b=agent] \
    [--knowledge fake|http] [--reasoner fake|http|module:factory]
python -m orchestrator serve [--host 127.0.0.1] [--port 8772] [--token-file F] \
    [--knowledge http|fake] [--reasoner http|fake|module:factory] \
    [--rag-token-file ~/.config/ragputin/rag-token] [--transcriber-url http://127.0.0.1:8770] \
    [--transcriber-code-file data/transcriber/access_code] [--transcripts DIR ...] [--log-dir data/live/serve]
```

- `domain/config.py` is the pydantic model of the product YAML (`config/products/*.yaml`). Rules checked: live products need vad, trigger and a streaming ASR; one call in flight; no audio kept live; expressed customer signals only in `live_assist.support`.
- `domain/triggers.py` decides which event asks for a suggestion: speaker role, min words, debounce (stream time), `final` or `turn_end`.
- `application/live_assist.py`: one call in flight; a newer trigger cancels the stale call (its answer is dropped, status `cancelled`) or, with `cancel_stale: false`, is skipped. Each call writes a JSONL record with `stages.knowledge_ms`, `reasoner_ms`, `total_ms` and the reasoner's own latency.
- `application/replay.py`: transcript.v1 to `live.event.v1` (finals stamped at sentence end, `turn_end` on speaker change). `--speed 1` is real time; `max` never waits and runs each call to the end.
- Ports: knowledge and reasoner exchange the contract JSON documents. Adapters: in-process fakes, `http_knowledge.py` (`POST /knowledge/query`), `http_reasoner.py` (`POST /reason`), and for `serve`: `http_transcriber.py` (start/stop live sessions), `ws_events.py` (the `live.event.v1` socket, `websockets` sync client), `fs_transcripts.py` (replayable transcripts).

## `serve`: the live loop for the live page (LV-9)

`adapters/inbound/http_server.py` (FastAPI, loopback only; with `--token-file` every route but `/health` needs `Authorization: Bearer <token>`):

| Route | Body | Answer |
|---|---|---|
| `GET /health` | | `{"status": "ok"}` |
| `GET /products` | | live products: `product`, `banner`, `signals` (support only), `script_phases` (interview), `roles`, `suggest_about`, `channels`, `live` (a transcriber is linked) |
| `GET /transcripts` | | `*.transcript.v1.json` of `--transcripts` (default `data/transcripts`, `examples/platform`) |
| `POST /sessions` | `{"product", "audio": "stereo"\|"mic", "session_id"?}` | 201 session; starts `POST /api/live/sessions` on the transcriber with the product's speakers and roles, then follows `WS /api/live/{id}/events` |
| `POST /sessions/{id}/replay` | `{"product", "transcript", "speed"?}` | 201 session; replays a transcript (speed 1 real time, `null` as fast as the services answer) |
| `POST /sessions/{id}/stop` | | the session (live: also stops it on the transcriber) |
| `GET /sessions`, `GET /sessions/{id}` | | state, counts of calls by status, skipped triggers, rejected events |
| `GET /sessions/{id}/suggestions/stream` | `?after=n` or `Last-Event-ID` | Server-Sent Events (below) |

Errors are `{"code", "message"}`: `not_found` 404, `busy`/`wrong_state` 409 (one session at a time), `invalid_request` 422, `transcriber` 502, `unavailable` 503 (no access code, so live sessions are off; replays still work).

The stream (`application/feed.py`, `application/sessions.py`) sends one `message` per item, numbered from 1 (`id:`), and `event: end` when the session is over:

- `{"type": "session", "session": {...}}` first: what the page draws (`page`: banner, signals, phases, roles).
- `{"type": "transcriber", "session": {...}}` (live only): the transcriber's answer.
- `{"type": "event", "event": <live.event.v1>}` for every event (partials, finals, speaker updates, warnings).
- `{"type": "suggestion", "record": <SuggestionRecord>}` per reasoner call: `status`, `suggestions` (with `cites` and `about`), `retrieved` (the chunks given to the reasoner, for citations), `stages.knowledge_ms`, `reasoner_ms`, `total_ms` and the reasoner's own latency.
- `{"type": "status", "session": {...}}` at the end (`ended` or `failed`).

Expressed customer signals (LV-14, `domain/signals.py`, `ARCHITECTURE.md` §8.1): in `live_assist.support` each suggestion's `signals` are kept only when one of its `about` utterances is a customer sentence; they become `expressed: [{signal, utterance, quote}]`, shown on the page as "El cliente dice: «…»". Suggestions are ranked by signal weight (churn_risk 5, complaint 4, stated_frustration 3, urgency 3, confusion 2, satisfaction 0; ×1.5 for each earlier time the same signal was expressed in the session, so a repeated complaint pushes escalation up). In every other product `signals` and `expressed` are removed before the record is shown or logged, and the page also refuses to draw them outside support. Text only: no voice or emotion model. Not measured yet: the precision of each signal on synthetic snippets (needs ~60 Haiku calls).

Every record is also written to `--log-dir/<session>.<product>.suggestions.jsonl`. A quota or auth answer stops the session's calls (skip reason `halted`) and the page shows a banner. Mic sessions are one channel heard as the person the product suggests about (customer, candidate); stereo sessions use the product's two channels. The transcriber side (start, stop, the events socket, the audio socket) is `docs/live/LIVE_SESSION.md`.

The web page reaches `serve` through the transcriber: `/api/assist/<route>` (`transcriber/adapters/inbound/live_assist_proxy.py`) forwards the routes above (fixed list) behind the access code and adds the bearer token read from the RAG token file (`~/.config/ragputin/rag-token`); `TRANSCRIBER_ORCHESTRATOR_URL` changes the address.

### Running it as a service

Like the RAG and the transcriber, as a user unit (not from a terminal):

```
systemd-run --user --unit=ragputin-orchestrator --collect -p WorkingDirectory=$REPO \
  /bin/bash -lc "cd $REPO && exec .venv/bin/python -m orchestrator serve \
    --token-file ~/.config/ragputin/rag-token >> ~/.local/state/ragputin/orchestrator.log 2>&1"
```

It needs the RAG up (knowledge and `/reason`) and, for live sessions, the transcriber started with `--keep-code` (it reads `data/transcriber/access_code`). No GPU, no torch. Start order: RAG, orchestrator, transcriber (the page works without the orchestrator; its live page then says the service is not available).

## In-process reasoner and the import ban

`orchestrator` may not import `src` (the ban test covers the three packages pairwise), so there is no `--inprocess` adapter inside the package. `--reasoner module:factory` loads a factory by name instead; `experiments/live/orchestrator_reasoner.py:build` wraps the RAG's `Reasoner` over `claude -p` and lives outside both packages. It goes away once `POST /reason` exists.

## Roles in a replay

A transcript has no roles. Speakers take the roles of the product's `source.channels` by order (the first speaker the first role, any further speaker the last one), or `--roles`.

## Smoke run (measured, 2026-11-21)

8 real `claude -p --model haiku` calls (cold one-shot path, fake knowledge with no chunks) on the first 25 utterances of `sim-reunion-verano`, `--product live_assist.support --speed max`: all 8 `ok`; reasoner p50 3802 ms, p90 4406 ms (the warm stream-json backend of `ARCHITECTURE.md` §6 is not used here, so this is not the live budget). Skipped triggers: 13 by role, 4 by min_words.

## Smoke of `serve` with the live page (measured, 2026-11-21)

`serve` on a spare port with the HTTP knowledge and reasoner adapters against the running RAG (warm Haiku pool), the page served by a fake-runtime transcriber, driven by headless Chrome: "Demo (reproducción)" at speed 1, stopped after a few calls. 8 real Haiku calls in total.

| Replay | calls | status | knowledge_ms | reasoner_ms |
|---|---|---|---|---|
| `sim-soporte-lumbre`, support | 5 | 4 ok, 1 no_suggestion | 9.3–32.6 | 948.8, 1886.7, 2073.9, 2346.8, 2574.3 |
| `sim-entrevista-backend`, interview | 3 | 3 ok | 19.4–37.2 | 1140.9, 1212.7, 1494.9 |

Support: 3 of the 4 answered calls carried quoted signals (urgency, stated_frustration, complaint); interview records carried none. Too few calls for percentiles.
