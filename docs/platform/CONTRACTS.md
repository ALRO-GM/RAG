==================== FILE B: docs/platform/CONTRACTS.md ====================
# Platform contracts

Same rules as `contracts/README.md`: fields are added only as optional
fields within a version. Removing or renaming a field means a new version.
Readers are deployed before writers. Each schema goes in
`contracts/<name>.schema.json` (JSON Schema 2020-12) with one example in
`contracts/examples/`, and each side tests against it.

| Contract | Producer | Consumers | Transport |
|---|---|---|---|
| live.event.v1 | transcriber | orchestrator, live page | WebSocket `/api/live/{session}/events` (alias `/live/{session}/events`; JSON per message; `docs/live/LIVE_SESSION.md`) |
| knowledge.query.v1 / knowledge.result.v1 | orchestrator, analyzer | RAG | `POST /knowledge/query` |
| reasoner.request.v1 / reasoner.response.v1 | orchestrator | RAG | `POST /reason`; `POST /reason/stream` (NDJSON: `{"type": "text", "text"}` pieces of the model's raw output as it writes them, then `{"type": "response", "response"}` or `{"type": "error", "message"}`; the voice bot, VB-2); role `structured` with the response's optional `data` object, added compatibly for the AI interviewer's planner, slot filler and report, #208) |
| tts.request.v1 | orchestrator | Piper adapter | in-process call or `POST /tts` (audio bytes back) |
| gpu-lease.v1 | transcriber (live) | IMG worker, transcriber JobRunner | file `$XDG_RUNTIME_DIR/gpu-lease/lease.json` |
| transcript.v1 (exists) | transcriber | RAG / analyzer | `POST /transcripts` |

## 1. live.event.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "live.event.v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "session_id", "seq", "type", "at", "stream_time"],
  "properties": {
    "schema": {"const": "live.event.v1"},
    "session_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"},
    "seq": {"type": "integer", "minimum": 0},
    "type": {"enum": ["session_started", "partial", "final", "speaker_update", "turn_end", "session_ended", "warning"]},
    "at": {"type": "string", "format": "date-time"},
    "stream_time": {"type": "number", "minimum": 0},
    "source": {
      "type": "object", "additionalProperties": false,
      "required": ["kind", "sample_rate", "channels"],
      "properties": {
        "kind": {"enum": ["stereo_channels", "mono_diarized", "single_speaker"]},
        "sample_rate": {"type": "integer"},
        "channels": {"type": "integer", "minimum": 1, "maximum": 2}
      }
    },
    "speakers": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "role"],
        "properties": {
          "id": {"type": "string", "minLength": 1},
          "name": {"type": ["string", "null"]},
          "channel": {"type": ["integer", "null"], "minimum": 0},
          "enrolled": {"type": ["boolean", "null"]},
          "role": {"enum": ["interviewer", "candidate", "agent", "customer", "user", "bot", "unknown"]}
        }
      }
    },
    "utterance": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "turn", "index", "speaker", "start", "end", "text", "revision"],
      "properties": {
        "id": {"type": "string", "pattern": "^t[0-9]+\\.s[0-9]+$"},
        "turn": {"type": "integer", "minimum": 0},
        "index": {"type": "integer", "minimum": 0},
        "speaker": {"type": "string", "minLength": 1},
        "channel": {"type": ["integer", "null"], "minimum": 0},
        "start": {"type": "number", "minimum": 0},
        "end": {"type": "number", "minimum": 0},
        "text": {"type": "string"},
        "revision": {"type": "integer", "minimum": 0},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "speaker_certainty": {"type": ["number", "null"], "minimum": -2, "maximum": 2},
        "speaker_provisional": {"type": ["boolean", "null"]},
        "language": {"type": ["string", "null"], "minLength": 2}
      }
    },
    "timings": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "endpoint_ms": {"type": "number", "minimum": 0},
        "asr_ms": {"type": "number", "minimum": 0},
        "attribution_ms": {"type": "number", "minimum": 0},
        "lag_ms": {"type": "number", "minimum": 0, "description": "speech end to emission"}
      }
    },
    "warning": {
      "type": "object", "additionalProperties": false,
      "required": ["code", "message"],
      "properties": {
        "code": {"enum": ["gpu_degraded", "asr_behind", "source_lost", "quota", "other"]},
        "message": {"type": "string"}
      }
    }
  },
  "allOf": [
    {"if": {"properties": {"type": {"enum": ["partial", "final", "speaker_update"]}}}, "then": {"required": ["utterance"]}},
    {"if": {"properties": {"type": {"const": "session_started"}}}, "then": {"required": ["source", "speakers"]}},
    {"if": {"properties": {"type": {"const": "warning"}}}, "then": {"required": ["warning"]}}
  ]
}
```
Semantics: a partial and the final it becomes share the same `id`, and
`revision` increases with each update. A final is never retracted, only
revised by `speaker_update`. `turn` increases when the speaker changes,
the same rule as `transcript.v1`.

Example (final):
```json
{"schema": "live.event.v1", "session_id": "int-2026-11-21-a", "seq": 42, "type": "final",
 "at": "2026-11-21T10:04:12.310Z", "stream_time": 251.9,
 "utterance": {"id": "t7.s1", "turn": 7, "index": 1, "speaker": "S2", "channel": 1,
   "start": 247.2, "end": 251.4, "text": "En mi último proyecto migramos el monolito a servicios.",
   "revision": 3, "confidence": 0.91, "speaker_certainty": null, "speaker_provisional": false, "language": "es"},
 "timings": {"endpoint_ms": 480, "asr_ms": 310, "attribution_ms": 1, "lag_ms": 510}}
```

### Mapping to transcript.v1 (at `session_ended`)

| live.event.v1 | transcript.v1 |
|---|---|
| last revision of each `final` | `utterances[]` (`id`, `turn`, `index`, `speaker`, `start`, `end`, `text`, `confidence`, `speaker_certainty`, `language`) |
| order of finals | `previous_speaker`, `previous_utterance` (computed by the assembler) |
| `speaker_update` | rewrites `speaker` of earlier finals before assembly |
| `speakers[]` (`id`, `name`, `enrolled`) | `speakers[]`. `role` and `channel` are **not** carried (no `transcript.v1` field; a `transcript.v2` only if needed) |
| `source.kind` | `pipeline.diarizer` = `"channels"` / `"sortformer-streaming-v2"` / `null`; `pipeline.recogniser` = `"faster-whisper-large-v3-turbo-int8-stream"` |
| partials, timings, warnings | dropped (kept only in the live log) |
| audio kept (opt-in) | preferred: re-run the batch pipeline and publish that `transcript.v1` instead |

## 2. knowledge.query.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "knowledge.query.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "request_id", "domain", "text"],
  "properties": {
    "schema": {"const": "knowledge.query.v1"},
    "request_id": {"type": "string", "minLength": 1, "maxLength": 64},
    "domain": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"},
    "text": {"type": "string", "minLength": 1, "maxLength": 4000},
    "language": {"type": ["string", "null"], "minLength": 2},
    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
    "mode": {"enum": ["hybrid", "lexical", "semantic"], "default": "hybrid"},
    "filters": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "intent": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "array", "items": {"enum": ["pregunta", "respuesta", "informacion", "opinion", "propuesta", "peticion", "aceptacion", "rechazo", "compromiso", "escucha", "gestion", "relacion", "otro"]}},
        "script_phase": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}}
      }
    },
    "deadline_ms": {"type": ["integer", "null"], "minimum": 1}
  }
}
```
A filter keeps a chunk when, for every filter given, the chunk's value is
one of the listed values. `intent`, `script_phase` and `tags` vocabularies
are declared in `knowledge/domains/<d>/manifest.yaml` and checked at
ingestion. `category` uses the dialogue-act categories
(`src/domain/skills/dialogue_acts.CATEGORIES`).

Example: `{"schema": "knowledge.query.v1", "request_id": "r-91", "domain": "interview.backend", "text": "migración de monolito a microservicios", "language": "es", "top_k": 4, "filters": {"script_phase": ["technical"]}}`

## 3. knowledge.result.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "knowledge.result.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "request_id", "domain", "index_version", "chunks"],
  "properties": {
    "schema": {"const": "knowledge.result.v1"},
    "request_id": {"type": "string"},
    "domain": {"type": "string"},
    "index_version": {"type": "string", "description": "hash of the manifest + files indexed"},
    "chunks": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["id", "text", "score", "source"],
        "properties": {
          "id": {"type": "string"},
          "text": {"type": "string"},
          "score": {"type": "number"},
          "source": {
            "type": "object", "additionalProperties": false,
            "required": ["file_path", "first_character_index", "last_character_index"],
            "properties": {
              "file_path": {"type": "string"},
              "first_character_index": {"type": "integer", "minimum": 0},
              "last_character_index": {"type": "integer", "minimum": 0}
            }
          },
          "metadata": {
            "type": "object", "additionalProperties": false,
            "properties": {
              "title": {"type": ["string", "null"]},
              "intent": {"type": ["string", "null"]},
              "category": {"type": ["string", "null"]},
              "script_phase": {"type": ["string", "null"]},
              "tags": {"type": "array", "items": {"type": "string"}}
            }
          }
        }
      }
    },
    "timings": {"type": "object", "properties": {"total_ms": {"type": "number", "minimum": 0}}, "additionalProperties": false}
  }
}
```
`source` is the existing `MinimalSource` of `Chunk`, so the 42 / BEIR
tools keep working.

## 4. reasoner.request.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "reasoner.request.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "request_id", "role", "template", "context"],
  "properties": {
    "schema": {"const": "reasoner.request.v1"},
    "request_id": {"type": "string", "minLength": 1, "maxLength": 64},
    "role": {"enum": ["judge", "suggester", "conversational", "structured"]},
    "model": {"type": "string", "default": "haiku", "description": "alias: haiku|sonnet|opus|local:<id>"},
    "template": {"type": "string", "pattern": "^[a-z_.]+@[0-9]+$", "description": "prompt template id@version"},
    "language": {"type": ["string", "null"], "minLength": 2},
    "context": {
      "type": "object", "additionalProperties": false,
      "required": ["window"],
      "properties": {
        "session": {"type": "object", "additionalProperties": false, "properties": {
          "id": {"type": "string"}, "type": {"type": ["string", "null"]}, "goal": {"type": ["string", "null"]}}},
        "speaker_roles": {"type": "object", "additionalProperties": {"type": "string"}},
        "window": {"type": "array", "maxItems": 40, "items": {
          "type": "object", "additionalProperties": false,
          "required": ["id", "speaker", "text"],
          "properties": {"id": {"type": "string"}, "speaker": {"type": "string"}, "text": {"type": "string"},
                         "start": {"type": "number"}, "end": {"type": "number"}}}},
        "focus": {"type": ["string", "null"], "description": "utterance id that triggered the call"},
        "script_phase": {"type": ["string", "null"]},
        "covered_phases": {"type": "array", "items": {"type": "string"}}
      }
    },
    "retrieved": {"type": "array", "maxItems": 10, "items": {
      "type": "object", "additionalProperties": false, "required": ["id", "text"],
      "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "metadata": {"type": "object"}}}},
    "payload": {"type": "object", "description": "role-specific; judge: the analyzer's existing candidate JSON"},
    "constraints": {"type": "object", "additionalProperties": false, "properties": {
      "max_output_tokens": {"type": "integer", "minimum": 1, "maximum": 4000},
      "max_suggestions": {"type": "integer", "minimum": 0, "maximum": 5},
      "deadline_ms": {"type": "integer", "minimum": 1},
      "stream": {"type": "boolean"}}}
  }
}
```

## 5. reasoner.response.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "reasoner.response.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "request_id", "role", "status", "model", "backend", "latency"],
  "properties": {
    "schema": {"const": "reasoner.response.v1"},
    "request_id": {"type": "string"},
    "role": {"enum": ["judge", "suggester", "conversational", "structured"]},
    "status": {"enum": ["ok", "no_suggestion", "timeout", "cancelled", "quota", "auth", "error"]},
    "suggestions": {"type": "array", "items": {
      "type": "object", "additionalProperties": false, "required": ["kind", "text"],
      "properties": {
        "kind": {"enum": ["question", "follow_up", "answer", "reminder", "caution"]},
          "signals": {"type": "array", "description": "support only: expressed customer signals from text, each backed by about[]", "items": {"enum": ["complaint", "stated_frustration", "urgency", "churn_risk", "confusion", "satisfaction"]}},
        "text": {"type": "string", "maxLength": 300},
        "rationale": {"type": ["string", "null"], "maxLength": 300},
        "cites": {"type": "array", "items": {"type": "string"}, "description": "chunk ids from retrieved"},
        "about": {"type": "array", "items": {"type": "string"}, "description": "utterance ids"},
        "script_phase": {"type": ["string", "null"]}}}},
    "reply": {"type": ["object", "null"], "additionalProperties": false, "properties": {
      "text": {"type": "string"}, "cites": {"type": "array", "items": {"type": "string"}}}},
    "verdict": {"type": ["object", "null"], "description": "judge: the analyzer's existing JSON, unchanged"},
    "model": {"type": "string", "description": "resolved model id"},
    "backend": {"enum": ["claude-code", "anthropic-api", "local", "opencode"]},
    "usage": {"type": "object", "additionalProperties": false, "properties": {
      "input_tokens": {"type": "integer", "minimum": 0},
      "output_tokens": {"type": "integer", "minimum": 0},
      "cache_creation_input_tokens": {"type": "integer", "minimum": 0},
      "cache_read_input_tokens": {"type": "integer", "minimum": 0},
      "cost_usd_equivalent": {"type": "number", "minimum": 0}}},
    "latency": {"type": "object", "additionalProperties": false, "required": ["total_ms"], "properties": {
      "queued_ms": {"type": "number", "minimum": 0},
      "process_start_ms": {"type": "number", "minimum": 0},
      "first_token_ms": {"type": ["number", "null"], "minimum": 0},
      "total_ms": {"type": "number", "minimum": 0}}},
    "error": {"type": ["object", "null"], "additionalProperties": false, "properties": {
      "kind": {"enum": ["quota", "auth", "transient", "timeout", "invalid_output"]},
      "message": {"type": "string"},
      "resets_at": {"type": ["string", "null"], "format": "date-time"}}}
  }
}
```
`usage` mirrors `src/domain/skills/usage.Usage`. Errors mirror
`QuotaExhaustedError`, `ModelAuthError` and `TransientModelError`. The
suggester's grounding check drops any suggestion whose `cites`/`about`
are not in the request. The suggester never returns ratings of a person
(template rule + test).

Example: `{"schema": "reasoner.response.v1", "request_id": "r-91", "role": "suggester", "status": "ok", "suggestions": [{"kind": "follow_up", "text": "¿Cómo gestionasteis las transacciones entre servicios?", "cites": ["interview.backend:q-arch-07"], "about": ["t7.s1"], "script_phase": "technical"}], "model": "claude-haiku-4-5", "backend": "claude-code", "usage": {"input_tokens": 1450, "output_tokens": 62, "cache_read_input_tokens": 1100}, "latency": {"queued_ms": 0, "process_start_ms": 700, "first_token_ms": null, "total_ms": 1650}}`
(The model id in the example is illustrative; the resolved id is
whatever `claude -p --model haiku` reports.)

## 6. tts.request.v1

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tts.request.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "request_id", "text", "voice"],
  "properties": {
    "schema": {"const": "tts.request.v1"},
    "request_id": {"type": "string", "minLength": 1, "maxLength": 64},
    "text": {"type": "string", "minLength": 1, "maxLength": 1000},
    "language": {"type": ["string", "null"], "minLength": 2},
    "voice": {"type": "string", "description": "Piper voice id, e.g. es_ES-davefx-medium"},
    "speaker": {"type": ["integer", "null"], "minimum": 0},
    "length_scale": {"type": "number", "minimum": 0.5, "maximum": 2.0, "default": 1.0},
    "format": {"enum": ["pcm_s16le", "wav"], "default": "pcm_s16le"},
    "sample_rate": {"type": "integer", "default": 22050},
    "stream": {"type": "boolean", "default": true},
    "interrupt_id": {"type": ["string", "null"], "description": "request this one replaces (barge-in)"}
  }
}
```
Response: audio bytes, with headers `X-Request-Id` and `X-Synth-Ms`.

Implemented in-process by the orchestrator's `TextToSpeech` port
(`orchestrator/adapters/outbound/piper_tts.py`, VB-1): Piper runs as its
own process from its own venv (GPL-3.0, never imported), one process per
voice kept warm. With `stream: true` the text is cut into sentences and
one audio chunk (`SpeechChunk`: `request_id`, `seq`, `text`, `audio`,
`format`, `sample_rate`, `synth_ms`) is yielded per sentence as soon as
it is synthesised; `sample_rate` is the voice's own (22050 Hz for the
`medium` voices), whatever the request says. A later request whose
`interrupt_id` names a running one stops it after its current sentence
(barge-in). No HTTP `POST /tts` exists yet.

## 7. gpu-lease.v1 (file, not HTTP)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "gpu-lease.v1",
  "type": "object", "additionalProperties": false,
  "required": ["schema", "holder", "priority", "pid", "acquired_at", "expires_at", "need_mib"],
  "properties": {
    "schema": {"const": "gpu-lease.v1"},
    "holder": {"type": "string"},
    "priority": {"enum": ["live", "batch"]},
    "pid": {"type": "integer"},
    "acquired_at": {"type": "string", "format": "date-time"},
    "expires_at": {"type": "string", "format": "date-time"},
    "starts_at": {"type": ["string", "null"], "format": "date-time"},
    "need_mib": {"type": "integer", "minimum": 0}
  }
}
```
Valid lease: `expires_at` is in the future and `pid` is alive. Ack file:
`{"schema": "gpu-lease.ack.v1", "holder": "img-worker", "freed": true, "at": "..."}`.
