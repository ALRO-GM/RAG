# Stage 1 end to end (measured 2026-11-21)

What was run: `python -m orchestrator replay` with the HTTP knowledge
adapter (`POST /knowledge/query`) and the HTTP reasoner adapter
(`POST /reason`, warm Haiku backend) against the **running** RAG service
on `127.0.0.1:8000`, on two synthetic Spanish calls (fictional, written
for this check, ~27 utterances each, `transcript.v1`):

- `examples/platform/sim-soporte-lumbre.transcript.v1.json`: customer
  service of Lumbre Hogar (domain `support.demo`), product config
  `experiments/platform/e2e_config/live_assist.support.yaml`.
- `examples/platform/sim-entrevista-backend.transcript.v1.json`: backend
  interview (domain `interview.generic`), config
  `.../live_assist.interview.yaml`.

The two configs are the repository's `live_assist.*` products with only
the knowledge domain changed to the seed domains (the shipped products
name `support.general` and `interview.backend`, which do not exist yet).
Everything else is unchanged: trigger on `final` of the customer or
candidate, min 4 words, `deadline_ms: 1800`, `--speed max` (each call
runs to its end). Domains were indexed with `python -m src domain index`
(65 and 67 chunks); the service picked the indexes up without a restart.

40 real Haiku calls in total (13 + 13 + 14); budget 40. Raw files:
`results/*.suggestions.jsonl` (one record per call, with stages and
signals), `results/*.events.jsonl`; summary `stage1_e2e.json`
(`stage1_e2e_summary.py`).

## Results

| Run | calls | ok | timeout | no_suggestion |
|---|---|---|---|---|
| support, run 1 | 13 | 3 (23 %) | 10 (77 %) | 0 |
| support, run 2 | 13 | 5 (38 %) | 8 (62 %) | 0 |
| interview | 14 | 10 (71 %) | 3 (21 %) | 1 (7 %) |

Latency in ms, p50 / p90 (nearest rank). "all" counts every call; a
timeout is pinned at the 1800 ms deadline plus overhead, so it says
nothing about Haiku's speed. "ok" counts only answered calls.

| Run | | knowledge_ms | reasoner_ms | total_ms |
|---|---|---|---|---|
| support 1 | all | 22.4 / 54.5 | 1820.4 / 1826.1 | 1842.9 / 1874.9 |
| support 1 | ok | 15.3 / 21.2 | 973.3 / 1342.4 | 988.6 / 1363.7 |
| support 2 | all | 14.0 / 21.5 | 1820.7 / 1823.4 | 1834.9 / 1844.5 |
| support 2 | ok | 12.7 / 22.3 | 1182.2 / 1566.8 | 1193.7 / 1586.2 |
| interview | all | 17.0 / 26.0 | 1287.4 / 1837.8 | 1308.1 / 1854.4 |
| interview | ok | 17.0 / 22.4 | 1205.4 / 1656.0 | 1217.8 / 1678.3 |

Findings:

1. **Knowledge is fast** (p90 under 60 ms) except the very first query
   after the service start: 2003 ms and reported `unavailable`
   (HttpKnowledge timeout 2.0 s). Issue #176.
2. **The 1.8 s reasoner budget is met only when the process is warm.**
   Timeouts come in bursts (support run 1: calls 1-2 and 6-13; run 2:
   1-5 and 11-13), all with `first_token_ms: null`. Reading
   `claude_code_stream.py`: a timeout kills the process and the next call
   starts a new one inside its own deadline, so it times out as well
   until one fits. Answered calls took p50 ~1.0-1.2 s, in line with
   `docs/live/HAIKU_LATENCY.md`. This is a hypothesis from the code and
   the pattern, not a profiled cause. Issue #175. Until it is fixed the
   live support product would show suggestions on roughly a third of the
   customer turns on this machine (support) to 70 % (interview).
3. Interview: all 10 ok calls cited a seed chunk (support run 2: 4 of 5);
   the suggestions read as on topic to the author; one call returned `no_suggestion` (valid).
4. Support signals are partly doubtful: call 6 tagged `confusion` for
   «Está completo, solo la probé un día…», which expresses no confusion.
   Three ok calls carried signals (`confusion`, `complaint`, `complaint`,
   `urgency`); the customer's explicit statements of frustration ("estoy
   bastante harto", "me doy de baja") fell in timed-out calls in run 2,
   so recall of signals could not be judged. Nothing here is a measured
   accuracy: 13 calls, one script, no labels.

## Examples (support, run 2)

- «Está completo, solo la probé un día, la bombilla se fundió enseguida.»
  `question` [confusion]: ¿Me confirma si el producto tiene el embalaje interior y los accesorios completos? (cita `support.demo:s60`)
- same turn, `reminder` [complaint]: Confirme que la devolución queda abierta hoy y que la bombilla fundida se registra como motivo.
- «Perfecto, porque yo no puedo ir a ninguna tienda entre semana.»
  `caution` []: No ofrezca recogida en tienda; el cliente no puede ir entre semana. (cita `support.demo:s31`)
- «Me vendría bien el jueves por la tarde, si es posible.»
  `question` []: ¿Le confirmo el jueves por la tarde? Indíqueme la dirección y un teléfono de contacto.
- «Necesito que me lo devuelvan ya, esto es urgente…» (call 10)
  [urgency]: ¿Cuál es la fecha límite exacta en la que necesita que el reembolso…

## Examples (interview)

- «Rediseñé el servicio de rutas…» `follow_up`: ¿Qué cambió exactamente al procesar en lotes, y cómo medisteis el impacto? (cita `interview.generic:guia-general.md#72`)
- same turn, `follow_up`: ¿Qué decisiones técnicas tomaste para el procesamiento por lotes y por qué?
- «Yo lideré el diseño y escribí casi todo el código…» `follow_up`: ¿Hubo algún desacuerdo con tu compañero en la revisión? ¿Cómo lo hablasteis? (cita `interview.generic:i32`)
- «Una vez discrepamos sobre migrar…» `question`: Explícame qué es una migración de base de datos como si fuera una persona de negocio. (cita `interview.generic:i23`)
- «Sí, primero añado pruebas de caracterización…» (call 11) `follow_up`: Entonces, ¿primero escribes pruebas que fijan el comportamiento…

## Limits

Synthetic scripts written by the author; one run per scenario (two for
support); the service had been restarted shortly before; no human
judged the suggestions. The p50/p90 are over 13-14 calls.
