# AI interviewer (VB-8, #208)

The voice-bot page (`#/voicebot`) now has a **topic**: the support
assistant (`voicebot`, whose knowledge domain can be changed on the
page) or a **job interview held by an AI interviewer**
(`voicebot.interview`). The interviewer works from a job offer and an
interview plan the recruiter reviews first; it asks one question at a
time, follows up until each *mínimo* slot has an answer, keeps the
time, closes politely, and never gives the candidate a score or
feedback. The recruiter sees a live coverage panel and, at the end, a
visual report; the transcript (both voices) and the filled slot sheet go
to the post-call analysis.

Design: `docs/live/VOICEBOT.md` (the voice bot it runs on),
`docs/KNOWLEDGE.md` (domains and uploads), `docs/SETTINGS.md`
(«Ajustes»). Contracts: `contracts/interview.plan.v1.schema.json`,
`contracts/interview.sheet.v1.schema.json` (examples in
`contracts/examples/`).

```
job offer (pasted, or a PDF/DOCX uploaded to a knowledge domain)
   │  GET /knowledge/domains/{d}/documents/{doc}/text   (RAG)
   ▼
InterviewPlanner ── POST /knowledge/query (interview.generic, method corpus)
   │             ── POST /reason  role structured, interview.planner@1
   ▼
interview.plan.v1 ── page: review (imprescindible, level, weight, questions…) ── PUT
   │
   ▼  POST /sessions {"product": "voicebot.interview", "plan", "candidate"}
VoiceBot + InterviewDirector
   each candidate final → slot filler (interview.slots@1 or rules)
                        → Agenda: open | ask | probe | clarify | answer | close
                        → interviewer@1 words the step (warm Haiku, streamed, Piper)
                        → {"type": "coverage"} to the page
   end of session → interview.report@1 scores the slots against the offer
                  → interview.sheet.v1 (kept with the plan) + transcript.v1
                  → POST /transcripts, optionally POST /analysis/sessions/{id}
```

## 1. Topics and products

`config/products/voicebot.interview.yaml` is a second voice-bot product:
one candidate on the microphone (`role: candidate`), VAD hang 700 ms
(candidates pause to think; the support bot keeps 400 ms; not
measured), the conversational template `interviewer@1` with a 4 s
deadline, Piper `es_ES-davefx-medium`, knowledge `interview.generic`,
and an `interview` section (`InterviewSection` in
`orchestrator/domain/config.py`): corpus, planner/filler/report
templates, `filler` (`haiku` | `heuristic`), `max_probes` (1),
`minutes` (20), `report` (true), `post_call` (false).

`GET /products` now says for each product its `domain` and whether it
is an `interview`; the page lists every product whose `ui.kind` is
`speaker`. A plain voice-bot session may name another knowledge domain
(`POST /sessions {"domain"}`, checked against the domain-id pattern);
an interview session needs a `plan` and takes an optional `candidate`
label (only for the comparison; the bot never says it).

## 2. The interview-method corpus

`knowledge/domains/interview.generic/metodo-entrevista.jsonl`: 170 own,
fictional records in Spanish, beside the 64 seed records (unchanged).
The chunk metadata schema is closed (`title`, `intent`, `category`,
`script_phase`, `tags`), so the role type and the competency are tags:

| Metadata | Values |
|---|---|
| `script_phase` | opening, experience, technical, behavioural, closing |
| `tags` role | `puesto_todos` (115), `puesto_tecnico` (10), `puesto_atencion_cliente` (10), `puesto_almacen` (10), `puesto_hosteleria` (9), `puesto_comercio` (8), `puesto_oficina` (8) |
| `tags` competency | `comp_<id>` of the rubric's 11 competencies (`knowledge/rubrics/core.es.json`), 6 to 9 records each |
| `tags` kind | `star` (46), `repregunta` (36), `requisito` (19), `legal` (17), `estructura` (14), `exploracion` (13), `evidencia` (11), `cierre` (8), `ejemplo_respuesta` (6) |

Content: structure and timing of a 15–30 minute interview and how to
introduce an AI interviewer; per role type what to explore, requirement
questions and a good vs a vague answer; per competency what a complete
answer shows, STAR questions and follow-ups; probing techniques (STAR,
"¿qué hiciste tú?", figures, when to stop); closing without feedback;
questions that must not be asked in Spain/EU (age, family, pregnancy,
religion, ideology, union membership, sexual orientation, health or
disability, origin, criminal record unless the job requires it) and how
to redirect, with the legal bases (Constitución art. 14, Estatuto de
los Trabajadores arts. 4.2.c and 17, Ley 15/2022, LO 3/2007, RGPD art. 9
and minimisation, Directivas 2000/78/CE and 2000/43/CE). The citations
were written from general knowledge and not checked against the
official texts: a lawyer should review them before the corpus is used
with real candidates. Counts are checked by
`tests/integration/test_seed_domains.py`.

## 3. The plan (`interview.plan.v1`)

`POST /interview/plans` (orchestrator; the page goes through the
transcriber's `/api/assist/interview/plans`): `{"offer": text}` or
`{"document": {"domain", "id"}}` (the RAG's new
`GET /knowledge/domains/{d}/documents/{doc}/text`, indexed documents
only, front matter cut), optional `title`, `role_type`, `minutes`. The
planner queries the method corpus (filtered by `puesto_<role_type>` and
`puesto_todos` when the role type is given) and calls the reasoner with
role **`structured`** (new: one JSON object in the response's new
`data` field) and template `interview.planner@1`, which asks for 6–10
slots: the offer's required items as `minimo`, nice-to-have as
`deseable`, 2–4 rubric competencies with STAR questions, anything else
`opcional`, each with question ideas, the evidence wanted and a
**weight 1–5**. The draft is validated in
`orchestrator/domain/interview.py`: unknown levels become `deseable`,
weights are clamped, competency ids checked, at most 12 slots, at least
one `minimo`, and **any slot or question about a protected
characteristic is dropped** (an accent-folded word list; "políticas de
devolución" and "salud laboral" pass, "¿tienes hijos?" does not). When
the model fails or gives nothing usable the plan is a generic one
(`planner.status: "fallback"`, with the error) for the recruiter to
edit. Plans live in `data/interview/plans/<id>.json`
(`--interview-dir`).

Review on the page: a checkbox per slot marks it *imprescindible*
(`minimo`), plus level, weight, competency, questions (one per line),
evidence, add/remove, title and time budget. `PUT
/interview/plans/{id}` carries the revision it was based on (an older
one: 409); an edit with a protected question or without a `minimo` is
refused.

## 4. The interview loop

`InterviewDirector` (`orchestrator/application/interview.py`) is the
voice bot's **turn director** (`VoiceBot(director=...)`): the bot asks
it what each turn does and sends its `payload` to `interviewer@1`; the
model only words the step. The bot **speaks first** (`VoiceBot.open`),
and finals after the close are skipped (`skipped.closed`).

After each candidate final the **slot filler** says which slots the
answer covers (`complete` or `partial`, with a quote):

- `haiku` (default): `interview.slots@1` on the reasoner (warm pool,
  6 s deadline); slot ids are checked against the plan and a quote that
  is not the candidate's words is replaced by the answer's first words.
  A failed call falls back to the rules for that answer.
- `heuristic`: no model; length, first-person past-tense verbs, figures,
  result words and words of the slot
  (`orchestrator/domain/interview_agenda.py`, `heuristic_fill`).

The **agenda** (`Agenda`, same module) then decides, per the owner's
rules:

1. `open`: greet, say it is an AI interviewer and the length, first
   `minimo` slot.
2. Answer covers the slot → next slot. The candidate says they do not
   know the topic («no lo conozco», «no tengo experiencia con…») →
   status **`unknown` («no lo conoce»)**, no more questions on it.
3. Not covered → **one** follow-up (`probe`, `max_probes`); still not
   covered → **`clarify`**: «No me ha quedado del todo claro esto;
   ¿quieres matizarlo o pasamos a la siguiente pregunta?». A skip
   («pasamos», «siguiente pregunta»…) marks it **`skipped`** (a possible
   gap); a refinement is filled again; either way the agenda moves on.
4. A question of the candidate is answered (`answer`, once per slot),
   then the slot is asked again.
5. `deseable` slots only once every `minimo` is settled and 30 % of the
   time is left; `opcional` with 50 % left.
6. `close` when nothing is left, when the time is up, or after 40
   turns: thanks, next steps from the team, goodbye; no question, no
   score.

`interviewer@1` (fixed system prompt, so it has its own warm Haiku
pool; the plan step goes in the user prompt) forbids scores, praise or
criticism and protected topics. The page's **coverage panel** (only
the recruiter sees it) shows each slot's status
(cubierto / parcial / sin respuesta / saltado / no lo conoce), the slot
being asked, attempts and quotes, and the time left. When the closing
sentence has played the page stops the session.

Each turn record has a new stage `director_ms` (the filler and the
agenda) inside the turn's latency: with the Haiku filler the next
question waits for that call.

## 5. Result: sheet, report, comparison, analysis

When the session ends the director:

1. **Scores** each slot with `interview.report@1`: 0–100 **relative to
   what this offer requires** (100 = meets the requirement for this job,
   e.g. junior level for a junior job), `above` (evidence beyond the
   requirement), the model's `confidence` 0–1, a note; plus «otras
   habilidades útiles para el puesto» with quotes and a short neutral
   summary. `unknown` stays 0; a slot never answered stays unscored.
   Without the model (failure or `report: false`) a coarse score from
   the status (filled 70, partial 40, skipped 10, unknown 0) with
   confidence 0.2 (`scored_by: "status"`).
2. Computes the **weighted suitability**: Σ weight·score / Σ weight over
   the scored slots, with unscored `minimo` slots counting 0.
3. Writes the **`interview.sheet.v1`** (answers point at the
   transcript's utterance ids) to
   `data/interview/results/<plan>/<session>.json`.
4. Builds the **`transcript.v1` with both voices** (bot `BOT`
   "Entrevistador IA", candidate `S1`; `session.type: "entrevista"`)
   and posts it to the RAG (`POST /transcripts`); with `post_call` it
   starts the analysis with the sheet as the **job sought** (title,
   summary, each slot's level, weight, status, score and quotes, at most
   4000 characters). The transcriber's own handoff is turned off for
   this product, so it does not overwrite the transcript with the
   candidate's side only.

The page draws the **report** (`views/interview.js`): the «borrador,
revisar por una persona» notice, the suitability in a coloured ring,
tiles (cumple / carencias / por encima / no lo conoce / saltados), a
bar per slot (green ≥ 70, amber ≥ 40, red below; heaviest first) with
the note, confidence and quotes, the gaps, the other skills, the facts
of the run and the link to the analysis. «Comparar candidatos de esta
oferta» lists the interviews of the plan (`GET
/interview/plans/{id}/results`): suitability and one bar per slot,
best first, marked when the plan was edited after the interview.

Screenshots (fakes, no model): plan review, live coverage and report
in the session's scratchpad (`interviewer/plan.png`, `coverage.png`,
`report.png`), drawn by a harness page from a fake interview run.

## 6. Settings («Ajustes», group «Entrevistador de IA»)

| Step | Options (first is the default) |
|---|---|
| `interview.slot_filler` | `haiku` · `heuristic` |
| `interview.max_probes` | `1` · `0` · `2` |
| `interview.minutes` (new plans) | `20` · `15` · `30` · `45` |
| `interview.post_call` | `off` · `on` |

`voicebot.reasoner_model` and `voicebot.voice` apply to the interviewer
too. The time budget of each plan is edited in its review.

## 7. What was run (no measurement)

The owner will test with two real interviews (shop assistant, backend
developer), so no synthetic interview was measured: no turns-to-minimums,
slot-fill accuracy or first-audio figure exists yet for the interviewer.

Smoke test with real Haiku (`experiments/voicebot/interviewer_smoke.py`,
result `experiments/voicebot/interviewer_smoke_results.json`,
2026-11-21): **8 calls**, all `ok`, through the in-process reasoner
(`claude -p`, one cold process per call, so the times are not the live
bot's; 39.8 s in all). The planner drafted 9 slots for the shop
assistant offer (6 `minimo`, incl. INIT and COMM competency slots) and
12 for the backend offer; no kept slot is about a protected topic (how
many the filter dropped was not recorded). The text
interview (opening + two scripted answers) worded `open`, `ask` and
`probe` as asked; the filler marked the first answer `s1` complete and
`s2` partial, the second (an availability answer to the till
question) `s3` complete, and the agenda then probed `s2`. The report
scored the three answered slots 80, 55 and 90 with quotes and notes.
It also gave 0 to the six slots that were never asked; that run led to
the rule above (an unanswered slot stays unscored), applied after it
(no new call): the suitability of that sheet goes from 31.4 to 41.4.

## 8. How to run the test (owner)

1. Restart the services after merging (`ragputin-servicios restart`):
   the RAG gets the new templates and warms the `interviewer@1` and
   `interview.slots@1` pools; the orchestrator gets the
   `voicebot.interview` product and `data/interview/`.
2. Optional: upload each job offer (PDF/DOCX) to a knowledge domain in
   «Conocimiento» and wait until it is *indexado*.
3. «Asistente de voz» → Tema: «Entrevista de trabajo con IA». Paste the
   offer (or pick the document), choose the kind of job and the length,
   «Preparar el plan» (one Haiku call, 5–60 s).
4. Review the plan: tick the *imprescindibles*, adjust weights,
   questions and time; «Guardar cambios». Write a candidate label.
5. Headset on, «Empezar la entrevista». Answer as the candidate; watch
   the coverage panel. Try: a vague answer (one follow-up, then the
   «matizar o pasar» offer), «no lo conozco», «pasamos a la siguiente»,
   a question to the interviewer.
6. At the end read the report; «Comparar candidatos de esta oferta»
   after the second interview with the same plan. With
   `interview.post_call: on` the analysis starts by itself; otherwise
   start it from the session.
7. What to note for the measurement: turns until every *imprescindible*
   was settled (the report's facts line), the slots the filler got
   wrong (coverage panel vs what was said), and the latency line under
   the conversation (end of the answer → first voice).

Haiku calls per interview: 1 (plan) + 2 per answer (filler and
interviewer; 1 with the `heuristic` filler) + 1 (report); for a
20-minute interview of 12–16 answers that would be about 25–35 (an
estimate from that count, not measured). `post_call: on` adds the analysis's calls.

## 9. Limits

- Synthetic only: the planner, filler, agenda thresholds and the report
  prompt were tried on two invented offers and two answers.
- The rules filler and the «no lo sé» / «pasamos» detection are Spanish
  word lists; English interviews are not handled yet.
- Bot utterance times in the transcript come from the orchestrator's
  clock (session start ≈ transcriber stream start); a turn cancelled by
  a newer final has already advanced the agenda.
- One interview at a time (the voice bot's one live session).
- Privacy: the sheet holds candidate quotes and scores
  (`data/interview/results/`); retention and who sees it are pending
  decisions, as for the rest of the analysis.
