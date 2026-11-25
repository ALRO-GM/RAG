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
VoiceBot + InterviewDirector + TurnTaking (#213)
   candidate finals add up; the answer ends after 3.5 s of silence (§5)
   each answer → slot filler (interview.slots@1 or rules)
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

`knowledge/domains/interview.generic/metodo-entrevista.jsonl`: 177 own,
fictional records in Spanish, beside the 64 seed records (unchanged).
The chunk metadata schema is closed (`title`, `intent`, `category`,
`script_phase`, `tags`), so the role type and the competency are tags:

| Metadata | Values |
|---|---|
| `script_phase` | opening, experience, technical, behavioural, closing |
| `tags` role | `puesto_todos` (122), `puesto_tecnico` (10), `puesto_atencion_cliente` (10), `puesto_almacen` (10), `puesto_hosteleria` (9), `puesto_comercio` (8), `puesto_oficina` (8) |
| `tags` competency | `comp_<id>` of the rubric's 11 competencies (`knowledge/rubrics/core.es.json`), 6 to 9 records each |
| `tags` kind | `star` (46), `repregunta` (36), `requisito` (20), `legal` (20), `estructura` (15), `exploracion` (15), `evidencia` (11), `cierre` (8), `ejemplo_respuesta` (6) |

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
were re-checked on 2026-11-24 against the BOE consolidated texts and the
EU Official Journal (#234, `docs/live/INTERVIEW_KNOWLEDGE.md`): the
records now give article numbers read in those texts, and the machine-
readable list of forbidden questions lives in
`knowledge/interview/forbidden-questions.json`. Not legal advice: a
lawyer should still review it before the corpus is used with real
candidates. Counts are checked by
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
`data` field) and template `interview.planner@2` (`@1` stays
registered; `@2` adds each slot's concepts, §4.1), which asks for 6–10
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
evidence, **concepts to cover (one per line, up to 6; §4.1)**,
add/remove, title and time budget. `PUT
/interview/plans/{id}` carries the revision it was based on (an older
one: 409); an edit with a protected question or without a `minimo` is
refused.

## 4. The interview loop

`InterviewDirector` (`orchestrator/application/interview.py`) is the
voice bot's **turn director** (`VoiceBot(director=...)`): the bot asks
it what each turn does and sends its `payload` to `interviewer@2`; the
model only words the step. The bot **speaks first** (`VoiceBot.open`),
and answers after the close are skipped (`skipped.closed`). With a
candidate profile (#235, `docs/live/CANDIDATE_PROFILE.md`) the opening
and the next turns ask the candidate's open core questions first (not
the work-mode one, whose blanks the plan cannot fill), and the salary
expectation is asked just before the close; answered ones never. When an
answer ends is decided by the turn-taking of §5, not by each final.

After each candidate answer the **slot filler** says which slots the
answer covers (`complete` or `partial`, with a quote):

- `haiku` (default): `interview.slots@2` on the reasoner (warm pool,
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

### 4.1 Concepts per point (#258)

A point (slot) of the plan can carry **`points`**: 0 to 6 short concrete
concepts (`{"id": "p1", "text": "Tipos de pisada"}`) a complete answer
talks about. `interview.planner@2` asks the model for them: for
knowledge, technical and requirement points what a person who knows the
topic would mention («tipos de pisada», «amortiguación y drop», «cómo
elegir la talla»); for competency points the parts of a complete
example («situación concreta», «lo que hizo él o ella», «resultado») or
what is specific to this job. `plan_from_draft` keeps strings or
`{"text"}` objects, trims them to 80 characters, drops empty, repeated
and protected-characteristic ones (same word list as the slots) and
numbers them; the fallback plan has generic ones (where / how long /
which tasks; the parts of an example; shifts and start date). The field
is additive: plans saved before it load with no concepts and behave as
before.

**What decides "treated".** After each answer the filler also says
which concepts of the slot it treats, with the candidate's words:

- `haiku` filler, `interview.slots@2`: the model lists, per slot, the
  concepts the answer talks about (in their words or a synonym) as
  `treated`, or `unknown` when the candidate says they do not know it;
  unknown concept ids and statuses are dropped and a quote that is not
  the candidate's words is replaced by theirs. Only when the reply gives
  the slot asked **without** a `points` field do the rules below decide
  its concepts; when the reply leaves the slot out (it covers none), the
  model's word stands and nothing is ticked. Reply cap 800 tokens (3
  slots × 6 concepts with quotes; an estimate, not measured); a cut
  reply falls back to the rules for that answer.
- rules (`orchestrator/domain/interview_points.py`, `point_hits`), the
  fallback of the model and the live page's matcher: the answer is cut
  at `.`, `;`, `?`, `!`, «pero» and «aunque»; a part treats a concept
  when it contains **all** the concept's content words if it has one
  or two (first five letters of each word of four letters or more;
  «tipo», «cómo», «cuánto», «tiempo», «hacia», «siempre»… do not count),
  at least half if it has more, or a cue of the concept's kind: a figure
  with years, months or weeks for «cuánto tiempo»; a first-person past
  verb («expliqué», «decidí», «hice») for «lo que hizo»; a result verb
  («conseguimos», «se resolvió», «redujimos») for «resultado»; «una
  vez», «cuando trabajaba…», «hace dos años» for «situación»; shifts,
  weekend days or «jornada» for «horario o turnos»; «empezar»,
  «incorporarme» for the start date; «trabajé de/en», «estuve … en una
  tienda» for the workplace; «me encargaba», «atendía» or a job gerund
  with its object («atendiendo clientes») for tasks. Concepts are only
  read for the slot asked, never for another slot the answer touches.
  A part saying «no lo conozco» / «no sé» and naming one word of a
  concept makes it `unknown`.

What the rules can and cannot do: they tick a concept only from its
own words or those cues, and they were written to prefer missing a
concept (it stays pending and the follow-up asks for it) to ticking one
the candidate did not talk about. Filler words do not tick anything:
«así es la vida», «aquí estoy», «cuando quieras», «siempre llego a
tiempo», «voy hacia allí», «estoy pensando», «esta tarde no puedo» and
«no sé qué decirte» leave every concept pending (unit tests). They do
**not** understand synonyms or paraphrases («elijo el número» is not
«cómo elegir la talla»; «la amortiguación» alone is not «amortiguación
y drop», which needs both words), so a concept written in words the
candidate will not use is rarely ticked by them, and a concept that
shares a word with an unrelated sentence can still be ticked (for
example «Tipos de pisada» by «dejé una pisada en el suelo mojado»). The Haiku filler
reads meaning and is the default for that reason; on the live page
there is no model and the rules are all there is.

"Treated" only means the candidate **talked about** the concept: it is
a hint for the interviewer, not a judgement of whether what they said
is right (that is the next ticket). Nothing about concepts is ever said
to the candidate.

**Completion rule** (`points_status`), for a slot with concepts, in
place of the filler's slot status:

- every concept treated → `filled` (the agenda moves on);
- every concept `unknown` → `unknown` («no lo conoce»);
- none pending, some treated and some unknown → `partial`, settled (the
  agenda moves on: nothing is left to ask);
- any concept pending → still open: the agenda follows up as in §4.

A candidate saying they do not know settles concepts instead of leaving
them pending forever: the ones they name in that sentence, or else the
ones the last follow-up asked about (all pending ones after the first
question).

**Follow-up.** For a slot with concepts, `probe` and `clarify` name the
pending ones, **at most three at once** (`MAX_ASKED_POINTS`), and the
ones already treated: the payload carries `pending_points` and
`treated_points`, and `interviewer@2` (same rules as `@1` plus this one)
says «Me has hablado de A. ¿Y de B y C?» or «No me ha quedado claro lo
de B y C; ¿quieres contarme algo o pasamos a la siguiente pregunta?»,
never that something is missing or how much was covered. Slots without
concepts keep the generic follow-up.

**Coverage and sheet.** The coverage panel lists, under each point, its
concepts with a sign and a word (✓ tratado / ○ aún no / ? dice que no lo
conoce; never the colour alone) and a line «Conceptos: N de M
tratados». The `interview.sheet.v1` slot gets `points`: id, text,
status, quote and the transcript utterance id that settled it (pointed
at the transcript like the answers). The report shows the count under
each bar and, under «Más información», each concept with its status and
quote.

**Live page with a person interviewing** (`live_assist.interview` with
a role and a candidate; `TopicTracker`,
`orchestrator/application/next_round.py`). When the interviewer asks
about a topic (its label, questions or concepts, as before), each
candidate answer to it ticks the concepts it names by the rules above;
the topic is done (✓) when none is pending and at least one was
treated, and one asked but not done stays «volver a tratar» for the next
round. «Temas del puesto» marks the topic being asked and shows
«Falta por tratar en «tema»: B, C» plus, under «Conceptos de cada tema»,
every topic's concepts with their status in this round (the status
line keeps one live region and the list is redrawn inside a disclosure
that keeps its state, so a screen reader hears the change). The pending
concepts of the topic being asked also go into the **goal of the
existing suggester call** (`suggester.interview@2`, «Tema en curso: …;
conceptos aún sin tratar: B y C.»), so the suggestions steer to them
with no extra model call. Which concepts were treated is decided by the
rules, not by the suggester: its reply is read only for the contract's
suggestion fields (anything else is dropped on purpose), so returning
concepts from it would need a change of `reasoner.response.v1`. The
scripted interview (`#/live/demo`, `entrevista-lucia.demo.json`) ticks
its topics' concepts by prepared lines (`asked_by`, `points[].covered_by`;
the spoken lines are unchanged), and leaves «Pruebas de integración»
pending. A topic becomes current only on a line that asks about it; the
opening «proyecto del que estés orgullosa» asks about none, so «Python y
APIs» becomes current when Lucía's answer covers it.

Not measured: how often the rules or Haiku mark a concept the candidate
did talk about (or miss one) has not been measured on any interview;
only invented examples in the tests.

## 5. Turn-taking: waiting for the candidate (VB-10, #213)

The owner's first test showed the interviewer cutting the candidate
off: every transcriber final (VAD hang 700 ms) ended the turn. Now an
interview session's `VoiceBot` has a **turn-taking policy**
(`orchestrator/domain/turn_taking.py`, `TurnTaking`: pure, time passed
in) and the session ticks it every 0.1 s (`LiveSessions(ticker=...)`,
a thread beside the event stream). The support voice bot keeps
answering each final.

| Rule | Default («Ajustes») | What happens |
|---|---|---|
| Think time | `interview.think_s` 6 s (3–15) | After the bot finishes a question (estimated end of its audio) nothing happens for 6 s. |
| End of the answer | `interview.turn_silence_s` 3.5 s (2–6) | The candidate's finals are **accumulated into one answer**; it ends only after 3.5 s of silence after the last speech. Any partial, final or voice activity restarts the wait. |
| Unfinished answer | up to `interview.max_wait_s` 10 s (5–20) | If the answer *looks unfinished* when the silence is reached (`looks_unfinished`: last word a conjunction, connector, filler, article or preposition — «y», «pero», «porque», «o sea», «entonces», «eh», «de»… — or a trailing comma, colon, dash or ellipsis; no model call) the bot waits up to 10 s. |
| No answer | `interview.max_wait_s` 10 s | 10 s of complete silence after a question (no voice, no noise): the bot **re-asks once** in other words (step `reask`). A second silence as long ends the turn with an empty answer, which the agenda treats as an answer that covers nothing (follow-up, then «matizar o pasar»). |
| Stuck | `interview.stuck_s` 7 s (4–15) | Silent for 7 s after part of an unfinished answer: one brief **hint** on how to go on, tied to the slot, ending in a choice question («¿Te referías a …, o prefieres hablar de …?», step `hint`, with the answer so far). At most one per slot; the answer heard so far is kept. |
| Never talk over | — | Speech of the candidate (a partial or final with words, or the page's voice detector) while a turn is being **prepared** drops it (`interrupted_by: "candidate"`, or `"vad"` from the page); a turn dropped while the slot filler runs does not move the agenda. Barge-in while the bot **speaks** works as before. A partial never drops the opening; if the page's detector stops it, the wait for an answer starts anyway. |
| Active listening | `interview.backchannel` `voice` (`visual`, `off`) | At natural pauses of a long answer (pause ≥ 1.2 s, shorter than the end of the answer, ≥ 8 s after the previous one and after the answer started): «Ajá.», «Vale.», «Entiendo.», «Mm-hm.» in turn, synthesised once when the session starts and played by the page at gain 0.35, beside the bot's voice queue (never counted as the bot speaking), plus an «Escuchando…» tag for 2 s. `visual`: the tag only. |

`reask` and `hint` are two new actions of `interviewer@1` (its system
prompt says how to word them; the user prompt carries `Answer so far`);
they do not count as agenda turns. The page now reports its voice
detector's starts during an interview even while the bot is silent
(`POST …/barge-in`, which also drops a turn being prepared), so a
candidate who resumes after a pause restarts the wait at once rather
than at their first partial. The 1.2 s pause, the 8 s gap and the gain
are constants of the code, not settings.

### Replay (simulation, not a recording)

`experiments/voicebot/turn_taking_replay.py` (results in
`turn_taking_replay_results.json`, 2026-11-21) feeds scripted answers
to the real `VoiceBot` + `TurnTaking` on a simulated clock: 40 answers
of 3–5 fragments, 125 pauses drawn from 2.02 to 4.96 s (34 over 4 s),
the final 0.674 s after each fragment (the p50 of
`docs/live/LIVE_ASR.md`), a partial every 0.8 s, the page's voice
detector 0.2 s after speech starts (150 ms `startMs` plus 50 ms
assumed), and the bot's first audio 0.912 s after it decides (the
support bot's p50 in `docs/live/VOICEBOT.md`; the interviewer's slot
filler adds an unmeasured call). A *premature decision* is a turn
decided before the answer's last final; a *premature turn* is one that
was not dropped and spoke.

| Pauses come after | `turn_silence_s` | Voice detector | Premature decisions | Premature turns (answers cut) |
|---|---|---|---|---|
| an unfinished fragment («…y», «…porque», «…o sea,») | 3.5 | on | 0 | 0 |
| an unfinished fragment | 3.5 | off | 0 | 0 |
| a complete sentence | 3.5 | on | 35 | 3 (3 of 40) |
| a complete sentence | 3.5 | off | 72 | 34 (25 of 40) |
| a complete sentence | 5 | on | 0 | 0 |
| a complete sentence | 5 | off | 11 | 0 |
| a complete sentence | 6 | on or off | 0 | 0 |

So, in this simulation: zero premature turns with pauses of 2–5 s
whenever the long pauses come after an unfinished fragment, at the
default 3.5 s. A pause of more than about 4 s after a **complete**
sentence (3.5 s of silence plus the 0.674 s the final takes) is, by
construction, the end of the answer at 3.5 s; the voice detector then
drops most of those turns before they speak (32 of 35), and
`interview.turn_silence_s` 5 removed them all here. Every answer ended
3.5 s (5 s, 6 s) after its last final, as configured. Not measured
with real speech: real pauses, ASR timings and the energy detector on
a real microphone may differ.

## 6. Result: sheet, report, comparison, analysis

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

## 7. Settings («Ajustes», group «Entrevistador de IA»)

| Step | Options (first is the default) |
|---|---|
| `interview.slot_filler` | `haiku` · `heuristic` |
| `interview.max_probes` | `1` · `0` · `2` |
| `interview.minutes` (new plans) | `20` · `15` · `30` · `45` |
| `interview.post_call` | `off` · `on` |
| `interview.turn_silence_s` | `3.5` s, a slider from 2 to 6 |
| `interview.think_s` | `6` s, 3 to 15 |
| `interview.max_wait_s` | `10` s, 5 to 20 |
| `interview.stuck_s` | `7` s, 4 to 15 |
| `interview.backchannel` | `voice` · `visual` · `off` |

The turn-taking values are the owner's (issue #213), not measured with
real candidates; they are also fields of the product's `interview`
section (`InterviewSection`), which the saved settings replace.

`voicebot.reasoner_model` and `voicebot.voice` apply to the interviewer
too. The time budget of each plan is edited in its review.

## 8. What was run (no measurement)

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

## 9. How to run the test (owner)

1. Restart the services after merging (`ragputin-servicios restart`):
   the RAG gets the new templates and warms the `interviewer@2` and
   `interview.slots@2` pools; the orchestrator gets the
   `voicebot.interview` product and `data/interview/`.
2. Optional: upload each job offer (PDF/DOCX) to a knowledge domain in
   «Conocimiento» and wait until it is *indexado*.
3. «Asistente de voz» → Tema: «Entrevista de trabajo con IA». Paste the
   offer (or pick the document), choose the kind of job and the length,
   «Preparar el plan» (one Haiku call, 5–60 s).
4. Review the plan: tick the *imprescindibles*, adjust weights,
   questions, the concepts of each point and time; «Guardar cambios».
   Write a candidate label.
5. Headset on, «Empezar la entrevista». Answer as the candidate; watch
   the coverage panel. Try: a vague answer (one follow-up, then the
   «matizar o pasar» offer), «no lo conozco», «pasamos a la siguiente»,
   a question to the interviewer.
6. At the end read the report; «Comparar candidatos de esta oferta»
   after the second interview with the same plan. With
   `interview.post_call: on` the analysis starts by itself; otherwise
   start it from the session.
7. Turn-taking (#213): pause for 2–5 s in the middle of an answer
   (after «y…», «porque…», and after a full sentence), stay silent
   after a question (re-ask after 10 s), stop in the middle and wait
   (hint after 7 s), start talking while the bot is «Pensando…» (it
   should stay silent). Watch the «Escuchando…» tag and listen for the
   quiet «ajá» during long answers. If it still cuts in, raise
   «Silencio que termina una respuesta» in «Ajustes».
8. What to note for the measurement: turns until every *imprescindible*
   was settled (the report's facts line), the slots the filler got
   wrong (coverage panel vs what was said), and the latency line under
   the conversation (end of the answer → first voice).

Haiku calls per interview: 1 (plan) + 2 per answer (filler and
interviewer; 1 with the `heuristic` filler) + 1 (report); for a
20-minute interview of 12–16 answers that would be about 25–35 (an
estimate from that count, not measured). `post_call: on` adds the analysis's calls.

## 10. Limits

- Synthetic only: the planner, filler, agenda thresholds and the report
  prompt were tried on two invented offers and two answers.
- The rules filler, the concept matching (#258) and the «no lo sé» /
  «pasamos» detection are Spanish word lists; English interviews are not
  handled yet. The concept rules miss synonyms and paraphrases and can
  still tick a concept from its own words used in another sense (§4.1).
- Bot utterance times in the transcript come from the orchestrator's
  clock (session start ≈ transcriber stream start). A turn dropped
  because the candidate went on talking does not move the agenda if
  the drop comes while the slot filler runs; dropped after that (while
  the interviewer's reply is being written), it already has, and its
  Haiku calls are spent.
- Turn-taking: the end of a pause is known only from the page's voice
  detector or the next partial, and a pause of more than about 4 s
  after a complete sentence ends the answer at the default 3.5 s (see
  the replay in §5). «Unfinished» is a Spanish/English word list. The
  end of the bot's audio is estimated from the chunks' durations. The
  backchannel tokens are said by the bot's voice (Piper or the clone);
  without a headset the microphone may hear them (one word, so never
  taken as the bot's echo; at worst it restarts the wait).
- One interview at a time (the voice bot's one live session).
- Privacy: the sheet holds candidate quotes and scores
  (`data/interview/results/`); retention and who sees it are pending
  decisions, as for the rest of the analysis.
