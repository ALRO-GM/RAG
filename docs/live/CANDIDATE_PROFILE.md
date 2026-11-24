# The candidate across interview rounds (LV-18, #235)

The owner asked for «general questions that must be asked in every
interview, few so the interview is not saturated, saved in the
candidate's profile so they need not be repeated in later rounds». The
questions are KN-4's five core questions
(`knowledge/interview/core-questions.json`, `interview.core.v1`,
`docs/live/INTERVIEW_KNOWLEDGE.md` §2); this note is how their answers are
captured, kept, shown, corrected and deleted.

Synthetic data only: no real candidate went through it, and nothing
here was measured on real interviews.

## 1. Role and candidate

- A **role** is an interview plan (`interview.plan.v1`,
  `data/interview/plans/<plan>.json`): the same plans the AI interviewer
  uses. Every candidate of a role gets the same core questions, with the
  same words (comparability).
- A **candidate** belongs to one role. On «En directo» (interview
  product) the start form now offers «Puesto» (a plan, optional) and
  «Candidatura» (one of the role's candidates, or a new name). Without a
  role nothing about the candidate is kept, as before. The AI interviewer
  takes the same `plan` plus `candidate_id` (or `candidate`, a new name).

## 2. What is stored, where, and how it is deleted

`interview.candidate.v1`, one JSON file per candidate:
`<interview dir>/candidates/<plan id>/<candidate id>.json`
(`orchestrator/adapters/outbound/fs_candidates.py`; folders mode 700,
files mode 600, written atomically; ids checked against
`^[a-z0-9][a-z0-9-]{0,63}$` before a path is built).

```
schema, id, plan_id, name (≤ 80), created_at, updated_at,
answers{ <question id>: {question, fields{path: value}, summary (≤ 280),
                         quote (≤ 160), source: captured|corrected,
                         session_id, at} },
rounds[ {session_id, at, kind: live|bot, covered[], weak[], talk{}} ]   # ≤ 20
```

- `fields` are the profile paths of the core question
  (`eligibility.work_authorization`, `salary.expectation_min`, ...).
- `quote` keeps the candidate's own words **only** for the start date and
  the salary; never for a yes/no answer (so a reason given with a «no» is
  not stored) and never for the motivation (its kept words are the
  summary).
- No audio, no transcript, no score of the person.

**Retention.** A profile is deleted once it has not changed for the
shortest `privacy.store_text_days` of the interview products (30 days in
`config/products/live_assist.interview.yaml` and
`voicebot.interview.yaml`); the check runs whenever profiles are listed
or read (`CandidateBook.purge`). Without that setting nothing expires.

**Deletion on request.** `DELETE
/interview/plans/{plan}/candidates/{candidate}` (on the page: «Borrar
ficha de la candidatura» under «Más información» of «Datos de la
candidata», after a confirmation) deletes the file. One answer can be
forgotten with `DELETE .../answers/{question}`. The sessions'
transcripts, hand-off records and analyses are separate and are not
touched (they follow their own retention).

**Pending.** The owner decisions of #151 (legal basis for candidates'
data, retention days, processor terms) are still open: the 30 days are
the products' current setting, not a decision. Consent is the
interview's: the mandatory items (`uso-datos`, `aviso-asistente`) still
ask the interviewer to say how the data are used.

## 3. How answers are captured (deterministic)

`orchestrator/domain/candidate.py`, no model call:

1. **The question must be heard first.** `asked_core` matches the
   interviewer's sentence (live page: role `interviewer`; AI interviewer:
   what the bot actually spoke) against word patterns per question, in
   Spanish and English, on the folded text. Only questions still open are
   looked for.
2. **The next candidate sentences are read** (at most 3; a new question
   from the interviewer lets it go):
   - work mode/location, work authorisation: a clear yes or no; «en
     trámite», «depende», «no sé», «quizás» or a sentence with both give
     nothing;
   - start date and notice: «de inmediato», «en dos semanas», «a
     principios de marzo»... and «N días/semanas/meses de preaviso» (in
     days), «sin preaviso» = 0;
   - salary: one figure or a range («entre 28 y 32 mil», «30.000»,
     «30k») in euros per year, 6 000–500 000; a monthly figure, a refusal
     or three figures give nothing (the interviewer types it);
   - motivation: the answer's words (≤ 280 characters) unless it is
     shorter than 6 words or names a personal circumstance (children,
     family, health, religion, union, origin, age...: then it is not
     recorded, as KN-4's legal note says).
3. **Nothing is overwritten.** A capture never replaces an answer
   already kept, and the profile is read again before writing, so an
   answer corrected meanwhile on another page stays.

**Correcting.** `PUT .../answers/{question} {"text"}` keeps the
interviewer's words as a `corrected` answer (the same rules fill the
fields when they can: «No» becomes `false`, «entre 30 y 34 mil» the two
figures). Only another correction or «Quitar respuesta» changes it.

**Limits.** Word patterns miss paraphrases («¿Tienes papeles?» is not
recognised as the authorisation question) and an answer said before
the question is not taken. A misheard «sí»/«no» from the recogniser is
taken as said: the answer is visible at once on the live page and on the
result page with «Leída de la conversación... corrígela si no es
exacta». Not measured on real interviews.

## 4. What later rounds show

- **Live page.** A quiet row «Preguntas comunes · N de 5 respondidas»
  under the topics: answered ones as done chips with their answer
  (hover: the whole answer), open ones as dashed chips (hover: the
  question). Suggestions that would ask an answered question are dropped
  before they reach the page (`CoreTracker.screen` in
  `LiveAssist`). The feed message is `{"type": "core", "profile": ...,
  "captured"?: id}`.
- **AI interviewer.** It opens with the open core questions of the
  opening and experience phases (authorisation, start date, motivation),
  then the plan's slots; the salary expectation is asked just before
  closing. Answered ones are never asked. The work mode/location
  question has blanks («[modalidad] en [ciudad]») the plan cannot fill,
  so the bot leaves it to a person.
- **Result page.** `#/analysis/<session>` shows «Datos de la candidata»
  under the overview when the orchestrator knows the session's candidate
  (`GET /interview/candidates/by-session/{session}`): one line per core
  question (answer or «Pendiente»), the number of rounds; under «Más
  información» how each answer was obtained, the kept words, a field to
  correct it and the deletion.

## 5. The next round's plan and the talk balance

**What a round leaves.** At the end of every round the profile gets a
`rounds[]` entry with the plan's slots `covered` and left `weak`:

- AI interviewer: from its sheet (`round_from_sheet`): covered = score
  70 or more (or `filled` without a score); weak = a `minimo` or
  `deseable` slot scored under 50, `partial`, `unknown` or `skipped`.
- Live page: `TopicTracker` (`orchestrator/application/next_round.py`)
  notes the slot an interviewer's sentence asks about (two content
  words of the slot's label and questions, one when the label has a
  single one) and reads the candidate's next answer with the AI
  interviewer's cheap filler (`heuristic_fill`): `complete` = covered;
  asked but not covered = weak. Word overlap only: a question in other
  words is missed, and a long answer counts as covered without anyone
  judging it.

**The next round** (`orchestrator/domain/next_round.py`): each slot is
`weak`, `open` (never covered) or `covered`, the newest round that
mentions it deciding. The live page's topic chips come in that order
(weak first, covered last and quieter; hover says why) under «Temas del
puesto · primero lo pendiente», and done ones tick as the round covers
them. The suggestions get the round's goal («Ronda N del puesto «T».
Empieza por lo que dejaron las rondas anteriores: por debajo del nivel
pedido: ...; sin tratar aún: ...», at most 480 characters) as the
conversation goal of `reasoner.request.v1` (`context.session.goal`,
which the suggester template already reads). The AI interviewer gets
the plan with its slots reordered the same way (levels kept: `minimo`
still first). A first round, or one with nothing weak or open, has no
goal. The analysis's own weak points (the synthesis' improvements) are
not read here: only the plan's slots.

**Talk balance.** The live page's header shows one thin bar and «Tú
62 % · Lucía 38 %»: each side's share of the time of its final
sentences (`LiveState.talkBalance`, the seconds between start and end of
every final). No threshold, colour change or alert. The round keeps the
same shares per role (`rounds[].talk`; for the AI interviewer, its spoken
audio against the candidate's finals). Pauses and overlaps are not
counted; a voice wrongly attributed in mono counts for the wrong side.

## 6. Speaker roles in the analysis

`transcript.v1` speakers may now carry `role` (optional, additive; see
`contracts/README.md`). The live hand-off writes the role the session
gave each voice (not `unknown`), the AI interviewer writes `interviewer`
and `candidate`, and replays keep a role the transcript gives. The RAG
keeps it as `Transcript.roles`; the `sentences` step lists it per speaker
(`speakers[].role`), the synthesis reads only the candidate (and anyone
without a role) against the job, and the result page draws the job
ring and bars only for the candidate (the interviewer gets the general
reading) and opens the candidate's card first. Results without roles
read as before.

## 7. API (orchestrator, through `/api/assist/...` on the transcriber)

| Route | Body | Answer |
|---|---|---|
| `GET /interview/plans/{plan}/candidates` | — | `candidates` (views), `questions`, `retention_days` |
| `POST /interview/plans/{plan}/candidates` | `{"name"}` | 201 the new candidate |
| `GET`/`DELETE /interview/plans/{plan}/candidates/{id}` | — | the candidate / `{"deleted": true}` |
| `PUT`/`DELETE .../candidates/{id}/answers/{question}` | `{"text"}` / — | the candidate |
| `GET /interview/candidates/by-session/{session}` | — | the candidate whose rounds include the session (404 otherwise) |
| `POST /sessions` | adds `candidate_id` (and `plan` for a live-page interview) | the session (`candidate_id`) |
