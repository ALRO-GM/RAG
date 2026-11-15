# Conversation events: the second layer

Issue #83 (SK-10). The first layer labels each sentence with one to
three of 13 dialogue acts (`CONVERSATION.md` §5,
`DIALOGUE_ACTS_GUIDE.md`). None of these labels is a decision, a task
or an unanswered question: those are **sequences** (`CONVERSATION.md`
§2, "From pairs to agreements", and §5.1, layer 2). This document
states the rules that derive them, their parameters, what the input
must look like and how far the rules were checked.

Everything here is counting over labels: no model is called, nothing is
rated. A per-person summary is a list of counts and rates ("3 proposals,
1 accepted"), context for a reader, never a score.

Code: `src/domain/skills/act_sequence.py` (input),
`src/domain/skills/conversation_events.py` (linking and events),
`src/domain/skills/conversation_summary.py` (counts),
`src/domain/skills/event_evaluation.py` (scoring),
`src/application/skills/conversation_events.py` (use cases),
`src/adapters/outbound/act_sessions.py` (files).

## 1. Input

The rules read an `ActSession`: per sentence its `id`, `speaker`,
`turn` (consecutive sentences of one speaker are one turn), position,
optional `start`/`end` seconds, one to three weighted labels and an
optional `responds_to`. It is read from any of:

- the Spanish set's text files (`examples/skills/dialogue_acts/*.txt`,
  one file or the whole directory) or its `dialogues.json`;
- a tagger's `<session>.acts.json` (#82), in this shape (`acts.v1`):

```json
{
  "schema": "acts.v1",
  "session": {"id": "demo_es"},
  "tagger": {"name": "…", "version": "…"},
  "utterances": [
    {"id": "t3.s1", "turn": 3, "speaker": "Ana", "text": "¿Lo vemos el lunes?",
     "start": 12.4, "end": 14.0,
     "labels": [{"label": "pregunta", "weight": 72},
                {"label": "propuesta", "weight": 21, "doubtful": true}],
     "rest": 7,
     "responds_to": null}
  ]
}
```

`session.id` (or a top-level `session_id`) and `utterances` with
`speaker` and either `labels` or a single `act` are required. `turn`
is counted from speaker changes when absent; `id` defaults to
`u<position>`; ids must be unique; `start`, `end`, `responds_to` are
optional; other keys (`schema`, `tagger`, `rest`, `phase`, …) are
ignored. Labels are the 13 category ids; weights are whole % (1–100)
and need not sum to 100 (a tagger keeps an explicit rest).

## 2. Parameters

All are `SkillsSettings` fields (`--event_reply_window 3` on the
command line) handed to the domain as `EventRules`. The defaults were
chosen by reading the Spanish set before any evaluation; they are not
tuned on reference data (there is none yet).

| Setting | Default | Used by |
|---|---|---|
| `event_reply_window` | 4 turns | how far back a missing reply link is searched (rule 1); how many turns an unanswered question or ignored proposal needs before its confidence is full (rules 2, 3) |
| `event_link_decay` | 0.15 | confidence lost per extra turn of distance of an inferred link |
| `event_min_link_confidence` | 0.3 | weakest inferred link kept |
| `event_min_label_weight` | 30 % | a label counts when it carries at least this weight, so `gestion:60 pregunta:40` is also a question |
| `event_use_doubtful` | false | whether tagger labels marked `doubtful` count |
| `event_min_acceptances` | 1 | acceptances by other speakers an accepted proposal needs to become a decision |
| `event_closing_acceptances` | 2 | acceptances that make a gestión sentence without closing words a decision (capped at the number of other speakers) |
| `event_closing_window` | 12 turns | a gestión closing absorbs the accepted proposals of the previous 12 turns into one decision |

## 3. The rules

A label "is present" on a sentence when its weight reaches
`event_min_label_weight` (and it is not doubtful, unless allowed).
*First-pair parts* are pregunta, propuesta and petición; *replies* are
respuesta, aceptación, rechazo and compromiso.

### Rule 1 — reply linking (when links are missing)

A given `responds_to` is kept with confidence 1 (dropped if it points to
the same speaker or to no sentence). A sentence with a reply label and
no link is attached to the earlier sentence of **another speaker**,
at most `event_reply_window` turns back, that maximises

```
prior(reply label, antecedent kind) × (1 − link_decay × (distance in turns − 1))
```

if that score reaches `event_min_link_confidence`; the score is the
link's confidence. Priors (`LINK_PRIORS`):

| Reply | Antecedent priors |
|---|---|
| respuesta | pregunta 1.0, gestión 0.5 |
| aceptación | propuesta, petición, gestión closing 1.0; opinión 0.6; información, gestión 0.5; pregunta 0.3 |
| rechazo | propuesta, petición, gestión closing 1.0; opinión, información 0.6; gestión 0.4; pregunta 0.3 |
| compromiso | petición 1.0, pregunta 0.6 — only if no other speaker has replied to it yet |

Escucha, otro and relación are never antecedents. Example (dec-01):
"Me gusta, así no nos jugamos todo de golpe." follows Irene's "¿Y si
hacemos una prueba de un mes…?" one turn later → linked, confidence 1;
Luis's "Sí, de acuerdo." two turns later → 0.85.

### Rule 2 — questions

- **question_answered**: a pregunta with at least one linked reply by
  another speaker. Lists who answered and the turns until the first
  reply. *Ex.* "¿Y eso cuánto tarda?" → "El banco nos dijo dos semanas…"
  (dec-05).
- **question_unanswered**: a pregunta with no linked reply. Confidence
  is the label weight × min(1, (turns left in the session + 1) /
  (`event_reply_window` + 1)): a question in the last turn is only
  weakly "unanswered".

### Rule 3 — proposals

For each propuesta, every other speaker's **last** stance stands (an
aceptación or compromiso accepts, a rechazo rejects; a speaker who first
says "vale" and later "bueno, no" rejects):

- **proposal_accepted**: standing acceptances, no standing rejection.
  *Ex.* "Podríamos preguntar a la segunda si tienen un plan…" → "Vale,
  eso sí." · "Bien." (dec-03).
- **proposal_rejected**: standing rejections only. *Ex.* "Yo quería
  lanzar el uno de mayo." → "Imposible." (dec-05).
- **proposal_contested**: both. *Ex.* "Propongo la segunda…" →
  "Me parece razonable." and "Sí, pero la de la intranet no tiene
  subtítulos…" (dec-03).
- **proposal_ignored**: neither (a backchannel "Mm, ajá" is not a
  reply). Confidence scaled as for an unanswered question. *Ex.* "Pues
  una escape room por equipos…" (llu-01).

### Rule 4 — decisions

- A **gestión closing** is a gestión sentence (not also a pregunta)
  with **no standing rejection** that either contains closing words
  (`CLOSING`: *resumiendo, en resumen, queda, quedamos, acordamos,
  acordado, decidido, cerrado, lo dicho*) or gets at least
  `event_closing_acceptances` acceptances (capped at the number of other
  speakers). *Ex.* "Entonces queda: intensiva julio y agosto, con turnos
  de tarde rotatorios…" → "Hecho." · "Sí." · "De acuerdo." (dec-02).
  "Pues lo dejamos aquí, nos vemos el lunes." is not one (no closing
  word, no acceptance).
- A closing absorbs the accepted proposals of the previous
  `event_closing_window` turns: one decision supported by the summary,
  the proposals and their acceptances.
- An **accepted proposal** with at least `event_min_acceptances`
  acceptances that no closing absorbed is a decision of its own.

### Rule 5 — action items

- An **accepted petición** gives one item per accepting speaker
  (owner), with who asked, the request, the acceptance and the owner's
  compromiso sentences linked to it or said in the same turn as the
  acceptance. *Ex.* "Luis, ¿puedes llamar tú a la empresa nueva esta
  semana?" → "Sí, claro." "Les llamo el jueves por la mañana." (dec-01):
  one item, owner Luis, what "Les llamo el jueves por la mañana.",
  deadline "el jueves por la mañana".
- Every other **compromiso** is an item of its speaker. *Ex.* "Yo
  preparo la lista de envíos a Valencia…" (dec-01).
- **Deadline**: the first Spanish time expression (`DEADLINE`: *hoy,
  mañana, esta tarde, esta semana, el jueves, antes del viernes, la
  semana que viene, en dos semanas, a finales de marzo…*) found in the
  commitment, then the acceptance, then the request.
- A rejected petición gives nothing; one nobody took up is an open
  issue.

### Rule 6 — open issues

What is still unresolved at the end: every unanswered question, every
petición nobody accepted or rejected, and every ignored or contested
proposal **with no decision after it** (a later decision is taken to
settle the competing proposals). Each open issue points (`source`) to
the event it restates.

### Confidence

Each event carries the smallest of: the weights (as fractions) of the
labels it rests on and the confidences of the links it uses (unanswered
and ignored events: as in rule 2). Human labels of the set give 1.0 on
most events; a tagger's probabilities and inferred links lower it. It
is a heuristic score, **not a calibrated probability**.

## 4. Summaries (counts and rates, never ratings)

Per person (`PersonEvents`): questions asked / answered / unanswered
and the answer rate; questions of others they answered; proposals made
and their fate (accepted, rejected, contested, ignored) with acceptance
and ignored rates; acceptances and rejections given; commitments
(action items owned); decisions they took part in; median turns from
someone else's question, proposal or request to their reply. A rate is
"n/a" when its denominator is zero. Per session (`SessionEvents`):
count of every event type, given and inferred links, median reply
latency, and whose proposals were ignored.

## 5. Commands

```sh
# events of one dialogue of the set (or of a tagger's acts file)
uv run --no-sync python -m src skills interaction \
    --acts examples/skills/dialogue_acts --session dec-01
# participation from the transcript, then its events; infer every link
uv run --no-sync python -m src skills interaction data/transcripts/demo.json \
    --acts data/transcripts/demo.acts.json --relink --output events.json
# the person report with the participant's event counts as context
uv run --no-sync python -m src skills report out/demo.json Ana --acts demo.acts.json
# precision/recall against the expected events
uv run --no-sync python -m src skills events_evaluate \
    --output experiments/skills/results/conversation_events_eval.json
```

## 6. Evaluation

**Reference.** `examples/skills/dialogue_acts/expected_events.json`:
70 expected events in 8 dialogues of the Spanish set (dec-02, dec-04,
llu-01, pro-01, neg-01, con-01, ven-01, seg-01), written on 2026-11-15
by the AI coding assistant that wrote the rules, from each dialogue's
content, before running the rules on those dialogues. **It is the
author's reading, not human gold**: one reader, no agreement measured,
the same author as the rules, on synthetic dialogues whose labels are
also the author's. What counts as a question or proposal follows the set
labels; the outcome (answered, accepted, decided, who took which task)
is read from the content. Matching: same type, the expected sentence
among the event's supporting sentences (key first) and, for action
items, the same owner; one-to-one. Deadlines and confidences are not
scored.

**Results** (`skills events_evaluate`, 2026-11-15, saved to
`experiments/skills/results/conversation_events_eval.json`; P =
precision, R = recall, n/a = nothing to divide):

| Event | Expected | With the set's links: derived / P / R | Links removed and inferred: derived / P / R |
|---|---|---|---|
| question_answered | 22 | 21 / 1.000 / 0.955 | 21 / 1.000 / 0.955 |
| question_unanswered | 0 | 1 / 0.000 / n/a | 1 / 0.000 / n/a |
| proposal_accepted | 10 | 9 / 1.000 / 0.900 | 8 / 1.000 / 0.800 |
| proposal_rejected | 8 | 8 / 1.000 / 1.000 | 7 / 0.857 / 0.750 |
| proposal_contested | 2 | 0 / n/a / 0.000 | 2 / 0.000 / 0.000 |
| proposal_ignored | 1 | 4 / 0.250 / 1.000 | 4 / 0.250 / 1.000 |
| decision | 11 | 9 / 1.000 / 0.818 | 9 / 0.778 / 0.636 |
| action_item | 16 | 18 / 0.889 / 1.000 | 19 / 0.842 / 1.000 |
| open_issue | 0 | 1 / 0.000 / n/a | 1 / 0.000 / n/a |

Inferred links against the set's 88 links in those dialogues: 72
recovered (0.818), 13 wrong, 3 missed, 1 inferred on a sentence the set
left unlinked.

**One rule changed after the first run.** In the first run an ignored
proposal was always an open issue: 5 open issues derived against 0
expected (both link modes; every other row identical to the table).
Reading them (the brainstorm ideas of llu-01 that the final decision
replaced), rule 6 was changed so that a later decision settles ignored
proposals as it already did contested ones. The table is after that
change, so the open-issue row is fitted to these 8 dialogues. Reading
the output also showed a bug in the deadline words ("a la dirección de
la clínica" read as a deadline): "a la / para el / hasta el / antes
de" now need a day, month, hour or similar after them. Deadlines are
not scored, so the table did not change.

**What the errors show.** Content the labels do not carry is missed:
an idea offered in answer to "¿Alguna más?" is a propuesta, not a
respuesta, so the question counts as unanswered; support voiced as an
opinión ("La clase de cocina me parece la más segura") is no
acceptance, so contested brainstorm ideas look ignored; a counter-offer
taken into the final deal (neg-01) looks ignored. A closing absorbs a
different accepted proposal said shortly before (llu-01, "llamar y
preguntar"). "Resumo:" is not in the closing words and gets one
acceptance of three, so seg-01's decision is missed. A promise of
politeness labelled compromiso ("Le robo quince minutos como mucho, se
lo prometo") becomes an action item, and an e-mail promised and then
requested again (ven-01) becomes two.

**Limits.** 8 synthetic dialogues, 70 events, one reader who also wrote
the rules: these figures show the rules run and where they break, not
how well they would do on real meetings or on a tagger's labels (which
will be noisier than the set's intended ones). A real measure needs
annotators marking decisions and action items on recorded sessions
(`CONVERSATION.md` §6.3, "Is layer 2 right?"); action items need κ for
their four parts (Purver et al.).

## 7. Known simplifications

- Turns are speaker changes; overlapping speech and long pauses are not
  used (times are only carried through).
- A later compromiso by the same speaker within the reply window joins
  the request they accepted, even when it is a different task.
- A gestión sentence with enough acceptances is a decision even when it
  is procedure ("empezamos") rather than an outcome; the two-acceptance
  threshold keeps most of these out.
- Closing and deadline words are Spanish only.
