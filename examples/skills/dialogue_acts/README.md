# Datasheet: Spanish dialogue-act set (synthetic)

Issue #64 (SK-5). Category definitions: `docs/skills/CONVERSATION.md`
§5, with Información split into Información and Opinión (owner decision
on #57). Public corpora and their mapping: `docs/skills/DIALOGUE_ACT_DATA.md`.

## Purpose

To develop and get a **first automatic baseline** for the sentence-level
dialogue-act classifier in Spanish, where no public meeting corpus with
dialogue-act labels exists. It is **not** for validating the categories
or for reporting the analyzer's accuracy to anyone: see *Limits*.

## Who made it and how

- **Author: synthetic.** Every dialogue was written for this project by
  an AI coding assistant at the owner's request (2026-11-15), in batches
  of one conversation type each. No sentence was copied or closely
  paraphrased from a corpus, book, article or website. Names are generic
  Spanish first names (20 of them); companies, places and figures are
  invented.
- **Labels: the author's intended labels.** Each sentence was written
  with the act it is meant to perform and labelled at the same time.
  Every Respuesta, Aceptación and Rechazo points at the sentence it
  replies to (`>anchor`); the parser refuses a set where one does not, or
  where a link goes forwards, to the same speaker, or (for Respuesta) to
  something other than a Pregunta or a turn-giving Gestión.
- **Register:** spoken Spanish of Spain, from formal (interviews, a
  dentist, a landlord) to colloquial (flatmates, colleagues), with
  fillers ("eh", "bueno", "o sea", "venga"), stalls, self-corrections
  ("No, hasta marzo no, hasta enero"), interruptions (a sentence cut
  with "—" or "…" followed by another speaker; 32 sentences end so),
  laughter tokens ("Jaja") and a few deliberate speech-recognition slips:
  "Haber, que empezamos" (*a ver*), "Echo" ×3 (*hecho*), "tres de"
  (*3D*), "modulo" (*módulo*), "y halla lo que sea" (*allá*), and one
  inaudible segment written as `[ininteligible]`. Real ASR output would
  have many more errors and no punctuation guarantees; this set is
  cleaner than a transcript.

## Format

`*.txt`, one file per batch, readable beside its labels:

```
=== neg-01 | negociacion | Renovación del contrato de limpieza
-- fase: posicionamiento
Hugo: Os adelanto que para el año que viene tenemos que subir un seis por ciento. | informacion #i1
Marta: Pero un seis por ciento ahora mismo no lo podemos asumir. | rechazo >i1
```

`=== id | type | title` opens a dialogue (`session.type`), `-- fase:`
sets the phase of what follows, consecutive sentences of one speaker are
one turn.

A sentence that clearly does two or three things carries **weighted
labels** (owner decision 2 on #57, issue #73): `etiqueta:peso` with whole
%, steps of 10, primary first, summing to 100; a bare label is the same
as `etiqueta:100`, so every single-label line stays valid:

```
Javier: Vale, se lo digo hoy. | compromiso:70 aceptacion:30 >r1
```

A reply rule holds for every label of the sentence (the `aceptacion:30`
above still needs its `>r1`). The grammar and its checks are in
`src/domain/skills/dialogue_act_set.py`.

`dialogues.json` is the same set in a `transcript.v1`-like shape
(`session` with `type`; `speakers`; `utterances` with `id` `t<turn>.s<n>`,
`turn`, `index`, `speaker`, `text`, `act` (the primary label),
`responds_to`, `phase`, and `labels` (`label`, `weight`) only on the
sentences with more than one label; no times or audio;
`"synthetic": true`). It is generated, never edited:

```sh
python -m src skills dialogue_acts_set --source examples/skills/dialogue_acts \
    --output examples/skills/dialogue_acts/dialogues.json
python -m src skills dialogue_acts_import --corpus own \
    --source examples/skills/dialogue_acts --output data/dialogue_acts/own.jsonl
```

`expected_events.json` (#83) lists the conversation events (questions
answered, fate of proposals, decisions, action items, open issues) a
reader expects in 8 of the dialogues. It is the author's reading of the
content, written by the same assistant that wrote the event rules, not
human gold; `skills events_evaluate` scores the rules against it
(`docs/skills/CONVERSATION_EVENTS.md` §6).

## Contents (measured with `dialogue_acts_set`, 2026-11-15)

51 dialogues, 1,534 sentences, 1,188 turns, 10,557 words (6.9 per
sentence), 20 speakers, 479 reply links. 6 sentences carry weighted
labels (#73, 2026-11-15): four promises answering a request
(`compromiso:70 aceptacion:30`) and two invitations to speak
(`gestion:60 pregunta:40`). They were added to exercise the format; the
rest of the set was **not** relabelled, so most sentences that could
carry a second act still have one label.

Counts below are by primary label.

| Category | Sentences | Share |
|---|---:|---:|
| pregunta | 189 | 12.3% |
| respuesta | 194 | 12.6% |
| informacion | 171 | 11.1% |
| opinion | 93 | 6.1% |
| propuesta | 104 | 6.8% |
| peticion | 52 | 3.4% |
| aceptacion | 165 | 10.8% |
| rechazo | 98 | 6.4% |
| compromiso | 97 | 6.3% |
| escucha | 123 | 8.0% |
| gestion | 97 | 6.3% |
| relacion | 115 | 7.5% |
| otro | 36 | 2.3% |
| **total** | **1,534** | |

Rare categories are **oversampled on purpose** (the ticket asked for at
least 50 Compromiso, 50 Rechazo and 50 Señal de escucha): Compromiso is
6.3% here against about 0.3% in MRDA, Rechazo 6.4% against about 1%.
The proportions are therefore not those of real meetings; weight or
resample before reading any accuracy figure as representative.

Per conversation type:

| | reunion_decision | lluvia_ideas | resolucion_problemas | negociacion | entrevista | feedback | conflicto | coaching | venta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dialogues | 8 | 5 | 7 | 5 | 5 | 6 | 5 | 5 | 5 |
| pregunta | 20 | 11 | 25 | 15 | 36 | 13 | 11 | 36 | 22 |
| respuesta | 25 | 12 | 25 | 16 | 34 | 13 | 11 | 35 | 23 |
| informacion | 30 | 8 | 29 | 10 | 29 | 23 | 18 | 11 | 13 |
| opinion | 10 | 10 | 14 | 11 | 9 | 16 | 13 | 7 | 3 |
| propuesta | 12 | 34 | 12 | 13 | 1 | 4 | 8 | 6 | 14 |
| peticion | 12 | 5 | 11 | 3 | 4 | 7 | 2 | 4 | 4 |
| aceptacion | 42 | 21 | 21 | 13 | 2 | 18 | 24 | 11 | 13 |
| rechazo | 15 | 18 | 17 | 15 | 0 | 7 | 15 | 0 | 11 |
| compromiso | 20 | 11 | 17 | 8 | 5 | 8 | 10 | 10 | 8 |
| escucha | 18 | 12 | 17 | 15 | 15 | 16 | 8 | 9 | 13 |
| gestion | 32 | 21 | 8 | 4 | 4 | 8 | 15 | 3 | 2 |
| relacion | 15 | 7 | 10 | 12 | 19 | 18 | 16 | 7 | 11 |
| otro | 9 | 7 | 7 | 2 | 2 | 1 | 3 | 4 | 1 |
| **total** | 260 | 177 | 213 | 137 | 160 | 152 | 154 | 143 | 138 |

Phases are written per dialogue with the names the type suggests
(apertura, información, discusión, decisión, tareas; divergencia,
convergencia; posicionamiento, problema, soluciones; GROW's objetivo,
realidad, opciones, compromiso; …). They follow the practitioner models
of `CONVERSATION.md` §4.2 and are **not** an empirical phase scheme.

## Limits

- **The intended labels are not a human annotation.** They are one
  author's intentions, written together with the text, so the sentences
  tend to be clearer examples of their category than real speech is, and
  any agreement figure against them measures agreement with the author,
  not reliability. Inter-annotator agreement (κ) still needs two trained
  human annotators labelling independently; the cheapest useful check is
  a stratified subset of about 300 sentences (`CONVERSATION.md` §6.3).
- **Synthetic speech.** Dialogues are tidier, shorter and more on-task
  than real meetings; there are few overlaps, few long turns, and only a
  handful of recognition errors. Expect a model to score higher here than
  on real transcripts.
- **One author, one style.** Lexical cues ("¿Y si…?" for proposals,
  "Lo hago yo" for commitments) recur; a classifier can learn the
  author's phrasing. Hold out whole files, not random sentences, when
  splitting.
- **Boundary cases were decided one way.** "Vale" closing a proposal is
  Aceptación, during a telling Señal de escucha; an invitation to speak
  ("Nuria, ¿tú cómo lo ves?") is Gestión and its answer a Respuesta; a
  request accepted with a promise is split into Aceptación + Compromiso
  where the text allows it, else labelled Compromiso. Second and third
  labels were allowed after the set was written (decision 2 of #57); only
  the 6 sentences above carry them, so the set says nothing about how
  often real sentences need one.
- **For development and a first baseline only**, never for validating the
  categories or for any statement about a real person.
