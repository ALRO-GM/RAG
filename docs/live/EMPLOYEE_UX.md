# The live page for the employee (LV-16, #206)

What the person who gets the suggestions sees and chooses on
«En directo» (`#/live`). Measurements: `ASSIST_MODES.md` (the two modes,
Haiku) and `MANDATORY_ITEMS.md` (the similarity threshold).

## Who is who

- **Employee.** The start form lists the people enrolled with their
  voice (`GET /api/speakers`) and links to «Registrar mi voz»
  (`#/enrol`) when there is nobody. The choice goes as `employee: {id,
  name}` with the session (`POST /sessions`, `POST /sessions/{id}/replay`).
  - Stereo: the channel of the employee's role (agent, interviewer)
    takes their name.
  - One microphone: the session is diarized (streaming Sortformer, see
    `MONO_DIARIZATION.md`) with `enrolled: [employee]`, so only their
    voiceprint can name a voice; every voice starts as the other party
    (customer, candidate) and the voice the voiceprint names becomes the
    employee (role agent or interviewer). Until it is named (about 4 s
    of their clean speech), the employee's first sentences count as the
    other party's. With «Voces de un solo micrófono» off in «Ajustes» the
    microphone stays one person, as before.
  - Replays: the transcript's speaker with the employee's role takes the
    name.
- **Names.** Every name on the page can be clicked and typed over during
  the session (`POST /sessions/{id}/speakers/{speaker} {"name"}`, at
  most 60 characters). The change is published (`{"type": "rename"}`),
  kept in the hand-off record and, in a live session, sent to the
  transcriber (`POST /api/live/sessions/{id}/speakers/{speaker}`) so the
  `transcript.v1` speakers carry it.

## Role and candidate (interviews, #235)

The start form of an interview also offers «Puesto» (an interview plan)
and «Candidatura» (a candidate of that role, or a new name). With both,
the candidate's answers to the core questions are read from the
conversation as proposals, which the employee confirms, edits or
discards in «Confirmar datos» when the session ends (or on the result
page); confirmed ones are kept for the next rounds, shown as a
«Preguntas comunes» row on the page and as «Datos de la candidata» on
the result page. Later rounds start from the topics earlier ones left
below the level, to revisit or open. The header shows a quiet talk-time
bar (you / the other person) on every live page:
`docs/live/CANDIDATE_PROFILE.md` §4–§5.

## What is shown

| Choice | Where it is set | Default |
|---|---|---|
| The employee's lines on/off | checkbox on the page; «Ajustes» `live.show_employee` | on |
| The other party's lines on/off (both off: suggestions only) | checkbox; `live.show_other` | on |
| Suggestions in view (older ones fold under "N earlier suggestions") | select; `live.max_suggestions` (1, 2, 3, 5) | 3 |
| Assistance mode, `pincelada` or `guiado` | buttons on the page (applies to the next call); `live.assist_mode` | pincelada |

The page's own choices are remembered in this browser
(`localStorage`, key `ragputin.live.view`, every access in try/catch:
a blocked storage only means the defaults); «Ajustes» gives the
defaults for a browser that chose nothing.

## Mandatory items

A sticky red box above the conversation lists the domain's items still
to say; an item turns green when the employee's sentence says it or
when it is ticked by hand after a confirmation dialog (logged). Every
`remind_seconds` (45 in the seed lists) it flashes and a toast names
what is missing; ending the session with items pending asks first and
lists them; the end-of-session summary on the page and the hand-off
record list what was not said. Details: `MANDATORY_ITEMS.md`.

## Discomfort (support only)

When the newest answered suggestion quotes a `complaint`,
`stated_frustration`, `churn_risk` or `urgency` signal from the
customer's own words, a banner says so with the quote, and the quoted
customer line and that suggestion are outlined. The signals are the
text-only ones of LV-14 (`docs/platform/ORCHESTRATOR.md`); interviews
never show signals, ratings or emotions about the candidate, and the
page refuses to draw them there too.

## Hand-off record

At the end of a live-page session the orchestrator publishes
`{"type": "handoff", "handoff": {...}}` and writes
`<log-dir>/<session>.<product>.handoff.json`: session, product,
assistance mode, employee, names, renames (with their time) and the
mandatory items (`said`, `ticked`, `missing`, manual `ticks`). The
`transcript.v1` the transcriber publishes has no place for the items,
so they stay in this record.

## Not done

- No measurement of the voiceprint marking on real calls (only the
  synthetic one of `MONO_DIARIZATION.md`).
- The page does not check that a renamed speaker exists in the session.

## The calm page (LV-17, #232)

Target: `docs/web/mockups/Live.dc.html` (approved 2026-11-24). One thing
at a time, nothing overlapping, phone width (the «Ahora» card comes
first there), light and dark.

- **Conversation** (left): the employee's lines muted under «Tú», the
  other party's larger under their name; a line types in as its
  partials arrive and fades in once. «Está hablando» under the list.
  The list is a box tied to the viewport's height and anchored to the
  bottom: the newest line stays in view and older ones leave at the
  top under a soft fade, with no scrollbar; scrolling back stops the
  following until the user is at the bottom again (`followsNewest`).
- **Ahora** (one card): a line better avoided while it is not
  rephrased (for at most 30 s of conversation, and only until a newer
  suggestion arrives: then it steps aside and stays as the line's label
  and in the history); a rephrased one (badge «Reformulado») until the other
  party answers or a newer suggestion arrives; else the newest
  suggestion the employee has not said yet, else the newest one with
  «Hecho». «Hecho» means an employee line after the suggestion holds
  at least 60 % of the content words (4 letters or more) of its
  example (or of a guided sentence): words, not meaning.
- **Topics**: a demo's job topics, else the product's script phases
  (done once a suggestion was about the phase). **Por decir**: the
  mandatory items as chips; a pending one is ticked by clicking it
  (confirmation, logged as before). Every `remind_seconds` the row is
  highlighted for 3 s with what is missing: no toast any more.
- **Under disclosures**: «Sugerencias anteriores y de dónde salen»
  (older suggestions, `about` sentences, sources, latency, the lines
  better avoided) and «Opciones» (the LV-16 choices: lines shown,
  earlier suggestions open, assistance mode; status and banner).
  Errors and one-time notices (failed, halted, reconnecting, warnings,
  questions not checked) are a line under the header, always in sight.

## Lines better avoided

Each final sentence of the **interviewer** (interview products only) is
checked by the RAG, `POST /knowledge/interview/avoid {"text",
"language"}`, against KN-4's `knowledge/interview/forbidden-questions.json`
read through `src/domain/knowledge/interview_rules.py`
(`src/adapters/outbound/fs_forbidden_rules.py`, which loads that module
by name). The answer lists the categories with title, why and a lawful
alternative; the orchestrator keeps them (`orchestrator/domain/avoid.py`)
and publishes `{"type": "avoid", "flags": [...]}`. Limits: regular
expressions only (no template refinement yet), Spanish and English; a
hint about the question, never about the candidate and not legal advice;
the rephrase rule is a display choice (a question within 30 s without
that category), not measured. A line a later `speaker_update` gives to
the candidate loses its flag. Without the rules file or module the RAG
answers 404 (notice `avoid_no_rules`); a RAG that does not answer gives
`avoid_unreachable`. Each cause is said once; after 3 failures in a row
the checks rest for 60 s of conversation, then try again.

## The scripted demo

`#/live/demo` (also «Ver una entrevista de ejemplo» on the home and the
start form) plays `examples/demo/entrevista-lucia.demo.json`
(`live.demo.v1`, `demo: true`): a fictitious backend interview with
«Lucía Marín», about 69 s, through the replay path
(`POST /sessions/{id}/replay {"demo": id}`). Its suggestions, the
age question's hint and the topics are prepared in the file and
published by `orchestrator/application/demo.py`; no knowledge, reasoner
or model call, so every run is the same. Its hints, suggestion texts,
topics and title also come in English (`en`) for a page in English;
the conversation stays Spanish. A new demo, a live session or a
recorded replay stops a demo still playing (a demo never stops a real
session), and leaving the page stops it. It ends on
`#/analysis/demo-entrevista-lucia-marin`: the RAG installs
`examples/demo/entrevista-lucia.analysis.json` (`analysis.demo.v1`)
when it starts: the synthesis (`format_version` 2: strengths solves with
data, clear communication, ownership; improvements automated testing,
below the required level, and delegating) plus the two results the
summary-first page needs, the sentences and an assessment with one
scorecard (the candidate's, no levels), so she is the only person of
it. The run is `synthetic`, every step carries «Demo data» and the
steps not given read `unavailable`. It is installed only over nothing
or over another synthetic run, under the analysis lock; a failure is
printed and the RAG starts anyway. The overall opinion is computed with
the real rules of `src/domain/skills/opinion.py` from the file's eight
skill scores: weighted mean 4.548, percent 88,7 (since #232 the percent
comes from the unrounded mean; the shown score stays 4.5), no caveats.
No hiring wording anywhere.
