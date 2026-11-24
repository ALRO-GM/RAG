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
- **Ahora** (one card): a line better avoided while it is not
  rephrased; a rephrased one (badge «Reformulado») until the other
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
  earlier suggestions open, assistance mode; status, warnings, banner).

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
that category), not measured; without the rules file or module the RAG
answers 404 and the page says once that questions are not checked.

## The scripted demo

`#/live/demo` (also «Ver una entrevista de ejemplo» on the home and the
start form) plays `examples/demo/entrevista-lucia.demo.json`
(`live.demo.v1`, `demo: true`): a fictitious backend interview with
«Lucía Marín», about 69 s, through the replay path
(`POST /sessions/{id}/replay {"demo": id}`). Its suggestions, the
age question's hint and the topics are prepared in the file and
published by `orchestrator/application/demo.py`; no knowledge, reasoner
or model call, so every run is the same. A new demo stops one still
playing, and leaving the page stops it. It ends on
`#/analysis/demo-entrevista-lucia-marin`: the RAG installs
`examples/demo/entrevista-lucia.analysis.json` (`analysis.demo.v1`,
synthesis `format_version` 2: 88,7 %, strengths solves with data,
clear communication, ownership; improvements automated testing, below
the required level, and delegating) when it starts, with the run marked
`synthetic` and every other step `unavailable` («Demo data»). The
overall score 4.548 is written with three decimals so that its percent
is 88,7 (the #231 rule rounds scores to one decimal, which on a 1–5
scale gives only multiples of 2,5 %). No hiring wording anywhere.
