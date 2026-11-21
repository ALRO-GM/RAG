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
