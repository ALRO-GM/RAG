# The call-centre example (DM-2, #250)

One fictitious call, shown three ways, for people who want to see what
the product does for a customer-service team without a microphone or a
model. Company, agent, customer and every sentence are invented
(`Lumbre Hogar`, Marta, Andrés Ferrer); the pages say so.

**The example in two lines.** Andrés Ferrer calls to cancel Lumbre Plus
because the October renewal was charged after he cancelled in September;
after a hurtful moment (he is asked to repeat his data for the third
time) the agent checks the two conditions of the knowledge base (cancelled
before the renewal, free shipping not used that month), refunds the charge
and he stays on the monthly Lumbre Plus with a 10 % voucher. The agent says
the data notice late and without the customer's rights, never asks for the
express acceptance and leaves «¿me llegará algo por escrito?» unanswered.
Every offer is backed by the knowledge base (`support.demo` chunks s17,
s18, s20, s43, s46, s52, s53, s54): Lumbre Plus renews monthly and can be
cancelled any time, so no 12-month contract is offered.

## Where it is

| What | Route | Data |
|---|---|---|
| Live support call | `#/live/demo/support` (also from `#/live` and the home) | `examples/demo/llamada-lumbre.demo.json` |
| Call report | `#/call-report/llamada-lumbre` (home «Ver un informe de ejemplo», end of the live call) | same file, `report` |
| Voice assistant example | `#/voicebot/demo` (home «Escuchar un ejemplo», link on `#/voicebot`) | same file, `voicebot` |

`#/live/demo` still opens the interview. The file is `live.demo.v1`
with extra optional parts (`sources`, `mandatory`, `report`,
`voicebot`); the orchestrator serves the last two at
`GET /demos/{id}/report` and `/voicebot` (proxied under
`/api/assist/demos/...`). The call has 19 sentences and its last sentence ends at 173 s (2 min 53 s).

## What is scripted, what works on a real call, what is missing

| Part | In the demo | On a real call today |
|---|---|---|
| Sentences of the call | written by hand | transcribed live or from a recording |
| Customer signals quoted in the suggestion («El cliente dice…») | listed in the file per sentence | **works**: the support reasoner labels signals and the page quotes them (`orchestrator/domain/signals.py`) |
| One suggestion at a time with its source | prepared; the source text is copied from `knowledge/domains/support.demo` (a test compares it) | **works**: the RAG retrieves chunks and the reasoner cites them |
| Mandatory items pending, said, reminded | the file names the sentence that says each | **works**: matched by the RAG (`MANDATORY_ITEMS.md`), reminder every 45 s, hand-off record of said, ticked and missing |
| Hand-off record at the end | published by the demo | **works** (`<log-dir>/<session>.<product>.handoff.json`) |
| Report: sentence by sentence (helped / neutral / hurt by the customer's next reaction, better wording) | written by hand | **not built**: nothing judges a line by the reaction today |
| Report: reference script followed / skipped / reordered, and whether the script looks improvable | written by hand | **not built**: no reference script is stored for a call type and nothing aligns a call to one |
| Report: top-scorer wording | **illustrative text written for the demo**, labelled so on the page; it is not from real agents and not a measurement | **not built** |
| Report: mandatory items said / said late / never | consistent with the call (checked by a test); the data notice is marked late **and incomplete** (the item asks for who, what for and the customer's rights; the agent's sentence has no rights) | the said / missing part is real data of the hand-off record; «late» and «incomplete» are not computed |
| Report: customer questions left unanswered | written by hand | **not built** |
| Voice assistant example | the answers are written; sources copied from the same domain | the real bot (`#/voicebot`) answers with the reasoner and the RAG |

Nothing in the three demos calls the reasoner, the RAG, the voice
service or the GPU at play time. The live demo holds the single session
slot like the interview demo does and is stopped by a real session. The
voice example is played entirely in the browser, takes no session slot,
and the assistant's lines are spoken by the browser's own speech
synthesis when it has a Spanish voice (otherwise only the text is shown;
a checkbox mutes it). No audio file is shipped, so there is no licence
question; the voice quality is whatever the viewer's browser has, and
was not measured.

## What a real top-scorer comparison needs

- A set of calls of the same type (for example «baja por cobro erróneo»)
  scored by people: who scores, with which sheet, and how their
  agreement is checked (the same reliability work as the interview
  analysis).
- The reaction of the customer after each agent sentence, scored the
  same way, to derive which wordings help.
- Enough calls per moment for a wording to be called «bien valorada»;
  one or two calls are not evidence.
- A decision on privacy (agents see their own report; who else does) and
  on wording: it is coaching material, never a ranking of people or an
  input to hiring or firing.

## Notes

- The report's two strengths and two things to improve, and every row,
  point at lines of the call («Línea 5»); the chips open the full
  conversation at the foot of the page.
- **What is translated.** English: the report (all its texts and the
  names of the mandatory items), the voice example's questions and
  answers and the source titles, the live page's suggestions and every
  label. **What is not:** the call's own sentences (the call is in
  Spanish, as the interview demo); the text of the knowledge-base chunks
  quoted as sources (live card and voice example), which quote a Spanish
  knowledge base and carry the label «(source text, in Spanish)» on the
  English page; the mandatory-items checklist on the live page, which
  lists the domain's Spanish wording exactly as a real call does.
- The top-scorer section is headed «Otras formas de decirlo (ejemplo)» /
  «Other ways to say it (example)» and carries a note: the sentences are
  written for the demo, not taken from real colleagues.
- The demo video tool (`tools/demo_video`) keeps using the interview
  demo; it was not re-recorded.
- Screenshots of a throwaway dev stack (fake models) are in
  `~/.local/share/ragputin/demo-shots/dm250/` (outside the repository).
