# Two assistance modes: «pincelada» and «guiado» (LV-16, #206)

| Mode | What the suggestion is | How the page shows it |
|---|---|---|
| `pincelada` (default) | a hint of at most 8 words saying where to steer ("Preguntar la fecha límite exacta del cargo") | the card's normal text |
| `guiado` | the exact sentence or question to say, polite, to the other person, under 35 words | a separate block, "Di:", large type, easy to read aloud |

Templates `suggester.support@2` and `suggester.interview@2`
(`src/domain/reasoning/templates.py`) take the mode from the request's
`constraints.mode` (`reasoner.request.v1`). The mode and the number of
suggestions go in the **user** prompt, so the system prompt is the same
in both modes and switching during a session keeps the warm Claude
process (it is keyed by the system prompt). A `pincelada` longer than 8
words is cut to 8 by the parser (ending "…"). Everything else is `@1`:
grounding, the support signals with their definitions, and for
interviews the ban on rating, scoring, labelling or describing the
candidate's emotions. Output cap 240 tokens (`@1` support: 200).

The default comes from «Ajustes» (`live.assist_mode`, read by the
orchestrator when a session starts, `orchestrator/domain/live_choices.py`);
the page switches it during a session with `POST /sessions/{id}/mode`
(the next call uses it).

## Measurement (2026-11-21)

Real calls through the RAG's own live path: the warm `claude -p` stream
backend, Haiku, effort `low`, one process per template, behind the
`Reasoner` (so parsing, grounding and the cut are the production ones).
Synthetic transcripts of `examples/platform`; each trigger (a customer's
or candidate's sentence) asked once in each mode with the products'
settings: support one suggestion, deadline 2,800 ms; interview three,
1,800 ms. Window of 12 utterances, retrieval approximated by word
overlap with the seed domain's chunks.

```sh
PYTHONPATH=. python experiments/live/lv16/assist_modes.py \
    --out experiments/live/lv16/assist_modes.json
PYTHONPATH=. python experiments/live/lv16/assist_modes_diag.py
```

29 real calls in all: 24 measured + 2 priming calls (one per process),
then 3 to look into the empty interview answers (below).

| | calls | ok | no suggestion | answer p50 / max (ms) | first token p50 (ms) | words per suggestion | output tokens median |
|---|---|---|---|---|---|---|---|
| support, pincelada | 6 | 5 | 1 | 1,319 / 1,710 | 1,048 | 7–8 (2 of 5 cut to 8) | 212.5 |
| support, guiado | 6 | 5 | 1 | 985 / 1,730 | 664 | 9–21 | 107 |
| interview, pincelada | 6 | 3 | 3 | 1,003 / 1,113 | 721 | 8 (1 of 3 cut) | 79 |
| interview, guiado | 6 | 3 | 3 | 986 / 1,330 | 716 | 8–13 | 82 |

Every call answered within its deadline (support 2.8 s, interview
1.8 s); none timed out. Support kept the same signals in both modes on
every trigger (stated_frustration, complaint, urgency, churn_risk on 5
of 6); the empty support trigger (`t7.s0`, a question about the return
period) was empty in both modes. `pincelada` spent about twice the
output tokens of `guiado` in support (hidden thinking counts there) and
was slower at the median; not investigated further with 6 calls each.

**Short quality check** (read by hand, all 16 answers in
`assist_modes.json`):

- `pincelada` reads as a direction when the model wrote a hint
  ("Preguntar el motivo y avisar a retención", "Pedir números: volumen
  de pedidos y mejora obtenida"); 3 of 8 answers were sentences the
  parser cut ("Entiendo que es la segunda vez; vamos a…"), which read
  unfinished.
- `guiado` gives sentences that can be read aloud as they are, in
  *usted* in support ("¿Hasta qué fecha exacta necesita que se lo
  devolvamos?", "Entiendo que quiera darse de baja; ¿me cuenta qué es lo
  que más le ha molestado?") and in *tú* in the interview, like the
  transcript; one interview question ("¿Puedes explicarme qué es una
  caché…?") came from a knowledge chunk rather than the candidate's
  last sentence.
- Interviews: no text rated or described the candidate (a word flag
  for ratings and emotions found 0, and none on reading).
- Interview: half the calls in each mode came back empty. Re-asking two
  of them (`assist_modes_diag.jsonl`): one answered three grounded
  suggestions, the other timed out at 1.8 s while writing its third
  suggestion. Three suggestions under the 1.8 s deadline are tight; the
  product settings were left as they are.

Limits: 6 calls per cell, one machine, synthetic Spanish transcripts,
subscription latency varies with load.
