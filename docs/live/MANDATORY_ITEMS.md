# Mandatory script items (LV-16, #206)

Some things must be said in every conversation of a product: the legal
data-protection notice and the contract conditions of an online
contract, telling a candidate what their data is used for. The live page
lists them in red until the employee says them.

## Where they live

In the knowledge domain's `manifest.yaml`, versioned, under `mandatory`
(`src/domain/knowledge/mandatory.py`):

```yaml
mandatory:
  version: 1            # raise it whenever an item or example changes
  threshold: 0.575      # cosine at which a sentence says an item (measured below)
  remind_seconds: 45    # how often the page brings an unsaid item back
  items:
    - id: desistimiento
      text: "Derecho de desistimiento de 14 días por ser una contratación a distancia"
      phase: closing
      must_say:
        - "Al contratar a distancia dispone de catorce días naturales para desistir sin dar ninguna explicación."
        - "Le recuerdo que tiene un plazo de 14 días para desistir del contrato sin coste."
```

The block is not indexed, so it does not change the domain's index
version. Seed lists (synthetic wording for fictitious companies, not
legal advice): `support.demo` has `aviso-datos`, `condiciones-contrato`,
`desistimiento`, `aceptacion-expresa`; `interview.generic` has
`uso-datos`, `aviso-asistente`, `siguientes-pasos`.

## How an item counts as said

| Step | Where |
|---|---|
| The page shows the list when the session starts | orchestrator `MandatoryTracker.start` → RAG `GET /knowledge/domains/{id}/mandatory` |
| Each **final** sentence of the employee (role `agent` or `interviewer`; also a sentence a `speaker_update` gives to the employee later, as the mono voiceprints do) is checked while items are pending, off the event loop | orchestrator → RAG `POST /knowledge/mandatory {"domain", "text"}` |
| The RAG embeds the sentence with the knowledge model (`paraphrase-multilingual-MiniLM-L12-v2`) and compares it with every `must_say` example; an item's score is its best example. **A sentence says at most one item: the closest one, when it reaches the threshold** | `src/application/knowledge/mandatory.py`, `item_scores` |
| A manual tick needs `{"confirm": true}`, is logged with its time (`ticks`) and shown as "marked by hand" | `POST /sessions/{id}/mandatory/{item}` |
| Nothing else dismisses an item. Every `remind_seconds` the red box flashes and a toast names what is missing; ending the session with items pending asks for confirmation and lists them | live page |
| At the end the hand-off record says what was said, ticked and left unsaid | `{"type": "handoff"}` and `<log-dir>/<session>.<product>.handoff.json` |

The customer's or candidate's sentences are never checked. Without the
RAG (`serve --knowledge fake`) the lists come from the manifests and the
check is a word-count cosine with its own threshold (0.5, not measured;
`orchestrator/adapters/outbound/fs_mandatory.py`), for demos only. A RAG
that does not answer leaves the items pending (they can still be ticked)
and the page says so once.

## Threshold (measured 2026-11-21)

`experiments/live/lv16/mandatory_lines.json`: 43 synthetic employee
sentences written before the run, 25 support and 18 interview; 20 say
one item in other words than its examples, 23 say none (ordinary lines
and near misses on the same topics). Model on the CPU:

```sh
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. \
    python experiments/live/lv16/mandatory_threshold.py \
    --out experiments/live/lv16/mandatory_threshold.json
```

With the closest-item rule (a first run without it gave 7 false hits at
0.55 and 3 at 0.60, most of them the runner-up item of a sentence that
said another one: conditions against acceptance, conditions against
withdrawal):

| threshold | hits of 20 | misses | false hits | precision | recall |
|---|---|---|---|---|---|
| 0.500 | 20 | 0 | 3 | 0.870 | 1.00 |
| 0.550 | 19 | 1 | 2 | 0.905 | 0.95 |
| **0.575** | **19** | **1** | **1** | **0.950** | **0.95** |
| 0.600 | 17 | 3 | 1 | 0.944 | 0.85 |
| 0.650 | 15 | 5 | 1 | 0.938 | 0.75 |
| 0.700 | 12 | 8 | 1 | 0.923 | 0.60 |
| 0.750 | 7 | 13 | 0 | 1.000 | 0.35 |

0.575 is the seed lists' threshold. The miss is "¿Da usted su
conformidad a lo que le he leído para formalizar el alta?" (0.526 to
`aceptacion-expresa`); the false hit that stays below 0.75 is "Para
darle de baja necesito que me confirme el titular del contrato." (0.730
to `aceptacion-expresa`). Labelled item scores ranged 0.526–0.919.

Limits: one small synthetic set, picked and evaluated on the same
sentences (no held-out set), two domains, Spanish only. A false "said"
hides an item that was not said, so a deployment with real legal items
should re-measure on its own sentences and may prefer a higher
threshold; the manual tick (confirmed and logged) covers the misses.

## Where the block sits (#287)

On the live page «Lo que hay que decir» is the **first block of the side
area**, above «Temas del puesto», in every layout (desktop, tablet, phone),
in the scripted demo and in the «Compartir pantalla» mirror (they all draw
the same page). The DOM order is the visual order, so keyboard and screen
readers read it first. While something is pending it is open with a red
border; once everything is said it folds into one calm line, «Todo lo
obligatorio está dicho», that can be reopened (it is a disclosure; the
employee's own open/closed choice is kept until the state changes). The
manual tick stays. Support sessions use the same code and the same order:
the block only exists when the session has mandatory items, so nothing
else moves.
