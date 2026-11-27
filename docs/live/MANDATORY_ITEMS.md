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

## From the first words (#286)

The check starts with the employee's **first** final sentence: the
tracker reads the list (`MandatoryTracker.start`) before the session's
events are followed, and nothing waits for a warm-up or skips the first
minute. Since #286 the start of a **live** session also sends one check
whose answer is ignored («Hola.»; it never ticks anything and a failure
says nothing), on its own thread, so it never waits in the checks' queue
or blocks anything, and the RAG loads its model and the examples'
vectors (computed once per process) before the first real sentence
instead of on it. A replay or a rebuild sends none.

Time from saying an item to its tick = speech end → final + the RAG
check + the feed (the orchestrator's part is negligible). Measured
2026-11-27 with `experiments/live/lv286/mandatory_timing.py --rag`
(`CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=.`):

| Part | Measured | How |
|---|---|---|
| speech end → final | p50 591–753 ms, p90 739–966 ms | `LIVE_MEASURES.md` (real transcriber, synthetic audio, three runs) |
| orchestrator: final → checklist published, first sentence of the session | p50 7.6 µs, p90 7.9–8.0 µs (500 runs, two runs) | the script's `orchestrator` part: a port that answers at once, inline executor |
| RAG check, **CPU**, in process | first check 3,848–5,313 ms (model load + examples); then p50 384–640 ms, p90 749–759 ms (90 checks of the 18 interview lines of `experiments/live/lv16/mandatory_lines.json`, two runs) | the script's `rag` part: `MandatoryItems.check` with the knowledge model |
| RAG check on the running service (GPU) | **not measured** | it needs the running services |

So on the CPU path an item said in the first sentence is ticked about
1–1.4 s after the speech ends once warm (p50); without the warm-up the
first sentence after a RAG start would have paid the 4–5 s cold check.
(An earlier run of the same measurement gave 6,213 ms, 549 / 774 ms and
5.3 / 5.7 µs: the machine was busy with other work both times.)

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
the same page). The DOM order is the visual order in every layout (the side is
before the conversation in the DOM; the two columns of wide screens are
placed with grid areas, not `order`), so keyboard and screen readers read
it first. The reminder line (item names, `role=status`) sits outside the
fold, so a folded block never hides an unsaid item; the folded line shows
a chevron that turns when open. While something is pending it is open with a red
border; once everything is said it folds into one calm line, «Todo lo
obligatorio está dicho», that can be reopened (it is a disclosure; the
employee's own open/closed choice is kept until the state changes). The
manual tick stays. Support sessions use the same code and the same order:
the block only exists when the session has mandatory items, so nothing
else moves.

## Lists the owner defines (#294)

The manifests' lists above are the **general list** of each product
(«Lista general»: `support.demo` for support calls, `interview.generic`
for interviews). The owner can define more lists from the web
(«Administración › Lo que hay que decir», `#/mandatory`; owner's guide:
[`DEFINE_MANDATORY.md`](DEFINE_MANDATORY.md)), for the kind of call being
made: one per call-centre campaign or commercial offer (`support`) and
one per job offer (`interview`). The general lists stay read-only there
and can be copied as a starting point.

| Part | Where |
|---|---|
| A list: id, name, kind, `add` / `replace`, version, items (`id`, `text`, `phase`: `opening` / `during` / `closing`, `must_say`: 1–6 examples); at most 20 items, 300 characters per text, 200 lists; ids are made by the server from the text | `orchestrator/domain/mandatory_lists.py` |
| Stored outside the repository, like the plans: `<dir>/lists/<id>.json` (current) and `<dir>/lists/<id>/v<version>.json` (every version), `<dir>/offers/<plan>.json` (the list of a job offer); files of mode 600 | `orchestrator/adapters/outbound/fs_mandatory_lists.py`, `serve --mandatory-dir` (default `data/mandatory`) |
| Create, copy, edit (names the version it was based on: 409 when older), delete, review; the offers' choice; the list of a new session | `orchestrator/application/mandatory_lists.py`, routes `/mandatory/...` and `/interview/plans/{id}/mandatory` (same bearer token; on the web behind the access code, through `/api/assist/...`) |

**Which list a session uses.** A support call uses the list chosen on the
live page's start form («Campaña u oferta», remembered in that browser;
default: the general list). A live interview started with a job offer
uses the offer's list (chosen on the offer's page), else the general
list. The list is read when the session starts: an edit made while it
runs applies to the next session.

**Adds or replaces.** A list **adds** its items to the general list by
default, so the data-protection notice is not lost by accident. A list
that **replaces** lists only its own items. A copy of the general list
adds too: the items it copied keep the general ids, and an item with a
general item's id takes that item's place, so nothing is listed twice.
Only a copied item keeps a general id: a new item gets an id made from
its text that is never one of the general list's (an item titled
«Desistimiento» does not replace the general legal item of that id).

**Size.** A save is refused, with a plain reason, when the list together
with the general list would have more than 30 items (the RAG's bound),
an item more than 12 examples, or a body that with a 2,000-character
sentence would pass the RAG's 64 KB limit; a session started with such a
list (the general list grew since) is refused the same way. A save reads
the general list for that and to reserve its ids: when it cannot be
read, nothing is saved (`unavailable`). Saves, creates and deletes run
one at a time.

**The check is the same.** The tracker still reads the domain's list
(its threshold, 0.575, and `remind_seconds` are kept: they belong to the
check, not to one list), combines it with the owner's list and sends the
combination with every sentence: `POST /knowledge/mandatory {"list",
"text"}` (the RAG checks it like a domain's list, with the closest-item
rule; the vectors of the last 16 lists sent are kept). With
`--knowledge fake` the word-count check does the same over the sent
list. The general list alone is still checked by domain, as before.

**What is recorded.** The checklist, the hand-off record
(`mandatory.list`) and the end-of-session summary name the list used:
`{"id", "name", "general", "version", "mode", "domain"}` (the general
list: its id `general.<kind>`, domain and version). A live interview's
stored report keeps it too (`interview.sheet.v1` `mandatory_list`, an
optional field added in #294), and the offer page's table shows it per
round. A later edit never changes what an old session recorded. A
recovered round (#265) is checked again with the list the session used:
the session's own while the orchestrator still has it in memory, else
the list and version its stored report names, read again at that version
(each version is kept). When that version is gone the general list
stands in and the record says so (`mandatory_list.instead_of`). The
rebuilt report names its list too. The offline `rebuild-round` command
still uses the general list (a follow-up). The interview report page
shows the list and version.

**Warnings, never a block.** Saving or «Revisar los ejemplos» returns:
examples of fewer than five words (few words match badly), and examples
whose cosine to an example of **another** item of the list the session
would check reaches the threshold (`POST /knowledge/mandatory/review`,
the same embedding model; with `add` the general items count too): a
sentence says at most one item, the closest, so such an example could
tick the other one. Both are **heuristics, not measurements**: the five
words and the use of the item threshold between two examples were not
measured, and the threshold itself was measured on the seed lists only
(above). Without the RAG only the short examples are checked and the
page says so.

Out of scope of #294: **the AI interviewer (`voicebot.interview`) does
not use these lists** (it has no «Lo que hay que decir» at all), so an
offer's list applies to live-page interviews only; no model writes or
proposes items.
