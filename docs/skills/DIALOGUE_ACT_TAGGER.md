# Dialogue-act tagger: three approaches, measured

Status: issue #82 (SK-9), 2026-11-15; §9 adds the retrieval taggers of
#94 (SK-15). The automatic tagger of the 13
first-layer categories of [`CONVERSATION.md`](CONVERSATION.md) §5 (ids in
[`DIALOGUE_ACT_DATA.md`](DIALOGUE_ACT_DATA.md)), the weighted output of
#73, and how each approach did on our Spanish set and the public corpora.
Every figure below comes from the runs listed in §8; the full tables
(per-category precision/recall/F1, confusion matrices) are in
`experiments/skills/results/act_tagger/` (`scores.json`, `summary.md`,
`margin.json`, `speed.json`).

## 0. Resumen en español

- Se han medido tres formas de etiquetar cada frase: **reglas** (signos
  de interrogación, fórmulas, «ajá»…), un **modelo de lenguaje** que elige
  una etiqueta (Qwen3-8B local en las 1.534 frases; Claude Code en 140) y
  un **codificador XLM-R** afinado con AMI + ICSI-MRDA (inglés) y, además,
  con nuestro conjunto español por validación cruzada por diálogo.
- **El codificador entrenado también con español es el mejor**: 73,1 %
  de acierto en la primera etiqueta y 93,7 % entre las tres primeras
  (validación cruzada de 5 particiones por diálogo, nunca se prueba con
  un diálogo visto al entrenar). El mismo codificador solo con inglés se
  queda en 43,0 %: no transfiere al español sin ejemplos españoles. Las
  reglas dan 51,9 %, Qwen3-8B 55,3 %, Claude Code 74,3 % (solo 140 frases).
- **Por defecto**: el codificador entrenado con todo el conjunto
  español. Combinarlo con las reglas no aporta (73,2 % frente a 73,1 %).
- **Confianza**: las probabilidades están calibradas (ECE 0,021). Una
  etiqueta se marca «dudosa» si no supera la tasa base de su categoría
  **+ 0,45**: es el margen más pequeño con el que el 80 % de las
  etiquetas no dudosas aciertan (propuesta; la curva está en §5).
- **Límites**: las etiquetas españolas son las que pretendía quien
  escribió los diálogos, no un acuerdo de anotadores humanos; los
  corpus públicos son reuniones en inglés.
- **Cuarto enfoque, RAG sobre fichas de teoría (#94, §9)**: recuperar
  fichas y frases etiquetadas sin más (R) acierta el 61,8 %, por debajo
  del codificador; **recuperar y preguntar a Claude Code (RG)** acierta
  el **86,4 %** en las mismas 140 frases en que Claude solo da 74,3 % y
  el codificador 69,3 %. Por defecto se queda el codificador (local,
  gratis, todas las frases); RG queda como revisión opcional de las
  etiquetas dudosas, una llamada por frase.

## 1. Approaches (one port)

All taggers implement `ActTagger` (`src/application/skills/ports.py`):
a probability per category for each sentence, read with its 3 previous
sentences.

| Id | What | Code | Calibration |
|---|---|---|---|
| A `rules` | First rule that fires: question mark, interrogative word, back-channel alone, greeting/thanks, can-you request, proposal/request/commitment/management cue, agreement/disagreement opener after another speaker, opinion marker, else default (split by "after a question", short, statement). Spanish and English cue lists, written before measuring and not tuned on the results (two rule-order bugs were fixed from unit tests; the Spanish figures did not change) | `src/domain/skills/act_rules.py`, `RuleTagger` | Outcome table per rule key: the share of each category among held-out sentences with that key, smoothed towards the base rates |
| B `llm:<model>` | The existing judge prompt (`skills judge_acts`, `src/domain/skills/act_prompting.py`: definitions + decision order, **no worked examples**), one label | `LlmTagger` | Outcome table per answered label |
| D `encoder` | XLM-R base, 13-way head, input `context [SEP] target` with `[=]`/`[>]` marking same/other speaker (no names), trained with the compatible-set loss `−ln Σ_{c∈set} p(c)` on AMI + MRDA train splits; frozen word embeddings (bf16), 8-bit AdamW, gradient checkpointing, so it fits ~1.6 GB of a shared GPU | `src/adapters/outbound/encoder_act_tagger.py`, `experiments/skills/act_tagger_train.py` | Temperature scaling on a held-out split |
| D+es `encoder-es` | D continued for 3 epochs on 4 of 5 Spanish dialogue folds; the shipped model on all 51 dialogues | same, `--spanish-folds`, `--spanish-all` | Temperature on the last fifth of its Spanish training data |
| F `combined` | Per category, the approach whose claims of it were most precise on the other folds; the most accurate approach answers the rest | `CombinedTagger`, `choose_routes` | Each part's own |

Splits of the public corpora are ours, by meeting (`split_of`: SHA-256
of the dialogue id; 15 % test, 10 % dev, rest train), not the
published ones. Switchboard-DA (non-commercial) and DIHANA are used for
evaluation only.

## 2. Results on the Spanish set (1,534 sentences, 51 dialogues)

5-fold cross-validation by dialogue for everything learned from it
(outcome tables, temperatures, routes, Spanish fine-tuning). Reference:
the primary label for top-k, F1, ECE; the weights for the log-loss.

| Approach | Sentences | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|---:|
| A rules | 1534 | 0.519 | 0.731 | 0.449 | 1.525 | 0.027 |
| D encoder (English only, T from English dev) | 1534 | 0.430 | 0.747 | 0.345 | 1.841 | 0.206 |
| D encoder + Spanish temperature | 1534 | 0.430 | 0.747 | 0.345 | 1.715 | 0.065 |
| F rules + English encoder | 1534 | 0.550 | 0.739 | 0.484 | 1.508 | 0.060 |
| **D+es encoder-es** | 1534 | **0.731** | **0.937** | 0.693 | **0.799** | **0.021** |
| F rules + encoder-es | 1534 | 0.732 | 0.938 | **0.699** | 0.792 | 0.021 |
| B Qwen3-8B (4-bit, local) | 1534 | 0.553 | 0.778 | 0.509 | 1.548 | 0.018 |
| B Claude Code (opus) | 140 | 0.743 | 0.879 | 0.660 | 1.158 | 0.124 |
| encoder-es on the same 140 | 140 | 0.700 | 0.914 | 0.637 | 0.866 | 0.085 |

Per-category F1:

| Approach | preg | resp | info | opin | prop | peti | acep | rech | comp | escu | gest | rela | otro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | .84 | .82 | .42 | .27 | .36 | .45 | .27 | .12 | .02 | .57 | .13 | .76 | .80 |
| encoder (English) | .85 | .43 | .40 | .20 | .26 | .26 | .28 | .22 | .02 | .39 | .00 | .83 | .34 |
| encoder-es | .93 | .90 | .68 | .34 | .70 | .64 | .67 | .61 | .75 | .67 | .53 | .87 | .71 |
| rules + encoder-es | .93 | .90 | .68 | .34 | .70 | .64 | .67 | .61 | .72 | .67 | .53 | .87 | .81 |
| Qwen3-8B | .81 | .33 | .57 | .38 | .51 | .41 | .65 | .47 | .47 | .62 | .37 | .67 | .36 |
| Claude Code (140) | .82 | .85 | .53 | .74 | .81 | .40 | .70 | .59 | .80 | .76 | .84 | .74 | .00 |

Reading:

- **English-only training does not transfer** to these Spanish
  workplace dialogues (0.430): it is beaten by the rules. Respuesta,
  Gestión and Otro collapse; the training corpora are dominated by
  statements and back-channels.
- **A few hundred Spanish sentences per fold close most of the gap**
  (0.430 → 0.731). This is the step `CONVERSATION.md` §6.1 option E
  expected; it is also the result most exposed to the limits of §7 (the
  set is written by one author, so test folds share style with training
  folds).
- **Opinión stays weak** for every approach except Claude (0.34 for the
  encoder): its border with Información is the hardest call.
- **Combining with the rules does not pay**: +0.001 top-1 (Otro .71 →
  .81, Compromiso .75 → .72). The rule of the evaluation script asks a
  combination for +0.01 top-1 over the best single approach.
- Claude Code is the best on its 140 sentences (0.743 vs 0.700 for
  encoder-es on the same ones), with poorer calibration of its single
  label (ECE 0.124 from a 140-sentence outcome table).

## 3. Results on the public corpora (test splits, compatible sets)

Only A and the English-trained D were run on them (B and D+es were not:
see §8).

| Corpus | Approach | Sentences | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| AMI | rules | 22805 | 0.521 | 0.750 | 0.376 | 1.523 | 0.017 |
| AMI | encoder | 22805 | 0.660 | 0.902 | 0.553 | 0.996 | 0.006 |
| MRDA | rules | 18507 | 0.657 | 0.823 | 0.268 | 1.167 | 0.186 |
| MRDA | encoder | 18507 | 0.776 | 0.947 | 0.546 | 0.685 | 0.029 |
| SwDA (NC, eval only) | rules | 30176 | 0.672 | 0.919 | 0.377 | 1.001 | 0.003 |
| SwDA (NC, eval only) | encoder | 30176 | 0.646 | 0.905 | 0.432 | 1.084 | 0.067 |
| DIHANA (es) | rules | 3012 | 0.714 | 1.000 | 0.662 | 0.581 | 0.034 |
| DIHANA (es) | encoder | 3012 | 0.382 | 0.690 | 0.400 | 2.335 | 0.301 |

Rules on the public corpora are calibrated on each corpus's own train
split. "Macro-F1" averages only the categories a corpus names alone
(`DIALOGUE_ACT_DATA.md` §3); with sets, a prediction inside the set is
right. The encoder learned AMI and MRDA (in-domain); SwDA (telephone) is
a domain shift where it does no better than the rules; DIHANA
(Spanish human–machine timetable dialogues) again shows that English
training does not carry to Spanish.

## 4. The default and its output

`knowledge/dialogue_acts/act_tagger.json` (written by the evaluation):

- tagger **`encoder`**: `data/models/act-tagger/xlmr-base-ami-mrda-es`
  (English model + 3 epochs on all 51 Spanish dialogues), temperature
  0.993 fitted on the last fifth of its Spanish data;
- the rule table (fitted on the whole Spanish set) is kept: `--tagger
  rules` or `combined`, and the fallback when the weights are missing;
- base rates = the Spanish set's mean label weights; **margin 0.45**.

```sh
python -m src skills acts --transcript data/transcripts/x.transcript.v1.json
#   --tagger default|rules|encoder|combined   --context 3   --output FILE
```

writes `<session>.acts.json` next to the transcript in the shape the
second layer reads (`src/domain/skills/act_sequence.py`; the sentence
below is an illustration of the shape, the threshold is the real one):

```json
{"schema": "acts.v1", "session": {"id": "x"},
 "tagger": {"name": "encoder", "version": "xlmr-base-ami-mrda-es",
            "margin": 0.45, "thresholds": {"pregunta": 0.574, "...": 0}},
 "utterances": [{"id": "t3.s1", "turn": 3, "speaker": "Ana", "text": "…",
   "start": 12.4, "end": 14.0,
   "labels": [{"label": "pregunta", "weight": 72, "doubtful": false},
              {"label": "gestion", "weight": 20, "doubtful": true}],
   "rest": 8, "responds_to": null, "by": "encoder"}]}
```

Top 3 in whole %, not renormalised (`rest` is what is left), a label
below its category's base rate + margin marked `doubtful`. The tagger
does not link replies (`responds_to` is null). On the Spanish demo
session (`examples/skills/transcripts/demo_es.txt`, 83 sentences) the
first label was doubtful in 24.

The weights are not in git. To rebuild them, §8 commands 1, 3.

## 5. The "dudoso" margin, from the data

`margin.json`, on the cross-validated encoder-es predictions (4,343
shown labels for 1,534 sentences):

| Margin | Clear labels | Right among clear | Right among doubtful | Clear first labels | Right among them |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 2825 | 0.491 | 0.036 | 1534 | 0.733 |
| 0.10 | 2127 | 0.611 | 0.065 | 1534 | 0.733 |
| 0.20 | 1753 | 0.695 | 0.086 | 1519 | 0.738 |
| 0.30 | 1502 | 0.751 | 0.111 | 1436 | 0.763 |
| 0.40 | 1319 | 0.792 | 0.132 | 1316 | 0.793 |
| **0.45** | 1225 | **0.815** | 0.142 | 1225 | 0.815 |
| 0.50 | 1141 | 0.838 | 0.152 | 1141 | 0.838 |

Proposal: **0.45**, the smallest margin at which ≥ 80 % of the labels
shown as clear are right (target chosen here, `--target 0.8`). At 0.45
only first labels stay clear and 309 of 1,534 first labels (20 %) are
marked doubtful. A lower target (e.g. 0.75 → margin 0.30) keeps more
labels clear; it is the owner's call.

## 6. Cost (measured on the workstation, RTX 3070 Ti 8 GB shared)

| What | Figure |
|---|---|
| English training (161,644 sentences, 1 epoch, 5,051 steps, batch 16 × 2, max 128 tokens) | 1,435 s; peak 1.56 GB VRAM; 1.77 GB free at start (another project's worker held ~5.6 GB); no batch lost to OOM |
| Spanish fold training (5 × ~2,900 sentence-passes, 3 epochs) | 23–70 s each; peak 1.52 GB |
| Shipped Spanish model (all 51 dialogues) | 32.4 s; peak 1.52 GB |
| Encoder inference, GPU (fp16, 1,534 sentences) | 1,821 sentences/s; peak 0.69 GB |
| Encoder inference, CPU (12 threads, machine shared with other jobs) | 23.8 sentences/s |
| Rules, CPU | 44,290 sentences/s |
| Qwen3-8B 4-bit on the GPU, 1,534 sentences | 15 min 48 s (1.62 sentences/s) |
| Claude Code (opus), 140 sentences | 5 min 47 s; 140 calls |
| Qwen3-4B on the CPU | ~24 s/sentence (10-sentence probe, then a 300-sentence run was stopped when the GPU freed up) |

## 7. Limits

- **The Spanish labels are the author's intended labels, not human
  gold.** No second annotator; κ is unknown. The cross-validation keeps
  dialogues apart, but every dialogue was written by the same person,
  so 0.731 measures agreement with that person's style, and is an upper
  bound for real meetings.
- **Text written as dialogue, not ASR output**: real transcripts lose
  punctuation and back-channels (`CONVERSATION.md` §5.4); not measured.
- **The public corpora are English meetings** (AMI two-thirds role-play,
  MRDA research groups), SwDA telephone (non-commercial, evaluation
  only), DIHANA a train-timetable machine. Their figures say how the
  tagger learned those corpora, not how it does on Spanish workplace
  speech.
- The label mappings to our categories are ours (`DIALOGUE_ACT_DATA.md`
  §6); the splits are ours, not the published ones.
- B used the existing prompt without worked examples (the ticket asked
  for few-shot): adding examples is untested.
- Claude Code was measured on 140 sentences only (owner's quota), so its
  per-category F1 rests on a handful of sentences per category.
- The "dudoso" margin and the +0.01 rule for combinations are proposals
  set on the same set they are reported on.
- Labels are evidence for a human assessor, never a decision
  (`CONVERSATION.md` §5.5).

## 8. How the figures were produced

```sh
# 0. Imports (DIALOGUE_ACT_DATA.md §5), into the git-ignored data/
python -m src skills dialogue_acts_import --corpus ami --source <zip> \
    --output data/dialogue_acts/ami.jsonl            # mrda, swda, dihana, own
# 1. English model
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/skills/act_tagger_train.py \
    --output data/models/act-tagger/xlmr-base-ami-mrda
# 2. Spanish folds (measurement)
... act_tagger_train.py --spanish-folds --init data/models/act-tagger/xlmr-base-ami-mrda \
    --output data/models/act-tagger/xlmr-es-folds --config '{"epochs": 3}'
# 3. Shipped model
... act_tagger_train.py --spanish-all --init data/models/act-tagger/xlmr-base-ami-mrda \
    --output data/models/act-tagger/xlmr-base-ami-mrda-es --config '{"epochs": 3}'
# 4. Judges (B)
python -m src --device cuda skills --judge local --local_model Qwen/Qwen3-8B \
    --local_bits 4 --local_policy fixed judge_acts --sample 0 \
    --output experiments/skills/results/act_tagger/llm_qwen3-8b_4bit.json
python -m src skills --judge claude_code judge_acts --sample 140 \
    --output experiments/skills/results/act_tagger/llm_claude_code.json
# 5. Evaluation, default card, summaries
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/skills/act_tagger_eval.py \
    --encoder data/models/act-tagger/xlmr-base-ami-mrda \
    --es-folds data/models/act-tagger/xlmr-es-folds \
    --es-all data/models/act-tagger/xlmr-base-ami-mrda-es \
    --llm experiments/skills/results/act_tagger/llm_qwen3-8b_4bit.json \
    --llm experiments/skills/results/act_tagger/llm_claude_code.json
```

Not run: Qwen3-4B on the full set (GPU memory was taken by other jobs;
on the CPU it needs ~24 s per sentence: command 4 with `Qwen/Qwen3-4B`
and `--device cpu`); B and D+es on the public corpora; a few-shot prompt
for B. Training with `--seed` other than 13 (no variance estimate).

## 9. Retrieval over theory cards: R and RG (#94)

A fourth approach, measured on the same Spanish set, folds and scores
as §2. Results: `experiments/skills/results/act_rag/scores.json` (every
configuration, ablation, combination and baseline, with answerer
counts), `rg_claude_code.json` (the 140 verdicts with their reasons)
and the summaries `act_tagger_rag_cards_folds_e5_small_dense.json`,
`act_tagger_encoder_es___r_ensemble.json`,
`act_tagger_encoder_es__r_on_dudoso.json`,
`act_tagger_rg_claude_code__140.json` that the study page charts.

### 9.1 Method

- **Theory corpus** (`knowledge/dialogue_acts/theory/`, Spanish,
  written for this project from `CONVERSATION.md` §5 and
  `DIALOGUE_ACTS_GUIDE.md`): `categories.json`, one card per category
  (definition, function, cues, what it is *not* with the neighbour it
  is confused with, boundary cases, examples); `sequences.json`, 21
  sequence cards (pregunta→respuesta, propuesta→aceptación/rechazo,
  petición→compromiso, opinión→rechazo→propuesta, cierre→aceptaciones,
  clarification chains…) with every position tagged. 369 card
  examples, all new sentences: `test_card_examples_are_not_evaluation_sentences`
  checks none is a sentence of the evaluation set. Chunks: 279 single,
  121 pair, 46 window.
- **Overlap between definitions** (cosine of definition + function):
  the draft's closest pairs were reduced by rewriting the cards. Before
  → after, closest pair: MiniLM propuesta/petición 0.828 → petición/
  compromiso 0.745; e5-small aceptación/rechazo 0.945 → 0.921 (e5's
  cosines sit high for any two Spanish texts; the draft figures are in
  `act_rag/overlap_draft.json`).
- **Three views**, each its own query and its own chunks: the target
  alone (single), previous + target (pair), up to three previous +
  target (window). A chunk votes only for the category of the position
  the target occupies in it, so a retrieved "pregunta→respuesta" pair
  votes respuesta for an answer and the previous sentence's evidence
  never votes for the target. Views are fused with weights and a
  temperature fitted on the other four folds (`FusionWeights`).
- **Index**: cards only, or cards + the labelled sentences of the other
  four folds (the test dialogue is never in the index); embedders
  `paraphrase-multilingual-MiniLM-L12-v2` and `multilingual-e5-small`;
  dense, or hybrid with BM25 (share 0.3, fixed before measuring).
- **R**: the fused, calibrated vote. **RG**: the same evidence split by
  position goes into a prompt (`src/domain/skills/act_rag_prompting.py`)
  with the target marked and the context marked as context; the judge
  answers up to three labels with weights and a one-line reason citing
  a card; its weights are mixed with R (share of R fitted on the other
  folds: 0.1 in every fold). Code: `src/domain/skills/act_retrieval.py`,
  `src/application/skills/rag_act_tagger.py` (`RetrievalActTagger`,
  `RetrievalJudgeTagger`, both `ActTagger`), cards loaded by
  `src/adapters/outbound/json_act_cards.py`.

### 9.2 R on the 1,534 sentences

| Index / embedder / similarity | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE | Answerer → pregunta |
|---|---:|---:|---:|---:|---:|---:|
| cards / MiniLM / dense | 0.335 | 0.615 | 0.266 | 2.088 | 0.071 | 3 |
| cards / MiniLM / hybrid | 0.424 | 0.683 | 0.363 | 1.887 | 0.078 | 2 |
| cards / e5-small / dense | 0.387 | 0.709 | 0.355 | 1.927 | 0.094 | 0 |
| cards / e5-small / hybrid | 0.426 | 0.698 | 0.395 | 1.875 | 0.081 | 6 |
| cards+folds / MiniLM / dense | 0.476 | 0.739 | 0.424 | 1.664 | 0.055 | 5 |
| cards+folds / MiniLM / hybrid | 0.533 | 0.782 | 0.485 | 1.497 | 0.045 | 4 |
| **cards+folds / e5-small / dense** | **0.618** | **0.852** | **0.553** | **1.271** | 0.071 | 1 |
| cards+folds / e5-small / hybrid | 0.569 | 0.809 | 0.524 | 1.405 | 0.064 | 2 |

"Answerer → pregunta" counts the error the owner feared: a sentence
right after another speaker's pregunta, not itself a pregunta, labelled
pregunta (186 such sentences). For comparison, encoder-es makes 0 of
these errors and Qwen3-8B 9.

Reading:

- **The cards alone are weak** (best 0.426, below the rules' 0.519).
  Adding the labelled sentences of the other folds lifts R to 0.618.
- **The definitions add nothing measurable to R**: removing the
  definition, cue and boundary chunks (keeping only the card examples
  and the fold sentences) gives 0.617 for the best configuration
  (0.618 with them) and 0.426 for cards/e5/hybrid (same). Nearest
  neighbour voting is driven by example sentences; the theory text
  matters only once a judge reads it (RG).
- Hybrid BM25 helps with the cards only and hurts with e5 + folds.
- Cost: about 36 s on the CPU for the 1,534 sentences of the first
  e5-small configuration, embedding included (later configurations
  reuse the cached embeddings: 1.3–4.6 s).

**View ablation** (best configuration; F1 of the context-dependent
categories):

| Views | Top-1 | Macro-F1 | Log-loss | preg | resp | acep | rech | comp | Answerer → pregunta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 0.521 | 0.482 | 1.594 | .87 | .35 | .61 | .31 | .53 | 3 |
| pair | 0.605 | 0.534 | 1.437 | .85 | .84 | .58 | .51 | .56 | 0 |
| window | 0.559 | 0.508 | 1.540 | .89 | .59 | .57 | .43 | .53 | 0 |
| single+pair | 0.614 | 0.547 | 1.310 | .85 | .80 | .58 | .52 | .60 | 2 |
| single+window | 0.572 | 0.525 | 1.404 | .88 | .57 | .60 | .38 | .55 | 2 |
| single+pair+window | 0.618 | 0.553 | 1.271 | .88 | .77 | .60 | .49 | .60 | 1 |

Context helps: the pair view takes respuesta from .35 to .84 and
rechazo from .31 to .51 while pregunta moves by at most .02; it also
removes the answerer-as-pregunta errors. Aceptación does not gain.
The fitted weights were single 1, pair 2, window 1 in all five folds.

### 9.3 Combinations with the encoder (1,534 sentences)

The encoder-es predictions here are recomputed from the fold models
(`data/models/act-tagger/cache/encoder_es_folds.json`) and score 0.729
top-1, 0.802 log-loss, against 0.731 / 0.799 in §2: the same models,
a different inference run (small numeric differences); all rows below
use the recomputed ones.

| Approach | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|
| encoder-es | 0.729 | 0.939 | 0.689 | 0.802 | 0.020 |
| R | 0.618 | 0.852 | 0.553 | 1.271 | 0.071 |
| encoder-es, R replaces "dudoso" (302 sentences) | 0.715 | 0.907 | 0.668 | 0.931 | 0.032 |
| encoder-es, R averaged in on "dudoso" | 0.739 | 0.941 | 0.699 | 0.816 | 0.032 |
| encoder-es + R ensemble (R share 0.1 in every fold) | 0.735 | 0.943 | 0.694 | 0.791 | 0.021 |

"Dudoso" uses §5's margin 0.45 (set on this same set). The best
combination gains +0.010 top-1, exactly the threshold §2 asks of a
combination, with a worse log-loss; the ensemble gains +0.006 top-1
and 0.011 log-loss. Neither is adopted.

### 9.4 RG with Claude Code on #82's 140 sentences

Same 140 sentences as the zero-shot Claude run of §2 (paired
comparison), judge `claude-code:opus` through `claude -p` on the
subscription; 140 calls in total (one per sentence, within the
ticket's 150), none unparsed. The run was cut by a reboot after 70
verdicts and resumed from the saved file; the 70 resumed calls took
223 s (3.2 s per sentence; the first half's time was lost).

| Approach (same 140) | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE | Answerer → pregunta |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.479 | 0.671 | 0.355 | 1.653 | 0.109 | 0 / 16 |
| R | 0.579 | 0.829 | 0.470 | 1.524 | 0.136 | 0 / 16 |
| encoder-es | 0.693 | 0.907 | 0.630 | 0.865 | 0.108 | 0 / 16 |
| Claude Code zero-shot (§2) | 0.743 | 0.879 | 0.660 | 1.158 | 0.124 | 0 / 16 |
| **RG Claude Code, weights + R** | **0.864** | **0.986** | **0.849** | **0.458** | **0.082** | 0 / 16 |
| RG Claude Code, outcome table | 0.864 | 0.929 | 0.849 | 0.779 | 0.149 | 0 / 16 |

Per-category F1 of RG (weights + R): preg .90, resp .89, info .67,
opin .89, prop .93, peti .80, acep .76, rech .82, comp .92, escu .91,
gest .88, rela .87, otro .80. Every category's F1 is higher than
with Claude zero-shot; the largest gains are peticion .40 → .80,
otro .00 → .80, opinion .74 → .89, gestion .84 → .88 the smallest.

With 140 sentences the 95 % normal-approximation interval is about
±0.057 around 0.864 and ±0.072 around 0.743; no paired significance
test was run.

RG with the local judge on all 1,534 sentences is in §9.7.

### 9.5 Decision

- **Keep encoder-es as the default tagger.** It is local, free and
  labels every sentence; R alone is 0.11 below it and its combinations
  do not clear the +0.01 rule with a better log-loss.
- **RG with Claude Code is the most accurate tagger measured** (0.864
  vs 0.743 for the same judge without retrieval, on the same
  sentences). It costs one subscription call per sentence (~3 s), so
  the proposal is to use it as an optional second pass over the
  encoder's "dudoso" labels (≈ 20 % of sentences, §5), not for every
  sentence. That route was not measured end to end (RG was run on the
  140 sentences of #82, not on the dudoso subset) and it is not wired
  into `skills acts` in this ticket; the taggers exist behind the
  `ActTagger` port and in the experiment script.
- The theory cards matter for the judge, not for the vote: R gains
  nothing from them (§9.2).

### 9.6 Limits and how to reproduce

- **Circularity.** The Spanish set, the theory cards and the judge's
  prompt come from the same author (and the cards were written knowing
  the categories' guide the set follows). The card examples are kept
  disjoint from the evaluation sentences, and the index never holds the
  test dialogue, but the style is shared: 0.864 measures agreement with
  that author, an upper bound for real meetings.
- Labels are the author's intended labels (§7), not human agreement.
- RG figures rest on 140 sentences: per-category F1 on a handful each.
- The fusion grid, the hybrid share 0.3 and the R share in the mix were
  fitted or fixed on this set (fusion weights and shares on the other
  folds of each test fold).

```sh
# R, ablations, combinations, scoring of saved RG verdicts, summaries
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. python \
    experiments/skills/act_rag_eval.py --es-folds data/models/act-tagger/xlmr-es-folds
# RG with Claude Code on #82's 140 sentences (resumes from the saved verdicts)
... act_rag_eval.py --configs cards+folds/e5-small/dense --run-judge claude_code \
    --ids experiments/skills/results/act_tagger/llm_claude_code.json
# RG with the local judge on every sentence (§9.7; resumable, ~1 h on the 3070 Ti)
... act_rag_eval.py --configs cards+folds/e5-small/dense --run-judge local \
    --local-model Qwen/Qwen3-8B
```

### 9.7 RG with the local Qwen3-8B on all 1,534 sentences, and as a second pass (#100)

Judge `Qwen/Qwen3-8B` 4-bit on the GPU (the 8B fitted; the 4B was not
needed), config cards+folds / e5-small / dense, no Claude Code or other
hosted call. 1,534 verdicts, 1,522 new calls in 3,697 s (2.4 s each; 12
came from a 12-sentence trial), 11 unparsed (fall back to R). Figures
in `experiments/skills/results/act_rag/rg_local_scores.json`.

| Approach (1,534) | Top-1 | Top-3 | Macro-F1 | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|
| encoder-es | 0.729 | 0.939 | 0.689 | 0.802 | 0.020 |
| R | 0.618 | 0.852 | 0.553 | 1.271 | 0.071 |
| RG local, weights + R (mix share 0.5-0.6) | 0.698 | 0.869 | 0.656 | 1.141 | 0.024 |
| RG local, outcome table | 0.699 | 0.817 | 0.659 | 1.282 | 0.027 |
| encoder-es, RG replaces "dudoso" | 0.741 | 0.915 | 0.695 | 0.909 | 0.040 |
| **encoder-es, RG averaged in on "dudoso"** | **0.747** | 0.942 | 0.703 | **0.795** | 0.022 |

The second pass sends only the 302 sentences "dudoso" at margin 0.45
(302 judge calls instead of 1,534). Paired against encoder-es on the
same sentences (exact McNemar on top-1; paired bootstrap, 2,000
resamples, on the per-sentence log-loss):

| Comparison | Only candidate right | Only encoder right | McNemar p | Log-loss gain [95 %] |
|---|---:|---:|---:|---|
| RG alone vs encoder-es (1,534) | 149 | 197 | 0.011 (encoder better) | -0.339 [-0.407, -0.268] |
| replace vs encoder-es (302 doubtful) | 66 | 48 | 0.111 | -0.546 [-0.767, -0.342] |
| average vs encoder-es (302 doubtful) | 66 | 38 | 0.008 | +0.035 [-0.032, +0.105] |

Reading. The local 8B judge is worse than the encoder alone (0.698 vs
0.729, unlike Claude Code's 0.864 on 140 sentences, §9.4: those figures
are not comparable, different judge and sentences). As a second pass
averaged into the encoder's "dudoso" labels it gains +0.018 top-1
(McNemar p = 0.008) and 0.007 log-loss: formally over the +0.01 rule of
§2 with a better log-loss, but the log-loss interval includes zero, the
margin 0.45 was set on this same set and the 50/50 average is fixed,
not fitted. Replacing is worse.

**Decision: not wired into `skills acts`.** The measured configuration
uses an index that includes the other folds' labelled sentences and
fusion weights fitted per fold; `acts` has no equivalent production
index or weights, so wiring would ship something other than what was
measured. The gain is small for a judge call on 20 % of sentences and
rests on the author's own set (§9.6). Re-open if a stronger local judge
or the production index is measured.
