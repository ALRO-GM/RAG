# Dialogue-act tagger: three approaches, measured

Status: issue #82 (SK-9), 2026-11-15. The automatic tagger of the 13
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
