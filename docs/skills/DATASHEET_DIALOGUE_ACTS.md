# Datasheet: the dialogue-act set and the data behind it

Date: 2026-11-16. Issue #86 (part 4). Format after Gebru et al. (2021),
"Datasheets for Datasets", adapted. This is the datasheet of the
dialogue-act category set and of the data that trains and tests the
tagger (`FAIRNESS.md` §3.5 row I4; the inventory of §3.1 row T2:
languages, accents, microphones against intended use). It complements
the datasheet of the Spanish set itself,
`examples/skills/dialogue_acts/README.md`. **Per-group metrics do not
exist**: no corpus here carries group labels we use, and no result is
split by sex, age, accent or first language. See `FAIRNESS.md` and the
audit template (`src/application/skills/fairness_report.py`,
`experiments/skills/fairness_audit.py`). No real recording is described
or copied here.

## 1. Motivation

A first-layer label per sentence ("what does this sentence do") as
evidence for a human assessor and as input of the conversation events
(`CONVERSATION.md` §5, `DIALOGUE_ACT_TAGGER.md`). The 13 categories
(`CATEGORIES`, `src/domain/skills/dialogue_acts.py`): pregunta,
respuesta, informacion, opinion, propuesta, peticion, aceptacion,
rechazo, compromiso, escucha, gestion, relacion, otro. Información is
split into Información and Opinión by the owner (decision 1 on #57). A
sentence may carry up to three weighted labels (decision 2 on #57).
Respuesta, aceptacion and rechazo must point at the sentence they reply
to. The categories are the project's choice, not an established
standard; ISO 24617-2 and other schemes are compared in
`CONVERSATION.md` §3-§5.

## 2. Composition and inventory (FAIRNESS T2)

| Source | Language | Setting and register | Real or synthetic | Size (read by the importer or by command) | Speakers' demographics, accents, microphones | Licence |
|---|---|---|---|---|---|---|
| Original Spanish set, `examples/skills/dialogue_acts/` | Spanish (Spain) | 9 conversation types, workplace and daily life, written text from formal to colloquial, with fillers and a few deliberate ASR slips | **Synthetic**: written by an AI coding assistant, labelled by its author | 51 dialogues, 1,534 sentences, 20 invented first names, 479 reply links, 6 weighted sentences (command below) | Not recorded: no audio, no demographics, no accent, no microphone; speakers are names only (`enrolled: false` for all 136) | Project's own text; licence not stated |
| AMI dialogue acts | English (many non-native) | Design-team meetings, two thirds role-play | Real speech, scripted scenarios in part | 139 meetings, 110,795 segments | Not recorded by us; not used as a factor | CC BY 4.0 |
| ICSI MRDA | English | Real research-group meetings | Real | 75 meetings, 111,179 segments | Not recorded by us | CC BY 4.0 |
| Switchboard-DA (SwDA) | English | Telephone, strangers, assigned topics | Real | 1,155 conversations, 223,487 utterances | Not recorded by us | CC BY-NC-SA 3.0; **internal evaluation only**, never training a shipped model |
| DIHANA | Spanish | Wizard-of-Oz telephone, train timetables, person and "machine" | Real recorded calls, task-bound | 900 dialogues, 23,493 segments read by the importer; 225 speakers, 23,547 segments in `CONVERSATION.md` §3.1 | Not recorded by us | CC BY 4.0; **evaluation only** |

Sources of the rows: `DIALOGUE_ACT_DATA.md` §1 and §4;
`examples/skills/dialogue_acts/README.md`; `DIALOGUE_ACT_TAGGER.md` §1.
Label distribution of the Spanish set by primary label (command
below): aceptacion 165, compromiso 97, escucha 123, gestion 97,
informacion 171, opinion 93, otro 36, peticion 52, pregunta 189,
propuesta 104, rechazo 98, relacion 115, respuesta 194. Rare classes are
oversampled on purpose (Compromiso 6.3% against about 0.3% in MRDA).
Public corpora: see the table in `DIALOGUE_ACT_DATA.md` §4.

**Coverage gaps against the intended use** (workplace meetings in Spanish
and English, from transcribed speech):

- No Spanish meeting corpus with dialogue acts exists publicly
  (`DIALOGUE_ACT_DATA.md` §1); Spanish evidence is synthetic text only.
- Real speech in the training corpora is English (meetings, two thirds
  role-play in AMI); English-only training reaches top-1 0.430 on the
  Spanish set, against 0.731 with Spanish folds (`DIALOGUE_ACT_TAGGER.md`
  §2).
- Regional varieties, non-Peninsular Spanish, code-switching, accents
  and microphones: not covered by any data here and not measured.
- Text, not ASR output: real transcripts lose punctuation and
  back-channels; not measured (`DIALOGUE_ACT_TAGGER.md` §7).

## 3. Collection and annotation process

- **Public corpora**: downloaded from the official pages on 2026-11-15
  (URLs, archive sizes and hashes in `DIALOGUE_ACT_DATA.md` §1); read by
  importers into a common JSONL kept in the git-ignored `data/dialogue_acts/`.
- **Label mapping**: each source label maps to the set of our
  categories it is compatible with; a prediction inside the set counts as
  right. The mappings are ours, made from tag definitions, **not checked
  by a second person** (`DIALOGUE_ACT_DATA.md` §2, §6).
- **Spanish set**: labels are the author's intended labels, written with
  the sentence; the parser (`dialogue_act_set.py`) checks reply links and
  weights. **Not a human annotation**; inter-annotator agreement was not
  measured (`DIALOGUE_ACT_TAGGER.md` §7).
- **Annotation guide** for future human annotators:
  `DIALOGUE_ACTS_GUIDE.md` (Spanish; target kappa 0.60 per category,
  borrowed from other studies, §10) and `ANNOTATION_GUIDE.md` for
  competencies. Annotators, their background and any human labels: none
  yet.

## 4. Preprocessing

Markup removed from SwDA; SwDA tags clustered into 43 classes; AMI
second parts derived from adjacency pairs; each record keeps the 3 previous
segments as context (`DIALOGUE_ACT_DATA.md` §1.1, §5). Our splits are by
meeting or dialogue (SHA-256 of the dialogue id: 15 % test, 10 % dev, rest
train), not the published ones; learned parts use 5-fold cross-validation
by dialogue (`DIALOGUE_ACT_TAGGER.md` §1-§2).

## 5. Uses

Used for: training and evaluating the encoder tagger and the rules;
comparing judges (`DIALOGUE_ACT_TAGGER.md`, `DESIGN.md` §4). Not for:
validating the categories, reporting the analyzer's accuracy to anyone, or
any statement about a real person (`examples/skills/dialogue_acts/README.md`).
Accuracy on the set measures agreement with its author and is an upper
bound for real meetings (`DIALOGUE_ACT_TAGGER.md` §9.6). SwDA and DIHANA must
not train a shipped model; check licences before any commercial use of a
model trained on AMI, ICSI or SwDA (`CONVERSATION.md` §6.1).

## 6. Distribution

The Spanish set and its README are in the repository
(`examples/skills/dialogue_acts/*.txt`, `dialogues.json` generated).
Public corpora are not redistributed; only aggregates are published. Only
aggregates of imported data appear in documents; sentences stay local.

## 7. Maintenance

Mappings and category definitions are revisable and re-importable without
touching the corpora (`DIALOGUE_ACT_DATA.md` §2). No owner other than the
project, no review calendar written down. Re-measure per group when real
sessions with declared groups exist (`FAIRNESS.md` §3.1 T1-T2, §3.3
J1-J4, J6).

## Commands run for the counts in §2

`python3` reading `examples/skills/dialogue_acts/dialogues.json`:
51 dialogues, 1,534 utterances, 20 distinct speaker names, 51 with
`"synthetic": true`, language `es` for all 51, 479 `responds_to` links,
6 utterances with `labels`, 13 acts, 136 speaker entries all with
`enrolled: false`.

## Sources

- `src/domain/skills/dialogue_acts.py`, `dialogue_act_set.py`
- `docs/skills/DIALOGUE_ACT_DATA.md`, `DIALOGUE_ACTS_GUIDE.md`,
  `DIALOGUE_ACT_TAGGER.md`, `ANNOTATION_GUIDE.md`, `CONVERSATION.md`,
  `FAIRNESS.md`
- `examples/skills/dialogue_acts/README.md`, `dialogues.json`
