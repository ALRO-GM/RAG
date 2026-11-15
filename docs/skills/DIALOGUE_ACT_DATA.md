# Dialogue-act data: public corpora and our Spanish set

Status: issue #64 (SK-5), 2026-11-15. Scope: which labelled sentences
exist to develop and evaluate the first-layer dialogue-act classifier,
how their labels map to our 13 categories, and the original Spanish set
written for this project. Owner decisions are on #57.

The 13 categories are the 12 of `CONVERSATION.md` §5, with Información
split into **Información** (a checkable fact or report) and **Opinión**
(an evaluation or stance), as the owner decided on #57. Their ids in
code and data are:

`pregunta, respuesta, informacion, opinion, propuesta, peticion,
aceptacion, rechazo, compromiso, escucha, gestion, relacion, otro`

Only aggregates are published here. The imported sentences stay in the
git-ignored `data/dialogue_acts/`.

---

## 0. Resumen en español

- **Corpus públicos utilizables:** AMI y ICSI-MRDA (reuniones, inglés,
  CC BY 4.0, con enlaces pregunta→respuesta y propuesta→aceptación),
  Switchboard-DA (teléfono, inglés, **no comercial**: solo evaluación
  interna) y DIHANA (español, CC BY 4.0, pero diálogos de información de
  trenes con una máquina: solo pregunta/respuesta y poco más).
- **No hay ningún corpus público de reuniones en español con actos de
  diálogo.** Por eso hemos escrito nuestro propio conjunto: 51 diálogos, 1.534 frases
  (`examples/skills/dialogue_acts/`, ficha en su `README.md`).
- Cada etiqueta de un corpus se traduce al **conjunto de categorías
  nuestras con las que es compatible**. Si el corpus no distingue algo
  (MRDA no separa hecho de opinión), la frase vale para las dos, y esa
  distinción no se puede medir con ese corpus (tabla §3).
- **Varias etiquetas con peso** (decisión 2 de #57, #73): una frase de
  nuestro conjunto puede llevar hasta tres, `compromiso:70 aceptacion:30`
  (enteros, pasos de 10, la principal primero, suman 100); una sola
  etiqueta vale 100, así que los datos de antes siguen valiendo. Los
  corpus públicos no tienen pesos.
- `python -m src skills dialogue_acts_import --corpus ami --source
  <zip> --output data/dialogue_acts/ami.jsonl` importa un corpus a un
  formato JSONL común y escribe solo recuentos en pantalla.

---

## 1. The corpora

Licences were read on the official pages and in the `LICENCE.txt` of
each archive (2026-11-15). Sizes are what the importer read from the
archives below (same date); archive sizes and SHA-256 are of the files
downloaded that day.

| Corpus | Language, setting | Download (official) | Licence | Archive | Read by the importer | Reply links |
|---|---|---|---|---|---|---|
| **AMI** dialogue acts | English (many non-native), design-team meetings, ⅔ role-play | https://groups.inf.ed.ac.uk/ami/download/ → `AMICorpusAnnotations/ami_public_manual_1.6.2.zip` | **CC BY 4.0**, "the AMI corpus and its annotations" (`LICENCE.txt` in the zip; https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) | 22,887,865 bytes, sha256 `b56e5bab…` | 139 meetings, 110,795 segments (24 with no type) | Yes: `dialogueActs/*.adjacency-pairs.xml`, types POS, NEG, UNC, PART, ELA; 24,689 segments linked |
| **ICSI MRDA** | English, real research-group meetings | https://groups.inf.ed.ac.uk/ami/icsi/download/ → `ICSICorpusAnnotations/ICSI_core_NXT.zip` | **CC BY 4.0**, "the ICSI corpus and its annotations" (`ICSI/LICENCE.txt` in the zip; the web page says "some of the annotations") | 19,488,722 bytes, sha256 `cf486024…` | 75 meetings, 111,179 segments | Yes: `adjacency` attribute (`5a`, `5b-2`, `5b+`); 20,901 segments linked |
| **Switchboard-DA** (SwDA) | English, telephone conversations between strangers on a set topic | https://github.com/cgpotts/swda → `swda.zip` | **CC BY-NC-SA 3.0** (http://compprag.christopherpotts.net/swda.html; HF card `swda`). Code GPL-2.0. **Non-commercial: internal evaluation only**, never training a shipped model (owner decision 6 on #57) | 14,449,197 bytes, sha256 `0a08b8dd…` | 1,155 conversations, 223,487 utterances | No (only answer tags imply them) |
| **DIHANA** | **Spanish**, Wizard-of-Oz telephone dialogues about train timetables (person ↔ "machine") | https://zenodo.org/records/5572470 → `dihanaCorpus.tgz` (includes audio) | **CC BY 4.0** (Zenodo record) | 177,396,693 bytes, sha256 `479f6e38…` | 900 dialogues, 23,493 segments | No |

Others checked and **not imported**:

| Corpus | Why not |
|---|---|
| MRDA repackaged on GitHub (NathanDuran/MRDA-Corpus) | GPL-3.0 repackaging that drops the reply links; the ICSI original is better |
| SILICONE, MIAM (Hugging Face) | Repackagings; their cards contradict themselves (metadata CC BY-SA, text BY-NC-SA). MIAM's `dihana` subset is the same data as the Zenodo record |
| DialogBank (ISO 24617-2) | No data licence statement found |
| CALLHOME Spanish dialogue acts | LDC, paid |
| NegotiAct, act4teams, Pugh et al. | Not public (`CONVERSATION.md` §3.1) |

No Spanish **meeting** corpus with dialogue-act labels was found, as
`CONVERSATION.md` §3.1 already reported.

### 1.1 Formats (what the readers parse)

- **AMI** (NXT XML, ISO-8859-1): `words/<meeting>.<spk>.words.xml`
  (`<w nite:id starttime>`), `dialogueActs/<meeting>.<spk>.dialog-act.xml`
  (`<dact>` with a `da-aspect` pointer into `ontologies/da-types.xml` and
  a `nite:child` word range `id(a)..id(b)`), and
  `dialogueActs/<meeting>.adjacency-pairs.xml` (pointers `source` = first
  part, `target` = second part, `type` into `ontologies/ap-types.xml`).
  Segments are ordered by the start time of their first word.
- **ICSI** (NXT XML): `ICSI/Words/…words.xml` (ids may contain commas),
  `ICSI/DialogueActs/…dialogue-acts.xml` (`<dialogueact type="s^aa"
  adjacency="5b" participant=…>`). Our reading of `adjacency` (number =
  pair, `a` = first part, a later letter = second part, `+` =
  continuation, `.` joins several) comes from the data, **not checked
  against the MRDA manual**; a second part is linked to the nearest
  earlier first part with the same number.
- **SwDA**: one CSV per conversation (`act_tag, caller, text,
  transcript_index, …`); Switchboard markup (`{D So, }`, `[ … + … ]`,
  `<laughter>`, `/`) is removed. Raw tags (over 200) are clustered into
  the usual 43 classes as in Potts' reader before mapping.
- **DIHANA**: `.dia` text files (Latin-1), one line per segment:
  `texto (Hablante:Nivel1:Nivel2:Nivel3) pal: i - j`. Only level 1 is
  task-independent; levels 2–3 are timetable slots.

## 2. Mapping to our categories

A source label maps to the **set of our categories it is compatible
with**. One category: the corpus makes our distinction. Several: it
does not, and a prediction inside the set counts as right. None (`–`):
not usable (non-speech, unlabelled, ambiguous). The tables are code
(`src/domain/skills/dialogue_acts.py`); a mapping can be revised and
re-imported without touching the corpus.

### 2.1 AMI (15 types)

| AMI type | Ours |
|---|---|
| `bck` Backchannel | escucha |
| `stl` Stall, `fra` Fragment, `oth` Other | otro |
| `inf` Inform | informacion |
| `ass` Assess | opinion |
| `sug` Suggest | propuesta \| peticion (AMI has no request class) |
| `off` Offer | compromiso \| propuesta (conditional and unconditional offers together) |
| `el.inf`, `el.ass`, `el.und`, `el.sug` (elicitations) | pregunta |
| `und` Comment-about-understanding | escucha |
| `be.pos`, `be.neg` | relacion |

Second parts, from the adjacency pairs:

- `inf`, `ass`, `sug` or `off` replying to an elicitation → **respuesta**.
- `ass` or `und` replying **POS** to `sug`, `off` or `ass` → **aceptacion**.
- `ass`, `und`, `inf` or `oth` replying **NEG** to `sug`, `off`, `ass`
  or `inf` → **rechazo** (disagreement or correction).
- Everything else keeps its type's mapping. In particular a POS reply to
  an **inform** is not an acceptance: AMI links introductions and
  acknowledgements that way (checked on meeting ES2002a), so taking them
  as acceptances would inflate the class.

### 2.2 ICSI MRDA (general tag + specific tags)

The first decisive specific tag wins, in this order; otherwise the
general tag decides.

| MRDA | Ours |
|---|---|
| `aa`, `aap` (accept, partial accept) | aceptacion |
| `ar`, `arp` (reject, partial reject) | rechazo |
| `cc` (commit) | compromiso |
| `co` (command) | peticion |
| `cs` (suggestion) | propuesta |
| `ny`, `nn`, `na`, `ng`, `no`, `nd`, `am` (answers, maybe) | respuesta |
| `ft`, `fa`, `fw`, `by`, `j` (thanks, apology, welcome, sympathy, joke) | relacion |
| `tc` (topic change) | gestion |
| `bu`, `br`, `d`, `g` (understanding check, repeat request, declarative and tag questions) | pregunta |
| `bk`, `ba`, `m`, `r` (acknowledgement, assessment/appreciation, mimic, repeat) | escucha |
| general `s` (statement) | informacion \| opinion |
| general `qy`, `qw`, `qr`, `qrr`, `qo` | pregunta |
| general `qh` (rhetorical question) | – (could be Rechazo or Opinión) |
| general `b` | escucha |
| general `fg` (floor grabber) | gestion |
| general `fh`, `h` (floor holder, hold) | otro |
| `%`, `%-`, `%--` (indecipherable, interrupted, abandoned) | otro |
| `x`, `z` (non-speech, not labelled) | – |

A segment marked as disrupted (`s.%--`, `b.%`) also admits **otro**:
`s.%--` → informacion \| opinion \| otro.

### 2.3 Switchboard-DA (43 clustered tags)

| SwDA | Ours |
|---|---|
| `sd` statement-non-opinion | informacion |
| `sv` statement-opinion | opinion |
| `^q` quotation | informacion \| opinion |
| `b`, `bk`, `ba`, `bh`, `b^m` (backchannels, acknowledgement, appreciation, repeat) | escucha |
| `aa`, `aap_am` (accept, partial accept, maybe) | aceptacion |
| `ar` reject | rechazo |
| `arp_nd` (partial reject, dispreferred answer) | respuesta \| rechazo |
| `qy`, `qw`, `qo`, `qrr`, `qy^d`, `qw^d`, `^g`, `br` | pregunta |
| `ny`, `nn`, `na`, `ng`, `no` (answers) | respuesta |
| `ad` action-directive | peticion |
| `oo_co_cc` (offer, option, commit) | compromiso \| propuesta |
| `bf` summarize/reformulate | gestion (weak: often a check) |
| `fp`, `fc`, `fa`, `ft`, `bd` (opening, closing, apology, thanks, downplayer) | relacion |
| `%`, `h`, `^h`, `t1`, `t3`, `^2` (abandoned, hedge, hold, self- and third-party talk, collaborative completion) | otro |
| `qh` rhetorical, `fo_o_fw_"_by_bc` (other), `x` non-verbal, `+` continuation | – |

### 2.4 DIHANA (level 1, Spanish)

| DIHANA L1 | Ours | Note |
|---|---|---|
| Pregunta | pregunta | The caller's request for a timetable is labelled Pregunta |
| Respuesta | respuesta | |
| Confirmacion | pregunta | System check question ("¿Quiere viajar a Madrid?") |
| Nueva_consulta | pregunta | "¿Desea algo más?" |
| No_entendido | pregunta | Clarification request |
| Afirmacion | respuesta \| aceptacion | "Sí" to a check |
| Negacion | respuesta \| rechazo | "No" to a check |
| Espera | gestion | "Un momento, por favor" |
| Apertura, Cierre | relacion | Greeting, goodbye |
| Indefinida | – | |

## 3. What each corpus can measure

✓ = some source labels map to this category alone; ~ = only shared with
another category (cannot be told apart); ✗ = never.

| Category | AMI | MRDA | SwDA | DIHANA (es) |
|---|---|---|---|---|
| pregunta | ✓ | ✓ | ✓ | ✓ |
| respuesta | ✓ (from pairs) | ✓ | ✓ | ✓ |
| informacion | ✓ | ~ (with opinion) | ✓ | ✗ |
| opinion | ✓ | ~ (with informacion) | ✓ | ✗ |
| propuesta | ~ (with peticion or compromiso) | ✓ | ~ (with compromiso) | ✗ |
| peticion | ~ (with propuesta) | ✓ | ✓ | ✗ |
| aceptacion | ✓ (from pairs) | ✓ | ✓ | ~ (with respuesta) |
| rechazo | ✓ (from pairs) | ✓ | ✓ | ~ (with respuesta) |
| compromiso | ~ (with propuesta) | ✓ | ~ (with propuesta) | ✗ |
| escucha | ✓ | ✓ | ✓ | ✗ |
| gestion | ✗ | ✓ (topic change, floor grabber) | ✓ (weak: reformulations) | ✓ (wait only) |
| relacion | ✓ | ✓ | ✓ | ✓ (greetings only) |
| otro | ✓ | ✓ | ✓ | ✗ |

Reading:

- **MRDA is the only public corpus that measures all of Propuesta,
  Petición and Compromiso separately**, and it cannot split Información
  from Opinión. AMI and SwDA split Información/Opinión but merge the
  other three in pairs. Between them every category except the
  Información/Opinión split *and* the Propuesta/Petición/Compromiso split
  at once is covered, but only in English.
- **Gestión** is poorly covered everywhere: meeting management
  ("vamos al segundo punto", summaries of decisions, time keeping) is
  not a class in any of the four; MRDA's topic changes and floor
  grabbers are a small part of it.
- **Spanish**: DIHANA measures only questions, answers, yes/no replies,
  greetings and "wait": the human–machine timetable setting has no
  proposals, opinions or commitments. For everything else in Spanish
  only our own set (`examples/skills/dialogue_acts/`, datasheet in its
  `README.md`) exists, and it is not human-annotated.
- English accuracy does not transfer to Spanish speech; the corpora are
  for development and a first cross-lingual sanity check, not for
  claiming Spanish performance.

## 4. Imported label distributions (measured)

`python -m src skills dialogue_acts_import --corpus <name> --source
<archive> --output data/dialogue_acts/<name>.jsonl`, run on 2026-11-15
on the archives of §1. Keys joined with `|` are compatible sets; `–` is
unmapped.

| Ours | AMI | MRDA | SwDA | DIHANA |
|---|---:|---:|---:|---:|
| pregunta | 7,384 | 7,001 | 9,306 | 13,098 |
| respuesta | 6,044 | 3,363 | 5,858 | 4,285 |
| informacion | 29,857 | — | 75,403 | — |
| opinion | 11,847 | — | 26,527 | — |
| informacion\|opinion | — | 36,346 | 986 | — |
| propuesta | — | 2,383 | — | — |
| peticion | — | 637 | 752 | — |
| propuesta\|peticion | 9,030 | — | — | — |
| compromiso | — | 354 | — | — |
| compromiso\|propuesta | 1,442 | — | 110 | — |
| aceptacion | 6,701 | 6,044 | 11,262 | — |
| rechazo | 1,406 | 1,013 | 345 | — |
| respuesta\|aceptacion | — | — | — | 990 |
| respuesta\|rechazo | — | — | 207 | 340 |
| escucha | 14,383 | 24,052 | 46,233 | — |
| gestion | — | 3,492 | 961 | 1,948 |
| relacion | 2,307 | 950 | 3,071 | 2,727 |
| otro | 20,368 | 12,838 | 18,415 | — |
| … \|otro (disrupted, MRDA) | — | 9,937 | — | — |
| – (unmapped) | 26 | 2,769 | 24,051 | 105 |
| **total** | **110,795** | **111,179** | **223,487** | **23,493** |

MRDA's disrupted segments are split as: informacion\|opinion\|otro
7,317, pregunta\|otro 1,117, escucha\|otro 651, propuesta\|otro 344,
respuesta\|otro 228, aceptacion\|otro 73, gestion\|otro 51,
relacion\|otro 48, rechazo\|otro 45, peticion\|otro 44,
compromiso\|otro 19. The unmapped are mostly `z` (2,422) and `qh` in
MRDA, and `+` continuations (17,810 incl. `+@`), `x` and `o` in SwDA.

Rare classes match the literature: Compromiso is 0.3% of MRDA (354),
Rechazo 0.9–1.3% of the meeting corpora. A balanced evaluation has to
sample per class and report per-class F1, not accuracy.

## 5. The common format

One JSON object per line (`ActRecord`):

| Field | Meaning |
|---|---|
| `corpus`, `licence`, `non_commercial`, `language` | Provenance; `non_commercial: true` for SwDA |
| `dialogue`, `id`, `order` | Meeting/conversation, the corpus's segment id, position |
| `speaker`, `text` | Who and what (markup removed) |
| `context` | The previous 3 segments (`--context N`), speaker and text |
| `source_label` | The corpus's own tag, unchanged (our set: the written label, `compromiso` or `compromiso:70 aceptacion:30`) |
| `categories` | Our compatible categories (empty: unusable); for our set, the primary label |
| `responds_to` | Segment id of the first part (AMI, MRDA, our set) |
| `session_type` | Declared conversation type (our set only) |
| `labels` | Weighted labels, `[{"label", "weight", "doubtful"}]`, primary first, weights in whole % summing to 100 (our set only; empty for public corpora) |

**Weighted labels** (owner decision 2 on #57, issue #73). A sentence may
carry 1 to 3 labels with their share of its function in whole %. Human
annotation: steps of 10, primary first, summing to 100
(`DIALOGUE_ACTS_GUIDE.md` §7); in the set's text format they are written
`etiqueta:peso`, and a bare label means 100, so every single-label file
and every JSONL written before the field existed still loads (a missing
`labels` reads as empty). A tagger fills the same shape with its top-3
calibrated probabilities, whatever is left to 100 kept as an explicit
rest rather than renormalised, and `doubtful: true` on a label below its
category's threshold (`CONVERSATION.md` §5.3.1 c). Public corpora keep
`labels` empty: they give one tag per segment, and what it is compatible
with is already in `categories`. The agreement and accuracy measures for
weighted labels are in `CONVERSATION.md` §6.3.1.

Commands:

```sh
python -m src skills dialogue_acts_import --corpus ami \
    --source data/dialogue_acts/raw/ami/ami_public_manual_1.6.2.zip \
    --output data/dialogue_acts/ami.jsonl        # also: mrda, swda, dihana
python -m src skills dialogue_acts_import --corpus own \
    --source examples/skills/dialogue_acts --output data/dialogue_acts/own.jsonl
python -m src skills dialogue_acts_set --source examples/skills/dialogue_acts \
    --output data/dialogue_acts/own.json         # check + transcript.v1-like JSON
```

## 6. Limits

- The mappings are ours, made from the tag definitions, and have not
  been checked by a second person. The AMI second-part rules (§2.1) are
  a judgement call; MRDA's adjacency encoding is our reading of the data.
- Text-only: prosody that the original annotators heard (declarative
  questions, irony) is not in the text we classify (`CONVERSATION.md`
  §5.4).
- AMI is two-thirds role-play by people who were not real teams; SwDA is
  strangers talking on assigned topics; DIHANA talks to a machine. None
  is a Spanish workplace meeting.
- SwDA is non-commercial: internal evaluation only.
