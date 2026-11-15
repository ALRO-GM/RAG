# The parts of a conversation and how to divide it

Status: research study for issue #57, compiled 2026-11-15. Scope: before
the analyzer extracts questions, proposals, answers and agreements, decide
on evidence **which unit to classify, which categories to classify it
into, and which context labels (conversation type, phase) to attach**.
Out of scope: implementation; follow-up tickets come from §5 and §6.

Conventions:

- Each figure carries a reference `[n]` (list at the end). A reference is
  tagged **[verified]** when the page or PDF was opened and the figure was
  seen there, **[unverified]** when it comes from a search snippet, a
  secondary source or memory. Several sources were read from the
  project's saved copies in `data/research/softskills/papers/` (noted
  "local copy").
- Agreement figures are Cohen's κ unless stated. A κ from another scheme,
  language or unit is quoted as an **analogue**, never as the agreement we
  will obtain.
- Spanish examples in §5 were written for this document; they are not
  corpus data.

---

## 0. Resumen en español

**Qué se pregunta.** Antes de extraer preguntas, propuestas, respuestas y
acuerdos, hay que decidir qué trozo de conversación se clasifica, en qué
grupos y con qué etiquetas de contexto.

**Qué dice la evidencia, en corto.**

1. **La unidad.** Los lingüistas no clasifican turnos enteros (un turno
   mezcla varias cosas) sino trozos con una sola función ("segmento
   funcional" en la norma ISO 24617-2 [1]; "unidad de pensamiento" en
   NegotiAct [12]). Nuestras frases de `transcript.v1` son una buena
   aproximación: se clasifica **cada frase**, mirando las frases de antes.
2. **La conversación tiene estructura en pares.** Una pregunta pide una
   respuesta; una propuesta pide aceptación o rechazo; una petición pide
   que se cumpla (pares adyacentes [21][22]). Un **acuerdo** o una
   **decisión** no está en una frase: es una propuesta seguida de
   aceptaciones sin rechazo. Por eso la propuesta separa dos capas:
   etiquetas por frase (capa 1) y objetos que se deducen de secuencias
   (capa 2: decisión, tarea asignada, pregunta sin contestar).
3. **Hay esquemas de etiquetas probados**, con corpus públicos en inglés
   (AMI y ICSI, licencia CC BY 4.0) y uno en español (DIHANA, CC BY 4.0,
   pero son llamadas para pedir información de trenes, no reuniones). No
   existe ningún corpus de **reuniones en español** etiquetado: habrá que
   anotar uno propio.
4. **La fiabilidad humana depende mucho de la categoría.** Preguntas y
   respuestas se anotan bien (κ ≈ .75–.80 en esquemas parecidos);
   acuerdo/desacuerdo regular (κ ≈ .42–.71); compromisos y tareas
   asignadas mal si no se definen con cuidado (κ entre .15 y .78 según el
   esquema). Lo que las personas no consiguen anotar con fiabilidad, la
   máquina tampoco lo hará.
5. **Los modelos de lenguaje sin entrenar fallan con etiquetas finas**
   (31–40 % de acierto con ~43 etiquetas en inglés [9]); con 5 etiquetas
   gruesas los modelos entrenados pasan del 90 % [6][9]. Conclusión: pocas
   categorías, bien definidas.

**Grupos recomendados (capa 1, una etiqueta principal por frase). Son 13 tras la decisión 1 del dueño (2026-11-15): Opinión separada de Información, y Propuesta y Petición por separado:**

| # | Categoría | En una frase |
|---|---|---|
| 1 | Pregunta | Pide información u opinión |
| 2 | Respuesta | Contesta a una pregunta anterior |
| 3 | Información | Aporta hechos, datos o explicaciones comprobables |
| 4 | Opinión | Expresa una valoración, juicio o postura sobre el tema |
| 5 | Propuesta | Plantea una idea, solución o forma de actuar para que el grupo la considere |
| 6 | Petición | Pide a alguien concreto que haga algo |
| 7 | Aceptación | Acepta o apoya una propuesta, petición u opinión anterior |
| 8 | Rechazo | Rechaza, discrepa o corrige |
| 9 | Compromiso | Quien habla se obliga a hacer algo |
| 10 | Señal de escucha | "Vale", "ajá", "sí, sí": sigue la conversación sin aportar contenido |
| 11 | Gestión de la reunión | Orden del día, turnos, tiempo, resúmenes, cambio de tema |
| 12 | Relación | Saludos, agradecimientos, disculpas, elogios o críticas personales, humor |
| 13 | Otro | Frases cortadas, ininteligibles o sin función clara |

Etiquetas de contexto: **tipo de conversación** (lo indica quien sube la
grabación; no se adivina) y **fase** (apertura, exploración, discusión,
decisión, cierre), deducida de las etiquetas y marcada como estimación.

**Límites.** El texto pierde la entonación (una pregunta sin "¿" puede
leerse como afirmación), muchas señales de escucha que el reconocedor de
voz no transcribe, los solapamientos y la ironía. La ley europea de IA
prohíbe deducir emociones de la voz en el trabajo: estas categorías
describen **qué hace** una frase, nunca **cómo se siente** quien habla.

**Primer paso recomendado.** Escribir la guía de anotación de estas 13
categorías (hecha: [`DIALOGUE_ACTS_GUIDE.md`](DIALOGUE_ACTS_GUIDE.md)), anotar entre dos personas unas 1.500 frases de reuniones en
español (con consentimiento o simuladas), y medir con ellas tres opciones
baratas: reglas, un modelo de lenguaje local con ejemplos, y un
clasificador pequeño entrenado con AMI/ICSI traducidos. Coste de GPU:
horas, no días.

**Decisiones pendientes (Alberto):** ver §6.4. La 1 está tomada; de la 2 a la 6 siguen abiertas.

---

## 1. Units

### 1.1 Candidate units

| Unit | Definition | Delimited by | Used by |
|---|---|---|---|
| Turn | Stretch of talk by one speaker until another speaker takes the floor | Speaker change; in CA, at a transition-relevance place [21] | CA; act4teams and NegotiAct start a new unit at every speaker change [12] |
| Utterance / slash-unit | A sentence-like stretch, usually one clause; SWBD-DAMSL and MRDA segment talk into these before tagging | Syntax and prosody | SWBD-DAMSL [5], MRDA [6] |
| Functional segment | "The minimal stretches of communicative behaviour that have a communicative function"; usually shorter than a turn; may be discontinuous, overlap, span speakers and carry several functions | Function, not form | ISO 24617-2 [1] [verified] |
| Thought / sense unit | "The smallest meaningful segment of behavior that can stand alone"; new unit at speaker change, at a new statement within the turn ("first making an offer, then substantiating it"), or at a second complete thought with the same code | Meaning | NegotiAct [12] [verified, local copy], act4teams, Bales [16] |
| Act | Bales: a discrete act, often 2–3 per sentence | Meaning | Bales IPA [16] [verified via encyclopedia entry] |
| Intervención / acto / subacto | Val.Es.Co: dialogue › exchange › intervention › act › subact (structural) plus turn (social); "every turn is an intervention, but not every intervention is a turn" | Pragmatic function and uptake | Spanish colloquial conversation [28] [verified] |
| Dialogue act | Not a unit but the label: the communicative function attached to one of the units above | — | All DA schemes |

### 1.2 How the units are delimited in speech transcripts

- Human segmentation of meeting speech into DAs is itself imperfect:
  MRDA reports 66.5% strict agreement on segment boundaries [6] [verified].
- NegotiAct avoided measuring segmentation agreement altogether: one coder
  segmented all 18 videos (5,365 units), and κ was computed only on the
  codes assigned to those fixed units [12] [verified, local copy]. act4teams
  follows the same practice [13].
- AMI asks that "each segment should contain a single speaker intention"
  [7] [verified].

### 1.3 Which unit to classify here

`transcript.v1` already delivers **utterances = sentences**, with id
`t<turn>.s<index>`, speaker, word-exact times, `previous_speaker` and
`previous_utterance` (`contracts/transcript.v1.schema.json`;
`docs/skills/AUDIO.md` §3). The sentence boundary comes from the ASR
punctuation plus `split_sentences`.

Recommendation:

1. **Classify each `transcript.v1` utterance** (sentence). It is the
   closest available approximation to a functional segment, and it is what
   tags, evidence and the player already use.
2. **Allow a second label** on the same sentence when it does two things
   ("Vale, lo hago yo" = acceptance + commitment). ISO 24617-2 allows
   multiple functions per segment [1]; a hard single label would force
   arbitrary choices on exactly the cases the product cares about.
3. **Always classify in context**: the sentence plus at least the previous
   three sentences with speakers. Pugh et al. found that adding adjacent
   utterances improves CPS classification [17] [verified, local copy];
   answers, acceptances and rejections are defined only relative to what
   came before.
4. **Do not re-segment at first.** Measure in the gold set (§6.3) how often
   annotators want to split a sentence. If above roughly 10% of sentences,
   add a sub-sentence split (at discourse markers such as "pero", "y
   además") as a separate ticket.

Known mismatches: a turn with no punctuation from ASR becomes one long
"sentence"; very short backchannels may be absent because the recogniser
drops them (§5.4).

---

## 2. Structure of conversation (conversation analysis)

| Concept | Key claim | Source | Consequence for us |
|---|---|---|---|
| Turn-taking | Turns are built from turn-constructional units; at each transition-relevance place the next speaker is selected by the current one or self-selects; the system minimises gap and overlap | Sacks, Schegloff & Jefferson 1974 [21] [unverified: page blocked, standard reference] | Turn = speaker change is a valid but coarse unit; interruptions are violations of this order and need overlap detection |
| Response timing | Across 10 languages the mean response offset after a polar question is +208 ms, mode 0 ms; language means range from +7 ms (Japanese) to +469 ms (Danish). Spanish is not in the sample | Stivers et al. 2009 [26] [verified] | Gaps are short and comparable across languages; a long delay before a reply is informative (often a dispreferred answer), and our word times can measure it |
| Adjacency pair | Two turns by different speakers, first pair part (question, offer, request, proposal, greeting) makes a type-fitted second part (answer, acceptance/decline, compliance, return greeting) relevant | Schegloff 2007 [22] [unverified, bibliographic] | Core of the category set: first-pair-part categories (question, proposal, request) and second-pair-part categories (answer, acceptance, rejection) |
| Sequence expansion | A base pair can be expanded before (pre-sequence: "¿Tienes un minuto?"), inside (insert sequence: a clarifying question before the answer) and after (post-expansion: "vale", "ah") | Schegloff 2007 [22] [unverified] | An answer may come several sentences after its question; linking needs a window, not just the previous sentence |
| Preference organisation | Second parts are not equal: agreement and acceptance are usually "preferred" (immediate, short); disagreement and rejection are "dispreferred" (delayed, hedged, with accounts: "Sí, pero…"). After self-deprecation disagreement is preferred | Pomerantz 1984 [23] [unverified] | "Sí, pero…" is mostly a **rejection**, not an acceptance; this is the hardest boundary for text classifiers and annotators alike |
| Repair | Trouble in speaking, hearing or understanding is repaired; self-initiated self-repair is preferred over other-correction | Schegloff, Jefferson & Sacks 1977 [24] [unverified] | Self-corrections and restarts are "Otro"/own-communication management, not content; other-initiated repair ("¿cómo?") is a clarification question |
| Openings and closings | Closings are organised: pre-closing ("bueno…", "vale, pues…"), then terminal exchange | Schegloff & Sacks 1973 [25] [unverified] | Phase signals for opening and closing |
| Change-of-state token | "Oh" marks a change in the speaker's knowledge | Heritage 1984 [27] [unverified] | Spanish "ah", "anda": acknowledgement with uptake, not a backchannel only |
| Continuers | "Uh huh" passes up the opportunity of a full turn; it says "go on" | Schegloff 1982 [27b] [unverified] | Backchannels are a function of their own, not agreement |
| Acknowledgement tokens and nods | Vocal continuers align with the telling; nods mid-telling also affiliate with the teller's stance | Stivers 2008 [27c] [unverified] | Part of listening is visual and absent from any transcript |
| Spanish affirmative operators | "sí", "ya", "claro", "vale", "exacto" often take the turn rather than act as pure backchannels | [29] [unverified] | In Spanish "vale"/"claro" are ambiguous between backchannel, acceptance and agreement; the boundary needs explicit rules (§5.2) |
| Topic shift | Topics change by stepwise drift or by marked shifts ("otra cosa…", "pasando a…") | Galley et al. 2003; Hsueh & Moore 2007 (empirical topic segmentation) [19][20] | Marked shifts are a "Gestión" act and a phase/agenda signal |

**From pairs to agreements.** None of these sources treats "agreement" or
"decision" as a property of one utterance. A decision is a sequence: a
proposal, uptake by others (acceptances), the absence of standing
rejections, and often a summary or formulation ("Entonces quedamos en…").
An action item likewise has parts: Purver et al. annotate four classes
(task description, timeframe, ownership, agreement), and only 1.4% of
utterances carry any of them [18] [verified]. This motivates the two-layer
design of §5.

---

## 3. Dialogue-act and interaction coding schemes

### 3.1 Comparison

| Scheme | Structure | Tags | Unit | Reliability (as reported) | Corpus | Licence / access | Languages |
|---|---|---|---|---|---|---|---|
| **ISO 24617-2** (from DIT++) [1][2] | 9 dimensions (Task, Auto-Feedback, Allo-Feedback, Turn Mgmt, Time Mgmt, Discourse Structuring, Own Comm. Mgmt, Partner Comm. Mgmt, Social Obligations Mgmt); 2nd edition (2020) adds Contact Mgmt and plug-ins [verified] | 56 functions (DIT++ has 88); about 26 general-purpose ones in four families: information-seeking (Question, Set-, Propositional-, Choice-, Check-Question), information-providing (Inform, Agreement, Disagreement, Correction, Answer, Confirm, Disconfirm), commissives (Offer, Promise, Address/Accept/Decline Request, Address/Accept/Decline Suggest), directives (Request, Instruct, Suggest, Address/Accept/Decline Offer). The 26/30 split is our count from Table 6.1 | Functional segment | Function κ per dimension (Dutch DIAMOND data; plain/taxonomically weighted): Task .47/.71, Auto-FB .21/.57, Allo-FB .42/.58, Turn .82, Time .58, Discourse Structuring .74, Own Comm. and Social Obligations 1.00. Recognising which dimension applies: .83–.99 [verified, in [1]] | DialogBank (English: Map Task, Switchboard, TRAINS, DBOX; Dutch: OVIS, DIAMOND, Map Task, Schiphol); DBOX: 338 dialogues, 16 h, κ .55–1.00 per dimension [verified] | Standard is paid (ISO); DialogBank online, licence not stated [verified] | Mainly English, Dutch |
| **DAMSL** [3] | 3 layers: forward-looking, backward-looking, utterance features; several labels per layer | ~ a few dozen | Utterance | 8 TRAINS dialogues, 604 utterances: Statement .66, Influence-on-Addressee-Future-Action .70, Committing-Speaker-Future-Action **.15**, Answer .76, Agreement **.42**, Understanding .57, Response-to .77. Authors: "on average 0.15 lower than the accepted kappa" [verified] | TRAINS | LDC [unverified] | English |
| **SWBD-DAMSL** [4][5] | Flat after clustering | ~50 basic tags + diacritics, ~220 combinations used, clustered into 42 (Potts' SwDA lists 43) | Slash-unit | 84% agreement, κ = .80 on the 42-tag set [verified, in [5]] | Switchboard DA: 1,155 telephone conversations, 205k utterances, 1.4M words [verified] | SwDA CC BY-NC-SA 3.0 [unverified: HF card snippet]; HF `cgpotts/swda`, SILICONE `swda` | English |
| **MRDA** (ICSI meetings) [6] | 1 obligatory general tag + 0..n specific tags; adjacency pairs annotated | 11 general + 39 specific; derived sets of 5 / 12 / 52 labels in common preprocessing | DA segment | κ .80 overall on the 6-way map (pairs .75–.82); segmentation 66.5% strict [verified] | 75 meetings, ~72 h, 180,218 tags; statement 68.0%, backchannel 13.4%, floor holder 7.4%, yes/no question 4.9% [verified] | ICSI corpus "and its annotations" CC BY 4.0, free download [verified] (the ICSI page says "some of the annotations" [verified, local copy]: check the DA files specifically); SILICONE `mrda` (5 labels) | English |
| **AMI** dialogue acts [7][8] | Flat + adjacency relations + addressee | 15: Backchannel, Stall, Fragment, Inform, Elicit-Inform, Suggest, Offer, Elicit-Offer-or-Suggestion, Assess, Comment-about-Understanding, Elicit-Assessment, Elicit-Comment-about-Understanding, Be-Positive, Be-Negative, Other [verified] | "Single speaker intention" | **"Reliability test results are not currently available"** [verified] | 100 h of meetings, ~2/3 scenario (design team role-play) [verified]; also topic segmentation, extractive/abstractive summaries with DECISIONS and ACTIONS headings [verified] | CC BY 4.0 [verified] | English (many non-native speakers) |
| **Bales IPA** [16] | 4 areas × 3 (positive socio-emotional, attempted answers, questions, negative socio-emotional), polar pairs | 12 | Act (2–3 per sentence) | No figure found; a training manual "has never been widely available" [verified, encyclopedia] | none public | Book under copyright | English |
| **act4teams** [13][14] | 4 facets: problem-focused, procedural, socio-emotional, action-oriented [unverified]; solution talk split into defining the objective, solution, describing, arguing for, problem with a solution [verified, local copy] | Code count not confirmed (often quoted as 43–44: **unverified**) | Sense unit | Fleiss κ = .81 on 6 videos / 2,795 units, after **200 h** of coder training [verified, local copy of [14]]; 92 meetings in [13] | Not public | Trademark; manual via the authors / training [unverified] | German (coded from video) |
| **NegotiAct** [12] | 7 categories: information (providing/asking priority, preference, positional info…), offers (single-/multi-issue, accepting, rejecting…), persuasive communication, socio-emotional statements, unethical behaviours, process-related communication (procedural suggestion, time management…), residual | 47 codes | Thought unit | κ = .80 on 18 videotaped negotiations (5,365 units); per-code agreement from 66.67% [verified, local copy] | Not public | Manual in supplementary material | German/English (coded from video) |
| **Sun et al. CPS** [15] | 3 facets (constructing shared knowledge, negotiation/coordination, maintaining team function), 2 sub-facets each | 17 indicators (from 32 pilot) | 4-min video segments (study 1); thin slices (study 2) | Coders trained to ICC .80; Spearman .92 in study 2 [verified] | Not public | — | English |
| **Pugh et al. 2021** (CPS, automated) [17] | 7 CPS skills | 7 | Turn/utterance | Raters' median ICC .93; BERT with context: micro-AUROC .91 on human transcripts vs **.80 on ASR** [verified, local copy] | 8,660 utterances, 44 teams | Not public | English |
| **IBIS / TAS** [30][31] | Issue–position–argument; TAS nodes: open issue, A/B issue, yes/no issue, statement, weak statement; relations support, object, clarification | ~5 nodes + relations | Argument unit; backchannels filtered out | **No reliability reported** ("will be investigated") [verified] | AMI subset, "speculative, not generally released" [verified] | — | English |
| **Action items** (Purver et al.) [18] | Hierarchical: task description, timeframe, ownership, agreement | 4 | Utterance, grouped into subdialogues | Flat scheme (Gruenstein et al. 2005) κ as low as .36; hierarchical scheme κ .64–.78 pairwise on 18 ICSI meetings [verified] | ICSI subset | — | English |
| **Decisions** (Hsueh & Moore) [32] | Decision-related DAs in AMI | binary | DA | Classifier F ≈ .35 [unverified, secondary]; annotation κ not found | AMI | CC BY 4.0 (AMI) | English |
| **DIHANA** [10][11] | 3 levels; level 1 task-independent | L1: 11 (Pregunta, Respuesta, Confirmación, Nueva Consulta, Espera, Cierre, Afirmación, Apertura, No Entendido, Negación, Indefinida); L2/L3 task-specific [verified] | DA segment | Not found | 900 Wizard-of-Oz telephone dialogues (railway information), 225 speakers, 23,547 DA segments, ~5.5 h [verified] | **CC BY 4.0**, Zenodo download [verified]; also in MIAM (`dihana`, splits 19,063/2,123/2,361; licence field inconsistent: BY-SA vs BY-NC-SA) | **Spanish** |
| **CALLHOME Spanish DA** [33] | DA (from SWBD-DAMSL), dialogue games, genre | — | Utterance | Not found | 120 telephone conversations | LDC, paid [verified] | Spanish |
| **Val.Es.Co** [28] | Hierarchical units, not a DA tag set | — | Act/subact | — | Corpus Val.Es.Co 3.0 (colloquial Spanish) | Not found | Spanish |

No Spanish **meeting** corpus with dialogue-act labels was found (searched:
MIAM, SILICONE, LDC, Val.Es.Co, DialogBank, HF). The Polish DiaBiz.Kom and
Dutch DialogBank corpora show that ISO annotation outside English exists,
but nothing comparable was found for Spanish.

### 3.2 Automatic classification: what has been reached

| Setting | Result | Source |
|---|---|---|
| SwDA 42 classes, words only, HMM (2000) | 71.0% (chance 35%; 65% on ASR output) | Stolcke et al. [5] [verified] |
| SwDA, supervised neural | 82.9–83.2% (human agreement 84%) | Raheja & Tetreault 2019; He et al. 2021 [34][35] [verified] |
| MRDA 5 classes, supervised neural | 91.1–91.6% | same [verified] |
| Fine-grained SwDA (43) / MRDA (~50), LLM few-shot, text | GPT-3.5 31.6% / 31.5%; LLaMA-3.1-70B 40.0% / 31.3%; GPT-4 (MRDA) 31.4%, macro-F1 24.7. Audio input (GPT-4 Audio, MRDA) 41.8% | Qamar, Tong & Huang, ACL 2025 [9] [verified]; the paper's caption and header disagree on zero- vs few-shot |
| Same, fine-tuned turn model | MRDA 67.0% / macro-F1 38.8; SwDA 81.0% / 58.0 | same |
| LLM DA on Spanish | No published result found | — |
| CPS codes on ASR vs human transcripts | AUROC .80 vs .91 | Pugh et al. [17] [verified] |

Reading: supervised models reach human-level accuracy on **coarse** sets in
English; LLMs prompted with **fine** label sets fail. Our set is coarse
(12), so the LLM gap may be smaller, but this was not measured for Spanish
and must be.

---

## 4. Conversation types and their internal phases

### 4.1 What the phase literature supports

- A phase is "a coherent period of interaction, characterized by a
  dominant constellation of communicative acts" (Holmes, on negotiation
  [36] [verified]). That definition makes phases **derivable from the
  act labels** of §5, which is how we propose to obtain them.
- Unitary phase models are contested. Bales & Strodtbeck's orientation →
  evaluation → control [37] [unverified] was argued to appear only in
  newly formed groups [unverified]; Poole's multiple-sequence model finds
  that groups follow different sequences, with task, relation and topic
  tracks and breakpoints [38] [verified, secondary page].
- Human agreement on where segments of a meeting begin is moderate: AMI
  topic segmentation κ .66 (top level) and .59 (sub-topics); ICSI .41/.45
  (as cited) [20] [verified]. Phase boundaries, which are less concrete than
  topics, should be expected to agree no better.
- **No empirical study validating "opening → discussion → decision →
  action items → closing" for workplace meetings was found.** Meeting
  science (Allen, Lehmann-Willenbrock & Rogelberg 2015 [39] [unverified])
  works with procedural behaviours and their sequences rather than fixed
  phases.

### 4.2 Types, phases and their signals

Confidence column: **E** = empirical model with coded data; **P** =
practitioner or normative model; **—** = no model found.

| Type | Phases (source) | Conf. | Signals in our categories and words |
|---|---|---|---|
| Decision / team meeting | Opening & agenda; information exchange; discussion of options; decision; action items; closing. Composite, no single validated source [39] | P | Opening: Relación (greetings) + Gestión ("el orden del día", "lo primero"); discussion: Información, Propuesta, Rechazo; decision: Gestión-summary ("entonces quedamos en"), Aceptación by several; actions: Petición, Compromiso with names and dates; closing: pre-closing "bueno, pues…", thanks |
| Problem solving / group decision | Orientation → evaluation → control (Bales & Strodtbeck [37]); orientation, conflict, emergence, reinforcement (Fisher 1970 [40]); multiple sequences (Poole [38]) | E, contested | Question/Información density early; Rechazo peaks in conflict; Aceptación rises in emergence |
| Brainstorming | Divergence (generate, defer judgment) then convergence (evaluate, select) (Osborn [41]) | P | Many Propuesta, few Rechazo in divergence; Rechazo/Aceptación and criteria ("lo más barato") in convergence |
| Negotiation | Relational positioning, identifying the problem, generating solutions, reaching agreement (Adair & Brett 2005 [42], coded data) | E | Information categories (NegotiAct priorities) early; offers (Propuesta) and Rechazo/Aceptación in bargaining; Compromiso and summary at close |
| Mediation / conflict | Moore's 12 stages, from opening session and information exchange through interest discovery, generating and evaluating options, final bargaining, formalisation [43] [verified] | P | As negotiation, with a third party doing most Gestión |
| Employment interview | Structured interview research defines components of structure, not phases (Campion et al. 1997 [44]) | — | Interviewer Pregunta / candidate Respuesta; opening and candidate-questions segment are common practice, not a validated model |
| Feedback / appraisal | No phase model with empirical support found | — | Información about behaviour, Pregunta for the employee's view, Compromiso at the end |
| Coaching | GROW (goal, reality, options, will), practitioner-derived; real coaching talk "darts about" [45] [unverified] | P | Pregunta-heavy coach; Compromiso at "will" |
| Medical consultation (reference only) | Byrne & Long's six phases [46] [verified]; Calgary-Cambridge five stages [47] [unverified] | E/P | Shows that phase models exist where the activity is institutionally fixed |
| Sales | SPIN (situation, problem, implication, need-payoff questions), proprietary data, no peer-reviewed source [48] [unverified] | P | Question subtypes; not usable as evidence |
| Casual conversation | No phases beyond opening and closing [25] | — | Openings/closings only |

Tuckman's forming–storming–norming–performing [49] [unverified] describes
group development over weeks, not phases within one conversation; it is
context only and should not be labelled from one recording.

### 4.3 Consequences

1. **Conversation type should be declared, not inferred.** The type is
   known to whoever uploads the recording (it is a meeting, a negotiation
   role-play, an interview). A model guessing it adds error and no value.
   Stored as session metadata (`transcript.v1` has `session.title`; a
   `session.type` field would be an optional addition to the contract).
2. **Phase is a derived, low-confidence label.** Compute it from the
   proportions of §5 categories in a sliding window plus marked
   transitions (Gestión acts, pre-closings), per type. Show it as an aid
   to navigation, never use it in a rating, until the gold set shows
   annotators agree on it.
3. **Phase matters for the rubric through opportunity.** THEORY.md already
   requires opportunity gating (no disagreement episode → no evidence on
   conflict handling). A Rechazo by A opens a CONFL opportunity for B; a
   Propuesta with no response is a decision-quality signal. These come from
   the act labels, not from the phase label.

---

## 5. Proposed category set for the product

### 5.1 Design

- **Layer 1 (per sentence):** one primary category from the 13 below, plus
  an optional secondary one when a sentence clearly does two things. Each
  label points to the earlier sentence it responds to (`responds_to`)
  when it is a second-part category (Respuesta, Aceptación, Rechazo) —
  the MRDA and AMI adjacency-pair annotation does the same [6][7].
- **Layer 2 (derived from sequences, rules first):**
  - *Decision candidate*: a Propuesta with ≥ 1 Aceptación by another
    speaker and no standing Rechazo, or a Gestión summary that states it.
  - *Action item*: Petición or Compromiso with an owner (speaker or a
    named addressee), plus Aceptación when it was a request; timeframe if
    stated (the four parts of [18]).
  - *Unanswered question*: Pregunta with no Respuesta in the next N
    sentences (rubric COMM.N4).
  - *Ignored proposal*: Propuesta with no second part.
- **Context labels:** conversation type (declared) and phase (derived).
- The names describe **acts**, not attitudes or emotions (§5.5).

### 5.2 The categories

Agreement column: **analogue** κ from schemes with a similar category
(not our expected value); "our target" is what the gold set (§6.3) must
show before the category is used for evidence.

#### 1. Pregunta (question)

- **Definition.** Asks for information, an opinion, a confirmation or a
  choice. Includes clarification requests ("¿cómo?", "¿a qué te
  refieres?"). Excludes requests for action ("¿me pasas el informe?" →
  Petición) and rhetorical questions that assert ("¿Y eso para qué sirve?"
  said dismissively → Rechazo).
- **Examples.** "¿Cuánto nos queda de presupuesto?" · "¿Tú cómo lo ves,
  Irene?" · "O sea, ¿lo entregamos el viernes?" (check question).
- **Boundary.** Declarative question without "¿" from ASR ("Entonces lo
  entregamos el viernes") → Pregunta only if the next sentence answers it;
  text cannot see intonation. Indirect question ("Me pregunto si llegamos")
  → Pregunta.
- **ISO 24617-2.** Question, Set-, Propositional-, Choice-, Check-Question
  (Task dimension); clarification requests often Auto/Allo-Feedback
  elicitation.
- **Rubric.** LISTEN.P2, LISTEN.P5, LISTEN.N3; SOLVE.P2, SOLVE.P3 (asks
  for alternatives); INFLU.P6; INCL.P1, INCL.P3, INCL.P4, INCL.N2;
  COMM.P5; FEEDB.P3, FEEDB.P5; ADAPT.P3; COORD.P1 (asks for the goal).
  The existing `count_questions` counts "?" only; it would become a count
  of this category.
- **Analogue agreement.** High: MRDA coarse κ .80 [6]; question tags are
  among the most reliable in SWBD-DAMSL (overall .80 [5]).
- **Evaluation corpora.** MRDA, AMI (Elicit-*), SwDA (English); DIHANA
  L1 "Pregunta" (Spanish, task domain).

#### 2. Respuesta (answer)

- **Definition.** Gives the information or opinion asked for by an
  earlier Pregunta (linked with `responds_to`). Includes "no lo sé".
- **Examples.** "Unos tres mil euros." · "Sí, el viernes." · "No lo sé,
  tendría que mirarlo."
- **Boundary.** An answer that also accepts/rejects a proposal ("Sí,
  hagámoslo") → Aceptación; Respuesta is for answers to questions. A
  reply that evades → Información with no link (and COMM.N4 for the
  unanswered question).
- **ISO.** Answer, Confirm, Disconfirm.
- **Rubric.** COMM.N4 (absence), COLLAB.N1 (deflects when asked),
  INIT.N2 (vague non-commitment when asked), ADAPT.P1 ("no lo sé").
- **Analogue agreement.** DAMSL Answer κ .76 [3].
- **Corpora.** MRDA, SwDA, AMI (via adjacency pairs); DIHANA "Respuesta".

#### 3. Información (inform)

- **Definition.** Reports a fact, an event, a figure, an explanation or a
  reason that could in principle be checked, not in reply to a question
  and not proposing action. The test: *could someone look it up and say
  it is true or false?*
- **Examples.** "El cliente llamó ayer quejándose del plazo." · "El
  informe tiene cuarenta páginas." · "El proveedor dijo que entrega el
  día quince."
- **Boundary.** Fact vs opinion is **separated in layer 1** (owner
  decision 1): checkable fact or report → Información; evaluation or
  stance → Opinión (#4). Mixed sentence ("el plazo es de dos semanas, y
  es imposible") → primary the first function in the sentence, secondary
  the other. A reason attached to a proposal in the same sentence →
  Propuesta (primary) + Información (secondary), which feeds INFLU.P1.
  Reported speech of someone's opinion ("Pablo cree que no llegamos") is
  an Información about what Pablo said; the opinion belongs to Pablo.
- **ISO.** Inform (and its subtypes in the Task dimension).
- **Rubric.** COLLAB.P3; SOLVE.P1, SOLVE.P5; INFLU.P1–P3 (as the reason
  part); COMM.P1–P3; FEEDB.P1, FEEDB.P4; INIT.P3; ADAPT.P5.
- **Analogue agreement.** DAMSL Statement κ .66 [3] (that class holds
  fact and opinion together); ISO Task dimension .47 plain / .71 weighted
  [1].
- **Corpora.** All; still a large class after the split (MRDA statements
  are 68% before any split [6]), so report per-class F1, not accuracy.

#### 4. Opinión (opinion / assessment)

- **Definition.** Expresses an evaluation, judgement, belief, preference
  or stance about the topic, a person's work or an option: it says how
  the speaker sees it, not what happened. Not in reply to a question and
  not a proposal for action. Words that often signal it: "creo que", "me
  parece", "para mí", "es mejor / peor / peligroso", "no me convence".
- **Examples.** "Creo que el proveedor no va a llegar." · "Me parece
  demasiado arriesgado." · "El problema no es el proceso, es que nadie lo
  revisa." · "Para mí es lo más importante de la semana."
- **Boundary (Opinión vs Información).** Evaluation or stance → Opinión;
  checkable fact or report → Información. "Creo que" does not decide it
  by itself: "Creo que la reunión era a las diez" is a hesitant fact
  (Información); "Creo que la reunión es una pérdida de tiempo" is a
  judgement (Opinión). A prediction that depends on the speaker's reading
  ("no va a llegar") → Opinión; a scheduled fact ("llega el jueves") →
  Información. Evaluation that replies to an earlier proposal or to
  someone else's opinion → Aceptación or Rechazo, not Opinión.
  Evaluation of a person → Relación.
- **ISO.** No separate class: Inform with an evaluative qualifier. AMI
  has a separate Assess class [7].
- **Rubric.** SOLVE.P4 (values options with criteria); SOLVE.P1 when the
  problem is stated as a judgement; CONFL.P1 and INFLU.P1 as the reason
  part (secondary label); COMM.P1.
- **Analogue agreement.** No figure found for Inform vs Assess alone.
  Expected to be one of the weakest pairs together with
  Propuesta/Petición; the gold set (§6.3) must test it. If κ for either
  class falls below .60 the two are merged back (the owner chose to
  collect both so that merging stays possible).
- **Corpora.** AMI Assess; MRDA statement subtypes [unverified names].

#### 5. Propuesta (proposal / suggestion)

- **Definition.** Puts forward an idea, solution, plan, offer or procedure
  for the group to consider and accept or reject. The speaker does not
  bind a specific person to do it.
- **Examples.** "Podríamos dividir el informe en dos partes." · "¿Y si lo
  probamos primero con un cliente?" (suggestion phrased as a question) ·
  "Propongo que votemos."
- **Boundary.** "¿Y si…?" → Propuesta, not Pregunta. Proposal of a
  procedure ("vamos punto por punto") → Gestión if it is about the
  conversation itself, Propuesta if it is about the work. Offer to do
  something oneself ("Si queréis, lo hago yo") → Compromiso when
  unconditional, Propuesta when conditional on acceptance.
- **ISO.** Suggest (directive), Offer (commissive).
- **Rubric.** INFLU.P1, P2, P4, P5; CONFL.P3, CONFL.P5; SOLVE.P3;
  INIT.P2; COORD.P2, COORD.P3; ADAPT.P4; COMM.P1. Negative: SOLVE.N1
  (jumps to a solution), LISTEN.N2 (repeats another's proposal).
  act4teams solution codes [14] and NegotiAct offers [12] are the closest
  coded analogues; solution proposals and the replies that follow them
  carry the leadership finding (lag-1 z = 4.01) cited in RESEARCH.md.
- **Analogue agreement.** DAMSL Influence-on-Addressee-Future-Action κ
  .70 [3].
- **Corpora.** AMI Suggest/Offer/Elicit-Offer-or-Suggestion; MRDA
  suggestion tags [unverified: exact specific-tag names not checked].

#### 6. Petición (request / instruction)

- **Definition.** Asks or tells a specific person (or the group) to do
  something.
- **Examples.** "Marta, ¿me mandas los datos esta tarde?" · "Revisad el
  documento antes del lunes." · "Necesito que alguien llame al
  proveedor."
- **Boundary.** "¿Me mandas…?" → Petición, not Pregunta. Group-addressed
  "habría que…" without addressee → Propuesta. The line between Propuesta
  and Petición is the expected main confusion; if the gold set shows κ
  below target for this pair, merge them into one "Propuesta/Petición"
  category and keep addressee as an attribute.
- **ISO.** Request, Instruct.
- **Rubric.** COORD.P3, COORD.N4 (dictates without checking); COLLAB.P2
  (asks for help); FEEDB.P3 (asks for feedback, often as Pregunta).
- **Analogue agreement.** Same DAMSL class as Propuesta (.70) [3].
- **Corpora.** MRDA command tag [unverified]; AMI has no separate
  request class (falls under Suggest) [7]; DIHANA no.

#### 7. Aceptación (acceptance / agreement)

- **Definition.** Accepts, supports or agrees with an earlier proposal,
  request, offer or opinion by another speaker (`responds_to`). Includes
  agreement plus addition ("Sí, y además…").
- **Examples.** "Vale, me parece bien." · "De acuerdo, hagámoslo así." ·
  "Totalmente, y eso nos ahorra una semana."
- **Boundary.** Bare "vale"/"sí" during someone else's telling →
  Señal de escucha; after a proposal or request and closing that
  sequence → Aceptación. "Sí, pero…" → Rechazo (dispreferred format
  [23]). Acceptance of a request that commits the speaker ("Vale, lo
  hago") → Aceptación + Compromiso.
- **ISO.** Agreement; Accept Suggest; Accept Request; Accept Offer;
  Confirm (when the first part was a proposal).
- **Rubric.** LISTEN.P3 (builds on a colleague's idea); COLLAB.P4
  (credits); ADAPT.P2, ADAPT.P6; FEEDB.N3 (inverse). Layer 2: decision
  candidate; INCL.N3 (consensus declared while others have not spoken)
  needs the acceptances counted per speaker.
- **Analogue agreement.** Moderate: DAMSL Agreement κ .42 [3]; ISO Task
  .47–.71 [1].
- **Corpora.** MRDA accept tags with adjacency pairs; AMI Assess +
  adjacency relations (positive); DIHANA "Afirmación" (only as yes/no in
  a task dialogue, weak analogue).

#### 8. Rechazo (rejection / disagreement / correction)

- **Definition.** Rejects a proposal or request, disagrees with an
  opinion, corrects a statement, or objects. Includes partial rejection
  ("Sí, pero…") and rhetorical questions used to object.
- **Examples.** "No, eso no va a funcionar porque el cliente no lo
  acepta." · "Sí, pero el plazo no da." · "Eso es una tontería, con un
  proceso sobra."
- **Boundary.** Rejection with reason vs without is a rubric distinction
  (CONFL.P1 vs CONFL.N2), resolved by the secondary label Información or
  by the judge, not by a new category. Personal attack without task
  content → Relación (and CONFL.N1), not Rechazo.
- **ISO.** Disagreement, Correction, Decline Suggest/Request/Offer,
  Disconfirm.
- **Rubric.** CONFL.P1, P2, N2, N4; INFLU.N4; FEEDB.N3; ADAPT.N1;
  COLLAB.N4; SOLVE.N3 (ignores disconfirming information); opportunity
  trigger for CONFL for the other speakers.
- **Analogue agreement.** As Aceptación (.42–.71).
- **Corpora.** MRDA reject tags [unverified names]; AMI adjacency
  relations (negative); NegotiAct "rejecting offer" (not public).

#### 9. Compromiso (commitment)

- **Definition.** The speaker binds themself to a future action ("lo
  hago yo", "te lo mando mañana"). Includes accepting a request when the
  acceptance is itself the commitment.
- **Examples.** "Yo me encargo de llamar al proveedor." · "Te lo mando
  mañana antes de las diez." · "Lo reviso esta semana."
- **Boundary.** Vague intention without ownership ("habría que mirarlo")
  → Propuesta; promise about others ("Juan lo hará") → Petición or
  Información depending on whether Juan accepts. Commitment reports of
  past commitments ("ya lo mandé") → Información (INIT.P3).
- **ISO.** Promise; Accept Request (when it commits); Offer (when
  unconditional).
- **Rubric.** INIT.P1, INIT.P4, INIT.N2; COLLAB.P1, COLLAB.P6; COMM.P6;
  COORD.P3; ADAPT.P6. Layer 2: action item.
- **Analogue agreement.** **Low unless carefully defined:** DAMSL
  Committing-Speaker-Future-Action κ .15 [3]; flat action-item scheme κ
  .36, hierarchical (task, timeframe, owner, agreement) .64–.78 [18]. Rare
  (≈ 1–2% of utterances in ICSI [18]): oversample in the gold set.
- **Corpora.** ICSI action-item annotations of Purver et al. (18
  meetings; availability not checked); AMI ACTIONS summaries (weak, not
  utterance-level).

#### 10. Señal de escucha (backchannel / acknowledgement)

- **Definition.** Short token that shows attention or understanding
  without taking the floor or adding content: "ajá", "sí, sí", "vale",
  "ya", "mm", "claro" during another's talk; also "ah, vale" as
  change-of-state with no further content.
- **Examples.** "Ajá." · "Sí, sí." · "Ya, ya."
- **Boundary.** Spanish "vale/claro/ya" are ambiguous (§2 [29]): if it
  closes a proposal sequence → Aceptación; if it starts a turn that
  continues with content → label the content, not the token. Floor
  holders ("eh…", "bueno…") → Otro (or Gestión when they announce a
  turn).
- **ISO.** Auto-Feedback (AutoPositive), Allo-Feedback; Turn Management
  for floor tokens.
- **Rubric.** None directly (a listening behaviour that the rubric does
  not score from text); used as a participation descriptor and to keep
  backchannels out of the speaking-time and "turns" counts in
  `interaction.py`.
- **Analogue agreement.** Function within feedback: ISO Auto-FB κ .21
  plain / .57 weighted; identifying that the dimension applies: .90 [1].
  MRDA backchannels 13.4% of tags [6].
- **Corpora.** MRDA, SwDA, AMI Backchannel. **Our transcripts may lack
  most of them** (§5.4).

#### 11. Gestión de la reunión (process / discourse management)

- **Definition.** Talk about the conversation itself: agenda, order,
  method, turn allocation, time, summaries of what was said or decided,
  topic changes, opening and closing moves.
- **Examples.** "Vamos al segundo punto." · "Nos quedan diez minutos." ·
  "Entonces, resumiendo: lo entrega Marta el viernes y Luis revisa." ·
  "Irene, no has dicho nada todavía, ¿cómo lo ves?" (Gestión + Pregunta)
- **Boundary.** Summaries of decisions are Gestión, and also feed the
  layer-2 decision. Inviting someone to speak is Gestión (turn
  allocation) with Pregunta as secondary. Procedure proposals about the
  work → Propuesta.
- **ISO.** Discourse Structuring, Turn Management (Turn Assign), Time
  Management.
- **Rubric.** COORD.P1, P2, P4, P5, P6, COORD.N1, COORD.N2; INCL.P1,
  P2, P3, P4, P5; INCL.N3; COMM.P2, COMM.P4; INFLU.N3 (pushes for
  closure); SOLVE.N4. act4teams procedural facet; NegotiAct process-
  related communication.
- **Analogue agreement.** ISO Discourse Structuring κ .74, Turn .82, Time
  .58 [1].
- **Corpora.** MRDA (floor grabber/holder, topic change tags
  [unverified names]); AMI topic segmentation (as phase signal).

#### 12. Relación (social / relational)

- **Definition.** Acts that manage the relationship rather than the task:
  greetings, thanks, apologies, praise and criticism of a person (not of
  an idea), encouragement, jokes, personal chat.
- **Examples.** "Gracias por traerlo tan preparado, Pablo." · "Perdona, me
  he pasado." · "Buenos días a todos."
- **Boundary.** Praise of specific work ("el análisis de la página 4 está
  muy bien hecho") → Opinión (an evaluation of work, not of the person;
  FEEDB.P1/P6); praise of the person → Relación. Criticism of an
  idea → Rechazo; of a person → Relación (CONFL.N1, COLLAB.N2,
  FEEDB.N4). Named by **act** (thanks, apology, criticism), never by an
  inferred feeling.
- **ISO.** Social Obligations Management (greeting, thanking, apologising,
  valediction); praise/criticism have no direct ISO function (sentiment is
  a qualifier).
- **Rubric.** COLLAB.P4, COLLAB.N2; CONFL.P6, CONFL.N1; FEEDB.P6, FEEDB.N4;
  INCL.N4; INIT.P6. Bales positive/negative socio-emotional areas [16];
  NegotiAct socio-emotional statements [12].
- **Analogue agreement.** ISO Social Obligations κ 1.00 on small Dutch
  data [1]; praise/criticism expected lower (not measured anywhere found).
- **Corpora.** AMI Be-Positive/Be-Negative (closest); SwDA thanking,
  apology tags.

#### 13. Otro (other)

- **Definition.** Abandoned or interrupted fragments, self-corrections,
  stalls, unintelligible or off-task talk with no function in the
  categories above.
- **Examples.** "Es que el… bueno, no." · "Eh…" · "[ininteligible]"
- **ISO.** Own Communication Management, Stalling; Partner Comm. Mgmt.
- **Rubric.** COMM.N3 (restarts without resolving) when frequent.
- **Analogue agreement.** DAMSL Abandoned .64, Unintelligible .14 [3].
- **Corpora.** AMI Fragment/Stall/Other; MRDA disruption tags.

### 5.3 Summary table

| # | Category | ISO 24617-2 | Main rubric links | Analogue κ | Public evaluation data |
|---|---|---|---|---|---|
| 1 | Pregunta | Question family | LISTEN.P2/P5, INCL.P1/P3/P4, SOLVE.P2, INFLU.P6 | ~.80 (MRDA overall) | MRDA, SwDA, AMI; DIHANA (es) |
| 2 | Respuesta | Answer, Confirm, Disconfirm | COMM.N4, COLLAB.N1, INIT.N2 | .76 (DAMSL) | MRDA, SwDA; DIHANA (es) |
| 3 | Información | Inform | COLLAB.P3, SOLVE.P1/P5, FEEDB.P1/P4 | .66 (DAMSL); .47/.71 (ISO Task) | all |
| 4 | Opinión | Inform with an evaluative qualifier (no own class) | SOLVE.P4, CONFL.P1 and INFLU.P1 (reason part) | none measured; AMI keeps Assess apart from Inform [7] | AMI Assess, MRDA |
| 5 | Propuesta | Suggest, Offer | INFLU.P1–P5, SOLVE.P3, INIT.P2, CONFL.P3/P5 | .70 (DAMSL) | AMI, MRDA |
| 6 | Petición | Request, Instruct | COORD.P3/N4, COLLAB.P2 | .70 (DAMSL) | MRDA |
| 7 | Aceptación | Agreement, Accept* | LISTEN.P3, ADAPT.P2/P6, layer-2 decision | .42 (DAMSL) – .71 (ISO) | MRDA, AMI relations |
| 8 | Rechazo | Disagreement, Correction, Decline* | CONFL.P1/P2/N2, INFLU.N4, ADAPT.N1 | .42–.71 | MRDA, AMI relations |
| 9 | Compromiso | Promise, Accept Request | INIT.P1/P4/N2, COLLAB.P1/P6, COORD.P3 | .15 (DAMSL) – .78 (hier. action items) | ICSI action items (if obtainable) |
| 10 | Señal de escucha | Auto/Allo-Feedback, Turn | participation descriptor only | .21/.57 function; .90 dimension (ISO) | MRDA, SwDA, AMI |
| 11 | Gestión | Discourse Structuring, Turn, Time | COORD.*, INCL.*, COMM.P2/P4 | .58–.82 (ISO) | MRDA, AMI topics |
| 12 | Relación | Social Obligations Mgmt | COLLAB.P4/N2, CONFL.P6/N1, FEEDB.P6/N4 | 1.00 (ISO, small) | AMI Be-Pos/Neg, SwDA |
| 13 | Otro | Own Comm. Mgmt, Stalling | COMM.N3 | .14–.64 | AMI, MRDA |

Target for our gold set: κ ≥ .60 per category between two trained
annotators before a category is used for evidence (Bakeman & Quera's .62
threshold, as cited in [12]); categories below that are shown as
"provisional" or merged (Propuesta+Petición; Información+Opinión;
Aceptación vs Señal de escucha are the likely candidates).

#### 5.3.1 Certainty per category (owner request, #63)

**Why a figure.** The owner wants to keep or drop categories that barely
beat chance. κ is the chance-corrected agreement between two annotators:
0 = no better than chance, 1 = perfect agreement. κ × 100 reads as "% of
the possible agreement above chance". Plain-Spanish explanation for
annotators: [`DIALOGUE_ACTS_GUIDE.md`](DIALOGUE_ACTS_GUIDE.md) §11.

**(a) Borrowed expectation, before we measure anything.** Every figure
below comes from another scheme, unit or language (the "analogue" lines of
§5.2) and is **not** the agreement we will obtain. Verdict rule: by the
*lowest* figure of the range (conservative); keep κ ≥ .60, provisional
.40–.60, at risk of removal or merging < .40.

| # | Category | Borrowed κ (range) | % of possible agreement above chance | Source scheme | Verdict (borrowed) |
|---|---|---|---|---|---|
| 1 | Pregunta | .75–.80 | 75–80 % | MRDA coarse, SWBD-DAMSL overall [5][6] | keep |
| 2 | Respuesta | .76 (single figure) | 76 % | DAMSL Answer [3] | keep |
| 3 | Información | .47–.71 | 47–71 % | DAMSL Statement .66 [3]; ISO Task .47 plain / .71 weighted [1]; these classes include opinion, so the split may lower it | provisional |
| 4 | Opinión | no figure found | — | AMI separates Assess [7] but no κ for the pair was found | no borrowed figure: provisional by default |
| 5 | Propuesta | .70 (single figure, pooled with Petición) | 70 % | DAMSL Influence-on-Addressee-Future-Action [3] | keep (the pooled class; the split is untested) |
| 6 | Petición | .70 (single figure, pooled with Propuesta) | 70 % | same class as 5 [3] | keep (same caveat) |
| 7 | Aceptación | .42–.71 | 42–71 % | DAMSL Agreement .42 [3]; ISO Task .47–.71 [1] | provisional |
| 8 | Rechazo | .42–.71 | 42–71 % | as 7 | provisional |
| 9 | Compromiso | .15–.78 | 15–78 % | DAMSL .15 [3]; flat action-item scheme .36, hierarchical .64–.78 [18] | at risk (rare class; depends on a careful definition) |
| 10 | Señal de escucha | .21–.57 (function); .90 only for "does this dimension apply" | 21–57 % | ISO Auto-feedback plain .21 / weighted .57 [1] | at risk |
| 11 | Gestión de la reunión | .58–.82 | 58–82 % | ISO Discourse Structuring .74, Turn .82, Time .58 [1] | provisional (lowest .58) |
| 12 | Relación | 1.00 on small Dutch data; praise/criticism never measured | — (1.00 is not reliable) | ISO Social Obligations [1] | no reliable figure: provisional by default |
| 13 | Otro | .14–.64 | 14–64 % | DAMSL Abandoned .64, Unintelligible .14 [3] | at risk |

Verdicts for 4 and 12 are a decision made in this document, not a
literature result.

**(b) Measured rule, after our own double annotation.** Each category gets
its own κ (two annotators, independently, on the gold set of §6.2) with a
95 % bootstrap confidence interval (bootstrap tooling exists in the lab).
Verdict from the measured κ with the same bands (keep ≥ .60, provisional
.40–.60, at risk < .40). A category is **dropped or merged** when the
upper bound of its 95 % interval stays below .40 (it cannot be shown to
beat chance by a useful margin), and when the interval is too wide to say
(few instances) more examples are annotated before deciding; rare
categories must reach ≥ 50 instances (§6.2). Candidate merges: Propuesta +
Petición, Información + Opinión, Aceptación + Señal de escucha. The
verdict for a merge is computed on the merged category.

**(c) The automatic tagger, later.** Each predicted label carries a
**calibrated confidence in %** (calibration measured against the gold
set: of the labels given 70 %, about 70 % are right). A label whose
confidence is not clearly above the **base rate** of its category (its
share among all sentences in the gold set) is shown as **"dudoso"** or not
shown. The margin over the base rate and the "dudoso"/hidden cut-off are
**not fixed yet**: they are set on the gold set when it exists. Reported
next to each per-category κ: the tagger's κ against the consensus (§6.3).

### 5.4 What text-only loses

| Lost | Effect | Evidence |
|---|---|---|
| Intonation | Declarative questions read as statements; irony and sarcasm read literally; "sí" vs "sí…?" | General DA literature uses prosody for exactly these cases [5]; GPT-4 with audio beat text on MRDA, 41.8% vs 31.4% [9] |
| Backchannels | ASR (Whisper) tends to omit short tokens; MRDA counts them as 13.4% of tags [6]. NegotiAct coded from video because "mm hmm" active listening is lost in transcripts [12] | Listening cannot be measured from our backchannel counts |
| Overlap and interruptions | Rubric INCL.N1, LISTEN.N4 need overlap; `transcript.v1` has `overlap` but the current diarizer rarely sets it | AUDIO.md §6 |
| Timing | Delays before dispreferred replies [23][26]; available from word times, so not lost, but not used yet | — |
| Nods, gaze, gestures | Visual agreement and listening [27c] | No video in the pipeline |
| ASR errors | Classifier performance falls: AUROC .91 → .80 on ASR transcripts [17]; SwDA 71% → 65% [5] | Flag low-confidence sentences |

### 5.5 AI Act limits

- Art. 5(1)(f) prohibits AI systems that infer emotions of a natural person
  in the workplace and education, except for medical or safety reasons;
  the Commission guidelines read it as covering inference from biometric
  data (voice, face), and state that inference from written text is not
  based on biometric data (paras 245, 251, 254) — details and the caveat
  that these were read from unofficial reproductions are in
  [RESEARCH.md §7](RESEARCH.md).
- Consequences for this category set:
  1. Categories are **communicative acts** computed from text. None is an
     emotion ("Relación" labels thanks or a criticism, not anger or
     happiness). AMI's Be-Positive/Be-Negative are mapped to acts, not
     feelings.
  2. **No prosodic feature may be used to infer emotion, stress or
     attitude.** Using audio features to decide whether a sentence is a
     question is a different purpose, but because voice is biometric data
     under the guidelines, any audio-based classifier needs a legal check
     before use; this study recommends text-only for layer 1.
  3. Using these labels to evaluate workers places the product in Annex
     III 4 (high-risk) and under GDPR Art. 22; the labels are evidence for
     a human assessor, not a decision (RESEARCH.md §7).

---

## 6. How to build and measure it

### 6.1 Options

Costs on the workstation's RTX 3070 Ti (8 GB, ~2.7 GB already in use by
the desktop and other jobs at the time of writing, so ~5 GB usable). Time
figures are **estimates** from model size, not measurements, except where
stated.

| Option | What | Data needed | GPU cost (estimate) | Expected quality (from §3.2) | Risks |
|---|---|---|---|---|---|
| A. Rules | "¿…?" → Pregunta; lexicon for backchannels ("ajá", "vale" alone in a short sentence), commitments ("me encargo", "lo hago yo"), management ("siguiente punto", "nos quedan") | None | None | Good for Pregunta (with "¿") and Señal de escucha; poor elsewhere | Brittle with ASR punctuation |
| B. Local LLM, few-shot | Qwen3 from the existing ladder, prompt with definitions + Spanish examples, 3–5 sentences of context, one label (+ optional second) | ~5 examples per category | Qwen3-4B at 4-bit ≈ 3 GB; Qwen3-8B at 4-bit ≈ 5–6 GB (tight with 2.7 GB in use). Measured on this GPU: Qwen3-1.7B took 49 s for 64 sentences in call 1 (AUDIO.md §4); a 4B model a few minutes per meeting | Unknown for 13 coarse Spanish categories; fine-grained LLM DA accuracy is 31–40% [9]; the 1.7B judge showed position bias | Position bias, instability; needs label shuffling and measurement |
| C. API LLM (Claude) | Same prompt | Same | None locally | Likely the best zero-shot option; not measured | Data leaves the machine: only with consent (DESIGN.md §4) |
| D. Fine-tuned multilingual encoder | XLM-R base or mDeBERTa-v3-base (~280 M params) trained on English corpora mapped to the 13 categories (AMI, ICSI-MRDA; SwDA only if NC licence is acceptable) + DIHANA for Spanish question/answer; context window of previous sentences; zero-shot transfer to Spanish meetings | Mapping tables; corpora are free | Training in fp16, seq 128–256, batch 16: fits in ~5 GB; ~1–3 h for ~200k utterances on this GPU (estimate). Inference: thousands of sentences per minute | Coarse-set supervised models reach 91% in English [34][35]; cross-lingual transfer loss to Spanish meetings unknown | Domain shift (design role-play, telephone, railway info → Spanish workplace); categories 8 and 11 poorly covered by mappings |
| E. D + machine-translated corpora / Spanish gold set fine-tuning | Translate AMI/MRDA to Spanish, or fine-tune D on our own annotated sentences | Our gold set (§6.3) | As D | Usually the step that closes the domain gap | Needs annotation effort first |
| F. Hybrid | Rules for the easy categories, D or B for the rest, LLM only for disagreements | — | Sum of parts | — | More moving parts |

Licences for training a model that may be used commercially: AMI and ICSI
are CC BY 4.0 (attribution only) [7][6]; DIHANA CC BY 4.0 [11]; **SwDA is
CC BY-NC-SA** [unverified] and MIAM/SILICONE state inconsistent licences:
use SwDA for evaluation only unless a legal review says otherwise.

### 6.2 Recommended first step

1. **Write the annotation guide** for the 13 categories (done:
   [`DIALOGUE_ACTS_GUIDE.md`](DIALOGUE_ACTS_GUIDE.md); §5.2 is its basis), with `responds_to` and the optional second label.
2. **Annotate a Spanish gold set** (two annotators, independently, then
   consensus): ~1,500 sentences from 6–10 meetings (role-played or with
   consent), oversampling meetings with decisions and task assignment so
   that Compromiso and Rechazo reach ≥ 50 instances each. Existing tooling
   (`skills agreement`, annotation forms with sentence ids) covers most of
   the mechanics.
3. **Build options A, B (Qwen3-4B) and D in parallel** — each is days of
   work, not weeks — and evaluate all three on the gold set. Keep C as a
   ceiling reference only on synthetic or consented data.
4. **Decide per category** which option to use (F), and which categories
   stay provisional.

Rationale: without the Spanish gold set no option can be judged, and the
gold set also answers whether the categories themselves are reliable,
which is the first decision the issue asks for.

### 6.3 Evaluation plan

| Question | Data | Metric | Pass mark |
|---|---|---|---|
| Are the categories reliable for people? | Spanish gold set, 2 annotators | Cohen's κ per category and overall; confusion matrix; % sentences annotators wanted to split | κ ≥ .60 per category (else merge or mark provisional) |
| Does the tagger match people? | Gold set consensus | Per-class precision/recall/F1, macro-F1, κ system vs consensus, compared with annotator–annotator κ (the ceiling) | System κ within .10 of the human κ |
| Does it transfer? | AMI and MRDA test splits mapped to our categories; DIHANA L1 for Pregunta/Respuesta | Macro-F1 by corpus | Reported, not gating (different domain) |
| Does ASR hurt? | Gold set on human-corrected vs raw `transcript.v1` text | Δ macro-F1 | Reported; flag sentences with low `confidence` |
| Is layer 2 right? | Gold set decisions and action items marked by annotators | Precision/recall of derived decisions and action items | Reported; action items need κ for the four parts [18] |
| Is it fair? | Gold set by speaker gender, accent, native vs non-native | Per-group F1 and error-rate ratios | Ratio ≥ .80 (four-fifths, RESEARCH.md §7) |

Sample-size note: with ~1,500 sentences and a majority class near 50–70%
(Información), rare categories will have tens of instances; κ for a
category with 50 instances has a wide confidence interval (bootstrap it;
the lab already has bootstrap tooling). Oversampling is required, not
optional.

### 6.4 Decisions for the owner

1. **Category set** — **decided 2026-11-15 (owner):** keep the
   categories separate. Propuesta and Petición stay apart; Opinión is
   split from Información. 13 first-layer categories; data is collected
   for each, and they are merged later only if the gold set shows κ below
   .60 (§5.3). Recorded in §0, §5.2 and
   [`DIALOGUE_ACTS_GUIDE.md`](DIALOGUE_ACTS_GUIDE.md).
2. **Second label** (open): allow a secondary label per sentence (more faithful,
   costlier to annotate) or force one?
3. **Gold set source**: role-plays written/recorded in-house, or real
   meetings with consent? Who annotates (the psychologist, students)?
4. **Conversation type**: add an optional `session.type` to
   `transcript.v1` (declared by the uploader) — yes/no.
5. **API judge**: may consented recordings be sent to Claude to obtain a
   ceiling reference?
6. **SwDA (NC licence)**: evaluation only, or exclude?

---

## 7. Limitations of this study

- **Not all primary sources were read in full.** Several CA works
  (Sacks et al. 1974, Schegloff 2007, Pomerantz 1984, Schegloff et al.
  1977) are cited from bibliographic records and standard summaries
  [unverified]. The SWBD-DAMSL and MRDA (Dhillon et al.) coding manuals
  could not be opened.
- **Reliability figures are analogues.** They come from other schemes,
  units, languages (ISO figures from Dutch data) and media (act4teams and
  NegotiAct coded from video). None measures our categories on Spanish
  meeting transcripts. The expected values in §5 are educated guesses
  until the gold set exists.
- **AMI publishes no dialogue-act agreement**, so its labels are an
  evaluation reference of unknown reliability.
- **Category counts** for ISO (26 general-purpose) are our own count from a
  table; act4teams' code count was not confirmed.
- **No Spanish meeting corpus with dialogue acts was found**; absence from
  our searches is not proof of absence (searched in English and Spanish,
  not in Portuguese or Catalan; no LDC catalogue search beyond CALLHOME).
- **Phase models** for meetings, appraisal and coaching lack empirical
  validation in what we found; the meeting phases in §4.2 are a composite.
- **GPU costs are estimates**; only the Qwen3-1.7B figure was measured on
  this machine.
- **Legal points** rely on RESEARCH.md §7, which read EU texts from
  unofficial reproductions.
- This study was compiled by an AI research assistant with web access and
  the project's saved papers; it has not been reviewed by a linguist or
  conversation analyst.

---

## References

[1] Bunt, H., Petukhova, V., Traum, D., Alexandersson, J. (2017).
Dialogue act annotation with the ISO 24617-2 standard. In Dahl (ed.),
*Multimodal Interaction with W3C Standards*, Springer. Tables 6.1–6.3, §6.4
(DBOX). https://people.ict.usc.edu/~traum/Papers/iso-2017.pdf [verified].
Annotation guidelines:
https://dialogbank.lsv.uni-saarland.de/wp-content/uploads/2015/12/ISO24617-2_Annotation_Guidelines2017.pdf
[verified]. DialogBank: https://dialogbank.lsv.uni-saarland.de/ [verified].

[2] Bunt, H., Petukhova, V., Gilmartin, E., et al. (2020). The ISO standard
for dialogue act annotation, second edition. LREC 2020.
https://www.lsv.uni-saarland.de/wp-content/publications/2020/LREC2020_ISO24617-2_Ed2.pdf
[verified].

[3] Core, M., Allen, J. (1997). Coding dialogs with the DAMSL annotation
scheme. AAAI Fall Symposium.
https://www.eecis.udel.edu/~carberry/CIS-885/Papers/Core-Allen-Coding-Damsl.pdf
[verified].

[4] Jurafsky, D., Shriberg, E., Biasca, D. (1997). Switchboard SWBD-DAMSL
shallow-discourse-function annotation coders manual. Manual not reachable
(TLS error) [unverified]. SwDA: https://compprag.christopherpotts.net/swda.html
[verified]; https://huggingface.co/datasets/cgpotts/swda (licence CC
BY-NC-SA 3.0 [unverified, snippet]).

[5] Stolcke, A., et al. (2000). Dialogue act modeling for automatic
tagging and recognition of conversational speech. *Computational
Linguistics* 26(3). https://aclanthology.org/J00-3003.pdf [verified].

[6] Shriberg, E., Dhillon, R., Bhagat, S., Ang, J., Carvey, H. (2004).
The ICSI Meeting Recorder Dialog Act (MRDA) corpus. SIGdial.
https://aclanthology.org/W04-2319.pdf [verified]. ICSI licence:
https://groups.inf.ed.ac.uk/ami/icsi/license.shtml [verified]. Preprocessing:
https://github.com/NathanDuran/MRDA-Corpus [verified].

[7] AMI Consortium (2005). Guidelines for dialogue act and addressee
annotation, v1.0.
https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf
[verified]. Annotation overview:
https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml [verified].

[8] AMI Meeting Corpus, overview and licence (CC BY 4.0).
https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml,
https://groups.inf.ed.ac.uk/ami/corpus/license.shtml [verified; local
copies `ami_overview.html`, `ami_license.html`].

[9] Qamar, A., Tong, J., Huang, R. (2025). Do LLMs understand dialogues? A
case study on dialogue acts. ACL 2025.
https://people.engr.tamu.edu/huangrh/papers/acl2025_main_LLM_DA.pdf
[verified].

[10] Benedí, J.-M., et al. (2006). Design and acquisition of a telephone
spontaneous speech dialogue corpus in Spanish: DIHANA. LREC. Figures as
reported in Ribeiro, Ribeiro, Martins de Matos (2019), Hierarchical
multi-label dialog act recognition on Spanish data,
https://arxiv.org/pdf/1907.12316 [verified].

[11] DIHANA corpus on Zenodo (CC BY 4.0). https://zenodo.org/record/5572469
[verified]. MIAM benchmark card:
https://huggingface.co/datasets/miam/raw/main/README.md [verified]; SILICONE:
https://huggingface.co/datasets/eusip/silicone [verified].

[12] Jäckel, E., Zerres, A., Hemshorn de Sanchez, C., Lehmann-Willenbrock,
N., Hüffmeier, J. (2024). NegotiAct: introducing a comprehensive coding
scheme to capture temporal interaction patterns in negotiations. *Group &
Organization Management* 49(3), 743–783. doi:10.1177/10596011221132600
[verified, local copy `jaeckel2024_negotiact.pdf`].

[13] Kauffeld, S., Lehmann-Willenbrock, N. (2012). Meetings matter.
*Small Group Research* 43, 128–156. doi:10.1177/1046496411429599.
https://www.mangold-international.com/research/publications/human-behavior/meetings-matter-effects-of-work-group-communication-on-organizational-success
[verified, summary page]. act4teams chapter, Cambridge Handbook of Group
Interaction Analysis (2018):
https://www.cambridge.org/core/books/cambridge-handbook-of-group-interaction-analysis/advanced-interaction-analysis-for-teams-act4teams-coding-scheme/1930BA23CB78C1881DD6EAC30CC4FB5B
[unverified].

[14] Lehmann-Willenbrock, N., Meinecke, A., Rowold, J., Kauffeld, S.
(2015). How transformational leadership works during team interactions.
*The Leadership Quarterly* 26, 1017–1033.
doi:10.1016/j.leaqua.2015.07.003 [verified, local copy].

[15] Sun, C., Shute, V., Stewart, A., Yonehiro, J., Duran, N., D'Mello, S.
(2020). Towards a generalized competency model of collaborative problem
solving. *Computers & Education* 143, 103672.
https://par.nsf.gov/servlets/purl/10140613 [verified].

[16] Bales, R. F. (1950). *Interaction Process Analysis*. Addison-Wesley.
Description:
https://www.encyclopedia.com/social-sciences/encyclopedias-almanacs-transcripts-and-maps/observation-systems
[verified, secondary].

[17] Pugh, S., Subburaj, S., Rao, A., Stewart, A., Andrews-Todd, J.,
D'Mello, S. (2021). Say what? Automatic modeling of collaborative problem
solving skills from student speech in the wild. EDM 2021.
https://par.nsf.gov/servlets/purl/10494306 [verified, local copy].

[18] Purver, M., Dowding, J., Niekrasz, J., Ehlen, P., Noorbaloochi, S.,
Peters, S. (2007). Detecting and summarizing action items in multi-party
dialogue. SIGdial. https://aclanthology.org/2007.sigdial-1.4.pdf [verified].
Gruenstein et al. (2005): https://aclanthology.org/2005.sigdial-1.13
[unverified].

[19] Galley, M., McKeown, K., Fosler-Lussier, E., Jing, H. (2003).
Discourse segmentation of multi-party conversation. ACL.
https://www.cs.jhu.edu/~yarowsky/acl03cd/main/pdfs/Galley.pdf [verified].

[20] Hsueh, P.-Y., Moore, J. (2007). Combining multiple knowledge sources
for dialogue segmentation in multimedia archives. ACL.
https://aclanthology.org/P07-1128.pdf [verified].

[21] Sacks, H., Schegloff, E. A., Jefferson, G. (1974). A simplest
systematics for the organization of turn-taking for conversation.
*Language* 50(4), 696–735. https://muse.jhu.edu/article/452679/summary
[unverified: page blocked].

[22] Schegloff, E. A. (2007). *Sequence Organization in Interaction*.
Cambridge University Press.
https://www.cambridge.org/core/books/sequence-organization-in-interaction/insert-expansion/7E01D8FD1B90854C305BFFA54B9FFFC1
[unverified].

[23] Pomerantz, A. (1984). Agreeing and disagreeing with assessments. In
Atkinson & Heritage (eds.), *Structures of Social Action*, 57–101.
https://emcawiki.net/Pomerantz1984a [unverified].

[24] Schegloff, E. A., Jefferson, G., Sacks, H. (1977). The preference for
self-correction in the organization of repair in conversation. *Language*
53(2), 361–382. https://emcawiki.net/Schegloff1977 [unverified].

[25] Schegloff, E. A., Sacks, H. (1973). Opening up closings. *Semiotica*
8, 289–327. https://nurc.fflch.usp.br/node/407 [unverified].

[26] Stivers, T., et al. (2009). Universals and cultural variation in
turn-taking in conversation. *PNAS* 106(26), 10587–10592.
https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608 [verified].

[27] Heritage, J. (1984). A change-of-state token and aspects of its
sequential placement. In *Structures of Social Action*, 299–345.
https://emcawiki.net/Heritage2006a [unverified].
[27b] Schegloff, E. A. (1982). Discourse as an interactional achievement:
some uses of "uh huh" and other things that come between sentences. GURT
1981, 71–93. [unverified].
[27c] Stivers, T. (2008). Stance, alignment, and affiliation during
storytelling. *Research on Language and Social Interaction* 41(1), 31–57.
https://forms.mpi.nl/node/48855 [unverified].

[28] Grupo Val.Es.Co. Unidades de la conversación.
https://www.uv.es/valesco/proy_unid.html [verified]; Grupo Val.Es.Co.
(2014), *Estudios de Lingüística del Español* 35, 11–71,
https://ddd.uab.cat/record/118916 [unverified].

[29] Spanish affirmative reply operators (sí, ya, claro, vale, exacto).
Edizioni Ca' Foscari chapter,
https://edizionicafoscari.it/media/pdf/books/978-88-6969-080-8/978-88-6969-080-8-ch-15.pdf
[unverified].

[30] Kunz, W., Rittel, H. (1970). Issues as elements of information
systems. Working Paper 131, Berkeley.
https://en.wikipedia.org/wiki/Issue-based_information_system [unverified].

[31] Rienks, R., Heylen, D., van der Weijden, E. (2005). Argument
diagramming of meeting conversations. MMMP workshop, ICMI.
https://www.idiap.ch/webarchives/sites/www.idiap.ch/workshop/icmi05/pdf/MMMP_paper24.pdf
[verified].

[32] Hsueh, P.-Y., Moore, J. (2007). What decisions have you made?
Automatic decision detection in conversational speech. NAACL.
https://aclanthology.org/N07-1004/ [unverified].

[33] CALLHOME Spanish Dialogue Act Annotation, LDC2001T61.
https://catalog.ldc.upenn.edu/LDC2001T61 [verified].

[34] Raheja, V., Tetreault, J. (2019). Dialogue act classification with
context-aware self-attention. NAACL. https://arxiv.org/pdf/1904.02594
[verified].

[35] He, Z., et al. (2021). Speaker turn modeling for dialogue act
classification. Findings of EMNLP. https://arxiv.org/pdf/2109.05056
[verified].

[36] Holmes, M. E. (1992). Phase structures in negotiation. In Putnam &
Roloff (eds.), *Communication and Negotiation*, Sage.
https://sk.sagepub.com/book/edvol/embed/communication-and-negotiation/chpt/phase-structures-negotiation
[verified].

[37] Bales, R. F., Strodtbeck, F. L. (1951). Phases in group problem
solving. *Journal of Abnormal and Social Psychology* 46(4), 485–495.
[unverified].

[38] Poole, M. S. (1983). Decision development in small groups III.
*Communication Monographs* 50(4), 321–341.
https://en.wikipedia.org/wiki/Poole%27s_multiple_sequence_model [verified,
secondary].

[39] Allen, J. A., Lehmann-Willenbrock, N., Rogelberg, S. G. (eds.) (2015).
*The Cambridge Handbook of Meeting Science*. CUP.
https://digitalcommons.unomaha.edu/facultybooks/335 [unverified].

[40] Fisher, B. A. (1970). Decision emergence: phases in group decision
making. *Speech Monographs* 37, 53–66. [unverified; phase names from a
weak secondary source].

[41] Osborn, A. F. (1953). *Applied Imagination*. Scribner.
https://en.wikipedia.org/wiki/Brainstorming [unverified].

[42] Adair, W. L., Brett, J. M. (2005). The negotiation dance: time,
culture, and behavioral sequences in negotiation. *Organization Science*
16(1), 33–51.
https://www.kellogg.northwestern.edu/academics-research/research/detail/2005/the-negotiation-dance-time-culture-and-behavioral-sequences-in/
[verified].

[43] Moore, C. W. *The Mediation Process* (summary).
https://www.beyondintractability.org/bksum/moore-mediation [verified,
summary].

[44] Campion, M. A., Palmer, D. K., Campion, J. E. (1997). A review of
structure in the selection interview. *Personnel Psychology* 50(3),
655–702. https://core.ac.uk/download/234763785.pdf [unverified].

[45] GROW model appraisal. Coaching at Work (2017).
https://coaching-at-work.com/2017/11/03/research-matters-the-grow-model-why-we-should-discard-blueprints
[unverified].

[46] Byrne, P. S., Long, B. E. L. (1976). *Doctors Talking to Patients*.
HMSO. Six phases:
https://gpnotebook.com/en-GB/pages/general-practice/byrne-and-long-phases-of-the-consultation/the-six-phases-of-the-consultation
[verified, secondary].

[47] Calgary-Cambridge guide (Kurtz, Silverman, Draper).
https://bradfordvts.co.uk/wp-content/onlineresources/communication-skills/teach-communication-skills/calgary-cambridge/01a%20calgary%20cambridge%20framework%20-%20basic.pdf
[unverified].

[48] Rackham, N. (1988). *SPIN Selling*. McGraw-Hill.
https://www.routledge.com/SPIN-Selling-1st-Edition/Rackham/p/book/9780566076893
[unverified].

[49] Tuckman, B. W. (1965). Developmental sequence in small groups.
*Psychological Bulletin* 63(6), 384–399.
https://garfield.library.upenn.edu/classics1984/A1984TD25600001.pdf
[unverified].
