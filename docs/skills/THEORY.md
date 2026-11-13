# Soft-skills analyzer — theoretical base

Status: research synthesis, compiled 2026-11-12. Scope: which frameworks, instruments and rating methods are scientifically defensible for rating **observable verbal behaviour in recorded, diarized group conversations**, and how to turn them into a rubric catalogue (competency → behavioural indicators → rating anchors).

Conventions used in this document:

- A dagger (†) marks a detail (a dimension name, a number, an anchor description) that is stated **from memory and was not re-confirmed against the source in this research pass**. Everything without a dagger was confirmed against the source, its abstract page, a publisher/Crossref record, or the published framework text.
- References are numbered `[n]` and listed in section 7 with a `[verified]` / `[unverified]` tag. "Verified" means bibliographic data and the specific claim attributed to it were confirmed; it does **not** mean the full text was read. Most journal articles were confirmed at abstract/record level only.
- Indicators are paraphrased. No rubric text is reproduced verbatim from copyrighted instruments.

---

## 1. Executive summary

1. **Do not build the catalogue on Belbin, Tuckman, MBTI or DISC.** Belbin's self-report inventory has weak discriminant validity and an ipsative format [44][45][46]; Tuckman is a descriptive group-stage model built from a literature review, not an individual assessment [48][49]; MBTI and DISC are type/style self-reports with no behavioural observation component [51][52][53]. None of them yields indicators that can be matched to transcript turns.
2. **Use a four-layer backbone instead:**
   - **Layer A — construct legitimacy (what HR calls the competencies):** competency-modelling best practice (Campion et al. 2011 [1]) and a cross-walk to frameworks HR actually uses and that are published in full: SHL UCF / "Great Eight" [2], UK Civil Service Success Profiles behaviours [8], OECD Competency Framework [9], UN Values and Behaviours Framework [10], SHRM BASK [12], EU LifeComp [14], plus PISA/ATC21S collaborative problem solving for the collaboration construct [16][17].
   - **Layer B — behavioural content (what the indicators are):** the team-process and teamwork-behaviour literature — Marks, Mathieu & Zaccaro [30], Rousseau, Aubé & Savoie [31], Salas, Sims & Burke [29], Stevens & Campion [28] — and the two instruments with published behavioural anchors or utterance-level codes: **CATME** (BARS, five dimensions) [32][33] and the **act4teams** meeting coding scheme [35]. Bales' IPA [26] and Benne & Sheats [27] remain useful as a coding vocabulary and team-level profile.
   - **Layer C — measurement method (how a rating is produced):** the assessment-centre process — observe, record, classify, evaluate — with behavioural checklists feeding a BARS-style summary scale, frame-of-reference calibration, and the 2015 International Taskforce guidelines as the professional standard [18][20][21][22][23].
   - **Layer D — team-level descriptors (context, never individual scores):** participation equality [39][40], Bales/act4teams category profiles, and group-development stage as low-confidence narrative context only.
3. **Be honest about the main scientific weakness.** Four decades of assessment-centre research show that dimension ("competency") ratings are dominated by exercise/situation and general-performance variance; dimension-specific variance is small [57][58][59][60][61][62]. The other side of the debate shows dimensions have criterion validity and fit better when collapsed to a few broad factors [63][64][65]. Design consequences: few, broad competencies; report per session with evidence; aggregate across several sessions and tasks; never present fine-grained competency differences from one meeting as stable traits.
4. **Scale:** two stages — turn-level behavioural incident coding (checklist, positive and negative indicators), then a **5-point behaviourally anchored summary scale with an explicit "insufficient evidence" outcome**. Silence is "not observed", not a low score.
5. **Position the product as developmental feedback with a human in the loop**, not as selection or performance evaluation. Automated coding of collaboration from transcripts is feasible but measurably degraded by ASR errors [71], and ASR error rates differ by speaker group [77][78]; LLM coders approach human agreement on some coding frames and fail on others [73][74].
6. **Catalogue (11 competencies, 4 clusters):** communicating clearly; active listening and uptake; influencing with reasons; collaboration and mutual support; inclusive facilitation; constructive disagreement and conflict handling; feedback and developing others; coordination and keeping the team on track; collaborative problem solving and decision making; initiative and commitment to action; openness to learning and adaptability. "Leadership" is reported as a derived profile, not rated on its own, because perceived leadership tracks speaking time [42].

---

## 2. Evidence table

Legend — *Transcript?*: can the construct be observed in a diarized transcript of group talk (yes / partly / no). *Evidence*: quality of the evidence **for our use** (not for the instrument's original purpose).

### 2.1 Competency modelling and HR frameworks

| Framework / instrument | What it measures | Unit | Transcript? | Evidence quality | How HR uses it | Role in our system |
|---|---|---|---|---|---|---|
| McClelland 1973 [3]; Boyatzis 1982 [4]; Spencer & Spencer 1993 [5] (competency dictionary, Behavioural Event Interview) | Characteristics of superior performers, elicited through critical-incident style interviews | Individual | Partly (BEI codes narrated past behaviour, not live interaction) | Moderate — historically foundational; McClelland's empirical claims were strongly contested (Barrett & Depinet 1991 [6]) | Origin of "competency dictionaries" with scaled behavioural levels | Descriptive context; borrow the *format* (definition + scaled behavioural levels), not the content |
| Campion et al. 2011, best practices in competency modelling [1] | 20 best practices for analysing, organising and using competency models | Method | n/a | Strong as professional consensus (SIOP-linked author team); it is a practice review, not a validation study | Reference standard for building defensible models | **Core method source** for how the catalogue is built and documented |
| SHL Universal Competency Framework / Great Eight (Bartram 2005 [2]) | 8 broad factors → 20 dimensions → 112 components†; criterion-centric validation across 29 studies, N = 4,861 [2] | Individual | Partly (Supporting & Cooperating, Interacting & Presenting, Leading & Deciding are; Enterprising & Performing largely is not) | Strong for the criterion structure; validity evidence is for predicting manager ratings from personality/ability, not for observing behaviour | Vendor backbone for many corporate models | **Core cross-walk** (top-level taxonomy) |
| Korn Ferry Leadership Architect / Lominger [7] | 38 competencies, 12 clusters, 4 factors (Thought, Results, People, Self) | Individual | Partly | Weak-to-moderate for us — widely used, but validation is vendor-published technical manual | Very common in corporate talent management, 360 | Secondary cross-walk (naming alignment only) |
| O*NET content model (Social Skills; Work Styles) [15] | Occupational descriptors; Work Styles taxonomy was revised (21 lower-order dimensions) and enters the database from release 30.1 [15] | Occupation | No (job requirements, not person behaviour) | Strong as taxonomy, irrelevant as a rating instrument | Job analysis, skills matching | Secondary cross-walk (vocabulary) |
| ESCO transversal skills [13] | Hierarchy of transversal skills: 6 categories, 24 groups, 96 skills (revised model) | Skill label | No | Moderate as EU-standard vocabulary; no behavioural anchors | Labour-market matching, CV/skills tagging in EU | Secondary cross-walk; useful for Spanish/EU labelling |
| OECD PISA 2015 collaborative problem solving [16]; Graesser et al. 2018 [70] | 3 collaboration competencies × 4 problem-solving processes = 12 skills | Individual (in a scripted task) | Yes (framework is about communicative acts) | Strong framework; assessed with computer agents, not free conversation | Education policy; increasingly cited in skills frameworks | **Core rubric source** for collaboration constructs |
| ATC21S (Hesse et al. 2015 [17]) | Social skills (participation, perspective taking, social regulation) and cognitive skills (task regulation, knowledge building) | Individual | Yes | Moderate — well-specified indicators with developmental levels; school-age samples | Education | **Core rubric source** (indicator wording) |
| EU Key Competences / LifeComp (Sala et al. 2020 [14]) | 9 competences in 3 areas; social area = Empathy, Communication, Collaboration | Individual | Partly | Moderate — conceptual, explicitly non-prescriptive, no psychometric validation | EU education and training policy | Secondary cross-walk |
| CIPD Profession Map [11]; SHRM BASK [12] | Behaviours for HR professionals; SHRM: Relationship Management (networking, relationship building, teamwork, negotiation, conflict management), Communication (delivering messages, exchanging organisational information, listening) | Individual (HR role) | Partly | Moderate — professional-body standards; practice-analysis based | HR certification | Secondary cross-walk; useful when presenting to HR directors |
| UK Civil Service Success Profiles — Behaviours [8] | 9 behaviours × 6 grade levels, each with published indicators | Individual | Partly (Communicating and Influencing, Working Together, Making Effective Decisions, Leadership, Developing Self and Others) | Moderate — fully published, used at scale for selection; little published psychometric evidence | Recruitment and promotion across UK government | **Core cross-walk and indicator source** |
| OECD Competency Framework [9] | 15 core competencies in 3 clusters, 5 levels with behavioural indicators | Individual | Partly | Moderate — fully published; organisational framework, not validated instrument | OECD recruitment and performance | **Core cross-walk and indicator source** |
| UN Values and Behaviours Framework (2021; ST/SGB/2024/4) [10] and the 1999 "Competencies for the Future" [10b] | 4 values + 5 behaviours (e.g. Connect and Collaborate); the 1999 framework had Communication, Teamwork etc. with positive/negative indicators† | Individual | Partly | Moderate — published, with official Spanish version | UN recruitment (competency-based interviews) | Secondary cross-walk; the positive/negative indicator *format* is a model |
| OPM ECQs / leadership competencies [11b] | ECQs were replaced in 2025: now Commitment to the Rule of Law…, Driving Efficiency, Merit and Competence, Leading People, Achieving Results. The older model (Building Coalitions, interpersonal skills, oral communication, conflict management, team building)† is superseded | Individual | Partly (older model) | Weak for us — the current ECQs are no longer a soft-skills framework | US federal senior executive selection | Exclude the current ECQs; cite the pre-2025 model only as historical cross-walk |
| NIH Competencies Dictionary [11c] | Competency definitions with key behaviours and a 1–5 proficiency scale | Individual | Partly | Moderate — published; page could not be opened in this pass (HTTP 403) | NIH HR | Secondary; verify content before citing indicators |

### 2.2 Behaviour-based rating methods

| Method | What it is | Unit | Transcript? | Evidence quality | How HR uses it | Role in our system |
|---|---|---|---|---|---|---|
| Critical incident technique (Flanagan 1954 [19]) | Collect concrete effective/ineffective incidents to define performance | Method | Yes | Strong (foundational) | Job analysis; source for BARS/BOS/interview questions | **Core** — how indicators and anchors are generated and validated |
| BARS (Smith & Kendall 1963 [20]) | Scale points anchored by scaled behavioural examples ("retranslation") | Individual | Yes | Moderate — clear face validity and acceptance; psychometric superiority over other formats is not established [24] | Performance appraisal, structured interviews, AC rating forms | **Core** — format of the summary scale |
| BOS (Latham & Wexley 1977 [21]) | Frequency ratings of listed behaviours | Individual | Yes | Moderate — users prefer it; critics argue frequency ratings behave like trait judgements† | Appraisal, 360 | **Core** — format of the incident checklist (but we count, we do not recall) |
| Behavioural checklists in ACs | Tick observed behaviours per dimension before rating | Individual | Yes | Moderate — recommended to reduce assessor cognitive load [18] | Assessment centres | **Core** |
| STAR / behavioural interviewing; structured interviews [25][25b] | Past-behaviour questions with anchored scoring | Individual | No (needs an interview) | Strong for structured interviews in general (top-ranked predictor in Sackett et al. 2022 [25b]) | Selection | Exclude as a method; relevant only as evidence that anchored scoring of verbal responses works |
| Assessment centres, leaderless group discussion; 2015 Guidelines [18]; ISO 10667 [18b] | Multiple exercises, trained assessors, ORCE process, behavioural dimensions | Individual | Yes — the LGD is the closest established analogue of our setting | Strong for criterion validity of overall ratings [56]; **contested** for construct validity of dimensions (see 2.5) | Selection and development of managers and graduates | **Core method source** |
| 360 / multisource feedback [66][67] | Ratings from peers, subordinates, supervisors | Individual | No | Moderate — low inter-source agreement; small average improvement after feedback [67] | Development | Convergent-validity criterion, not a rubric source |
| Frame-of-reference rater training (Woehr & Huffcutt 1994 [22]; Roch et al. 2012 [23]) | Train raters on dimension definitions and calibrated examples | Method | n/a | Strong — two meta-analyses show improved rating accuracy | AC assessor training | **Core** — for human annotators *and* as the template for LLM prompt calibration (definitions + scored exemplars) |
| Rater-error literature (halo, leniency) [54][55] | Systematic rating distortions | Method | n/a | Strong | Appraisal design | **Core** — motivates evidence-first rating and halo checks |
| Inter-rater reliability standards [80]–[86] | ICC, Cohen's kappa, Krippendorff's alpha and conventional thresholds | Method | n/a | Strong (conventions, not laws) | Validation reporting | **Core** — see section 5 |

### 2.3 Teamwork-specific, observable-behaviour models

| Framework / instrument | What it measures | Unit | Transcript? | Evidence quality | How it is used | Role in our system |
|---|---|---|---|---|---|---|
| Bales IPA (1950) [26]; SYMLOG [26b] | 12 act categories† (socio-emotional positive, task answers, task questions, socio-emotional negative); SYMLOG adds 3 rating dimensions | Act → individual and team profiles | Yes (designed for act-by-act coding) | Moderate — foundational and reliable to code; categories are coarse and not evaluative | Research; facilitation training | Secondary: coding vocabulary and **team-level profile**; not a competency scale |
| Benne & Sheats functional roles (1948) [27] | Task roles, group building/maintenance roles, individual (self-oriented) roles | Role behaviours | Yes | Weak-to-moderate — conceptual typology from a training workshop, not empirically validated | Teaching, facilitation | Secondary: source of indicator ideas (e.g. gatekeeping, harmonising, blocking) |
| Stevens & Campion teamwork KSAs (1994) [28] | 2 categories, 5 subcategories, 14 KSAs (conflict resolution, collaborative problem solving, communication, goal setting/performance management, planning/task coordination) | Individual | Partly (it is a knowledge taxonomy; the test is a situational judgement test) | Moderate — taxonomy influential; the test has reliability problems [89] | Team staffing | **Core** taxonomy source |
| Salas, Sims & Burke "Big Five" (2005) [29] | Team leadership, mutual performance monitoring, backup behaviour, adaptability, team orientation + coordinating mechanisms (shared mental models, closed-loop communication, mutual trust) | Team/individual | Partly | Moderate — integrative review with propositions | Team training design | **Core** taxonomy source |
| Marks, Mathieu & Zaccaro (2001) [30]; LePine et al. 2008 meta-analysis [30b] | 10 processes in 3 categories: transition, action, interpersonal | Team | Partly | Strong — meta-analytically supported structure and links to effectiveness | Research standard | **Core** taxonomy source |
| Rousseau, Aubé & Savoie (2006) [31] | Integrated hierarchy of teamwork behaviours (regulation of performance; management of team maintenance) | Team/individual | Partly | Moderate — review/integration | Research | **Core** taxonomy source |
| CATME (Loughry et al. 2007 [32]; Ohland et al. 2012 [33]) | 5 dimensions: Contributing to the team's work; Interacting with teammates; Keeping the team on track; Expecting quality; Having relevant KSAs. BARS version with behavioural descriptions at high/medium/low levels | Individual (peer/self rating) | Partly (Interacting, Keeping on track, Expecting quality: yes; Contributing and KSAs need work products) | Strong for a peer-rating instrument — theory-based, large development samples (N = 2,777 and 1,157 [32]), three validity studies for the BARS [33] | Very widely used in university team projects | **Core rubric source** and **convergent-validity criterion** |
| TeamSTEPPS: T-TPQ, T-TAQ, TPOT [36] | Team structure, leadership, situation monitoring, mutual support, communication (observation tool + perception/attitude questionnaires) | Team | Partly (healthcare-specific phrasing) | Moderate — widely used; psychometrics mixed across studies | Healthcare team training | Secondary: model of an observation tool; T-TPQ as team-level criterion |
| Behavioural marker systems: NOTECHS [37], ANTS [37b], NOTSS [37c] | Non-technical skills in categories → elements → example good/poor behaviours, 4-point-style ratings with "not observed" | Individual | Partly (many markers are non-verbal or task-specific) | Strong as a *method* — developed with task analysis, reliability studies, mandatory rater training | Aviation, anaesthesia, surgery | **Core method source** (category → element → good/poor marker; "not observed" option) |
| Edmondson psychological safety (1999) [38]; meta-analysis [38b] | Shared belief that the team is safe for interpersonal risk taking (7-item survey†); learning behaviours | Team climate | No for the construct (it is a belief); partly for *learning behaviours* (asking for help, admitting error, seeking feedback) | Strong for the survey construct | Engagement/culture surveys | Team-level survey criterion; do **not** claim to measure psychological safety from transcripts — code learning behaviours instead |
| act4teams (Kauffeld & Lehmann-Willenbrock 2012 [35]; [35b]) | Utterance-level codes in four facets: problem-focused, procedural, socio-emotional, action-oriented, each with functional and dysfunctional codes | Utterance → individual/team | **Yes** (built for coding meeting talk) | Strong for our purpose — 92 real team meetings, functional vs dysfunctional statements linked to meeting satisfaction, team productivity and later organisational success [35] | Research; meeting diagnostics | **Core rubric source** for turn-level indicators |
| Woolley et al. 2010 collective intelligence [39]; Riedl et al. 2021 [40] | Group-level "c" factor; correlates with social sensitivity and equality of conversational turn-taking | Team | Yes (turn-taking distribution) | Moderate — replicated in pooled data (22 studies, 1,356 groups [40]) but the generality of "c" is disputed | Popular in management writing | **Team-level descriptor** (participation equality) only |
| Pentland sociometric work [41] | Energy, engagement, exploration from badge signals | Team | Partly (turn-taking yes; body signals no) | Weak-to-moderate — suggestive, small samples, much in practitioner outlets | Workplace analytics | Descriptive context only |
| Conversation-analytic measures: interruptions/overlap [75][76][76b], back-channels, question asking [43], uptake [72] | Low-level interaction features | Turn | Yes, but overlap and back-channels depend heavily on diarization quality | Mixed — uptake and follow-up questions have validated computational measures; "interruption" is an interpretation, not an acoustic fact [76] | Research | Features and evidence for indicators; **never scored directly** (see 4.6) |

### 2.4 Popular but weak instruments

| Instrument | What it claims | Unit | Transcript? | Evidence quality | How HR uses it | Role in our system |
|---|---|---|---|---|---|---|
| Belbin team roles [44][45][46] | 9 preferred team roles via self-perception inventory (+ observer assessment) | Individual | No (self-report preference) | Weak — ipsative scoring; early psychometric study found poor properties [44]; review of 43 studies finds adequate convergent but weak discriminant validity between several roles [45]; observer form also questioned [46] | Team-building workshops | **Exclude from scoring.** At most an optional, clearly labelled "role vocabulary" in narrative feedback — and Benne & Sheats covers the same ground with behaviour-based roles |
| Tuckman stages [48][49][50] | Forming–storming–norming–performing(–adjourning) | Group | Partly | Weak as measurement — derived from a literature review dominated by therapy/training groups; Bonebright's historical review notes its limitations; alternative models exist (e.g. punctuated equilibrium [50]) | Teaching, facilitation | **Descriptive context only**, optional, low confidence, never an individual score |
| MBTI [51][52][53] | 16 personality types | Individual | No | Weak — dichotomised types, questionable test-retest of type assignment, poor fit to the theory's own claims; the underlying scales map onto four of the Big Five [52] | Team-building, coaching | **Exclude** |
| DISC | 4 behavioural styles | Individual | No | Weak — we found no independent peer-reviewed validation literature comparable to the above in this pass (absence of evidence, stated as such) | Sales/team workshops | **Exclude** |

### 2.5 The assessment-centre construct-validity debate (both positions)

- **Position A — "dimensions are not what is measured".** Sackett & Dreher (1982) found that AC ratings factor by exercise, not by dimension [57]. Lance (2008) argued this is not a design flaw but reflects true cross-situational variability in performance, and that ACs should score exercises/tasks [58]. Meta-analytic MTMM work [59][60], generalizability studies (Putka & Hoffman 2013 [61]; Jackson et al. 2016 [62]) and Dewberry's 2024 review [62b] conclude that dimension-specific variance is very small compared with general performance and exercise-specific variance (single-digit percentages as reported in secondary summaries; exact figures not re-confirmed here).
- **Position B — "dimensions are valid when specified properly".** Arthur et al. (2003) collapsed 168 dimension labels into six (consideration/awareness of others, communication, drive, influencing others, organising and planning, problem solving) and showed dimension-level criterion validity (34 articles; a composite of four dimensions explained more variance than the overall AC rating in the earlier meta-analysis) [63]. Meriac et al. (2008) showed incremental validity over cognitive ability and personality [64]. Meriac, Hoffman & Woehr (2014) found a three-factor structure — administrative skills, relational skills, drive — fits AC data best, with weaker nomological support for relational skills [65]. Hoffman et al. (2015) show that exercises themselves carry valid variance [56].
- **What both sides agree on (and what we adopt):** behaviour is situation-specific; a small number of broad dimensions is more defensible than many narrow ones; ratings must rest on recorded behaviour; aggregation over multiple situations is required before generalising about a person.

### 2.6 Automatic / computational assessment

| Work | What was done | Relevance | Evidence quality |
|---|---|---|---|
| ETS collaborative-problem-solving coding: Flor et al. 2016 [68]; CPS-Rater, Hao et al. 2017 [69] | Supervised classification of chat turns into CPS categories; sequence models (CRF) beat turn-independent models | Direct precedent for turn-level coding; context of neighbouring turns matters | Moderate (text chat, scripted tasks) |
| Pugh et al. 2021 [71] | BERT models of 7 CPS skill facets on 8,660 spoken utterances from 44 dyads; AUROC ≈ .91 on human transcripts vs ≈ .80 on ASR transcripts; worse in noisy classrooms than in the lab | **Key quantitative evidence that ASR noise costs accuracy** | Strong for the point made |
| Demszky et al. 2021 [72] | Computational measure of conversational uptake validated against expert annotation and instruction quality | Supports "building on others' contributions" as a measurable indicator | Moderate (teacher–student talk) |
| Okada et al. 2016 [72b] | Predicting expert-rated communication skill in group discussions from dialogue acts + non-verbal features | Precedent for LGD-style scoring | Moderate (small corpus) |
| Naim et al. 2018 [72c] | Predicting interview ratings from prosodic, lexical and facial features | Shows such models largely learn fluency/engagement proxies | Moderate; mock interviews with students |
| Hickman et al. 2022 [72d] | Automated video-interview personality scores, N = 1,073; models trained on interviewer reports show some validity, models trained on self-reports show little | "Automated scores inherit the validity of the labels they are trained on" | Strong |
| Hao et al. 2024/2025 (arXiv) [73] | ChatGPT-family models coding CPS chat data; satisfactory on some frameworks/tasks, poor where talk is dense with domain terminology; newer reasoning models not necessarily better; prompt refinement helps inconsistently | Best current evidence on LLM-as-coder for collaboration | Moderate (preprint) |
| Friedman et al. 2024 (arXiv) [74] | LLM coding of negotiation transcripts with a coding manual | Second domain showing near-human agreement is attainable with a good codebook | Moderate (preprint) |
| Tay et al. 2022 [79]; Landers & Behrend 2023 [79b] | Frameworks for machine-learning measurement bias and for auditing AI assessment | The audit template for section 5 | Strong (conceptual) |
| Raghavan et al. 2020 [79c] | Review of vendors' algorithmic-hiring claims and bias-mitigation practices | Context for vendor critiques (e.g. video-interview scoring) | Moderate |

Known validity and bias problems (each needs a check in section 5):

- **ASR error disparities.** Commercial ASR word error rate was about .35 for Black speakers vs .19 for white speakers in Koenecke et al. 2020 [77]. For Spanish, accent- and gender-related differences have been reported for Whisper [78]; other work reports variation across Spanish varieties (not opened in full here).
- **Diarization errors** misattribute turns; an indicator credited to the wrong speaker is worse than a missed one. No single reference is cited; this must be measured on our own data.
- **Interruption is an attribution, not a signal.** Meta-analysis shows only a small-to-moderate gender difference in intrusive interruptions (d = .33 across 43 studies) [75]; listeners judge the same overlap differently, and male listeners rated interrupting women more harshly in a large perception experiment [76]. Overlap is also a normal feature of turn-taking [76b].
- **Talk quantity confound.** Speaking time predicts being seen as a leader independently of intelligence and personality [42]; women speak less in some group compositions and decision rules [43b]. A frequency-driven rubric would reward dominance.
- **Language-proficiency confound.** Fluency, vocabulary and grammatical accuracy of non-native speakers leak into "communication" ratings; LLM raters and ASR both amplify this. No single definitive study is cited; treat as an open risk to be tested.
- **Halo.** Human raters produce correlated dimension ratings [54][55]; an LLM that rates all competencies in one pass will too.

Legal pointers (not developed): EU AI Act (Regulation (EU) 2024/1689) Annex III point 4 — employment, worker management, performance/behaviour monitoring are high-risk; Article 5(1)(f) prohibits inferring emotions in workplace and education settings from biometric data [L1]. NYC Local Law 144 (automated employment decision tools; bias audit) [L2]. US EEOC Uniform Guidelines on Employee Selection Procedures, 29 CFR 1607, four-fifths rule [L3]. GDPR Art. 22 and Art. 9 (voice as biometric data when used for identification) [L4].

---

## 3. Recommended competency catalogue

### 3.1 Selection criteria and structure

A competency is included only if it (a) appears in several mainstream HR frameworks, (b) is observable in spoken group interaction, and (c) has published behavioural indicators in at least one source. The eleven competencies are grouped in four clusters that mirror the broad factors recovered in assessment-centre research (communication and influencing others [63]; relational skills, administrative skills, drive [65]).

| Cluster | Competencies |
|---|---|
| I. Communication and influence | C1 Communicating clearly · C2 Active listening and uptake · C3 Influencing with reasons |
| II. Relational | C4 Collaboration and mutual support · C5 Inclusive facilitation · C6 Constructive disagreement and conflict handling · C7 Feedback and developing others |
| III. Task and coordination | C8 Coordination and keeping the team on track · C9 Collaborative problem solving and decision making |
| IV. Drive and self-management | C10 Initiative and commitment to action · C11 Openness to learning and adaptability |

Cross-walk naming note: dimension names from the Great Eight, UK Success Profiles, OECD, UN VBF, SHRM BASK, LifeComp, PISA, ATC21S, CATME, Marks et al., Salas et al., Stevens & Campion and act4teams facets were confirmed in this pass. Names marked † (Korn Ferry competencies, O*NET skills, individual act4teams codes, Bales categories, Alles dictionary entries) are from memory and must be checked against the source before publication.

For every competency the indicators below are **our paraphrased synthesis**, phrased as verbal behaviour that can be quoted from a transcript. They are candidates for expert review (section 5), not validated items.

### C1. Communicating clearly

- **Definition.** Expresses ideas, information and reasoning so that others in the group can understand and use them, adapting to the audience.
- **Cross-walk.** Great Eight: *Interacting and Presenting* [2] · UK: *Communicating and Influencing* [8] · OECD: (oral side of) *Influencing*; *Drafting Skills* is the written counterpart [9] · SHRM: *Communication — delivering messages* [12] · LifeComp: *S2 Communication* [14] · AC meta-dimension: *Communication* [63] · Salas: *closed-loop communication* [29] · Korn Ferry: *Communicates effectively*† [7] · O*NET: *Speaking*† [15].
- **Positive indicators.** (1) States the point first, then supports it. (2) Structures a longer contribution explicitly (signposting, enumerating). (3) Explains a term or gives an example when others signal confusion. (4) Summarises own position concisely when asked. (5) Checks that the message landed ("does that make sense?", asks someone to restate). (6) Closes the loop on requests (confirms what was heard and what will be done).
- **Negative indicators.** (1) Long, unstructured turns after which others ask what was meant. (2) Uses unexplained jargon and does not repair when asked. (3) Repeatedly restarts or contradicts own statements without resolving them. (4) Leaves direct questions unanswered.
- **Utterance patterns.** "There are two options: first… second…"; "In short, what I'm proposing is…"; "Let me put that differently…"; "So you need X by Friday — I'll send it Thursday."
- **Not inferable from a transcript.** Vocal delivery, pace and prosody (lost in text; and inferring affect from voice is legally sensitive); visual aids; written communication; whether the content was *correct*. Disfluency counts must not be used: they confound with second-language status and ASR errors.

### C2. Active listening and uptake

- **Definition.** Attends to what others say and demonstrably uses it: paraphrases, asks follow-up questions, and builds on others' contributions.
- **Cross-walk.** UK: *Communicating and Influencing* ("listen and ask questions") and *Working Together* ("listen genuinely") [8] · OECD: *Diplomatic Sensitivity* (level 1: listens actively) [9] · SHRM: *Communication — listening* [12] · LifeComp: *S1 Empathy*, *S2 Communication* [14] · ATC21S: *Perspective taking* (responsiveness, audience awareness)† [17] · PISA: *Establishing and maintaining shared understanding* [16] · CATME: *Interacting with teammates* [33] · Great Eight: *Supporting and Cooperating* [2] · Korn Ferry: *Communicates effectively*† · O*NET: *Active Listening*† · computational: conversational uptake [72], follow-up questions [43].
- **Positive indicators.** (1) Paraphrases or summarises another speaker before responding. (2) Asks a follow-up question that depends on the content of the previous turn. (3) Explicitly builds on a named colleague's idea. (4) Refers back to something said earlier by someone else. (5) Asks a clarifying question instead of assuming. (6) Corrects own understanding after an answer.
- **Negative indicators.** (1) Responds with an unrelated point after a direct contribution (topic hijack). (2) Repeats a proposal someone else already made without acknowledging it. (3) Asks for information that was just given. (4) Cuts in to finish others' sentences with a different meaning, followed by the original speaker's repair.
- **Utterance patterns.** "If I understood you, you're saying…"; "Building on what Ana said…"; "What did you mean by…?"; "Going back to Luis's point about…".
- **Not inferable.** Silent attention, eye contact, note-taking; whether understanding was real. Back-channels ("mm-hm", "sí, sí") are often deleted by ASR and vary culturally — do not score their presence or absence.

### C3. Influencing with reasons

- **Definition.** Gains others' agreement through evidence, argument and appeal to shared goals, while remaining open to counter-argument.
- **Cross-walk.** Great Eight: *Interacting and Presenting* (persuading and influencing component†) and *Leading and Deciding* [2] · UK: *Communicating and Influencing* [8] · OECD: *Influencing*; *Negotiating* [9] · SHRM: *Relationship Management — negotiation* [12] · AC meta-dimension: *Influencing others* [63] · Korn Ferry: *Persuades*† · O*NET: *Persuasion*, *Negotiation*† · act4teams: problem-focused statements (naming solutions with reasons)† [35].
- **Positive indicators.** (1) Gives a reason, datum or example with a proposal. (2) Links the proposal to a goal the group has stated. (3) Anticipates and addresses an objection. (4) Adjusts the proposal after a counter-argument and says so. (5) Proposes a trade-off or package that incorporates others' interests. (6) Asks what would be needed for others to agree.
- **Negative indicators.** (1) Repeats the same assertion without new support. (2) Appeals to status or seniority instead of reasons. (3) Pushes for closure while objections are unanswered. (4) Dismisses alternatives without engaging with their content.
- **Utterance patterns.** "I'd go with B because…"; "Given we agreed the priority is X…"; "You're right about the cost; what if we…"; "What would convince you?".
- **Not inferable.** Whether influence actually changed minds beyond the meeting; side-channel lobbying; the quality of evidence cited (needs domain knowledge); whether the eventual group choice was good. **Do not use "proposal adopted by the group" as a positive indicator** — it rewards status and talk time [42].

### C4. Collaboration and mutual support

- **Definition.** Works toward shared goals: offers and asks for help, shares information, credits others, and puts the team result ahead of personal position.
- **Cross-walk.** Great Eight: *Supporting and Cooperating* [2] · UK: *Working Together* [8] · OECD: *Teamwork and Team Leadership* (e.g. shares relevant information, seeks others' input, invites and builds on ideas) [9] · UN: *Connect and Collaborate* [10]; 1999 *Teamwork*† [10b] · SHRM: *Relationship Management — teamwork* [12] · LifeComp: *S3 Collaboration* [14] · Salas: *backup behaviour*, *team orientation* [29] · Marks: *team monitoring and backup*; *motivating and confidence building* [30] · Rousseau: collaborative behaviours (cooperation, information exchange)† [31] · CATME: *Interacting with teammates*; *Contributing to the team's work* [33] · Stevens & Campion: interpersonal KSAs [28] · AC meta-dimension: *Consideration/awareness of others* [63] · Korn Ferry: *Collaborates*† · O*NET: *Coordination*, *Service Orientation*; work style *Cooperation*† · Bales: shows solidarity / agrees† [26].
- **Positive indicators.** (1) Offers concrete help with a teammate's task. (2) Asks for help or input when stuck. (3) Shares information unprompted that another member needs. (4) Credits a named colleague for an idea or work. (5) Uses collective framing for goals and problems ("what do *we* need"). (6) Volunteers for an unglamorous shared task.
- **Negative indicators.** (1) Withholds or deflects when asked for information ("that's my part, don't worry"). (2) Disparages a teammate's contribution or competence. (3) Claims shared work as own. (4) Refuses a reasonable request for help without explanation.
- **Utterance patterns.** "I can take that off your plate"; "Could someone look at this with me?"; "That was Marta's idea"; "You'll need the figures I got yesterday — I'll share them."
- **Not inferable.** Whether offered help was delivered; quality and quantity of actual work contributed (CATME's *Contributing* and *Having relevant KSAs* need work products and peers [33]); trust; behaviour outside recorded sessions.

### C5. Inclusive facilitation

- **Definition.** Helps everyone contribute: invites quieter members, balances airtime, protects speakers from being talked over, and makes room for different viewpoints.
- **Cross-walk.** UK: *Working Together* (inclusive environment, seek diverse input) and *Leadership* (inclusion) [8] · OECD: *Teamwork and Team Leadership* (encourages people with opposing viewpoints to express concerns; encourages and values input) [9] · UN: value *Inclusion*; *Connect and Collaborate* [10] · ATC21S: *Participation*; *Social regulation* [17] · PISA: *Establishing and maintaining team organisation* [16] · Benne & Sheats: gatekeeper/expediter, encourager (maintenance roles)† [27] · Salas: *team leadership* [29] · Korn Ferry: *Values differences*, *Drives engagement*† · team-level link: equality of turn-taking [39].
- **Positive indicators.** (1) Directly invites a member who has not spoken on the topic. (2) Returns the floor to someone who was cut off ("let her finish"). (3) Asks for dissenting or missing views before closing a topic. (4) Checks the whole group for agreement instead of assuming it. (5) Limits own turn to make room ("I've talked a lot — others?"). (6) Rephrases a hesitant contribution so the group considers it.
- **Negative indicators.** (1) Talks over or resumes own point when another member begins, repeatedly, with the other speaker abandoning the turn. (2) Addresses questions only to the same one or two members. (3) Declares consensus when others have not spoken. (4) Dismisses a contribution with reference to who said it.
- **Utterance patterns.** "Carlos, we haven't heard from you on this"; "Hang on, Lucía was saying…"; "Does anyone see it differently?"; "Is everyone OK with that, or just the three of us?".
- **Not inferable.** Intent behind overlaps (cooperative vs intrusive overlap [76][76b]); gaze and gesture used to invite; whether a member's silence is exclusion or choice; chat/side channels. Negative indicator (1) requires reliable overlap timing and diarization — mark as "low-confidence evidence" when either is poor.

### C6. Constructive disagreement and conflict handling

- **Definition.** Raises and handles disagreement about tasks and ideas openly and respectfully, separates issues from persons, and works toward resolution.
- **Cross-walk.** Stevens & Campion: *conflict resolution KSAs* [28] · Marks: *conflict management*; *affect management* [30] · Rousseau: integration/conflict management (team maintenance)† [31] · OECD: *Teamwork and Team Leadership* (resolves conflict sensitively and fairly), *Diplomatic Sensitivity*, *Negotiating* [9] · UK: *Communicating and Influencing* (handle difficult conversations), *Leadership* (manage conflict), *Making Effective Decisions* (invite challenge) [8] · SHRM: *Relationship Management — conflict management* [12] · LifeComp: *S3 Collaboration*, *P1 Self-regulation* [14] · ATC21S: *Social regulation* (negotiation) [17] · Korn Ferry: *Manages conflict*† · act4teams: socio-emotional facet, including dysfunctional criticising/complaining† [35] · Bales: disagrees vs shows antagonism† [26] · Benne & Sheats: harmoniser, compromiser vs aggressor, blocker† [27].
- **Positive indicators.** (1) States disagreement explicitly with a reason. (2) Acknowledges the valid part of the other view before differing. (3) Names the disagreement and proposes a way to resolve it (criteria, data, test, vote). (4) Reframes a personal remark as a task issue. (5) Proposes a compromise or integrative option. (6) De-escalates (apologises, takes back wording) after tension.
- **Negative indicators.** (1) Personal criticism or sarcasm aimed at a member. (2) Blanket rejection without reasons ("that will never work"). (3) Repeated complaining without proposals. (4) Withdraws after disagreement ("fine, do whatever you want") — code only when explicit.
- **Utterance patterns.** "I see it differently, because…"; "I agree on the goal, not on the timing"; "We keep going round on this — can we agree how to decide?"; "Sorry, that came out harsher than I meant."
- **Not inferable.** Emotional state, tone and sarcasm are unreliable from text (and emotion inference from voice at work is legally restricted [L1]); unexpressed disagreement; conflict history outside the recording. A session without disagreement provides **no evidence** for this competency → "insufficient evidence", not a high score.

### C7. Feedback and developing others

- **Definition.** Gives specific, usable feedback, asks for feedback on own work, explains and teaches so that others can do the task.
- **Cross-walk.** UK: *Developing Self and Others* [8] · OECD: *Developing Talent* [9] · UN: *Learn and Develop* [10] · CATME: *Keeping the team on track* (gives teammates specific, timely, constructive feedback†) and *Expecting quality* [33] · Salas: *mutual performance monitoring* [29] · Marks: *team monitoring and backup* (includes feedback/coaching) [30] · Great Eight: *Supporting and Cooperating*; *Leading and Deciding* [2] · Korn Ferry: *Develops talent*† · O*NET: *Instructing*† [15] · Edmondson learning behaviours (seeking feedback) [38].
- **Positive indicators.** (1) Gives feedback that names the specific behaviour or piece of work and its effect. (2) Pairs criticism with a concrete suggestion. (3) Asks for feedback on own contribution. (4) Explains *how* or *why*, not only *what*, when a teammate lacks knowledge. (5) Checks whether the explanation helped. (6) Recognises improvement or good work specifically.
- **Negative indicators.** (1) Global judgement without specifics ("this is bad"). (2) Takes over a teammate's task instead of explaining, after a request for help to learn. (3) Rejects or argues with feedback received without considering it. (4) Feedback delivered as a put-down in front of the group.
- **Utterance patterns.** "The intro works; the second section loses me at… maybe move…"; "Can you tell me what you'd change in my part?"; "The reason we do it this way is…".
- **Not inferable.** Accuracy of the feedback; whether the recipient developed; private 1:1 feedback; written review comments. Many working sessions contain no feedback episode → expect frequent "insufficient evidence".

### C8. Coordination and keeping the team on track

- **Definition.** Structures the group's work: clarifies goals, proposes process, allocates tasks and time, monitors progress and summarises.
- **Cross-walk.** Marks: transition processes (*mission analysis, goal specification, strategy formulation and planning*) and action processes (*monitoring progress toward goals, coordination*) [30] · Rousseau: preparation of work accomplishment; work assessment behaviours† [31] · Stevens & Campion: *goal setting and performance management*; *planning and task coordination* [28] · CATME: *Keeping the team on track* [33] · act4teams: *procedural* statements (goal orientation, clarifying, procedural suggestions, time management, summarising; dysfunctional: losing the train of thought)† [35] · Great Eight: *Organizing and Executing* [2] · AC meta-dimension: *Organising and planning* [63] · UK: *Delivering at Pace*; *Leadership* (clear objectives and timeframes) [8] · OECD: *Teamwork and Team Leadership* (clear directions and priorities), *Managing Resources* [9] · UN: *Analyse and Plan* [10] · PISA: *Establishing and maintaining team organisation* [16] · ATC21S: *Task regulation* [17] · Korn Ferry: *Plans and aligns*, *Directs work*† · O*NET: *Coordination*, *Time Management*†.
- **Positive indicators.** (1) States or asks for the goal/output of the session. (2) Proposes an agenda, order or method. (3) Proposes or negotiates who does what by when. (4) Notes time and redirects ("ten minutes left; decide X first"). (5) Summarises decisions and open points. (6) Brings the discussion back after a digression.
- **Negative indicators.** (1) Introduces unrelated topics repeatedly while a task is open. (2) Reopens settled decisions without new information. (3) Leaves task ownership ambiguous when asked ("someone will do it"). (4) Dictates assignments without checking capacity or agreement.
- **Utterance patterns.** "What do we need to have decided by the end?"; "Let's do requirements first, then split tasks"; "So: Pablo the draft by Wednesday, I review Thursday"; "We're drifting — back to the budget."
- **Not inferable.** Whether plans were realistic or executed; coordination done in tools (boards, chat); who holds a formal role (must come from metadata — a designated facilitator showing these behaviours is fulfilling a role; say so in the report).

### C9. Collaborative problem solving and decision making

- **Definition.** Contributes to analysing problems and reaching decisions with the group: defines the problem, shares and requests relevant information, generates and evaluates options, and makes reasoning explicit.
- **Cross-walk.** PISA: *Taking appropriate action to solve the problem*; *Establishing and maintaining shared understanding* [16] · ATC21S: *Task regulation*; *Knowledge building* [17] · Stevens & Campion: *collaborative problem solving KSAs* [28] · act4teams: *problem-focused* statements (differentiating a problem, cross-linking, differentiating/cross-linking a solution)† [35] · UK: *Making Effective Decisions* [8] · OECD: *Analytical Thinking*; *Flexible Thinking* [9] · UN: *Analyse and Plan* [10] · Great Eight: *Analyzing and Interpreting*; *Leading and Deciding* [2] · AC meta-dimension: *Problem solving* [63] · Korn Ferry: *Decision quality*† · O*NET: *Complex Problem Solving*, *Judgment and Decision Making*† · Bales: gives/asks for orientation, opinion, suggestion† [26].
- **Positive indicators.** (1) Defines or reframes the problem; separates symptom from cause. (2) Asks for facts or assumptions before judging an option. (3) Offers more than one option, or asks for alternatives. (4) States criteria and weighs options against them (costs, risks, users). (5) Makes the reasoning chain explicit ("if… then…"). (6) Checks the decision against the goal and names remaining risks.
- **Negative indicators.** (1) Jumps to a solution and resists examining the problem. (2) Asserts conclusions with no stated basis when challenged. (3) Ignores disconfirming information raised by others. (4) Pushes to decide without stating what is being decided.
- **Utterance patterns.** "What exactly is the problem we're solving?"; "Do we know that, or are we assuming it?"; "Option A is cheaper but riskier because…"; "What would have to be true for this to fail?".
- **Not inferable.** Correctness of the analysis and **technical expertise** (needs domain ground truth); thinking done silently or in writing; decision quality (needs outcomes). The rubric rates the *process contribution*, not being right.

### C10. Initiative and commitment to action

- **Definition.** Moves the work forward: volunteers, proposes next steps, commits to concrete actions and reports on earlier commitments.
- **Cross-walk.** act4teams: *action-oriented* statements (expressing interest in change, taking responsibility, action planning; dysfunctional: no interest in change, complaining, denying responsibility)† [35] · CATME: *Contributing to the team's work*; *Expecting quality* [33] · AC factor: *Drive* [63][65] · Great Eight: *Enterprising and Performing*; *Organizing and Executing* [2] · UK: *Delivering at Pace*; *Leadership* (ownership) [8] · OECD: *Achievement Focus* [9] · UN: *Deliver Results with Positive Impact* [10] · Marks: *motivating and confidence building* [30] · Korn Ferry: *Action oriented*, *Ensures accountability*† · O*NET work styles: *Initiative*, *Dependability*†.
- **Positive indicators.** (1) Volunteers for a task with a concrete deliverable or date. (2) Proposes a next step when the group stalls. (3) Reports status of an earlier commitment without being asked. (4) Admits a missed commitment and re-plans. (5) Raises the quality bar with a specific standard ("let's test it with two users first"). (6) Expresses confidence in the team's ability to deliver, tied to a plan.
- **Negative indicators.** (1) Deflects responsibility ("not my job", "nobody told me"). (2) Vague non-commitment when asked directly ("I'll see"). (3) Pessimism about any change without alternatives ("there's no point"). (4) Repeated unexplained non-delivery acknowledged in the session.
- **Utterance patterns.** "I'll take it — you'll have it Monday"; "Next step: I'll call the client today"; "I didn't finish the analysis; I'll have it tomorrow morning".
- **Not inferable.** **Follow-through itself** — a single transcript shows commitments, not delivery. Delivery is observable only by linking sessions (commitment in session *n*, status in session *n+1*). Effort, working hours, quality of output: not observable.

### C11. Openness to learning and adaptability

- **Definition.** Shows willingness to revise views and approach: admits gaps and mistakes, asks for help, incorporates new information and adjusts to changed constraints.
- **Cross-walk.** Edmondson: team *learning behaviours* (asking for help, admitting errors, seeking feedback, discussing mistakes) [38] · Salas: *adaptability* [29] · Great Eight: *Adapting and Coping* [2] · OECD: *Flexible Thinking* [9] · UK: *Changing and Improving*; *Developing Self and Others* (act on feedback) [8] · UN: *Adapt and Innovate*; *Learn and Develop*; value *Humility* [10] · LifeComp: *P2 Flexibility*; *L1 Growth mindset* [14] · ATC21S: *Social regulation* (self-evaluation) [17] · Korn Ferry: *Situational adaptability*, *Demonstrates self-awareness*† · O*NET work styles: *Adaptability/Flexibility*†.
- **Positive indicators.** (1) Says "I don't know" or "I was wrong" in plain terms. (2) Changes position after an argument or new information and says why. (3) Asks how someone else would approach it. (4) Proposes adjusting the plan when a constraint changes. (5) Reflects on what the team could do differently next time. (6) Accepts feedback and states what will change.
- **Negative indicators.** (1) Defends a position after it has been factually corrected. (2) Blames others or circumstances for an error raised in the session. (3) Rejects a changed constraint without proposing how to cope ("we've always done it this way"). (4) Dismisses a retrospective or feedback discussion as a waste of time.
- **Utterance patterns.** "Good point, I hadn't considered that — then I'd change…"; "My mistake: I used last year's data"; "How would you do it?"; "Next time we should start with…".
- **Not inferable.** Actual learning or behaviour change (needs later sessions); private reflection; emotional resilience/stress tolerance (do not infer). Admitting error depends on team climate: low frequency may reflect low psychological safety in the *team*, not the individual — interpret together with team-level descriptors.

### 3.2 Derived profile: leadership (not rated separately)

Leadership appears in every framework (Great Eight *Leading and Deciding*; UK *Leadership*; OECD *Teamwork and Team Leadership*; Salas *team leadership*). In peer groups it is emergent and shared, and perceptions of it are driven by speaking time [42]. Recommendation: report a "leadership behaviours profile" composed from evidence already coded under C5 (inclusive facilitation), C8 (coordination), C3 (influencing), C7 (developing others) and C10 (initiative), without a separate numeric score.

### 3.3 Competencies we must not claim to measure from a transcript

| Not measured | Why |
|---|---|
| Integrity, honesty, ethics | Require knowledge of facts and of behaviour when unobserved; transcript shows only claims. UK, UN and OPM frameworks treat these as values assessed by other means. |
| Technical/professional expertise; correctness of ideas | Needs domain ground truth and work products (CATME *Having relevant KSAs* is a peer judgement [33]). |
| Reliability, work quality, actual contribution, follow-through | Occur outside the conversation (CATME *Contributing to the team's work*). Only cross-session commitment tracking gives partial evidence. |
| Results orientation / delivering results | Outcome constructs (Great Eight *Enterprising and Performing*; UN *Deliver Results*). |
| Strategic thinking, vision, organisational knowledge | Not sampled in peer work sessions; inference would rest on vocabulary. |
| Emotional intelligence, empathy as a trait, emotional state, stress tolerance, resilience | Internal states; text is a poor proxy; voice-based emotion inference in workplace/education is prohibited under the EU AI Act [L1]. We code *expressed* acknowledgement behaviours only (C2, C4, C6). |
| Psychological safety, trust, cohesion | Shared beliefs measured by team survey [38]; the transcript shows behaviours that co-vary with them, not the construct. |
| Personality traits, "types", Belbin roles, motivation, values | Dispositional constructs; automated personality inference has weak validity [72d]. |
| Creativity/innovation | Idea novelty needs domain judgement; we can count idea generation as part of C9 only. |
| Written communication, presentation delivery, non-verbal communication | Not in an audio transcript. |
| Cognitive ability | Not a behavioural competency; verbal fluency is a biased proxy. |

---

## 4. Rating method recommendation

### 4.1 Process: ORCE, enforced by the pipeline architecture

The assessment-centre process separates four steps [18]; each maps to a pipeline stage, and the separation is what makes the output auditable.

1. **Observe** — the diarized transcript (with timestamps, speaker labels, ASR confidence).
2. **Record** — extract candidate behavioural incidents per speaker as verbatim quotes with turn IDs. No evaluation at this step. A programmatic check must confirm every quote exists verbatim in the transcript and is attributed to the right speaker.
3. **Classify** — assign each incident to at most one primary competency and one indicator (positive or negative), using the rubric retrieved for that indicator. Unclassifiable incidents are dropped, not forced.
4. **Evaluate** — only after classification, apply the anchored scale per competency using the classified incidents as the sole input (the evaluator stage should not see the raw transcript again, and should rate one competency at a time, to limit halo [54][55]).

### 4.2 Scale: two stages

- **Stage 1 — behavioural incident checklist (BOS-like, but counted, not recalled).** For each indicator: number of independent incidents, each with quote, turn ID, polarity, and a quality flag (basic / clear / exemplary). Unlike a human-completed BOS, frequencies are counted from the record rather than estimated from memory, which removes the main criticism of BOS.
- **Stage 2 — 5-point BARS-style summary rating per competency per session, plus "insufficient evidence" (IE).** Anchors are written at 1, 3 and 5; 2 and 4 are "between" points — the same structure as the CATME BARS† [33] and comparable to the 1–5 proficiency scales in public frameworks (e.g. NIH [11c]).

| Point | Generic anchor (each competency gets its own behavioural version built from its indicators) |
|---|---|
| 5 | Several different positive indicators shown, at clear/exemplary quality, across different phases of the session; visible effect on the group's interaction (others take up, respond, continue); no or isolated minor negative indicators. |
| 4 | Between 3 and 5. |
| 3 | Positive indicators present but narrow (one or two indicator types), or basic quality, or mixed with some negative indicators; little visible effect on the group. |
| 2 | Between 1 and 3. |
| 1 | Negative indicators predominate and recur; positive indicators absent or isolated. |
| IE | Fewer than the minimum number of independent incidents, or the session offered no opportunity (e.g. no disagreement arose for C6). **Not a score; never converted to a number.** |

Why five points: it matches CATME and HR practice, raters (human or model) do not reliably discriminate more levels on behavioural dimensions, and "between" points avoid forcing. Why BARS for the summary and BOS for the checklist: no rating format has shown clear psychometric superiority [24]; the hybrid gives the *traceability* of counted behaviours and the *interpretability* of anchors, which is what an I/O psychologist will ask for.

Anchor construction should follow a shortened Smith & Kendall retranslation [20]: subject-matter experts (a) generate incidents, (b) independently re-assign each incident to a competency — drop those with low agreement, (c) scale the survivors for effectiveness — keep those with low rating variance as anchors.

### 4.3 Minimum-evidence rules

These are design rules, not thresholds taken from the literature; they should be tuned on the gold set.

- **Independent incident** = a distinct episode. Consecutive turns in the same exchange count once; the same sentence repeated counts once.
- **Rating requires ≥ 3 independent incidents** (positive or negative) for that competency in the session, from **≥ 2 different episodes** (e.g. separated by a topic change or by several minutes). Otherwise IE.
- **A 5 requires ≥ 2 different positive indicator types**; a 1 requires ≥ 2 negative incidents.
- **Opportunity gating** for C6 and C7: if no disagreement / no feedback episode occurred in the session for anyone, the competency is IE for all participants.
- **Low-confidence evidence** (ASR confidence below threshold, overlapped speech, uncertain diarization) cannot be the only support for a rating.
- **Minimum participation floor for any report**: below a set amount of speech (e.g. a few minutes or a few dozen turns — to be set empirically), the participant gets a participation descriptor only.

### 4.4 Frequency versus quality

- Frequency must be **normalised by opportunity** (per 100 own turns or per speaking minute), then **capped**: beyond the minimum, more instances do not raise the rating by themselves. This is the main defence against rewarding talkativeness [42].
- Quality (specificity, responsiveness to the preceding turn, effect on the group) differentiates 3 from 5.
- Negative indicators are tallied separately and are not cancelled out by positive ones; the report shows both.
- Low participation yields IE and a neutral participation descriptor. **Silence is never scored as a deficit.**

### 4.5 Aggregation across sessions

Because behaviour is situation-specific (section 2.5), a single session supports only a statement about that session.

- Always store and show **session-level** ratings with their evidence.
- A **person-level profile** requires rated (non-IE) evidence from at least 3 sessions for that competency, ideally with different tasks or group compositions. Report the median and the range, not a single mean with false precision; flag high variability across sessions as a finding in itself.
- Estimate reliability of the aggregate with a generalizability study (persons × sessions × raters) once data exist [61][62]; use it to state how many sessions are needed for a dependable person-level score.
- Prefer cluster-level summaries (the four clusters) when competency-level evidence is thin; AC research supports broad factors better than narrow ones [65].
- Track C10 commitments across sessions (commitment → status) as a separate, factual "commitments log".

### 4.6 Team-level descriptors (context only; never individual scores)

- **Participation equality**: share of speaking time and of turns per member; a dispersion index (e.g. Gini or normalised entropy). Basis: turn-taking equality correlates with collective intelligence [39][40]. Present as a team property with the caveat that roles and task type legitimately create inequality.
- **Interaction profile**: proportion of turns in Bales-style macro-categories (task: giving vs asking; socio-emotional: positive vs negative) [26], and/or act4teams facets with functional vs dysfunctional split [35]. Kauffeld & Lehmann-Willenbrock found dysfunctional statements related more strongly (negatively) to outcomes than functional ones related positively [35].
- **Interaction features**: question rate, uptake rate [72], overlap rate. Overlap is reported descriptively at team level with an explicit note that overlap ≠ interruption [76][76b].
- **Group development stage (Tuckman)**: optional narrative label, low confidence, with the caveat that the model is descriptive and not all groups follow it [48][49][50]. Never used in scoring.
- **Psychological safety, team processes**: only via short team surveys (Edmondson scale [38]; team-process survey [30c]), shown next to — not derived from — transcript analytics.

### 4.7 Reporting rules

Every rating shows: competency definition, session, the classified incidents (quotes + turn IDs), the anchor matched, IE where applicable, and a standing notice of what cannot be inferred. Reports are developmental, addressed to the participant first, and reviewed by a human facilitator before any use beyond self-development (consistent with the AC guidelines' requirements on feedback and assessor competence [18]).

---

## 5. Validation plan for our tool

### 5.1 Human-annotated gold set

- **Material.** Recorded sessions spanning task types (planning, problem solving, retrospectives), group sizes (3–6), and speaker groups (gender, native/non-native, regional variety). Keep both human-corrected and raw ASR transcripts for every session.
- **Annotators.** Minimum 3 trained annotators on a fully crossed reliability subset (all three code the same material); 2 annotators plus adjudication on the remainder. At least one annotator with I/O psychology or assessor experience to produce expert "target scores".
- **Frame-of-reference training** [22][23]: (1) teach definitions, indicators and anchors; (2) practise on transcripts; (3) compare with expert target scores and discuss discrepancies; (4) certification round — an annotator starts production only after reaching the agreement thresholds below; (5) drift checks: re-insert calibrated transcripts periodically.
- **Two annotation layers**, mirroring the pipeline: (a) turn-level incident coding (competency, indicator, polarity); (b) session-level 1–5/IE rating per participant per competency.
- **Agreement statistics and thresholds** (conventions; report confidence intervals, not just point estimates [80]):
  - Turn-level nominal codes: **Krippendorff's alpha** (handles >2 coders and missing data) and Cohen's kappa per pair. Target α ≥ .80; .667–.80 acceptable for tentative use; below .667 revise the indicator or merge categories [82]. Landis & Koch's kappa labels (.61–.80 "substantial", > .80 "almost perfect")† are more lenient and should be cited only as secondary [83]. Report per-indicator agreement — rare codes usually fail first.
  - Unitising (did both coders mark an incident at all?): report separately (precision/recall against adjudicated set), because chance-corrected coefficients assume a fixed set of units.
  - Session-level ratings: **ICC(2,1)** (two-way random, absolute agreement, single rater) and **ICC(2,k)** for the averaged gold label [81][84]. Target ICC(2,k) ≥ .75 ("good" under Koo & Li [80]; "excellent" under Cicchetti† [85]) and ICC(2,1) ≥ .60. For context, single-rater reliability of supervisor performance ratings is only about .5† [86] and structured-interview inter-rater reliability is higher† [87]; do not demand of the model what trained humans do not achieve — the fair benchmark is **model–human agreement vs human–human agreement on the same items**.
  - IE decisions: agreement on rated-vs-IE as a binary kappa.
  - Within-group agreement indices (rwg) are not a substitute for ICC here [84].
- **Sample-size reasoning.**
  - Session-level ICC: using Bonett's approximation [88], estimating an ICC near .70 with a 95% CI of total width .20 needs roughly 65–70 participant-sessions with 3 raters and about 100 with 2 raters (our calculation from the formula as recalled†; re-derive before pre-registration). Participants are nested in sessions, so plan for **≥ 30 sessions / ≥ 120 participant-sessions** in the reliability subset and use a mixed model or G-study rather than treating rows as independent.
  - Turn-level coding: precision is driven by the rarest indicator. Aim for ≥ 50 adjudicated positive instances per indicator; oversample sessions likely to contain rare behaviours (conflict, feedback). As an order of magnitude, Pugh et al. coded 8,660 utterances for seven facets [71].
  - Hold out a test split by **session and by team** (no participant in both tuning and test sets).

### 5.2 Model evaluation against the gold set

- Turn-level: per-indicator precision/recall/F1 and chance-corrected agreement of the model treated as an additional coder; compare with human–human.
- Session-level: ICC and weighted kappa of model vs averaged human rating; confusion on IE.
- **Evidence faithfulness**: 100% of quoted evidence must match the transcript verbatim and the right speaker (automated check); human audit of whether quotes actually instantiate the indicator.
- **Stability**: repeat runs (same input, different seeds/temperature; prompt paraphrases; model version changes) → report run-to-run ICC. Hao et al. show results vary by model and that newer models are not automatically better [73].
- **Robustness**: human transcript vs ASR transcript (expect a drop [71]); simulated diarization errors; shuffled speaker names; session truncated.
- **Halo check**: inter-competency correlations from the model vs from humans; a model with much higher inter-correlations is rating general impression.
- **Talk-time check**: correlation of each rating with speaking time; partial out speaking time and confirm the model still agrees with humans.

### 5.3 Validity evidence

- **Content validity**: SME panel (I/O psychologists, HR practitioners) rates each indicator for relevance and observability; retranslation agreement per indicator (section 4.2); document the cross-walk (Campion et al. practices on linking competencies to evidence [1]).
- **Convergent validity**: **CATME peer ratings** collected for the same teams [33] — predicted pattern: C2/C4/C5/C6 ↔ *Interacting with teammates*; C8/C7 ↔ *Keeping the team on track*; C10 ↔ *Contributing* and *Expecting quality*; expect modest correlations (different sources rarely agree strongly [66]). Team-level: participation equality and functional/dysfunctional ratios vs team-process survey [30c], T-TPQ [36] and psychological safety [38].
- **Discriminant validity**: low correlations with *Having relevant KSAs*, grades/technical test scores, speaking time, language-proficiency level; multitrait–multimethod matrix (competency × source: model, human observers, peers, self).
- **Structural evidence**: G-study for variance due to person, session, person × session, rater — expect a large person × session component (section 2.5) and report it openly.
- **Criterion-related evidence** (later, larger N): team output quality, instructor/manager ratings, subsequent peer nominations. Pre-register predictions.
- **Consequential evidence**: participant reactions (perceived accuracy, fairness, usefulness), and whether feedback changes behaviour in later sessions.

### 5.4 Fairness checks

Follow a measurement-bias framework [79] and audit template [79b]:

- **Input disparity**: WER and diarization error by gender, native/non-native status and regional variety [77][78]; set a maximum tolerated gap and route low-quality audio to IE.
- **Score disparity**: standardised mean differences (d) and IE rates by subgroup, for model and for human gold ratings side by side — a gap present in the model but not in humans is model bias; a gap in both needs construct-level discussion.
- **Conditional bias**: model-minus-human rating differences by subgroup (differential accuracy), and differential model–human agreement by subgroup (the ETS group reports checking coder agreement across demographic subgroups; we could confirm that only from secondary summaries).
- **Counterfactual tests**: swap speaker names/gender markers; replace a transcript with a grammatical-error-injected or accented-ASR version of the same content — ratings on C2–C11 should not move.
- **Interruption/overlap**: verify the negative indicator in C5 is applied equally across genders given the same overlap pattern [76].
- **If cut scores are ever used** (not recommended): adverse-impact ratios under the four-fifths rule [L3] and a bias audit in the LL144 sense [L2].
- Documentation aligned with ISO 10667 (client/provider responsibilities, evidence of validity, reliability and fairness) [18b] and, for Spain, the COP test-review model [91].

---

## 6. Spanish-language considerations

### 6.1 Frameworks in use in Spain and Latin America

- **Martha Alles — *Diccionario de competencias: La Trilogía*** (Granica): Tomo 1 competencies (60 competencies in cardinal, managerial-specific and area-specific groups, each opened in four grades), Tomo 2 behaviours, Tomo 3 interview questions [92]. It is the de facto reference in Latin American HR and common in Spain. It is a practitioner dictionary (consultancy methodology), with no published psychometric validation that we found. Use: **naming and cross-walk for Spanish-speaking HR audiences** (e.g. *Trabajo en equipo*, *Comunicación*, *Liderazgo*, *Adaptabilidad al cambio*†), not as evidence. The text is copyrighted: paraphrase, do not copy grade descriptors.
- **Spanish public sector**: INAP's *Marco Básico de Competencias Directivas* for senior public managers in the Administración General del Estado, linked to Orden TDF/379/2024, with an INAP–UCM self-assessment questionnaire [93]. Content was not opened in this pass; obtain it before cross-walking. Spain has no single cross-government behavioural framework comparable to the UK Success Profiles.
- **EU and international frameworks with official Spanish versions**: ESCO (all skills labels in Spanish) [13]; LifeComp [14]; the UN Values and Behaviours Framework (official Spanish PDF) [10]. These give ready-made, authoritative Spanish terminology.
- **Universities**: Spanish degree programmes use "competencias transversales/genéricas" (Tuning-derived lists; teamwork is ubiquitous) [94] — relevant if peer-to-peer sessions take place in higher education.

### 6.2 Instruments with Spanish validation evidence

- **Teamwork Competency Test (TWCT)** — Aguado, Rico, Sánchez-Manzanares & Salas (2014): developed by a Spanish team after showing reliability and structural problems in the Stevens & Campion Teamwork KSA Test; better reliability and dimensional support [89]. Best candidate for a Spanish-language convergent measure of teamwork *knowledge*.
- **Psychological safety**: a Spanish adaptation of Edmondson's scale (Navarro; back-translated) exists in PsychArchives with pilot reliability/validity data [90]; we did not find a peer-reviewed Spanish validation in this pass. Treat as "adapted, validation evidence limited" and run our own reliability/CFA.
- **CATME**: no validated Spanish version was identified in this pass (not an exhaustive search). Plan a translation/back-translation and measurement-invariance check before using it as a criterion with Spanish speakers.
- **act4teams**: developed with German-speaking teams and applied to English-speaking ones; no Spanish validation identified. Our gold set would effectively be that validation.
- **Test quality standard**: the Spanish Psychological Association's test-review model (CET-R; Hernández, Ponsoda, Muñiz, Prieto & Elosua 2016), aligned with the EFPA model, is what a Spanish psychologist will use to judge our evidence [91].

### 6.3 Assessing Spanish speech

- **ASR**: accent and gender effects on Whisper/Alexa error rates for Spanish varieties have been reported [78]; other studies report differences across national varieties. Measure WER on our own speakers by variety (Peninsular north/south, Canarian, Mexican, Rioplatense, Andean, Caribbean, etc.) and by native vs non-native status. Code-switching (Spanish–English technical jargon; Spanish–Catalan/Galician/Basque) degrades ASR and diarization — test it explicitly.
- **Turn-taking norms**: descriptions of Peninsular Spanish conversation report more simultaneous talk and collaborative overlap than Anglo-American norms, and overlap is frequently affiliative† [95]. Thresholds for the C5 negative indicator tuned on English data will over-flag Spanish speakers; calibrate on Spanish data and keep overlap team-level and descriptive.
- **Politeness and directness**: unmitigated disagreement and direct requests are less marked in Peninsular Spanish than in many English varieties, with variation across Latin American varieties† [96]. Indicators for C6 must be anchored on *reasons given and issue-vs-person focus*, not on hedging or politeness formulas. Annotators should include speakers of the varieties assessed.
- **Address forms and morphology**: tú/usted/vos and diminutives carry relational meaning that English-trained rubrics miss; do not score them, but do not treat "usted" as distance or "tú" as disrespect.
- **Rubric language**: write indicators and exemplars natively in Spanish (not translated) for retrieval — embedding similarity between an English indicator and a Spanish utterance is an extra, unmeasured error source. Validate the Spanish rubric separately (agreement statistics per language).
- **LLM raters**: the published LLM-coding evidence is on English data [73][74]; agreement on Spanish transcripts must be established on our own gold set.

---

## 7. References

Tags: **[verified]** = authors/year/venue and the claim attributed here confirmed in this pass against the source page, abstract, publisher/Crossref record or the published document; **[unverified]** = cited from memory or not opened. DOIs marked "(Crossref)" were retrieved from the Crossref registry in this pass.

### Competency modelling and HR frameworks

- [1] Campion, M. A., Fink, A. A., Ruggeberg, B. J., Carr, L., Phillips, G. M., & Odman, R. B. (2011). Doing competencies well: Best practices in competency modeling. *Personnel Psychology, 64*, 225–262. https://doi.org/10.1111/j.1744-6570.2010.01207.x (Crossref) **[verified]**
- [2] Bartram, D. (2005). The Great Eight competencies: A criterion-centric approach to validation. *Journal of Applied Psychology, 90*(6), 1185–1203. https://doi.org/10.1037/0021-9010.90.6.1185 **[verified]** (29 studies, N = 4,861 confirmed from abstract-level sources; the 20/112 breakdown of the UCF is from memory)
- [3] McClelland, D. C. (1973). Testing for competence rather than for "intelligence". *American Psychologist, 28*, 1–14. https://doi.org/10.1037/h0034092 (Crossref) **[verified]**
- [4] Boyatzis, R. E. (1982). *The competent manager: A model for effective performance*. Wiley. **[unverified]**
- [5] Spencer, L. M., & Spencer, S. M. (1993). *Competence at work: Models for superior performance*. Wiley. **[unverified]**
- [6] Barrett, G. V., & Depinet, R. L. (1991). A reconsideration of testing for competence rather than for intelligence. *American Psychologist, 46*(10), 1012–1024. https://doi.org/10.1037/0003-066X.46.10.1012 **[verified]**
- [7] Korn Ferry. *Korn Ferry Leadership Architect* (38 competencies, 12 clusters, 4 factors) and its Research Guide and Technical Manual. https://www.kornferry.com/capabilities/talent-suite/korn-ferry-assess/leadership-architect **[verified]** for structure (vendor page and Stanford HR summary); individual competency names cited in section 3 are **unverified**
- [8] UK Cabinet Office / Civil Service HR. *Success Profiles: Civil Service Behaviours* (page updated 29 January 2025). https://www.gov.uk/government/publications/success-profiles/success-profiles-civil-service-behaviours **[verified]** (page opened; 9 behaviours, 6 levels)
- [9] OECD. *Competency Framework* (15 core competencies, 3 clusters, 5 levels). https://oecd.org/content/dam/oecd/en/about/careers/apply/competency_framework_en.pdf **[verified]** (PDF opened). A newer six-competency OECD Core Competency Framework also exists (https://oecd.org/en/about/careers/apply/core-competency-framework.html) — **[unverified]**, not opened
- [10] United Nations. *UN Values and Behaviours Framework* (2021; formalised in ST/SGB/2024/4). English: https://hr.un.org/sites/default/files/2025-05/English-Values%20and%20Behaviours%20Framework_Final.pdf ; Spanish: https://hr.un.org/sites/hr.un.org/files/Spanish-Values%20and%20Behaviours%20Framework_Final_0.pdf **[verified]** for structure (4 values, 5 behaviours) via UN pages in search results; PDFs not opened
- [10b] United Nations (1999). *United Nations Competencies for the Future*. https://hr.un.org/sites/hr.un.org/files/Competencies%20for%20the%20Future_0.pdf **[unverified]** (URL seen, document not opened; content from memory)
- [11] CIPD. *The Profession Map*. https://www.cipd.org/en/the-people-profession/the-profession-map/ **[unverified]**
- [11b] US Office of Personnel Management. *Executive Core Qualifications* (chart updated 1 October 2025). https://www.opm.gov/policy-data-oversight/senior-executive-service/executive-core-qualifications **[verified]** (page opened; current five ECQs). Pre-2025 ECQ content cited from memory **[unverified]**
- [11c] US National Institutes of Health, Office of Human Resources. *Competencies Dictionary* and proficiency scale. https://hr.nih.gov/working-nih/competencies/competencies-dictionary ; https://hr.nih.gov/about/faq/working-nih/competencies/what-nih-proficiency-scale **[unverified]** (HTTP 403 on the dictionary; 1–5 scale seen only in a search summary)
- [12] SHRM. *Body of Applied Skills and Knowledge (BASK)*. https://www.shrm.org/certification/about/body-of-applied-skills-and-knowledge **[verified]** for the sub-competency lists (via a copy of the 2026 BASK PDF surfaced in search; official page not opened in full)
- [13] European Commission. *ESCO — transversal skills and competences* (revised hierarchy: 6 categories, 24 groups, 96 skills). https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/transversal-skills-and-competences **[verified]** for the structure via ESCO pages in search results; exact URL of the English page not opened
- [14] Sala, A., Punie, Y., Garkov, V., & Cabrera Giraldez, M. (2020). *LifeComp: The European Framework for Personal, Social and Learning to Learn Key Competence* (EUR 30246 EN, JRC120911). Publications Office of the EU. https://publications.jrc.ec.europa.eu/repository/handle/JRC120911 **[verified]** (two DOIs appear in the JRC record — 10.2760/302967 and 10.2760/922681 — for the online and print editions; not disambiguated)
- [15] National Center for O*NET Development. *O*NET Content Model*; and *Revisiting the Work Styles Domain of the O*NET Content Model* (2024). https://www.onetcenter.org/content.html ; https://www.onetcenter.org/reports/Work_Styles_New.html **[verified]** for the Work Styles revision; Social Skills element names cited from memory **[unverified]**
- [16] OECD (2017). *PISA 2015 Assessment and Analytical Framework: Science, Reading, Mathematic, Financial Literacy and Collaborative Problem Solving* (revised ed.). OECD Publishing. https://doi.org/10.1787/9789264281820-en **[verified]** for the 3 × 4 structure via secondary sources quoting the framework; DOI from memory, not resolved
- [17] Hesse, F., Care, E., Buder, J., Sassenberg, K., & Griffin, P. (2015). A framework for teachable collaborative problem solving skills. In P. Griffin & E. Care (Eds.), *Assessment and Teaching of 21st Century Skills: Methods and Approach* (pp. 37–56). Springer. https://doi.org/10.1007/978-94-017-9395-7_2 (Crossref) **[verified]**

### Rating methods, assessment centres, rater training

- [18] International Taskforce on Assessment Center Guidelines (2015). Guidelines and ethical considerations for assessment center operations. *Journal of Management, 41*(4), 1244–1273. https://doi.org/10.1177/0149206314567780 **[verified]**
- [18b] ISO 10667-1:2020. *Assessment service delivery — Procedures and methods to assess people in work and organizational settings — Part 1: Requirements for the client* (Part 2: service provider). https://www.iso.org/standard/74716.html **[verified]** (catalogue record; standard text is paywalled and was not read)
- [19] Flanagan, J. C. (1954). The critical incident technique. *Psychological Bulletin, 51*, 327–358. https://doi.org/10.1037/h0061470 (Crossref) **[verified]**
- [20] Smith, P. C., & Kendall, L. M. (1963). Retranslation of expectations: An approach to the construction of unambiguous anchors for rating scales. *Journal of Applied Psychology, 47*, 149–155. https://doi.org/10.1037/h0047060 (Crossref) **[verified]**
- [21] Latham, G. P., & Wexley, K. N. (1977). Behavioral observation scales for performance appraisal purposes. *Personnel Psychology, 30*, 255–268. https://doi.org/10.1111/j.1744-6570.1977.tb02092.x (Crossref) **[verified]**
- [22] Woehr, D. J., & Huffcutt, A. I. (1994). Rater training for performance appraisal: A quantitative review. *Journal of Occupational and Organizational Psychology, 67*, 189–205. https://doi.org/10.1111/j.2044-8325.1994.tb00562.x (Crossref) **[verified]**
- [23] Roch, S. G., Woehr, D. J., Mishra, V., & Kieszczynska, U. (2012). Rater training revisited: An updated meta-analytic review of frame-of-reference training. *Journal of Occupational and Organizational Psychology, 85*, 370–395. https://doi.org/10.1111/j.2044-8325.2011.02045.x (Crossref) **[verified]** (effect size not quoted because not re-confirmed)
- [24] Tziner, A., & Kopelman, R. E. (2002). Is there a preferred performance rating format? A non-psychometric perspective. *Applied Psychology, 51*, 479–503. https://doi.org/10.1111/1464-0597.00104 (Crossref) **[verified]** (record only); see also DeNisi, A. S., & Murphy, K. R. (2017). Performance appraisal and performance management: 100 years of progress? *Journal of Applied Psychology, 102*, 421–433. https://doi.org/10.1037/apl0000085 (Crossref) **[verified]** (record only)
- [25] Levashina, J., Hartwell, C. J., Morgeson, F. P., & Campion, M. A. (2014). The structured employment interview: Narrative and quantitative review of the research literature. *Personnel Psychology, 67*, 241–293. https://doi.org/10.1111/peps.12052 (Crossref) **[verified]** (record only)
- [25b] Sackett, P. R., Zhang, C., Berry, C. M., & Lievens, F. (2022). Revisiting meta-analytic estimates of validity in personnel selection: Addressing systematic overcorrection for restriction of range. *Journal of Applied Psychology, 107*(11), 2040–2068. https://doi.org/10.1037/apl0000994 **[verified]**
- [54] Thorndike, E. L. (1920). A constant error in psychological ratings. *Journal of Applied Psychology, 4*, 25–29. https://doi.org/10.1037/h0071663 (Crossref) **[verified]** (record only)
- [55] Murphy, K. R., Jako, R. A., & Anhalt, R. L. (1993). Nature and consequences of halo error: A critical analysis. *Journal of Applied Psychology, 78*, 218–225. https://doi.org/10.1037/0021-9010.78.2.218 (Crossref) **[verified]** (record only)
- [56] Hoffman, B. J., Kennedy, C. L., LoPilato, A. C., Monahan, E. L., & Lance, C. E. (2015). A review of the content, criterion-related, and construct-related validity of assessment center exercises. *Journal of Applied Psychology, 100*, 1143–1168. https://doi.org/10.1037/a0038707 (Crossref) **[verified]** (record only); Gaugler, B. B., Rosenthal, D. B., Thornton, G. C., & Bentson, C. (1987). Meta-analysis of assessment center validity. *Journal of Applied Psychology, 72*, 493–511. https://doi.org/10.1037/0021-9010.72.3.493 (Crossref) **[verified]** (record only)
- [57] Sackett, P. R., & Dreher, G. F. (1982). Constructs and assessment center dimensions: Some troubling empirical findings. *Journal of Applied Psychology, 67*, 401–410. https://doi.org/10.1037/0021-9010.67.4.401 (Crossref) **[verified]**
- [58] Lance, C. E. (2008). Why assessment centers do not work the way they are supposed to. *Industrial and Organizational Psychology, 1*, 84–97. https://doi.org/10.1111/j.1754-9434.2007.00017.x (Crossref) **[verified]**
- [59] Lievens, F., & Conway, J. M. (2001). Dimension and exercise variance in assessment center scores: A large-scale evaluation of multitrait–multimethod studies. *Journal of Applied Psychology, 86*, 1202–1222. https://doi.org/10.1037/0021-9010.86.6.1202 (Crossref) **[verified]** (record only)
- [60] Bowler, M. C., & Woehr, D. J. (2006). A meta-analytic evaluation of the impact of dimension and exercise factors on assessment center ratings. *Journal of Applied Psychology, 91*, 1114–1124. https://doi.org/10.1037/0021-9010.91.5.1114 (Crossref) **[verified]** (record only)
- [61] Putka, D. J., & Hoffman, B. J. (2013). Clarifying the contribution of assessee-, dimension-, exercise-, and assessor-related effects to reliable and unreliable variance in assessment center ratings. *Journal of Applied Psychology, 98*, 114–133. https://doi.org/10.1037/a0030887 **[verified]** (variance percentages known only from secondary summaries)
- [62] Jackson, D. J. R., Michaelides, G., Dewberry, C., & Kim, Y.-J. (2016). Everything that you have ever been told about assessment center ratings is confounded. *Journal of Applied Psychology, 101*, 976–994. https://doi.org/10.1037/apl0000102 (Crossref) **[verified]** (record only)
- [62b] Dewberry, C. (2024). Assessment centers do not measure competencies: Why this is now beyond reasonable doubt. *Industrial and Organizational Psychology, 17*(2), 154–175. https://doi.org/10.1017/iop.2024.5 **[verified]**
- [63] Arthur, W., Jr., Day, E. A., McNelly, T. L., & Edens, P. S. (2003). A meta-analysis of the criterion-related validity of assessment center dimensions. *Personnel Psychology, 56*, 125–153. https://doi.org/10.1111/j.1744-6570.2003.tb00146.x (Crossref) **[verified]** (abstract-level; dimension-by-dimension validity coefficients not quoted because not re-confirmed)
- [64] Meriac, J. P., Hoffman, B. J., Woehr, D. J., & Fleisher, M. S. (2008). Further evidence for the validity of assessment center dimensions: A meta-analysis of the incremental criterion-related validity of dimension ratings. *Journal of Applied Psychology, 93*, 1042–1052. https://doi.org/10.1037/0021-9010.93.5.1042 **[verified]**
- [65] Meriac, J. P., Hoffman, B. J., & Woehr, D. J. (2014). A conceptual and empirical review of the structure of assessment center dimensions. *Journal of Management, 40*(5), 1269–1296. https://doi.org/10.1177/0149206314522299 **[verified]**
- [66] Conway, J. M., & Huffcutt, A. I. (1997). Psychometric properties of multisource performance ratings: A meta-analysis of subordinate, supervisor, peer, and self-ratings. *Human Performance, 10*, 331–360. https://doi.org/10.1207/s15327043hup1004_2 (Crossref) **[verified]** (record only)
- [67] Smither, J. W., London, M., & Reilly, R. R. (2005). Does performance improve following multisource feedback? A theoretical model, meta-analysis, and review of empirical findings. *Personnel Psychology, 58*, 33–66. https://doi.org/10.1111/j.1744-6570.2005.514_1.x (Crossref) **[verified]** (record only; "small average improvement" from memory)

### Teamwork models and observation instruments

- [26] Bales, R. F. (1950). A set of categories for the analysis of small group interaction. *American Sociological Review, 15*(2), 257–263. https://doi.org/10.2307/2086790 (Crossref) **[verified]** (record; the 12 category names are from memory)
- [26b] Bales, R. F., & Cohen, S. P. (1979). *SYMLOG: A system for the multiple level observation of groups*. Free Press. **[unverified]**
- [27] Benne, K. D., & Sheats, P. (1948). Functional roles of group members. *Journal of Social Issues, 4*(2), 41–49. https://doi.org/10.1111/j.1540-4560.1948.tb01783.x **[verified]**
- [28] Stevens, M. J., & Campion, M. A. (1994). The knowledge, skill, and ability requirements for teamwork: Implications for human resource management. *Journal of Management, 20*(2), 503–530. https://doi.org/10.1177/014920639402000210 (Crossref) **[verified]**
- [29] Salas, E., Sims, D. E., & Burke, C. S. (2005). Is there a "Big Five" in teamwork? *Small Group Research, 36*(5), 555–599. https://doi.org/10.1177/1046496405277134 **[verified]**
- [30] Marks, M. A., Mathieu, J. E., & Zaccaro, S. J. (2001). A temporally based framework and taxonomy of team processes. *Academy of Management Review, 26*(3), 356–376. https://doi.org/10.2307/259182 (Crossref; the publisher also uses 10.5465/amr.2001.4845785) **[verified]**
- [30b] LePine, J. A., Piccolo, R. F., Jackson, C. L., Mathieu, J. E., & Saul, J. R. (2008). A meta-analysis of teamwork processes: Tests of a multidimensional model and relationships with team effectiveness criteria. *Personnel Psychology, 61*, 273–307. https://doi.org/10.1111/j.1744-6570.2008.00114.x (Crossref) **[verified]** (record only)
- [30c] Mathieu, J. E., Luciano, M. M., D'Innocenzo, L., Klock, E. A., & LePine, J. A. (2020). The development and construct validity of a team processes survey measure. *Organizational Research Methods, 23*, 399–431. https://doi.org/10.1177/1094428119840801 (Crossref) **[verified]** (record only)
- [31] Rousseau, V., Aubé, C., & Savoie, A. (2006). Teamwork behaviors: A review and an integration of frameworks. *Small Group Research, 37*(5), 540–570. https://doi.org/10.1177/1046496406293125 (Crossref) **[verified]**
- [32] Loughry, M. L., Ohland, M. W., & Moore, D. D. (2007). Development of a theory-based assessment of team member effectiveness. *Educational and Psychological Measurement, 67*(3), 505–524. https://doi.org/10.1177/0013164406292085 **[verified]**
- [33] Ohland, M. W., Loughry, M. L., Woehr, D. J., Bullard, L. G., Felder, R. M., Finelli, C. J., Layton, R. A., Pomeranz, H. R., & Schmucker, D. G. (2012). The Comprehensive Assessment of Team Member Effectiveness: Development of a behaviorally anchored rating scale for self- and peer evaluation. *Academy of Management Learning & Education, 11*(4), 609–630. https://doi.org/10.5465/amle.2010.0177 **[verified]** (abstract-level; the scale-point layout and anchor contents are from memory). Dimension list also confirmed at https://info.catme.org/features/catme-five-teamwork-dimensions/
- [35] Kauffeld, S., & Lehmann-Willenbrock, N. (2012). Meetings matter: Effects of team meetings on team and organizational success. *Small Group Research, 43*(2), 130–158. https://doi.org/10.1177/1046496411429599 **[verified]** (abstract-level; individual code names are from memory)
- [35b] Lehmann-Willenbrock, N., & Allen, J. A. (2018). Modeling temporal interaction dynamics in organizational settings. *Journal of Business and Psychology, 33*(3), 325–344. https://doi.org/10.1007/s10869-017-9506-9 **[verified]**
- [36] Agency for Healthcare Research and Quality. *TeamSTEPPS measurement tools* (T-TPQ, T-TAQ, TPOT). https://www.ahrq.gov/teamstepps-program/resources/tools/index.html **[verified]** (via AHRQ pages in search results)
- [37] Flin, R., Martin, L., Goeters, K.-M., Hörmann, H.-J., Amalberti, R., Valot, C., & Nijhuis, H. (2003). Development of the NOTECHS (non-technical skills) system for assessing pilots' CRM skills. *Human Factors and Aerospace Safety, 3*(2), 97–119. https://elib.dlr.de/55190/ **[verified]** (repository record only)
- [37b] Fletcher, G., Flin, R., McGeorge, P., Glavin, R., Maran, N., & Patey, R. (2003). Anaesthetists' Non-Technical Skills (ANTS): Evaluation of a behavioural marker system. *British Journal of Anaesthesia, 90*(5), 580–588. https://doi.org/10.1093/bja/aeg112 **[verified]**
- [37c] Yule, S., Flin, R., Paterson-Brown, S., Maran, N., & Rowley, D. (2006). Development of a rating system for surgeons' non-technical skills. *Medical Education, 40*(11), 1098–1104. https://doi.org/10.1111/j.1365-2929.2006.02610.x (Crossref) **[verified]**
- [38] Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly, 44*(2), 350–383. https://doi.org/10.2307/2666999 **[verified]**
- [38b] Frazier, M. L., Fainshmidt, S., Klinger, R. L., Pezeshkan, A., & Vracheva, V. (2017). Psychological safety: A meta-analytic review and extension. *Personnel Psychology, 70*, 113–165. https://doi.org/10.1111/peps.12183 (Crossref) **[verified]** (record only)
- [39] Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N., & Malone, T. W. (2010). Evidence for a collective intelligence factor in the performance of human groups. *Science, 330*(6004), 686–688. https://doi.org/10.1126/science.1193147 **[verified]**
- [40] Riedl, C., Kim, Y. J., Gupta, P., Malone, T. W., & Woolley, A. W. (2021). Quantifying collective intelligence in human groups. *PNAS, 118*(21), e2005737118. https://doi.org/10.1073/pnas.2005737118 **[verified]**
- [41] Pentland, A. (2012). The new science of building great teams. *Harvard Business Review, 90*(4), 60–69. https://hbr.org/2012/04/the-new-science-of-building-great-teams **[unverified]**; Kim, T., Chang, A., Holland, L., & Pentland, A. (2008). Meeting mediator: Enhancing group collaboration using sociometric feedback. *CHI '08 Extended Abstracts*, 3183–3188. https://doi.org/10.1145/1358628.1358828 (Crossref) **[verified]** (record only)
- [42] MacLaren, N. G., Yammarino, F. J., Dionne, S. D., Sayama, H., Mumford, M. D., Connelly, S., et al. (2020). Testing the babble hypothesis: Speaking time predicts leader emergence in small groups. *The Leadership Quarterly, 31*(5), 101409. https://doi.org/10.1016/j.leaqua.2020.101409 **[verified]**
- [43] Huang, K., Yeomans, M., Brooks, A. W., Minson, J., & Gino, F. (2017). It doesn't hurt to ask: Question-asking increases liking. *Journal of Personality and Social Psychology, 113*, 430–452. https://doi.org/10.1037/pspi0000097 (Crossref) **[verified]** (record only)
- [43b] Karpowitz, C. F., Mendelberg, T., & Shaker, L. (2012). Gender inequality in deliberative participation. *American Political Science Review, 106*(3), 533–547. https://doi.org/10.1017/S0003055412000329 (Crossref) **[verified]** (record only; finding summarised from memory)

### Critical appraisal of popular instruments

- [44] Furnham, A., Steele, H., & Pendleton, D. (1993). A psychometric assessment of the Belbin Team-Role Self-Perception Inventory. *Journal of Occupational and Organizational Psychology, 66*, 245–257. https://doi.org/10.1111/j.2044-8325.1993.tb00535.x (Crossref) **[verified]** (findings via secondary sources)
- [45] Aritzeta, A., Swailes, S., & Senior, B. (2007). Belbin's team role model: Development, validity and applications for team building. *Journal of Management Studies, 44*(1), 96–118. https://doi.org/10.1111/j.1467-6486.2007.00666.x **[verified]**
- [46] Broucek, W. G., & Randell, G. (1996). An assessment of the construct validity of the Belbin Self-Perception Inventory and Observer's Assessment from the perspective of the five-factor model. *Journal of Occupational and Organizational Psychology, 69*, 389–405. https://doi.org/10.1111/j.2044-8325.1996.tb00625.x (Crossref) **[verified]** (record only)
- [48] Tuckman, B. W. (1965). Developmental sequence in small groups. *Psychological Bulletin, 63*(6), 384–399. https://doi.org/10.1037/h0022100 (Crossref) **[verified]** (record only); Tuckman, B. W., & Jensen, M. A. C. (1977). Stages of small-group development revisited. *Group & Organization Studies, 2*, 419–427. https://doi.org/10.1177/105960117700200404 (Crossref) **[verified]** (record only)
- [49] Bonebright, D. A. (2010). 40 years of storming: A historical review of Tuckman's model of small group development. *Human Resource Development International, 13*(1), 111–120. https://doi.org/10.1080/13678861003589099 **[verified]**
- [50] Gersick, C. J. G. (1988). Time and transition in work teams: Toward a new model of group development. *Academy of Management Journal, 31*, 9–41. https://doi.org/10.2307/256496 (Crossref) **[verified]** (record only)
- [51] Pittenger, D. J. (2005). Cautionary comments regarding the Myers-Briggs Type Indicator. *Consulting Psychology Journal: Practice and Research, 57*(3), 210–221. https://doi.org/10.1037/1065-9293.57.3.210 (Crossref) **[verified]**
- [52] McCrae, R. R., & Costa, P. T. (1989). Reinterpreting the Myers-Briggs Type Indicator from the perspective of the five-factor model of personality. *Journal of Personality, 57*, 17–40. https://doi.org/10.1111/j.1467-6494.1989.tb00759.x (Crossref) **[verified]** (record only)
- [53] Stein, R., & Swan, A. B. (2019). Evaluating the validity of Myers-Briggs Type Indicator theory: A teaching tool and window into intuitive psychology. *Social and Personality Psychology Compass, 13*(2). https://doi.org/10.1111/spc3.12441 (Crossref) **[verified]** (record only)

### Computational assessment, bias

- [68] Flor, M., Yoon, S.-Y., Hao, J., Liu, L., & von Davier, A. (2016). Automated classification of collaborative problem solving interactions in simulated science tasks. *Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications*, 31–41. https://doi.org/10.18653/v1/W16-0504 **[verified]**
- [69] Hao, J., Chen, L., Flor, M., Liu, L., & von Davier, A. A. (2017). CPS-Rater: Automated sequential annotation for conversations in collaborative problem-solving activities. *ETS Research Report Series, 2017*, 1–9. https://doi.org/10.1002/ets2.12184 **[verified]**
- [70] Graesser, A. C., Fiore, S. M., Greiff, S., Andrews-Todd, J., Foltz, P. W., & Hesse, F. W. (2018). Advancing the science of collaborative problem solving. *Psychological Science in the Public Interest, 19*(2), 59–92. https://doi.org/10.1177/1529100618808244 (Crossref) **[verified]** (record only); Andrews-Todd, J., & Forsyth, C. M. (2020). Exploring social and cognitive dimensions of collaborative problem solving in an open online simulation-based task. *Computers in Human Behavior, 104*, 105759. https://doi.org/10.1016/j.chb.2018.10.025 (Crossref) **[verified]** (record only)
- [71] Pugh, S. L., Subburaj, S. K., Rao, A. R., Stewart, A. E. B., Andrews-Todd, J., & D'Mello, S. K. (2021). Say what? Automatic modeling of collaborative problem solving skills from student speech in the wild. *Proceedings of the 14th International Conference on Educational Data Mining*, 55–67. https://par.nsf.gov/biblio/10494306 **[verified]**
- [72] Demszky, D., Liu, J., Mancenido, Z., Cohen, J., Hill, H., Jurafsky, D., & Hashimoto, T. (2021). Measuring conversational uptake: A case study on student-teacher interactions. *Proceedings of ACL-IJCNLP 2021*, 1638–1653. https://doi.org/10.18653/v1/2021.acl-long.130 **[verified]**
- [72b] Okada, S., Ohtake, Y., Nakano, Y. I., Hayashi, Y., Huang, H.-H., Takase, Y., & Nitta, K. (2016). Estimating communication skills using dialogue acts and nonverbal features in multiple discussion datasets. *Proceedings of ICMI 2016*, 169–176. https://doi.org/10.1145/2993148.2993154 **[verified]**
- [72c] Naim, I., Tanveer, M. I., Gildea, D., & Hoque, M. E. (2018). Automated analysis and prediction of job interview performance. *IEEE Transactions on Affective Computing, 9*(2), 191–204. https://doi.org/10.1109/TAFFC.2016.2614299 **[verified]** (interpretation "fluency/engagement proxies" is ours)
- [72d] Hickman, L., Bosch, N., Ng, V., Saef, R., Tay, L., & Woo, S. E. (2022). Automated video interview personality assessments: Reliability, validity, and generalizability investigations. *Journal of Applied Psychology, 107*(8), 1323–1351. https://doi.org/10.1037/apl0000695 **[verified]**
- [73] Hao, J., Cui, W., Kyllonen, P., Kerzabi, E., Liu, L., & Flor, M. (2024, rev. 2025). Automated coding of communications in collaborative problem-solving tasks using ChatGPT (earlier title: Scaling up the evaluation of collaborative problem solving: Promises and challenges of coding chat data with ChatGPT). arXiv:2411.10246. https://arxiv.org/abs/2411.10246 **[verified]** (abstract; agreement statistics not extracted)
- [74] Friedman, R., et al. (2024). An application of large language models to coding negotiation transcripts. arXiv:2407.21037. https://arxiv.org/abs/2407.21037 **[verified]** (search-level summary only; author list beyond the first author not confirmed)
- [75] Anderson, K. J., & Leaper, C. (1998). Meta-analyses of gender effects on conversational interruption: Who, what, when, where, and how. *Sex Roles, 39*(3–4), 225–252. https://doi.org/10.1023/A:1018802521676 **[verified]**
- [76] Hilton, K. (2018). *What does an interruption sound like?* (Doctoral dissertation, Stanford University). https://purl.stanford.edu/vf660gm5432 **[verified]** (repository record and university press summary)
- [76b] Schegloff, E. A. (2000). Overlapping talk and the organization of turn-taking for conversation. *Language in Society, 29*, 1–63. https://doi.org/10.1017/S0047404500001019 (Crossref) **[verified]** (record only)
- [77] Koenecke, A., Nam, A., Lake, E., Nudell, J., Quartey, M., Mengesha, Z., Toups, C., Rickford, J. R., Jurafsky, D., & Goel, S. (2020). Racial disparities in automated speech recognition. *PNAS, 117*(14), 7684–7689. https://doi.org/10.1073/pnas.1915768117 **[verified]**
- [78] Nacimiento-García, E., Díaz-Kaas-Nielsen, H. S., & González-González, C. S. (2024). Gender and accent biases in AI-based tools for Spanish: A comparative study between Alexa and Whisper. *Applied Sciences, 14*(11), 4734. https://doi.org/10.3390/app14114734 **[verified]** (abstract/search-level). See also Google Research, "Dialect bias in speech recognition across 10 Spanish and French varieties", https://research.google/pubs/dialect-bias-in-speech-recognition-across-10-spanish-and-french-varieties/ **[unverified]** (title seen, not opened)
- [79] Tay, L., Woo, S. E., Hickman, L., Booth, B. M., & D'Mello, S. (2022). A conceptual framework for investigating and mitigating machine-learning measurement bias (MLMB) in psychological assessment. *Advances in Methods and Practices in Psychological Science, 5*(1). https://doi.org/10.1177/25152459211061337 **[verified]**
- [79b] Landers, R. N., & Behrend, T. S. (2023). Auditing the AI auditors: A framework for evaluating fairness and bias in high stakes AI predictive models. *American Psychologist, 78*, 36–49. https://doi.org/10.1037/amp0000972 (Crossref) **[verified]** (record only)
- [79c] Raghavan, M., Barocas, S., Kleinberg, J., & Levy, K. (2020). Mitigating bias in algorithmic hiring: Evaluating claims and practices. *Proceedings of FAT\* 2020*, 469–481. https://doi.org/10.1145/3351095.3372828 (Crossref) **[verified]** (record only)

### Reliability statistics

- [80] Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155–163. https://doi.org/10.1016/j.jcm.2016.02.012 (Crossref) **[verified]** (thresholds <.50 / .50–.75 / .75–.90 / >.90 as widely quoted)
- [81] Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin, 86*, 420–428. https://doi.org/10.1037/0033-2909.86.2.420 (Crossref) **[verified]** (record only)
- [82] Krippendorff, K. (2004). Reliability in content analysis: Some common misconceptions and recommendations. *Human Communication Research, 30*(3), 411–433. https://doi.org/10.1093/hcr/30.3.411 (Crossref) **[verified]** (record; the α ≥ .800 / ≥ .667 convention is attributed to Krippendorff's *Content Analysis* textbook by several secondary sources)
- [83] Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*, 159–174. https://doi.org/10.2307/2529310 (Crossref) **[verified]** (record only; benchmark labels from memory)
- [84] LeBreton, J. M., & Senter, J. L. (2008). Answers to 20 questions about interrater reliability and interrater agreement. *Organizational Research Methods, 11*, 815–852. https://doi.org/10.1177/1094428106296642 (Crossref) **[verified]** (record only); Hallgren, K. A. (2012). Computing inter-rater reliability for observational data: An overview and tutorial. *Tutorials in Quantitative Methods for Psychology, 8*(1), 23–34. https://doi.org/10.20982/tqmp.08.1.p023 (Crossref) **[verified]** (record only)
- [85] Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for evaluating normed and standardized assessment instruments in psychology. *Psychological Assessment, 6*, 284–290. https://doi.org/10.1037/1040-3590.6.4.284 (Crossref) **[verified]** (record only; thresholds from memory)
- [86] Viswesvaran, C., Ones, D. S., & Schmidt, F. L. (1996). Comparative analysis of the reliability of job performance ratings. *Journal of Applied Psychology, 81*, 557–574. https://doi.org/10.1037/0021-9010.81.5.557 (Crossref) **[verified]** (record only; the ≈ .5 figure is from memory)
- [87] Conway, J. M., Jako, R. A., & Goodman, D. F. (1995). A meta-analysis of interrater and internal consistency reliability of selection interviews. *Journal of Applied Psychology, 80*, 565–579. https://doi.org/10.1037/0021-9010.80.5.565 (Crossref) **[verified]** (record only)
- [88] Bonett, D. G. (2002). Sample size requirements for estimating intraclass correlations with desired precision. *Statistics in Medicine, 21*, 1331–1335. https://doi.org/10.1002/sim.1108 (Crossref) **[verified]** (record only; formula applied from memory)

### Spanish-language sources

- [89] Aguado, D., Rico, R., Sánchez-Manzanares, M., & Salas, E. (2014). Teamwork Competency Test (TWCT): A step forward on measuring teamwork competencies. *Group Dynamics: Theory, Research, and Practice, 18*(2), 101–121. https://doi.org/10.1037/a0036098 (Crossref); preprint: https://repositorio.uam.es/bitstream/10486/666276/3/teamwork_aguado_gd_2014_pre.pdf **[verified]**
- [90] Navarro, J. *Cuestionario de Seguridad Psicológica* (Spanish adaptation of Edmondson, 1999). PsychArchives. https://www.psycharchives.org/en/item/695f048d-eb39-45df-a7a3-9b7b44bcf3a2 **[verified]** (existence via search summary; psychometric results not seen)
- [91] Hernández, A., Ponsoda, V., Muñiz, J., Prieto, G., & Elosua, P. (2016). Revisión del modelo para evaluar la calidad de los tests utilizados en España. *Papeles del Psicólogo, 37*(3), 192–197. **[unverified]** (cited from memory; no DOI or URL confirmed)
- [92] Alles, M. A. *Diccionario de competencias: La Trilogía. Tomo 1 — Las 60 competencias más utilizadas en gestión por competencias* (nueva edición). Ediciones Granica. ISBN 9789506418724; with Tomo 2 *Diccionario de comportamientos* and Tomo 3 *Diccionario de preguntas*. **[verified]** for existence, structure (cardinal / managerial-specific / area-specific; four grades) via publisher/bookseller descriptions; entry names and grade labels are **unverified**
- [93] Instituto Nacional de Administración Pública (INAP). *Marco Básico de Competencias Directivas* (linked to Orden TDF/379/2024) and INAP–UCM competency self-assessment questionnaire. News items: https://laadministracionaldia.inap.es/noticia.asp?id=1517348 ; https://laadministracionaldia.inap.es/noticia.asp?id=1517222 **[verified]** for existence via search summaries; framework content not opened
- [94] González, J., & Wagenaar, R. (Eds.) (2003). *Tuning Educational Structures in Europe. Final Report, Phase One*. Universidad de Deusto. **[unverified]**
- [95] Berry, A. (1994). Spanish and American turn-taking styles: A comparative study. *Pragmatics and Language Learning, 5*, 180–190; and Cestero Mancera, A. M. (2000). *El intercambio de turnos de habla en la conversación*. Universidad de Alcalá. **[unverified]**
- [96] Márquez Reiter, R. (2000). *Linguistic politeness in Britain and Uruguay*. John Benjamins; Hernández-Flores, N. (1999). Politeness ideology in Spanish colloquial conversation. *Pragmatics, 9*(1), 37–49. **[unverified]**

### Legal pointers

- [L1] Regulation (EU) 2024/1689 (Artificial Intelligence Act), Annex III point 4 and Article 5(1)(f). https://eur-lex.europa.eu/eli/reg/2024/1689/oj **[verified]** for the content of both provisions via the Commission AI Act Service Desk and legal commentaries in search results; EUR-Lex text not opened. Application dates for Annex III obligations were reported inconsistently across sources and are not stated here.
- [L2] New York City Local Law 144 of 2021 (Automated Employment Decision Tools). https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page **[unverified]**
- [L3] Uniform Guidelines on Employee Selection Procedures, 29 CFR Part 1607 (four-fifths rule at §1607.4(D)). https://www.ecfr.gov/current/title-29/part-1607 **[unverified]**
- [L4] Regulation (EU) 2016/679 (GDPR), Articles 9 and 22. https://eur-lex.europa.eu/eli/reg/2016/679/oj **[unverified]**

### Sources searched for but not confirmed

- A peer-reviewed, independent validation literature for DISC: none located in this pass.
- Peer-reviewed I/O-journal studies (2023–2026) of general-purpose LLMs scoring interview or group-discussion competencies against trained assessors: none confirmed; only preprints, medical-education studies and vendor documents surfaced. Treat LLM-as-rater validity for workplace competencies as **not yet established in the peer-reviewed literature we could confirm**.
- A validated Spanish version of CATME or act4teams: none located.
- The exact inter-rater reliability reported for act4teams coding: not confirmed.
