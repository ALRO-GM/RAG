# Soft-skills assessment from recorded conversations: what the evidence supports

Status: research synthesis, 2026-11-14. It supersedes the evidence table of [`THEORY.md`](THEORY.md) where the two differ, and its glossary (§13) fixes the terms (types of reliability and validity) used across `docs/skills/`. The Spanish page https://rag.peer4u.app/research.html presents the same evidence with charts.

Synthesis of six research passes over the literature, published as the repository's reference for the soft-skills work. Scores for 70 models, instruments and methods are in [`research/scores.json`](research/scores.json); counts are in [`research/coverage.json`](research/coverage.json); the rubric is [`research/RUBRIC.md`](research/RUBRIC.md). Source ids in backticks refer to [`research/sources.json`](research/sources.json), which lists authors, year, title, venue, DOI or URL and how far each source was read (full text, abstract only, or secondary). The figures drawn on the public research page (https://rag.peer4u.app/research.html) are in [`research/figures.json`](research/figures.json), each with its location in the source. The source records, notes and saved papers themselves are kept privately (copyright). Every figure below comes from those records or from a saved paper. Where I derived a number myself (for example with Spearman-Brown), I say so.

How to read the scores. Each entry has three separate scores (0-100), never merged:
- **A**: credibility of the evidence.
- **B**: acceptance in organisations.
- **C**: fit to this product, which scores individuals from a recorded, transcribed conversation.

A high C with a low A means "the right shape for the product, but nobody has shown it works". Read C on its own and it misleads (see the leaderless group discussion below). The rubric was applied as written. The interpretation rules I added where RUBRIC.md gives no anchors are listed in §12.

---

## 1. One-page summary

**What the evidence supports**

1. **Sampling behaviour beats asking about it.** The best re-estimated predictors of job performance are:
   - structured interview, ρ = .42
   - work sample, .33
   - assessment centre (AC), .29
   - situational judgement test (SJT), .26

   Self-reported Big Five traits come lower: conscientiousness .19 (.25 when contextualised), the other four traits .05-.10 (`sackett2022`, full text).
2. **Trained observers can agree with each other.** Some figures:
   - AC assessors: G = .80 for a single rating, .97 for the overall score (`jackson2016`); in a second operational AC, .76 for one assessor and .87 for two (Putka & Hoffman 2013, verified, JAP 98, Results).
   - Negotiation coding: κ = .80 (`jaeckel2024_negotiact`).
   - Meeting coding: κ = .81 after about 200 h of training per coder (`lehmannwillenbrock2015`).
   - Collaborative problem solving (CPS) coding: ICC .80-.93 (`sun2020`, `pugh2021`).
3. **Observer ratings of personality are more useful than self-reports.** Other-ratings predict job performance better than, and in addition to, self-reports (`connelly2010`, meta-analysis; `ohwangmount2011`, direction verified in the PubMed abstract, coefficients not seen). Automatic personality models work only when trained on observer labels: convergent validity .40, against .12 for models trained on self-reports (`hickman2022`).
4. **A machine can stand in for one trained rater, in one small study.** On manual transcripts of 3-minute role-plays, ML matched about one human assessor. Role player + ML kept the panel's criterion validity (.55 against .57; `hickman2023`, N = 96).

**What it does not support**

1. **No validated instrument scores an individual's soft skills from one transcribed workplace meeting.** None was found in any area.
2. **Several situations are needed per person.** Agreement across exercises is low:
   - .26 for a single rating
   - .55 for a dimension over 3 exercises
   - .65 for an overall score over 3 exercises (`jackson2016`)

   One meeting is one exercise.
3. **Text loses information.** Strangers judging personality from text or electronic communication agree only .11-.28, against .36-.59 from natural audio-visual behaviour (`connelly2010`, Table 4). The developers of the best meeting and negotiation schemes coded from video for this reason.
4. **The observational schemes closest to the product were validated at team or dyad level.**
   - act4teams: team outcomes.
   - NegotiAct: joint gains of the dyad, N = 18.
   - The talk dimensions that look most like conversation add the least in ACs: communication ΔR² = .003 (n.s.), consideration of others .005 (`meriac2008`).
5. **Speaking time "predicts" leadership only as peer-perceived emergence.** It also rewards talkativeness and carries a gender effect (`maclaren2020`, verified in full text 2026-11-15; `sanchez2010`).
6. **An LLM given interview transcripts did not work for competences.** Its test-retest reliability was "insufficient", and R² was below 0 for job-related competences (`zhang2024`; which model and criterion the R² refers to should be checked in the paper).
7. **Regulation constrains the design.** Inferring emotions from voice or face at work, including from candidates, is prohibited. Text-only inference is outside that ban. Selection and evaluation of workers is high-risk (AI Act Art. 5(1)(f), Annex III 4, as reproduced unofficially; §7).

**Bottom line.** Build the product as an **aid to trained human raters**, on the assessment-centre model:
- several short situations per person;
- behaviourally anchored, transcript-codable indicators;
- machine scores validated against a human panel and a later criterion.

Do not sell it as an automatic judge of personality or leadership.

---

## 2. Ranked tables per family

Sorted by C (fit) inside each family, with A and B beside it. A "popular-but-weak" flag means measured use or belief is substantial while the collected evidence is weak.

### Selection methods

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Leaderless group discussion and leader emergence (incl. speaking time) | 19 | 37 | 100 | Low | avoid |
| 2 | Assessment centre (interpersonal exercises: role-play, group discussion, presentation) | 69 | 74 | 93 | High | use |
| 3 | Multiple speeded assessments (many 3-minute role-plays) | 23 | 6 | 90 | Medium | use-as-reference |
| 4 | Structured (behavioural / competency) interview | 67 | 77 | 59 | High | use-as-reference |
| 5 | Situational judgement test (paper or video) | 59 | 8 | 45 | Medium | use-as-reference |
| 6 | Work sample / job simulation | 55 | 49 | 42 | High | cite-only |
| 7 | Unstructured interview (popular-but-weak) | 37 | 53 | 42 | High | avoid |
| 8 | Teamwork KSA test (Stevens & Campion) | 20 | 0 | 35 | Low | cite-only |
| 9 | Graphology (popular-but-weak) | 0 | 15 | 15 | Low | avoid |

### Observational coding schemes

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Generalized collaborative problem solving competency model (Sun, Shute, D'Mello et al.) | 33 | 6 | 75 | High | use |
| 2 | Negotiation tactic/sequence coding (Weingart et al.; Adair & Brett; Olekalns & Smith) | 25 | 0 | 70 | Low | use-as-reference |
| 3 | NegotiAct negotiation coding scheme | 23 | 6 | 70 | High | use |
| 4 | Riverside Behavioral Q-sort (RBQ) | 23 | 6 | 68 | Low | use-as-reference |
| 5 | Behavioural expressiveness coding in group interaction (Bleckmann & Wagner) | 21 | 6 | 68 | High | use-as-reference |
| 6 | Twente leader-behaviour video coding scheme (Hoogeboom & Wilderom) | 15 | 0 | 68 | Medium | use-as-reference |
| 7 | Bales Interaction Process Analysis / SYMLOG | 2 | 0 | 57 | Low | cite-only |
| 8 | act4teams (Advanced Interaction Analysis for Teams) | 37 | 8 | 55 | Medium | use-as-reference |
| 9 | Clinical behavioural marker systems (ANTS, Oxford NOTECHS II, TeamSTEPPS TPOT/PACT, Co-ACT) | 37 | 0 | 53 | Medium | use-as-reference |
| 10 | Observer-rated transformational leadership from video (German TLI) | 23 | 0 | 47 | High | cite-only |
| 11 | PISA 2015 / ATC21S collaborative problem solving frameworks | 7 | 0 | 44 | Low | cite-only |

### Peer / other-report instruments

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Other-/observer-rated Big Five (acquaintances, strangers, thin slices) | 61 | 0 | 73 | High | use-as-reference |
| 2 | CATME / CATME-B (peer evaluation with behavioural anchors) | 28 | 12 | 52 | Low | use-as-reference |
| 3 | Multifactor Leadership Questionnaire (MLQ) | 48 | 12 | 47 | Medium | use-as-reference |
| 4 | Follower-rated leader-behaviour questionnaires (LBDQ; Yukl's task/relations/change taxonomy) | 48 | 0 | 47 | Medium | use-as-reference |
| 5 | Multisource (360-degree) feedback ratings | 39 | 0 | 44 | Low | use-as-reference |

### Automatic methods

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Automated coding of collaborative problem solving talk (Pugh 2021; Hao 2025; Venkatesha 2025) | 35 | 6 | 90 | Medium | use |
| 2 | Emergent-leader detection from speaking activity (ELEA; speaking time) | 19 | 0 | 87 | Medium | avoid |
| 3 | Conversational uptake (Demszky et al.) | 25 | 6 | 83 | High | use-as-reference |
| 4 | ML scoring of transcribed AC role-plays (Hickman et al. 2023) | 23 | 16 | 83 | High | use |
| 5 | Conversational receptiveness and follow-up questions (Yeomans; Huang) | 21 | 6 | 70 | Low | use-as-reference |
| 6 | Automatic negotiation analysis (outcome prediction; LLM tactic annotation) | 21 | 0 | 63 | Low | use-as-reference |
| 7 | LLM zero-shot personality rating from interview transcripts (Zhang et al. 2024) | 14 | 6 | 63 | High | avoid |
| 8 | Politeness classifier (Danescu-Niculescu-Mizil et al.) | 13 | 6 | 57 | High | cite-only |
| 9 | Automated interview-performance prediction (Naim et al. 2015) | 10 | 6 | 57 | High | avoid |
| 10 | Personality from language (Mairesse; Schwartz; Park) | 33 | 0 | 56 | Medium | avoid |
| 11 | Automated video-interview personality assessment (Hickman et al. 2022) | 35 | 6 | 55 | High | use-as-reference |
| 12 | Apparent personality from short video (ChaLearn First Impressions) | 0 | 0 | 47 | Medium | avoid |
| 13 | Hybrid AI-expert O*NET Work Style ratings | 25 | 14 | 29 | High | use-as-reference |
| 14 | Sociometric badges / 'honest signals' | 0 | 0 | 19 | Low | avoid |
| 15 | LLM-as-a-judge (chat answers, essays) | 19 | 6 | 18 | High | use-as-reference |

### Questionnaires and tests

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Big Five / FFM self-report inventories (NEO PI-R/3, BFI-2, IPIP, HEXACO) | 55 | 75 | 45 | High | use-as-reference |
| 2 | Belbin Team Roles (SPI and observer forms; Interplace II) | 24 | 0 | 44 | Medium | avoid |
| 3 | Hogan HPI / HDS / MVPI | 27 | 8 | 30 | Medium | cite-only |
| 4 | SHL OPQ32 | 27 | 8 | 30 | Low | cite-only |
| 5 | CompeTEA (TEA Ediciones) (popular-but-weak) | 26 | 70 | 30 | Medium | use-as-reference |
| 6 | 16PF-5 (Spanish TEA edition) (popular-but-weak) | 19 | 63 | 30 | Medium | cite-only |
| 7 | Myers-Briggs Type Indicator (popular-but-weak) | 0 | 15 | 30 | Medium | avoid |
| 8 | DISC | 0 | 0 | 30 | Low | avoid |
| 9 | Self-report EI (EQ-i 2.0, TEIQue, TMMS-24) (popular-but-weak) | 51 | 12 | 24 | Low | cite-only |
| 10 | MSCEIT (ability emotional intelligence) | 47 | 12 | 24 | Medium | cite-only |
| 11 | Team psychological safety scale (Edmondson) | 66 | 8 | 23 | Low | use-as-reference |
| 12 | Gallup CliftonStrengths | 0 | 0 | 20 | Low | avoid |

### Competency frameworks and taxonomies

| Rank (by C) | Model / instrument / method | A evidence | B acceptance | C fit | Confidence | Verdict |
|---|---|---|---|---|---|---|
| 1 | Negotiation competency model (Smolinski & Xiong) | 0 | 6 | 68 | High | use-as-reference |
| 2 | CONOCER EC0554.01 Teamwork standard (Mexico) | 0 | 12 | 65 | High | use-as-reference |
| 3 | UK Civil Service Success Profiles: Behaviours | 0 | 14 | 52 | High | use-as-reference |
| 4 | Korn Ferry Leadership Architect (Lominger) | 8 | 8 | 41 | Medium | cite-only |
| 5 | LifeComp (and EntreComp), EU JRC frameworks | 0 | 6 | 37 | Medium | use-as-reference |
| 6 | NACE career readiness competencies | 0 | 0 | 35 | Medium | cite-only |
| 7 | Hogan Competency Model | 10 | 8 | 32 | Medium | cite-only |
| 8 | McClelland / Spencer & Spencer / Boyatzis competency approach | 2 | 0 | 31 | Low | cite-only |
| 9 | O*NET Work Styles (and skills) | 45 | 14 | 29 | High | use-as-reference |
| 10 | DDI Success Profiles | 2 | 8 | 27 | Low | cite-only |
| 11 | ESCO transversal skills | 0 | 14 | 27 | Low | cite-only |
| 12 | Competency modelling as practised (Campion et al. best practices; SIOP task force) (popular-but-weak) | 14 | 47 | 25 | Medium | use-as-reference |
| 13 | SHL Universal Competency Framework / Great Eight | 31 | 8 | 24 | Low | cite-only |
| 14 | Marks-Mathieu-Zaccaro team process taxonomy (LePine et al. meta-analysis) | 48 | 0 | 23 | Low | use-as-reference |
| 15 | Google Project Aristotle (team effectiveness dynamics) | 2 | 8 | 23 | High | cite-only |
| 16 | Collective intelligence factor (Woolley et al.) | 21 | 0 | 20 | Medium | cite-only |
| 17 | Big Five of teamwork (Salas, Sims & Burke) | 9 | 0 | 18 | Low | cite-only |
| 18 | SFIA 9 behavioural factors | 0 | 0 | 12 | Low | cite-only |

### Why the top entries rank where they do

**Assessment centre (A 69 / B 74 / C 93, use).** This is the only method that is strong on all three scores.
- Native mode: trained observers rating interaction. The Guidelines allow rating from recordings (`rupp2015`, p. 1251).
- Criterion validity: re-estimated ρ = .29 (`sackett2022`); dimensions add ΔR² = .10 over cognitive ability plus the Big Five (`meriac2008`).
- Weaknesses: exercise effects swamp dimension effects (`jackson2016`, `hoffman2011`; in Putka & Hoffman 2013 exercise-related effects were 56.8% of between-candidate variance against 23.0% for all assessor-related effects, verified), and the role player's behaviour shifts the candidate's (`oliver2014`).
- A fully automated scorer "could no longer be referred to as an assessment center" (`rupp2015`, Section X).

**Multiple speeded assessments (23 / 6 / 90, use-as-reference).**
- 18 three-minute role-plays, aggregated, predicted later ratings at .54 to .57.
- Single role-plays were not reliable (single-assessor G = .38), and their validity averaged only .26 (`herde_lievens2023`, verified).

This is the design template: many short conversations rather than one long one. The evidence is a single dataset of 96 MBA students.

**Leaderless group discussion and emergence (19 / 37 / 100, avoid).** It gets maximum fit because it is observer-rated, individual, and its automatic proxy (speaking activity) is replicated. But:
- There is no work-criterion validity. Emergence is perception, not effectiveness.
- It rewards talkativeness: speaking time correlated .67 with emergence votes (zero-order, article p. 6), and being male was worth about one additional expected vote, roughly 45 s of speaking (p. 8) (`maclaren2020`, verified in full text). A speaking-time classifier picked out female emergent leaders far less often than male ones (29% vs 81%, p. 10).

It is the clearest case of a trap to avoid: a construct that fits the product perfectly and has no validity for the claim it would be used for.

**Automated CPS coding (35 / 6 / 90, use).**
- Utterance-level collaboration codes from speech agree with trained coders: AUROC .80 on ASR transcripts, .91 on human transcripts (`pugh2021`).
- Two abstract-level replications: `hao2025_chatgpt_cps` and `venkatesha2025_cps_asr`.
- Limits: students only, and no outcome validity yet.

**ML scoring of AC role-plays (23 / 16 / 83, use).** This is the validation template:
- the machine is one rater among humans;
- contamination is checked (human scores correlate r = .44 with word count; some models .69-.88);
- a later criterion is used.

**Conversational uptake (25 / 6 / 83, use-as-reference).**
- A transcript-native measure of a speaker building on the other's turn, ρ = .54 with expert labels.
- The experts themselves agreed only κ = .29.

**Sun et al. CPS model (33 / 6 / 75, use).**
- Individual indicators coded within triads, with discriminant evidence against personality and intelligence (`sun2020`).
- The best seed for teamwork indicator cards.

**Observer-rated Big Five (61 / 0 / 73, use-as-reference).**
- The credible way to keep "personality" in scope, defined as personality perceived by observers.
- The ceiling from text is low (.11-.28).

**NegotiAct (23 / 6 / 70, use).**
- Open, synthesised from 89 earlier schemes, κ = .80.
- Validation is tiny and at dyad level.
- The licence is CC BY-NC: commercial use needs permission.

**Structured interview (67 / 77 / 59, use-as-reference).**
- Best-validated way of scoring spoken answers (.42), and widely used: UK 69.4%, Spanish SMEs 52%.
- C is low because answers describe other situations. The one transcript-LLM test was negative (`zhang2024`).

**Frameworks.** None has validity evidence for person scoring. Three are worth reusing as text:
- **CONOCER EC0554** (C 65): Spanish-language observable teamwork criteria.
- **Smolinski & Xiong** (C 68): negotiation indicators with proficiency levels.
- **UK Civil Service Behaviours** (C 52): Open Government Licence, level-graded, picked per job.

**O*NET** (A 45) is the open job-profile side.

**Questionnaires.** **CompeTEA** is the only Spanish-native instrument with leadership and negotiation scales. It is self-report. Use it as a convergent measure in validation, not as the scoring basis.

### Popular but weak (why)

- **CompeTEA (A 26 / B 70).**
  - Used by 21.7% of Spanish work psychologists (`muniz2020`), and the COP review rated its predictive validity 3.5/5 (`cop2013`).
  - But the manual extract lists only "country and administration format" as criteria, and no job-performance figure was seen. The two records disagree; the COP report itself did not download.
- **16PF-5 (19 / 63).**
  - The most used test by Spanish work psychologists (27.5%) and in the COP Madrid survey.
  - The COP review shows predictive validity "-" (none reported in the manual) and no bias analysis (`cop2011`).
- **Unstructured interview (37 / 53).**
  - Used by 90.7% of Galician SMEs and 41.8% of UK organisations.
  - Re-estimated validity is .19 (`sackett2022`).
  - HR executives still rate it above tests (`highhouse2008`).
- **Competency modelling as practised (14 / 47).**
  - Competency interviews and scales are widespread in Spain.
  - Its own best-practice authors call it "experience based rather than empirically based" (`campion2011`). Untrained raters' competency inferences have poor inter-rater reliability (`lievens2004`, secondary).
- **MBTI (0 / 15).**
  - About 35% change at least one letter on retest after about a month (`pittenger2005`, secondary).
  - SIOP says types should not be used for hiring (`siop2021_persona`).
  - Only 20.1% of US/Canadian HR respondents rejected the MBTI myth, against 49% in 2002 (`fisher2019_selection_myths`).
- **Self-report EI and graphology** are popular as beliefs:
  - 52.7% of HR respondents believe EI predicts better than cognitive ability;
  - 47.0% do not reject graphology (`fisher2019_selection_myths`).

  Self-report EI has criterion validity (.30), but it overlaps personality (incremental ΔR² .002, secondary). Graphology has no evidence at all.
- **Belbin and DISC** cannot be called "popular" from these sources: no usage survey was collected.
  - Belbin: weak discriminant validity (`vandierendonck2008`, `aritzeta2007`) and no criterion validity.
  - DISC: no source record at all.

  Excluding both is consistent with the evidence.

---

## 3. The central tension, with figures

**1. What is validated and adopted is mostly self-report or interview.**
- In Spain, psychologists most often use 16PF (27.5%) and CompeTEA (21.7%), both self-report (`muniz2020`).
- Personality questionnaires: 40% of Spanish SMEs, 25.6% of UK organisations.
- The validity of those questionnaires is modest: conscientiousness .19-.25, the rest ≤ .10 (`sackett2022`).
- Faking cannot be avoided (`morgeson2007`).

**2. Direct observation is the right mode for this product, but its evidence has four limits.**

*a. It sits at team or dyad level.*
- act4teams: criterion links to team satisfaction, productivity and success 2.5 years later (`kauffeld2012`, abstract).
- NegotiAct: one dyad-level link, rs = .50, N = 18.
- Clinical marker systems: mostly team level, with weak criterion links (NOTECHS II r = −.26 with glitches, `robertson2014`).

*b. It depends on trained human raters.*
- About 200 h of training per act4teams coder.
- AC Guidelines: one day of training for a simple AC, 2 days per AC day for novices (`rupp2015`, p. 1257).
- Frame-of-reference training improves accuracy: d = .83 and .50 (secondary).
- Training mattered more than team size (`wirz2013`).
- Untrained or single raters are poor. A transformational-leadership questionnaire rated from video gave ICC1 = .28 for one rater and ICC2 = .66 for six (`lehmannwillenbrock2015`).
- A single peer rating has inter-rater reliability of only .37 (`conway1997`, secondary).

*c. It needs several situations per person (generalizability).*

| What is generalised | Figure | Source |
|---|---|---|
| Across assessors, single rating | .80 | `jackson2016`, full text |
| Across assessors, overall score | .97 | `jackson2016` |
| Across exercises, single rating | .26 | `jackson2016` |
| Across exercises, dimension over 3 exercises | .55 | `jackson2016` |
| Across exercises, overall over 3 exercises | .65 | `jackson2016` |
| Person × dimension share of variance | .13-1.11% | `jackson2016` |
| Second operational AC (3 samples, N = 633; 3–4 exercises, 2 assessors per exercise) | one assessor .76, two .87; same dimension in two exercises .27 (.18 with different assessors); a dimension averaged over 3 exercises against a new set of 3, .52 (.36 with a different assessor); person × dimension 0.4–1.4% of between-candidate variance | Putka & Hoffman 2013 (verified, JAP 98, Results, Tables 3–5) |
| Single assessor, one 3-min role-play | G = .38 (`hickman2023`); single-rater ICC .27–.42 across rating conditions (`herde_lievens2023`, manuscript p. 32) | `hickman2023`, `herde_lievens2023` |
| 18 role-plays aggregated, criterion validity | .54 / .57; single role-plays averaged .26 (range .03–.43) | `herde_lievens2023` (verified, manuscript pp. 2, 21; N = 96, criterion 7 months later) / `hickman2023` |
| Rating only the first minute | validities .43–.56 across four conditions; lowest (.43) with highly standardised raters and 1 min | `herde_lievens2023` (verified, manuscript p. 33) |
| Distinct social skills, 2 exercises per skill | skill dimensions 20% of reliable variance per rating, 28% at dimension level; about four exercises per skill recommended | `breil2022` (verified, EJPA 38, pp. 230, 233; N = 589) |
| Personality impressions across 15 episodes | .43 (range .29–.61); little gain beyond ~6 | `borkenau2004` (verified, JPSP 86, pp. 599, 604; N = 600) |
| Expressiveness, aggregated over 30-s sequences | ICC(2,k) > .89, > 49% within-person variance | `bleckmann2025` |

*My projection with Spearman-Brown, not stated in any source.* A per-exercise generalizability of about .38 is implied by .65 over 3 exercises. On that basis:
- about 4 comparable situations reach .70;
- about 6-7 reach .80.

The conclusion: observers are not the bottleneck; situations are.

*d. Accuracy falls sharply with text-only material.* Single-rater consensus among strangers (`connelly2010`, Table 4; k = 3-4 for text):

| Channel | Emotional stability | Extraversion | Openness | Agreeableness | Conscientiousness |
|---|---|---|---|---|---|
| Text / electronic communication only | .11 | .28 | .14 | .23 | .16 |
| Audio only | .38 | .53 | .24 | .43 | .30 |
| Natural audio-visual behaviour | .38 | .59 | .36 | .39 | .43 |

Two more cautions:
- The act4teams team rated leadership from video because transcripts capture interpersonal constructs less accurately (Nicolai et al. 2010, as cited in `lehmannwillenbrock2015`).
- NegotiAct coded from video because back-channel "mm-hmm" active listening is lost in transcripts.

**3. What automatic scoring has actually reached, and against which human criterion.**

| Method | Human criterion | Agreement / validity reached | Source |
|---|---|---|---|
| ML on manual transcripts of AC role-plays | trained assessors | about one assessor's reliability; OAR .76-.79; later criterion .47 (ML alone), .55 (with role player) vs .57 (panel) | `hickman2023` |
| AVI personality | interviewer ratings vs self-report | convergent .40 vs .12; test-retest .50 / .46 | `hickman2022` |
| LLM zero-shot on interview text | observers / self-report | comparable to a trained model for traits; test-retest "insufficient"; R² < 0 for competences | `zhang2024` |
| CPS utterance classifier | trained coders (ICC .93) | AUROC .80 (ASR) / .91 (human transcripts) | `pugh2021` |
| Conversational uptake | expert annotators (κ .29) | ρ .54 | `demszky2021` |
| Speaking activity | peer-perceived emergent leader | up to 80% identification | `sanchez2010` |
| Interview "performance" | untrained crowd (α .19 overall) | r > .65 | `naim2015` |
| LLM judge of chat answers | crowd preferences | > 80% agreement (human-human level), with position/verbosity biases | `zheng2023` |
| LLM work-style ratings of occupations | expert panels (ICC .86) | r .84 vs analysts .76 | `onet_hybrid_ai_2025` |

No study rated soft skills from **workplace** transcripts against trained observers and a job-performance criterion.

---

## 4. Proposal for the first prototype

Scope: personality, negotiation and leadership, plus core teamwork. Each part is split into **[S] supported by sources** and **[J] my judgement**.

### Common design (all four)

[S] Facts the design rests on:
- Several situations per person (`jackson2016`, `herde_lievens2023`).
- Several trained raters with FOR training (`rupp2015`).
- Automated scores must be compared with two independent human raters (`itc2022`, p. 56).
- Word-count contamination is the main threat (`hickman2023`).
- The counterpart shapes behaviour (`oliver2014`).
- Emotion inference from voice is prohibited at work (`aiact_art5`, `ec_guidelines_prohibited` para 245, 251).

[J] Design choices:
- Work on the text plus speaker labels and timings only. Use no prosody or facial cues.
- Score behaviour **frequencies and sequences per speaker turn** using indicator cards (§8). Aggregate to a person score only across ≥ 4 situations, and show the reliability with the score.
- Present the output as **evidence for a human rater**: highlighted turns mapped to indicators. Do not present it as a verdict.

### Personality

- **Build on:**
  - observer-rated Big Five, `connelly2010`;
  - Riverside Behavioral Q-sort, `funder2000_rbq`, coefficients still to be obtained;
  - expressiveness coding, `bleckmann2025`.
  - If self-reported personality is wanted, use a short open questionnaire (IPIP, `goldberg2006`, secondary) as a separate input.
- **Codable from transcript [J]:**
  - share and length of turns;
  - initiating topics and questions (extraversion-related, observable);
  - building on others (`demszky2021`);
  - politeness / receptive language (`danescu2013`, `yeomans2020`).
- **Must NOT claim:**
  - a Big Five self-report score, emotional stability, or any emotion;
  - anything inferred from voice (Art. 5(1)(f)).

  [S] The text-only ceiling (.11-.28) and the failure of self-report-trained models (.12) both say no.
- **Validation study [J]:**
  - Design: 5+ trained observers rate each person's transcripts (and separately the audio/video); people also complete IPIP; and 2+ acquaintances rate them.
  - Success: machine–observer agreement ≥ the mean single observer–observer agreement on the same material (ITC logic). Below that, report nothing.
  - Sample: see the sample-size note at the end of this section.

### Negotiation

- **Build on:**
  - NegotiAct codes (`jaeckel2024_negotiact`, manual saved);
  - Adair & Brett, a scheme natively coded from transcripts (`adair2005`, secondary);
  - Smolinski & Xiong indicators for anchors (`smolinski2020`).
- **Codable [S, from NegotiAct categories]:**
  - asking for and providing priority information;
  - single- and multi-issue offers;
  - persuasive arguments;
  - procedural statements;
  - paraphrasing the other party.

  Not codable from text alone: lying and positional misrepresentation, which need role instructions; back-channel listening if ASR drops it.
- **Must NOT claim:** a "negotiation skill" score from one negotiation. *Corrected 2026-11-15 from the full text:* the earlier line "28% individual vs 42% dyadic" is not in `elfenbein2008`. In that round-robin study (N = 106 in the variance analysis), consistent individual differences explained 46% of objective outcomes (28% the negotiator's own effect, 19% the effect they have on counterparts), but dyadic effects could not be separated from error for objective outcomes; for subjective value, individual differences were 19% and the dyad 28% (JRP 42, Table 1, p. 1466; p. 1468). The 2018 follow-up figure remains unverified. No individual-level criterion validity exists.
- **Validation study [J]:**
  - Design: round-robin. Each participant negotiates a standard multi-issue case with 3-4 different counterparts. Two trained coders per transcript.
  - Criteria: joint and individual outcomes, separated with a social-relations-model analysis.
  - Success: code-level κ machine vs coder ≥ .70 (NegotiAct human level .80); person-level consistency across counterparts measurable; behaviour patterns predicting outcomes replicate the direction of `jaeckel2024_negotiact` (multi-issue offer → active listening).

### Leadership

- **Build on:**
  - the AC "influencing others" dimension (`meriac2008`, ΔR² .013);
  - Twente leader-behaviour codes (`hoogeboom2012`): directing, structuring, verifying, informing, visioning, intellectual stimulation, individualized consideration, listening;
  - act4teams leader solution-focused talk (`lehmannwillenbrock2015`);
  - UK Civil Service Leadership text (OGL);
  - MLQ or LBDQ as follower criteria (`derue2011`).
- **Codable [S]:**
  - procedural structuring (goal orientation, summarising, task distribution, time management; `bavendiek2015`);
  - solution proposals and the members' responses that follow them (lag-1 z = 4.01, `lehmannwillenbrock2015`);
  - questions and verifying.
- **Must NOT claim:**
  - leadership from speaking time or participation share (`maclaren2020`);
  - effectiveness from behaviour frequencies alone (`hoogeboom2012`: no difference between highly and moderately effective leaders).
- **Validation study [J]:**
  - Design: recorded real meetings of 60+ leaders, at least 4 meetings each, 3 trained coders.
  - Criteria: follower MLQ/LBDQ, which is other-report, and an independent superior effectiveness rating collected later.
  - Success:
    - the person score across meetings reaches G ≥ .70;
    - machine–coder agreement reaches the human level;
    - the score relates to the independent criterion after controlling talk time;
    - gender differences in scores are reported with impact ratios.

### Core teamwork

- **Build on:**
  - Sun et al. CPS indicators (`sun2020`), which are individual-level;
  - automated CPS coding (`pugh2021`);
  - act4teams categories (licence needed);
  - CONOCER EC0554 Spanish criteria (`conocer_ec0554`);
  - CATME anchors for peer ratings.
- **Codable [S]:**
  - constructing shared knowledge (sharing information, building on ideas);
  - coordination (proposing plans, assigning tasks);
  - maintaining team function (validating contributions, inviting others);
  - counterproductive talk (criticising, complaining; `kauffeld2012`).
- **Must NOT claim:**
  - an individual effect on team performance (validity is team-level);
  - stable traits from one meeting.
- **Validation study [J]:** as for leadership, with team outcome criteria and CATME peer ratings from 3+ peers per person. A single peer rating is unreliable (.37).

### Sample-size reasoning

All of this is my derivation from the collected figures.

1. **Convergence and criterion correlations.** Hickman's N = 96 gave r = .47 with 95% CI [.30, .61]. For a half-width of about ±.10 around r ≈ .5, Fisher-z gives N ≈ 220 people. About 100 is acceptable only for an exploratory first pass.
2. **Raters.**
   - With single-rater reliability .28 (leadership questionnaire rated from video), Spearman-Brown gives about 6 raters for .70 and about 10 for .80.
   - With AC-type trained assessors (.80), 2-3 suffice.

   Pilot the human rater reliability first: it fixes the ceiling for the machine.
3. **Situations.** About 4 per person for .70, about 6-7 for .80 for an overall score (from `jackson2016`). A single dimension generalises worse: about .27 per exercise (Putka & Hoffman 2013, verified), which by my Spearman-Brown projection needs about 6 exercises for .70.

**What counts as success overall [J]:**
- machine–human agreement not below human–human agreement;
- person-level G ≥ .70 across situations;
- criterion validity no more than about .05 below the human panel (as in `hickman2023`);
- no residual relation to word count beyond what humans show (.44);
- subgroup impact ratios reported and acceptable under the 4/5 rule (`cfr1607`).

---

## 5. How competencies link to job profiles

**What the frameworks do:**
- **Hogan:** subject-matter experts rate each competency's relation to job success, with SME inter-rater .92/.89 on N = 9 (`hogan_sales_techreport`). Synthetic validity then comes from an archive of 250+ jobs (vendor claim).
- **SHL:** equations turn O*NET job data into competency profiles. These converge r = .86 with incumbent profiles over 125 jobs (vendor, `shl_ucf_whitepaper`).
- **O*NET:** profiles each occupation as a vector of work styles and skills, with expert inter-rater ICC(C,5) .86. An LLM reproduces expert ratings at r = .84 (`onet_hybrid_ai_2025`).
- **UK Civil Service:** recruiters pick the behaviours relevant to a job, at grade-specific levels (`ukcs_success_profiles`).
- **SFIA:** maps skills to role levels.
- **The competency-modelling literature:** prescribes job analysis (interviews, focus groups, surveys), behavioural indicators, then mapping to jobs and levels (`campion2011`). It also warns that linkage judgements by untrained raters are unreliable (`lievens2004`, secondary).

**What this implies [J]:**
- A job profile in the product should be a **weighted list of indicator-based competencies**. Build it from a short job analysis, with O*NET or an LLM draft reviewed by SMEs.
- The weights must be documented and published before assessment. The Spanish psychologists' council asks for this for selection: Rec. 4.3, `elosua2025_cop_recomendaciones`, secondary via Infocop.
- The link is itself a judgement that needs reliability checks: several SMEs, and ICC reported.
- Person scores must never be "fitted" to a profile beyond the reliability of each score.

---

## 6. Market questions

**Evidence for demand:**
- 72.3% of 1.7 M Catalan job ads ask for some soft skill (`duran2024`).
- Employers in vendor surveys say soft-skill failures are common: 78% report a technically strong hire who failed on soft skills (`testgorilla2025`, vendor).
- 56% of SHRM respondents use pre-employment assessments (`shrm2022_assessments`, vendor-side).
- Structured observational methods grow with organisation size:
  - Spain: AC 11.9% of mid-size vs 3.0% of small firms (`alonso2015`);
  - UK: use rises with organisation size (`zibarras2010`).

**Evidence against or for caution:**
- **Candidates:**
  - Algorithmic decisions are judged less fair: k = 21, N = 7,818, secondary (`moritz2023`).
  - Behavioural control and social presence are perceived negatively (`robots2022`).
  - Automatic interview evaluation worsens reactions (`langer2020`, secondary).
  - 38% of US candidates say they withdrew because of an AI interview, and 70% were not told (`greenhouse2026_ai_interviews`, vendor).
  - Spanish candidates in 2020 were 76% positive about new tools such as video and personality tests (`michaelpage2020_candidatos`, vendor).
- **HR practitioners:**
  - Recruiters feel threatened, and AI assessment meets resistance (`li2021_aies_recruiters`, 15 interviews).
  - Recruiters state distrust of algorithms yet follow an inconsistent algorithmic recommendation more often (55.4% vs 40%; `lacroux2022_trust`). This is a liability risk, not a selling point.
  - About half of HR leaders do not monitor their AI tools (`greenhouse2023_ai_survey`, vendor, ~100 respondents).
- **Practice in Spain:**
  - 5% of firms use AI in selection now, 16% use or plan to; the main objection is loss of empathy and intuition (40%) (`infojobs2023_ia`, vendor, N not stated).
  - 56% of firms hold no technical documentation for their tests (`cop_madrid_encuesta2015`).
  - Practitioners choose tests by popularity and network, not evidence (`diekmann_konig_personality`); HR beliefs often diverge from research (`fisher2019_selection_myths`).

  So rigorous validation may not by itself sell. It does reduce legal risk.
- **I-O psychologists:** they use GenAI themselves (> 90% monthly) and insist AI should go with I-O knowledge, not replace it (`siop_tip63_2_genai_survey`).

**Unknown (no evidence found):**
- whether high-volume employers are the main buyers (only the "applicant glut" framing in `li2021_aies_recruiters`);
- willingness to pay;
- employee acceptance of recorded meetings used for peer evaluation or development;
- Spanish HR acceptance of automated behavioural assessment specifically;
- whether "aid to the psychologist/assessor" positioning sells better than "replacement". `hunkenschroer2021` suggests augmenting framing may improve fairness perceptions; effect sizes were not extracted.

---

## 7. Regulation

All EU texts were read from **unofficial reproductions**: EUR-Lex returned a bot challenge. Verify wording against OJ L 2024/1689 before relying on it.

**AI Act Art. 5(1)(f)** (`aiact_art5`).
- What it says: it prohibits AI systems "to infer emotions of a natural person in the areas of workplace and education institutions", except for medical or safety reasons.
- The Commission guidelines interpret it as follows (`ec_guidelines_prohibited`; copy hosted by a law firm; non-binding):
  - It covers inference from **biometric** data only (voice, face, keystrokes), para 245.
  - "An AI system inferring emotions from written text … is not based on biometric data", para 251.
  - Mere detection of a raised voice is not covered unless used to infer emotion, para 249.
  - "Workplace" includes candidates and probation.
  - Inferring emotional tone in hybrid team calls from voice or imagery is given as a prohibited example, para 254.
- Consequence: score text only. Do not label emotions, stress or "attitude" from audio.

**Art. 3(39)** (`aiact_art3`). Defines an emotion recognition system as one that infers emotions or intentions from biometric data (verified, OJ p. 48/144; recital 18, p. 5, lists examples and excludes mere detection of a raised voice). Note: Art. 5(1)(f) itself (OJ p. 51) does not mention biometric data; the biometric limit comes from recital 44 (p. 12) and the guidelines.

**Annex III 4(a),(b)** (`aiact_annex3`; verified against the Official Journal text, OJ p. 127/144). 4(a) covers recruitment and selection, including evaluating candidates; 4(b) covers decisions on work terms, promotion and termination, task allocation "based on individual behaviour or personal traits", and monitoring and evaluating performance and behaviour. *Correction:* the quoted phrase belongs to task allocation; a skills tool falls under the "evaluate performance and behaviour" clause. A peer-evaluation or skills tool used by employers falls here.

**Arts. 6(3), 26, 4 and 113** (read 2026-11-15 in the OJ text):
- Art. 6(3) (p. 54): an Annex III system can escape high-risk status only under narrow conditions, and **never if it profiles natural persons**. Scoring an individual's skills is very likely profiling, so the derogation should not be relied on (our reading; legal review pending).
- Art. 26 (pp. 67–69): deployers must follow the instructions for use, assign competent human oversight, and, if they are employers, **inform workers' representatives and the affected workers before use** (26(7)).
- Art. 4 (p. 51): providers and deployers must ensure sufficient AI literacy of their staff.
- Art. 113 (p. 123): the Regulation applies from 2 August 2026; Chapters I–II (including Arts. 4 and 5) from 2 February 2025.

**GDPR** (`gdpr_art22`, `gdpr_art9`; unofficial):
- Art. 22(1): no decision based solely on automated processing with legal or similarly significant effects. Where exceptions apply, 22(3) requires human intervention, a right to express a view, and a right to contest.
- Art. 9(1): biometric data are a special category when processed "for the purpose of uniquely identifying". Diarisation or speaker identification may need legal review. The AI Act and GDPR definitions of biometric data differ (guidelines fn. 160).

**US** (in case of US clients):
- 29 CFR 1607.4(D), the four-fifths rule (`cfr1607`).
- NYC Local Law 144 final rule (`nyc_ll144_rule`):
  - an "automated employment decision tool" is one that substantially assists or replaces discretionary decisions;
  - an annual independent bias audit is required, with impact ratios by sex, race/ethnicity and intersections;
  - verified 2026-11-15 in the DCWP final rule (§§ 5-300 to 5-304, pp. 3–10): impact ratio = a category's selection rate divided by that of the most selected category, or, for scores, the share above the sample median divided by that of the highest-scoring category (p. 4); an auditor may leave out a category under 2% of the data, with a justification and its counts still reported (p. 7); results must be posted before use and kept up at least 6 months after last use (p. 9); notice at least 10 business days before use (pp. 9–10);
  - enforcement from 5 July 2023 (not stated in the rule text read; unverified).
- The Illinois AI Video Interview Act and the EEOC 2023 guidance were seen only as summaries.

**Professional standards:**
- ITC/ATP 2022 (`itc2022`): automated scoring should be compared with at least two independent human raters, and biases tested.
- AC Guidelines (`rupp2015`): fully automated scoring is not an AC.
- SIOP 2023 AI-based assessment recommendations (read in full 2026-11-15): AI-based assessments should meet the same standards as traditional employment tests (p. 6); a vendor algorithm validated elsewhere needs evidence for the new context (p. 13); differences in microphones, cameras or connectivity are a source of measurement bias, and removing subgroup differences inside the algorithm can leave that bias unresolved and may be unlawful in the US (p. 19).
- Spanish council 2025 recommendations (`elosua2025_cop_recomendaciones`, secondary): justified and published scoring and weights, candidate access to results, qualified psychologists. They do not mention AI.
- ISO 10667: catalogue level only.

---

## 8. Knowledge-base design

### Evaluation of the two-layer recommendation

The recommendation already given was: structured cards for retrieval at scoring time, plus raw passages as an audit/evidence layer.

**Right in principle:**
1. Retrieving raw studies at scoring time would let the scorer "reason from" findings that do not hold at individual level. Speaking time → leadership is the example.
2. Cards force the separation of reliability from validity and carry the verification status.
3. Passages give a reviewing psychologist something to audit.

**What I would change, given what is now known:**
1. **Scoring needs indicators and anchors, not evidence.** At scoring time, retrieve only indicator cards and competency cards. Evidence cards and passages serve reports, audit and human reviewers. They must never be in the scorer's prompt as instructions.
2. **Cards are an interpretation layer, so errors propagate silently.** Every claim on a card must point to a passage id and carry `verified: full-text | abstract-only | secondary`. Cards need versions and a reviewer sign-off.
3. **The evidence layer will have holes.** 148 of the 263 records (56%) were not read in full text. Cards must say "not found" explicitly rather than stay silent.
4. **Licences limit what the evidence layer may contain.**
   - Paywalled and CC BY-NC texts cannot be shipped in a commercial product: keep full texts private for internal audit, and expose only citations and short quotations.
   - Gated vendor content (SHL, Hogan, Korn Ferry, DDI, CompeTEA) must not enter the cards beyond names.
5. **Sanitise the passages.** One saved PDF carries embedded text aimed at AI summarisers. Passages must be stored as quoted data, stripped of such text, and never concatenated into a prompt as instructions.
6. **Indicators must encode context.** The cards need fields for counterpart effects, the minimum number of situations, transcript observability (is it lost without audio?) and known contaminants (word count).

**Verdict:** two layers, built from a synthesis, are right. Make it three card types plus a passage store, and keep the scorer's retrieval limited to indicators and competencies.

### (a) Behavioural indicator card

Schema:

```yaml
id:                 # stable slug
name:
definition:         # observable, in the source's words where licence allows
examples: []        # utterance examples
delimit: []         # what does NOT count
unit:               # utterance | turn | turn-pair | sequence; speaker-attributed
requires:           # e.g. preceding turn by another speaker; role instructions; timing
transcript_observability:   # full | partial (lost without audio) | none
competencies: []    # competency card ids, with direction (+/-)
reliability:        # [{type, value, source, verified}]
validity:           # [{type, value, level: individual|dyad|team, criterion, source, verified}]
contaminants: []    # word count, counterpart behaviour, talk time
licence:            # e.g. CC BY-NC: permission needed for commercial use
source_passages: [] # passage ids
review: {status, reviewer, date}
```

Filled example, built only from `jaeckel2024_negotiact` and its saved manual:

```yaml
id: negotiact.active_listening
name: Active listening (NegotiAct, socio-emotional category)
definition: "Paraphrasing the other party's statements and generic paraverbal responses such as 'mm hmm' or 'yeah'"
examples: ["I understand.", "Mm hmm.", "Hmm, I see.", "I see where you're going with this."]
delimit: ["Also: repeating the other's statement",
          "Not: 'Good' or 'Okay' as a signal to proceed (Encouragement)",
          "Not: agreeing with the other party's substantiation, e.g. 'Yes, indeed!' (Positive relationship remark)"]
unit: thought unit attributed to one speaker; needs the other party's preceding turn
requires: diarised transcript; back-channels preserved by ASR
transcript_observability: partial  # authors coded from video because paraverbal cues are better recognised there
competencies: [{id: negotiation, direction: +}]
reliability: [{type: inter-coder, value: "Cohen's kappa .80 for the whole scheme; per-code agreement 66.7-100% (value for this code not recorded)", source: jaeckel2024_negotiact, verified: full-text}]
validity: [{type: criterion (exploratory), level: dyad, value: "multi-issue offer -> active listening sequence rs = .50 with joint gains (p = .03); active listening alone .42 n.s.", sample: "18 negotiations, German students", source: jaeckel2024_negotiact, verified: full-text}]
contaminants: [counterpart behaviour (dyad-level validation), ASR dropping back-channels]
licence: CC BY-NC (article and manual); commercial use needs permission
review: {status: pending psychologist review}
```

### (b) Competency card

Schema:

```yaml
id:
name:
definition:          # sourced text, licence-compatible
levels: []           # level-graded behavioural descriptions
indicators: []       # indicator card ids, with weights set by psychologist
not_claimed: []      # explicit exclusions
evidence_summary: {reliability: [], validity: [], unit_of_validation:}
min_situations:      # from generalizability evidence
job_profile_link:    # how it is selected/weighted per job
sources: []
review: {}
```

Filled example. Text from `ukcs_success_profiles` (OGL v3.0); evidence from `meriac2008`, `demszky2021`, `jackson2016`:

```yaml
id: communicating_and_influencing
name: Communicating and Influencing (UK Civil Service Behaviours)
definition: "Communicate clearly and concisely both orally and in writing ... Listen to and value different ideas, views and ways of working. Respond constructively and objectively to comments and questions. Handle challenging conversations with confidence and sensitivity."
# frameworks notes label this Level 2; in the saved PDF the text sits just before the 'Level 2' heading, so check the level
indicators: [uptake.builds_on_previous_turn (demszky2021), negotiact.active_listening, negotiact.persuasive_argument]
not_claimed: [written communication, choice of channel, 'confidence' as an inner state]
evidence_summary:
  reliability: "none for this framework; AC assessors agree .80 per rating but only .26 across exercises (jackson2016)"
  validity: "AC communication dimension adds dR2 .003 (n.s.) beyond GMA + Big Five; influencing others .013 (meriac2008)"
  unit_of_validation: individual (AC dimensions)
min_situations: ">= 4 for ~.70 (my Spearman-Brown projection from jackson2016)"
job_profile_link: "recruiters choose relevant behaviours per role and grade (ukcs_success_profiles)"
licence: Open Government Licence v3.0 (attribution)
review: {status: pending}
```

### (c) Instrument / evidence card

Schema:

```yaml
id:
name:
native_mode:
unit:
owner_licence:
reliability: []    # typed
validity: []       # typed: structural / convergent-discriminant / nomological / criterion / incremental
generalizability: []
known_problems: []
adoption: []
scores: {A:, B:, C:, confidence:}   # link to scores.json
verified:
sources: []
```

Filled example, built from collected sources:

```yaml
id: assessment_centre
native_mode: trained-assessor rating of simulations; may be rated from audio/video recordings (rupp2015 p.1251)
unit: individual
owner_licence: method free; Guidelines open; exercise content owned by builder
reliability:
  - {type: inter-assessor G, value: ".80 single rating; .97 overall", source: jackson2016}
generalizability:
  - {type: across exercises, value: ".26 single; .55 dimension/3 ex.; .65 overall/3 ex.", source: jackson2016}
validity:
  - {type: criterion, value: "operational .29", source: sackett2022}
  - {type: incremental, value: "dR2 .10 over GMA + Big Five (20% -> 30%)", source: meriac2008}
  - {type: structural, value: "person x dimension .13-1.11% of variance; exercise 24.7-53.7%", source: jackson2016}
known_problems: [exercise effects, role-player disposition (oliver2014), communication dimension n.s. incrementally]
adoption: ["UK 17.3% (zibarras2010)", "Spanish SMEs 8% (alonso2015)", "Germany 35% (diekmann_konig)"]
standard: "fully automated scoring is not an AC (rupp2015 Section X)"
scores: {A: 69, B: 74, C: 93, confidence: High}
```

### Evidence layer contents

- The source records.
- Passage-level excerpts with page or table locations, matching the `numbers` fields.
- File hashes (sha256 already recorded).
- Verification status.
- Licence per file.
- The disagreements list (§11).
- The full texts themselves are stored privately.

### Must be reviewed by a qualified psychologist before use

- Every indicator definition, Spanish translation and delimitation.
- The mapping and weights from indicators to competencies.
- Level anchors and cut-offs.
- The list of "not claimed" statements and report wording to candidates and employees.
- The exclusion list: MBTI, Belbin, DISC, speaking time as leadership, emotion labels.
- The validation protocol, rater training and fairness analysis.
- The job-profile weighting procedure.

---

## 9. Consequences for the existing product rubric

The repository's rubric (`knowledge/rubrics/`, theory in [`THEORY.md`](THEORY.md)) has 11 competencies, some of whose details were written from memory (marked † in THEORY.md). This synthesis was written without reviewing it line by line. Given the evidence, any such rubric must satisfy the following.

1. **Every competency must be defined by indicators observable in a speaker-labelled transcript.** Drop or relabel anything that needs inner states, self-knowledge or emotion. Emotion from voice is prohibited at work (§7).
2. **Every definition and indicator must be re-sourced** to a card with a licence status. Text that came from memory and cannot be sourced must be flagged. Gated vendor content (SHL, Hogan, Korn Ferry/Lominger, DDI, CompeTEA) cannot be copied.
3. **Eleven separate scores from one conversation exceed what the evidence supports.**
   - Dimension-specific variance in ACs is about 1% (`jackson2016`; 0.4–1.4% in Putka & Hoffman 2013, verified, Table 3).
   - Meta-analytic CFA finds three broad factors: administrative, relational, drive (`meriac2014`, abstract).
   - Negotiation judges could not rate more than ten competencies in 60 minutes (`smolinski2020`).

   Report fewer, broader scores, or report all 11 only as aggregated multi-situation profiles with their reliabilities.
4. **Aggregate across situations.** Show reliability per score (G or ICC) and refuse a score below a threshold. Never give a single-meeting verdict.
5. **Control word count and talk time**, and report scores conditional on them. Account for the counterpart and the role.
6. **Make individual claims only where individual-level validity exists.** Present team-level schemes (act4teams) as team feedback.
7. **Personality, if kept, means personality as observers perceive it.** No self-report trait claims.
8. **Use anchored levels (BARS) calibrated with trained raters.** FOR training is required by the AC Guidelines.
9. **Weight per job from documented job analysis**, published before use.
10. **Keep a human decision-maker.** Run bias audits with impact ratios. Keep documentation sufficient for high-risk obligations.

---

## 10. Papers to obtain (merged, ranked)

DOIs are as recorded in the sources or notes. "No DOI" means none was recorded.

| # | Paper | DOI | Why it matters |
|---|---|---|---|
| 1 | Herde & Lievens (2023), Multiple speeded assessments under scrutiny, JAP | 10.1037/apl0000603 | Design template for many short scored conversations; **obtained and read 2026-11-15** (accepted manuscript) |
| 2 | Kauffeld, Lehmann-Willenbrock & Meinecke (2018), act4teams chapter, pp. 422–431 | 10.1017/9781316286302.022 | Full code list, reliability, licence of the main meeting scheme; **no legal open copy found 2026-11-15** (request to the authors drafted) |
| 3 | Kauffeld & Lehmann-Willenbrock (2012), Meetings matter, SGR 43(2), 130–158 | 10.1177/1046496411429599 | Criterion coefficients for meeting talk (abstract only); **no legal open copy found 2026-11-15** (request drafted) |
| 4 | Putka & Hoffman (2013), AC variance sources, JAP | 10.1037/a0030887 | How many raters/exercises; **read 2026-11-15** (copy posted by the second author's department): jackson2016's summary of it confirmed (person × dimension 1.1%, person × dimension × exercise 23.4%, Table 4) |
| 5 | Oh, Wang & Mount (2011), observer-rated FFM validity, JAP | 10.1037/a0021832 | Coefficients for observer-rated personality vs performance; direction confirmed in the PubMed abstract 2026-11-15; an authors' copy is listed on SSRN but could not be fetched automatically |
| 6 | Elfenbein et al. (2008), Are some negotiators better than others?, JRP; and the 2018 JAP follow-up | 10.1016/j.jrp.2008.06.010 (follow-up DOI not recorded) | Individual vs dyadic variance in negotiation; **2008 paper read 2026-11-15**; 2018 follow-up still needed |
| 7 | Breil, Forthmann & Back (2022), Multiple speed assessments of social skills, EJPA | 10.1027/1015-5759/a000657 | Distinct social skills from short interactions; **read 2026-11-15** |
| 8 | Arthur et al. (2003), AC dimensions criterion validity, PPsych | 10.1111/j.1744-6570.2003.tb00146.x | Resolves 6 vs 7 dimensions; per-dimension validity |
| 9 | Funder, Furr & Colvin (2000), RBQ, J Pers | 10.1111/1467-6494.00103 | Observer coding of personality-relevant behaviour; coefficients |
| 10 | Borkenau et al. (2004), Thin slices, JPSP | 10.1037/0022-3514.86.4.599 | How many situations for personality impressions; **read 2026-11-15** |
| 11 | MacLaren et al. (2020), Babble hypothesis, LQ | 10.1016/j.leaqua.2020.101409 | r = .67 and gender effect **verified 2026-11-15** |
| 12 | Kuncel & Sackett (2014), AC construct validity, JAP | 10.1037/a0034147 | Counter-position to jackson2016 |
| 13 | Roch et al. (2012), FOR training meta-analysis, JOOP | 10.1111/j.2044-8325.2011.02045.x | Rater-training effect (d .50 secondary) |
| 14 | Lievens (2001), assessor training, JAP | 10.1037/0021-9010.86.2.255 | Training effect on reliability/discriminant validity |
| 15 | Bowler & Woehr (2006), dimension vs exercise factors, JAP | 10.1037/0021-9010.91.5.1114 | Meta-analytic variance decomposition |
| 16 | Hüffmeier et al. (2014), hard/softline negotiation meta-analysis, J Mgmt | 10.1177/0149206311423788 | Which behaviours pay off; effect sizes not seen |
| 17 | Lievens, Sanchez & De Corte (2004), competency inference, PPsych | 10.1111/j.1744-6570.2004.00009.x | Reliability of job-to-competency linkage |
| 18 | Conway & Huffcutt (1997), multisource ratings, HP | 10.1207/s15327043hup1004_2 | Peer-rating reliability (secondary figures) |
| 19 | Ohland et al. (2012), CATME-B, AMLE | 10.5465/amle.2010.0177 | Peer-anchor validity |
| 20 | Christian, Edwards & Bradley (2010), SJT constructs, PPsych | 10.1111/j.1744-6570.2009.01163.x | Leadership/teamwork SJT coefficients |
| 21 | Bartram (2005), Great Eight, JAP | 10.1037/0021-9010.90.6.1185 | Only criterion evidence for a commercial framework; abstract only |
| 22 | Judge et al. (2002), personality and leadership, JAP | 10.1037/0021-9010.87.4.765 | **Read 2026-11-15**: Big Five multiple R .53 emergence, .39 effectiveness, .48 overall; extraversion ρ .31 (pp. 765, 773) |
| 23 | LePine et al. (2008), teamwork processes meta-analysis, PPsych | 10.1111/j.1744-6570.2008.00114.x | Dimension names and coefficients |
| 24 | Bass (1954), Leaderless group discussion, Psych Bull (+ Bass et al. 1953) | 10.1037/h0056881 | LGD reliability |
| 25 | Fletcher et al. (2003), ANTS, BJA | 10.1093/bja/aeg112 | Individual-level marker system |
| 26 | Langer, König & Hemsing (2020), automatic interviews, JMP | 10.1108/JMP-03-2019-0156 | Applicant reactions |
| 27 | SIOP (2023) AI-based assessment recommendations | no DOI | Professional standard; **read 2026-11-15** (§7) |
| 28 | AI Act official text, OJ L 2024/1689 (EUR-Lex), Arts. 3, 4, 5, 6, 26, Annex III | — | **Read 2026-11-15** (§7) |
| 29 | COP CET-R report on CompeTEA (cop.es) | — | Resolve the predictive-validity disagreement. The file obtained is the blank CET-R questionnaire (Hernández et al., 2016), not the CompeTEA report: still open |
| 30 | Hausknecht, Day & Thomas (2004), applicant reactions meta-analysis | no DOI | Per-procedure favourability |

---

## 11. Disagreements between records

- **Structured interview validity:** .42 (`sackett2022`) vs .51 (Schmidt & Hunter via Sackett) vs .63 (`salgado2008`) vs .70 (`huffcutt2014`, secondary). The planning value is .42.
- **AC validity:** .29 (`sackett2022`) vs .37 with reliability .70 (`salgado2008`).
- **Conscientiousness:** .19/.25 (`sackett2022`) vs .30 (`salgado2008`).
- **AC dimensions:** 6 (`arthur2003` abstract) vs 7 (`meriac2008`, `hoffman2011`).
- **act4teams κ:** .90 / .71 (secondary snippets) vs .81 (`lehmannwillenbrock2015`) vs .63 (`bavendiek2015`).
- **CompeTEA:** COP predictive validity rating 3.5 vs manual extract listing no job-performance criterion.
- **Verification status:**
  - `fiori2014` is "abstract-only" in its record but described as read in full in the area notes. I used the record.
  - `derue2011` is full text in instruments, abstract-only in behaviour.
  - `hickman2022` is full text in computational, secondary in instruments.
- **`rupp2015`:** its `notes` field contains both "content not opened … unverified" and "Second pass: read in full". I treated it as full text.
- **Practice notes** say 14 full-text records; the records have 17.
- **Kuncel & Sackett 2014 vs Jackson 2016:** they disagree on whether AC dimension variance matters.

---

## 12. How the rubric was applied

- **A1, A2, A3, C1-C4:** only the RUBRIC anchors were used. A1 = 20 requires adequate values on every type reported, including across situations; otherwise 14.
- **Vendor meta-analyses** count as meta-analyses in A3. Their lack of independence is scored in A4.
- **A4** (my levels):
  - 20: mostly independent, replicated, and held under critical re-analysis;
  - 15: independent, replicated, no critical test found;
  - 10: mixed vendor/independent, or weakened by critical re-analysis;
  - 5: single independent study or group;
  - 0: vendor only or none.
- **A5:**
  - 10: meta-analytic or > 1,000 people, recent work samples;
  - 7: large but older, or moderate;
  - 4: small or student;
  - 2: vendor statements only;
  - 0: none.
- **A6** is subtracted. Totals are floored at 0 (the raw sum is noted in `scores.json` when it went negative).
- **B1:** 35 wide independent use / 25 sizeable minority / 15 small independent use / 8 vendor or owner claims / 0 not found.
- **B2:** 25 standard or review supports it including validity / 18 favourable with gaps / 10 general guidance only / 0 not found or negative.
- **B3:** 20 both legal and candidate evidence / 12 one of them / 6 weak / 0.
- **B4:** 20 Spanish version with norms / 12 Spanish version limited, or method practised in Spain / 6 open but not Spanish / 0.
- **C1 for automatic methods** = the anchor of the human criterion the method was validated against (untrained crowd counts as 12).
- **C2:** 25 requires replication *and* no contrary transcript evidence; with contrary evidence, 18.
- **C4:** 15 explicit open licence or a free method / 8 readable, or CC BY-NC / 0 gated or not found.
- **C5:** 10 targets personality, negotiation or leadership / 7 core teamwork, or scope within a broader set / 4 names only or adjacent (EI, communication) / 0.

**Known weaknesses of the rubric:**
- C ignores validity. That is intended, but it puts the leaderless group discussion at 100.
- A3 rewards vendor meta-analyses at the same level as independent ones.
- B rewards measured popularity, which is not quality.

---

## 13. Glossary (plain language)

- **Reliability:** consistency of measurement. A ruler shifted by +1 cm is reliable and not valid. Types:
  - *Internal consistency* (Cronbach's α): do the items of one scale agree?
  - *Test-retest:* same person, same test, later. Same score?
  - *Inter-rater / inter-coder* (ICC, Cohen's or Fleiss' κ): do different raters agree? Single-rater values (ICC1) are lower than averages of k raters (ICC2/ICC(k)).
  - *Across situations / exercises:* does the person score the same in a different meeting? This is the one that matters most here.
- **Generalizability (G) theory:** splits score variance into person, rater, situation, dimension and their interactions. It gives a G coefficient for a chosen design, for example "2 raters × 4 meetings".
- **Validity:** does the score mean what we say it means? Each type is separate:
  - *Content:* do the items or indicators cover the construct (expert judgement)?
  - *Structural / factorial:* do the items group as the theory says?
  - *Convergent / discriminant (construct):* does it correlate with measures of the same thing, and not with different things?
  - *Nomological:* does it fit the expected network of relations with other constructs?
  - *Criterion* (concurrent or predictive): does it relate to an outcome such as job performance, measured at the same time or later?
  - *Incremental:* does it add prediction beyond what is already measured (for example GMA + Big Five)?
- **Operational validity:** a criterion correlation corrected for criterion unreliability and range restriction. Overcorrection inflated older estimates (`sackett2022`).
- **Meta-analysis:** a *quantitative* pooling of effect sizes across studies.
- **Narrative / systematic review:** a qualitative synthesis. It is not a meta-analysis. `lance2008`, `campion2011`, `salgado2008`, `robots2022` and `murphy_hall2021` are reviews.
- **Second-order meta-analysis:** a meta-analysis of meta-analyses (`wilmot2019`).
- **Native assessment mode:** how an instrument was built and validated (self-report, other-report, observer coding, test, interview). Changing the mode means the validity has to be shown again.

---

## 14. Limitations of this research

Counts are computed from the source records ([`research/coverage.json`](research/coverage.json)).

| Area | Records | Full text | Abstract-only | Secondary | Files saved |
|---|---|---|---|---|---|
| behaviour | 79 | 25 | 27 | 27 | 31 |
| computational | 50 | 37 | 5 | 8 | 39 |
| frameworks | 37 | 13 | 11 | 13 | 22 |
| instruments | 43 | 10 | 14 | 19 | 21 |
| practice | 24 | 17 | 1 | 6 | 10 |
| spanish | 30 | 13 | 7 | 10 | 21 |
| **total** | 263 | 115 | 65 | 83 | 144 records with a file |

- **Overall:** 263 records, 251 unique ids.
  - 12 records duplicate an id across areas.
  - 5 papers appear under two different ids: `arthur2003`/`arthur2003_ac`, `salgado2008`/`salgado_moscoso2008`, `alonso2015`/`alonso2015_pymes`, `esco_tsc2020`/`esco_transversal2020`, `hogan_tech2016`/`hogan_sales_techreport`.
  - By record: 115 full text (44%), 65 abstract-only, 83 secondary.
  - 157 saved files (kept privately).
  - Of the 152 source ids cited in `scores.json`: 75 full text, 41 abstract-only, 36 secondary.
- **Key meta-analyses are known only from abstracts or other papers:** Arthur 2003, Oh/Wang/Mount 2011, Conway & Huffcutt 1997, Christian 2010, Joseph & Newman 2010, O'Boyle 2011, Hüffmeier 2014, Kauffeld 2012. (Herde & Lievens 2023, Judge 2002, Borkenau 2004, MacLaren 2020, Elfenbein 2008, Breil 2022 and Putka & Hoffman 2013 were read in full on 2026-11-15; for Oh/Wang/Mount 2011 only the PubMed abstract was checked.) Scores built on them carry Low or Medium confidence.
- **No forum evidence:** Reddit, Stack Exchange and Quora were blocked, and CIPD returned 403. The "opinion climate" rests on surveys, many of them by vendors (TestGorilla, Greenhouse, SHRM, Michael Page, InfoJobs).
- **Legal texts were unofficial reproductions** (EUR-Lex bot challenge). On 2026-11-15 the AI Act articles cited in §7 and the NYC LL144 final rule were checked against official copies. The Commission guidelines, the GDPR, the Illinois statute and EEOC guidance were not re-checked.
- **Spanish evidence is thin:**
  - one regional company survey (75 firms, 2013);
  - one psychologists' survey (Work specialty N = 69);
  - no Spanish validation of any observational scheme or AC;
  - no Spanish corpus with behaviour annotations.
- **Populations:** almost all observational and automatic evidence comes from students, labs or clinical settings, in English or German.
- **Some author lists were not captured** (`robots2022`, `savani2023`, `langer2024`, `twitchell2013`). Several figures were read from layout-extracted text and flagged "check" (`naim2015`, `mairesse2007`, `zhang2024`).
- **The scoring is mine.** It was applied uniformly but involves judgement on the unanchored lines (§12). A second scorer should rescore a sample to estimate inter-rater agreement on this rubric itself.
