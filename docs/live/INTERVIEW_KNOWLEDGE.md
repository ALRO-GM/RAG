# Interview knowledge: forbidden questions and core questions (KN-4, #234)

**This is not legal advice.** It is a working note built from the
official texts read on 2026-11-24. Law changes, and how it applies
depends on the case: a labour-law professional must review it before it
is used with real candidates. What could not be checked is marked
*unverified* below and in the files.

Two data files feed the live detector (#232, not wired here) and the AI
interviewer:

| File | Schema | What |
|---|---|---|
| `knowledge/interview/forbidden-questions.json` | `interview.forbidden.v1` | questions not to ask, per category, with legal basis |
| `knowledge/interview/core-questions.json` | `interview.core.v1` | the few questions asked once per candidate |

Both are plain JSON, validated by `src/domain/knowledge/interview_rules.py`
(pydantic models; `parse_forbidden`, `parse_core`, `fold`,
`ForbiddenRules.hits`) and by `tests/unit/test_interview_rules.py`.
The loader does no I/O: read the file, `json.load`, parse.

## 1. `interview.forbidden.v1`

```
schema, version, language_of_patterns, checked_on, disclaimer{es,en}, pattern_rules
sources{ <id>: {title, id, url, kind, text_read, how_read} }      # the research log
categories[ {
  id, category, title{es,en},
  patterns{es:[regex], en:[regex]},      # applied to fold(question)
  examples{es:[...], en:[...]},          # questions to avoid; each trips its patterns
  why{es,en},
  legal_basis[ {source, ref, url, quote, verified, note?} ],
  lawful_alternative{es,en},             # what to ask instead
  exceptions[ {es, en, basis[...]} ],    # legitimate cases
  verification: "verified" | "partly_verified",
  status_note?{es,en}
} ],
ai_limits[ {id, limit, status, why, basis[...], product_rule} ]
```

- **Matching.** `fold(text)` lower-cases, removes accents and straightens
  apostrophes; each pattern is a Python regular expression searched in
  that text, in either language. A hit is a hint for a person, never a
  verdict. Tests check that every example trips its own category and that
  no recommended alternative or core question trips any category.
- **Basis.** `source` is a key of `sources`; `ref` the article as the law
  numbers it; `url` the official page (BOE consolidated text with the
  article anchor, EUR-Lex, or the AEPD guide); `quote` the exact words
  read there (checked against the fetched text when the file was
  generated); `verified` is true only when the wording was read in the
  official text; otherwise `note` says what was not checked.
- **`verification`** of a category is `verified` only when every basis
  and exception basis is verified and unannotated; any note, unverified
  basis or exception without basis makes it `partly_verified`.

### Categories (15) and their main legal basis

| Category | Main basis (verified in the official text unless marked) |
|---|---|
| `age` | ET art. 4.2.c, 17.1; Ley 15/2022 art. 2.1, 9.1; CE art. 14; CP art. 314; Directiva 2000/78/CE art. 4.1. Exceptions: minimum age (ET art. 6.1, 6.2); essential occupational requirement |
| `family_status` (civil status, partner, children, plans) | ET art. 4.2.c, 17.1; LO 3/2007 art. 5; CE art. 18.1; CP art. 314 |
| `pregnancy` | LO 3/2007 art. 8, 5; ET art. 4.2.c; RGPD art. 9.1. PRL art. 26 is after hiring |
| `health` | **Ley 15/2022 art. 9.5** («El empleador no podrá preguntar sobre las condiciones de salud del aspirante al puesto»), art. 2.1, 2.3; RGPD art. 9.1; LOPDGDD art. 9.1; LISOS art. 16.1.c; PRL art. 22.1 (after hiring) |
| `disability` | ET art. 4.2.c, 17.1; Ley 15/2022 art. 2.1, 9.5; LGD art. 63; exceptions: reasonable adjustments (Ley 15/2022 art. 4.1; LGD art. 40.2), PRL art. 25 |
| `religion_ideology` | CE art. 16.2, 14; ET art. 4.2.c, 17.1; RGPD art. 9.1; LOPDGDD art. 9.1. Tendency-organisation exception: **unverified** |
| `union` | CE art. 28.1; ET art. 4.2.c, 17.1; RGPD art. 9.1; LOPDGDD art. 9.1 |
| `origin_nationality_language` | Ley 15/2022 art. 2.1; ET art. 4.2.c, 17.1; RGPD art. 9.1. Directiva 2000/43/CE art. 3.1.a **unverified**. Exceptions: work authorisation (LO 4/2000 art. 36.1), language the job requires (Ley 15/2022 art. 4.2) |
| `sexual_orientation_gender` | ET art. 4.2.c, 17.1 (as amended by Ley 4/2023); Ley 15/2022 art. 2.1; RGPD art. 9.1 |
| `criminal_record` | RGPD art. 10; AEPD guide pp. 20-23; exception: contact with minors, certificate of the Registro Central de Delincuentes Sexuales, **LO 8/2021 art. 57.1** (the former LO 1/1996 art. 13.5 no longer exists; the AEPD guide of December 2025 still cites it). Sector rules cited by the AEPD (aviation, anti-money-laundering): **unverified** |
| `physical_appearance` | open clause of Ley 15/2022 art. 2.1 and CE art. 14; whether each criterion is protected: **unverified** |
| `pay_history` | Directiva (UE) 2023/970 art. 5.2 (verified); transposition deadline 7 June 2026 (art. 34.1); **Spain's transposition not checked** |
| `excess_personal_data` | RGPD art. 5.1.c; AEPD guide; LISOS art. 16.1.c |
| `indirect_proxies` | Ley 15/2022 art. 4.1, 9.2; ET art. 17.1; LISOS art. 16.1.c |
| `past_claims` | ET art. 4.2.h, 17.1 (text speaks of workers; applicability to applicants **unverified**) |

LISOS art. 16.1.c classifies as a very serious infraction to «solicitar
datos prohibidos por la normativa legal o reglamentaria o de carácter
personal en cualquier proceso de intermediación o colocación» and to
adopt discriminatory decisions in access to employment; the AEPD guide
applies it to selection processes (its wording speaks of intermediation
and placement: whether it reaches an employer's own interview was not
checked beyond that guide).

### AI limits (7), linked to #151 (owner decisions)

`ai_limits` records what the EU AI Act (Reglamento (UE) 2024/1689), the
GDPR and Law 15/2022 say about AI in recruitment, with the rule it
imposes on the product: no emotion inference at work (art. 5.1.f; that a
selection interview counts as «lugar de trabajo» is **unverified**),
no biometric categorisation (art. 5.1.g), recruitment systems are
high-risk (art. 6.2, annex III 4.a; obligations not reviewed in detail;
whether art. 113's date of 2 August 2026 was postponed by a later
amendment is **unverified**), tell people it is an AI (art. 50.1), no
solely automated significant decision (RGPD art. 22.1), employer informs
workers (art. 26.7), and Ley 15/2022 art. 3.1.o applies equal treatment
to AI.

## 2. `interview.core.v1`

```
schema, version, max_questions (<= 6), disclaimer{es,en}, rules{es,en}
questions[ {id, order, phase, question{es,en}, profile_fields[{path, type, unit?, optional?, max_chars?}],
            why, follow_up, legal_note, related_forbidden[ids], legal_basis[source ids]} ]
```

Five core questions, asked once per candidate with the same words,
whatever the role. They are few on purpose (the ticket said not to
saturate the interview; a sixth slot is free). Each fills profile fields
(`path` is a dotted name for the candidate profile of the next ticket).

| # | id | Question (ES) | Fills | Why this one |
|---|---|---|---|---|
| 1 | `work_mode_location` | «El puesto es [modalidad] en [ciudad]. ¿Te encaja esa modalidad y esa ubicación?» | `availability.work_mode_ok`, `availability.location_note` | knock-out requirement worth knowing first; described, not asked «dónde vives» |
| 2 | `work_authorization` | «¿Tienes autorización para trabajar en España?» | `eligibility.work_authorization` | LO 4/2000 art. 36.1; asked to everyone, never nationality |
| 3 | `start_date_notice` | «¿Cuándo podrías incorporarte, contando con el preaviso que tengas que dar en tu trabajo actual?» | `availability.earliest_start`, `availability.notice_days` | availability and notice in one answer (two of the ticket's examples merged) |
| 4 | `motivation` | «¿Qué te ha llevado a presentarte a este puesto?» | `motivation.summary` | the only open core question; guides behavioural follow-ups |
| 5 | `salary_expectation` | «¿Qué expectativa salarial tienes para este puesto?» | `salary.expectation_min`, `salary.expectation_max` | expectation, never pay history (Directiva 2023/970 art. 5.2); asked last |

If the answer is already in the profile or the offer, it is not asked
again; a «no» is recorded without asking why.

## 3. Review of the interview-method corpus

`knowledge/domains/interview.generic/metodo-entrevista.jsonl`
(`docs/live/VOICEBOT_INTERVIEWER.md` §2) went from 170 to 177 records.
Found and fixed:

| Problem | Records | Fix |
|---|---|---|
| Several questions in one quoted question (not one thing per turn, bad for voice) | 46 records in all: 17 compound STAR/follow-up questions (e.g. m020, m055, m094, m118) and the quoted runs of 2-3 questions (e.g. m016, m042, m101, m124) | first question kept; the rest worded as «si falta, en otro turno». A test now forbids more than one `?` per quoted question outside the legal guard |
| Closed (yes/no) behavioural questions | m083, m102, m142, m016 | rewritten as open («¿Qué hiciste tú?», «Cuéntame una vez…») |
| Invites a health disclosure | m160: «con o sin algún ajuste» | removed; asks «¿Puedes realizar estas tareas?» to everyone; cites Ley 15/2022 art. 9.5 |
| Outdated citation | m163: LO 1/1996 art. 13.5 | replaced by LO 8/2021 art. 57 (13.5 was removed) and RGPD art. 10 |
| Inconsistent with the product | m140: «tras dos seguimientos» | the AI asks one (`interview.max_probes`); two for a person |
| Duplicate | m161 repeated m005 (ajustes) | m161 now the legal point (denying adjustments is discrimination) |
| Vague / uncited law | m154-m158, m162, m168, m169 | article numbers added and checked (CE 16.2, LOPDGDD 9.1, LO 4/2000 36.1, AI Act 50.1); the Directiva 2000/43 mention marked unchecked |
| Gaps | pay history, past claims, emotion inference, the core questions, driving/delivery, vulnerable persons, team leads | m171-m177. The three role families have no `puesto_*` type (a code constant in `ROLE_TYPES`), so they are tagged `puesto_todos` until a follow-up adds the types |

Not changed: the records on structure, closing and per-role requirements
read correctly; m057, m087 and m134 overlap (concrete example requests) but each is
reached by a different situation, so they were kept.

## 4. How the sources were read (the research log)

`sources` of the forbidden file is the log: for each, title, official
id, URL, whether the text was read and how.

- **BOE:** the open-data API `legislacion-consolidada/id/<BOE-A-...>/texto`
  (current consolidated version, 2026-11-24); every BOE quote was checked
  as a substring of the current version of its article.
- **EU:** `eur-lex.europa.eu` answers automated requests with a bot
  challenge, so the Official Journal text was read through the
  Publications Office resource `publications.europa.eu/resource/celex/<CELEX>`
  (same text; the URLs recorded are the EUR-Lex ones). Directive 2000/43/CE
  could not be retrieved that way: unverified.
- **AEPD:** «La protección de datos en las relaciones laborales» (guide
  updated December 2025), PDF read as text; no verbatim quote kept (the
  layout is in columns).
- Not read: case law (the AEPD guide cites STS and STSJ rulings),
  collective agreements, regional rules, the AI Act recitals and the
  Commission's guidelines.

## 5. Open doubts

1. Whether LISOS art. 16.1.c, written for intermediation and placement
   agencies, applies to an employer interviewing directly (the AEPD guide
   applies it to selection).
2. Whether a selection interview is a «lugar de trabajo» for AI Act art.
   5.1.f, and whether the high-risk date was postponed.
3. Whether and how Spain transposed Directive 2023/970 (salary history).
4. Tendency organisations, sector criminal-record rules, protected status
   of appearance: need a lawyer.
5. Pattern lists are word lists in Spanish and English; they will miss
   paraphrases and flag some innocent sentences (tests only cover the
   examples and the corpus). The detector (#232) should treat them as hints.
