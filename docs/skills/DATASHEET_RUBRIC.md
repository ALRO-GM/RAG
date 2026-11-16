# Datasheet: the competency rubric (`core`)

Date: 2026-11-16. Issue #86 (part 4). Format after Gebru et al. (2021),
"Datasheets for Datasets", adapted to a hand-authored catalogue. This is
the rubric's datasheet asked for in `FAIRNESS.md` §3.5 row I4 and §4
action 4. The rubric is text written by the project, not data collected
from people. **No per-group result exists for it**: it has not been
applied to real people (`FAIRNESS.md` §1), see also
`src/application/skills/fairness_report.py` and
`experiments/skills/fairness_audit.py`.

## Contents at a glance

Counted by reading the two JSON files (script in the Sources list):

| Item | `core.en.json` | `core.es.json` |
|---|---|---|
| Name, version, language | `core`, 0.1.0, `en` | `core`, 0.1.0, `es` |
| Competencies | 11 | 11 |
| Indicators | 110 (66 positive, 44 negative); 10 per competency (6 + 4) | 110 (66 positive, 44 negative) |
| Examples attached to indicators | 73 | 75 |
| Scale | 1 to 5; anchors written at 1, 3 and 5; 2 and 4 are "between" | same |
| Excluded competencies, with reasons | 12 | 12 |
| Sources cited per competency | 5 to 10 | 5 to 10 |

The eleven (same ids in both files): COMM, LISTEN, INFLU, COLLAB, INCL,
CONFL, FEEDB, COORD, SOLVE, INIT, ADAPT. Each competency also has a
`definition`, `limits` (what a transcript cannot show) and `practice`
(hand-written "helps / hurts" examples for the person report).

## 1. Motivation

Created to give the analyzer a fixed, inspectable list of observable
behaviours to match transcript sentences against and to rate on
(`DESIGN.md` §1, §3). Scope chosen from what can be observed in a
speaker-labelled transcript of spoken group interaction (`THEORY.md`
§3.1). Authored inside the project; no external requester is recorded.

## 2. Composition

Competency, indicator (id, text, polarity, examples), anchors, sources,
limits, practice (`src/domain/skills/rubric.py`, `Competency`,
`Indicator`). The schema checks unique ids, anchors on the scale and at
least one positive indicator per competency. Clusters in `THEORY.md`
§3.1: communication and influence; relational; task and coordination;
drive and self-management.

Deliberately excluded (12, each with a reason in the file), for example
integrity and ethics ("a transcript only shows claims"), technical
expertise, and reliability or work quality ("happen outside the
conversation"). Leadership is a derived profile, not rated separately
(`THEORY.md` §3.2).

## 3. Collection and authoring process

- Indicators are the project's own paraphrased synthesis, phrased as
  verbal behaviour that can be quoted; **candidates for expert review,
  not validated items** (`THEORY.md` §3.1). No rubric text is
  reproduced from copyrighted instruments (`THEORY.md`, conventions).
- Sources cited per competency in the JSON: frameworks such as the SHL
  Great Eight, UK Success Profiles, OECD, UN Values and Behaviours,
  SHRM BASK, LifeComp, PISA 2015 CPS, ATC21S, CATME, Marks et al. 2001,
  Salas et al. 2005, Stevens and Campion 1994, assessment-centre
  dimensions, act4teams and Woolley et al. 2010. Cross-walks and
  appraisal: `THEORY.md` §2-§3, `RESEARCH.md`.
- Some details are stated from memory and not re-checked (marked with a
  dagger in `THEORY.md`); `RESEARCH.md` §9 says some details of the rubric were written from
  memory and that its synthesis did not review the rubric line by line. Which indicators are affected is not tracked: unknown.
- Spanish was written natively with the same ids
  (`docs/TASKS.md` T-32); a test enforces the same ids in both files
  (`DESIGN.md` §7). Written by one author; **not reviewed by assessors**
  (`docs/TASKS.md` T-32). Native Spanish speakers of varied origin did
  not review it: not recorded.

## 4. Preprocessing and use in the pipeline

Indicator text plus examples form the query matched against sentences
(`Indicator.query`, BM25 and MiniLM embeddings, `DESIGN.md` §3). A
rating needs at least `min_incidents` verified turns from `min_episodes`
exchanges (default 3 incidents, `src/config.py`), otherwise "insufficient
evidence", which is never converted to a number (`THEORY.md` §4.2-§4.3).

## 5. Uses

Intended: a draft, for a trained human assessor, of observable
behaviours with quotes. Not for: a sole decision about a person,
inferring traits or emotions, or claims beyond the evidence in
`RESEARCH.md` §9. Known limits stated there: eleven separate scores
from one conversation exceed what the evidence supports (dimension
variance in assessment centres is small); scores should be aggregated
across situations, with reliability shown.

Fairness-relevant gaps (all open): Peninsular Spanish overlap, directness
and address forms may be over-flagged by indicators tuned on English
data (`THEORY.md` §6.3); anchors have not been calibrated with raters
(`THEORY.md` §4.2); no item-level bias check, no differential-item
analysis, no validity study: **not measured**.

## 6. Distribution

In the repository: `knowledge/rubrics/core.en.json`,
`knowledge/rubrics/core.es.json`; served by `json_catalogue_repository`
(`skills catalogue --language es|en`). Licence of the catalogue text:
not stated in the repository. Source frameworks keep their own licences
(vendor content such as SHL cannot be copied, `RESEARCH.md` §9).

## 7. Maintenance

Version `0.1.0` in both files; last commit touching the folder is #58
(`git log`). No maintainer or review calendar is written down in the
repository. Re-audit when
the rubric changes (`FAIRNESS.md` §3.7 G5). Next steps in `THEORY.md`
§5: a human-annotated gold set, annotator agreement, validity evidence,
fairness checks.

## Sources

- `knowledge/rubrics/core.en.json`, `knowledge/rubrics/core.es.json`
- `src/domain/skills/rubric.py`
- `docs/skills/DESIGN.md` §1, §3, §7; `src/config.py`; `THEORY.md` §2-§6;
  `RESEARCH.md` §9; `FAIRNESS.md`; `docs/TASKS.md` (T-32)
- Counting script: a short `python3` reading `scale`, `competencies`,
  `indicators[].polarity`, `indicators[].examples` and `excluded` of
  each JSON file (output quoted above).
