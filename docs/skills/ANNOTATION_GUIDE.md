# Building the reference data

The analyzer is only as good as the data it is tuned and judged
against. This guide is the protocol for producing that data; the
reasoning and the sources are in [`THEORY.md`](THEORY.md) §5.

The files under `examples/skills/` are **synthetic** (written to
exercise the tools). They are not reference data.

## 1. What is annotated

Two layers per session, mirroring the pipeline:

| Layer | Unit | What the annotator records |
|---|---|---|
| Incidents | A turn of a participant | The indicator it shows (`LISTEN.P1`, `CONFL.N2`, ...). A turn may carry several indicators; most turns carry none |
| Ratings | Participant × competency | 1–5 with the anchors of the rubric, or *insufficient evidence* (`null`) |

Rules the annotators follow, the same ones the system enforces:

1. **Observe, record, classify, then evaluate.** Mark the incidents of
   the whole session first; rate afterwards, one competency at a time,
   from the marked incidents only.
2. **A rating needs three incidents from two separate exchanges.**
   Otherwise the answer is *insufficient evidence*. It is a valid,
   frequent answer, never a low rating.
3. **Only the participant's own words count**, and only what was said:
   no intentions, personality, tone, accent, grammar or fluency.
4. **How much someone talks is not evidence.** Extra instances beyond
   the minimum do not raise a rating; variety and quality do.
5. **No opportunity, no rating.** A session without disagreement gives
   no evidence about handling disagreement, for anybody.
6. Read the `limits` of each competency in the rubric
   (`skills catalogue`) before annotating it.

## 2. People and training

- **Three annotators** on the reliability subset (all three annotate
  the same sessions, independently); two plus adjudication on the rest.
  At least one with assessor or I/O psychology experience, who also
  produces the expert target scores. For Spanish sessions, annotators
  who speak the varieties being assessed.
- **Frame-of-reference training** before any production annotation:
  1. Study the definitions, indicators and anchors.
  2. Annotate practice transcripts.
  3. Compare with the expert target scores and discuss every
     difference.
  4. Certification round: an annotator starts production only after
     reaching the thresholds of §4.
  5. Drift checks: already-calibrated transcripts are re-inserted
     periodically.

## 3. Workflow

```bash
# 1. One empty form per annotator and session
uv run python -m src skills annotation_template \
    --transcript data/transcripts/s01.json --annotator A1

# 2. The annotator fills rating and evidence (turn + indicator id),
#    working from the transcript with turn numbers.

# 3. Check the form: unknown turns, turns of another speaker,
#    indicators of another competency, ratings without evidence
uv run python -m src skills validate_annotations \
    --annotations A1/s01.json --transcript data/transcripts/s01.json

# 4. Agreement between annotators, overall and per competency
uv run python -m src skills agreement \
    --annotations A1/s01.json,A2/s01.json,A3/s01.json

# 5. Only if agreement is sufficient: the consensus reference
uv run python -m src skills consensus \
    --annotations A1/s01.json,A2/s01.json,A3/s01.json

# 6. Score the system and tune retrieval against the consensus
uv run python -m src skills evaluate --assessments ... --reference ...
uv run python -m src skills lab_dataset --reference ... --output D.json
```

Annotators never see the system's output for a session before
annotating it.

## 4. Agreement thresholds

| What | Statistic | Usable | Tentative | Action below |
|---|---|---|---|---|
| Ratings, all annotators | Krippendorff's ordinal alpha | ≥ 0.80 | 0.667–0.80 | Revise anchors, retrain |
| Ratings, one annotator | ICC(2,1) | ≥ 0.60 | | Retrain |
| Ratings, consensus of k | ICC(2,k) | ≥ 0.75 | | Add annotators or retrain |
| Rated or insufficient | Cohen's kappa | ≥ 0.667 | | Clarify the evidence rule |
| Evidence turns | Pairwise F1 | report | | Review per indicator |

Per-competency statistics are printed with the number of co-rated
units and are not interpreted below ten: with a handful of units alpha
swings from 0 to 1 on a single disagreement. Indicators that annotators
cannot agree on are rewritten or merged; that is the main way the
rubric improves.

The consensus rating is the median of the annotators who rated (the
lower middle value on a tie) when more than half of them rated;
evidence is a turn marked by more than half.

## 5. How much data

| Purpose | Minimum |
|---|---|
| Reliability of session ratings | ≥ 30 sessions, ≥ 120 participant-sessions, 3 annotators |
| Per-indicator incident coding | ≥ 50 adjudicated positive instances per indicator; oversample sessions likely to contain rare behaviours (disagreement, feedback) |
| Tuning versus testing | Split by session **and by team**: no participant on both sides |

Vary task type (planning, problem solving, retrospective), group size
(3–6) and speakers (gender, native and non-native, regional variety).
Keep both the human-corrected and the raw automatic transcript of every
session: the difference between them is a measured source of error.

## 6. What to check in the system once the data exists

| Check | How |
|---|---|
| Agreement with assessors | `skills evaluate`, read against the annotators' own agreement, which is the ceiling |
| Quote faithfulness | Share of quotes verified must be 100% after grounding; audit by hand whether quotes show the indicator |
| Stability | Same session, repeated runs and model versions |
| Robustness | Human transcript versus automatic transcript; shuffled speaker names |
| Halo | Correlation between competencies, system versus humans |
| Talk time | Correlation of each rating with words spoken, system versus humans |
| Fairness | Rating and insufficient-evidence rates by subgroup, system next to humans; name-swap tests |
| Convergent validity | Peer ratings (CATME) collected for the same teams |
