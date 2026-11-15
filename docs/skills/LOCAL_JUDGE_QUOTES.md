# The judges' quotes in the assessment, and the similarity floor

Ticket #104. In step g of `skills analyze` (assessment, `DESIGN.md`
§1) the judge quotes the participant, files each quote under an
indicator and rates; grounding keeps the quotes found verbatim in a
turn of the participant (`domain/skills/grounding.py`), and a second
opinion drops a quote when the cosine between its turn and the
indicator (MiniLM, `paraphrase-multilingual-MiniLM-L12-v2`) is below a
floor. On two real recordings the local judge Qwen3-8B (4 bits)
proposed 2 quotes in 22 calls and 20 in 44, and the floor (then
`min_similarity` 0.3) dropped 2 of 2 and 15 of 20; Claude had proposed
113 in 44 on the simulated discussion. This document measures why,
what changes it, and where the floor should be.

**Everything measured here is synthetic** (§1). The two real
recordings were read only on the owner's machine to diagnose; nothing
of them is in this document, the tests or the results files.

## 1. Material

| Session | What | Turns | People |
|---|---|---|---|
| `sim-reunion-verano` | the simulated discussion of #88 (`SIMULATION.md`), a four-person decision meeting | 117 | 4 |
| `sim-entrevista-corta` | a short interview written for this ticket, `examples/skills/simulated/entrevista_corta.txt` (2.8 min at the simulator's 2.4 words per second, 367 words) | 20 | 2 |

Judges: Claude Code (`opus`, subscription; its 44 replies on the
discussion come from the reply cache of the #88 demo, the 22 on the
interview were asked for this ticket: 22 calls in all), Qwen3-8B and
Qwen3-4B at 4 bits on the shared RTX 3070 Ti. Same settings as
`skills analyze` (8 candidate turns, 3 incidents over 2 exchanges).

**Hand labels.** Every grounded quote of every run was labelled
`good` or `bad` by the agent that implemented the ticket (Claude, not
a human annotator, and not blind to which judge wrote the quote):
*does this turn show the behaviour the indicator describes?* In the
interview, a candidate describing what they did (behavioural interview
convention) counts. 740 labels (175 good, 565 bad), in
`experiments/skills/results/assessment_quotes_labels.json`. They are
one reader's judgement, enough to compare judges and thresholds on the
same quotes, not a reference set.

Reproduce (no model, from the stored replies):

```sh
S=examples/skills/simulated
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/skills/assessment_quotes_eval.py \
    --replay --sessions $S/sim-reunion-verano.transcript.v1.json,$S/sim-entrevista-corta.transcript.v1.json \
    --judges claude-code:opus,Qwen/Qwen3-8B,Qwen/Qwen3-4B --styles standard,guided,precise \
    --replies experiments/skills/results/assessment_quotes_replies.json \
    --labels experiments/skills/results/assessment_quotes_labels.json \
    --output experiments/skills/results/assessment_quotes.json
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/skills/assessment_quotes_eval.py \
    --output experiments/skills/results/assessment_quotes.json --table 0,0.3
```

Without `--replay` the local models run again (and Claude, only with
`--claude_calls N`).

## 2. Diagnosis

What the replies show, on the real recordings (read locally, not
recorded) and on the synthetic sessions:

- **Not a format problem.** The replies parse (0 unparseable in the
  14 synthetic runs) and none is cut off: all 462 stored replies end
  with a closed JSON object, the longest 2,034 characters. Thinking is
  off (`enable_thinking=False`) and decoding is greedy. Nothing is
  lost between the model and the parser.
- **Not a language problem.** The instructions are in English and the
  transcripts in Spanish; the 8B answers in Spanish and copies Spanish
  words.
- **The low counts are not reproduced on synthetic sessions.** With
  the same prompt as Claude the 8B proposed 136 quotes in 44 calls on
  the simulated discussion (Claude: 113) and 60 in 22 calls on the
  synthetic interview (Claude: 20). On the real recordings it mostly
  answered "no evidence"; a 2.5-minute interview and a four-person debate
  may simply show few of the rubric's work behaviours, and Claude was
  never run on them (they are real people), so there is no reference
  saying how many quotes they hold.
- **The real weakness is precision.** Of the 8B's grounded quotes on
  the synthetic sessions only 60 of 174 (34%) show the indicator they
  are filed under (Claude: 122 of 131, 93%). It cites whole turns and
  files them under any indicator of the competency (a commitment under
  "gives feedback", an "I agree" under "proposes a middle option").
  The 4B: 50 of 215 (23%).
- **Attribution slips are caught.** In the two-person interview a
  third or more of the local models' proposals were not grounded (8B:
  19 of 60; 4B: 30 of 76). The 8B's, checked one by one, all quote the
  other speaker's `(context)` lines while assessing the interviewer.
  Grounding rejects them; Claude made none.
- **The similarity floor punished the good judge.** At 0.3 it dropped
  51 of Claude's 111 grounded quotes on the discussion (2 of them bad)
  and cut its ratings from 21 to 7. The cosine does separate good from
  bad quotes somewhat (AUC 0.63 to 0.84 for the local runs), but the
  floor was one number for two very different judges.

## 3. Results

Counts of the 14 runs (`--table 0,0.3`). *Good / labelled*: hand labels
over every grounded quote. *Kept* and *Rated* with the floor at 0.3 and
at 0 (the good/bad split of the kept quotes at every floor is in
`assessment_quotes.json`, `by_threshold`; it can differ by one from
*Kept* because the stored cosine is rounded).

| Session | Judge | Prompt | Proposed | Grounded | Good / labelled | Kept ≥0.3 | Rated ≥0.3 | Kept ≥0 | Rated ≥0 | Cosine AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| discussion | Claude (opus) | standard | 113 | 111 | 104 / 111 | 53 | 7 | 111 | 21 | 0.733 |
| interview | Claude (opus) | standard | 20 | 20 | 18 / 20 | 11 | 1 | 20 | 4 | 0.528 |
| discussion | Qwen3-8B 4-bit | standard | 136 | 133 | 45 / 133 | 58 | 5 | 132 | 31 | 0.644 |
| interview | Qwen3-8B 4-bit | standard | 60 | 41 | 15 / 41 | 23 | 2 | 41 | 9 | 0.841 |
| discussion | Qwen3-8B 4-bit | guided | 204 | 204 | 50 / 204 | 78 | 14 | 202 | 44 | 0.677 |
| interview | Qwen3-8B 4-bit | guided | 79 | 51 | 19 / 51 | 35 | 7 | 51 | 11 | 0.749 |
| discussion | Qwen3-8B 4-bit | precise | 214 | 206 | 58 / 206 | 92 | 17 | 204 | 40 | 0.666 |
| interview | Qwen3-8B 4-bit | precise | 81 | 54 | 16 / 54 | 31 | 7 | 54 | 11 | 0.741 |
| discussion | Qwen3-4B 4-bit | standard | 182 | 169 | 42 / 169 | 64 | 7 | 167 | 36 | 0.63 |
| interview | Qwen3-4B 4-bit | standard | 76 | 46 | 8 / 46 | 32 | 4 | 46 | 9 | 0.753 |
| discussion | Qwen3-4B 4-bit | guided | 220 | 201 | 50 / 201 | 76 | 14 | 199 | 40 | 0.694 |
| interview | Qwen3-4B 4-bit | guided | 111 | 59 | 12 / 59 | 33 | 5 | 59 | 9 | 0.74 |
| discussion | Qwen3-4B 4-bit | precise | 187 | 176 | 43 / 176 | 62 | 9 | 174 | 43 | 0.738 |
| interview | Qwen3-4B 4-bit | precise | 86 | 49 | 10 / 49 | 31 | 4 | 49 | 7 | 0.797 |

The prompts (`domain/skills/prompting.py`): *standard* is Claude's.
*guided* walks the model through the participant's turns one by one,
asks for up to 6 quotes of 5 to 25 words copied character by character
and shows one invented example reply in the transcript's language.
*precise* does the same but asks, before each item, what the words do
("ask, propose, inform, agree...") and to cite a turn only if that is
exactly what an indicator describes, with that reason (`why`) written
before the indicator id, and says that `(context)` lines are never the
participant's. The reply format is unchanged (the parser ignores
`why`).

Both variants raise the number of good quotes (pooled over both local
models and both sessions: standard 110, guided 131, precise 127) but
the bad ones more, so the share of good ones does not improve (28%,
25%, 26%). The 8B with a variant still quotes `(context)` lines in the
interview.

Time of the live runs (8B on the discussion, 44 calls): standard
358 s, guided 462 s, precise 587 s; on the interview (22 calls) 127 s,
161 s, 187 s. 4B: 318 s, 318 s, 423 s and 154 s, 188 s, 203 s.
Claude's 22 calls on the interview took 130 s.

## 4. The similarity floor

Share of good quotes kept (recall) and share of good quotes among the
kept ones (precision) at each floor, pooled over the runs (computed
from `assessment_quotes.json` and the labels):

| Floor | Claude: precision · recall | Local, standard prompt: precision · recall | Local, all prompts: precision · recall |
|---|---|---|---|
| none | 0.93 · 1.00 | 0.28 · 1.00 | 0.26 · 1.00 |
| 0 | 0.93 · 1.00 | 0.28 · 1.00 | 0.27 · 1.00 |
| 0.1 | 0.94 · 0.95 | 0.28 · 0.92 | 0.27 · 0.93 |
| 0.2 | 0.96 · 0.77 | 0.32 · 0.84 | 0.30 · 0.86 |
| 0.3 | 0.95 · 0.50 | 0.38 · 0.61 | 0.39 · 0.65 |
| 0.35 | 0.96 · 0.39 | 0.44 · 0.49 | 0.45 · 0.54 |
| 0.4 | 1.00 · 0.25 | 0.54 · 0.36 | 0.53 · 0.41 |
| 0.45 | 1.00 · 0.20 | 0.61 · 0.23 | 0.62 · 0.28 |
| 0.5 | 1.00 · 0.13 | 0.62 · 0.15 | 0.65 · 0.19 |

The numbers do not pick a floor by themselves, so the criterion is
stated: a quote under a rating that does not show the behaviour
misrepresents a person, while a missing quote leaves "insufficient evidence", the
analyzer's safe answer. So precision weighs more than recall: the
floor maximises F0.5 where precision is already low, and F1 where it
is already high.

- **Claude: 0** (`quote_min_similarity`). Its precision is 0.93 with
  no floor and no floor raises it much; F1 is highest with no floor
  (0.964) and falls to 0.656 at 0.3. 0 only drops turns that point
  away from the indicator. On the discussion it keeps 111 quotes
  instead of 53 and rates 21 competencies instead of 7.
- **Local models: 0.4** (`local_quote_min_similarity`). F0.5 rises
  from 0.33 with no floor to 0.49 at 0.4 (standard prompt; all
  prompts: 0.31 to 0.50) and falls after it (0.46 at 0.45). Half the
  kept quotes are good instead of a quarter, at the cost of two thirds
  of the good ones.

**Prompt, at that floor.** At 0.4, pooled over both local models and
both sessions, *precise* keeps 58 good and 50 bad quotes, *standard*
40 and 34, *guided* 54 and 52: precise gives the most good quotes at
the same precision (0.54), so it is the local default
(`local_prompt`). It is about 1.3 to 1.6 times slower than standard.

Before and after, each judge with the old setting (standard prompt,
floor 0.3 for all) and the new one; *good / bad* are the hand labels
of the kept quotes:

| Session | Judge | Before: kept (good / bad) · rated | After: kept (good / bad) · rated |
|---|---|---|---|
| discussion | Claude | 53 (51 / 2) · 7 | 111 (104 / 7) · 21 |
| interview | Claude | 11 (10 / 1) · 1 | 20 (18 / 2) · 4 |
| discussion | Qwen3-8B | 58 (25 / 34) · 5 | 45 (23 / 22) · 7 |
| interview | Qwen3-8B | 23 (12 / 11) · 2 | 14 (9 / 5) · 1 |
| discussion | Qwen3-4B | 64 (23 / 41) · 7 | 34 (19 / 15) · 4 |
| interview | Qwen3-4B | 32 (7 / 25) · 4 | 15 (7 / 8) · 1 |

Claude now keeps every good quote it finds. The local models keep
fewer quotes, nearly as many good ones (25 to 23, 12 to 9, 23 to 19,
7 to 7) and far fewer bad ones (34 to 22, 11 to 5, 41 to 15, 25 to
8); they rate a little less often (13 competencies instead of 18 over
the four runs), still on evidence that is about half wrong.

## 5. What changed

- `domain/skills/prompting.py`: the `guided` and `precise` prompt
  styles (`PromptStyle`); `standard` is unchanged, so Claude's prompts
  and its cached replies are the same as before.
- `application/skills/assessment.py`: `JudgeTuning` (prompt style and
  similarity floor) chosen per call from the name of the judge that
  answers, so a fallback from Claude to the local model in the middle
  of a session switches both. A quote below that judge's floor is not
  thrown away: it becomes a `weak_match` observation of the
  provisional tier (#103, `DESIGN.md` §13), never counts towards a
  firm level and keeps the estimate's confidence low.
- `config.py` / `skills_bootstrap.py`: `quote_min_similarity` (Claude,
  default 0), `local_quote_min_similarity` (local models, 0.4) and
  `local_prompt` (`precise`), also as `skills` options.
  `min_similarity` (0.3) now only filters the candidate tags of each
  sentence (`AUDIO.md` §4), which this ticket did not measure.
- `experiments/skills/assessment_quotes_eval.py`, its stored replies,
  labels and results under `experiments/skills/results/`; the
  synthetic interview `examples/skills/simulated/entrevista_corta.txt`
  and its `transcript.v1`.

## 6. Limits

- Two synthetic sessions written in this project and one labeller (an
  AI agent, not blind to the judge). The thresholds and the prompt
  choice can move with real data and a human reference set
  (`ANNOTATION_GUIDE.md`); they depend on the embedding model and must
  be measured again if it changes.
- The labels judge the quote, not the rating; whether a competency's
  rating is right was not measured.
- The local models' kept evidence is still about half wrong at the new
  floor: their ratings should be read as provisional, and Claude
  remains the judge to prefer when the session may leave the machine.
- The few quotes on the two real recordings are not explained by
  anything measurable here; a reference for them would need someone
  to label real sessions with consent.
- The 4B and 8B at 4 bits only; `enable_thinking` was not tried (it
  would need far more than 700 new tokens per call).
