# Model card: the judges of the soft-skills analyzer

Date: 2026-11-16. Issue #86 (part 4). Format after Mitchell et al.
(2019), "Model Cards for Model Reporting". This is the "model card for
the judge" of `FAIRNESS.md` §3.5 row I4 and §4 action 4. One page; it
states what is known and what is not. **Per-group metrics do not exist
yet** (no real sessions, no group data): see `FAIRNESS.md` and the
audit template (`src/application/skills/fairness_report.py`,
`experiments/skills/fairness_audit.py`). Every figure below is on
synthetic data unless said otherwise.

## 1. Model details

A *judge* is the language model behind the `ChatModel` port that
confirms tags, labels dialogue acts, quotes evidence and writes the
profile (`src/domain/skills/judging.py`, `DESIGN.md` §4). It is not
trained or fine-tuned here; the project only prompts it.

| Judge | What it is | Id in the repo | Licence | Where it runs |
|---|---|---|---|---|
| `claude_code` | Claude through `claude -p` on the owner's subscription, isolated (`--safe-mode`, no tools, empty working directory, text on stdin; `DESIGN.md` §4.1) | `opus` (`DEFAULT_CLAUDE_CODE_MODEL`, `src/config.py`) | not recorded in the repo (hosted service) | Anthropic; the transcript text leaves the machine |
| `local` | Qwen3 at 4 bits (NF4, `bitsandbytes`), policy `preferred` | `Qwen/Qwen3-8B`, then `Qwen/Qwen3-4B` (`DEFAULT_LOCAL_PREFERENCE`); ladder `Qwen3-0.6B/1.7B/4B/8B/14B` (`SKILLS_MODEL_LADDER`) | not recorded in the repo | This machine; nothing leaves it |
| `both` | Both on the same items; both answers kept; agreement measured (`compare_labels`: agreement, Cohen's kappa, per-category kappa, confusion) | as above | as above | Both |
| `api` | Anthropic SDK, `claude-opus-5-5` (`DEFAULT_SKILLS_API_MODEL`) | kept, not to be used (owner decision on #74) | not recorded | Anthropic |

Choice: `judge = selected` (default) reads the web choice kept in
`data/skills/judge.json`, `claude_code` until something is chosen
(`DEFAULT_JUDGE_MODE`). Fallback (`FallbackJudge`,
`src/application/skills/judges.py`): Claude first; on the first
subscription usage-limit refusal (`QuotaExhaustedError`) the job
answers that item and the rest with the local judge and records
`fallback_reason`. A job that starts while the quota is known to be
out (until the reset the service gave, else 5 hours, unless Claude
answered since) goes straight to the local judge. Transient errors
(429, 5xx) and sign-in errors do not trigger the fallback. Every item
records the judge that produced it; replies are cached per judge.

## 2. Intended use

A draft aid for a trained human assessor (`DESIGN.md` §2): the judge
acts in step f (confirms or rejects candidate indicators per sentence,
up to `max_confirmations` = 150), step g (quotes the participant, files
each quote under an indicator, proposes a rating or none; grounding
and a similarity floor check it), step i (an explanatory profile per
person, Claude by default; local only if the owner does not allow
Claude) and `skills judge_acts` (one dialogue-act label per sentence;
the default tagger of step d is the encoder, not a judge,
`DIALOGUE_ACT_TAGGER.md` §9.5). Language: prompts in English,
transcripts in Spanish or English.

## 3. Out-of-scope uses

A sole decision on hiring, promotion or dismissal (`FAIRNESS.md` §3.6
H5); inferring emotions or traits; use on recordings whose participants
did not consent to external processing with `claude_code`, `both` or
`api` (`DESIGN.md` §4); rating from a session whose speakers were not
verified (`AUDIO.md` §4); treating the local judge's ratings as more
than provisional (`LOCAL_JUDGE_QUOTES.md` §6).

## 4. Factors

Relevant factors not yet measured by group: speaker sex, age, first
language and accent (through the transcript), register, Spanish vs
English, and transcription quality. The judge sees text only, never
audio, so accent and timbre enter only through transcription errors
(`FAIRNESS.md` §2). Group membership is not recorded anywhere in the
analysis; the judges are never given it.

## 5. Metrics

Accuracy and macro-F1 against intended labels (dialogue acts);
confirmed / rejected counts and position bias (tags); grounded-quote
share and hand-labelled precision, with a similarity floor per judge
(assessment); Cohen's kappa between judges (`both`). No metric by
group; no human-agreement metric on rated competencies.

## 6. Evaluation data

All synthetic or written by the project's own author; no human
reference set exists (`ANNOTATION_GUIDE.md`).

| Measure | Data | Source |
|---|---|---|
| Acts, 200 sentences of the synthetic Spanish set (51 dialogues, intended labels) | synthetic | `DESIGN.md` §4.2 |
| Tags, demo session `examples/skills/transcripts/demo_es.txt` | synthetic | `DESIGN.md` §4.2, `AUDIO.md` §4 |
| Quotes: simulated discussion (117 turns, 4 people) and a short interview (20 turns, 2 people); 740 hand labels by the implementing agent, not blind | synthetic | `LOCAL_JUDGE_QUOTES.md` §1 |
| Acts, 1,534 sentences (Qwen3-8B) and 140 (Claude) | synthetic Spanish set | `DIALOGUE_ACT_TAGGER.md` §2 |

## 7. Training data

Not applicable to the project (no judge is trained). The base models'
training data is not documented in this repository; "unknown" here.

## 8. Quantitative analyses

Synthetic data, one run each, greedy decoding for the local models.

| Result | Value | Source |
|---|---|---|
| Acts accuracy, Qwen3-1.7B / 4B (4 bit) / 8B (4 bit) | 0.365 / 0.545 / 0.620 on 200 sentences | `DESIGN.md` §4.2 |
| Tags confirmed of 64, same three | 40 / 63 / 58 (the 4B and 8B confirm almost everything) | `DESIGN.md` §4.2 |
| Acts top-1, Qwen3-8B (1,534) vs Claude `opus` (140) | 0.553 vs 0.743 | `DIALOGUE_ACT_TAGGER.md` §2 |
| Claude, 20 sentences: right / local 4B right / agreement / kappa | 14 / 10 / 8 / 0.36 ("smoke, not a benchmark") | `DESIGN.md` §4.4 |
| Grounded quotes that show their indicator: Claude vs Qwen3-8B vs Qwen3-4B | 122 of 131 vs 60 of 174 vs 50 of 215 | `LOCAL_JUDGE_QUOTES.md` §2 |
| Similarity floor: Claude / local | 0 / 0.4 | `LOCAL_JUDGE_QUOTES.md` §4 |
| Peak VRAM, Qwen3-8B / 4B at 4 bits | 6.31 GB / 2.87 GB | `DESIGN.md` §4.2 |
| By group (sex, age, language, accent) | not measured | `FAIRNESS.md` §3.3 J1, J4, J6 |

## 9. Ethical considerations

- The report profiles people; the AI Act treats employee evaluation as
  high risk (`FAIRNESS.md` §1). The report is a draft, every item
  cites a verbatim quote, and "insufficient evidence" is a valid answer.
- With `claude_code` the transcript leaves the machine; `local` keeps
  it. The `claude` call carries no project files or settings, but Claude
  Code still adds the OS, the date, the model name, the working
  directory and the account e-mail (`DESIGN.md` §4.1).
- Quote precision of the local judge (about a quarter before the floor,
  about a half after) means a quote under a rating can misrepresent a
  person; the floor trades recall for precision on that reasoning
  (`LOCAL_JUDGE_QUOTES.md` §4).
- Open fairness checks (`FAIRNESS.md` §3.3): J1 swap test, J2 order and
  length bias, J4 Spanish-English parity, J6 rejection rate by group:
  all pending. NIST AI 600-1 warns that an LLM may do worse in
  languages other than English or in some dialects (`FAIRNESS.md` §2).

## 10. Caveats and recommendations

- Synthetic data, one author, one hand labeller (an AI agent): figures
  may move with real data and human reference labels.
- Local models confirm most tags (position bias moved from "first
  indicator" to "never none"); none is a usable assessor of indicators
  (`DESIGN.md` §4.2). Prefer Claude when the session may leave the
  machine.
- Claude's real quota and sign-in messages were never seen; the
  patterns are tested with fakes only (`DESIGN.md` §4.1).
- Judge run-to-run variation was not separated (`SIMULATION.md` §3,
  `TASKS.md` SK-16). No timing of the CPU fallback was measured.
- Re-measure and re-audit when the judge, prompt, floor, rubric or
  embedding model changes (`FAIRNESS.md` §3.7 G5).

## Sources

- `src/application/skills/judges.py`, `src/domain/skills/judging.py`
- `src/adapters/outbound/claude_code_judge.py`, `local_judge.py`
- `src/config.py` (`SkillsSettings`, `DEFAULT_*`), `src/skills_bootstrap.py`
- `docs/skills/DESIGN.md` §4, §12, §14; `LOCAL_JUDGE_QUOTES.md`;
  `AUDIO.md` §4–5; `DIALOGUE_ACT_TAGGER.md` §2, §9; `FAIRNESS.md`
- `docs/TASKS.md` (SK-7, SK-16, SK-20)
- `experiments/skills/results/judge_local_qwen3-8b-4bit.json`,
  `assessment_quotes.json`
