# Soft-skills analyzer: design

A recorded group session becomes, per participant, a competency
scorecard in which every rating rests on quotes that can be checked in
the transcript. This document describes how the module is built and
used. The theory behind the rubric is in [`THEORY.md`](THEORY.md); how
to build reference data is in
[`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md).

Recordings are transcribed, diarized and attributed by the
transcriber, a separate package
([`docs/transcriber/ARCHITECTURE.md`](../transcriber/ARCHITECTURE.md)),
which hands over a `transcript.v1` document (`contracts/`). What the
analyzer does with it (sentences, candidate tags per sentence,
confirmation) is in [`AUDIO.md`](AUDIO.md). This document starts from
the diarized transcript, however it was produced.

## 1. Pipeline

```
transcript ──► turns ──► evidence retrieval ──► judge ──► grounding ──► scorecard
(JSON/text)              (per participant ×      (LLM)     (rules)      (JSON + Markdown)
                          competency)
```

| Step | What it does | Where |
|---|---|---|
| Parse | Reads the transcriber's `transcript.v1` (validated strictly; one turn per contract turn, utterance ids kept as sentence ids, speakers by name, previous speaker kept), `[hh:mm:ss] Speaker: text` lines or diarizer JSON (`segments` with `speaker`, `text`, `start`, `end`) into ordered turns; merges consecutive segments of one speaker | `domain/skills/transcript.py`, `domain/skills/transcript_contract.py` |
| Describe | Counts turns, words, share of speech, questions, participation equality. No model, no judgement | `domain/skills/interaction.py` |
| Retrieve | Ranks the participant's own turns against each behavioural indicator of the competency: `bm25/max + w·cosine`, the fusion of the document search | `application/skills/evidence.py` |
| Judge | A language model receives the rubric, the candidate turns (each after the turn it replies to) and returns quotes, the indicator each quote shows, a rating or null, and a rationale | `domain/skills/prompting.py`, `ChatModel` port |
| Ground | Deterministic checks on everything the model returned (next section) | `domain/skills/grounding.py` |
| Report | Scorecard as JSON (machine) and Markdown (reader), each rating followed by its quotes with turn number and timestamp; a plain-language HTML page per participant (§10) | `application/skills/report.py`, `person_report*.py` |

## 2. What is verified, and what is not

The model's output never reaches the scorecard unchecked.

| Check | Rule | Catches |
|---|---|---|
| Quote exists | ≥ 80% of the quote's words appear as one contiguous run (≥ 3 words) in a candidate turn; case, accents and punctuation ignored | Invented or paraphrased quotes |
| Speaker | The turn must be spoken by the assessed participant | Credit for someone else's words |
| Citation | A quote found in another of the participant's turns is re-attributed to it | Wrong turn number |
| Indicator | The indicator id must belong to the competency; polarity is read from the rubric, never from the model | Invented indicators, flipped signs |
| Second opinion | The cosine between the turn and the cited indicator must reach `min_similarity` | A real quote filed under a behaviour it does not show |
| Minimum evidence | A rating needs `min_incidents` (3) distinct turns of surviving evidence from `min_episodes` (2) separate exchanges, else "insufficient evidence" | Ratings from one remark or one exchange |
| Breadth at the extremes | The top rating needs two different positive indicators, the bottom one two negative incidents; otherwise it moves one point to the centre | Extreme ratings on narrow evidence |
| Participation floor | A participant under `min_participant_words` words gets no rating at all | Scoring silence as a deficit |
| Scale | Ratings off the scale are discarded | Format errors |
| Consistency | A high rating with only negative evidence (or the reverse) is flagged for review | Rating that contradicts its own evidence |

Counters `proposed`, `rejected` and `unsupported` are stored per
competency, so the hallucination rate of a judge model is a measured
number.

**Not verified by rules:** whether a verbatim quote really shows the
behaviour (the second opinion is a heuristic), and whether the rating
is the one a trained assessor would give. Those two are measured
against human reference data (`skills evaluate`) and are the reason
the output is labelled a draft for a human assessor.

## 3. Languages

| Layer | English | Spanish |
|---|---|---|
| Lexical analysis | Tokenizer of the RAG | Accent folding, Spanish stopwords, light stemmer (`domain/languages.py`) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, one vector space for both languages | same |
| Rubric | `knowledge/rubrics/core.en.json` | `knowledge/rubrics/core.es.json`, same ids |
| Judge | Prompt in English, rationale in the transcript's language | same |

The transcript language is read from the file or detected. With
`rubric_language auto` the rubric of that language is used, so the
lexical part of retrieval contributes; a rubric in another language
still works through the embeddings alone.

## 4. Judge models

Owner decision on #74 (2026-11-15): the paid Anthropic API is not
used. The judge is **Claude through Claude Code** (the `claude`
command line, signed in with the owner's subscription), with a **local
model** behind it when the subscription's quota runs out, and a
**both** mode to compare them. Consented recordings may be sent out
(decision 5 on #57); only synthetic examples were sent while building
this.

| `--judge` | What runs | Data leaves the machine | Status |
|---|---|---|---|
| `selected` (default) | Whatever was chosen from the web (`GET/PUT /analysis/judge`, kept in `judge_state_path`, `data/skills/judge.json`); `claude_code` until something is chosen | as the choice | Tested with fakes; the choice file holds no transcript text |
| `claude_code` | `claude -p` with `--model opus` (`--claude_model`), isolated as in §4.1; on a quota refusal the job goes on with the local judge (§4.3) | Yes, to Anthropic through the subscription | Run on 20 synthetic sentences (§4.4) |
| `local` | On a GPU, Qwen3-8B at 4 bits if it fits the free memory, otherwise Qwen3-4B at 4 bits after waiting for room; never a smaller model (`local_policy`, `local_preference`, §4.2). On a CPU, the machine's choice as before. `--local_model` forces one | No | Measured on synthetic data (§4.2) |
| `both` | `claude_code` and `local` on the same items; both answers kept, agreement reported | Yes | Run on 20 synthetic sentences (§4.4) |
| `api` | Claude through the Anthropic SDK (`AnthropicChatModel`) | Yes | Kept, **not to be used** (costs extra money) |

Every item records the judge that produced it (`TagDecision.model`,
`ActVerdict.model`); the output file of `skills tag --confirm` and
`skills judge_acts` has a `judge` record with the mode, the model and,
after a fallback, `fallback_reason`. Replies are cached per judge, so a
reply of the local judge is never filed under Claude's name.

### 4.1 Claude Code judge and its isolation

`adapters/outbound/claude_code_judge.py` runs one call per item:

```
claude -p --safe-mode --setting-sources "" --strict-mcp-config --tools "" \
  --disable-slash-commands --no-session-persistence --output-format json \
  --model opus --system-prompt <our prompt>      (user message on stdin)
```

from a new empty temporary directory (removed afterwards), with the
environment reduced to `PATH`, `HOME`, `LANG`, `LC_ALL`. The transcript
text goes on standard input, never on the command line (readable in the
process list), and is never logged nor put in an error message.

- `--bare` would isolate as well, but then Claude Code accepts only an
  API key (`ANTHROPIC_API_KEY`), not the subscription's OAuth sign-in
  (`claude --help`, 2.1.325), so it is not used.
- `--safe-mode` disables CLAUDE.md, skills, plugins, hooks, MCP
  servers, custom agents and output styles while keeping the sign-in;
  `--setting-sources ""` loads no settings file; `--strict-mcp-config`
  without `--mcp-config` loads no MCP server; `--tools ""` gives no tool.
- Checked with real calls (Claude Code 2.1.325, 2026-11-15): a
  `CLAUDE.md` with a secret word and a `.claude/notes.md` with another
  were put in the working directory, and the judge was asked what
  instructions, CLAUDE.md, memory, project files, tools and secret
  words it could see. It answered: our system prompt only; no
  CLAUDE.md, memory or project files; no tools; no secret word. No
  session was written under `~/.claude/projects`. What Claude Code
  still adds and cannot be turned off with the subscription: the
  operating system, the date, the model name, the working directory
  (the empty temporary one) and the account's e-mail address.
- A first try that asked for the system prompt "verbatim" was refused
  by the model's safeguards (`is_error: true`); such refusals are
  reported as ordinary failures, not as quota.

The JSON result is parsed (`result`, `is_error`, `api_error_status`).
Failures are sorted into four kinds:

- **Quota**: only the subscription's usage-limit messages ("Claude AI
  usage limit reached|<timestamp>", "hit your … limit", "5-hour /
  session / weekly limit reached", "out of extra usage") raise
  `QuotaExhaustedError`. The reset time is kept as written and, when
  it is a Unix timestamp, as a moment (`resets_at`).
- **Transient**: a per-minute rate limit (`429`, "rate limit", "Too
  Many Requests"), an overload or a `5xx`. These are retried twice
  (5 s, then 10 s), change no stored state, and end as a plain
  `ModelError`. They never send the job to the local judge.
- **Sign-in**: only `401` or the not-logged-in messages ("Not logged
  in", "Please run /login", "Invalid API key", "OAuth token has
  expired") raise `ModelAuthError`; the fix is to run `claude` once
  and log in.
- **Anything else** (a time-out after `claude_timeout`, 180 s; a
  missing program; a safeguard refusal) is a `ModelError`.

An error message carries at most Claude Code's own error text, never
a reply. `claude` starts in a new session (process group); on a
time-out or any other exception the whole group is killed, and
helpers left behind by a finished call are killed too. The real
quota and sign-in messages were not seen, because they could not be
triggered. The patterns come from the messages Claude Code is known
to print and are tested with fakes.

### 4.2 Local judge

The GPU (8 GB) is shared with the live transcriber, the RAG and another
project's worker, so the free memory changes from job to job.
`adapters/outbound/local_judge.py` reads the free memory
(`torch.cuda.mem_get_info`) when a job first needs the judge, keeps
0.4 GB aside and takes each model's needs from its measured peak on
the GPU when there is one (`MEASURED_GPU_PEAK_GB` in
`domain/runtime.py`: Qwen3-8B at 4 bits 6.31 GB, Qwen3-4B at 4 bits
2.87 GB, from the table below), otherwise estimates them as
`params × bits/8 × 1.3`. The formula said 5.3 GB for the 8B, so with
5.7–6.3 GB free the policy would have tried it and run out of memory;
it now needs 6.71 GB free (6.31 + 0.4). The weights are quantised to 4 bits (NF4)
with `bitsandbytes`, installed by `uv sync --group local-judge`
(`bitsandbytes`, `accelerate`; not a default group). The job releases
the model when it ends (`release()`), so the memory comes back.

`local_policy` decides which model loads:

- **`preferred`** (default; owner decision on #74):
  - The list is `local_preference`, by default Qwen3-8B then Qwen3-4B,
    both at 4 bits.
  - The first model of the list that fits the free memory now is
    loaded.
  - When none fits, the job waits for the last model of the list,
    checking every 15 s and logging what it waits for. It gives up
    after `local_wait_seconds` (600 s) with a clear error.
  - It never takes a model outside the list, so it never downgrades
    to 1.7B.
  - The list is a setting, so the 8B can be dropped if it measures
    worse.
- **`largest_fitting`**: the largest model of `model_ladder` that fits
  the free memory now. Without the quantisation packages it logs a
  warning and chooses among the models that fit at 16 bits.
- **`fixed`**: pins `--local_model` and waits for it, like `preferred`.

`--local_model` set explicitly loads that model under `preferred` and
`largest_fitting` too. With `preferred` and no quantisation packages,
the 8B and 4B do not fit an 8 GB card at 16 bits. The judge then
fails at once, saying to install the group, rather than load another
model.

Measured on 2026-11-15 on the RTX 3070 Ti (one run each, greedy
decoding). The 8B row comes from the committed
`experiments/skills/judge_local_eval.py`, run on the idle GPU; its
output is `experiments/skills/results/judge_local_qwen3-8b-4bit.json`
(GPU memory back to 0.24 GB after release). The 1.7B and 4B rows come
from a scratch version of the script, which computes the same values,
and have no committed result file. To reproduce:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/skills/judge_local_eval.py Qwen/Qwen3-8B --bits 4 \
    --output experiments/skills/results/judge_local_qwen3-8b-4bit.json
```

(`Qwen/Qwen3-4B --bits 4` for the 4B row.)

Dialogue acts: 200
sentences spread over the 51 dialogues of the synthetic Spanish set,
with the 3 previous sentences, against their intended labels. Tag
confirmation: the synthetic Spanish demo session
(`examples/skills/transcripts/demo_es.txt`), as in `AUDIO.md` §4.

| Model | Peak VRAM of the judge | Load | Acts: accuracy (unparsed) | s / sentence | Tags: confirmed / rejected / not understood | First indicator (P1) among confirmed | Negative indicators confirmed |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B, 16 bit | 3.54 GB | 4.8 s | 0.365 (3) | 0.16 | 40 / 23 / 1 | 35 of 40 | 0 |
| Qwen3-4B, 4 bit | 2.87 GB | 39.1 s | 0.545 (1) | 0.32 | 63 / 1 / 0 | 10 of 63 | 5 |
| Qwen3-8B, 4 bit | 6.31 GB | 19.7 s | 0.620 (0) | 0.39 | 58 / 6 / 0 | 13 of 58 | 7 |

The 1.7B run reproduces `AUDIO.md` §4 exactly. The 4B at 4 bits is
better on dialogue acts (0.545 against 0.365) and its position bias is
gone, but on tag confirmation it confirms almost everything (63 of 64):
the failure moved from "always the first indicator" to "never none".
Qwen3-8B at 4 bits is better than the 4B on both tasks: 0.620 on
dialogue acts (0.545) and it rejects 6 of 64 tags instead of 1, at
0.39 s per sentence (tags: 0.99 s). It still confirms most tags (58 of
64), so none of the three is a usable assessor of indicators; the 8B
stays first in `local_preference`. The preferred policy loads it when
6.71 GB are free (measured 6.31 GB + 0.4 GB margin); otherwise it
waits for Qwen3-4B (2.87 + 0.4 GB). One run on synthetic data each.

### 4.3 Selection and fallback

`application/skills/judges.py`. `JudgeSelection` keeps the web's
choice and Claude's quota record (`JudgeState`: last refusal, its reset
time, last answer). `FallbackJudge` asks Claude; on the first
`QuotaExhaustedError` of a job it records the refusal and answers that
item and all the following ones with the local judge, giving the reason.
A job that starts while the quota is known to be out goes to the local
judge without calling Claude. "Known to be out" means: until the reset
moment the service gave or, without one, for 5 hours (the
subscription's window), unless Claude answered since. The state file
(`judge_state_path`, relative to the repository root, not to the
working directory) is changed under a file lock. The web's choice and
a job's quota record therefore never overwrite each other. The web
shows a reset moment in local time. The quota shown in the web
is therefore what the last jobs saw (`available`, `exhausted`, or
`unknown` before the first call), not a live probe: a probe would
spend quota.

In `both` mode, the first judge falls back to the same local model as
the second when the quota runs out. Items that one model answered for
both are left out of the agreement figures and counted in
`same_judge`, and the command says so. Otherwise κ would tend to 1 by
comparing the local model with itself.

The RAG serves `GET/PUT /analysis/judge` behind the transcript token
(`adapters/inbound/http_api.py`); the transcriber forwards
`GET/PUT /api/analysis/judge` (access code) to it on loopback with the
same token (redirects are never followed with the token, here or when
publishing), so the RAG stays out of the tunnel; the sessions page has
a «Análisis de habilidades: modelo que juzga» section with the three
choices and the quota line.

### 4.4 Both mode

`compare_confirmations` (tags) and `compare_act_judges` (dialogue acts)
run the two judges on the same items and return `LabelComparison`
(`domain/skills/judging.py`): agreement, Cohen's κ over all labels,
κ per category (one against the rest), the confusion between the two
judges and, when the items have reference labels, each judge's
accuracy overall and per category. `skills --judge both judge_acts
--sample 20` does it on the Spanish set.

**Smoke figure, not a benchmark** (2026-11-15, 20 sentences spread
over the set, Claude Code `opus` against Qwen3-4B at 4 bits, intended
labels of a synthetic set written by one author): Claude 14 of 20 right
(0.70), the local judge 10 of 20 (0.50); the two agreed on 8 of 20,
κ 0.36. Twenty sentences cannot rank two models; this shows the
mechanism works end to end, nothing more.

### 4.5 Earlier measurement (assessment, local 1.7B)

Measured with the local judge on the synthetic Spanish session (5
speakers, 39 turns, 55 judgements, 58 minutes on 20 cores), against the
synthetic two-annotator consensus: the retriever placed 91% of the
reference evidence turns among the candidates; of 245 quotes proposed,
26 were not verbatim and 120 were dropped by the second opinion;
evidence precision 0.19 and recall 0.43; it rated 14 units where the
reference rates 9, with 4 in common (2 of them identical). It gave
Luis, who dismisses a colleague twice, 5/5 in listening. This judge is
a wiring test, not a usable assessor.

Hosted judges (`claude_code`, `both`, `api`) must only be used with
recordings whose participants consented to external processing.

## 5. Architecture

Same hexagon as the RAG; the analyzer reuses its ports and adapters and
adds its own.

| Layer | New | Reused |
|---|---|---|
| Domain | `languages`, `skills/transcript`, `rubric`, `assessment`, `interaction`, `grounding`, `agreement`, `prompting`, `profile` | `ranking` (fusion, top-k), `experiments` (metrics, bootstrap, Pareto) |
| Application | `skills/evidence`, `assessment`, `gold`, `evaluation`, `lab`, `report`, `person_report*`, `profile`, `profile_report*`; ports `ChatModel`, `TranscriptRepository`, `CatalogueRepository`, `AssessmentStore` | ports `Embedder`, `LexicalIndexBuilder`, `Cache`, `Progress`, `TrialStore`; `ExperimentService`, `Tuner`, `AnalysisService` |
| Adapters out | `fs_transcript_repository`, `fs_assessment_store`, `json_catalogue_repository`, `transformers_chat_model`, `local_judge`, `claude_code_judge`, `json_judge_state`, `anthropic_chat_model` (kept, not used) | `bm25_index`, `minilm_embedder`, `sqlite_cache`, `jsonl_trial_store` |
| Adapters in | `skills_cli` | `lab_cli` (now hosted by any container offering `experiments`, `tuner`, `analysis`) |
| Composition | `skills_bootstrap.SkillsContainer`, `config.SkillsSettings` | |

Changing the diarizer output format, the judge model, the embedding
model or the rubric touches one adapter or one JSON file.

## 6. Commands

Options go between `skills` and the command. The cache of judge
replies is disabled with the global `--no_cache` after the command.

```bash
# No model needed
uv run python -m src skills catalogue --language es
uv run python -m src skills interaction --transcript FILE

# Retrieval only (embedding model)
uv run python -m src skills evidence --transcript FILE \
    --participant NAME --competency ID

# Full scorecards → data/output/skills/<session>.assessment.json + .scorecard.md
uv run python -m src skills assess --transcript FILE
uv run python -m src skills --judge local assess --transcript FILE

# Judges (§4): dialogue acts of the synthetic Spanish set, two judges compared
uv run python -m src skills --judge both judge_acts --sample 20

# One participant, for a non-technical reader (§10) → <session>.<name>.report.html
uv run python -m src skills report --assessment S.assessment.json \
    --participant NAME [--language en] [--transcript FILE] [--synthetic]

# One person across sessions (§11) → <output_dir>/<person id>.profile.html
uv run python -m src skills profile --participant PERSON_ID \
    [--sessions S1,S2,...] [--assessments_dir DIR] [--format json] \
    [--language en] [--synthetic]

# Reference data
uv run python -m src skills annotation_template --transcript FILE --annotator A1
uv run python -m src skills validate_annotations --annotations F --transcript FILE
uv run python -m src skills agreement --annotations A1.json,A2.json
uv run python -m src skills consensus --annotations A1.json,A2.json
uv run python -m src skills evaluate --assessments S.json --reference GOLD.json

# Tuning evidence retrieval with the experiment lab
uv run python -m src skills lab_dataset --reference GOLD.json --output D.json
uv run python -m src skills --transcripts_dir DIR lab experiment \
    --suite retrieval --datasets D.json
uv run python -m src skills --transcripts_dir DIR lab tune \
    --suite retrieval --datasets D.json
uv run python -m src skills lab analyze
```

## 7. Data formats

Transcript (either form):

```
[00:00:05] Ana: Vale, empezamos.
[00:00:21] Luis: Yo ya lo tengo claro.
```

```json
{"session_id": "s1", "language": "es",
 "segments": [{"speaker": "Ana", "start": 5.0, "end": 20.5, "text": "Vale, empezamos."}]}
```

Rubric catalogue (`knowledge/rubrics/<name>.<lang>.json`): `name`,
`version`, `language`, `scale`, and `competencies`, each with `id`,
`name`, `definition`, `indicators` (`id`, `text`, `polarity`,
`examples`), `anchors` (scale point → behavioural description),
`sources` and `limits`. Every language file of a catalogue has the same
ids; a test enforces it.

Annotation set (one annotator, one session): `session_id`, `annotator`,
`catalogue`, `catalogue_version`, and one `annotation` per participant
× competency with `rating` (or `null` for insufficient evidence) and
`evidence` (`turn`, `indicator_id`).

## 8. Metrics

| Question | Metric | Command |
|---|---|---|
| Is the reference data reliable? | Krippendorff's ordinal alpha and ICC(2,1)/(2,k) on ratings, kappa on rated-or-not, pairwise F1 on evidence turns | `skills agreement` |
| Does retrieval find the evidence? | Share of reference turns among the candidates; recall@k and MRR in the lab | `skills evaluate`, `skills lab` |
| Does the judge pick the right turns? | Turn-level precision, recall, F1 against the reference | `skills evaluate` |
| Does it rate like the assessors? | Exact and within-one agreement, MAE, quadratic weighted kappa, alpha between system and consensus | `skills evaluate` |
| Does it abstain when it should? | Rated-or-not table against the reference | `skills evaluate` |
| How much does the judge invent? | Share of proposed quotes found verbatim; count dropped by the second opinion | `skills evaluate` |

The system's agreement with the consensus should be read against the
annotators' agreement with each other: that is the ceiling.

## 9. Open items

Recommended by `THEORY.md` and not implemented yet:

| Recommendation | Status |
|---|---|
| Classify and evaluate in two separate model calls, the second seeing only the classified incidents (limits halo) | Call 1 exists per sentence (`skills tag --confirm`, `AUDIO.md` §4) but the 1.7 B local judge is unreliable at it; the scorecard still uses one call |
| Quality flag per incident (basic / clear / exemplary) | Not recorded |
| Opportunity gating: no disagreement or feedback episode in the session → insufficient evidence for everybody | Left to the judge and the evidence rules |
| Frequency normalised by opportunity and capped | Only the minimum-evidence and breadth rules |
| Low-confidence evidence from transcription confidence and overlap | Turns can carry a confidence (from `transcript.v1`, `AUDIO.md`); the transcriber's Whisper adapter reports none yet and overlap is not detected |
| Person-level profile from ≥ 3 sessions (median and range), commitments log across sessions | Profile done (`skills profile`, §11), exercised on a synthetic example only; commitments log not done |
| Team interaction profile (Bales / act4teams categories), leadership profile | Participation counts only |
| Validation with assessors, fairness checks | Tooling ready (§8), no data |

Other open items:

- All tunable values are provisional until reference data exists.
- The sample sessions and their annotations under `examples/skills/`
  are synthetic and only exercise the tooling.
- Data protection (consent, retention, access control, the EU AI Act
  obligations for employment uses) is out of scope for now and must be
  addressed before any real recording is processed.

## 10. Person report

`skills report --format html` turns one participant's scorecard into a
single page for someone who is not a specialist: Spanish by default
(`--language en`), short sentences, no request to any server (inline
CSS and SVG), readable at phone width, printable, light or dark.

| Part | Content | From |
|---|---|---|
| Resumen | 3–5 fixed sentences: length of the session, how many competencies have a level, strengths, points to improve, "draft for a human assessor" | assessment + `person_report_texts.py` |
| Qué significa | Situations seen (1) against the ~4 recommended (`RESEARCH.md` §1, §6), competencies with a level, length; what a level is; the limits box (no emotion or personality from the voice, speaking time is not leadership) | assessment, transcript, rubric scale |
| Gráficas | One row per competency: the level on its scale, a written band and a shape (circle strong, square middle, triangle to improve), filled dots for the evidence against the minimum; dashed and "sin datos suficientes" when unrated, never low. A radar of the rated ones only when there are ≥ 5 | assessment |
| Puntos fuertes / a mejorar | Rated above / below the centre of the scale, with their verified quotes (speaker, minute, indicator) and two examples of how the competency helps or hurts at work, in a team or in a negotiation | assessment, rubric `practice` |
| Conductas aisladas | Verified negative quotes outside the points to improve: too few for a level, shown for the assessor to check | assessment |
| Notas | Footnotes for competency, assessor, situation, level and evidence | `person_report_texts.py` |

No sentence is written by a model when the page is built. The
"helps / hurts" examples are hand-written in the catalogue
(`practice.helps`, `practice.hurts` per competency, both languages; not
yet reviewed by assessors) and the judge's free-text rationale is left
out. `--synthetic` adds a banner saying the data are made up; the
sample pages are built from `examples/skills/output/demo_es.assessment.json`,
the run of the 1.7 B local judge, so they show its mistakes as they are
(for example, Luis's "Eso es una tontería, Marta" filed as active
listening).

## 11. Profile across sessions

One conversation is one situation, and the same competency agrees only
about .27 between two situations (Putka & Hoffman 2013, `RESEARCH.md`
§6c): a person profile needs several sessions. `skills profile
--participant ID` joins the assessments of one person and shows, per
competency, each session's level and evidence side by side.

**Identity.** Sessions are joined by the id the transcriber enrolled
the speaker under: a `transcript.v1` speaker with `enrolled: true`
keeps its `id` as `Transcript.people`, and `skills assess` writes it
as `ParticipantScorecard.person_id` (with the session's `started_at`
and `conversation_type`). Names are never used to join: two Anas in
two sessions are not assumed to be one person. A session without
person ids (a text transcript, an older assessment, anonymous voices)
cannot be joined; when it is named with `--sessions` it is listed as
left out with the reason, never dropped silently. A person named
through the expected-speakers floor (`identification:
expected_floor`) is joined like any other; that weaker identification
is not yet down-weighted in the profile.

**Store.** The assessments are the `<session>.assessment.json` files
`assess` writes (`FileSystemAssessmentStore` over `output_dir`, or
`--assessments_dir`). Without `--sessions`, every stored session where
the person id appears is used; `--sessions` takes session ids or file
paths.

| Per competency | Rule | Where |
|---|---|---|
| Sessions | One cell per session in date order (undated after, by id); a competency not assessed in a session is an empty cell | `domain/skills/profile.py` |
| Situations with evidence | Sessions where the competency has a level; their declared conversation types (#65) are counted | same |
| Overall level | Median of those levels, only when at least `profile_min_situations` (setting, default 3) sessions have one. Sessions without a level are left out, never counted as zero. Below the minimum: "en construcción", with how many more sessions are missing | same |
| Consistency | Range and standard deviation of the levels; "consistent" when the range is at most one level | same |
| Trend | Least-squares slope of level against session number, with ≥ 3 levels; ±0.25 per session or more is up or down, otherwise flat | same |
| Reliability | With ≥ 3 levels: Spearman-Brown from .27 per situation (a projection from a published figure, not a measurement; about 6 situations for .70). When the store has at least three enrolled people with two or more levels in that competency, also ICC(1) among them, projected the same way | same |
| Repeated strengths / points to improve | Competencies in the same band (above or below the centre of the scale) in two sessions or more, with one verified quote per session | `application/skills/profile_report.py` |

The thresholds for "consistent" (range ≤ 1), the trend step (0.25)
and the two sessions for "repeats" are choices, not measurements; with
a handful of sessions a trend can be chance and the page says so.

**Page.** `--format html` (default) is the profile mode of the person
report (§10), same style and constraints: summary sentences, sessions
seen against the ~4 recommended and the ~6 one competency needs, what
the overall level is, the limits box (the person behaves differently
by situation; no emotion or personality; speaking time is not
leadership), a strip of session boxes per competency (dashed box: no
evidence in that session, never low), competencies never seen listed
apart, repeated strengths and points to improve with their quotes and
the rubric's `practice` lines, the list of sessions and those left out,
and footnotes. `--format json` writes the `PersonProfile` model.
Wording: `application/skills/profile_report_texts.py` (es, en).

**Example.** `examples/skills/profile/` holds four invented Spanish
conversations (decision meeting, problem solving, sprint review,
negotiation) of the same four invented people (Marta, Luis, Pablo,
Irene), with levels written by hand and checked against the evidence
rules of §2 by `experiments/skills/build_profile_example.py`, which
also writes the `transcript.v1` documents and the assessments (`check`
fails if they are out of date). It exercises the tooling and shows
nothing about real people:

```bash
uv run python -m src skills \
    --transcripts_dir examples/skills/profile/transcripts \
    profile --participant marta \
    --assessments_dir examples/skills/profile/output --synthetic
```

## 12. Whole flow in one run (`skills analyze`, #88)

`skills analyze --session <id>` runs every step on a session's
`transcript.v1` (found in `transcripts_dir`, where the RAG keeps the
transcripts the transcriber publishes) and stores each result under
`analysis_dir/<id>/` (default `data/output/skills/analysis`).

| Step | Result file | Who runs it |
|---|---|---|
| a. `sentences`: sentences of each person with their times | `sentences.json` | plain splitting (`domain/skills/speech.py`) |
| b. `interaction`: turns, words, share, overlaps, response times | `interaction.json` | `domain/skills/interaction.py` |
| c. `candidates`: candidate rubric indicators per sentence (the RAG step) | `candidates.json` | retrieval, embedding model named in the record |
| d. `acts`: dialogue acts per sentence, up to three weighted labels, "dudoso" below each category's threshold | `acts.json` (`acts.v1`) | port `DialogueActTagger`, adapter `TaggerActStep` over the #82 tagger (`skills acts`): the card's choice, the fine-tuned encoder (`data/models/act-tagger/…`), or the rules when its weights are missing; the record names the approach, the margin and how many first labels are doubtful; `unavailable` only when no tagger card exists |
| e. `events`: decisions, action items, open questions, fate of proposals | `events.json` | port `EventExtractor`: the rules of #83 (`conversation_events.analyse`) over step d's acts; `unavailable` without them |
| f. `confirmation`: the judge confirms or rejects the candidates | `confirmation.json` | the selected judge (§4.3); at most `max_confirmations` sentences, best candidate first |
| g. `assessment`: evidence per person and scorecards | `assessment.json`, `scorecard.md` | the same judge (§1) |
| h. `reports`: one HTML page per person, the session summary | `report-<n>.html`, `summary.json` | `person_report*.py` (§10) |

`run.json` holds one record per step: status (`pending`, `running`,
`done`, `failed`, `unavailable`), start time, seconds, who ran it
(model, judge mode, fallback reason) and counts. It is saved before and
after every step, so a failure keeps the earlier results; running the
command again resumes at the first step that is not `done` or
`unavailable` (`--restart` runs all; `--steps acts,events` reruns only
those, keeping the others and asking no judge). Every run also carries the three
things the flow leaves to people (`open_issues`): consent and retention
before real people (#19), human review of the labels, several
situations per person for a profile (#84).

One analysis per session at a time, whether started from the CLI or
the API: the run holds an `flock` on `analysis_dir/<id>/analysis.lock`
(which holds the worker's process id, also stored as `worker_pid` in
`run.json`); a second run gets `BusyError` (`409` through the API). The
operating system drops the lock when the worker dies, so a `run.json`
left `running` without a holder is shown as `interrupted` and the next
run resumes it at the step that was running.

Judge calls: step f makes one call per sentence with candidates (up to
`max_confirmations`, default 150) and step g one per person and
competency (11 in the core rubric). Replies are cached, so a resumed
run does not ask again.

Through the RAG (`docs/DEPLOYMENT.md`: loopback only, bearer token):
`POST /analysis/sessions/{id}` (body `{"restart", "synthetic"}`, `202`,
`409` while running, `404` without a transcript) starts
`python -m src skills ... analyze` as a child process (its output in
`<id>/analyze.log`); `GET /analysis/sessions/{id}` returns `run.json`
plus `active`; `GET …/steps/{step}` a step's result;
`GET …/reports/{n}` a person report with
`Content-Security-Policy: sandbox; default-src 'none'; style-src
'unsafe-inline'; img-src data:` and `Content-Disposition: attachment`.
The transcriber forwards the same four routes under
`/api/analysis/sessions/…` (access code; ids, step names and report
numbers checked before forwarding; redirects never followed with the
token) and its web has an «Análisis» page per session
(`#/analysis/<id>`). Reports are downloaded as files, never shown in
the pages' origin.

Simulated discussion: `examples/skills/simulated/reunion_verano.txt`
(4 invented people of an invented bakery, a decision meeting with
disagreement, proposals and commitments, ~15 minutes, original text)
becomes `transcript.v1` with `skills simulate` (times from a speaking
rate of 2.4 words per second, `+` marks a turn that cuts in 0.6 s
early). `make demo-analysis` runs `simulate` and `analyze` on it.

Demo, steps d and e added afterwards with `--steps acts,events` (no
judge call; `confirmation.json` byte-identical before and after),
2026-11-15, CPU: acts 11.4 s, encoder `xlmr-base-ami-mrda-es` on all
238 sentences, 51 with a doubtful first label (margin 0.45); events
0.0 s: 18 questions answered, 5 unanswered, 6 proposals accepted, 17
with no reply, 6 decisions, 18 action items, 6 open issues. First labels:
información 59, aceptación 32, propuesta 28, pregunta 24, respuesta 20,
compromiso 18, relación 12, opinión 12, escucha 10, rechazo 10,
gestión 8, petición 5. Not checked against a human reading of the
script.
