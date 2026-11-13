# Soft-skills analyzer: design

A recorded group session becomes, per participant, a competency
scorecard in which every rating rests on quotes that can be checked in
the transcript. This document describes how the module is built and
used. The theory behind the rubric is in [`THEORY.md`](THEORY.md); how
to build reference data is in
[`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md).

Recordings are transcribed and diarized by the stage described in
[`AUDIO.md`](AUDIO.md) (`skills transcribe`), which also splits turns
into sentences and proposes candidate tags per sentence. This document
starts from the diarized transcript, however it was produced.

## 1. Pipeline

```
transcript ──► turns ──► evidence retrieval ──► judge ──► grounding ──► scorecard
(JSON/text)              (per participant ×      (LLM)     (rules)      (JSON + Markdown)
                          competency)
```

| Step | What it does | Where |
|---|---|---|
| Parse | Reads `[hh:mm:ss] Speaker: text` lines or diarizer JSON (`segments` with `speaker`, `text`, `start`, `end`) into ordered turns; merges consecutive segments of one speaker | `domain/skills/transcript.py` |
| Describe | Counts turns, words, share of speech, questions, participation equality. No model, no judgement | `domain/skills/interaction.py` |
| Retrieve | Ranks the participant's own turns against each behavioural indicator of the competency: `bm25/max + w·cosine`, the fusion of the document search | `application/skills/evidence.py` |
| Judge | A language model receives the rubric, the candidate turns (each after the turn it replies to) and returns quotes, the indicator each quote shows, a rating or null, and a rationale | `domain/skills/prompting.py`, `ChatModel` port |
| Ground | Deterministic checks on everything the model returned (next section) | `domain/skills/grounding.py` |
| Report | Scorecard as JSON (machine) and Markdown (reader), each rating followed by its quotes with turn number and timestamp | `application/skills/report.py` |

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

| `--judge` | Model | Data leaves the machine | Status |
|---|---|---|---|
| `local` (default) | Chosen automatically (`llm = "auto"`): the largest model of the ladder (Qwen3 0.6B to 14B) that fits the memory and, on a CPU, is estimated to reach 4 tokens/s; here `Qwen/Qwen3-1.7B`. `python -m src system` shows the choice and why the others were rejected; `--local_model` forces one | No | Tested. Quotes are verbatim, but indicator classification is weak; about 60 s per participant × competency on 20 cores |
| `api` | Claude (`claude-opus-5-5`) through the Anthropic SDK (`uv sync --group api`) | Yes | Adapter unit-tested against a fake client; not run against the live API in this repository |

Measured with the local judge on the synthetic Spanish session (5
speakers, 39 turns, 55 judgements, 58 minutes on 20 cores), against the
synthetic two-annotator consensus: the retriever placed 91% of the
reference evidence turns among the candidates; of 245 quotes proposed,
26 were not verbatim and 120 were dropped by the second opinion;
evidence precision 0.19 and recall 0.43; it rated 14 units where the
reference rates 9, with 4 in common (2 of them identical). It gave
Luis, who dismisses a colleague twice, 5/5 in listening. This judge is
a wiring test, not a usable assessor.

The API judge must only be used with recordings whose participants
consented to external processing.

## 5. Architecture

Same hexagon as the RAG; the analyzer reuses its ports and adapters and
adds its own.

| Layer | New | Reused |
|---|---|---|
| Domain | `languages`, `skills/transcript`, `rubric`, `assessment`, `interaction`, `grounding`, `agreement`, `prompting` | `ranking` (fusion, top-k), `experiments` (metrics, bootstrap, Pareto) |
| Application | `skills/evidence`, `assessment`, `gold`, `evaluation`, `lab`, `report`; ports `ChatModel`, `TranscriptRepository`, `CatalogueRepository` | ports `Embedder`, `LexicalIndexBuilder`, `Cache`, `Progress`, `TrialStore`; `ExperimentService`, `Tuner`, `AnalysisService` |
| Adapters out | `fs_transcript_repository`, `json_catalogue_repository`, `transformers_chat_model`, `anthropic_chat_model` | `bm25_index`, `minilm_embedder`, `sqlite_cache`, `jsonl_trial_store` |
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
uv run python -m src skills --judge api assess --transcript FILE

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
| Classify and evaluate in two separate model calls, the second seeing only the classified incidents (limits halo) | One call does both |
| Quality flag per incident (basic / clear / exemplary) | Not recorded |
| Opportunity gating: no disagreement or feedback episode in the session → insufficient evidence for everybody | Left to the judge and the evidence rules |
| Frequency normalised by opportunity and capped | Only the minimum-evidence and breadth rules |
| Low-confidence evidence from transcription confidence and overlap | Turns can carry a confidence and timed words (`AUDIO.md`); the Whisper adapter reports none yet and overlap is not detected |
| Person-level profile from ≥ 3 sessions (median and range), commitments log across sessions | Session level only |
| Team interaction profile (Bales / act4teams categories), leadership profile | Participation counts only |
| Validation with assessors, fairness checks | Tooling ready (§8), no data |

Other open items:

- All tunable values are provisional until reference data exists.
- The sample sessions and their annotations under `examples/skills/`
  are synthetic and only exercise the tooling.
- Data protection (consent, retention, access control, the EU AI Act
  obligations for employment uses) is out of scope for now and must be
  addressed before any real recording is processed.
