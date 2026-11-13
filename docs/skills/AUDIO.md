# From a recording to tagged sentences

Status: first working version, 2026-11-13. It extends the analyzer
upstream (audio in, instead of a transcript written by someone) and
downstream (tags per sentence, instead of evidence per turn). Numbers
below come from synthetic meetings and are **optimistic**; nothing
here has been measured on a real meeting yet.

## 1. Pipeline

```
recording ─► words with times ─► speaker per word ─► turns ─► sentences ─► candidate tags ─► (judge / annotator) ─► scorecard
            Whisper              voice embeddings     domain    domain      retrieval            existing analyzer
```

| Stage | What it does | Where |
|---|---|---|
| Read | WAV, FLAC, OGG, MP3 to mono 16 kHz (other containers through `ffmpeg` if installed) | `adapters/outbound/soundfile_audio_reader.py` |
| Recognise | Words with start and end times; language given or detected | `adapters/outbound/whisper_recogniser.py` (`openai/whisper-large-v3-turbo`) |
| Diarize | Who spoke when, in two passes: 5 s windows find the voices, 1.5 s windows are assigned to the nearest voice | `adapters/outbound/xvector_diarizer.py` (`microsoft/wavlm-base-plus-sv`), maths in `domain/skills/speech.py` |
| Join | Each word takes the speaker whose span overlaps it most; one-word speaker flips are undone; consecutive words of a speaker form a turn | `domain/skills/speech.py`, `application/skills/transcription.py` |
| Split | Turns into sentences (`t3.s1`) that keep speaker and exact times from their words | `domain/skills/speech.py` |
| Tag | Each sentence gets the rubric indicators it may show | `application/skills/tagging.py` |

Everything runs on this machine: no audio or text leaves it. The
speech recogniser and the diarizer are ports (`SpeechRecogniser`,
`Diarizer`), so either can be replaced without touching the rest.

```sh
# who said what, and when
uv run python -m src skills transcribe --audio meeting.wav \
    --speakers 4 --language es --names Ana,Luis,Marta,Pablo
# -> data/transcripts/meeting.json (turns, words, sentences)

uv run python -m src skills sentences --transcript data/transcripts/meeting.json
uv run python -m src skills tag --transcript data/transcripts/meeting.json
# -> data/output/skills/meeting.tags.json and a timeline

# the rest of the analyzer reads the same file
uv run python -m src skills assess --transcript data/transcripts/meeting.json
```

`--speakers` is optional but matters: with the number of people known,
diarization is clearly more reliable (below). `--names` maps voices in
order of first appearance; check the first turns before trusting it.

## 2. What was measured

`experiments/audio/meeting_eval.py` builds meetings by concatenating
utterances of several readers from Multilingual LibriSpeech (Spanish,
public, with transcripts and speaker ids) with 0.2–1.0 s of silence in
between. Six meetings, four speakers, sixteen turns, about four minutes
each; rows in `experiments/audio/meetings.jsonl`.

| Measure | Result |
|---|---|
| Word error rate (accents and punctuation ignored) | 5.4% mean (2.2%–8.6%) |
| Words attributed to the right speaker, number of speakers given | 91.5% mean (76.3%–99.7%) |
| Same, number estimated (threshold 0.5) | 90.2% mean (79.0% worst); voices found: 5, 4, 6, 3, 5, 4 for 4 real |
| Speed on the RTX 3070 Ti | about 22× real time for recognition; 1.5 s per meeting for the two diarization passes |
| GPU memory | 3.3 GB (two 30 s windows at a time) |

How the diarizer got there (same six meetings, speech boundaries given):

| Voice windows | Words right, speakers known |
|---|---|
| 1.5 s, raw embeddings | 55% |
| 1.5 s, centred on the recording's mean voice | 78% |
| 5 s, centred | 95% |

Short windows are too noisy to tell voices apart, and uncentred
embeddings share one large component, so the final design finds the
voices with long windows and only then labels short ones.

Two implementation notes that cost time: asking Whisper for word
timestamps makes `transformers` return every attention map, and the
encoder's take about 3 GB per window for nothing (they are switched
off); beam search multiplied memory further (decoding is greedy).

**Why these numbers are optimistic.** Read speech from audiobooks, one
person at a time, clean audio, no interruptions, no laughter, no
overlap, a pause between turns. `THEORY.md` §2.6 cites word error
rates of .19 to .35 by speaker group for commercial recognisers, and
automatic coding that drops from AUROC .91 on human transcripts to .80
on recognised ones. The diarizer cannot
represent two people talking at once: the window goes to one of them.

## 3. Sentences

A turn is too coarse for a rubric: one turn often holds a question, a
dismissal and a proposal. `split_sentences` cuts at `. ! ? …` when a
plausible sentence start follows, and does not cut after abbreviations
("Sr.", "etc.", "e.g."), initials or decimals. A sentence keeps:

```json
{"id": "t1.s0", "turn": 1, "index": 0, "speaker": "Tom",
 "text": "That is nonsense, really.", "start": 2.0, "end": 3.3}
```

Times are exact when the transcript comes from audio (first and last
word of the sentence). For transcripts written by hand they are
interpolated inside the turn by length, which is an estimate.

`sentence_transcript` presents the sentences as the "turns" of a
transcript, so evidence retrieval, the judge and the grounding checks
run at sentence level without a second implementation.

## 4. Tags: where the research meets the transcript

The rubric (`knowledge/rubrics/core.*.json`) is the operational form of
`THEORY.md`: 11 competencies, 110 observable indicators, positive and
negative. A **tag** is one link:

```
sentence (who, when, what was said)  ──►  indicator (which behaviour, of which competency, for or against)
```

`skills tag` produces **candidate** tags: for every sentence, the
indicators closest to it by the same lexical + semantic retrieval the
analyzer already uses, kept when the cosine reaches `min_similarity`.
No language model is involved. On the Spanish demo session:

```
[00:00:44] t3.s0 Luis: Eso es una tontería, Marta, con un proceso sobra.  <+COLLAB.P4 0.49, -CONFL.N4 0.52>
[00:00:51] t4.s0 Pablo: Espera, Luis, déjala terminar.  <+LISTEN.P4 0.54, +COORD.P1 0.37>
```

Read against the rubric, **none of these four candidates is right**.
The first sentence is `CONFL.N2` ("rejects an idea outright without
giving reasons") and the second is `INCL.P2` ("gives the floor back to
someone who was interrupted"); neither is among the six best candidates
of its sentence. The *competency* is found in both cases (`CONFL` for
the first, `INCL` for the second, through other indicators). Two
sentences of a synthetic session prove nothing, but they match what
the method can do: similarity finds what a sentence is *about*, not
which behaviour *happened*, and an indicator such as "gives the floor
back" shares no words with "Espera, Luis, déjala terminar".

So candidates should be used at the level they are reliable at: they
say **which competencies to look at for a sentence**; choosing the
indicator, and deciding whether the behaviour occurred, needs a reader
that sees the context. Tags therefore have three states:

| State | Made by | Meaning |
|---|---|---|
| candidate | retrieval (`skills tag`, done) | worth a look: which competencies this sentence may bear on |
| confirmed / rejected | the judge (`skills tag --confirm`, below) or an annotator | which indicator of that competency, if any, the sentence shows |
| counted | the existing evidence rules | enough confirmed tags from enough separate exchanges to support a rating |

### Confirmation by the judge (call 1), and what it is worth today

`skills --judge local tag --transcript FILE --confirm=True` asks the
judge, for each sentence with candidates, which indicator of the
candidate competencies the sentence shows, or none. It sees the three
sentences said just before it, with their speakers, and the full indicator
list of those competencies (not only the retrieved candidates). The
reply is one line, parsed strictly; an indicator outside the list
counts as not understood. Decisions are cached and written next to the
candidates (`[+ID]`, `[-ID]`, `[none]`, `[?]` in the timeline).

Run on the Spanish demo session with the local judge this machine
selects (Qwen3-1.7B), 49 s on the GPU: 64 sentences with candidates,
40 confirmed, 23 rejected, 1 not understood. **The result is not
usable**:

- 35 of the 40 confirmed tags are the *first* indicator of their
  competency (`P1`), whatever the sentence says: "En la página cuatro."
  became `LISTEN.P1`. That is position bias, not reading.
- No negative indicator was confirmed in a session written to contain
  several. "Eso es una tontería, Marta, con un proceso sobra."
  (`CONFL.N2`) came back as none.
- "Espera, Luis, déjala terminar." came back as `LISTEN.P1` instead of
  `INCL.P2`.
- About five look right on reading, all of them explicit: `INCL.P1`
  for "Irene, no has dicho nada todavía, ¿cómo lo ves tú?",
  `COORD.P5` for the closing summary, `CONFL.P1` for the apology,
  `COLLAB.P2` for asking for help in advance.

So the mechanism works and a 1.7 B model cannot do the task. What
would: a larger judge (the next local step, Qwen3-4B, does not fit in
8 GB at float16 without quantisation; the API judge sends text to a
third party, see §5), indicators shown in shuffled order to measure and
cancel position bias, and above all reference annotations to measure
any of it (T-36). No rating should be computed from these tags.

### What the research requires of the next step

| From `THEORY.md` | Consequence for tagging |
|---|---|
| Classify and evaluate in two separate calls (limits halo) | Call 1 takes a sentence, its context and the full indicator list of the candidate competencies, and returns the indicator shown or none; call 2 rates the competency from confirmed tags only. Candidates keep call 1 to two or three competencies instead of eleven. |
| A behaviour is defined in context ("lets the other finish", "builds on a previous idea") | The judge must see the sentences around the candidate and who said them; a sentence alone is not enough. Sentence ids make that window explicit. |
| A rating needs ≥ 3 incidents from ≥ 2 exchanges | Already enforced on turns; with sentences, two tags in one turn must still count as one exchange (`episode_gap` in sentence units needs re-setting). |
| Opportunity: no disagreement episode, no evidence on handling disagreement | Tags of *other* speakers define opportunities: a `CONFL` candidate by A opens an opportunity for B. Needs tags per sentence, which now exist. |
| Interaction measures: interruptions, question asking, uptake, share of speech | Speaking rate (words per minute) and median response latency are computed from turn times (`skills interaction`). Interruptions need a diarizer that detects overlap. |
| ASR errors lower automatic coding (AUROC .91 on human transcripts vs .80 on ASR ones) | Each turn keeps the recogniser's confidence when available (not with the current Whisper adapter, which reports none); low-confidence sentences should be flagged, not scored. |
| Diarization errors credit the wrong person | Show the speaker and the time with every tag so that a person can check it against the audio; never rate from a session whose speakers were not verified. |
| Developmental feedback with a human in the loop | Tags are shown as a timeline to the participant and the assessor; a rating is a summary of confirmed tags, each traceable to a sentence and a time. |

### Three ways to confirm tags, in order of cost

1. **Judge per candidate** (no new data): implemented as call 1 above;
   not reliable with the small local judge. Call 2 (rating from
   confirmed tags only) is not built and should wait for a call 1 that
   can be trusted.
2. **Annotators** (needs people): an evidence entry of an annotation
   form may now carry a `sentence` id besides its turn (validated
   against the transcript; forms with and without it can be mixed).
   This produces the reference data everything else needs (T-36).
3. **A trained classifier** (needs that data): once a few thousand
   sentences are annotated, a small supervised model per indicator
   family is cheaper and steadier than a judge, as in the CPS
   literature cited in `THEORY.md` §2.6.

## 5. Before the first real recording

The user deferred data protection while the tooling was built
(2026-11-12). It has to be settled **before any real meeting is
recorded**, because audio changes the picture:

- A voice recording of an identifiable person is personal data, and
  rating people from it is profiling. Participants need to know what
  is recorded, what is derived and who sees it, and to agree.
- The pipeline is local by design. Keep it local: the API judge sends
  transcript text to a third party and needs its own decision.
- Voice embeddings are computed in memory and never stored; storing
  them to recognise a person across sessions would be biometric
  identification and should not be added casually.
- Decide how long audio is kept. The transcript, not the audio, is
  what the analyzer needs after `transcribe`.

## 6. Open items

| Item | Why it matters |
|---|---|
| Measure on real meetings (overlap, noise, accents) | Every number above is from read speech |
| A diarizer that handles overlap (for example pyannote; its models are licence-gated) behind the same port | Interruptions are both an error source and a behaviour the rubric cares about |
| Word confidence (Whisper reports none through this adapter) | Flagging unreliable evidence |
| A judge that can do call 1 (larger model, shuffled indicator order), then call 2 | The 1.7 B local judge picks the first indicator of the list |
| Interruptions and overlap | Needs the overlap-aware diarizer |
