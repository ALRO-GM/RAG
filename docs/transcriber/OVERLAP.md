# Two people around one phone: identification and overlapped speech

Status: TR-14 (issue #21), experiment, 2026-11-14; what TR-17 built on
it is in §9. When this study ran nothing in
`transcriber/` changed. Every figure below is a row of
`experiments/transcriber/overlap.jsonl`, written by
`experiments/transcriber/overlap_study.py` (the row `kind` is given
with each table), except where a line says it comes from another
document or is a judgement.

## 1. Material and what can be claimed

- **Three real sessions** (R1, R2, R3: 388, 288 and 309 s) of two
  enrolled people, A and B, talking around one phone, recorded with the
  web app; both were listed as expected speakers and the count was 2.
  The audio, the text, the voiceprints and the intermediate vectors
  stay on the workstation; the repository only has the aggregates.
  A is the person named on all three sessions; B came out as `S1` on R2
  and R3.
- **There is no ground truth.** Nobody has marked who speaks when or
  where both speak at once. No accuracy is claimed. What is measured:
  similarity distributions, agreement between methods, how much each
  model marks as overlap, speed and memory. Section 7 says what the
  owner would have to label to turn the open questions into
  measurements.
- **"Reference"** below is a third method, not a truth: the
  pipeline's own two-way clustering of the 5 s windows (centred, count
  given), each cluster named by the one-to-one assignment of its mean
  to the two voiceprints. The two clusters' means are far apart at
  every recording (§2.2), so naming them is unambiguous; *which
  windows* go in each cluster is the clustering's opinion.
- Windows are rebuilt from each session's stored words exactly as the
  pipeline builds them (speech regions, 5 s / 2.5 s and 1.5 s / 0.75 s
  windows) and embedded with the pipeline's WavLM adapter. Run with the
  voiceprints and the default parameters, the pipeline's seeding on
  these vectors gives what the stored transcripts show: both named on
  R1, A and `S1` on R2 and R3.

## 2. Why B is anonymous on R2 and R3

### 2.1 Window against voiceprint (`window_similarity`)

Cosine of each 5 s window with each voiceprint (raw unit vectors, as
the pipeline compares), windows grouped by the reference. Quantiles
p5 / p25 / p50 / p75 / p95.

| Recording | Person (windows) | To own voiceprint | To the other's | Own ≥ 0.95 | Own ≥ 0.90 | Own ≥ 0.85 | AUC |
|---|---|---|---|---|---|---|---|
| R1 | A (73) | 0.837 / 0.942 / 0.962 / 0.973 / 0.980 | p50 0.567, p95 0.654 | 65.8% | 93.2% | 93.2% | 0.985 |
| R1 | B (51) | 0.626 / 0.769 / 0.865 / 0.909 / 0.940 | p50 0.663, p95 0.884 | 3.9% | 31.4% | 54.9% | 0.970 |
| R2 | A (29) | 0.620 / 0.765 / 0.875 / 0.912 / 0.940 | p50 0.552, p95 0.779 | 3.4% | 37.9% | 69.0% | 0.945 |
| R2 | B (45) | 0.528 / 0.765 / 0.844 / 0.890 / 0.917 | p50 0.521, p95 0.717 | 0.0% | 20.0% | 44.4% | 0.893 |
| R3 | A (54) | 0.422 / 0.756 / 0.904 / 0.943 / 0.962 | p50 0.536, p95 0.689 | 18.5% | 51.9% | 63.0% | 0.882 |
| R3 | B (32) | 0.661 / 0.766 / 0.826 / 0.870 / 0.906 | p50 0.529, p95 0.755 | 0.0% | 9.4% | 46.9% | 0.978 |

AUC: chance that a window of the person is closer to that person's
voiceprint than a window of the other person is.

The seeding claims a person when **at least one** 5 s window reaches
the threshold (0.95). B never has such a window on R2 and R3, and 2 of
51 on R1: on R1 B was named on the strength of two windows. A is named
everywhere because A's windows reach 0.95 far more often (66% on R1).
With the count given (2), the single-voice rule plays no part.

### 2.2 Voices against voiceprints and across sessions

Mean of each reference voice against the voiceprints
(`window_similarity`, `group_to`):

| Recording | A's voice: to A / to B | B's voice: to A / to B |
|---|---|---|
| R1 | **0.991** / 0.595 | 0.730 / **0.922** |
| R2 | **0.942** / 0.638 | 0.599 / **0.917** |
| R3 | **0.935** / 0.611 | 0.594 / **0.899** |

At the level of a whole voice the two people are well apart: the
lowest own similarity is 0.899 and the highest cross similarity
0.730, a gap of 0.17. The two voiceprints have a cosine of 0.566 with
each other (`voiceprints`).

The same voices compared across sessions
(`centroids_across_recordings`):

| | Same person, other session | Other person |
|---|---|---|
| A | 0.950 – 0.975 | 0.638 – 0.819 |
| B | 0.958 – 0.993 | |

B's voice is very stable from session to session (0.958–0.993), and
further from B's own voiceprint (0.899–0.922) than from B's voice in
another session. For A the voiceprint is as close as another session
is (0.935–0.991). **B's voiceprint is the outlier**: it was made under
conditions that differ from how B sounds at the table.

### 2.3 Enrolment conditions

The stored voiceprints hold only the voice model, the vector, the
consent version and the enrolment time: no device, room or distance,
and the enrolment audio was deleted (as `ARCHITECTURE.md` §6
specifies). Both were enrolled 3–4 minutes before R1 started, from
the read text. What can be said from the vectors is §2.2: B's
enrolment sounds unlike B in the conversations; which difference
(another phone, a different distance, reading against talking) cannot
be told from the data. **The owner knows which device and position B
enrolled with; that is the first thing to check.**

What a better-matched voiceprint does, measured by using one
session's reference voices as the voiceprints of another session
(`session_voiceprints`; same phone, conversational speech, never the
recording being scored):

| Scored | Voiceprints from | B to own, p50 | B windows ≥ 0.95 | Seeding at 0.95 names |
|---|---|---|---|---|
| R1 | R2 / R3 | 0.895 / 0.883 | 7.8% / 7.8% | both / both |
| R2 | R1 / R3 | 0.873 / 0.885 | 15.6% / 4.4% | both / **B only** |
| R3 | R1 / R2 | 0.882 / 0.913 | 12.5% / 12.5% | both / both |

A voiceprint taken in the conditions of the conversation raises B's
typical window by 0.02–0.09 and names both people in 5 of 6 cases at
the current threshold. It does not make 0.95 safe: even then only
0–22% of a person's windows reach it, and in the sixth case A was
missed instead.

### 2.4 Language

`language` rows: Whisper's detected language per 5 s window. R1:
122 of 124 windows Spanish. **R2: 28 Spanish, 39 English, 7 other.
R3: 6 Spanish, 80 English.** The sessions were recorded with language
`es`, so Whisper was forced to Spanish on mostly English speech (what
that does to the text was not measured here).

| Person | Spanish windows, own p50 | English windows, own p50 |
|---|---|---|
| A | R1 0.962 (73), R2 0.875 (15) | R2 0.884 (12), R3 0.907 (51) |
| B | R1 0.865 (49), R2 0.860 (13) | R2 0.833 (27), R3 0.819 (29) |

B's English windows sit 0.03–0.05 below B's Spanish ones; A shows no
such difference. Recording and language are confounded (R3 is almost
all English), and the windows are few. Even B's Spanish windows have a
median of 0.86, so **language is at most a small part of the gap to
0.95**; the voiceprint (§2.2) is the larger part.

### 2.5 Rules that would name both (`seeded`, `expected_rule`)

**Lower threshold, same seeding.** Sweep from 0.70 to 0.99:

| Threshold | R1 | R2 | R3 |
|---|---|---|---|
| 0.70 – 0.91 | both | both | both |
| 0.92 – 0.94 | both | both | A only |
| 0.95 (default) | both | A only | A only |
| 0.96 | both | nobody | A only |
| 0.97 | A only | nobody | A only |

From 0.70 to 0.91 both are named on every session and nobody is named
as the other at the level of a person. At window level the seeding and
the reference disagree on 2.4–9.5% of the windows in that range (the
`other` field); without labels it is not known which of the two is
right on those windows.

But a lower global threshold is unsafe for the case it was set for:
on read speech, a *different* reader's voice reached 0.934 / 0.954
(p95 / p99, tuning readers) and 0.958 / 0.970 (held-out readers)
against a voiceprint (`IDENTIFICATION.md` §3.1). At 0.91 an unenrolled
person, or an enrolled person who is absent from the list, would often
be named as someone else.

**Expected-speakers rule.** When the session gives the list of people
present and the count equals its length: cluster the windows into
that many voices, assign the voices to the listed voiceprints one to
one (maximum total cosine), and accept a pair above a low floor. The
whole-voice similarities of §2.2 decide; at floors from 0 to 0.85 it
names both on all three sessions; at 0.90 it loses B on R3 (0.899).
Its windows agree with the reference by construction (it *is* the
reference clustering plus a floor), so the evidence for it is only the
0.17 gap between own and cross similarity of §2.2, and the fact that
it names both with any floor from 0 to 0.85.

### 2.6 Recommendation for (a)

1. **Adopt the expected-speakers rule** when a session lists exactly
   who is present (count = length of the list): cluster, assign one to
   one, floor about **0.8**. On these sessions the lowest true pair is
   0.899 and the highest cross pair 0.730, so 0.8 sits in the middle
   of the gap (judgement on three sessions of two people; see §7 for
   what would test it). The floor is not a defence against a wrong
   list: on read speech a different person reaches 0.95, so the list
   is what guards the names. Keep the similarity in
   `speaker_confidence`, as now.
2. **Keep 0.95** for open sessions (no list, or a count that differs
   from the list): that is where the impostor risk measured in TR-05
   applies, and nothing here measures it.
3. **Re-enrol B** with the phone where it lies during the
   conversations, speaking as in a conversation, and check the device
   used for the first enrolment. Expected effect, from §2.3: B's
   typical window rises by 0.02–0.09. Re-enrolment alone does not
   rescue the 0.95 seeding (§2.3), so it supports point 1 rather than
   replacing it.
4. Record mixed-language sessions with language detection, not a
   forced `es` (§2.4); the effect on the text is outside this study.

## 3. Candidates for simultaneous speech

Licence and gating were read from the Hugging Face API (`gated`,
`license:` tag, file sizes) and the model cards, on 2026-11-14.

| Candidate | Licence | Gated | Size | Ran here | GPU peak | Speed |
|---|---|---|---|---|---|---|
| `nvidia/diar_streaming_sortformer_4spk-v2` (NeMo; 4 speakers, frame posteriors every 80 ms) | CC BY 4.0 | no | 471 MB `.nemo` | **yes** | 0.98 GB | 427× real time |
| `microsoft/wavlm-base-plus-sd` (transformers; 2 speakers, 20 ms frames, fine-tuned on LibriMix) | card points to the UniSpeech licence: CC BY-SA 3.0 | no | 378 MB | **yes** | 0.47 GB | 725× real time |
| `nvidia/Nemotron-3-Diarization` (NeMo; up to 8 speakers) | OpenMDW 1.1, card: commercial use allowed | no | 199 MB `.nemo` | **no**: needs NeMo's development branch (the release 3.0.0 lacks its `rope` attention), whose lhotse requires Python ≥ 3.11; the project is on 3.10 | – | – |
| `pyannote/segmentation-3.0` (powerset segmentation, overlap native) | MIT | **yes** | 5.9 MB | no: needs the owner to accept the conditions and provide a token | – | – |
| `pyannote/speaker-diarization-3.1` | MIT | **yes** | (uses the above) | no: same | – | – |
| `pyannote/speaker-diarization-community-1` | CC BY 4.0 | **yes** | 5.9 + 26.6 MB | no: same | – | – |
| `pyannote/overlapped-speech-detection` | MIT | **yes** | – | no: same | – | – |
| `onnx-community/pyannote-segmentation-3.0` | MIT | no | – | no: a third-party copy of the gated model; using it would sidestep the gate the owner is meant to accept | – | – |
| `BUT-FIT/diarizen-wavlm-large-s80-md` (DiariZen, pyannote fork) | CC BY-NC 4.0 | no | 278 MB | no: non-commercial | – | – |
| `nvidia/diar_sortformer_4spk-v1` | CC BY-NC 4.0 | no | 493 MB | no: non-commercial, superseded by v2 | – | – |
| `Revai/reverb-diarization-v1` / `-v2` | other | **yes** | – | no | – | – |

Speed and memory (`speed` rows): the three sessions, 985 s of audio,
in 2.31 s (Sortformer) and 1.36 s (WavLM-SD) after loading, on the
RTX 3070 Ti, peak memory reserved by torch around the runs. For
comparison, the recogniser reserved 2.51–2.80 GB on synthetic meetings
(`IDENTIFICATION.md` §5, not re-measured) and the WavLM voice encoder
2.08 GB here (`recording` rows); the transcriber loads models per job and releases them,
so either candidate fits beside them on 8 GB.

NeMo was installed in a separate virtual environment layered on the
project's (the project's torch 2.14.1+cu126, numpy 2.2.6 and
transformers 5.20.2 were used, not replaced). `nemo_toolkit[asr]`
3.0.0 brought about 100 packages (Lightning, Hydra, lhotse, librosa,
wandb, ...; 592 MB) and pins older `packaging` (24.2 against the
project's 26.3) and `fsspec` (2025.12.0 against 2026.9.0): adding it
to the project's single environment would need those conflicts
resolved first.

## 4. How much overlap, and do the candidates agree

### 4.1 Overlap marked (`overlap`)

A frame is a voice when its posterior is at least 0.5 and overlap when
two voices are. "Speech regions" are the pipeline's regions (from the
words): 332.9, 193.7 and 228.9 s.

| Recording | Model | Overlap (s) | Share of the model's own speech | Share of the speech regions |
|---|---|---|---|---|
| R1 | Sortformer | 37.6 | 12.6% | 10.7% |
| R1 | WavLM-SD | 114.5 | 37.6% | 32.4% |
| R2 | Sortformer | 19.3 | 11.7% | 8.7% |
| R2 | WavLM-SD | 59.7 | 30.3% | 26.2% |
| R3 | Sortformer | 13.6 | 6.1% | 4.3% |
| R3 | WavLM-SD | 51.7 | 21.0% | 16.7% |

Sortformer used more than two of its four speakers on R1 (four with
more than 1% of the frames; two on R2 and R3).

### 4.2 Agreement between the two (`agreement`)

| Recording | Both mark (s) | IoU | Sortformer's overlap also in WavLM-SD's | WavLM-SD's also in Sortformer's |
|---|---|---|---|---|
| R1 | 33.4 | 0.281 | 88.9% | 29.2% |
| R2 | 17.4 | 0.283 | 90.2% | 29.1% |
| R3 | 10.8 | 0.197 | 79.0% | 20.8% |

What Sortformer calls overlap, WavLM-SD mostly calls overlap too;
WavLM-SD marks about three times as much. Which one is right cannot be
told without labels. Judgement: WavLM-SD was fine-tuned on LibriMix,
synthetic mixtures of read speech that overlap most of the time, so it
is likely to over-detect on a conversation; Sortformer was trained for
meetings. That is a reason to trust its smaller figure more, not a
measurement.

### 4.3 Utterances of the current transcripts on overlap (`utterances`)

Share of the stored utterances with at least 0.5 s on frames a model
marks as overlap (in brackets: with at least 20% of their time):

| Recording (utterances) | Sortformer | WavLM-SD | Both models |
|---|---|---|---|
| R1 (79) | 20.3% (21.5%) | 67.1% (64.6%) | 20.3% (20.3%) |
| R2 (94) | 13.8% (13.8%) | 40.4% (50.0%) | 11.7% (13.8%) |
| R3 (72) | 11.1% (4.2%) | 33.3% (38.9%) | 6.9% (2.8%) |

By the more conservative model, one utterance in five to one in nine
holds half a second or more of two voices.

### 4.4 Sortformer's speakers against the reference (`sortformer_vs_reference`)

Each 5 s window takes Sortformer's speaker with the most activity in
it, mapped one to one to the reference people: the two agree on 78.2%
(R1), 82.4% (R2) and 95.3% (R3) of the windows. Two independent
diarizations agree that far and no further; which one is right where
they differ is again a labelling question.

### 4.5 The cheap score from the embeddings (`cheap_score`)

Three per-window "two voices" scores from vectors the pipeline already
has, on the 1.5 s windows: the lower of the two voiceprint cosines,
one minus their difference, and the lower of the cosines to the two
reference voices (centred). AUC against windows with at least 20% of
their time marked as overlap by each model:

| Score | vs Sortformer (R1 / R2 / R3) | vs WavLM-SD (R1 / R2 / R3) |
|---|---|---|
| lower voiceprint cosine | 0.547 / 0.556 / 0.608 | 0.660 / 0.636 / 0.570 |
| voiceprint cosines close | 0.608 / 0.548 / 0.659 | 0.525 / 0.476 / 0.543 |
| lower cosine to the voices | 0.477 / 0.481 / 0.616 | 0.440 / 0.458 / 0.551 |

0.5 is chance. **The cheap score does not follow either model**
(0.44–0.66): a 1.5 s x-vector of mixed speech does not look like
"between the two voices". Not worth adopting as an overlap detector.

## 5. Recommendation for the product (c)

Measured above: amounts, agreement, speed, memory. The rest of this
section is judgement.

1. **Identification first.** The anonymous B is not an overlap
   problem; it is a voiceprint and threshold problem (§2). The
   expected-speakers rule (§2.6) is a domain change of a few dozen
   lines with fakes-based tests and no new dependency.
2. **Overlap: adopt Sortformer v2 as an overlap *flag*, behind a new
   outbound port** (for example `SpeakerActivity.activity(audio) →
   frames × speakers` posteriors), not as the diarizer yet. The
   pipeline would keep its WavLM grouping and use the frames only to
   mark overlap. Cost: NeMo in its own environment or the dependency
   conflicts of §3 resolved, 471 MB of weights, about 1 GB of GPU
   memory, under a second per session; licence CC BY 4.0
   (attribution). Replacing the WavLM diarization with Sortformer's
   speakers is possible behind the same port but not justified until
   §7's labels show which of the two is right where they disagree
   (18–22% of windows on R1 and R2).
3. **Pyannote** (MIT / CC BY 4.0, overlap-aware) is the obvious
   second candidate once the owner accepts the conditions and provides
   a token; it was not run.
4. **What the reader should see as doubtful**, per utterance:
   - *overlap*: at least 0.5 s on frames Sortformer marks as two voices
     (20%, 14% and 11% of the utterances here); the text and the
     speaker of such an utterance may mix both people;
   - *weak identification*: `speaker_confidence` below about 0.9 for a
     named person, or a name given by the expected-speakers rule rather
     than by the threshold (both people here would show 0.90–0.99);
   - *language*: windows detected in a language other than the
     session's (R2 and R3 here), until mixed-language recording is
     handled.

## 6. Reproducing

The data stay where the transcriber keeps them and are only read; the
cache holds private vectors and goes in a directory outside the
repository, deleted afterwards.

```sh
C=/some/private/tmp
PYTHONPATH=. python experiments/transcriber/overlap_study.py embed --cache $C
PYTHONPATH=. python experiments/transcriber/overlap_study.py wavlm-sd --cache $C
PYTHONPATH=. python experiments/transcriber/overlap_study.py language --cache $C
# NeMo, in a separate environment that reuses the project's packages:
uv venv --seed --python .venv/bin/python /tmp/nemo-venv
.venv/bin/python -c "import site; print(site.getsitepackages()[0])" \
  > "$(/tmp/nemo-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/project.pth"
/tmp/nemo-venv/bin/python -m pip install "nemo_toolkit[asr]==3.0.0"
PYTHONPATH=. /tmp/nemo-venv/bin/python \
  experiments/transcriber/overlap_study.py sortformer --cache $C
PYTHONPATH=. python experiments/transcriber/overlap_study.py analyse --cache $C
rm -r $C /tmp/nemo-venv
```

A second `analyse` run wrote a byte-identical `overlap.jsonl`.

## 7. What the owner would have to label

- **Overlap (decides between the models).** 90 clips of 2 s drawn
  from R1–R3, 30 from each of three strata: both models mark overlap;
  only WavLM-SD does; neither does. For each clip one answer: do two
  people speak at the same time (yes / no). About 3 minutes of audio
  and 15 minutes of listening; format `recording,start,end,both` (CSV,
  seconds). It gives each model's precision and how much real overlap
  Sortformer misses where WavLM-SD alone fires. The script can write
  the clip list (times only) on request.
- **Who speaks (decides between the diarizations).** The 5 s windows
  where Sortformer and the reference disagree: 27 (R1), 13 (R2) and 4
  (R3); for each, which person speaks most (A / B / both). About 4
  minutes of audio.
- **The expected-speakers floor.** These sessions cannot test the
  case it must survive: a listed person absent, or an unlisted person
  present. One more session with a third, unenrolled person in the
  room (listed or not, as an error would be) would show what a floor
  of 0.8 does then.

## 8. Limits

- Three sessions, two people, one phone, one afternoon. Every share
  above can move by several points with one more session.
- The reference is a clustering, not a truth (§1); the
  expected-speakers rule shares it.
- Language detection is Whisper's argmax on 5 s windows, unchecked.
- Speeds exclude decoding and model loading (`load_seconds` of the
  `speed` rows: 16.0 s for WavLM-SD, its download included; 0.9 s for
  Sortformer, already in the local cache). The Sortformer timings
  follow one warm-up call.

## 9. What TR-17 (#25) built on this study

- **Expected-speakers rule** (§2.6 point 1): `TranscriptionParameters.
  expected_floor` (0.8, the judgement above; not a measured error
  rate). A person it names below 0.95 is marked
  `identification: "expected_floor"` in the transcript (optional field
  of `transcript.v1`); the player shows it as a less certain
  identification and recommends re-enrolling. Open sessions keep the
  seeding at 0.95.
- **Overlap detector port** (§5 point 2): `OverlapDetector.overlaps
  (audio) → [(start, end)]`, used only to set `overlap` (at least
  `overlap_seconds`, 0.5 s, on a detected stretch) with
  `overlap_source` telling which source fired. Settings:
  `overlap_detector: "off"` (default) or `"fake"` (a deterministic
  stand-in for tests). **No real adapter ships.** Sortformer v2 could
  not be loaded with what the project's environment has: its `.nemo`
  checkpoint is a NeMo model (FastConformer encoder, Transformer
  encoder, streaming speaker cache) with no transformers port, and a
  pure-Python reimplementation was not attempted. To ship it, one of:
  1. a second virtual environment with `nemo_toolkit[asr]==3.0.0`
     layered on the project's (as in §6), and an adapter that runs a
     small worker script in it as a subprocess (audio in, overlap
     stretches out as JSON), so the project's own environment keeps
     its pins; or
  2. adding NeMo to the project, which first needs the conflicts of
     §3 resolved: NeMo 3.0.0 pins `packaging` 24.2 (project 26.3) and
     `fsspec` 2025.12.0 (project 2026.9.0), and brings about 100
     packages (592 MB).
  Either way the torch, torchaudio, transformers and numpy of the
  project must stay as they are; the licence (CC BY 4.0) needs the
  attribution in the docs.
- **Language** (§2.4, §2.6 point 4): a session's language may be
  `auto` (Whisper detects each window; the web app offers
  "Automático / mezcla"); every sentence carries the language detected
  in its scoring window, and with a fixed language a sentence detected
  in another gets `language_differs: true`. The detection reuses the
  encoder output of the scoring pass (one decoder step per scoring
  window, no second pass over the audio), so it is on by default; its
  cost, measured below, was within the run-to-run noise.
- **Enrolment advice** (§2.6 point 3) in the tutorial and on the enrol
  page.
- Not done: the owner labelling of §7, which is still what any
  accuracy claim needs.

### 9.1 R1–R3 rerun with TR-17

The three sessions copied to a scratch directory and transcribed again
with the TR-17 pipeline (both people listed, count 2, so the
expected-speakers rule applies; no overlap detector), on the RTX 3070
Ti; aggregates printed by `experiments/transcriber/real_sessions.py`
(people A/B as in §1; no audio, text, ids or names):

```sh
PYTHONPATH=. python experiments/transcriber/real_sessions.py \
    --data-dir data/transcriber --work /tmp/tr17 --first 3
# the same with --no-detect-languages (timing) and --language auto
rm -r /tmp/tr17
```

Recorded language (`es`, as the sessions were started):

| | R1 | R2 | R3 |
|---|---|---|---|
| A named by | threshold | floor | floor |
| B named by | floor | floor | floor |
| Anonymous voices | 0 | 0 | 0 |
| Sentences | 58 | 82 | 53 |
| Doubtful | 13 (22.4%) | 11 (13.4%) | 3 (5.7%) |
| Overlap (voice windows) | 11 (19.0%) | 4 (4.9%) | 3 (5.7%) |
| Sentences out of time order | 0 | 0 | 0 |
| Seams (steps back over 1 s) | 2 | 5 | 4 |
| Words returned / dropped by the seam step | 837 / 24 | 441 / 52 | 397 / 35 |
| … of the dropped, with a kept copy | 16 | 44 | 30 |
| Sentences detected in another language | 0 | 49 | 52 |
| Scoring seconds, with / without language detection | 3.1 / 1.8 | 1.3 / 1.3 | 1.5 / 1.5 |
| Recognition seconds | 12.7 | 6.4 | 7.0 |

Before TR-17 B was anonymous on R2 and R3 (§2.1); now both people are
named on all three, B always through the floor, A through the floor on
R2 and R3 (A's whole voice there is 0.942 and 0.935 from A's
voiceprint, §2.2). Whether every sentence went to the right person is
not known: there are no labels (§7). The dropped words without a kept
copy (8, 8 and 5) are words of one window replaced by different words
of the other at the same moment; which window heard right is not
known either. The R1 scoring time with detection is the first scoring
call of the process; on R2 and R3 detection cost nothing measurable.

With `--language auto`: R1 58 sentences, all detected as Spanish; R2
79 sentences, 47 English and 32 Spanish; R3 55 sentences, all English.
Doubtful 13 / 9 / 6, overlaps 11 / 8 / 7, out of order 0 / 0 / 0; seams
2 / 7 / 1 with 24 / 77 / 11 words dropped (16 / 58 / 11 with a kept
copy). Names as above. What forcing Spanish did to the English text
was not measured (no reference transcript).
