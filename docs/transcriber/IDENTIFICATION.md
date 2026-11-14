# Putting a user id on a voice: what was measured

Status: TR-05, 2026-11-14. Synthetic recordings of read speech only.
Every figure below is a row (or an aggregate of rows) written by
`experiments/transcriber/identification_eval.py` to
`experiments/transcriber/*.jsonl`; the file and the selector are given
with each table. Nothing here has been measured on a phone on a table.

## 1. Method

**Source.** Multilingual LibriSpeech, Spanish (`facebook/
multilingual_librispeech`, public audiobook readings with transcripts
and reader ids), as in `docs/skills/AUDIO.md` §2. Two splits with
disjoint readers:

- `dev` (20 readers): everything chosen in this document (rules,
  thresholds, normalisation) was chosen on it;
- `test` (20 other readers): held out, only reported. Where the two
  disagree, the `test` figure is the one to believe.

**Recordings** (per split, no utterance used twice, seed 0; plan in
the cache, `recognition.jsonl` lists every recording):

| Kind | How it is built | Count |
|---|---|---|
| enrolment sample | other utterances of one reader, 0.3 s apart, cut at 30.0 s | 20 (one per reader) |
| two-reader sample | 30 s where a second reader takes the last 50%, 33% or 20% | 18 (6 of each) |
| one-voice recording | 6 utterances of one reader, 0.2–1.0 s of silence between | 20 |
| two-voice recording | 8 alternating turns of two readers | 20 |
| meeting | 16 turns of 4 readers, never the same twice in a row | 20 (about 4 min each) |

**Pipeline**, through the transcriber's own code: Whisper
(`WhisperRecogniser`, `openai/whisper-large-v3-turbo`, batch 2,
float16 on the RTX 3070 Ti) gives timed words; the speech-activity gate
(`heard_words`), speech regions, 5 s / 1.5 s windows and WavLM
x-vectors (`WavLMVoiceEncoder`, `microsoft/wavlm-base-plus-sv`) follow
`TranscribeSession`; enrolment samples follow `EnrolSpeaker` (regions
from all recognised words, 5 s windows, `enrolment_vector`). Words and
vectors are cached, so the analysis reruns without the GPU.

**Truth.** A word belongs to the reader whose utterance holds its
middle (nearest utterance otherwise); a window belongs to a reader when
at least 90% of its overlap with speech is theirs (mixed windows are
left out of §2).

## 2. Raw embeddings: do they separate readers?

Cosine between windows of the same meeting, pairs of the same reader
against pairs of different readers (`windows.jsonl`). AUC: chance that
a same-reader pair is more similar than a different-reader pair; EER:
equal error rate of a single threshold.

| Split | Windows | Vectors | Same reader p5 / p50 | Different readers p50 / p95 | AUC | EER (at cosine) |
|---|---|---|---|---|---|---|
| dev | 5 s | raw | 0.926 / 0.968 | 0.689 / 0.914 | 0.991 | 3.8% (0.92) |
| dev | 5 s | centred on the recording | 0.557 / 0.849 | -0.427 / 0.535 | 0.992 | 4.6% (0.55) |
| dev | 5 s | centred on a mean of other readers | 0.749 / 0.896 | -0.115 / 0.719 | 0.992 | 4.1% (0.73) |
| dev | 1.5 s | raw | 0.737 / 0.903 | 0.643 / 0.886 | 0.914 | 16.0% (0.83) |
| dev | 1.5 s | centred on the recording | 0.097 / 0.642 | -0.250 / 0.518 | 0.942 | 13.8% (0.31) |
| test | 5 s | raw | 0.918 / 0.965 | 0.672 / 0.931 | 0.984 | 6.7% (0.92) |
| test | 5 s | centred on the recording | 0.426 / 0.819 | -0.384 / 0.563 | 0.980 | 7.3% (0.48) |
| test | 5 s | centred on a mean of other readers | 0.699 / 0.888 | -0.186 / 0.805 | 0.972 | 9.4% (0.76) |
| test | 1.5 s | raw | 0.714 / 0.896 | 0.619 / 0.896 | 0.892 | 18.9% (0.83) |
| test | 1.5 s | centred on the recording | 0.006 / 0.593 | -0.222 / 0.524 | 0.921 | 16.0% (0.27) |

Read: raw vectors *do* separate readers, but everything happens
between 0.9 and 1.0. Different readers have a median raw cosine of
about 0.68 and one pair in twenty is above 0.91–0.93. Centring moves
the scale, not the overlap: on 5 s windows the AUC is the same within a
point. Short windows are much noisier, as found under `src`.

### The single-voice test

`is_single_voice` decides, when the number of people is unknown,
whether a recording is one voice (and must not be clustered after
centring, which would split it). The old rule: the *lowest* cosine of
a window with the mean direction is at least 0.9. Two voices whose raw
cosine is `c` put their windows at about `sqrt((1 + c) / 2)` from the
mean; with the median `c` of 0.68 that is 0.92, above 0.9: the rule
called many two-voice recordings one voice, as its author feared.

Four statistics on raw 5 s windows were compared on 40 one-voice
recordings (20 one-voice recordings and the 20 enrolment samples), 20
two-voice and 20 four-reader recordings per split
(`single_voice.jsonl`, rows `single_voice_rule`; errors are recordings
called the wrong way):

| Rule (one voice when …) | dev: 1-voice split / 2-voice merged / 4-voice merged | test: same |
|---|---|---|
| lowest cosine to the mean ≥ 0.90 (old default) | 3/40 · 5/20 · 0/20 | 2/40 · 10/20 · 1/20 |
| lowest cosine to the mean ≥ 0.93 (among its best on dev) | 5/40 · 1/20 · 0/20 | 4/40 · 1/20 · 0/20 |
| mean pairwise cosine ≥ 0.935 | 3/40 · 1/20 · 0/20 | 2/40 · 1/20 · 0/20 |
| **10% quantile of the cosines to the mean ≥ 0.95 (new default)** | **2/40 · 1/20 · 0/20** | **0/40 · 3/20 · 0/20** |
| cosine between the two halves of a 2-way split ≥ 0.92 | 3/40 · 1/20 · 0/20 | 2/40 · 5/20 · 0/20 |

No statistic of raw vectors separates the cases cleanly; the overlap of
§2 is the limit. The lowest cosine fails on one-voice recordings
because a single stray window (a breath, a noisy stretch) is enough; a
quantile ignores it. The 10% quantile at 0.95 had the fewest errors on
`dev` (3 of 80) and made 3 of 80 on `test`, all of them two readers
merged. It is now the default (`single_voice_threshold` 0.95,
`single_voice_quantile` 0.1, new parameter of `is_single_voice` and
`group_voices`).

What it does to anonymous diarization, words right under the best
one-to-one mapping of labels to readers (`diarization.jsonl`,
clustering threshold 0.5):

| Recordings | Speakers | Old rule (dev / test) | New rule (dev / test) |
|---|---|---|---|
| one voice | estimated | 93.7% / 96.9% | 96.6% / 100.0% |
| two voices | estimated | 84.0% / 71.0% | 87.2% / 71.2% |
| four readers | estimated | 89.9% / 83.1% | 89.9% / 84.0% |
| two voices | given | 98.6% / 96.2% | same |
| four readers | given | 87.5% / 87.8% | same |

With the number of speakers unknown, two-voice recordings stay poor
with either rule: the clustering after centring finds from 1 to 10
voices (`voices_found` in the rows). Lowering the clustering threshold
helps two-voice recordings (0.2: 93.7% dev, 84.9% test) and hurts
meetings (0.2: 79.7% dev, 78.3% test), so no single value is right
for both; `group_threshold` was left at 0.5. **Give the number of
people when it is known.**

## 3. Identification

Each split's 20 enrolment samples go through the real
`enrolment_vector` with the new sample rules (§4); all 40 were
accepted. For each meeting four enrolment conditions are drawn once
(`identification` rows, `condition`):

- `all_enrolled`: the 4 readers have a voiceprint;
- `one_unenrolled` / `two_unenrolled`: 1 or 2 of them do not;
- `enrolled_absent`: the 4 readers and one reader who is not in the
  meeting.

Only the condition's voiceprints are candidates (as with
`expected_speakers`). Measures, pooled over the 20 meetings of a split:

- **right**: share of words whose final speaker is right: the reader's
  id when enrolled; for an unenrolled reader, the anonymous label that
  best maps to them (one-to-one);
- **impostor**: share of the unenrolled readers' words that got an
  enrolled id (in brackets: unenrolled readers whose words mostly got
  one / unenrolled readers);
- **missed**: share of the enrolled readers' words left anonymous
  (readers mostly anonymous / enrolled readers present);
- **wrong**: share of the enrolled readers' words given another
  enrolled id; **absent**: words given the id of the enrolled reader
  who was not there.

Methods:

- **cluster then match** (the current pipeline): `group_voices`, then
  `match_voices` on the group vectors;
- **voiceprints as seeds**: each 5 s window takes the enrolled person
  it is most similar to when the cosine reaches the threshold; the
  unclaimed windows are clustered as in the pipeline (into the people
  still missing when the count is given); the 1.5 s windows then go to
  the nearest group. Implemented in the script only.

Normalisations (both sides go through it): `unit` (raw unit vectors,
the default), `centre_on_session` (mean of the session groups; for
seeds, of the windows), `recording_mean` (mean of the recording's 5 s
windows), `global_mean_other_split` (fixed mean of the other split's
windows), `pool_mean_other_readers` (mean of the voiceprints of the
split's readers not in the meeting), `centre_on_enrolled` (mean of the
candidate voiceprints, raw below 3). The threshold was swept from -0.2
to 0.85 in steps of 0.05 and from 0.86 to 0.99 in steps of 0.01.

### 3.1 Voiceprint against the true voice of a reader

Before any diarization error: cosine between a reader's true group in
a meeting (mean of their pure 5 s windows) and every voiceprint
(`group_to_voiceprint_cosines` rows).

| Split | Normalisation | Same reader p1 / p50 | Others p95 / p99 | AUC | EER (at cosine) |
|---|---|---|---|---|---|
| dev | unit | 0.978 / 0.994 | 0.934 / 0.954 | 1.000 | 0.03% (0.969) |
| dev | global mean, other split | 0.893 / 0.974 | 0.745 / 0.835 | 1.000 | 0.26% (0.858) |
| dev | pool mean, other readers | 0.902 / 0.977 | 0.763 / 0.849 | 1.000 | 1.05% (0.861) |
| dev | centre on session | 0.822 / 0.968 | 0.730 / 0.850 | 0.999 | 1.41% (0.825) |
| test | unit | 0.953 / 0.993 | 0.958 / 0.970 | 0.997 | 3.75% (0.962) |
| test | global mean, other split | 0.832 / 0.976 | 0.868 / 0.911 | 0.997 | 3.75% (0.882) |
| test | pool mean, other readers | 0.850 / 0.978 | 0.850 / 0.892 | 0.998 | 2.50% (0.877) |
| test | centre on session | 0.667 / 0.963 | 0.816 / 0.892 | 0.991 | 6.02% (0.804) |

Thirty seconds of enrolment match the same reader's voice in a meeting
at cosine 0.99 (median), but another reader can reach 0.95–0.97 in raw
vectors. The old default threshold of 0.5 is far below every
different-reader cosine: any group would take the nearest name.
Centring on a fixed mean spreads the scale; it does not separate the
readers better on `test`, and centring on the session is worse.

### 3.2 Words on the right id

Best threshold on `dev` per method, normalisation and count (mean of
the four conditions), and the `test` figure at that same threshold
(`identification` rows):

| Method | Normalisation | Speakers | Threshold (dev) | dev | test |
|---|---|---|---|---|---|
| voiceprints as seeds | unit | given | 0.95 | 94.0% | 91.4% |
| voiceprints as seeds | pool mean, other readers | given | 0.80 | 93.7% | 91.5% |
| voiceprints as seeds | global mean, other split | given | 0.80 | 94.1% | 90.5% |
| voiceprints as seeds | centre on session | given | 0.70 | 94.1% | 89.9% |
| voiceprints as seeds | unit | estimated | 0.92 | 93.4% | 88.2% |
| voiceprints as seeds | centre on session | estimated | 0.50 | 93.8% | 85.5% |
| cluster then match | unit | given | 0.95 | 85.9% | 86.1% |
| cluster then match | global mean, other split | given | 0.75 | 86.2% | 85.8% |
| cluster then match | pool mean, other readers | given | 0.80 | 86.4% | 85.7% |
| cluster then match | centre on session | given | 0.60 | 85.9% | 85.9% |
| cluster then match | centre on enrolled | given | 0.70 | 85.5% | 85.0% |
| cluster then match | unit | estimated | 0.94 | 89.1% | 82.6% |
| cluster then match | centre on session | estimated | 0.55 | 89.2% | 82.4% |
| cluster then match | global mean, other split | estimated | 0.70 | 89.1% | 82.2% |

The normalisation hardly matters once each has its own threshold; raw
unit vectors are as good as any on the held-out readers and need no
stored mean. The method matters: seeding with the voiceprints gains 5
to 6 points on `test` over cluster then match (91.4% against 86.1%
given, 88.2% against 82.6% estimated), because the diarization, not the match, is what loses words.

### 3.3 Per condition

Cluster then match, `unit`, at the old threshold 0.5 and the new 0.95;
voiceprints as seeds, `unit`, at its `dev` best (`test` rows; the `dev`
rows are in the same file):

| Setting (test) | all enrolled | one unenrolled | two unenrolled | enrolled reader absent |
|---|---|---|---|---|
| cluster/match, given, 0.5 | right 85.0% · wrong 15.0% | 84.5% · impostor 40.8% (8/20) · missed 1.5% | 83.3% · impostor 19.1% (8/40) · missed 2.9% | 85.3% · absent 15 words |
| cluster/match, given, 0.95 | 86.1% · missed 1.2% (1/80) · wrong 12.7% | 86.0% · impostor 40.8% (8/20) · missed 1.7% | 86.5% · impostor 11.7% (5/40) · missed 5.7% (2/40) | 86.1% · absent 0 |
| cluster/match, estimated, 0.5 | 82.9% · missed 7.3% (3/80) | 80.7% · impostor 33.6% (8/20) | 81.4% · impostor 16.2% (6/40) · missed 12.4% (3/40) | 82.0% · absent 309 words |
| cluster/match, estimated, 0.95 | 82.2% · missed 8.2% (4/80) | 82.3% · impostor 28.1% (6/20) | 81.6% · impostor 13.5% (5/40) · missed 14.5% (5/40) | 82.2% · absent 115 words |
| seeds, given, 0.95 | 93.9% · wrong 6.1% | 91.2% · impostor 17.5% (2/20) · missed 3.3% | 87.1% · impostor 7.7% (2/40) · missed 8.1% (1/40) | 93.6% · absent 35 words |
| seeds, estimated, 0.92 | 90.3% · missed 4.5% (1/80) | 86.3% · impostor 30.2% (8/20) | 86.2% · impostor 12.4% (5/40) · missed 8.1% (1/40) | 90.0% · absent 75 words |

What the impostor figures mean: with cluster then match, an
unenrolled reader whose words land in an enrolled reader's group
(because the diarization merged them) is named whatever the threshold;
the 40.8% at 0.5 and at 0.95 with one unenrolled reader is that. A
higher threshold only removes impostors whose *own* group was named.
With the count given, the impostor share pooled over the conditions
with unenrolled readers is 28.1% / 17.4% at 0.5 / 0.95 (dev) and
26.2% / 21.1% (test); the missed share pooled over all four conditions
is 0.5% / 2.0% (dev) and 0.8% / 2.1% (test) (sums of the rows' word
counts).

## 4. Enrolment sample rules

The 30 s genuine samples had 15.3–30.0 s of recognised speech and 5–11
voiced windows (`enrolment_sample` rows); that one sample sat at the
edge of the untuned `min_speech_ratio` (0.5) and `min_windows` (5),
which were left as they were. Outcomes per rule pair
(`sample_rules` rows; two-reader samples accepted out of 6 per share of
the second reader):

| `same_voice_threshold` / `max_other_voice_share` | Split | Genuine accepted | Two readers accepted (50% · 33% · 20%) |
|---|---|---|---|
| 0.5 / 0.2 (old) | dev | 20/20 | 6 · 5 · 6 (17/18) |
| 0.5 / 0.2 (old) | test | 20/20 | 4 · 5 · 6 (15/18) |
| **0.85 / 0.1 (new)** | dev | 20/20 | 0 · 2 · 1 (3/18) |
| **0.85 / 0.1 (new)** | test | 20/20 | 2 · 2 · 3 (7/18) |
| 0.9 / 0.1 | dev | 19/20 | 0 · 1 · 1 (2/18) |
| 0.9 / 0.1 | test | 20/20 | 1 · 0 · 1 (2/18) |
| 0.85 / 0.2 | dev | 20/20 | 0 · 2 · 5 (7/18) |

With the old rules a two-person sample was nearly always accepted: the
clustering threshold of 0.5 lies below the raw cosine of most pairs of
readers (§2), and a second reader on 20% of the sample (about 2 of 11
windows) stays under a 20% share. The new pair is the strictest that
kept every genuine `dev` sample (0.9 rejected one); on held-out
readers it still let 7 of 18 through. The check is a safety net, not a
guarantee: the person running the enrolment must make sure one person
reads.

## 5. Speech-activity gate and recognition

`activity.jsonl`, 60 recordings per split (meetings, one- and
two-voice). A word is "said" when its span overlaps a reader's
utterance and "over silence" when it lies entirely in a pause.

| Split | Frame / level | Said words dropped | Words over silence: dropped / all | WER |
|---|---|---|---|---|
| dev | no gate | 0 of 21,616 | 0 / 18 | 5.61% |
| dev | 0.025 s / 0.003 (default) | 0 of 21,616 | 3 / 18 | 5.60% |
| dev | 0.025 s / 0.02 | 237 | 5 / 18 | 6.55% |
| test | no gate | 0 of 22,379 | 0 / 56 | 4.99% |
| test | 0.025 s / 0.003 (default) | 5 of 22,379 | 7 / 56 | 4.94% |
| test | 0.025 s / 0.02 | 125 | 16 / 56 | 5.41% |

On this audio Whisper put few words in the pauses (18 and 56 of about
22,000) and most of them touch a voiced frame, so the gate removes
little; at the default level it drops almost no real word. Raising the
level costs real words before it removes the rest. The defaults were
kept. **The pauses here are digital silence and the speech is clean
studio-like reading; a phone on a table records room noise in every
pause and quieter, farther voices**, so both what the gate removes and
what it costs must be measured again on a real, consented recording.

Recognition of the 20 meetings per split through the transcriber's
adapters (`activity.jsonl`, rows `recognition_meetings`; per-recording
timing in `recognition.jsonl`):

| | dev | test | earlier, under `src` (6 meetings, `AUDIO.md` §2) |
|---|---|---|---|
| WER (pooled; accents and punctuation ignored) | 5.13% | 4.55% | 5.4% (mean) |
| WER per meeting, mean (min–max) | 5.14% (1.45–14.58%) | 4.46% (1.91–8.38%) | 2.2–8.6% |
| Recognition speed | 36.4× real time | 39.3× real time | about 22× |
| GPU memory, peak reserved by torch | 2.51 GB | 2.80 GB | 3.3 GB |
| Voice embedding, all windows of 20 meetings | 13.2 s | 13.4 s | 1.5 s per meeting |

The move from `src` lost nothing measurable: the error rate is in the
same range on more meetings, with a GPU no slower and no hungrier.
Speed and memory were measured here with `torch.cuda` peak counters
around each `recognise` call; how the earlier figures were taken is
not recorded, so the differences are not a gain to claim. Words on the right
speaker with the count given: 87.5% (dev) and 87.8% (test) here against
91.5% on the six earlier meetings; different meetings, and the earlier
figure varied from 76% to 99.7% between meetings, so this does not show
a regression either.

## 6. Defaults chosen

| Setting | Old | New | Why |
|---|---|---|---|
| `match_threshold` | 0.5 | 0.95 | best of the sweep on `dev` for the current pipeline (§3.2); 0.5 is below the raw cosine of most different readers (§3.1) |
| `normalisation` | `unit` | `unit` (kept) | no normalisation beat it on `test` (§3.2) and it needs no stored mean |
| `single_voice_threshold` / new `single_voice_quantile` | 0.9 / (minimum) | 0.95 / 0.1 | fewest errors on `dev`; merged 3 of 20 held-out two-voice recordings instead of 10 (§2) |
| `SampleRules.same_voice_threshold` / `max_other_voice_share` | 0.5 / 0.2 | 0.85 / 0.1 | strictest pair keeping every genuine `dev` sample (§4) |
| `activity_frame` / `activity_threshold` | 0.025 / 0.003 | kept | drops no real word on `dev` (§5) |
| `group_threshold` | 0.5 | kept | no value suits both two-voice recordings and meetings (§2) |

**What the pipeline uses now.** Voiceprints as seeds is the default
of `TranscriptionParameters.identification` (`"seeded"`;
`"cluster_then_match"` keeps the previous method). It applies whenever
the session has at least one voiceprint to compare with (the expected
people, or everyone enrolled with the same voice model); with none, the
voices are clustered as before. The pipeline calls
`transcriber.domain.voices.seed_voices` (and `name_seeded` for the
labels), and so does this script now, instead of its own copy, with
the same `match_threshold` (0.95 by default) and `unit` vectors: the
"given" row above (91.4% on `test`) is the default's setting; the
"estimated" row (88.2%) is at 0.92, its best on `dev`, not at the
default. Unclaimed windows are clustered into anonymous voices (`S1`,
`S2`, ...) on vectors centred on the recording's mean, whatever
`centre_windows` says.

With seeding, a given number of speakers is a lower bound, not a cap:
every enrolled person whose voiceprint clears the threshold on at least
one 5 s window is named, even when that makes more people than the
count (two enrolled voices with a count of one give two named people);
the count only decides how many voices the unclaimed windows are split
into. That is the method measured above, and it is what lets the
voiceprints win over a wrong count; `seed_voices`'s docstring and
`test_a_count_is_a_lower_bound_for_the_voiceprints` pin it.

After the switch to the domain function the cache was rebuilt
(`prepare`, on the GPU) and `analyse --what identification` rerun
(18 min 37 s on the workstation): `identification.jsonl` came out
byte-identical (6,922 rows), so every figure above stands for the
pipeline's own code.

## 7. Limits

- **Read speech, one person at a time, no overlap**, pauses of digital
  silence, synthetic turn-taking (each turn is one MLS utterance of
  10–20 s; real turns are often shorter, which leaves fewer windows per
  turn).
- **Each reader has their own microphone and room** (MLS comes from
  LibriVox audiobooks recorded by volunteers). Part of what separates the voices here is the
  channel; around one phone every voice shares it, so raw cosines
  between people will be higher and the thresholds above may be too
  low. This is the main reason the numbers are optimistic.
- **Enrolment and meeting come from the same recordings** (the same
  book, microphone and day for almost every reader), which is the best
  case the tutorial asks for, not the general one.
- 20 meetings per split, 20 readers per split: one meeting more or less
  moves a percentage by several points (per-meeting minimum in the
  rows' `right_share_worst_meeting`).
- The threshold sweeps and rules were chosen on `dev`; `test` was used
  only to report. The single-voice statistics and the enrolment rules
  were chosen on the same `dev` recordings they are reported on.

## 8. Reproducing

```sh
# downloads about 315 MB to data/tr05 (ignored by git), runs the GPU
# once; an interrupted run resumes from the cache
PYTHONPATH=. python experiments/transcriber/identification_eval.py prepare
# rewrites the analysis rows from the cache, CPU only, with the chosen
# rules as defaults (20 minutes on the workstation; a second run wrote
# byte-identical rows)
PYTHONPATH=. python experiments/transcriber/identification_eval.py analyse
rm -r data/tr05   # the audio is not kept
```

`recognition.jsonl` gets one row per recording the first time it is
recognised; delete it to time a fresh run.

## Appendix: the speaker certainty of a sentence (TR-16)

Every utterance of a transcript carries, besides `speaker_confidence`
(the cosine of the speaker's whole voice group with their voiceprint,
one value per speaker), its own `speaker_certainty`, `doubtful`,
`overlap` and `alternative_speaker`
(`transcriber/domain/certainty.py`). How they are computed:

- Each short analysis window (1.5 s every 0.75 s) has a cosine with
  each voice group's centroid, in the space the voices were grouped in
  (unit vectors centred on the recording's mean, as the grouping does
  by default). The window goes to
  the group it scores best; those cosines are kept.
- The **margin** of a speaker over a stretch of audio is the mean, over
  the short windows that overlap the stretch (each weighted by the
  seconds it shares with it), of the window's cosine with that speaker
  minus its cosine with the closest other voice heard in the
  transcript. `speaker_certainty` is the margin of the sentence's
  speaker over `[start, end]`. It lies in [-2, 2]: positive when the
  windows prefer the speaker, near 0 when they cannot tell, negative
  when they prefer someone else (possible after smoothing or after a
  sentence is kept whole).
- `doubtful` is a margin below `doubt_margin` (0.1). `overlap` is a
  window lying wholly inside the sentence that prefers another voice by
  at least 0.1. `alternative_speaker` is given only when the sentence
  is doubtful (the other voice the windows over its weakest stretch
  prefer most) or an overlap (the voice of the window that flagged it;
  with three or more voices it can differ from the runner-up over the
  whole sentence).
- A sentence that holds a stretch `keep_sentences_whole` gave to its
  speaker (below) takes the margin over that stretch when it is lower:
  a moved stretch is never hidden behind a confident sentence. With no
  window over the moved stretch the sentence is doubtful.
- With one voice in the session, or no window over a sentence, the
  fields are absent (`null` in the model).

What it is not: **a similarity margin, not a probability**. It was not
calibrated against sentences whose speaker is known on real speech, so
a margin of 0.5 does not mean "50% sure" and no error rate goes with
the 0.1 cut. `overlap` cannot tell two people speaking at once from a
quick exchange inside one sentence, and a sentence shorter than one
window (1.5 s) never has a window inside it. Separating simultaneous
voices needs a dedicated model (TR-14).

### Keeping a sentence whole (`whole_sentences`)

A change of speaker inside a sentence often comes from one stray short
window at a sentence's edge. `keep_sentences_whole` gives a stretch of
words to the speaker of a neighbouring stretch of the same sentence
when all of these hold: it lasts at most `whole_sentence_max_move`
seconds; its own speaker has fewer than two windows preferring it or a
margin below `whole_sentence_margin` (0.1); and the evidence favours
the move (at most one window prefers its own speaker, or its own
margin is at most 0, or the neighbour's margin over it is higher). A
stretch whose windows prefer its own speaker, however weakly, stays,
and the sentence is cut there (each part becomes its own turn). It has
its own switch, independent of `doubt_margin`.

Measured on the synthetic four-reader meetings of §3 (20 per split;
the default pipeline: seams resolved, words over silence dropped,
voiceprints as seeds at 0.95 on unit vectors, smoothing; the same
cached windows with the rule off and on; words on the right id, mean
of the four enrolment conditions):

| Split | Speakers | Off | 1.5 s | 3 s | 6 s | Unbounded |
|---|---|---|---|---|---|---|
| dev (tuning) | given | 94.1% | 95.5% | 95.9% | 96.2% | 96.1% |
| dev (tuning) | estimated | 89.0% | 92.5% | 93.5% | 93.7% | 91.3% |
| test (held out) | given | 91.4% | 94.0% | 94.7% | 95.0% | 94.6% |
| test (held out) | estimated | 81.8% | 85.9% | 87.2% | 88.1% | 84.1% |

With any of the four bounds the rule on put more words on the right id
than off in each of the 16 cells (two splits, four enrolment
conditions, number given or estimated; `whole_sentences.jsonl`), so it
stays on by default; 6 s was the best bound on every dev cell (tied
with unbounded on three) and is the default `whole_sentence_max_move`. Without a bound the rule hands long
stretches to the wrong neighbour when the number of speakers is
estimated. The "off" column with the number given reproduces the
91.4% of §3 for seeding. These are read speech meetings with clean
turns; on a phone on a table the rule was not measured against labels.

```sh
# needs the TR-05 cache (prepare, §8; GPU); this step is CPU only
PYTHONPATH=. python experiments/transcriber/identification_eval.py \
    analyse --what sentences
```

### On the first real recordings

Two people around one phone, three recordings R1–R3 (985 s; no ground
truth). Rerun with the current pipeline and printed by
`experiments/transcriber/real_sessions.py`, which copies the sessions
and the voiceprints to a scratch directory and prints aggregate counts
only (no audio, text, ids or names):

```sh
PYTHONPATH=. python experiments/transcriber/real_sessions.py \
    --data-dir data/transcriber --work /tmp/real-sessions --first 3
rm -r /tmp/real-sessions
```

| | R1 | R2 | R3 | All |
|---|---|---|---|---|
| Short windows | 405 | 205 | 269 | 879 |
| Windows whose two closest voices are within 0.1 | 1.7% | 3.9% | 3.4% | 2.7% (24) |
| 10% quantile of that gap | 0.374 | 0.192 | 0.185 | |
| Sentences | 58 | 82 | 59 | 199 |
| Doubtful | 13 | 10 | 4 | 27 (13.6%) |
| Overlap | 11 | 5 | 8 | 24 (12.1%) |
| Stretches moved by `keep_sentences_whole` | 17 | 6 | 3 | 26 |
| Sentences out of time order | 0 | 0 | 0 | 0 |
| Words returned / dropped by the seam step | 837 / 24 | 441 / 52 | 397 / 35 | 1675 / 111 |

Why 0.1. Few windows are near ties (2.7% within 0.1): a sentence
margin under 0.1 means its windows are, on average, as undecided as
those few. It is a parameter (`TranscriptionParameters.doubt_margin`).
With the sentences now cut where a supported change of speaker falls
and moved stretches carried into the certainty, 13.6% of the sentences
are doubtful (6.4% with the TR-16 rule, whose moves were silent).

The table above is the open-session pipeline (seeding at 0.95), as
TR-16 left it: B is anonymous on R2 and R3. With the TR-17
expected-speakers rule (these sessions list both people and give the
count 2) the same recordings give other groups and both people named;
those figures are in `OVERLAP.md` §9.1.
