# Model lab: denoisers × diarizers × recognisers on one recording

Status: TR-20 (issue #30), backend, 2026-11-14. The web tab "Pruebas"
is built on the API of §7 by a later change.

The lab runs a recorded session through every chosen combination of a
**denoiser**, a **diarizer** and a **recogniser** (a *pipeline*),
compares what the pipelines produce with one another and lets the owner
mark, per stretch of the recording, who really spoke. Until there are
labels, every number it gives is an **agreement** between models, never
an accuracy (§4).

```
 session audio ──► denoise (once per denoiser) ──► audio_d
                                                   ├─► diarize (once per d × diarizer) ─► segments ─► names (§3.3)
                                                   └─► recognise (once per d × recogniser) ─► timed words
 pipeline d+z+r = words of (d, r) attributed with the segments of (d, z)
 comparison (§4) ◄── all tracks, transcripts and pipelines
 scores (§5)     ◄── the owner's reference labels
```

## 1. Layers

| Piece | Where |
|---|---|
| Catalogue, stage results, runs, labels, cache keys | `transcriber/domain/lab.py` |
| Comparison (pure arithmetic) | `transcriber/domain/comparison.py` |
| Scoring against labels | `transcriber/domain/lab_scoring.py` |
| Public reference corpora, scoring against them (§13) | `transcriber/domain/lab_reference.py` |
| Import of a corpus, corpus summary (`ImportReference`, `ReferenceSummary`) | `transcriber/application/lab_reference.py` |
| Corpus shards from the Hugging Face Hub | `transcriber/adapters/outbound/hf_reference_corpus.py` |
| Ports `LabTool`, `LabRepository`, `CatalogueSource`, `LabEnvironment`, `ProcessTable` | `transcriber/application/lab_ports.py` |
| Use cases (`ShowCatalogue`, `StartLabRun`, `RunLab`, `LabRunStatus`, `ListLabRuns`, `CancelLabRun`, `ResumeLabRun`, `LabResultsView`, `LabLabels`, `LabScores`, `LabAudio`) | `transcriber/application/lab.py` |
| In-process tools (built-in diarization, the project's Whisper) and speaker mapping | `transcriber/application/lab_builtin.py` |
| Assembling the comparison of a run | `transcriber/application/lab_compare.py` |
| Subprocess runner, environment check, catalogue file, process table | `transcriber/adapters/outbound/subprocess_lab_tool.py` |
| File-system storage | `transcriber/adapters/outbound/fs_lab_repository.py` |
| API routes `/api/lab/...` | `transcriber/adapters/inbound/lab_api.py` |
| CLI `python -m transcriber lab ...` | `transcriber/adapters/inbound/lab_cli.py` |
| Catalogue, runner scripts, environment definitions | `lab/catalogue.json`, `lab/runners/`, `lab/envs/` |

The service never imports a runner: runners are scripts run by their
own environment's interpreter. `src/` and `transcriber/` still never
import each other. Settings: `lab` in `TranscriberSettings`
(`catalogue`, `runners_dir`, `envs_dir` default
`~/.local/share/ragputin/lab-envs`, `step_timeout` 3600 s, the
comparison tolerances of §4 and `labels_max_bytes`).

## 2. Environments

Each **family** of models (one runner script, one environment) lives
outside the repository in `~/.local/share/ragputin/lab-envs/<family>/`.
The project's own environment is never modified.

A full PyTorch environment costs about 6 GB here (torch 1.7 GB, the
NVIDIA libraries 3.6 GB, triton 0.9 GB), so the families are **layered**
on the project's environment: `uv venv --python .venv/bin/python` plus a
`.pth` file that appends the project's `site-packages`. The family's own
packages come first on `sys.path`; torch, numpy, transformers and the
CUDA libraries are the project's, read-only. uv does not see packages
reached through a `.pth`, so `lab/envs/build.py` resolves the family's
requirements with the project's versions as overrides, drops what the
project already has, and installs the rest with `--no-deps`:

```
lab/envs/<family>/requirements.in    top-level pins (edited by hand)
lab/envs/<family>/requirements.txt   resolved, minus the project's packages
lab/envs/<family>/requirements.lock  uv pip freeze of the environment
```

After installing, the builder runs the runner's `--selftest` and, when
it passes, leaves `<env>/.ragputin-lab-ready`; a family without that
marker is reported `not_installed` in the catalogue.

| Make target | Does |
|---|---|
| `make lab-env FAMILY=nemo` | create or update one environment, then self-test it |
| `make lab-envs` | every family, going on past one that fails |
| `make lab-selftest` | the self-test of every existing environment |

## 3. Runner contract

```
<env>/bin/python lab/runners/<family>.py <stage> --model <id> \
    --in <wav> --out <dir> [--params '<json object>'] [--device cuda|cpu]
<env>/bin/python lab/runners/<family>.py --selftest
```

- `<stage>`: `denoise`, `diarize` or `recognise`. `--in` is a 16 kHz
  mono 16-bit WAV; the runner resamples to its model's rate.
  `--out` is an empty directory.
- The runner always writes `<out>/result.json`:

```json
{"schema": "lab.result.v1", "ok": true, "stage": "diarize",
 "model": "sortformer-streaming-v2",
 "model_version": "nvidia/diar_streaming_sortformer_4spk-v2@<rev>",
 "device": "cuda", "seconds": 2.1, "load_seconds": 4.0,
 "peak_gpu_mb": 1003.0, "audio_seconds": 387.9, "notes": [],
 "segments": [{"start": 0.48, "end": 2.16, "speaker": "spk0"}],
 "overlap": [[12.3, 13.1]]}
```

  `denoise` adds `"audio": "audio.wav", "sample_rate": 16000` and writes
  `<out>/audio.wav` (16 kHz, mono, 16-bit, the input's length);
  `recognise` adds `"words": [{"start", "end", "text", "probability"?}]`
  and `"language"`. Segments of different speakers may overlap.
- Failure: `{"ok": false, "error": {"code": C, "message": M}}` and the
  exit code of C: `failed` 1, `unknown_model`/`bad_input` 2,
  `needs_token` 3, `out_of_memory` 4, `unsupported_language` 5,
  `not_installed` 6. The service adds `timeout` and `cancelled`. A
  runner that dies without `result.json` is reported by its exit code
  and the last lines of its log (private paths replaced by their file
  name).
- `params`: the catalogue's defaults, then the standard values the
  service passes where the model declares them (`language` for
  recognisers: the session's or the run's, `auto` to detect;
  `num_speakers` and, for the built-in diarization,
  `expected_speakers`: the session's), then the run's overrides.
  The service checks each override before the run is stored (400
  `invalid_input` otherwise): it must have the JSON type of the
  catalogue default (boolean, integer, number or string; a boolean is
  not a number; an integer is accepted for a number, not the other
  way round; a `null` default, where the model chooses, takes `null`
  or the type the entry declares for it in `types[name]`: `boolean`,
  `integer`, `number`, `string` or `string_list`; every `null`
  default must have one, and only those: `num_speakers`,
  `min_speakers` and `max_speakers` are integers, `atten_lim_db` a
  number, `language` and `streaming_mode` strings, `expected_speakers`
  a list of strings), and a number must be finite and greater than 0,
  or at least the entry's `minimums[name]` when the catalogue declares one
  (`ecapa-spectral`: `window` and `hop` at least 0.01 s, one frame).
  The runners refuse a step that would never end too: a piece of
  fewer than one sample (`piece_seconds`) or a window or hop below
  one 10 ms frame is `bad_input`.
- Memory must not grow with the recording. The denoisers that would
  hand the whole file to the model run it on pieces of
  `piece_seconds` with 1 s linear cross-fades (`_common.pieces`; the
  output always has the input's length): `deepfilternet3` (30 s,
  revision 2, #45; revision 1 took the whole file and ran out of GPU
  memory on every AMI meeting of §13, asking for 1.8–5.3 GiB more on
  25–40 min recordings) and the SpeechBrain `sepformer-dns4` and
  `metricgan-plus` (10 s). Each SpeechBrain piece is scaled back to
  its input's RMS level before the cross-fade (their output level is
  arbitrary); DeepFilterNet's pieces are not rescaled, as before.
  ClearVoice windows long audio itself (only a window's features reach
  the model) and RNNoise runs 10 ms frames on the CPU, so they are not
  cut and keep revision 1.
- `--selftest` imports the family's libraries without loading weights
  or using the network and prints one JSON line
  (`{"ok": true, "family", "python", "packages", "cuda"}`).

The subprocess adapter starts the runner from an argument list (no
shell) in its own process group, with a minimal environment (`PATH`,
`HOME`, `LANG`, the Hugging Face and torch cache variables,
`CUDA_VISIBLE_DEVICES`, `HF_HUB_OFFLINE`, `PYTORCH_CUDA_ALLOC_CONF`);
**the Hugging Face token is added only for a gated model**, read from
`HF_TOKEN` or the file `huggingface-cli login` writes, and never appears
on a command line, in a message or in a log the service writes. The
runner's output goes to `runner.log` beside its result. A runner over
`step_timeout` seconds, or of a cancelled run, is killed with its group.

### 3.1 The project family (in process)

`none` (denoise) returns the audio unchanged. `builtin` (diarize) and
the project's Whisper recognisers run the real transcription use case
on a scratch, in-memory session holding the lab's audio, so they are
exactly what a transcription does (seams resolved, words over silence
dropped, words scored, identification with the session's expected
speakers and count, sentences kept whole). The built-in diarization's
segments are the **turns** of the attributed words (a pause inside a
turn counts as speech, and it never reports overlap); its time includes
its own recognition.

### 3.2 Cache

A stage result is stored under `sha1(audio key, stage, model, params,
extra)`: the audio key is the SHA-256 of the original samples or the
key of the denoise result that wrote the audio, and `extra` holds what
an in-process tool also depends on (the transcription parameters and a
hash of the voiceprints). The cost of a run is therefore one denoise
per denoiser and one diarization and one recognition per (denoiser,
model), never the whole product, and a second run (or a resumed one)
computes only what is missing. Failures are not cached: they are tried
again. A step that fails with `out_of_memory` is also tried once more
inside the same run, right after the last step of its stage (all the
denoisers, all the diarizations or all the recognitions): another
process may have freed the GPU by then, and a denoiser is still
retried before any step reads its audio. Other failures are not
retried until the run is resumed.

### 3.3 Names for each diarizer's speakers

Each diarizer has its own labels (`SPEAKER_00`, `spk1`...). To compare
them they are mapped to the enrolled people:

1. `voiceprint`: the label's **non-overlapped** speech (stretches of
   at least 0.4 s where it is the only speaker) is cut into the
   service's short windows (1.5 s every 0.75 s, at most 120 per label),
   embedded with the service's voice model, averaged, and the averages
   are matched one to one to the voiceprints (`match_voices`: the
   session's expected people at `expected_floor` 0.8 when it lists
   them, else everyone at `match_threshold` 0.95). `similarity` is kept.
2. `builtin_overlap`: a label left without a person takes, one to one,
   a person the built-in diarization named on the same audio (else on
   the original), by the most shared time.
3. `none`: the label stays anonymous (`~label`, which never equals an
   enrolled id or another track's anonymous voice).

The built-in diarization names its people itself (`pipeline`). The
mapping is cached like a stage.

## 4. Comparison (agreement, not accuracy)

Time is cut into 10 ms frames; a frame belongs to a speaker when its
middle lies in one of that speaker's segments, so each frame holds a
set of names. Tolerances: `collar` 0.25 s, `change_tolerance` 0.5 s,
`cut_merge` 0.2 s, `word_time_tolerance` 0.2 s.

**Tracks** (one diarizer on one denoised audio), pairwise, A taken as
reference:

- `agreement`: share of frames with speech on either side where both
  give the same set of speakers **after the best one-to-one mapping of
  their labels**; `name_agreement`: the same with the names as mapped in
  §3.3 (no mapping between the two). Frames within the collar of either
  track's boundaries, and frames silent on both sides, are not counted.
- `missed`, `false_alarm`, `confusion` (seconds of speaker time: a frame
  of two people counts twice; NIST's frame definitions after the
  optimal mapping, collar on A's boundaries) and `error_rate` = their
  sum over A's speaker time — what the DER of B would be *if A were
  right*.
- `change_points`: the times the set of speakers changes, matched one
  to one within 0.5 s; `f1`.
- `overlap`: seconds each marks with two or more speakers, both, and
  their intersection over union.

**Pipelines**, pairwise: `words` = share of words given the same
speaker. Each pipeline's words are attributed with both pipelines'
segments (the service's rule: the segment a word overlaps most, the
nearest when none), and the two directions are averaged.

**Transcripts** (one recogniser on one denoised audio), pairwise:
`wer` of B with A as reference after normalisation (Unicode NFKC, lower
case, punctuation removed, numbers as written: `25` ≠ `veinticinco`),
and `time_agreement`: share of the aligned equal tokens whose words
start and end within 0.2 s in both.

**Denoiser effect**: for every recogniser, the WER of its words on each
denoised audio against its words on the original; for every diarizer,
the agreement, name agreement and error rate of its track on each
denoised audio against its track on the original, and the change of its
agreement with the consensus.

**Consensus**: per frame, the set of speakers with the most votes. Each
**vote group** has one vote, shared by its tracks: a group is a
catalogue `lineage` (near-duplicates: two pyannote pipelines) or else
the diarizer itself, so one diarizer on five denoised versions is one
vote, not five. A frame whose highest vote is tied has no consensus.
Each track's `agreement_with_consensus` is counted on frames with a
consensus and speech in it or in the track; `words_on_consensus` is the
share of a pipeline's words whose speaker is in the consensus set (on
frames whose consensus names someone). The consensus is a majority of
models, not the truth: models that fail the same way agree.

**Fragments**: the recording cut at every change point of any track,
cuts closer than 0.2 s merged into their mean. Each fragment gives every
track's answer (the set of speakers on most of its frames), the number
of distinct answers, `disagreement` = 1 − the vote share of the most
voted answer (vote groups as above; 0 when all agree), and
`distinct_texts` (how many different normalised texts the recognisers
give). Sorted by disagreement, the most disputed fragments come first:
those are the ones worth labelling.

## 5. Reference labels and scores

The owner labels any stretch (`start`, `end`, no overlap between
labels): `person` (one enrolled id), `both` (two or more ids at once),
`nobody`, or `unsure` (left out of every score); optionally the correct
`text`. Every pipeline is scored on the labelled time only:

| Field | Meaning |
|---|---|
| `labelled_seconds` | labelled time that is not `unsure` (shown with every score) |
| `reference_speech` | speaker time in it (two people = twice) |
| `missed`, `false_alarm`, `confusion`, `error_rate` | as §4, against the labels, with the names as mapped (no optimal mapping: naming the wrong person is an error) |
| `overlap_labelled_seconds`, `overlap_detected`, `overlap_exact` | on `both` labels: share of frames with two or more speakers, and with exactly the labelled people |
| `words_counted`, `words_right`, `words_right_share` | words whose middle is in a `person`/`both` label, and those given one of its people |
| `words_in_silence` | words inside `nobody` labels |
| `text_labels`, `reference_tokens`, `wer` | WER of the pipeline's words inside the labels that have a text |
| `rank` | by error rate, then words on the right speaker, then WER |

## 6. Storage (private: never in the repository)

```
<data_dir>/lab/<session_id>/
  original.json              key of the original audio
  audio/<key>.wav            the original (16 kHz mono 16-bit) and each denoiser's output
  stages/<key>/outcome.json  one stage result (+ runner.log, …)
  mappings/<key>.json        speaker mappings
  runs/<run_id>.json         a run; .cancel asks it to stop; .results.json its comparison
  labels.json                the owner's labels
  reference.json             a corpus recording's reference (§13)
  work/<random>/             stages being computed
```

Ids and keys are checked against their patterns before a path is made;
JSON is written atomically; a stage's directory appears in one rename.
A run records the process that queued or runs it: a `queued` or
`running` run whose process is gone is reported `interrupted` and can be
resumed. Cancelling leaves a marker file the running process checks
between steps and while a runner works (it then kills the runner).

Deleting a session (`DELETE /api/sessions/{id}`, `sessions delete`)
removes `lab/<session_id>/` with it, through the `SessionLabData`
port, before the session itself: a new session that reuses the id
starts with no runs, labels or audio. While one of the session's lab
runs is `queued` or `running` the deletion is refused (409
`wrong_state`): cancel the run first. An `interrupted` run does not
block it.

Starting or resuming a run holds the session's lock (the one the
deletion holds) from the session check through storing the original
audio, saving the run and queueing it. A deletion and a start (or a
resume) of the same session therefore never interleave: either the
deletion comes first and the start finds no session (404), or the
start comes first and the deletion sees the queued run (409). A start
decoding a long recording never writes `original.json`, the WAV copy
or `runs/<id>.json` after its lab directory was removed, so a new
session with the same id cannot inherit them.

## 7. API

All routes are under `/api/lab`, need the access code header
(`X-Access-Code`) and answer errors as `{"code", "message"}` like the
rest of the API (`invalid_input` 400, `not_found` 404, `wrong_state`
409, `invalid_request` 422, `payload_too_large` 413). JSON bodies are
limited to `api.json_max_bytes` (64 KiB), labels to
`lab.labels_max_bytes` (2 MiB, at most 5000 labels). Runs execute one
at a time on the service's job queue, after any transcription queued
before them.

### `GET /api/lab/catalogue`

```json
{"token_present": false,
 "models": [
  {"id": "pyannote-3.1", "stage": "diarize", "family": "pyannote",
   "display_name": "pyannote speaker-diarization 3.1",
   "source_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
   "licence": "MIT", "licence_commercial": true, "gated": true,
   "token_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
   "languages": ["any"], "vram_gb": 1.5,
   "parameters": {"num_speakers": null}, "types": {"num_speakers": "integer"},
   "weights": ["pyannote/speaker-diarization-3.1"],
   "weights_mb": 32, "notes": "…", "lineage": "pyannote",
   "state": "needs_token",
   "reason": "needs token: accept the conditions at https://huggingface.co/pyannote/speaker-diarization-3.1 and log in (huggingface-cli login)",
   "weights_cached": false}]}
```

`state`: `ready`, `needs_token` or `not_installed` (`reason` says why).
`weights_cached`: whether the weights are already downloaded (`null`:
unknown); a first run downloads them.

### `POST /api/lab/sessions/{session_id}/runs` → 202

```json
{"denoisers": ["none", "deepfilternet3"],
 "diarizers": ["builtin", "sortformer-streaming-v2"],
 "recognisers": ["whisper-large-v3-turbo"],
 "params": {"sortformer-streaming-v2": {"num_speakers": 2}},
 "language": null, "num_speakers": null}
```

Every field is optional: `denoisers` defaults to `["none"]`, `diarizers`
to `["builtin"]`, `recognisers` to `[]` (at least one diarizer or
recogniser). At most 16 models per stage and 600 pipelines; `params`
may only override parameters the model declares, with a value of the
default's type (for a `null` default, `null` or its declared `types`
entry) and, for a number, greater than 0 or the catalogue's minimum
(§3). `language` and
`num_speakers` default to the session's. The session must have its
audio (any state but `recording`). Answer: the run (as below, state
`queued`).

### `GET /api/lab/sessions/{session_id}/runs`

```json
{"runs": [{"id": "9f…", "session_id": "R1", "created_at": "…",
  "finished_at": "…", "state": "done", "request": {…},
  "message": null, "pipeline_count": 4, "done_steps": 5,
  "total_steps": 5}]}
```

### `GET /api/lab/sessions/{session_id}/runs/{run_id}`

```json
{"id": "9f…", "state": "running", "pipeline_count": 4,
 "done_steps": 2, "total_steps": 5,
 "steps": [{"name": "denoise:deepfilternet3", "stage": "denoise",
            "model": "deepfilternet3", "denoiser": "deepfilternet3",
            "state": "done", "key": "3c…", "cached": false,
            "error": null, "seconds": 3.1}, …],
 "pipelines": [{"id": "none+sortformer-streaming-v2+whisper-large-v3-turbo",
   "denoiser": "none", "diarizer": "sortformer-streaming-v2",
   "recogniser": "whisper-large-v3-turbo", "state": "failed",
   "error": {"code": "out_of_memory", "message": "…"}}],
 "job": {"state": "running", "stage": "diarize:none+sortformer-streaming-v2",
         "fraction": 0.4, "message": null}}
```

Run states: `queued`, `running`, `done`, `partial` (some steps failed,
some pipelines finished), `failed` (no pipeline finished), `cancelled`,
`interrupted`. Step states: `pending`, `running`, `done`, `failed`,
`cancelled`. A pipeline is `done` when all its steps are.

### `POST …/runs/{run_id}/cancel` → 200, `POST …/runs/{run_id}/resume` → 202

Cancel: only a `queued` or `running` run (else 409 `wrong_state`);
answers the status. Resume: only a run that is not queued or running;
done steps stay, the others are tried again; answers the run. The check
and the queueing hold the session's lock and then the run's lock, so of
two concurrent resumes one is 202 and the other 409 `wrong_state`, and
a concurrent deletion of the session waits and is then refused (§6).

### `GET …/runs/{run_id}/results`

The comparison of §4 without the fragments (404 until the run ended):

```json
{"schema_name": "lab.results.v1", "run_id": "9f…", "session_id": "R1",
 "duration": 387.9,
 "parameters": {"frame": 0.01, "collar": 0.25, "change_tolerance": 0.5,
                "cut_merge": 0.2, "word_time_tolerance": 0.2},
 "models": [{"id": "diarizen-wavlm-large-s80-md", "stage": "diarize",
             "licence": "CC BY-NC 4.0", "licence_commercial": false, …}],
 "tracks": [{"id": "none+builtin", "denoiser": "none", "diarizer": "builtin",
             "vote_group": "builtin", "speakers": ["A", "B"],
             "speaker_seconds": {"A": 201.3, "B": 120.4},
             "speech_seconds": 320.1, "overlap_seconds": 0.0, "labels": 2,
             "mapping": [{"label": "A", "speaker": "A", "method": "pipeline",
                          "similarity": null, "overlap_seconds": null,
                          "speech_seconds": 201.3}],
             "agreement_with_consensus": 0.91, "consensus_seconds": 300.2,
             "seconds": 15.2, "peak_gpu_mb": null, "device": null,
             "model_version": "openai/whisper-large-v3-turbo"}],
 "transcripts": [{"id": "none+whisper-large-v3-turbo", "words": 830,
                  "language": "es", "seconds": 13.0, …}],
 "pipelines": [{"id": "none+builtin+whisper-large-v3-turbo", "track": "none+builtin",
                "transcript": "none+whisper-large-v3-turbo", "speakers": ["A", "B"],
                "speech_seconds": 320.1, "overlap_seconds": 0.0, "words": 830,
                "seconds": 28.2, "peak_gpu_mb": null,
                "agreement_with_consensus": 0.91, "words_on_consensus": 0.93,
                "licence_commercial": true}],
 "diarization": [{"reference": "none+builtin", "hypothesis": "none+sortformer-…",
                  "agreement": 0.8, "name_agreement": 0.78, "scored_seconds": 290.0,
                  "reference_speech": 300.0, "missed": 10.0, "false_alarm": 20.0,
                  "confusion": 30.0, "error_rate": 0.2,
                  "change_points": {"reference": 120, "hypothesis": 150, "matched": 90, "f1": 0.67},
                  "overlap": {"reference_seconds": 0.0, "hypothesis_seconds": 37.6,
                              "both_seconds": 0.0, "iou": 0.0}}],
 "words": [{"left": "…", "right": "…", "agreement": 0.88}],
 "recognition": [{"reference": "none+whisper-large-v3-turbo",
                  "hypothesis": "none+faster-whisper-large-v3-turbo",
                  "wer": 0.12, "reference_tokens": 830, "matched_tokens": 760,
                  "time_agreement": 0.8}],
 "consensus": {"coverage": 0.95, "vote_groups": {"none+builtin": "builtin"},
               "weights": {"none+builtin": 1.0},
               "spans": [{"start": 0.0, "end": 0.48, "speakers": [], "agreed": true}]},
 "denoiser_effect": [{"denoiser": "deepfilternet3", "stage": "recognise",
                      "model": "whisper-large-v3-turbo", "wer": 0.05,
                      "time_agreement": 0.9, "agreement": null, …}],
 "fragment_count": 812,
 "reference": null}
```

`reference` is `null` except on a session imported from a public
corpus, where it holds the scores against the corpus's reference (§13).

(The numbers above illustrate the shape; they are not measurements.)
Ids: a track or transcript is `denoiser+model`, a pipeline
`denoiser+diarizer+recogniser` (`-` for a missing stage). Speaker names
are enrolled ids or `~label` for a voice nobody was found for.

### `GET …/runs/{run_id}/tracks`

Each track's named segments and the consensus, for a timeline:

```json
{"tracks": {"none+builtin": [{"start": 0.48, "end": 6.1, "speaker": "A"}]},
 "consensus": [{"start": 0.0, "end": 0.48, "speakers": [], "agreed": true}]}
```

### `GET …/runs/{run_id}/fragments?offset=0&limit=50&sort=disagreement`

`sort`: `disagreement` (most disputed, then most different texts, then
longest) or `time`; `limit` 1–200; `unlabelled=true` keeps only the
fragments no reference label covers (`total` counts those).

```json
{"total": 812, "offset": 0, "limit": 50, "sort": "disagreement",
 "fragments": [{"index": 31, "start": 44.12, "end": 45.3,
   "answers": {"none+builtin": ["A"], "none+sortformer-…": ["A", "B"]},
   "distinct_answers": 2, "disagreement": 0.5, "modal": ["A"],
   "distinct_texts": 2,
   "pipelines": {"none+builtin+whisper-large-v3-turbo": {
       "speakers": ["A"], "text": "…",
       "words_by_speaker": [{"speaker": "A", "text": "…"}]}},
   "label": null}]}
```

`label` is the reference label holding the fragment's middle, if any.

### `GET …/runs/{run_id}/fragment-spans`

Every fragment in time order without the pipelines' answers (a few
kilobytes instead of a page of answers): what the review tab needs to
count labelled fragments and to mark the disputed stretches.

```json
{"spans": [{"index": 0, "start": 0.0, "end": 0.48,
            "disagreement": 0.0, "distinct_answers": 1}]}
```

### `GET …/runs/{run_id}/audio/{denoiser}`

The original (`none`) or that denoiser's output as WAV (16 kHz mono),
whole or by one byte range (206), like the session audio; it also
accepts the playback cookie of `POST /api/access/cookie`, so an
`<audio>` element can play it. Play a fragment by seeking to its start.
The route only reads: the original is stored when a run is started or
resumed, and a missing one (or a denoiser's audio not made yet) is 404
`not_found`.

### `GET /api/lab/sessions/{session_id}/labels`, `PUT` the same

```json
{"labels": [
  {"start": 44.12, "end": 45.3, "kind": "both", "speakers": ["A", "B"],
   "text": null, "updated_at": "2026-11-14T18:00:00Z"},
  {"start": 45.3, "end": 47.9, "kind": "person", "speakers": ["A"],
   "text": "texto corregido", "updated_at": null}],
 "speakers": ["A", "B"]}
```

`PUT` replaces every label (body `{"labels": [...]}`) and answers the
stored set; labels must not overlap, must lie in the recording and may
name only enrolled ids (`speakers` lists them). `updated_at` is set
when missing.

### `GET …/runs/{run_id}/scores`

```json
{"labels": 40, "labelled_seconds": 61.5, "unsure_seconds": 4.0,
 "text_labels": 12,
 "scores": [{"pipeline": "none+…", "rank": 1, "labelled_seconds": 61.5,
             "reference_speech": 70.2, "missed": 1.2, "false_alarm": 0.8,
             "confusion": 2.0, "error_rate": 0.057, "overlap_labelled_seconds": 8.7,
             "overlap_detected": 0.6, "overlap_exact": 0.5, "words_counted": 150,
             "words_right": 141, "words_right_share": 0.94, "words_in_silence": 0,
             "text_labels": 12, "reference_tokens": 95, "wer": 0.08}]}
```

### The "Pruebas" tab

The web pages have a tab for all of this (`#/lab`, behind the access
code; texts in Spanish and English): pick a recording, choose models by
stage (licence, non-commercial badge, token and installed state, VRAM),
see how many pipelines and steps that is, start, follow the steps,
cancel and resume; then for an ended run an overview (sortable pipeline
table, heat map of pairwise track agreement for one denoiser or one
diarizer with a table alternative, recognisers' pairwise WER, denoiser
effect), a timeline (one row per diarizer, consensus and labels, click
to play the original or a denoised version through the audio route and
the playback cookie), the fragments to label (grouped answers, text
variants, keys 1/2/B/N/U/X/space/J/K, saved with debounced `PUT
…/labels`, kept in the browser and retried when saving fails) and the
scores with a client-side JSON export (a `blob:` download link, which
the site's CSP does not restrict: it is a navigation, not a fetch). The
results are fetched once per run. Code: `web/js/views/lab*.js`, pure
logic in `web/js/lib/lab.js` (tests in `tests/web/lab.test.mjs`).

## 8. CLI

```sh
python -m transcriber lab catalogue [--json]
python -m transcriber lab run SESSION [--all] [--denoise a,b] [--diarize x,y] \
    [--recognise r] [--language es] [--speakers 2] [--params JSON] [--json]
python -m transcriber lab runs SESSION [--json]
python -m transcriber lab summary SESSION RUN [--json]
python -m transcriber lab resume SESSION RUN [--json]
python -m transcriber lab import-reference ami|callhome [--limit N] [--split S] [--json]
python -m transcriber lab reference-summary [--corpus ami|callhome] [--json]
```

The last two are described in §13.

`run` runs in the command's own process (its own job queue) and prints
aggregate numbers only. It uses the GPU: do not start it while `serve`
is working on a job. It does not take `serve.lock` and writes only under
`<data_dir>/lab`, so it may run beside a running service.

## 9. Catalogue on this machine (2026-11-14)

`lab/catalogue.json` is written by `lab/make_catalogue.py`; licences,
gating and sizes were read from the model cards and the Hugging Face API
on 2026-11-14. "Peak MB" is the runner process's peak (torch reserved or
`nvidia-smi`, the larger) on R1; in-process models are not measured.

| Stage | Id | Family | Licence | State | Peak MB on R1 |
|---|---|---|---|---|---|
| denoise | `none` | project | – | ran | – |
| denoise | `deepfilternet3` | enhance | MIT / Apache-2.0 | ran (revision 2: 30 s pieces) | (1.9 GB on the test clip, whole file) |
| denoise | `mossformer2-se-48k` (ClearerVoice-Studio) | enhance | Apache-2.0 | ran | (0.46 GB, test clip) |
| denoise | `mossformergan-se-16k` (ClearerVoice-Studio) | enhance | Apache-2.0 | ran | (6.3 GB, test clip) |
| denoise | `sepformer-dns4` (SpeechBrain) | speechbrain | Apache-2.0 | ran (revision 2) | (1.8 GB, test clip) |
| denoise | `metricgan-plus` (SpeechBrain) | speechbrain | Apache-2.0 | ran (revision 2) | (0.3 GB, test clip) |
| denoise | `rnnoise` | enhance | BSD-3-Clause | ran (CPU) | – |
| diarize | `builtin` | project | – | ran | – |
| diarize | `pyannote-3.1` | pyannote | MIT, **gated** | ran (owner's token) | 2216 |
| diarize | `pyannote-community-1` | pyannote | CC BY 4.0, **gated** | ran (owner's token) | 2216 |
| diarize | `sortformer-streaming-v2` | nemo | CC BY 4.0 | ran | 1260–1454 |
| diarize | `nemotron-3-diarization` | transformers | OpenMDW 1.1 | ran | 618 |
| diarize | `diarizen-wavlm-large-s80-md-v2` | diarizen | **CC BY-NC 4.0 (non-commercial)** | ran | 3742 |
| diarize | `ecapa-spectral` (SpeechBrain ECAPA + spectral clustering, written here) | speechbrain | Apache-2.0 | ran | 1612–1616 |
| diarize | `nemo-clustering` (NeMo clustering diarizer, TitaNet-L) | nemo | CC BY 4.0 | ran | 2904–2982 |
| recognise | `whisper-large-v3-turbo` | project | MIT | ran | – |
| recognise | `whisper-large-v3` | project | Apache-2.0 | ran (full run, §11.7) | – (in-process, not measured) |
| recognise | `faster-whisper-large-v3-turbo` (CTranslate2 int8, converted here from the cached turbo) | ctranslate2 | MIT | ran | 1362–1394 |
| recognise | `parakeet-tdt-0.6b-v3` (NVIDIA, 25 European languages incl. Spanish) | nemo | CC BY 4.0 | ran | 6346 |

Nemotron-3 diarization did not need NeMo's development branch: the
project's transformers 5.20.2 has the model, so it runs in the
`transformers` family (Python 3.10). The environments take 0.93 GB in
all (hard-linked files shared), the converted and ClearerVoice models
1.1 GB, the downloaded weights about 4 GB (parakeet 2.5 GB). Postponed
for disk: nothing now (`whisper-large-v3`, 3.1 GB of weights, was
installed and ran, §11.7); optional DiariZen v1 or base variants
(0.28 / 0.10 GB) are still not installed. `--all` never selects a model whose weights
are known to be missing.

To enable the gated pyannote pipelines on another machine: accept the
conditions at https://huggingface.co/pyannote/speaker-diarization-3.1,
https://huggingface.co/pyannote/segmentation-3.0 and
https://huggingface.co/pyannote/speaker-diarization-community-1, then
`huggingface-cli login` (or set `HF_TOKEN` for the service). No further
gated dependency was needed on R1.

## 10. Future work: speech separation

ClearerVoice-Studio's `MossFormer2_SS_16K` (Apache-2.0, 670 MB) splits
a mixture into two speaker streams. That is two WAVs, which does not fit
the one-WAV denoise contract honestly: mixing the streams back gives the
input again, and keeping one drops a person. It needs a stage of its
own: separate, then diarize and recognise each stream, then match the
streams to the voiceprints. It only helps where voices overlap and may
invent a second voice in single-speaker stretches, so it should run on
the overlap regions the diarizers mark. The `enhance` environment
already has ClearerVoice; no new environment is needed.

## 11. Results on R1 (first run, 2026-11-14)

R1 is the first real session (388 s, two enrolled people A and B,
listed as expected, count 2; recorded with language `es`), chosen by the
owner because the two talk over each other. One suite: 7 denoisers
(`none` and six) × 8 diarizers × 3 recognisers = **168 pipelines** from
83 steps (6 denoise, 56 diarize, 21 recognise), every step finished.
The results stay in `data/transcriber/lab/` for the web tab; below are
aggregates only. Every figure is **agreement between models**: nothing
here says which model is right (§11.6).

Two runs were made. In the first, `sepformer-dns4` wrote audio at about
17 times the input's RMS level (clipped) and `metricgan-plus` at twice
it; on the SepFormer audio no diarizer's voice matched a voiceprint and
DiariZen found no speech at all. The runner now scales each piece back
to its input's level (revision 2, §3.2 cache key); the second run
recomputed only those two denoisers and their 22 dependent steps. The
numbers below are the second run. A third run added
`whisper-large-v3` (224 pipelines, §11.7); §11.1 to §11.6 are the same
in it.

### 11.1 Cost (seconds of processing without model loading; R1 = 388 s)

| Denoiser | s | | Diarizer on the original | s | peak MB | | Recogniser on the original | s | peak MB | words |
|---|---|---|---|---|---|---|---|---|---|---|
| deepfilternet3 | 3.2 | | builtin (incl. its recognition) | 53.8 | – | | whisper-large-v3-turbo (project) | 13.6 | – | 825 |
| mossformer2-se-48k | 13.2 | | pyannote-3.1 | 8.1 | 2216 | | faster-whisper-large-v3-turbo | 9.2 | 1362 | 867 |
| mossformergan-se-16k | 68.3 | | pyannote-community-1 | 6.6 | 2216 | | parakeet-tdt-0.6b-v3 | 1.3 | 6346 | 654 |
| sepformer-dns4 | 15.7 | | sortformer-streaming-v2 | 1.7 | 1454 | | | | | |
| metricgan-plus | 1.7 | | nemotron-3-diarization | 10.3 | 618 | | | | | |
| rnnoise (CPU) | 12.2 | | diarizen-wavlm-large-s80-md-v2 | 20.2 | 3742 | | | | | |
| | | | ecapa-spectral | 1.9 | 1612 | | | | | |
| | | | nemo-clustering | 17.0 | 2982 | | | | | |

The built-in diarization's time includes its own recognition; it took
53.8 s on the original (the first in-process step of the job) and
14–19 s on the denoised versions (60.8 s on `mossformergan-se-16k`; the
reason was not investigated). The comparison of the whole suite
took 27.8 s on the CPU; a page of 50 fragments 0.8 s; the stored
results are 6.8 MB.

### 11.2 What each diarizer found (original audio)

| Diarizer | labels | people named | speech s | overlap s | agreement with consensus |
|---|---|---|---|---|---|
| builtin | 2 | A, B (itself) | 332.0 | 0.0 | 0.765 |
| pyannote-3.1 | 2 | A, B | 303.9 | 31.6 | 0.855 |
| pyannote-community-1 | 2 | A, B | 303.9 | 31.6 | 0.853 |
| sortformer-streaming-v2 | 4 | A, B + 2 anonymous (5 and 22 s) | 297.2 | 37.6 | 0.739 |
| nemotron-3-diarization | 4 | A, B + 2 anonymous (44 and 4 s) | 280.3 | 24.6 | 0.666 |
| diarizen-wavlm-large-s80-md-v2 | 2 | A, B | 296.1 | 30.7 | 0.859 |
| ecapa-spectral | 2 | A, B | 288.6 | 0.0 | 0.843 |
| nemo-clustering | 2 | A, B | 336.3 | 0.0 | 0.801 |

Every diarizer's two main labels were named A and B by their own voice
(`voiceprint`, cosine 0.98–0.99 for A and 0.92–0.95 for B on the
original); the built-in names them itself. The labels left anonymous
are the extra voices Sortformer and Nemotron split off (§11.6).
Consensus: 7 vote groups (the two pyannote pipelines share one), a
consensus on 99.6% of the frames.

### 11.3 Diarizers against each other (original audio)

Agreement after the best label mapping (`name_agreement` is the same
except Nemotron, 0.74–0.77, because of its anonymous voice):

| | builtin | pya-3.1 | pya-c1 | sortformer | nemotron | diarizen | ecapa | nemo-cl. |
|---|---|---|---|---|---|---|---|---|
| builtin | – | 0.82 | 0.82 | 0.71 | 0.70 | 0.82 | 0.85 | 0.88 |
| pyannote-3.1 | | – | 1.00 | 0.82 | 0.85 | 0.99 | 0.87 | 0.87 |
| pyannote-community-1 | | | – | 0.82 | 0.85 | 0.99 | 0.87 | 0.87 |
| sortformer-streaming-v2 | | | | – | 0.85 | 0.83 | 0.78 | 0.77 |
| nemotron-3-diarization | | | | | – | 0.86 | 0.77 | 0.76 |
| diarizen | | | | | | – | 0.89 | 0.87 |
| ecapa-spectral | | | | | | | – | 0.93 |

- The two pyannote pipelines agree on 99.5% or more and mark the same
  speech and overlap (both use `segmentation-3.0`); DiariZen (a
  pyannote fork with WavLM) agrees with them at 0.99 and an error rate
  of 0.01–0.04 against either.
- Overlap: the overlap-aware models agree with each other on where it
  is (IoU 0.60–0.74 between pyannote, Sortformer, Nemotron and DiariZen;
  they mark 25–38 s); the built-in, ECAPA and NeMo clustering mark none.
- Change points within 0.5 s: F1 0.67–0.87 among the overlap-aware
  models, 0.62–0.70 between them and ECAPA, 0.34–0.50 between them and
  the built-in or NeMo clustering.
- Over all 3080 ordered pairs of the 56 tracks the agreement has a mean
  of 0.861 (0.545 to 1.000).

### 11.4 Recognisers against each other (original audio)

WER of the row's words with the column's as reference (time agreement of
the matched words in brackets):

| | whisper-turbo (project) | faster-whisper-turbo | parakeet-v3 |
|---|---|---|---|
| hypothesis whisper-turbo | – | 0.165 (0.73) | 0.494 (0.62) |
| hypothesis faster-whisper | 0.173 (0.73) | – | 0.569 (0.64) |
| hypothesis parakeet | 0.392 (0.62) | 0.429 (0.65) | – |

(read: with whisper-turbo as reference, faster-whisper's words have a
WER of 0.173.) Parakeet returned 654 words against 825 and 867; the
two Whisper builds of the same weights differ on about one word in six.
R1 is mostly Spanish; parakeet detects the language itself (it reports
none).

### 11.5 Effect of each denoiser (against the same model on the original)

| Denoiser | WER whisper-turbo | WER faster-whisper | WER parakeet | diarizer agreement (8 diarizers, min–max) | change of agreement with consensus (min–max) |
|---|---|---|---|---|---|
| deepfilternet3 | 0.108 | 0.152 | 0.456 | 0.85–0.99 | −0.03 … +0.19 |
| mossformer2-se-48k | 0.091 | 0.140 | 0.431 | 0.80–0.97 | −0.02 … +0.24 |
| mossformergan-se-16k | 0.142 | 0.149 | 0.450 | 0.82–0.97 | −0.04 … +0.21 |
| sepformer-dns4 (rev. 2) | 0.115 | 0.134 | 0.304 | 0.85–1.00 | −0.03 … +0.21 |
| metricgan-plus (rev. 2) | 0.355 | 0.393 | 0.454 | 0.71–0.90 | −0.17 … +0.14 |
| rnnoise | 0.133 | 0.205 | 0.485 | 0.84–0.97 | −0.05 … +0.18 |

- Every denoiser changes the recognisers' words (WER 0.09–0.21 for the
  Whisper builds, 0.30–0.49 for parakeet, which heard 654 words on the
  original and 773–785 on the denoised versions, except SepFormer's 590
  and MetricGAN+'s 640). Whether for better or worse is not known
  without corrected text.
- On the denoised audio the overlap-aware diarizers mark much less
  overlap (pyannote 6–19 s instead of 32 s; Sortformer 7–28 instead of
  38) and most agree more with the consensus (Sortformer +0.12 to +0.17,
  except +0.02 on SepFormer and −0.09 on MetricGAN+; Nemotron +0.14 to
  +0.24, splitting off less anonymous speech). Whether the removed overlap was real or noise is the first
  thing the labels must decide.
- MetricGAN+ (trained on read speech with synthetic noise) is the
  outlier: the Whisper builds hear 577 and 601 words on it instead of
  765–867 on the other versions, every diarizer finds 20–81 s less
  speech than on the original, and 7 of 8 diarizers agree less with the
  consensus.
- The pipelines with the highest agreement are all on denoised audio:
  `mossformer2-se-48k` with pyannote (0.923 / 0.919), DiariZen (0.910)
  or Nemotron (0.904); on the original the best is DiariZen (0.859).
  This is agreement with a consensus the same denoised tracks vote in,
  not evidence that they are right.

### 11.6 What cannot be concluded without the owner's labels

- **Which diarization is right.** The consensus is a majority of
  models; correlated models (the pyannote family and DiariZen) agree
  with one another by construction. 395 fragments (mean 0.98 s); 319
  have some disagreement, 6 (7.5 s) have half or more of the vote
  against the most common answer. The disputed fragments are listed
  first by the API: labelling them is the cheapest way to settle it.
- **How much overlap there is.** 0 s (built-in, clustering models),
  25–38 s (overlap-aware models on the original), 7–28 s on denoised
  audio. Only `both` labels (§5) can tell.
- **Whether the extra voices are real.** Sortformer and Nemotron give
  a third and fourth anonymous voice on the original (up to 44 s); only
  two people are enrolled and listed, but a third sound (a laugh, a
  television) is possible.
- **Which recogniser is right**, and whether a denoiser helps the
  text: there is no reference transcript; the WERs above are
  disagreements. A few `text` corrections on labelled fragments give a
  real WER (§5).
- **Generality.** One session of two people and one phone; every
  figure can move with another recording.

### 11.7 Full run with `whisper-large-v3` (run `e7a37b38`, 224 pipelines)

The suite of §11.1 plus the fourth recogniser `whisper-large-v3` on the
original and the six denoised tracks: 7 denoisers x 8 diarizers x 4
recognisers = **224 pipelines** from 90 steps (6 denoise, 56 diarize,
28 recognise). The 83 steps of the second run came from the cache; only
the 7 `whisper-large-v3` steps ran. The run needed one resume (one step
died with CUDA out of memory because another process held the GPU); it
ended `done` with every step finished, 2026-11-14 18:15 to 18:58. The
stored results are 8.9 MB.

**Did the earlier figures change?** No. Compared with the second run
(§11.1 to §11.6), the 56 tracks, the 3080 diarizer pairs, the 168
earlier pipelines, the 420 earlier recogniser pairs, the 14028 earlier
word-agreement pairs, the 66 earlier denoiser-effect rows, the
transcripts' word counts and the consensus (coverage 0.996, 332 spans,
same vote groups) are identical field by field. Only the fragments'
`distinct_texts` changed (184 of 395, more text variants because of the
new recogniser): fragments with more than one text 381 instead of 369;
the 319 fragments with disagreement and the 6 with half or more of the
vote against the modal answer are the same.

Cost against the other recognisers (R1 = 388 s; seconds without model
loading, as in §11.1):

| Recogniser | s on the original | peak MB | words on the original | s on the 6 denoised tracks |
|---|---|---|---|---|
| whisper-large-v3-turbo (project) | 13.6 | – | 825 | – |
| whisper-large-v3 (project) | 313.2 | – (in-process) | 756 | 29.1 to 41.0 |
| faster-whisper-large-v3-turbo | 9.2 | 1362 | 867 | – |
| parakeet-tdt-0.6b-v3 | 1.3 | 6346 | 654 | – |

`whisper-large-v3` is the slowest recogniser: about 23 times the turbo
build's time on the original, 34 times faster-whisper's. The 313 s of
the first step are ten times the other six (29 to 41 s on the denoised
tracks); it is the first step that used the 3.1 GB of weights and the
run needed a resume, but the cause of the difference was not
investigated (the step records do not say), so the 29 to 41 s are the
better estimate of its cost. Its
peak GPU memory was not measured (it runs inside the job, like the
project's turbo build).

WER on the original, hypothesis in the row, reference in the column (the
convention of §11.4; time agreement in brackets):

| | whisper-turbo (project) | whisper-large-v3 | faster-whisper-turbo | parakeet-v3 |
|---|---|---|---|---|
| hypothesis whisper-turbo | – | 0.136 (0.78) | 0.165 (0.73) | 0.494 (0.62) |
| hypothesis whisper-large-v3 | 0.148 (0.78) | – | 0.216 (0.58) | 0.417 (0.38) |
| hypothesis faster-whisper | 0.173 (0.73) | 0.188 (0.58) | – | 0.569 (0.64) |
| hypothesis parakeet | 0.392 (0.62) | 0.482 (0.38) | 0.429 (0.65) | – |

(read: with whisper-large-v3 as reference, the turbo build's words have a
WER of 0.136.) The columns and rows without `whisper-large-v3` are the
numbers of §11.4. `whisper-large-v3` is closest to its turbo sibling
(0.14 to 0.15), further from the CTranslate2 build (0.19 to 0.22) and
furthest from parakeet (0.42 to 0.48), with the lowest time agreement
of any pair (0.38). It returned 756 words, between parakeet's 654 and
the turbo build's 825.

Effect of each denoiser on `whisper-large-v3` (WER against its own words
on the original; words in brackets; time agreement of matched words
0.80 to 0.93):

| Denoiser | WER whisper-large-v3 | words | (whisper-turbo, §11.5) |
|---|---|---|---|
| deepfilternet3 | 0.107 | 744 | 0.108 |
| mossformer2-se-48k | 0.123 | 770 | 0.091 |
| mossformergan-se-16k | 0.159 | 788 | 0.142 |
| sepformer-dns4 (rev. 2) | 0.098 | 784 | 0.115 |
| metricgan-plus (rev. 2) | 0.378 | 587 | 0.355 |
| rnnoise | 0.132 | 733 | 0.133 |

It behaves like the turbo build: every denoiser except MetricGAN+ changes
10 to 16% of the words (WER), MetricGAN+ is the outlier (0.378, 587 words against 733 to
788 on the other versions), SepFormer changes it least (0.098). Whether
any of that is better is still unknown without corrected text (§11.6).

Over the pipelines, the share of words falling inside the consensus
speech (`words_on_consensus`) is 0.959 on average for each of
`whisper-large-v3`, the turbo build and parakeet, 0.956 for
faster-whisper; the range for `whisper-large-v3` is 0.770 to 0.990. The
highest agreement with the consensus is still `mossformer2-se-48k` with
pyannote-3.1 (0.923): the recogniser does not enter the diarization, so
the four recognisers share every diarization figure and the new
recogniser adds text, not speaker information.

## 12. Status and how to resume (2026-11-14)

Merged in main (PR #32, fixes #35, #36, #38): domain, use cases,
adapters, API, CLI, make targets, tests, seven environments built and
self-tested, and the web tab "Pruebas" (deployed). **Every model of §9
ran**: the full run `e7a37b38` of session R1 has all 224 pipelines
(`done`, §11.7); its results are in `data/transcriber/lab/<R1>/runs/`
(the newest run). Optional, not installed: DiariZen v1 / base variants.

Left: the owner's labels on the disputed fragments (§11.6), which
decide every "which is right" question; DiariZen v1 / base if wanted.

Resumability was checked twice on R1. The second run of the same suite
found 59 of its 83 steps in the cache and recomputed only the 24 that
depended on the two revised denoisers. The full run found 83 of 90 steps
in the cache, ran the 7 `whisper-large-v3` steps, and was resumed once
after a CUDA out-of-memory on one step (another process held the GPU):
`resume` kept the done steps and ran the rest. A stage becomes visible
only when its directory is renamed into `stages/` with its
`outcome.json` (written atomically); an interrupted stage leaves a
directory under `work/`, which is never read as a result. A run stopped
by a reboot is reported `interrupted`.

To resume after a reboot (from the repository root):

```sh
make lab-selftest                                  # environments still ready
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab runs <session>
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab resume <session> <run>
# or the whole suite again (cached steps are skipped):
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab run <session> --all \
  --diarize builtin,pyannote-3.1,pyannote-community-1,sortformer-streaming-v2,nemotron-3-diarization,diarizen-wavlm-large-s80-md-v2,ecapa-spectral,nemo-clustering
```

The GPU must be free of other jobs (a resume after an out-of-memory
only needs the other process to be gone; within a run, a step that
ran out of memory is tried once more after the rest of its stage,
§3.2).

`deepfilternet3` revision 2 (30 s pieces, #45) was checked on the CPU
only (the GPU was busy with the §13 run) on 60 s of `ami-TS3003a`
(from 120 s), with 20 s pieces so there are two seams, against
revision 1 on the same file: same length; identical inside the first
piece; at the recording's level (RMS 0.0033) both outputs are almost
silent (RMS 0.00002 and 0.00006); scaled ×10 (RMS 0.033) the outputs
correlate at 0.996 (RMS 0.0172 against 0.0177, difference 9% of
revision 1's RMS; at most 0.47 of it in a near-silent second close to a
seam). Its GPU peak on a long meeting is not measured yet.

## 13. Scoring against a public reference (AMI, CALLHOME)

Every figure of §4 is agreement between models. To get real error
rates without waiting for our own labelled recording, the lab imports
recordings of public corpora that come with a human reference and
scores every pipeline against it.

| Corpus | Hugging Face | Licence | Language, setting |
|---|---|---|---|
| `ami` | `diarizers-community/ami`, config `sdm`, splits `train`/`validation`/`test` (default `test`: 16 recordings in 3 shards, 313–331 MB each) | CC BY 4.0 (commercial use allowed) | English, 3–5 people, single distant microphone |
| `callhome` | `diarizers-community/callhome`, config `spa`, split `data` (5 shards, 459–501 MB) | **CC BY-NC-SA 4.0: internal evaluation only** | Spanish telephone calls, 2+ speakers |

Both datasets have one row per recording with the columns `audio`
(`{bytes, path}`), `timestamps_start`, `timestamps_end` and `speakers`
(one value per speaker turn; turns of different people overlap where
they talk at once). Neither has a transcript, so for now no WER is
measured on them; the reference format and the scoring take reference
words with times when a corpus has them. CALLHOME is gated: the
account whose token is used must accept its terms on the dataset's
page first (`huggingface-cli login`, or `HF_TOKEN`).

### 13.1 Import

```sh
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab import-reference ami --limit 2
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab import-reference callhome --limit 5
```

- Only the shards that hold the recordings asked for are downloaded, in
  order, into `<data_dir>/lab-corpora/<corpus>/` (under `data/`, ignored
  by git); offline, the shards already there are read.
- Each recording becomes a session `<corpus>-<recording>` (AMI:
  `ami-ES2004a` from the file name; CALLHOME rows have no file name, so
  `callhome-data-0001`... in corpus order), state `uploaded`, the
  corpus's language (`en`, `es`), no expected speakers or count (the
  diarizers estimate them), title `<corpus> <recording>`. The corpus's
  audio is stored as given; the lab decodes it to 16 kHz mono like any
  recording (§3).
- Beside it, `lab/<session>/reference.json` (`lab.reference.v1`): the
  speaker turns (anonymous corpus ids such as `MEE071`, or `A`/`B`;
  turns past the end of the audio cut there), the reference words when
  there are any, and `corpus`, `split`, `licence`, `licence_commercial`
  and `source_url`. It goes with the session when the session is
  deleted (§6).
- A recording whose session exists is skipped, so an interrupted import
  can simply be run again. A gated corpus whose terms were not accepted
  fails before downloading anything, with the page where to accept them.

A session with a reference is never compared with the people enrolled
on this machine: the run uses no voiceprint for it (no `voiceprint` or
`builtin_overlap` naming, §3.3, and the built-in diarization runs with
nobody enrolled), so every diarizer's labels stay anonymous (`~label`)
and a corpus speaker can never be given a household member's id. The
scores below pair labels with the reference's speakers themselves, so
they do not need names.

If an import stops between creating a session and storing its
reference, the next import stores the reference (`repaired`) instead of
skipping the session; a session of that id not made by the import (its
title differs) is refused.

### 13.2 Scores

A run on an imported session is run and compared as any other (§4);
its results gain `reference`, where the reference is treated as one
more track and one more transcript (the same code as §4; it takes no
part in the consensus or the fragments):

| Field | Meaning |
|---|---|
| `tracks[].der` | `compare_tracks` with the reference as reference: missed + false alarm + confusion over the reference's speaker time, after the best one-to-one mapping of labels, frames within the collar (`collar`, 0.25 s) of the reference's boundaries not scored (`missed`, `false_alarm`, `confusion`, `reference_speech` in seconds) |
| `tracks[].jer` | Jaccard error rate on the same frames: reference and track speakers paired one to one to maximise the sum of their intersection over union; per reference speaker 1 − IoU (1 without a partner), averaged over the reference speakers; a track speaker without a partner adds nothing |
| `tracks[].overlap_precision`, `overlap_recall` | time both mark two or more speakers over the time the track marks it, over the time the reference marks it (all frames, no collar; `null` when the denominator is 0) |
| `tracks[].speaker_count_error` | the track's speakers minus the reference's (each with at least 1 s of speech) |
| `transcripts[].wer` | `compare_transcripts` against the reference words (same normalisation as §4); only when the corpus has words |
| `pipelines[]` | each pipeline's `der`, `jer`, overlap, speaker-count error (its track's) and `wer` (its transcript's); `licence_commercial` false when a model **or the corpus** is non-commercial |
| `licence`, `licence_commercial`, `source_url` | the corpus's; `licence_commercial: false` flags CALLHOME's scores as internal-only |

The 10-second example of the tests (reference A 0–4 s, B 3–8 s; a track
with three anonymous voices) gives by hand DER 3.2/9 without a collar
and 2.5/7 with the 0.25 s collar, JER 0.275 without a collar and
0.2632 with it ((0 + 1 − 2.25/4.75) / 2), overlap precision 1.0 and
recall 0.3, and a speaker-count error of +1
(`tests/unit/transcriber/test_lab_reference.py`).

`lab summary` prints these per pipeline; the web overview adds DER, JER
and WER columns (sorted by DER) and says which corpus and licence the
recording comes from.

### 13.3 Over the corpus

```sh
uv run --no-sync python -m transcriber lab reference-summary [--corpus ami] [--json]
```

takes, per imported session, its newest `done` or `partial` run with
reference scores and gives the mean ± sample SD (with min and max in
`--json`) over the recordings of: per track (denoiser + diarizer) DER,
missed / false alarm / confusion as shares of the reference speech,
JER, overlap precision and recall and the speaker-count error (signed
and absolute); per transcript the WER; per pipeline DER, JER and WER.
Sessions without a scored run are listed apart; a non-commercial corpus
in the mix is flagged. Running the full suite on the corpus is a
separate experiment.

### 13.4 Smoke check (2026-11-14, one AMI recording)

Not an experiment, only a check that import and scoring work on the
real corpus. `lab import-reference ami --limit 1` downloaded the first
test shard (328 MB) and imported `ami-EN2002b` (1786.8 s, 4 speakers,
490 turns) in 75 s. A run with two diarizers on the original audio
took 60 s in all. Against the reference (1476.8 s of speech, 389.9 s
of overlap, collar 0.25 s):

| Track | DER | missed | false alarm | confusion | JER | overlap P | overlap R | speaker count error |
|---|---|---|---|---|---|---|---|---|
| `pyannote-community-1` | 0.1819 | 0.1144 | 0.0275 | 0.0400 | 0.2302 | 0.879 | 0.645 | 0 |
| `sortformer-streaming-v2` | 0.2907 | 0.2251 | 0.0356 | 0.0300 | 0.3266 | 0.894 | 0.481 | 0 |

(missed, false alarm and confusion as shares of the reference speech.)
One recording says nothing general; the corpus-wide run with every
model is the follow-up experiment.
