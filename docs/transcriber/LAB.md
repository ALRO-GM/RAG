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
- A recording with no speech (silence, or noise alone) is a diarization
  with no segments, not a failure. pyannote, Sortformer and Nemotron
  already answer so; `diarizen-wavlm-large-s80-md-v2` and
  `nemo-clustering` aborted (#48, §13.5) and now return no segments
  with a note (`no speech found…`). Only that path changed, so a
  recording with speech gives the same result: both keep revision 1.
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

### `GET /api/lab/overview`

Every session with a lab run or a corpus reference (§13), by id, with
its newest run's steps counted for the tab's progress view:

```json
{"sessions": [{"session_id": "ami-EN2002b", "title": "…", "duration": 1786.8,
  "corpus": "ami", "licence": "CC BY 4.0", "licence_commercial": true,
  "run_id": "8f…", "run_state": "partial", "created_at": "…",
  "finished_at": "…", "message": null, "results_run": "8f…",
  "excluded": null,
  "steps": {"done": 15, "running": 0, "pending": 0, "failed": 14,
            "failed_codes": {"failed": 10, "out_of_memory": 4},
            "open": [{"name": "diarize:none+…", "stage": "diarize",
                      "model": "…", "denoiser": "none", "state": "failed",
                      "code": "out_of_memory", "message": "…"}]}}]}
```

`corpus` is `null` for an own recording; `excluded` says why an
imported recording is left out of the results (`no speech in the
audio`, §13.6), `null` when it is used; `run_*` and `steps` are `null`
without a run; `results_run` is the newest `done` or `partial` run (the
one with results). `run_state` is reported as in the runs list
(`interrupted` when its process is gone). A step `running` in a run
that is no longer queued or running counts as `pending`, and so does a
`cancelled` step: a resume would try both again. `open` lists every
step not done, with its error code and message when it failed.

### `GET /api/lab/reference-summary?corpus=ami`

The summary of §13.3 (`lab reference-summary --json`) over the
imported sessions of one corpus (all when `corpus` is omitted; an
unknown one is 400 `invalid_input`): `corpora`, `non_commercial`,
`recordings`, `unscored`, `excluded` (sessions left out: `session_id`,
`corpus`, `recording`, `reason`, `speech` figures, §13.6) and `models` (`kind` `diarization`,
`recognition` or `pipeline`; each metric's `n`, `mean`, `sd`, `min`,
`max`).

### `GET /api/lab/sessions/{session_id}/reference-turns`

An imported session's reference speaker turns (anonymous corpus ids,
overlap kept, no words): `{"turns": [{"start", "end", "speaker"}]}`;
404 without a reference. The tab picks its audio samples from them.

### `GET …/runs/{run_id}/excerpt/{denoiser}?start=120.5&seconds=20`

A stretch of the original (`none`) or of a denoiser's output as a
short WAV (16 kHz mono), cut on the server so the phone downloads only
those seconds (20 s: 640 044 bytes). `start` ≥ 0 and before the end of
the audio (else 400 `invalid_input`), `seconds` more than 0 and at
most 60 (default 20; else 422); cut at the end of the audio. Read-only,
with byte ranges, and like the audio route it accepts the playback
cookie.

### The "Pruebas" tab

The web pages have a tab for all of this (`#/lab`, behind the access
code; texts in Spanish and English). It opens on an overview across
sessions (`web/js/views/lab-progress.js`, pure logic in
`web/js/lib/lab-progress.js`, tests in `tests/web/lab-progress.test.mjs`),
refreshed every 10 s while a run is queued or running and the page is
visible: totals (steps done and left, runs running) and one stacked bar
per session grouped by corpus (own, AMI, CALLHOME), failures split by
error code; the results so far (per corpus, plain-SVG charts of DER per
diarizer on the chosen audio, mean with min-max over the recordings, and
per denoiser, the mean over the diarizers that ran on every denoiser
with the best-to-worst range; CALLHOME flagged non-commercial; for own
recordings, on request, agreement with the consensus, flagged as
agreement only); the pending and failed steps with their reasons and
how to resume; and, per session, 20-second samples of the original and
each denoised audio of its newest run with results, through the excerpt
route, beside that audio's mean and best DER (corpus) or mean agreement
(own). The stretch is the 20 s window with most overlapped speech
(counted twice) plus speaker changes in the reference turns (corpus) or
the consensus (own), else from 10 % in; on a corpus recording the
consensus names nobody (no one is enrolled), so agreement is not shown
there. Below the overview: pick a recording, choose models by
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
python -m transcriber lab import-reference ami|callhome --check [--json]
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

Scored against the public references (§13): the lab suite ran on four AMI
meetings and ten CALLHOME Spanish calls and its results are in §13.8
(#44). Still no scores on a recording of our own.

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
  can simply be run again. A recording whose audio holds no speech is
  not imported (§13.6): it is reported as `skipped: no speech in the
  audio` with the check's figures, and `--limit N` reads on until N
  usable recordings are in (sessions already imported count unless they
  failed the check). A recording whose reference is invalid (a turn
  that ends before it starts, no turn at all...) is not imported either
  (§13.7): it is reported as `skipped: invalid reference (…)`, and
  counts the same way. A gated corpus whose terms were not accepted
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
in the mix is flagged. Imported sessions whose audio failed the speech
check (§13.6) count nowhere: they are listed in `excluded` (session,
reason, figures) and the text says how many were left out and why
(`1 excluded: no speech in the audio.`); the web overview leaves them
out of the progress bars and the results and says the same. Running the full suite on the corpus is a
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

### 13.5 CALLHOME recordings with no speech in them (#48)

In #44 `diarizen-wavlm-large-s80-md-v2` failed on `callhome-data-0002`
and `-0003` ("negative dimensions are not allowed", on the original and
the DeepFilterNet3 audio) and `nemo-clustering` too ("All files present
in manifest contains silence", on the original and the MossFormer2
audio), while `callhome-data-0001` passed.

**The audio of those recordings has no speech.** The corpus row's WAV
is already 16 kHz mono 16-bit (the lab stores it as given and changes
nothing), but its level is a constant noise: 20 ms frame RMS from
0.18 (5th percentile) to 0.27 (95th), against 0.0003 to 0.08 for
`callhome-data-0001`, with a flat spectrum up to 4 kHz (within 2.1 dB
in 1 kHz bands) and nothing above. pyannote 3.1 and community-1,
Nemotron and ECAPA find no segment on it, and Whisper's whole
transcript is "Gracias por ver el video." five times (once every 80 s),
the usual hallucination on no speech.
Of the 28 recordings of the first `spa/data` shard, 7 look like this
(rows 2, 3, 9, 15, 18, 23 and 28; 5th-percentile frame RMS above 0.17,
the others below 0.007); their reference still has speaker turns
(144 on row 2), so every diarizer's DER on them is about 1 and says
nothing about the diarizer. Why the dataset's audio is noise for those
rows was not investigated (not a byte-order or two-channel μ-law
misreading: neither gives speech-like samples).

**Why the two diarizers aborted.** With no speaker anywhere, DiariZen
marks every local speaker of every chunk inactive (cluster −2); its
pyannote fork then sizes the reconstruction with
`max(cluster) + 1 = −1` clusters (`speaker_diarization.py`,
`reconstruct`) and numpy refuses the shape. NeMo's VAD keeps no
stretch of speech and `validate_vad_manifest` aborts the run. The same
happens on the denoised audio, which is near silence (DeepFilterNet3
lowers row 2 from an RMS of 0.227 to 0.0014).

**Fix (runners, any input).** `diarizen.py` wraps the pipeline's
`reconstruct`: when no hard cluster is 0 or more it stops, and the
runner returns no segments with the note "no speech found: every
speaker of every chunk is inactive". `nemo.py` catches the
`ValueError` of `diarize()` only when the VAD manifest
(`vad_outputs/vad_out.json`) exists and has no stretch longer than
0 s, and returns no segments with the note "no speech found by the
VAD"; any other failure is raised as before. A recording with speech
never reaches either path, so its result is the same and the catalogue
revisions stay 1. Tests (no models,
`tests/unit/transcriber/test_lab_runner_guards.py`): fakes of both
pipelines that fail as the real ones do on synthetic noise (the
DiariZen fake raises the same numpy error), the runners' empty result
and note on it, an unchanged result on a synthetic tone, and a NeMo
failure after speech that is still raised.

Checked on the CPU with the real models on the stored
`callhome-data-0002` audio: before the fix DiariZen failed with the
error of #44 in `reconstruct`, NeMo with the "silence" error; after
it NeMo returns no segments and the note on the original audio of
`-0002` (20.3 s of processing) and the MossFormer2 audio of `-0003`
(60.2 s), and DiariZen returns no segments and its note on the original audio
of `-0002` (400.0 s of processing on the CPU). Neither denoised
DiariZen input nor a GPU run was repeated.

### 13.6 Recordings with no speech (2026-11-14)

Found while running the suite on CALLHOME (#48, §13.5): 7 of the first 28
rows of `diarizers-community/callhome` (`spa`, shard 0: rows 2, 3, 9,
15, 18, 23 and 28, sessions `callhome-data-0002`...) hold steady
broadband noise up to 4 kHz instead of speech, while their reference
says people talk. Every diarizer finds nobody there, so their DER of
about 1 measures the data, not the models. The import now checks the
decoded audio first (`transcriber/domain/speech_check.py`, numpy only):

- frames of 32 ms (512 samples at 16 kHz); only the loudest 30 % are
  measured (where speech would be);
- per frame, the **lag-1 autocorrelation** (mean removed) and the
  **spectral flatness** from the first bin above 0 Hz to 4 kHz
  (geometric over arithmetic mean of the Hann-windowed power
  spectrum); the recording's figure is the median over those frames;
- **no speech** when the autocorrelation is below **0.8** *and* the
  flatness is above **0.2** (both must say noise, so a noisy recording
  with a voice in it is not dropped).

Measured on the decoded audio (16 kHz mono, as the lab reads it) of
the 28 rows of CALLHOME shard 0 and the 6 recordings of the first AMI
`test` shard:

| Recordings | lag-1 autocorrelation | spectral flatness (0–4 kHz) | Verdict |
|---|---|---|---|
| CALLHOME rows 2, 3, 9, 15, 18, 23, 28 | 0.6666–0.6866 | 0.3939–0.3998 | no speech (all 7) |
| CALLHOME, the other 21 rows | 0.9011–0.9847 | 0.0039–0.0311 | speech |
| AMI EN2002b, TS3003a, ES2004c, ES2004b, TS3003b, IS1009a | 0.9684–0.9899 | 0.0065–0.0219 | speech |

Per CALLHOME row (autocorrelation / flatness): 1 0.9270/0.0225,
**2 0.6866/0.3952**, **3 0.6759/0.3949**, 4 0.9227/0.0111,
5 0.9096/0.0289, 6 0.9287/0.0199, 7 0.9291/0.0169, 8 0.9192/0.0311,
**9 0.6773/0.3958**, 10 0.9786/0.0082, 11 0.9705/0.0055,
12 0.9011/0.0121, 13 0.9753/0.0052, 14 0.9507/0.0200,
**15 0.6823/0.3939**, 16 0.9568/0.0166, 17 0.9500/0.0084,
**18 0.6823/0.3968**, 19 0.9490/0.0109, 20 0.9350/0.0071,
21 0.9760/0.0105, 22 0.9847/0.0042, **23 0.6853/0.3948**,
24 0.9676/0.0079, 25 0.9346/0.0146, 26 0.9585/0.0063,
27 0.9705/0.0039, **28 0.6666/0.3998**. AMI: EN2002b 0.9797/0.0147,
TS3003a 0.9899/0.0065, ES2004c 0.9796/0.0156, ES2004b 0.9827/0.0151,
TS3003b 0.9831/0.0120, IS1009a 0.9684/0.0219.

The check flags exactly the seven rows found in #48 and none of
the others; the gap between the groups is wide on both figures
(autocorrelation 0.69 against 0.90, flatness 0.40 against 0.03), and
the thresholds sit in the middle of it. The noise rows are also flat in
level (the 90th and 10th percentiles of frame energy within 2.0–2.5 dB
of each other, against 13.2 dB or more for every speech recording),
which the check does not use. The check took 0.02–0.05 s per CALLHOME
row and 0.06–0.18 s per AMI meeting on this machine's CPU, after
decoding. 34 recordings of two corpora are all it was calibrated on:
a corpus with another kind of noise may need another look.

Sessions imported before the check have no figures (`speech: null` in
`reference.json`) and are used as before until checked:

```sh
TRANSCRIBER_DATA_DIR=data/transcriber \
  uv run --no-sync python -m transcriber lab import-reference callhome --check
```

decodes each imported session of the corpus again (no download), stores
the figures in its `reference.json` (`speech`: `autocorrelation`,
`flatness`, `frames`, `speech`) and prints `<session>: unusable: no
speech in the audio (…)` for the ones that fail. Nothing is deleted: a
failing session stays, and `lab reference-summary`, `GET
/api/lab/reference-summary` (`excluded`) and the overview (`excluded`
per session, §7) leave it out and say why.

### 13.7 Rows with an invalid reference (#53, 2026-11-15)

`lab import-reference callhome --limit 10` stopped the whole import at
row 12 of `diarizers-community/callhome` (`spa`, shard 0, session
`callhome-data-0012`): `Invalid reference turn 342.684--56.709999999999994`.
Read from the shard (the timestamp columns only), that row has 188
turns, 2 of them ending before they start: speaker `B2` at 342.684 s
"to" -56.71 s and speaker `A` at 348.964 s "to" -48.89 s; the turns
around them are ordinary (B2 338.19–340.71 s, A 345.06–348.67 s). The
other 27 rows of the shard have no such turn. The right end of those
two turns cannot be recovered from the row, and dropping or guessing
them would score the diarizers against a reference with holes, so the
whole row is skipped:

- the import builds and checks the reference before the speech check;
  a new recording whose reference is refused (a turn that ends before
  it starts or is not a number, columns of different lengths, a turn
  without a speaker, no turn at all) gets no session and is reported
  as `<session>: skipped: invalid reference (<what is wrong>)`, with
  `speech: null` in `--json`;
- `--limit N` reads on until N usable recordings are in, as for the
  recordings without speech (§13.6), and the last line counts both
  (`…; 1 with an invalid reference; 3 without speech.`).

With the shard already downloaded, `lab import-reference callhome
--limit 9` into an empty data directory now reports rows 2, 3 and 9 as
without speech, row 12 as `skipped: invalid reference (Invalid
reference turn from 342.684 s to -56.71 s)` and imports rows 1, 4–8,
10, 11 and 13.

### 13.8 Results of #44 (2026-11-15)

The suite of §13 ran on four AMI meetings and ten CALLHOME Spanish
calls, scored against the human reference. The figures below come from
`lab reference-summary --corpus ami --json` and `--corpus callhome
--json` (aggregates only; the audio and the runs stay in the private
data directory, §6).

**Setup.**

| | AMI | CALLHOME Spanish |
|---|---|---|
| Recordings | 4 meetings: EN2002b (1787 s), ES2004b (2345 s), ES2004c (2334 s), TS3003a (1506 s) | 10 calls of 300–605 s (`data-0001`, `0004`–`0008`, `0010`, `0011`, `0013`, `0014`) |
| Language, audio | English, single distant microphone (`sdm`) | Spanish, telephone |
| Speakers in the reference | 4 | 2 |
| Left out | none | `0002`, `0003`, `0009` (no speech, §13.5, §13.6); `0012` (invalid reference, §13.7) |
| Licence of the data | CC BY 4.0 | CC BY-NC-SA 4.0: results are non-commercial |

- Suite: denoisers `none`, `deepfilternet3` (revision 2, 30 s pieces,
  #45) and `mossformer2-se-48k` × the 8 diarizers of §9 ×
  `whisper-large-v3-turbo`. Every diarizer estimates the number of
  speakers; none is told it. DER and JER as in §13.2 (collar 0.25 s).
- Neither corpus has reference words, so there is **no WER** (the `wer`
  entry of the pipeline rows has no value). The recogniser does not
  change the speaker turns: the DER of a pipeline on `none` equals the
  DER of its diarizer (checked for all 8 on both corpora), so the
  tables use the diarizer rows (24 per corpus).
- Each figure is the mean over recordings (SD between recordings,
  `n` = recordings). Missed, false alarm and confusion are fractions of
  the reference speech time and add up to the DER. "Speakers" is the
  estimated number minus the true one, averaged (positive = too many),
  and in brackets the mean absolute difference. Overlap P / R are
  precision and recall of the overlapped-speech regions; "–" for
  precision means the diarizer marked no overlap at all (recall 0).
- The GPU was shared with other jobs during the run; out-of-memory on
  `deepfilternet3` was handled by the 30 s pieces (#45) and by a
  supervisor that resumed the run. On AMI one recording has no
  `deepfilternet3` + `builtin` result (n = 3 in that row, and 3 recordings
  in every `deepfilternet3` pipeline row); the cause was not looked at.
  The row stays in the table with its n.

#### 13.8.1 AMI (4 meetings, 4 speakers each)

| # | denoiser | diarizer | n | DER (mean ± SD) | missed | false alarm | confusion | JER | overlap P / R | speakers: error (abs. error) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | none | diarizen (NC) | 4 | 0.089 ± 0.036 | 0.052 | 0.022 | 0.016 | 0.178 | 0.82 / 0.73 | +1.00 (1.00) |
| 2 | none | pyannote-3.1 | 4 | 0.130 ± 0.058 | 0.078 | 0.023 | 0.029 | 0.264 | 0.81 / 0.47 | +0.00 (1.00) |
| 3 | none | pyannote-community-1 | 4 | 0.134 ± 0.063 | 0.078 | 0.023 | 0.033 | 0.271 | 0.81 / 0.47 | +0.25 (1.25) |
| 4 | mossformer2-se-48k | pyannote-3.1 | 4 | 0.176 ± 0.094 | 0.132 | 0.020 | 0.024 | 0.297 | 0.76 / 0.25 | +0.00 (1.00) |
| 5 | mossformer2-se-48k | pyannote-community-1 | 4 | 0.181 ± 0.094 | 0.132 | 0.020 | 0.029 | 0.277 | 0.76 / 0.25 | +0.50 (1.00) |
| 6 | mossformer2-se-48k | diarizen (NC) | 4 | 0.190 ± 0.126 | 0.163 | 0.009 | 0.018 | 0.252 | 0.87 / 0.41 | +2.00 (2.00) |
| 7 | mossformer2-se-48k | nemo-clustering | 4 | 0.277 ± 0.184 | 0.214 | 0.020 | 0.043 | 0.376 | – / 0.00 | -0.50 (0.50) |
| 8 | none | nemotron-3 | 4 | 0.297 ± 0.076 | 0.257 | 0.008 | 0.032 | 0.393 | 0.86 / 0.40 | +0.00 (0.00) |
| 9 | none | sortformer-v2 | 4 | 0.333 ± 0.097 | 0.218 | 0.017 | 0.098 | 0.443 | 0.79 / 0.42 | +0.00 (0.00) |
| 10 | mossformer2-se-48k | sortformer-v2 | 4 | 0.338 ± 0.119 | 0.247 | 0.009 | 0.082 | 0.434 | 0.80 / 0.21 | +0.00 (0.00) |
| 11 | mossformer2-se-48k | nemotron-3 | 4 | 0.346 ± 0.054 | 0.290 | 0.006 | 0.050 | 0.436 | 0.87 / 0.25 | +0.25 (0.25) |
| 12 | mossformer2-se-48k | ecapa-spectral | 4 | 0.367 ± 0.203 | 0.156 | 0.136 | 0.075 | 0.407 | – / 0.00 | +0.00 (1.00) |
| 13 | none | nemo-clustering | 4 | 0.425 ± 0.175 | 0.404 | 0.005 | 0.016 | 0.491 | – / 0.00 | -0.75 (0.75) |
| 14 | none | builtin | 4 | 0.498 ± 0.276 | 0.209 | 0.053 | 0.236 | 0.536 | 0.26 / 0.00 | +5.50 (5.50) |
| 15 | none | ecapa-spectral | 4 | 0.521 ± 0.124 | 0.437 | 0.038 | 0.046 | 0.561 | – / 0.00 | +0.00 (1.00) |
| 16 | mossformer2-se-48k | builtin | 4 | 0.555 ± 0.212 | 0.272 | 0.027 | 0.256 | 0.605 | 0.06 / 0.00 | +4.25 (4.25) |
| 17 | deepfilternet3 | pyannote-3.1 | 4 | 0.760 ± 0.200 | 0.743 | 0.004 | 0.013 | 0.750 | 0.84 / 0.08 | +0.25 (0.25) |
| 18 | deepfilternet3 | pyannote-community-1 | 4 | 0.774 ± 0.188 | 0.743 | 0.004 | 0.028 | 0.765 | 0.84 / 0.08 | -0.50 (1.00) |
| 19 | deepfilternet3 | diarizen (NC) | 4 | 0.775 ± 0.200 | 0.760 | 0.001 | 0.014 | 0.771 | 0.94 / 0.10 | -1.00 (1.00) |
| 20 | deepfilternet3 | ecapa-spectral | 4 | 0.777 ± 0.142 | 0.648 | 0.020 | 0.109 | 0.769 | – / 0.00 | -0.50 (1.00) |
| 21 | deepfilternet3 | nemo-clustering | 4 | 0.778 ± 0.193 | 0.741 | 0.004 | 0.033 | 0.767 | – / 0.00 | +0.00 (1.50) |
| 22 | deepfilternet3 | sortformer-v2 | 4 | 0.796 ± 0.181 | 0.787 | 0.001 | 0.008 | 0.788 | 0.92 / 0.08 | -1.00 (1.00) |
| 23 | deepfilternet3 | nemotron-3 | 4 | 0.799 ± 0.172 | 0.789 | 0.001 | 0.009 | 0.790 | 0.92 / 0.08 | -1.00 (1.00) |
| 24 | deepfilternet3 | builtin | 3 | 0.844 ± 0.077 | 0.745 | 0.008 | 0.090 | 0.825 | 0.20 / 0.00 | +4.33 (4.33) |

| diarizer | none | deepfilternet3 | mossformer2-se-48k |
|---|---|---|---|
| diarizen (NC) | 0.089 | 0.775 | 0.190 |
| pyannote-3.1 | 0.130 | 0.760 | 0.176 |
| pyannote-community-1 | 0.134 | 0.774 | 0.181 |
| nemotron-3 | 0.297 | 0.799 | 0.346 |
| sortformer-v2 | 0.333 | 0.796 | 0.338 |
| nemo-clustering | 0.425 | 0.778 | 0.277 |
| builtin | 0.498 | 0.844 | 0.555 |
| ecapa-spectral | 0.521 | 0.777 | 0.367 |

Top 5 by DER: DiariZen on the original (0.089), pyannote-3.1 (0.130),
pyannote-community-1 (0.134) and the same two on `mossformer2-se-48k`
(0.176, 0.181).

#### 13.8.2 CALLHOME Spanish (10 calls, 2 speakers each)

| # | denoiser | diarizer | n | DER (mean ± SD) | missed | false alarm | confusion | JER | overlap P / R | speakers: error (abs. error) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | none | nemotron-3 | 10 | 0.106 ± 0.041 | 0.075 | 0.019 | 0.011 | 0.119 | 0.82 / 0.61 | +0.20 (0.20) |
| 2 | none | sortformer-v2 | 10 | 0.122 ± 0.068 | 0.073 | 0.022 | 0.027 | 0.149 | 0.77 / 0.63 | +0.20 (0.40) |
| 3 | none | diarizen (NC) | 10 | 0.163 ± 0.048 | 0.145 | 0.010 | 0.008 | 0.174 | 0.89 / 0.45 | +0.10 (0.10) |
| 4 | mossformer2-se-48k | sortformer-v2 | 10 | 0.174 ± 0.066 | 0.123 | 0.011 | 0.040 | 0.216 | 0.79 / 0.36 | +0.40 (0.80) |
| 5 | mossformer2-se-48k | diarizen (NC) | 10 | 0.188 ± 0.040 | 0.173 | 0.007 | 0.009 | 0.201 | 0.90 / 0.29 | +0.10 (0.10) |
| 6 | mossformer2-se-48k | nemo-clustering | 10 | 0.193 ± 0.051 | 0.154 | 0.015 | 0.025 | 0.218 | – / 0.00 | +0.00 (0.00) |
| 7 | mossformer2-se-48k | nemotron-3 | 10 | 0.194 ± 0.057 | 0.161 | 0.007 | 0.027 | 0.236 | 0.89 / 0.28 | -0.10 (0.10) |
| 8 | none | pyannote-community-1 | 10 | 0.195 ± 0.046 | 0.151 | 0.015 | 0.028 | 0.217 | 0.81 / 0.39 | +0.00 (0.00) |
| 9 | none | nemo-clustering | 10 | 0.211 ± 0.063 | 0.160 | 0.011 | 0.040 | 0.226 | – / 0.00 | +0.30 (0.30) |
| 10 | mossformer2-se-48k | pyannote-community-1 | 10 | 0.230 ± 0.033 | 0.192 | 0.010 | 0.028 | 0.254 | 0.78 / 0.19 | +0.00 (0.00) |
| 11 | mossformer2-se-48k | ecapa-spectral | 10 | 0.243 ± 0.090 | 0.195 | 0.005 | 0.043 | 0.298 | – / 0.00 | -0.20 (0.20) |
| 12 | none | pyannote-3.1 | 10 | 0.248 ± 0.126 | 0.151 | 0.015 | 0.081 | 0.297 | 0.81 / 0.39 | -0.20 (0.20) |
| 13 | none | ecapa-spectral | 10 | 0.248 ± 0.105 | 0.201 | 0.006 | 0.042 | 0.304 | – / 0.00 | -0.20 (0.20) |
| 14 | mossformer2-se-48k | pyannote-3.1 | 10 | 0.283 ± 0.116 | 0.192 | 0.010 | 0.081 | 0.332 | 0.78 / 0.19 | -0.20 (0.20) |
| 15 | deepfilternet3 | diarizen (NC) | 10 | 0.287 ± 0.120 | 0.246 | 0.004 | 0.036 | 0.316 | 0.94 / 0.21 | -0.10 (0.10) |
| 16 | deepfilternet3 | nemo-clustering | 10 | 0.289 ± 0.168 | 0.174 | 0.013 | 0.102 | 0.349 | – / 0.00 | -0.30 (0.50) |
| 17 | deepfilternet3 | nemotron-3 | 10 | 0.298 ± 0.103 | 0.256 | 0.004 | 0.039 | 0.351 | 0.94 / 0.16 | -0.10 (0.30) |
| 18 | deepfilternet3 | sortformer-v2 | 10 | 0.314 ± 0.138 | 0.211 | 0.007 | 0.096 | 0.365 | 0.82 / 0.19 | +0.30 (0.70) |
| 19 | deepfilternet3 | pyannote-community-1 | 10 | 0.361 ± 0.151 | 0.252 | 0.008 | 0.100 | 0.419 | 0.75 / 0.11 | -0.30 (0.30) |
| 20 | deepfilternet3 | pyannote-3.1 | 10 | 0.381 ± 0.165 | 0.252 | 0.008 | 0.121 | 0.472 | 0.75 / 0.11 | -0.40 (0.40) |
| 21 | deepfilternet3 | ecapa-spectral | 10 | 0.395 ± 0.158 | 0.265 | 0.003 | 0.128 | 0.513 | – / 0.00 | -0.90 (0.90) |
| 22 | none | builtin | 10 | 0.487 ± 0.146 | 0.189 | 0.013 | 0.284 | 0.488 | 0.06 / 0.00 | +6.40 (6.40) |
| 23 | mossformer2-se-48k | builtin | 10 | 0.506 ± 0.153 | 0.199 | 0.015 | 0.292 | 0.507 | 0.11 / 0.00 | +7.00 (7.00) |
| 24 | deepfilternet3 | builtin | 10 | 0.586 ± 0.186 | 0.282 | 0.012 | 0.291 | 0.584 | 0.21 / 0.00 | +6.50 (6.50) |

| diarizer | none | deepfilternet3 | mossformer2-se-48k |
|---|---|---|---|
| nemotron-3 | 0.106 | 0.298 | 0.194 |
| sortformer-v2 | 0.122 | 0.314 | 0.174 |
| diarizen (NC) | 0.163 | 0.287 | 0.188 |
| pyannote-community-1 | 0.195 | 0.361 | 0.230 |
| nemo-clustering | 0.211 | 0.289 | 0.193 |
| pyannote-3.1 | 0.248 | 0.381 | 0.283 |
| ecapa-spectral | 0.248 | 0.395 | 0.243 |
| builtin | 0.487 | 0.586 | 0.506 |

Top 5 by DER: Nemotron on the original (0.106), Sortformer (0.122),
DiariZen (0.163), then Sortformer and DiariZen on `mossformer2-se-48k`
(0.174, 0.188).

#### 13.8.3 Effect of the denoiser

Mean DER over the 8 diarizers (each diarizer counts once; the same
rows as the tables):

| Denoiser | AMI | CALLHOME |
|---|---|---|
| none | 0.304 | 0.223 |
| mossformer2-se-48k | 0.304 | 0.251 |
| deepfilternet3 | 0.788 | 0.364 |

- `mossformer2-se-48k` is better than the original for only 2 of 8
  diarizers on each corpus (AMI: ECAPA 0.521 → 0.367, NeMo clustering
  0.425 → 0.277; CALLHOME: ECAPA 0.248 → 0.243, NeMo clustering 0.211 →
  0.193) and worse for the rest. The equal AMI means hide that: the
  two big gains are on the two weakest diarizers, the best ones lose
  (DiariZen 0.089 → 0.190, pyannote-3.1 0.130 → 0.176, Nemotron 0.297 →
  0.346). On the original the best commercial option is always better
  than the best option on any denoised audio.
- On AMI `deepfilternet3` leaves most of the speech unlabelled: the
  missed rate is 0.74–0.79 for seven of the eight diarizers (0.648 for
  ECAPA) with false alarm and confusion at or
  below 0.03 for the overlap-aware models. Consistent with §12, where
  the output on one minute of TS3003a was almost silent at the
  recording's level: the denoiser treats far-field speech as noise.
  On the telephone calls it is milder (DER 0.287–0.395 for the seven,
  0.586 for the built-in) but worse than the original for every
  diarizer.
- With less overlap marked on the denoised audio the overlap recall
  falls (pyannote 0.47 → 0.25 with `mossformer2-se-48k` and 0.08 with
  `deepfilternet3` on AMI; Sortformer 0.63 → 0.36 → 0.19 on CALLHOME),
  while precision stays at 0.75–0.94.

#### 13.8.4 Licence column

| Diarizer | Licence (catalogue) | Commercial use |
|---|---|---|
| builtin | project | yes |
| pyannote-3.1 | MIT (gated) | yes |
| pyannote-community-1 | CC-BY-4.0 (gated) | yes |
| sortformer-streaming-v2 | CC-BY-4.0 | yes |
| nemotron-3-diarization | OpenMDW-1.1 (card: commercial use allowed) | yes |
| diarizen-wavlm-large-s80-md-v2 | CC-BY-NC-4.0 (weights); MIT (code) | **no** |
| ecapa-spectral | Apache-2.0 (weights) | yes |
| nemo-clustering | CC-BY-4.0 (TitaNet-L); NeMo VAD weights per NGC | yes |

Denoisers: `deepfilternet3` MIT OR Apache-2.0, `mossformer2-se-48k`
Apache-2.0. The best AMI result is therefore not usable in a product;
the best commercial one is pyannote-3.1 (0.130). The CALLHOME audio is
CC BY-NC-SA, so using it for evaluation is allowed here but the data
cannot go into a product; the summary carries `non_commercial: true`.

#### 13.8.5 Findings

1. **The ranking flips between corpora.** On the original audio the
   order on AMI is DiariZen 0.089, pyannote 0.130 / 0.134, Nemotron
   0.297, Sortformer 0.333; on CALLHOME it is Nemotron 0.106,
   Sortformer 0.122, DiariZen 0.163, pyannote-community-1 0.195. Nemotron
   and Sortformer, best on two-person calls, have the highest missed
   rates among the overlap-aware models on the meetings (0.257 and
   0.218; Sortformer also confusion 0.098) and count exactly 4
   speakers on every meeting (error 0.00), so the loss is missed speech,
   not the speaker count. DiariZen is first on AMI and third on
   CALLHOME. pyannote-community-1 is third of eight on AMI and fourth
   on CALLHOME, the most stable of the commercial options;
   pyannote-3.1 is second on AMI but sixth (tied with ECAPA) on CALLHOME
   (0.248, SD 0.126, worst call 0.479; confusion 0.081 against 0.028 for
   community-1). Nothing here says which will rank first on our own
   home recordings.
2. **Denoising does not help on average**, and the denoiser that helps
   only the weakest diarizers does not change which diarizer is best
   (§13.8.3). `deepfilternet3` is the worst on both corpora and removes
   most of the speech on the quiet distant-microphone meetings.
3. **The built-in diarizer is the worst of the eight on CALLHOME and
   second worst on AMI** (DER 0.487 on CALLHOME and 0.498 on AMI on the
   original, only ECAPA 0.521 is worse on AMI). Its confusion (0.28 and
   0.24) is far above the next highest on the original (0.081 on CALLHOME,
   0.098 on AMI) because it over-counts: +6.4 speakers on two-person calls, +5.5 on four-person
   meetings. It marks almost no overlap (precision 0.06 and 0.26, recall
   below 0.001).
4. **The overlap detectors catch about half or less of the overlap**:
   recall 0.73 (DiariZen) and 0.47 (pyannote) on AMI, 0.61–0.63
   (Nemotron, Sortformer) and 0.39 (pyannote) on CALLHOME; precision is
   0.77–0.89 for all of them. ECAPA, NeMo clustering and the built-in
   do not model overlap, so their overlap recall is 0.
5. **Speaker count.** On two-person calls every model except the
   built-in is within 0.4 speakers on average (|error| 0.00–0.40 on the
   original). On AMI only Nemotron and Sortformer are exact; DiariZen
   over-counts by 1 on the original and by 2 on `mossformer2-se-48k`.

#### 13.8.6 Against §11 (R1, agreement between models)

§11 could only measure agreement. The pipelines that agreed most there
were on denoised audio (`mossformer2-se-48k` with pyannote, DiariZen or
Nemotron, 0.90–0.92) and the original was the best only for DiariZen
(0.859; §11.5). Against a human reference, `mossformer2-se-48k` is
**worse** than the original for those three diarizers on both corpora
(AMI: pyannote 0.130 → 0.176 / 0.181, DiariZen 0.089 → 0.190, Nemotron
0.297 → 0.346; CALLHOME: pyannote-community-1 0.195 → 0.230, DiariZen
0.163 → 0.188, Nemotron 0.106 → 0.194). So the agreement of §11 did not
predict accuracy: denoised tracks agree with each other because they
lose the same overlap and quiet speech, not because they are right,
which is the caveat §11.5 already made. R1 is a private home recording
of a different kind, so the two rankings are compared in direction
only, not figure by figure.

#### 13.8.7 Limits

- **Four meetings and ten calls**: the SD between recordings is large
  (up to 0.28 on the built-in; DiariZen on AMI 0.036 but a
  JER SD of 0.19). Differences of a few hundredths between two
  diarizers (pyannote-3.1 against community-1, the two on the same
  segmentation) are inside that noise.
- **Different data**: English far-field meetings with 4 speakers
  against Spanish telephone calls with 2. Neither is a home
  recording through a phone with several people; DER on one corpus
  does not transfer.
- **No WER**: nothing here says whether denoising helps the words, only
  the turns. The §11 word-level figures are still disagreement.
- **CALLHOME is non-commercial** (CC BY-NC-SA) and DiariZen's weights
  are CC-BY-NC: those figures are for choosing, not for shipping.
- **Collar 0.25 s**; a different collar
  changes the absolute DER (the ranking was not recomputed with other
  settings).
- The AMI `deepfilternet3` result of 0.76–0.84 measures a denoiser
  that removes the speech; it says nothing about the diarizers
  themselves.
