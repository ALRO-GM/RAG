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
again.

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
  work/<random>/             stages being computed
```

Ids and keys are checked against their patterns before a path is made;
JSON is written atomically; a stage's directory appears in one rename.
A run records the process that queued or runs it: a `queued` or
`running` run whose process is gone is reported `interrupted` and can be
resumed. Cancelling leaves a marker file the running process checks
between steps and while a runner works (it then kills the runner).

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
   "parameters": {"num_speakers": null}, "weights": ["pyannote/speaker-diarization-3.1"],
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
may only override parameters the model declares. `language` and
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
done steps stay, the others are tried again; answers the run.

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
 "fragment_count": 812}
```

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
longest) or `time`; `limit` 1–200.

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

### `GET …/runs/{run_id}/audio/{denoiser}`

The original (`none`) or that denoiser's output as WAV (16 kHz mono),
whole or by one byte range (206), like the session audio; it also
accepts the playback cookie of `POST /api/access/cookie`, so an
`<audio>` element can play it. Play a fragment by seeking to its start.

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

## 8. CLI

```sh
python -m transcriber lab catalogue [--json]
python -m transcriber lab run SESSION [--all] [--denoise a,b] [--diarize x,y] \
    [--recognise r] [--language es] [--speakers 2] [--params JSON] [--json]
python -m transcriber lab runs SESSION [--json]
python -m transcriber lab summary SESSION RUN [--json]
python -m transcriber lab resume SESSION RUN [--json]
```

`run` runs in the command's own process (its own job queue) and prints
aggregate numbers only. It uses the GPU: do not start it while `serve`
is working on a job. It does not take `serve.lock` and writes only under
`<data_dir>/lab`, so it may run beside a running service.
