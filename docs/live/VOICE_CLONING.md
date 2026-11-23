# The voice bot's voice: Piper controls and operator voice cloning (VB-7, #207)

Two parts: (1) Piper's own controls on the «Ajustes» page, with a
sample button and matching an operator's speaking rate; (2) the bot
speaking with an enrolled operator's own voice, only with their
explicit, revocable consent.

**Owner's decision:** no spoken notice is played when the bot speaks
with a cloned voice. It is tracked as a compliance item in #151; this
work adds no notice.

## 1. Piper controls on «Ajustes»

New steps of `settings.v1` (group *Bot de voz*, read by the
orchestrator when a voice-bot session starts; `docs/SETTINGS.md`):

| Step | Piper flag | Accepted | Default |
|---|---|---|---|
| `voicebot.length_scale` (speed) | `--length-scale` | 0.7 to 1.5 | 1 |
| `voicebot.noise_scale` (intonation) | `--noise-scale` | 0 to 1 | 0.5 |
| `voicebot.noise_w` (cadence) | `--noise-w-scale` | 0 to 1 | 0.333 |
| `voicebot.sentence_silence` (pause) | `--sentence-silence` | 0 to 1 s | 0 |
| `voicebot.voice` | voice model | `pocket:lola`, `qwen3`, `davefx`, `sharvard`, `clone:<person>` | `pocket:lola` (#214) |
| `voicebot.match_rate` | sets the length scale | `off` or a person's id | `off` |

The defaults are the voices' own (`es_ES-davefx-medium.onnx.json` and
`es_ES-sharvard-medium.onnx.json` both say `noise_scale` 0.5, `noise_w`
0.333), so nothing saved changes nothing. Numbers are kept as text
with at most three decimals (`"1.1"`) and checked three times: the
RAG's allow-list (`src/domain/pipeline_settings.py`, `StepSpec.range`
and `pattern`), the shape check of the transcriber before forwarding,
and the orchestrator's own (`orchestrator/domain/live_choices.py`);
`tts.request.v1` gained the optional `noise_scale`, `noise_w` and
`sentence_silence` fields. On the page a number is a slider with
«Restablecer».

**«Escuchar ejemplo».** `POST /api/assist/voicebot/sample` (behind the
access code; forwarded to the orchestrator's `POST /voicebot/sample`)
with the voice bot's *unsaved* choices says a fixed Spanish sentence
(`orchestrator/domain/voices.py`, `SAMPLE_TEXT`) and answers one
`audio/wav`; the headers `X-Voice-Clone`, `X-Voice-Length-Scale` and
`X-Voice-Note` say what spoke and why not the clone.

**Matching the operator's rate.** At enrolment the transcriber now
measures the person's speaking rate on the reading: recognised words
over the seconds of speech regions (`speaking_rate`,
`transcriber/domain/enrolment.py`), stored as
`Voiceprint.words_per_second` (people enrolled before this have none
and must enrol again to get one). With `voicebot.match_rate = <id>`
the length scale becomes Piper's rate over the operator's, clamped to
0.7–1.5 (`matched_length_scale`). Piper's rate at length scale 1 was
measured on ten fixed sentences, words over voiced seconds (trimmed at
-40 dBFS): **davefx 4.065 words/s, sharvard 3.332 words/s**
(`experiments/live/piper_rate.py`, `piper_rate_results.json`). The two
rates are measured differently (speech regions from word times versus
trimmed synthetic audio); the match is approximate and has not been
checked against a listener.

## 2. Consent at enrolment

The enrolment page (`#/enrol`) has a second, optional checkbox under
the consent notice: «Además, acepto que el asistente de voz hable con
mi voz». It is a separate consent from identification:

- Ticked (`clone_consent=true`, only that exact value counts): the
  consent record says `voice_clone: true` and a clip of the reading
  (from the first to the last speech, at most 30 s) is kept as
  `<data>/voiceprints/<id>.clone.wav` (16-bit mono) beside the
  voiceprint. Without it nothing of the audio is kept, as before.
- Enrolling again without the box deletes an earlier clip.
- **Withdrawing:** «Retirar permiso de clonación» in the list of
  enrolled voices, `DELETE /api/speakers/{id}/voice-clone`: the clip is
  deleted and the record says `voice_clone: false` with
  `voice_clone_withdrawn_at`. The voiceprint stays. Forgetting the
  person deletes the clip too.
- The orchestrator reads the same two files (`fs_voice_profiles.py`,
  plain JSON and WAV, never the transcriber's code). A voice is
  clonable only with `voice_clone: true` **and** the clip;
  `plan_voice` refuses anything else and `ClonedVoice` cannot be built
  without it (tests: `test_without_consent_cloning_is_impossible`,
  `tests/unit/transcriber/test_voice_clone_consent.py`).

## 3. Which model clones: measured

`experiments/live/voice_cloning.py`, result
`experiments/live/voice_cloning_results.json` (run 2026-11-21 on the
workstation's RTX 3070 Ti with the GPU lease taken). 10 synthetic
Spanish bot replies; two **synthetic** reference voices, Piper
`es_ES-sharvard-medium` reading the es-1 enrolment text (25.5 s) and
`es_ES-mls_9972-low` (63.8 s). Latency is wall time from sending a
sentence to the worker until its WAV is written, model warm (the
bot's "sentence → first PCM": neither engine streams within a
sentence). Similarity is the cosine between WavLM embeddings
(`microsoft/wavlm-base-plus-sv`, the enrolment voiceprints' model) of
each whole clip and the reference. WER is faster-whisper large-v3
turbo (int8, beam 5) against the text, as a rough naturalness proxy.
Rows average the two references.

| | Cold load ms | Per sentence p50 / p90 / max ms | Similarity | WER | GPU peak MiB | Disk |
|---|---|---|---|---|---|---|
| Chatterbox multilingual (MIT), GPU | 12,676 | 2,029 / 2,243 / 2,411 | 0.960 | 0.033 | 4,336 | 3.0 GB weights + 0.26 GB venv (deleted) |
| **OpenVoice v2 (MIT), GPU** (Piper davefx, then the converter) | 4,236 | **118.5 / 169.8 / 203.1** | 0.917 | 0.076 | 782 | 126 MB weights + 324 MB venv |
| OpenVoice v2, CPU | 4,376 | 1,261.5 / 1,439.0 / 1,534.8 | 0.915 | 0.047 | 0 | same |
| Baseline: Piper davefx (today's bot voice) | — | 93 (`VOICEBOT.md`) | 0.598 | 0.034 | — | — |
| Baseline: the reference voice itself | — | — | 0.968 | 0.036 | — | — |

Per reference, similarity was 0.9725 / 0.9482 (Chatterbox), 0.8958 /
0.9385 (OpenVoice GPU) and 0.8098 / 0.3858 (plain davefx); two halves
of each reading score 0.994 / 0.991 against each other. WER varies
between runs (OpenVoice GPU with the sharvard reference: 0.041 in a
first run, 0.061 in this one) and Chatterbox samples at random: read
WER only as "intelligible, roughly as Piper".

**Shipped: OpenVoice v2.** It is the only one within the voice bot's
budget: one Chatterbox sentence (p50 2.0 s) would on its own nearly
use the 2.5 s from the end of the user's sentence to the first audio,
which already takes 1.6–1.7 s without cloning. OpenVoice also needs
782 MiB of GPU memory against 4.3 GB (the live recogniser shares the
8 GB GPU) and 0.45 GB of disk against 3.3 GB. Chatterbox sounds more
like the reference (0.960 against 0.917). XTTS-v2 and F5-TTS were not
measured (non-commercial weights; references only in the ticket).

Install (the worker's own venv, torch reused read-only from the shared
`.venv` through a `.pth`, like the lab's environments; no torch was
installed or changed): `~/.local/share/ragputin/clone-openvoice-venv`
(324 MB) and the converter's weights in
`~/.local/share/ragputin/clone-models/openvoice/converter/` (126 MB).
The worker is `lab/runners/voice_clone.py` (both engines; line
protocol in its docstring). OpenVoice's own watermark is off (it needs
the `wavmark` package, not installed).

## 4. How the bot uses it

Since #214 the cloner is chosen with `voicebot.clone_engine`: `qwen3`
(the default: Qwen3-TTS 0.6B, streamed, with Pocket TTS and Piper
behind it; `docs/live/NATURAL_TTS.md` §6) or `openvoice` (what follows).
The consent rules below are the same for both.

`voicebot.voice = clone:<person>` → at session start the orchestrator
looks the person up (`DirVoiceProfiles`, `--voiceprints-dir`, default
`data/transcriber/voiceprints`) and plans the voice (`plan_voice`):

- no profile, no consent, no clip, or no cloner installed → Piper
  davefx, and the session's feed says why (`{"type": "voice",
  "clone": false, "reason": ...}`);
- otherwise `ClonedVoice` (`orchestrator/application/voice.py`) speaks
  each sentence through `SubprocessCloner` (the worker in its venv,
  `--clone-engine openvoice`, `--clone-python`, warmed at session
  start). A sentence not ready within **800 ms** (`--clone-budget-ms`)
  is said by Piper; while the cloner is still busy the next sentences
  are Piper too; a cloner that fails is not asked again that session.
  The 800 ms come from the measured budget: 2.5 s target − (1,731 ms
  p90 end to end − 93 ms Piper) = 862 ms, rounded down. During the
  ~4.2 s the worker takes to load, the first sentences are Piper.

**The GPU lease (#211).** The cloner no longer runs on the GPU on its
own: like the natural voices (`NATURAL_TTS.md` §6) a cloner on the GPU
runs **under the live session's lease** (`gpu-lease.v1`, the same
`LeaseGpuGate` as `NaturalVoice`, `--gpu-lease-dir`):

- the worker (782 MiB measured, 800 MiB asked: `CLONE_NEED_MIB`) is
  started only while a valid `live` lease is held and `nvidia-smi`
  says the memory is free; the warm-up waits up to 30 s for the lease
  (the live session takes it and the image worker leaves the card),
  and if it never comes Piper speaks the whole session, with the
  reason in the feed's `voice` message;
- when the lease goes during a session the worker is stopped at once
  (its memory is free before the image worker resumes), and Piper
  speaks; when the session ends the worker is stopped too (before, it
  stayed on the GPU for the life of the orchestrator, lease or not);
- no gate wired, no GPU cloner: it is never used without the lease;
  `--clone-device cpu` needs no lease and stays loaded;
- «Escuchar ejemplo» outside a session **takes** the lease when nobody
  holds it (holder `ragputin-voicebot`, as for the natural voices),
  waits for the memory, loads the worker, speaks and stops it; if
  somebody else holds the lease (the image worker) the sample is Piper
  and says so in `X-Voice-Note`. The orchestrator never signals another
  process and never takes the lease during a session (the transcriber's
  live session is the holder).
- The Qwen3-TTS clone goes through `NaturalVoice` and was already under
  the lease (NATURAL_TTS.md §6): both cloner paths are now covered.

**Speed and the controls.** `voicebot.length_scale` (and the speed
`match_rate` sets), `noise_scale`, `noise_w` and `sentence_silence`
reach the **OpenVoice** clone: the worker's Piper base voice speaks
with them (one Piper process per distinct set of flags, at most two
kept; `length_scale` 1 is Piper's default and uses the one started at
load) and the converter recolours that audio, so pace, intonation and
pauses are Piper's and the timbre is the person's. The session warms
the worker with its own controls. Checked with «Escuchar ejemplo» on
the spare orchestrator (synthetic enrolled voice, §6): the same
sample sentence lasted 6.12 s at speed 0.8, 7.04 s at 1 and 9.23 s at
1.4 (and 6.98 s at 1 again), all spoken by the clone (`X-Voice-Engine:
openvoice`).

They do **not** reach **Qwen3-TTS** (the default `clone_engine`) or
Pocket TTS: neither has a speed or intonation parameter in the
installed `faster-qwen3-tts` 0.5.4 / `pocket-tts` 4.0.1 (nothing in
their generate calls) and the worker protocol carries none. The
«Ajustes» page disables the four sliders and says so when the chosen
voice is Pocket, Qwen3 or a Qwen3 clone (`controlsReach`,
`lib/settings.js`); with OpenVoice or a Piper voice they are live. A
time-stretch of Qwen3's audio was not tried. The values stay saved and
still shape Piper when it says a sentence as the fallback.

On the CPU OpenVoice would miss the budget (p50 1.26 s) and Piper would
speak.

## 6. End to end with a clone (#211)

Synthetic only: no human voice. `experiments/voicebot/first_audio.py`
(VB-5's script) against a spare orchestrator of this branch
(`natural_serve.py`, saved «Ajustes» untouched, `--voiceprints-dir` a
folder holding one **synthetic** enrolled person: Piper
`es_ES-mls_9972-low` reading a 42 s text as the clone clip, consent
`voice_clone: true`), the running transcriber (real ASR on the GPU, the
live session holding the lease) and the running RAG (Haiku via
`claude -p`). Questions are Piper `sharvard`; 8 turns per run.

| Clone (`voicebot.voice = clone:synth`) | Runs | Voice end → first audio p50 / p90 / max ms | Final → first audio p50 / p90 ms | First audio of the TTS p50 / p90 ms | Turns answered / within 2.5 s |
|---|---|---|---|---|---|
| OpenVoice v2, GPU, run 1 | 8 turns | 2,088 / 2,208 / 2,212 | 1,443 / 1,540 | 214 / 251 | 6 / 6 (2 hit the reply deadline: status `timeout`) |
| OpenVoice v2, GPU, run 2 | 8 turns | 2,067 / 2,270 / 2,274 | 1,385 / 1,554 | 158 / 204 | 7 / 7 (1 `timeout`) |
| Qwen3-TTS clone (speaker embedding only), GPU | 8 turns | 1,981 / 2,086 / 2,116 | 1,350 / 1,419 | 138 / 140 | 7 / 7 (1 `timeout`) |

Rows are the answered turns only (`ok`); the `timeout` turns are
Haiku not answering within the product deadline, and their first audio
came at 2.51-2.58 s. Results:
`experiments/voicebot/clone_first_audio_openvoice_run{1,2}.json` and
`clone_first_audio_qwen3.json`. 24 real Haiku calls. The feed said
`engine: openvoice` / `qwen3-cuda`, `clone: true`, no fallback reason,
once per session. Qwen3 audio is 24 kHz; OpenVoice's 22,050 Hz is also
Piper's, so the rate alone does not tell them apart: the proof that
the clone spoke is the worker process on the GPU during the session
(OpenVoice 642-724 MiB, Qwen3 2.8-2.96 GB).

Compared with the earlier rows (`NATURAL_TTS.md` §7: Pocket 1,653,
Qwen3 non-clone 1,740, Piper 1,584 ms p50) these runs are not
better or worse by the clone: the reply model's first token p50 was
980-1,018 ms here against 566-660 ms then, which alone explains the
~0.3-0.4 s. The clone's own part is the TTS first audio (138-214 ms
p50), inside the 800 ms budget with room. Few turns, one session per
run: read it as "the clone fits the 2.5 s target", not as a ranking.

**Lease, observed** (nvidia-smi compute apps and `lease.json` every
2 s during the sessions): the transcriber's `ragputin-live` lease was
held the whole time; the clone worker's process appeared after the
session started and was gone from the GPU within seconds of the session
ending (OpenVoice and Qwen3 both); no lease file was left. For the
sample button the lease holder was `ragputin-voicebot` while it spoke
and the file was deleted afterwards. The IMG worker was not running
during the measurements, so its pausing was not exercised end to end
here; the lease protocol is the same as the natural voices' (unit-tested
with a fake gate).

**Pending (owner):** real enrolled voices, a listener judging likeness
and naturalness, and the cloned voice's behaviour with an IMG job
actually queued.
## Limits

- Synthetic voices only: no enrolled human voice, no noise, no
  listener judging naturalness or likeness.
- End to end with a clone: synthetic enrolled voice only (§6), few
  turns, one run each; real voices still need the owner (pending).
- Speed and intonation do not reach the Qwen3-TTS clone (§4).
- Rate matching compares two differently measured rates and was not
  checked by ear.
- The services were not restarted: the cloner and the new settings
  take effect when the orchestrator and the transcriber restart.
