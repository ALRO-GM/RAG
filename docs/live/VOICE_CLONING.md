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

The Piper controls apply to Piper; with a clone, OpenVoice converts
the Piper davefx it runs itself at its own defaults (speed and rate
matching do not reach the clone yet). The worker runs on the GPU
without taking the GPU lease (782 MiB beside the live recogniser); on
the CPU it would miss the budget (p50 1.26 s) and Piper would speak.

## Limits

- Synthetic voices only: no enrolled human voice, no noise, no
  listener judging naturalness or likeness.
- Only per-sentence synthesis time was measured; the end-to-end first
  audio of a voice-bot session with a clone was not.
- Rate matching compares two differently measured rates and was not
  checked by ear.
- The services were not restarted: the cloner and the new settings
  take effect when the orchestrator and the transcriber restart.
