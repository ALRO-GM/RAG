# A more natural Spanish voice for the voice bot (VB-11, #214): research, measurement and integration

The owner finds Piper `es_ES-davefx-medium` robotic: candidates would
distance themselves from the bot. This page is the research and
measurement part of #214: which open TTS speaks the most natural
Spanish **on the workstation's RTX 3070 Ti (8 GB), beside the live
recogniser (~2.5 GB)**, fast enough for the voice bot (first audio of a
sentence < 500 ms, streaming preferred). §1–5 are the study; the owner
listened to the samples and chose **both** Pocket TTS (`lola`) and
Qwen3-TTS 0.6B (cloning), now selectable on «Ajustes» (§6, measured
end to end in §7).

Script `experiments/live/natural_tts.py`, results
`experiments/live/natural_tts_results.json`, listening samples
`experiments/live/tts_samples/` (run 2026-11-21, GPU lease taken).

## 1. Research (November 2026)

Quality reputation comes from the Artificial Analysis Speech Arena
(blind listener preference, Elo; "provider voice" board, open-weights
entries; Elo values move by ±10–20 between snapshots). The arena is
mostly English; no Spanish-only arena exists.

| Model | Size | Licence | Spanish | Arena Elo | Streaming | Cloning | Fits 8 GB with the recogniser? | Measured |
|---|---|---|---|---|---|---|---|---|
| Fish Audio Drama 3 / S2 Pro | 4B+ | research, **NC** | yes | 1,277 / ~1,110 | — | yes | no (8–16 GB) | no |
| Breeze TTS 2 | 3B | research, **NC** | ? | ~1,220 | — | yes | no (7.7 GB floor) | no |
| **Ming-3.0-tts-0.5B** (Ant Group) | 0.5B | MIT | yes | ~1,180 (#1 open on the controlled-voice board) | sample loop yields patches; streamed here (§2) | yes | yes | **yes** |
| Step Audio EditX | ~3B | Apache-2.0 | ? | ~1,100 | — | — | no (6–8 GB) | no |
| Voxtral TTS (Mistral) | 4B | **CC BY-NC** | yes | ~1,090 | — | 3 s | no | no |
| Magpie-Multilingual (NVIDIA) | 357M | NVIDIA Open Model | yes | ~1,064 | — | — | yes | no (NeMo stack; next candidate) |
| **Kokoro-82M v1.0** | 82M | Apache-2.0 | yes (3 voices) | ~1,060 | per segment | no | yes | **yes** |
| Higgs Audio V3 | 4B | **NC** | yes | ~1,040 | — | yes | no | no |
| Chatterbox multilingual | 0.5B | MIT | yes | ~1,020 | no (forks) | yes | yes (4.3 GB) | in #207: 2.0 s per sentence |
| Zonos v0.1 | 1.6B | Apache-2.0 | yes | ~1,000 | — | yes | no (6 GB) | no |
| OpenVoice v2 | — | MIT | via Piper | ~954 | — | yes | yes | baseline |
| XTTS-v2 | 0.5B | CPML, **NC** | yes | ~914 | yes | yes | yes | no (NC, older) |
| **Qwen3-TTS 12Hz 0.6B / 1.7B** | 0.6B / 1.7B | Apache-2.0 | yes (10 languages) | not on the board | yes (vendor: 97 ms first packet) | 3 s | 0.6B yes; 1.7B marginal | **0.6B** |
| **Pocket TTS** (Kyutai) | 100M | CC-BY-4.0 | yes (`spanish`, voice `lola`) | — | yes, CPU-first | yes, but the cloning weights are gated | yes (CPU) | **yes** |
| VoxCPM2 | 2B | Apache-2.0 | yes (30) | — | — | yes | probably | no (disk) |
| CosyVoice 3, F5-TTS Spanish (NC), FireRedTTS3, Orpheus multilingual, Kyutai TTS 1.6B (en/fr), IndexTTS2 (zh/en), VibeVoice (withdrawn) | | | | | | | | no |

Shortlist measured: **Qwen3-TTS 0.6B Base**, **Ming-3.0-tts-0.5B**,
**Kokoro-82M**, **Pocket TTS**; baselines Piper davefx and OpenVoice v2
(today's clone).

## 2. How it was measured

Each engine runs the way the bot would run it: a worker process in its
own virtual environment (`natural_tts.py --worker ENGINE`), one
sentence at a time, model warm (two warm-up sentences first). Sentences:
the 5 interview sentences of the issue plus the 10 bot replies of
`voice_cloning.py` (15 per engine).

- **First audio**: time from the sentence reaching the worker until
  the first audio chunk exists (streaming engines) or the whole
  sentence (Piper per sentence, Kokoro per segment, OpenVoice whole).
- **RTF**: synthesis time over audio length (< 1: faster than real time).
- **GPU peak**: `nvidia-smi` memory of the worker process (includes the
  CUDA context and, for Qwen3, the captured CUDA graphs).
- **WER**: faster-whisper large-v3-turbo (int8, beam 5): intelligibility.
- **UTMOS**: `utmos22_strong` (SpeechMOS v1.2.0), a predicted listener
  MOS (1–5). It was trained on English and also scores recording
  quality: the human reference clip (a LibriVox recording) gets 2.65.
  Use it as a rough ranking only; the owner's ear decides.
- **Similarity** (cloners): cosine of WavLM x-vectors to the reference.
- Cloners clone one *human* Spanish voice: CML-TTS (CC BY 4.0) Spanish
  test row 217, speaker 6306, female, 12.5 s (`--fetch-reference`); its
  accent was not checked.

Engine settings: Qwen3 through `faster-qwen3-tts` 0.5.4 (CUDA graphs),
4 codec steps (333 ms of audio) per chunk, **repetition penalty 1.3**:
with the default 1.05, 5 of 16 test generations ran on past the text
(babble until the token cap; lower temperature made it worse: 10 and 12
of 16), with 1.3 none of 16 did. Ming streamed patch by patch (its
`sample` loop plus the audio VAE's incremental decode, `use_cache`),
whole sentence as one text, greedy (`temperature 0`), as its README.
Pocket TTS `spanish` model, catalogue voice `lola` (a Common Voice
voice; cloning needs the gated weights). Kokoro `lang_code="e"`
(espeak-ng G2P).

## 3. Results

| Engine | First audio p50 / p90 ms | Whole sentence p50 ms | RTF | GPU peak MiB | WER | UTMOS (all 15 / interview 5) | Similarity | Cold load ms | Disk |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-TTS 0.6B** (clone), GPU | 187.5 / 187.9 | 1,918.5 | 0.458 | 4,310 | 0.016 | **3.93 / 4.00** | **0.963** | 4,069 | 2.6 GB (2.4 weights + 0.2 venv) |
| **Pocket TTS** `lola`, GPU | **17.8** / 25.2 | 258.6 | 0.068 | 754 | 0.023 | 3.85 / 3.91 | — | 2,492 | 0.22 GB |
| **Pocket TTS** `lola`, **CPU** | 81.0 / 86.1 | 898.0 | 0.229 | **0** | 0.010 | 3.74 / 3.84 | — | 2,148 | same |
| Kokoro `ef_dora` (female), GPU | 46.1 / 51.3 | 46.1 | 0.013 | 962 | 0.010 | 3.71 / 3.83 | — | 3,719 | 0.59 GB (0.31 + 0.28 venv) |
| Kokoro `em_alex` (male), GPU | 47.7 / 52.0 | 47.7 | 0.013 | 934 | 0.010 | 3.47 / 3.58 | — | 3,736 | same |
| Ming-3.0-tts-0.5B (clone), GPU | 103.7 / 104.9 | 2,940.6 | 0.658 | 3,410 | 0.010 | 3.42 / 3.23 | 0.961 | 5,151 | 2.9 GB (**deleted**) |
| Baseline: OpenVoice v2 (clone), GPU | 100.0 / 113.5 | 100.0 | 0.035 | 622 | 0.025 | 2.89 / 3.11 | 0.939 | 4,579 | 0.45 GB |
| Baseline: Piper davefx, CPU | 62.0 / 70.6 | 62.2 | 0.019 | 0 | 0.016 | 2.67 / 2.86 | — | 349 | (installed) |

WER is low everywhere and mostly not the TTS's fault: every engine has
its error on "entre las cuatro y las seis" and most on "unos cinco
días", where Whisper writes digits ("4", "6", "5") against the words of
the text. Every model is intelligible. Every streaming engine meets the < 500 ms first-audio
target by a wide margin; for the bot, what matters after the first
chunk is RTF < 1 (all are).

Notes per engine:

- **Qwen3-TTS 0.6B**: highest predicted naturalness and closest clone,
  Apache-2.0, streams in ~190 ms. Costs: 4.3 GB of GPU beside the
  recogniser (2.5 GB) and the services (~0.7 GB) leaves almost nothing
  (the CUDA graphs reserve memory for `max_seq_len` 2048; a smaller value
  should shrink it, not measured); sampling makes every rendering
  different; it needed the repetition-penalty fix above; installing it
  needed a Python-only torchaudio and a one-line patch of `qwen_tts`'s
  RoPE shim for transformers 5.20 (in the lab venv only). The 1.7B
  model was not measured (≈ 3.4 GB of weights alone).
- **Pocket TTS**: nearly as natural by the predictor, tiny, streams in
  18 ms on the GPU and **81 ms on the CPU at 0.23 RTF, using no GPU at
  all**. Only one Spanish catalogue voice (`lola`); cloning is in a
  separate, gated checkpoint (accept the terms on Hugging Face). Its
  sentences sound like one recording-quality voice; the voice's accent
  was not checked.
- **Kokoro**: fastest (a whole sentence in ~47 ms) and small; no
  cloning; Spanish through espeak-ng phonemes. `ef_dora` scores above
  `em_alex`.
- **Ming-3.0-tts**: top of the open arena (English) and an intelligible
  clone, but the lowest predictor score of the new models here, the
  slowest (RTF 0.66), 2.9 GB of disk, and it only ran with transformers
  4.46 pinned plus a patch of its iSTFT for torch 2.14 (no bf16 complex
  product). Rejected for now; its 5 samples stay for the owner's ear and
  reinstalling is the README's steps plus those two patches.

## 4. Recommendation

1. **Listen first** (`experiments/live/tts_samples/`, 5 sentences per
   engine: `qwen3_cuda_N`, `pocket_cpu_lola_N`, `pocket_cuda_lola_N`,
   `kokoro_cuda_ef_dora_N`, `kokoro_cuda_em_alex_N`, `ming_cuda_N`, and
   the baselines `piper_cpu_N`, `openvoice_cuda_N`). UTMOS is an English
   predictor; the choice is by ear.
2. If the ear agrees with the predictor, **integrate Pocket TTS on the
   CPU as the new default voice**: natural enough (3.74–3.85 against
   Piper's 2.67), 81 ms to first audio, streaming, CC-BY-4.0, 0.22 GB,
   and **no GPU memory**, so it cannot compete with the live recogniser
   or the IMG worker. Add **Qwen3-TTS 0.6B** as the "best quality /
   cloned voice" option in `voicebot.voice` when the GPU has room (it
   needs ~4.3 GB; with the lease held it fits, beside a busy recogniser
   it is tight). It would also replace OpenVoice for `clone:<person>`
   (similarity 0.963 vs 0.939, UTMOS 3.93 vs 2.89).
3. Kokoro `ef_dora` is the cheap fallback if Pocket's `lola` does not
   convince (no streaming within a segment, but 47 ms per sentence).

Licences: Pocket TTS CC-BY-4.0 (attribution; the `lola` voice is
derived from Common Voice, CC0), Qwen3-TTS Apache-2.0,
`faster-qwen3-tts` MIT, Kokoro Apache-2.0, Ming MIT. All commercial-use
compatible; none is NC.

## 5. Install (lab, reproducible)

Everything under `~/.local/share/ragputin/tts-lab/`: one venv per
engine created from the shared `.venv`'s interpreter, torch 2.14.1+cu126
reused read-only through a `zz_shared_torch.pth` (no torch installed, the
shared `.venv` untouched; packages installed with `uv pip install
--no-deps`, skipping anything the shared `.venv` has and every
`nvidia-*`/`cuda-*` wheel), weights in `tts-lab/hf` (`HF_HOME`), UTMOS in
`tts-lab/torch-hub`, a `score-venv` (shared `.venv` + Python-only
torchaudio 2.11.1, its C libraries moved aside) runs the orchestrator.

| Kept | Disk |
|---|---|
| `qwen3-venv` + Qwen3-TTS-12Hz-0.6B-Base | 0.21 + 2.4 GB |
| `kokoro-venv` + Kokoro-82M | 0.28 + 0.31 GB |
| `pocket-venv` + Pocket TTS (no cloning) | 0.01 + 0.21 GB |
| `score-venv` + UTMOS checkpoint | 0.4 GB |
| Deleted: Ming (`ming-venv`, source, weights) | 2.9 GB freed |

## 6. Integration: two natural voices behind the TTS port

**Choices** (`settings.v1`, group *Bot de voz*; checked by the RAG's
allow-list `src/domain/pipeline_settings.py`, the transcriber's shape
check and `orchestrator/domain/live_choices.py`):

| Step | Options (first = default) |
|---|---|
| `voicebot.voice` | `pocket:lola` · `qwen3` · `davefx` · `sharvard` · `clone:<person>` |
| `voicebot.clone_engine` | `qwen3` · `openvoice` |
| `voicebot.pocket_device` | `cpu` · `cuda` |

- `pocket:lola`: Pocket TTS, Spanish model, catalogue voice `lola` (the
  only Spanish one; the others are English, Italian, German, Portuguese
  and French speakers). On the CPU by default: no GPU memory, so it can
  never compete with the live recogniser or the image worker.
- `qwen3`: Qwen3-TTS 0.6B Base on the GPU cloning the study's
  reference, CML-TTS Spanish test row 217, speaker 6306 (**CC BY 4.0**,
  LibriVox recording; kept with its attribution and transcript in
  `~/.local/share/ragputin/natural-tts/voices/cml-tts-es-6306.{wav,txt}`),
  in-context with that transcript, as measured.
- `clone:<person>` with `voicebot.clone_engine = qwen3`: Qwen3-TTS clones
  the person's enrolment clip. The clip has no transcript, so it clones
  from the speaker embedding only (`xvec_only`); its likeness and
  naturalness were **not measured** (the study measured the in-context
  mode). The consent rules are unchanged (`docs/live/VOICE_CLONING.md`
  §2: `voice_clone: true` **and** the clip, `plan_voice`). With
  `openvoice`, or when Qwen3-TTS is not installed but OpenVoice is, the
  #207 cloner speaks as before.

**Default.** New installs and the product files
(`config/products/voicebot*.yaml`: `engine: pocket, pocket_voice: lola`)
speak with Pocket TTS on the CPU; where its venv is not installed the
same choice falls back to Piper davefx and the session's feed says
`Pocket TTS not installed`, so an install without the natural voices
behaves as before. Choices saved earlier on «Ajustes» (e.g. `davefx`)
keep winning.

**How a sentence is spoken** (`orchestrator/application/natural_voice.py`,
`NaturalVoice`). The voice is a chain: the chosen engine (Qwen3 or Pocket
on the GPU), then Pocket on the CPU, then Piper davefx. For each
sentence the first engine that can speak does:

- a GPU engine is used only while the **live session's GPU lease**
  (`gpu-lease.v1`) is held and, before it loads, `nvidia-smi` says the
  memory it needs is free (4,400 MiB for Qwen3, 800 for Pocket: the
  study's peaks rounded up). The voice bot's own live session takes
  that lease (the image worker leaves the card), so nothing here takes
  it during a session: the lease file has one holder and taking it
  would push the transcriber to the CPU. When the lease goes, a loaded
  GPU engine is stopped at once (its memory is free before the image
  worker resumes 30 s later). No other process is signalled, ever;
- its first audio must come within **800 ms** (`--natural-budget-ms`,
  the clone's budget, `VOICE_CLONING.md` §4); otherwise that sentence
  is cancelled on it and the next voice says it, and while it is still
  busy (loading, or stopping) the next sentences skip it;
- an engine that fails is not asked again in that session;
- the sentence is **streamed**: the worker sends audio as the model
  makes it (Pocket every 80 ms, Qwen3 every 4 codec steps = 333 ms);
  the first piece goes to the page at once, the next ones are held
  until they are as long as what was already sent (at most 1 s), so a
  model faster than real time keeps ahead of playback and a sentence
  takes a few messages (`AudioJoiner`). A piece later than 4 s ends
  the sentence there;
- the feed says who speaks: `{"type": "voice", "engine": "pocket-cpu" |
  "pocket-cuda" | "qwen3-cuda" | "openvoice" | "piper", "clone",
  "length_scale", "reason"}` at the start and again whenever the speaker
  changes, with why the voices before it did not speak (`qwen3-cuda: no
  live GPU lease`, `… missed the 800 ms budget`, `… failed (…)`).

Pocket on the CPU is warmed when the orchestrator starts and shared by
the sessions (one process, kept loaded for the orchestrator's life; a
session warms it again only if it is not loaded); a GPU engine is
started for its session (warmed in the background once the lease and
memory allow, waiting up to 30 s) and stopped when the session ends.
The engines of a chain load side by side, from the moment the session
is created (before the transcriber is asked to start), and a session's
first sentences wait for them (§8). Piper's controls (speed,
intonation…) only reach Piper (§8).

**«Escuchar ejemplo»** says the fixed sentence with the voice being
chosen; the header `X-Voice-Engine` (and the page) says which voice
spoke and why not the chosen one. Outside a session nobody holds the
lease, so for a GPU voice the sample **takes** it (holder
`ragputin-voicebot`, renewed every 5 s, deleted right after), waits up
to 20 s for the memory, loads the model (~4–5 s cold for Qwen3), speaks
and stops it. A live session started during those seconds would find
the lease taken and run its recogniser on the CPU.

**Workers.** `lab/runners/natural_tts.py` runs in each engine's own
venv (never imported by the services), keeps the model loaded and
speaks a line protocol (JSON requests in; base64 PCM pieces, then a
`done` line, out; `{"cancel": id}` stops a sentence). The orchestrator
side is `orchestrator/adapters/outbound/subprocess_speech.py`
(`WorkerSpeech`, `WorkerEngines`); the gate is
`orchestrator/adapters/outbound/gpu_gate.py` (`LeaseGpuGate`). Flags of
`python -m orchestrator serve`: `--natural-tts on|off`, `--natural-dir`
(default `~/.local/share/ragputin/natural-tts`), `--natural-budget-ms`,
`--gpu-lease-dir`. The services are not restarted by this change: it
takes effect when the orchestrator (and the RAG, for the new settings)
restart.

### Install

Everything lives in `~/.local/share/ragputin/natural-tts/` (moved there
from the study's `tts-lab/`, which keeps symbolic links so
`experiments/live/natural_tts.py` still runs):

| Path | What | Disk |
|---|---|---|
| `pocket-venv/` | `pocket-tts` 4.0.1 (+ `einops`, `sentencepiece`) | 6.5 MB |
| `qwen3-venv/` | `faster-qwen3-tts` 0.5.4, `qwen-tts-hf` 0.1.1.post1, Python-only `torchaudio` 2.11.1 (its C libraries in `disabled-torchaudio-libs/`), `librosa`, `numba`… | 212 MB |
| `hf/` | Hugging Face cache (`HF_HOME`): `kyutai/pocket-tts-without-voice-cloning` (Spanish), `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (and the study's Kokoro) | 2.9 GB |
| `voices/cml-tts-es-6306.wav` + `.txt` | Qwen3's default reference voice, CC BY 4.0, attribution and transcript | 0.6 MB |

Steps (the shared `.venv` is never changed; torch 2.14.1+cu126 is
reused read-only):

1. `uv venv --python 3.10 <dir>/pocket-venv` (and `qwen3-venv`), then
   put a `zz_shared_torch.pth` in each venv's `site-packages` holding
   the path of the shared `.venv`'s `site-packages`.
2. `uv pip install --python <venv>/bin/python --no-deps <packages>`:
   `pocket-tts einops sentencepiece` for Pocket; `faster-qwen3-tts
   qwen-tts-hf torchaudio==2.11.1 librosa numba llvmlite soxr sox pooch
   lazy_loader audioread decorator msgpack platformdirs` for Qwen3. Skip
   anything the shared `.venv` has and every `nvidia-*`/`cuda-*` wheel;
   move torchaudio's C extension libraries aside (Python-only).
3. Qwen3 with transformers 5.20: `qwen_tts/_transformers_compat.py`
   needs the default RoPE initialiser registered
   (`ROPE_INIT_FUNCTIONS.setdefault("default", ...)`, the one-line patch
   of the study, already in the venv).
4. Weights: `HF_HOME=<dir>/hf` and load each model once (Pocket:
   `TTSModel.load_model(language="spanish")`; Qwen3:
   `FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")`);
   the services run the workers offline (`HF_HUB_OFFLINE=1`).
5. Reference voice: `experiments/live/natural_tts.py --fetch-reference
   <dir>/voices/cml-tts-es-6306.wav` (CML-TTS, CC BY 4.0) and its
   attribution file.

Check: `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m slow
tests/slow/test_natural_voice_worker.py` (Pocket on the CPU through the
adapter).

## 7. Measured end to end: end of the user's turn → first audio

`experiments/voicebot/first_audio.py` (VB-5's script, now also keeping
the sample rate of each turn's first audio and the feed's `voice`
messages) against a spare orchestrator of this branch
(`experiments/voicebot/natural_serve.py`: `serve` with the voice
choices fixed for that process, the saved «Ajustes» untouched), the
running transcriber (real ASR on the GPU, live session with its GPU
lease) and the running RAG (warm Haiku pool). 8 synthetic turns per
engine (the first 8 questions of VB-5), feed followed on the
orchestrator (VB-5 followed it through the transcriber's proxy).
Result `experiments/voicebot/natural_first_audio_results.json`, run
2026-11-21; 24 real Haiku calls.

| Engine (`voicebot.voice`, `pocket_device`) | Voice end → first audio p50 / p90 / max ms | Final → first audio p50 / p90 ms | Turn start → first sentence p50 ms | First audio of the TTS (`first_tts_ms`) p50 / p90 ms | Turns within 2.5 s |
|---|---|---|---|---|---|
| Pocket TTS `lola`, CPU (`pocket:lola`, `cpu`) | 1,653 / 1,873 / 1,915 | 952 / 1,280 | 856 | 102 / 118 | 8 of 8 |
| Pocket TTS `lola`, GPU (`pocket:lola`, `cuda`) | 1,620 / 1,698 / 1,819 | 974 / 1,113 | 931 | 40 / 56 | 8 of 8 |
| Qwen3-TTS 0.6B, GPU (`qwen3`) | 1,740 / 1,900 / 1,920 | 1,095 / 1,324 | 893 | 201 / 204 | 8 of 8 |
| Baseline: Piper davefx, CPU (VB-5, 14 turns, through the proxy) | 1,584 / 1,731 / 1,850 | 915 / 1,134 | 831 | 93 / 155 | 14 of 14 |

In every turn of the three runs the chosen engine spoke the first
audio (24 kHz) and the feed reported no fallback. The time to the first
audio is still mostly the recogniser's final (~0.6 s) and the reply's
first sentence (~0.9 s); the natural voices add their first piece:
~0.1 s for Pocket on the CPU, ~0.04 s on the GPU, ~0.2 s for Qwen3
(per-sentence first audio inside the worker, measured in §3: 81, 18 and
188 ms). Eight turns per engine, one session each: the differences
between the three rows (≈ 0.1 s at p50) are within what the reply
model's own variation moves (its first token p50 566–660 ms across the
runs). The cloned `clone:<person>` with Qwen3 (speaker embedding only)
and the sample button were not measured end to end.

## 8. The first sentence of a session (cold voice)

**The report.** With `voicebot.voice = clone:<person>`,
`voicebot.clone_engine = qwen3` and `pocket_device = cpu`, the
interviewer's opening line sounded bad and the later sentences fine.

**Cause, reproduced.** `experiments/voicebot/opening_voice.py` starts
`voicebot.interview` sessions on a spare orchestrator
(`natural_serve.py` on port 8792, the saved «Ajustes» choices: the
clone with Qwen3, Pocket on the CPU, the Piper controls tuned) against
the running transcriber (live session and its GPU lease) and RAG, and
reads the feed's `voice` messages until the opening turn has spoken.
Result `experiments/voicebot/opening_voice_results.json`, 3 sessions
before and 3 after, 2026-11-22, one Haiku call each:

| | Opening line, 1st sentence | 2nd sentence | 1st audio after the session started |
|---|---|---|---|
| Before (main `9995595`) | Pocket `lola` once, **Piper** twice | Pocket, Pocket, Piper | 2,078 / 2,260 / 1,827 ms |
| After (this branch) | **Qwen3 clone** 3 of 3 | Qwen3 clone 3 of 3 | 7,543 / 7,256 / 7,239 ms |

Before, the feed said why: the Qwen3 model was still loading when the
opening was ready (`qwen3-cuda missed the 800 ms budget`, then
`qwen3-cuda busy`), so the next voice said it, and the backchannels
(«ajá», «vale»…, `interview.backchannel = voice`) were being
synthesised at the same moment, also by Pocket (in 2 of the 3
sessions `pocket-cpu` spoke 75 and 93 ms after the start, before any
reply existed, because `qwen3-cuda: no live GPU lease` yet), so Pocket
was `busy` too and **Piper** said
the opening, with its own (tuned) controls. So the first sentence was
either another voice (Pocket's `lola`, not the clone) or Piper. Pocket
was not cold: its worker is one process kept loaded by the orchestrator
since it started (the running one had been up 17 h, across the owner's
sessions).

**Fix.**

1. A session's first sentences wait for its voice: `NaturalVoice`
   counts a bound of `READY_WAIT_S` = 12 s from the start of the
   warm-up; `synthesize` waits until every engine of the chain has
   been tried (loaded or failed) or the bound passes, then applies
   the 800 ms budget as before (after 12 s the fallbacks speak as
   before). The opening's reply is written meanwhile, so the wait
   only adds what the model still needs. The feed says
   `{"type": "voice_state", "state": "warming"}` and then `ready` (or
   `fallback`); the page shows «Preparando la voz…» / "Preparing the
   voice…" meanwhile.
2. The warm-up starts as early as possible: the voice is built (and
   starts loading) when the session is created, before the transcriber
   is asked to start; the engines of the chain load side by side, so
   Pocket on the CPU never waits for the GPU lease; the gate is asked
   every 0.25 s instead of every 1 s while it refuses.
3. Pocket on the CPU stays resident: it already was (one shared worker
   per voice, never stopped by a session); a session now skips warming
   an engine that is already loaded, and warms Pocket again only if
   its worker is not running.
4. The backchannels are synthesised after the opening turn is over, so
   they never take a voice from the opening line.
5. A session that ends while a reply is still streaming no longer
   crashes the bot's thread (`RuntimeError: cannot schedule new
   futures after shutdown` in `orchestrator.log`): after `close()`,
   `synthesize` stops quietly (no Piper either).

The price is the wait: the opening's first audio came 7.2–7.5 s after
the session started instead of 1.8–2.3 s (Qwen3 loading, with the
lease and the memory check); the page says the voice is being prepared.

**Piper's controls and the natural voices.** `length_scale` (and the
speed `match_rate` sets from a person's rate), `noise_scale`,
`noise_w` and `sentence_silence` are not applied to Pocket or Qwen3:
`NaturalVoice` passes only the sentence and the language to their
workers (the worker protocol has no such fields), and the controls
reach Piper alone when it says a sentence (a unit test checks both).
They did not make the opening worse by themselves; they only shaped
the Piper fallback that said it. The `length_scale` of the feed's
`voice` message is the plan's, used only by Piper.

## Limits

- One reference voice for the cloners, one run per engine (15
  sentences), no listener yet: the naturalness numbers are a predictor.
- §3 measures first audio inside the worker; §7 the whole turn, on 8
  synthetic turns per engine only.
- Latency was measured with the GPU lease held (the IMG worker paused);
  with the recogniser busy at the same time, the GPU engines will be
  slower. Pocket on the CPU is unaffected by the GPU, but the CPU is
  shared with the services.
- Accent (Spain vs Latin American) of `lola`, the Kokoro voices and the
  reference was not checked.
