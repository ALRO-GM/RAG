# A more natural Spanish voice for the voice bot (VB-11, #214): research and measurement

The owner finds Piper `es_ES-davefx-medium` robotic: candidates would
distance themselves from the bot. This page is the research and
measurement part of #214: which open TTS speaks the most natural
Spanish **on the workstation's RTX 3070 Ti (8 GB), beside the live
recogniser (~2.5 GB)**, fast enough for the voice bot (first audio of a
sentence < 500 ms, streaming preferred). Nothing is integrated yet: the
owner listens to the samples first.

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

## Limits

- One reference voice for the cloners, one run per engine (15
  sentences), no listener yet: the naturalness numbers are a predictor.
- First audio is measured inside the worker; the end-to-end first audio
  of a voice-bot turn (recogniser, LLM, TTS) was not.
- Latency was measured with the GPU lease held (the IMG worker paused);
  with the recogniser busy at the same time, the GPU engines will be
  slower. Pocket on the CPU is unaffected by the GPU, but the CPU is
  shared with the services.
- Accent (Spain vs Latin American) of `lola`, the Kokoro voices and the
  reference was not checked.
