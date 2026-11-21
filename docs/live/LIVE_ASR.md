# Live ASR: Silero VAD, endpointing, faster-whisper stream (LV-3, LV-4)

Design: `docs/platform/ARCHITECTURE.md` §2, §5, §6, §9; events: `live.event.v1`
(`docs/platform/CONTRACTS.md` §1).

## What runs

| Piece | Code | How |
|---|---|---|
| Endpointing | `transcriber/domain/live.py` `Endpointer` | 32 ms frames; start after 96 ms at p ≥ 0.5 with 192 ms pre-roll; stay while p ≥ 0.35; end 500 ms (300–700) after the last voiced frame; forced cut at 15 s |
| VAD | `transcriber/adapters/outbound/silero_vad.py` | Silero VAD ONNX bundled with faster-whisper (`silero_vad_v6.onnx`, i.e. v6, not the v5 named in §9), onnxruntime, 1 CPU thread, context and LSTM state carried per frame |
| Stream ASR | `transcriber/adapters/outbound/faster_whisper_stream.py` | `large-v3-turbo` (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`) int8, beam 1, no previous-text conditioning, warm-up decode at start; partials without word times, finals with them |
| Partials | `LocalAgreement` (domain) | the open segment is decoded again every 800 ms; the prefix two consecutive hypotheses agree on goes out |
| Sentences | `SentenceAssembler` (domain) | finals split into sentences, ids `t<turn>.s<index>`; a partial and its final share the id, `revision` grows |
| Use case | `transcriber/application/live.py` `LiveTranscription` | one stream per channel, one shared model; `timings` per final |
| CLI | `python -m transcriber live-file <wav> [--channels] [--out events.jsonl] [--fast]` | the file is read 32 ms at a time in real time (`--fast`: as fast as possible); stereo with `--channels`: one stream and speaker per channel |

Install: dependency group `live` (`faster-whisper`, which brings
`ctranslate2` and `onnxruntime`; no torch). On CUDA the adapter loads
cuBLAS 12 and cuDNN 9 from the `nvidia-*` wheels of the CUDA torch build.

Timings (real time): `endpoint_ms` from the end of the last voiced frame to the
endpoint decision (the hang, plus any backlog), `asr_ms` the final decode,
`lag_ms` from the end of speech to the emission of the final. One value per
segment (every sentence of a segment carries the same timings).

## Measurement (2026-11-21)

Audio: `sim-reunion-verano`, 881.8 s of mono Piper speech, four voices
(`experiments/skills/render_simulation.py`, `~/.local/share/ragputin/sim96/audio/`).
RTX 3070 Ti shared with the running services, GPU lease held
(`transcriber gpu-lease acquire --need-mib 1500`), defaults above.

```sh
python -m transcriber live-file ~/.local/share/ragputin/sim96/audio/sim-reunion-verano.wav \
    --session sim-reunion-verano-live --out events.jsonl
PYTHONPATH=. python experiments/live/live_asr_latency.py events.jsonl \
    ~/.local/share/ragputin/sim96/data/sessions/sim-reunion-verano-audio/transcript.json \
    --reference ~/.local/share/ragputin/sim96/audio/sim-reunion-verano.reference.json \
    --out experiments/live/live_asr_results.json
```

Wall time 14 min 44.6 s for 14 min 41.8 s of audio; peak RSS 2.1 GB.
212 segments, 287 finals, 621 partials. Results: `experiments/live/live_asr_results.json`.

| ms | p50 | p90 | max | budget (§6) |
|---|---|---|---|---|
| endpoint_ms | 512 | 589 | 1852 | 500 |
| asr_ms (final) | 157 | 206 | 1756 | 350 |
| lag_ms (end of speech → final) | 674 | 767 | 3399 | — |
| partial lag_ms (end of decoded audio → partial) | 162 | 200 | 1012 | — |

Word error rate of the finals (normalised as in the lab):

| Hypothesis | against | WER |
|---|---|---|
| live finals | batch transcript (whisper-large-v3-turbo fp16, same audio) | 0.126 |
| live finals | the text Piper read | 0.123 |
| batch transcript | the text Piper read | 0.088 |

Reading: the medians are inside the budget (the endpoint p50 is the 500 ms
hang rounded up to 16 frames of 32 ms); the tails come from a few segments
where a decode took 0.5–1.8 s and the frames queued behind it. The live
finals have about 4 points more WER than the batch run on this audio; the
live run emits 2,157 tokens against 2,027 for the batch one. Neither the
slow decodes nor the extra words were investigated yet. Synthetic speech
(Piper), one run: no claim about real recordings.
