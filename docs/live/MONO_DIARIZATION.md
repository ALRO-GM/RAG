# Mono live diarization: streaming Sortformer v2 + voiceprint naming (LV-6)

Design: `docs/platform/ARCHITECTURE.md` §2 (SpeakerAttribution, stream), §5
(VRAM), §6 (attribution budget 150 ms mono), §9; events: `live.event.v1`
(`docs/platform/CONTRACTS.md` §1). Session API: `LIVE_SESSION.md`.

## What runs

| Piece | Code | How |
|---|---|---|
| Port | `transcriber/application/ports.py` `StreamDiarizer` | `start`, `feed(samples)`, `frames(until, timeout)` → `(first, rows)`, `finish`, `close`; `frame_seconds`, `slots` |
| Rules (domain) | `transcriber/domain/live_diarization.py` | `SpeakerFrames` (frames so far), `pick_slot` (slot, coverage, purity of a time span), `MonoSpeakers` (slot labels, finals sent, revisions), `NamingPlan`, `name_slots` |
| Use case | `transcriber/application/live_diarization.py` `MonoAttributor`, wired in `application/live.py` | every 32 ms frame is fed (sent in 160 ms pieces); each final sentence gets the slot that talks most in its span; finals whose frames were incomplete are re-picked as frames arrive; changes go out as `speaker_update` |
| Adapter | `transcriber/adapters/outbound/sortformer_stream_diarizer.py` | starts the worker per session (GPU memory held only while a mono session runs), writer and reader threads, reduced environment, process group killed on close |
| Worker | `lab/runners/sortformer_stream.py`, run by the lab's `nemo` environment (`make lab-env FAMILY=nemo`) | `nvidia/diar_streaming_sortformer_4spk-v2`, model card low-latency setting: chunk 6 frames, right context 7, FIFO 188, cache update 144, cache 188 → 1.04 s from a sample to its frame; features computed per chunk window |
| Naming | `MonoAttributor` with `WavLMVoiceEncoder` (the enrolment's model) and the voiceprint repository | clean speech of each slot kept in memory only (never written): sentences whose frames are complete and ≥ 80 % that slot alone; at 4 s, then ×2 up to 30 s, the new speech is embedded on the CPU in a worker thread (the stream never waits) and the slot's mean voice is matched one-to-one against the candidates (`match_voices`, cosine ≥ 0.95, the batch threshold) |
| Wiring | `transcriber/bootstrap.py` `live_builder`, `live_attributor`; `LiveSessions` | request `diarize: true` (channels 1); lease asks `need_mib` + `diarize_need_mib` (1500); `transcript.v1` hand-off applies `speaker_update` and writes `pipeline.diarizer = "sortformer-streaming-v2"` and the enrolled speakers |
| Orchestrator | `orchestrator/application/sessions.py` | a product with `attribution: {kind: sortformer_stream}` and `mic` audio sends every declared speaker on channel 0 and `diarize: true` |
| CLI | `python -m transcriber live-file <wav> --diarize [--enrolled a,b]` | the measurement below |

NeMo is not a dependency of the service's environment (it brings a
large tree of libraries, and installing it there could change the
shared torch the running services use), so Sortformer runs as a
subprocess of the lab's `nemo` environment and talks JSON lines:

| Request (stdin) | Answer (stdout) |
|---|---|
| — (start-up) | `{"type": "ready", "model", "revision", "frame_seconds": 0.08, "speakers": 4, "latency_seconds": 1.04, "device"}` |
| `{"type": "start"}` | `{"type": "started"}` |
| `{"type": "audio", "pcm": "<base64 s16le 16 kHz mono>"}` | `{"type": "frames", "first": n, "probs": [[p0, p1, p2, p3], ...]}` (frames completed by this audio; may be empty) |
| `{"type": "flush"}` | `frames` of the rest, with the right context there is |
| `{"type": "stop"}` or end of input | exit |
| any failure | `{"type": "error", "message"}` |

Only the protocol owns stdout (library output goes to stderr, kept in
`<data_dir>/live/diarizer.log`).

## Behaviour

- **Slots and ids.** Slot `n` is the `n`-th voice Sortformer hears. It
  takes `speakers[n]` of the start request (ids and roles from the
  product config, e.g. the agent who answers first), else `S1`… (never
  an enrolled id). `session_started` lists the slots and the enrolled
  candidates (`enrolled: true`, name from their consent).
- **Final and provisional.** At the endpoint the final waits up to
  `wait_ms` (150, the §6 budget) for the frames of its span. Frames lag
  the audio by 1.04 s plus chunking, so most finals go out with
  `speaker_provisional: true` and the speaker of the frames already
  there (or the last speaker). When the frames complete, a
  `speaker_update` (same id, revision + 1, full utterance) confirms or
  corrects it.
- **Naming.** When a slot matches a voiceprint, every final of that
  slot is revised with the person's id (`speaker_update`), and later
  finals use it. A slot that stops matching goes back to its label.
- **Turns.** The turn of an id is decided when its segment opens (by
  the speaker guessed then); ids never change, so after a revision
  `turn` may not follow the speaker. The `transcript.v1` hand-off
  regroups turns from the final speakers.
- **Failures.** A worker that is not installed, fails or exits stops
  only the attribution: one `warning` (`other`), and later sentences
  keep the last speaker known. The session goes on.
- **CPU (degraded).** The worker follows the session's device; on the
  CPU it was not measured.

## Measurement (2026-11-21)

RTX 3070 Ti shared with the running services (RAG, other agents' jobs),
GPU lease held (`transcriber gpu-lease acquire --need-mib 4000`).
Synthetic speech for the first table: no claim about real recordings.

### Piper simulation, mono mix, real time

Audio: `sim-reunion-verano`, 881.8 s, four Piper voices (two from the
same multi-speaker model), the four voices enrolled
(`~/.local/share/ragputin/sim96/data/voiceprints`, WavLM base-plus-sv).

```sh
TRANSCRIBER_DATA_DIR=~/.local/share/ragputin/sim96/data python -m transcriber live-file \
    ~/.local/share/ragputin/sim96/audio/sim-reunion-verano.wav --diarize --out diar.jsonl
TRANSCRIBER_DATA_DIR=~/.local/share/ragputin/sim96/data python -m transcriber live-file \
    ~/.local/share/ragputin/sim96/audio/sim-reunion-verano.wav --out base.jsonl
PYTHONPATH=. python experiments/live/mono_diarization.py events diar.jsonl \
    ~/.local/share/ragputin/sim96/audio/sim-reunion-verano.reference.json
```

| | without diarization | with `--diarize` |
|---|---|---|
| finals | 289 | 290 |
| `speaker_update` events | — | 188 |
| finals sent provisional | — | 61.4 % |
| speaker right at first emission | — | 98.3 % |
| speaker right after the last revision | — | 99.3 % (all four named with their enrolled id) |
| lag_ms p50 / p90 (end of speech → final) | 680 / 771 | 823 / 915 |
| attribution_ms p50 / p90 | 0 / 0 | 151 / 151 |
| peak GPU memory, service process (nvidia-smi) | 1394 MiB | 1394 MiB |
| peak GPU memory, Sortformer worker | — | 1140 MiB |
| wall time for 881.8 s | 14:45.8 | 14:46.5 |

Added cost: +143 ms lag at p50 (+144 at p90), which is the 150 ms
attribution wait; +1140 MiB of GPU memory (the worker). A first run
with the voice encoder on the GPU measured the service process at
2742 MiB (+1348 MiB: torch's CUDA context and WavLM), which broke the
1.5 GB budget; naming was moved to the CPU in a thread (the run above).
The wait rarely completes a final: 61 % still go out provisional, so a
`wait_ms` of 0 would save ~150 ms at the price of more provisional
finals (not measured).

Frame-level DER of the worker on the same audio (frames at 0.5, collar
0.25 s, `experiments/live/mono_diarization.py frames` then `der`):
0.119 (missed 80.7 s, false alarm 0.0 s, confusion 0.1 s of 679.7 s);
the misses are pauses inside the reference sentences.

### Lab sessions (real speech), frames only

Same worker, streamed as fast as it answers (0.5 s messages; p50 20 ms
per message), against the lab references
(`data/transcriber/lab/<id>/reference.json`); "offline setting" is the
lab runner's 30 s chunks (chunk 340, right 40, FIFO 40, update 300) in
the same worker and scorer.

| session | DER low latency (1.04 s) | DER offline setting |
|---|---|---|
| callhome-data-0001 | 0.329 | 0.288 |
| callhome-data-0002 | 0.508 | 0.491 |
| callhome-data-0003 | 0.458 | 0.443 |
| callhome-data-0004 | 0.198 | 0.169 |
| callhome-data-0005 | 0.130 | 0.132 |
| ami-ES2004b | 0.421 | 0.442 |
| ami-TS3003a | 0.358 | 0.378 |
| mean CALLHOME (5) / AMI (2) | 0.324 / 0.390 | 0.305 / 0.410 |

Check of the worker's chunked features: NeMo's own `diarize()` in
streaming mode with the low-latency setting gave exactly the same DER
on callhome-data-0002 (0.5078) and -0003 (0.4579), peak 1224 and
1393 MiB allocated.

Reading: the low-latency setting costs little against the 30 s one
on these sessions (+0.02 CALLHOME, −0.02 AMI). Both are far from the
lab table (§13.8.2: Sortformer 0.122 on 10 CALLHOME calls); these five
calls have large confusion (85–153 s) in both settings. Whether the lab
pipeline scores or cuts differently was not investigated. Naming was
not measured on real speech (no enrolled voices for these corpora).

## Not done

- Roles of anonymous voices beyond the order of first appearance.
- The CPU (degraded) mode of the worker was not measured.
- More than four voices: Sortformer has four slots; what happens with
  a fifth was not measured.
