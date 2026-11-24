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

## Voice check: two people on one microphone (#249)

The owner's sessions of 2026-11-24 showed the gap: up to four slots in
11 s of speech, and the employee named only after 4 s of clean speech
at cosine ≥ 0.95 against an enrolment made with another microphone
(sometimes never). A two-party product (interview, support) on one
microphone now starts with a **voice check** and keeps exactly two
people.

### What runs

| Piece | Code | What |
|---|---|---|
| Request | `LiveSessionRequest.parties` (`{"employee", "other", "voice_check"}`), sent by `orchestrator/application/sessions.py` (`two_parties`, `party_speakers`) for every `mic` session of a product with both an employee role and the other party's | speakers: the employee (the chosen person's id, else the employee channel's `S1`) and the other party (`S2`); `enrolled: [employee]` or `[]` (nobody else may be named) |
| Rules | `transcriber/domain/voice_check.py` (`VoiceCheck`, `PartySides`, `VoiceCheckSettings`) | the check's state; which side each slot is on |
| Use case | `MonoAttributor(parties=PartyPlan(...))` in `application/live_diarization.py` | below |
| Events | `voice_check` (`listening` with seconds heard of needed → `recognised` / `short` / `skipped` / `failed`); `speaker_unsure` on utterances | `live.event.v1`, added compatibly |
| Requests | `POST /api/live/sessions/{id}/voice` `{"action": "skip"}` or `{"action": "mine" \| "not_mine", "utterance"}` | queued, applied by the session's worker with its next frame |
| Diagnosis | `<data_dir>/live/<id>/attribution.jsonl` (mode 600) and one log line | every final's slot, coverage, purity, side, provisional, unsure; the check (seconds, sentences, state); every comparison of the slots with the check (cosines, seconds, side, source); the enrolment's nearest cosines; corrections; each sentence's last side. Ids and numbers only, no text, no audio |

1. **Check.** The finals heard while the check listens are the
   employee's (pinned: no later frame changes them) and their audio is
   kept in memory. At 5 s of speech (or 3 s once 20 s have passed since
   the first one; less is `short`) the audio is embedded with WavLM on
   the CPU in the naming thread (as the slots' voices are): that is the
   session's reference of the employee's voice, with this microphone and
   room. It is never written and never becomes an enrolment. The slot
   that spoke most of the check stands in for the employee until voices
   can be compared.
2. **Every later sentence** takes its Sortformer slot as before (same
   150 ms wait, same provisional finals and `speaker_update`), and the
   slot's side: the employee's or the other person's id. No anonymous
   label is ever shown, so a third or fourth slot is never a third or
   fourth person.
3. **Sides of the slots.** Each slot's clean speech (as for naming,
   from 2 s, then each time it doubles) is embedded; the slot whose
   voice is nearest to the check, at cosine ≥ 0.93, is the employee's,
   and so is any other slot at ≥ 0.97 (the employee split in two
   slots); every other slot is the other person's. A slot not compared
   yet keeps the stand-in's side and its sentences say `speaker_unsure`
   (except the slot that spoke the check).
   An enrolment, when the employee has one, still names their slot
   (0.95, as before) unless the employee said that voice is not theirs.
4. **Corrections.** «Esta soy yo» / «Esta no soy yo» on a line pins
   that line and its slot to the side chosen: the slot's earlier lines
   are revised (`speaker_update`) and its later ones follow.
5. **Skip.** Without the check (skipped, short, failed) the session
   works as before (slots, the enrolment if any), except that every
   slot not named is the other person and corrections still work.

GPU: no new model; the lease asks the same `need_mib` + 1500. Latency:
the attribution wait stays 150 ms (§6); the check's and the slots'
embeddings run off the stream. One WavLM pass on the CPU over a
sentence-long clip took p50 110 / p90 370 ms (3,761 clips, 8 to 20 CPU
threads, shared with other jobs; 4 s and longer: p50 256, p90 436), which is
why the reference is not computed in the final's path.

### Measurement (2026-11-24): `experiments/live/voice_check.py`

Material: the lab's references (`data/transcriber/lab/<id>/reference.json`)
of 13 CALLHOME Spanish telephone calls and 4 AMI meetings (single
distant microphone: one microphone in a room). Three calls
(callhome-data-0002, -0003, -0009) were set apart: their audio does not
follow their reference (the level inside the reference's speech over
the level outside is 0.99–1.01, against 2.3–23.6 for the others; on
them even an oracle centroid of each speaker's sentences reaches only
0.56–0.60). That may explain their high DER in the table above (0.508,
0.458) and is worth a look in #201. Each of the two speakers who talk
most is the "employee" in turn: 20 trials on 10 calls (10 on the five
calls with only two people), 8 on the AMI meetings (only the two main
speakers' sentences scored, the two others' left out). The check is
the speaker's earliest segments with under 10 % overlap and at least
1 s, until 3, 5 or 8 s (whole segments: 8.7 s on average for "5 s");
every later reference segment is a sentence to attribute (a segment
with 20 % or more covered by someone else is "overlapped"). Voices:
WavLM base-plus-sv on the CPU; slots: the streaming Sortformer worker
of this page (low-latency setting) over the whole recording, on the GPU
under the lease. Commands:

```sh
PYTHONPATH=. python experiments/live/voice_check.py sanity <sessions> --lab data/transcriber/lab --work W
PYTHONPATH=. python experiments/live/voice_check.py embed  <sessions> --lab data/transcriber/lab --work W
PYTHONPATH=. python experiments/live/voice_check.py frames <sessions> --lab data/transcriber/lab --work W --device cuda
PYTHONPATH=. python experiments/live/voice_check.py score --work W   # -> experiments/live/voice_check_results.json
```

Share of sentences on the right side, 5 s check (after the last
revision unless said; CALLHOME 3,384 sentences, two-person calls 1,460,
AMI 1,815):

| Approach | CALLHOME | two-person calls | AMI pairs |
|---|---|---|---|
| Each sentence alone vs the check, cosine ≥ 0.89 (best on CALLHOME) | 0.691 | 0.717 | 0.527 |
| Same, vectors centred on the session's mean | 0.712 | 0.741 | 0.583 |
| Two references (check, and the other person learnt as heard) | 0.697 | 0.717 | 0.599 |
| The check's slot only | 0.734 | 0.792 | 0.634 |
| The slot nearest to the check (today's naming, check as voiceprint) | 0.806 | 0.792 | 0.598 |
| **Chosen: nearest slot ≥ 0.93, split ≥ 0.97, check slot until then** | **0.814** (0.780 at first emission) | **0.792** | **0.593** (0.605) |

The chosen rule by kind of sentence (CALLHOME, 5 s): clean 0.974,
overlapped 0.715; under 1 s 0.561, 1–2 s 0.836, 2–4 s 0.855, 4 s and
more 0.932. On the two-person calls clean sentences are 0.996: what is
left is overlap and very short sentences, where the slots themselves
are wrong. Check length (chosen rule, CALLHOME / AMI): 3 s 0.796 /
0.664, 5 s 0.814 / 0.593, 8 s 0.812 / 0.591. The threshold barely
matters on CALLHOME (0.779–0.813 flat from 0.50 to 0.93 at every
length; lower at 0.95+ with short checks); on AMI the higher values are
better, because there the check's slot is a better guess than any
comparison: 0.93 is the highest value that loses nothing on CALLHOME.
The split rule changes CALLHOME by +0.018 (3 s), +0.001 (5 s), 0 (8 s)
and AMI by 0 at 0.97; at 0.95 it costs up to 0.038 on CALLHOME (5 and
8 s). Sentences
sent `speaker_unsure` (their slot not compared yet, not the check's):
50 of 3,384 on CALLHOME, all shown as the other person and right as
often as the rest (0.820 vs 0.780); 127 of 1,815 on AMI, mostly wrong
(0.221 vs 0.634; 72 of them had no slot at all). The mark is a hint, a
reason to look at the correction, not a measured confidence.

Reading: telling the two people apart sentence by sentence with this
voice model is weak (≈ 0.69 on telephone speech, ≈ 0.53 in the room);
the slots carry the information and the check only has to say which
slot is the employee. The far-field room (AMI) stays poor with every
approach (0.53–0.66): Sortformer's slots there are much worse (DER
0.39 above), the check is taken in the middle of a four-person meeting
and the two other people add confusions; corrections are the way out
there.

Not measured: the owner's microphone or any real interview (the lab's
own recordings have no reference; the failed sessions kept no audio);
a check spoken at the very start of a session, when the employee is the
first and only voice (the simulation takes it in mid-conversation, so
the check's slot is likely worse here than live); the enrolment's help
together with the check (no enrolments for these corpora); the CPU
time of the check's embedding inside a running session; real
end-to-end latency with the check on.

## Not done

- Roles of anonymous voices beyond the order of first appearance
  (two-party products have the voice check instead, #249).
- More than two people on one microphone with the voice check
  (meetings): out of #249's scope.
- The CPU (degraded) mode of the worker was not measured.
- More than four voices: Sortformer has four slots; what happens with
  a fifth was not measured.
