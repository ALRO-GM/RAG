# Live measures: start delay, end-to-end latency, streaming vs batch, signal precision

Measured 2026-11-21 on the workstation (RTX 3070 Ti, shared with the running
services; every live session held the GPU lease). All audio is synthetic
(Piper), every number is one run: they say nothing about real calls.
Raw results: `experiments/live/` (`lv12/`, `lv13/`, `signal_results.json`, `signal_results_lv15.json`).
Tickets: #173 (LV-12), #174 (LV-13), #183, and a fix found on the way.

## 0. Start delay and `asr_behind` at start

Before: the audio socket said `ready` at once while the worker still placed
the session on the GPU and loaded the model, so the client's first ~6 s
queued and every session began with an `asr_behind` warning (2 of 2 runs,
`experiments/live/start_delay.py` against `main`). Now the socket says `ready`
only when the recogniser is loaded (`LiveSession.wait_ready`), as the live
page already waits for it: create to `ready` 5.6 to 7.0 s over 6 runs (about 4 s
of it is the lease's grace wait without an image worker, §GPU in
`LIVE_SESSION.md`), no warning in any run. The model is still loaded per
session; keeping it warm across sessions would save the load but not the grace
wait and was not done.

## 1. End-to-end latency (LV-12, #173)

Calls: `examples/platform/sim-soporte-lumbre` (108 s) and
`sim-entrevista-backend` (142 s), rendered by `experiments/live/render_calls.py`
with two Piper voices, one per channel (stereo), and the channels summed
(mono). `experiments/live/live_latency.py` sends the wav in real time (100 ms
chunks) to the audio socket of a live session; an orchestrator (spare port,
real RAG `POST /knowledge/query` and `/reason`, warm Haiku pool) follows it,
and the client times each stage on its own clock from the SSE stream.
"Speech end" is the moment the microphone would have delivered the last
sample of the reference sentence.

| Run | Haiku calls (ok / no suggestion / timeout) | speech end → final p50 / p90 | knowledge p50 | reasoner p50 / p90 | final → on screen p50 / p90 | **speech end → on screen p50 / p90** |
|---|---|---|---|---|---|---|
| support, stereo | 13 (9 / 2 / 2) | 753 / 966 ms (31 finals) | 10 ms | 2184 / 2411 ms | 2194 / 2424 ms | **2697 / 3127 ms** |
| interview, stereo | 16 (9 / 5 / 2) | 656 / 768 ms (34 finals) | 9 ms | 1262 / 1523 ms | 1273 / 1533 ms | **1509 / 1951 ms** |
| support, mono, first 45 s | 11 (9 / 2 / 0) | 591 / 739 ms (14 finals) | 9 ms | 1320 / 1705 ms | 1328 / 1715 ms | **2065 / 2411 ms** |

40 real Haiku calls in all. The stage columns count the `ok` calls only (9
per run); a timeout appears on screen as nothing after the product's
deadline (2.8 s support, 1.8 s interview), and in both stereo runs 2 calls hit
it. The orchestrator adds nothing measurable between the final and the call
(the part of "final → on screen" that is neither knowledge nor reasoner is
0 to 4 ms) and retrieval is 9 to 27 ms: the reasoner is the whole cost after
the recogniser. Speech end → final is the endpoint hang plus the final decode
plus the relay (LIVE_ASR measured 674 ms p50 without the relay, mono, 882 s).
The first run (support, stereo) had the slowest reasoner (p50 2.2 s against
1.3 s in the later two): the support template is longer, but a cold start of
the Haiku pool in that first run was not ruled out and one run each cannot
tell. Mono sessions are one channel heard as the customer (the diarization of
LV-6 is not done), so every sentence of the mixed call is a customer sentence
there. No `asr_behind` or `gpu_degraded` warning in any run.

## 2. Streaming against batch (LV-13, #174)

`experiments/live/stream_vs_batch.py` compares the transcript a live session
handed over with the batch pipeline's transcript (`transcriber import --wait`
on the mono mix: whisper-large-v3-turbo fp16 and the batch diarizer) and with
the text Piper read. Word error rate as in the lab (`normalise_text`).

| Call | live vs script | batch vs script | live vs batch | speaker accuracy, live stereo / batch |
|---|---|---|---|---|
| support, stereo (108 s) | 0.044 | 0.038 | 0.013 | 1.00 / 1.00 (31 / 28 sentences) |
| interview, stereo (142 s) | 0.108 | 0.095 | 0.085 | 1.00 / 1.00 (34 / 35 sentences) |
| support, mono, first 40 s | 0.071 | 0.056 | 0.032 | not available (no mono diarization live) |
| `sim-reunion-verano`, mono, 882 s (`LIVE_ASR.md`) | 0.123 | 0.088 | 0.126 | n/a |

Reading: the stream is 1 to 5 points of WER behind the batch on these calls,
and on a clean synthetic channel pair speaker accuracy is perfect in both:
live stereo gets it by construction (channel n is speaker n), and the batch
diarizer also separates two non-overlapping Piper voices. Neither says
anything about crosstalk or about real voices: these calls have no overlap
across channels and no channel leakage. In mono the live session gives every
sentence one speaker; the 0.54 of `experiments/live/lv13/support_mono_40s.json`
is that, not diarization.

CALLHOME Spanish (lab data, `data/transcriber/lab-corpora/callhome/spa`) has
speaker turns but no text, so only the agreement between the two recognisers
exists: first 120 s of two sessions, live vs batch WER 0.353 and 0.216 (261 and
331 live tokens). Telephone Spanish is a much harder input than Piper, and
the live number is "how far from the batch", not an error rate against a truth.
Two other sessions of the first parquet file (rows 1 and 2) are full-scale
noise in the dataset (their batch transcript is "Gracias por ver el video")
and were left out; a third clip's batch run was abandoned because the GPU was
busy with another job (not measured). No live diarization, so no speaker
accuracy on CALLHOME.

## 3. Precision of each customer signal (#183)

`experiments/live/signal_snippets.json`: 60 synthetic support snippets (an agent
line and a customer line; 8 per signal and 12 with none), labelled by the author
before the run. `experiments/live/signal_precision.py` sends each once to the
RAG's `/reason` with the product's settings (`suggester.support@1`, one
suggestion), reads the signals as the live loop does (`expressed`: only those
backed by a customer sentence), 60 Haiku calls, no retries, template unchanged.

| Signal | shown | correct | gold | precision | recall |
|---|---|---|---|---|---|
| complaint | 13 | 6 | 8 | 0.46 | 0.75 |
| stated_frustration | 0 | 0 | 8 | none shown | 0.00 |
| urgency | 8 | 8 | 8 | 1.00 | 1.00 |
| churn_risk | 4 | 4 | 8 | 1.00 | 0.50 |
| confusion | 7 | 6 | 8 | 0.86 | 0.75 |
| satisfaction | 7 | 7 | 8 | 1.00 | 0.88 |

Confusion matrix (gold in rows, signals shown in columns; "none" is no signal
shown):

| gold \ shown | complaint | stated_frustration | urgency | churn_risk | confusion | satisfaction | none |
|---|---|---|---|---|---|---|---|
| complaint | 6 | 0 | 0 | 0 | 0 | 0 | 2 |
| stated_frustration | 7 | 0 | 0 | 0 | 0 | 0 | 1 |
| urgency | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| churn_risk | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| confusion | 0 | 0 | 0 | 0 | 6 | 0 | 2 |
| satisfaction | 0 | 0 | 0 | 0 | 0 | 7 | 1 |
| none | 0 | 0 | 0 | 0 | 1 | 0 | 11 |

Reading: when a signal is shown it is mostly the right one (urgency,
churn_risk, satisfaction 1.00, confusion 0.86), with no wrong label across
the other pairs. Two problems: (a) `stated_frustration` is never produced:
7 of its 8 snippets come out as `complaint` (precision of complaint 0.46) and
the model does not use the label at all; the template's only example of the
shape is `"signals": ["complaint"]` and the system text names the six labels
without defining them, which is the likely cause (not tested); (b) 10 of the 60
calls answered `no_suggestion` (4 of the 8 churn_risk snippets), so no signal
could be shown: recall is limited by the one-suggestion budget as much as by
labelling. One false signal on a snippet with no signal (a question about the
return window read as confusion, 1 of 12).

### LV-15 (#196): definitions and one example per signal

`suggester.support@1` now carries a short definition and one example customer
line per signal (`SIGNAL_DEFINITIONS` in `src/domain/reasoning/templates.py`),
says that a fault without annoyance words is only `complaint` and annoyance
words are `stated_frustration`, and its JSON shape example is `urgency`
instead of `complaint`. The cap stays 200 output tokens and the call keeps
`--effort low`. The same 60 snippets were sent again (60 Haiku calls plus 10
for a latency baseline plus 2 from an aborted first start: 72 of the 80
allowed), through a RAG started from the branch on another port
(`experiments/live/signal_results_lv15.json`).

| Signal | precision before | precision after | recall before | recall after |
|---|---|---|---|---|
| complaint | 0.46 (6/13) | 0.70 (7/10) | 0.75 | 0.88 |
| stated_frustration | none shown | 0.86 (6/7) | 0.00 | 0.75 |
| urgency | 1.00 | 0.88 (7/8) | 1.00 | 0.88 |
| churn_risk | 1.00 | 1.00 (6/6) | 0.50 | 0.75 |
| confusion | 0.86 | 0.88 (7/8) | 0.75 | 0.88 |
| satisfaction | 1.00 | 1.00 (5/5) | 0.88 | 0.62 |

Answers: `ok` 48, `no_suggestion` 12 (before: 10), no timeout. Snippets with no
signal that stayed clean: 11 of 12 (same one false `confusion` as before).
Reading: the definitions make the model use `stated_frustration`
(6 of 8 right, 3 still read as `complaint`); complaint precision goes from
0.46 to 0.70. Costs: urgency lost one (7/8) and satisfaction recall fell to
5/8 (3 snippets with no signal shown, as the one-suggestion budget allows
`no_suggestion`); with 8 snippets per signal one snippet is 0.125, so these
small moves are inside the noise and only the frustration/complaint change
is large. One run, no repetition.

Latency: over the 60 calls the end-to-end `/reason` time was median 1.41 s,
mean 1.49 s, max 2.78 s (deadline 2.8 s). For the baseline the old template
was run on the first 10 snippets only (median 1.04 s, mean 1.25 s, 1 of the
10 timed out and 3 were `no_suggestion`); the same 10 with the new template:
median 1.74 s, mean 1.69 s. So the longer prompt costs roughly +0.4 to +0.7
s on that sample of 10, an unreliable estimate (n=10, different server
processes: the baseline ran on the production RAG, the new one on a CPU-only
copy started for the run). #183's own run did not record times.

The template does not change the answer's size (cap and `--effort low` as
before). Synthetic text by the author of the labels: the real precision on real
customers is unknown.

## Cost and limits

Haiku calls: 40 (LV-12) + 60 (#183) + 72 (LV-15) = 172, through the owner's subscription;
no paid API. Not done: LV-12 on a real microphone and a real network (the
audio and SSE are on loopback), several repetitions, the mono diarization,
and a CALLHOME truth text.

## The tick loop with the coverage of #286 (orchestrator side, 2026-11-27)

What #286 adds to each final and each tick call, with an instant fake
model (so the model's own latency is not in it), CPU, 4,800 finals of an
invented round (200 replays of 24 lines), 2,400 tick calls; before =
main `c7a190d`, after = the #286 branch with both reviews' fixes, three
runs each (`experiments/live/lv286/tick_cost.py <checkout>` with that
checkout on `PYTHONPATH`):

| | before p50 / p90 | after p50 / p90 |
|---|---|---|
| `TopicTracker.on_event` per final | 86 / 92 µs | 264–269 / 299–303 µs |
| one tick call's handling around the model | 9 / 53–54 µs | 12–23 / 205–211 µs |

Both stay under a millisecond against 0.6–0.9 s from speech end to final
and seconds of model. The tick call (`interview.ticks@2`) runs on its own
thread on the cold backend and yields, beside the suggestions; its
prompt grows from 543 to 841 system characters and its output cap from
160 to 260 tokens. Its real latency with the model was not measured
(it needs the running services).
