import test from "node:test";
import assert from "node:assert/strict";
import {
  corpusOf, tallyOf, stepTotal, groupByCorpus, overviewTotals, anyActive, barSegments, failureCodes,
  pendingGroups, niceMax, linearScale, axisTicks, summaryDenoisers, derByDiarizer, derByDenoiser,
  agreementByDiarizer, chooseExcerpt, sampleRows, splitExcluded, turnsToSpans,
} from "../../transcriber/adapters/inbound/web/js/lib/lab-progress.js";

function tally(done, running, pending, failed, codes = {}, open = []) {
  return { done, running, pending, failed, failed_codes: codes, open };
}

const ENTRIES = [
  { session_id: "callhome-data-0002", corpus: "callhome", run_id: "r3", run_state: "done", steps: tally(10, 0, 0, 0) },
  { session_id: "rec-1", corpus: null, run_id: "r1", run_state: "running",
    steps: tally(5, 1, 3, 2, { out_of_memory: 1, timeout: 1 }, [
      { name: "diarize:none+b", stage: "diarize", model: "b", denoiser: "none", state: "pending" },
      { name: "diarize:none+c", stage: "diarize", model: "c", denoiser: "none", state: "failed", code: "timeout", message: "slow" },
      { name: "diarize:none+a", stage: "diarize", model: "a", denoiser: "none", state: "running" },
    ]) },
  { session_id: "ami-B", corpus: "ami", run_id: "r2", run_state: "partial",
    steps: tally(4, 0, 0, 3, { out_of_memory: 3 }, [
      { name: "diarize:none+d", stage: "diarize", model: "d", denoiser: "none", state: "failed", code: "out_of_memory", message: "OOM" },
    ]) },
  { session_id: "ami-A", corpus: "ami", run_id: null, run_state: null, steps: null },
  { session_id: "zz-1", corpus: "other", run_id: null, steps: null },
];

test("sessions are grouped by corpus: own, AMI, CALLHOME, then the rest", () => {
  const groups = groupByCorpus(ENTRIES);
  assert.deepEqual(groups.map((g) => g.corpus), ["own", "ami", "callhome", "other"]);
  assert.deepEqual(groups[1].sessions.map((s) => s.session_id), ["ami-A", "ami-B"]);
  assert.equal(corpusOf({ corpus: null }), "own");
  assert.deepEqual(tallyOf({ steps: null }).open, []);
});

test("totals over every session", () => {
  const totals = overviewTotals(ENTRIES);
  assert.deepEqual(totals, {
    done: 19, running: 1, pending: 3, failed: 5, left: 9, steps: 28, runsRunning: 1, notStarted: 2, sessions: 5,
  });
  assert.equal(anyActive(ENTRIES), true);
  assert.equal(anyActive(ENTRIES.filter((e) => e.run_state !== "running")), false);
  assert.equal(anyActive([{ run_state: "queued" }]), true);
});

test("a stacked bar splits the failures by code and fills [0, 1]", () => {
  const segments = barSegments(ENTRIES[1].steps);
  assert.deepEqual(segments.map((s) => [s.kind, s.code, s.count]), [
    ["done", null, 5], ["running", null, 1], ["pending", null, 3], ["failed", "out_of_memory", 1], ["failed", "timeout", 1],
  ]);
  assert.equal(segments[0].start, 0);
  const last = segments.at(-1);
  assert.ok(Math.abs(last.start + last.width - 1) < 1e-12);
  assert.equal(stepTotal(ENTRIES[1].steps), 11);
  assert.deepEqual(barSegments(tally(0, 0, 0, 0)), []);
  // a failure count without codes is shown as unknown
  assert.deepEqual(barSegments(tally(1, 0, 0, 2, { x: 1 })).map((s) => s.code), [null, "x", "unknown"]);
  assert.deepEqual(failureCodes(ENTRIES), [{ code: "out_of_memory", count: 4 }, { code: "timeout", count: 1 }]);
});

test("pending work: sessions with failures first, failed steps first", () => {
  const groups = pendingGroups(ENTRIES);
  assert.deepEqual(groups.map((g) => g.entry.session_id), ["ami-B", "rec-1"]);
  assert.deepEqual(groups[1].steps.map((s) => s.state), ["failed", "running", "pending"]);
  assert.equal(groups[1].active, true);
  assert.equal(groups[0].active, false);
});

test("chart scales", () => {
  assert.equal(niceMax(0.63), 1);
  assert.equal(niceMax(0.42), 0.5);
  assert.equal(niceMax(0.18), 0.2);
  assert.equal(niceMax(0.25), 0.25);
  assert.equal(niceMax(1.3), 2);
  assert.equal(niceMax(0), 1);
  assert.equal(niceMax(Number.NaN), 1);
  const x = linearScale(0.5, 100, 300);
  assert.equal(x(0), 100);
  assert.equal(x(0.25), 200);
  assert.equal(x(0.9), 300);
  assert.equal(x(-1), 100);
  assert.deepEqual(axisTicks(0.5, 5), [0, 0.1, 0.2, 0.3, 0.4, 0.5]);
});

const SUMMARY = {
  models: [
    { kind: "diarization", id: "none+a", metrics: { der: { n: 3, mean: 0.3, min: 0.2, max: 0.4 } } },
    { kind: "diarization", id: "none+b", metrics: { der: { n: 3, mean: 0.2, min: 0.1, max: 0.5 } } },
    { kind: "diarization", id: "none+c", metrics: { der: { n: 0, mean: null, min: null, max: null } } },
    { kind: "diarization", id: "dn+a", metrics: { der: { n: 3, mean: 0.5, min: 0.4, max: 0.6 } } },
    { kind: "diarization", id: "dn+b", metrics: { der: { n: 3, mean: 0.3, min: 0.2, max: 0.4 } } },
    { kind: "diarization", id: "xx+a", metrics: { der: { n: 1, mean: 0.1, min: 0.1, max: 0.1 } } },
    { kind: "pipeline", id: "none+a+-", metrics: { der: { n: 3, mean: 0.3 } } },
  ],
};

test("DER per diarizer and per denoiser from the corpus summary", () => {
  assert.deepEqual(summaryDenoisers(SUMMARY), ["none", "dn", "xx"]);
  assert.deepEqual(derByDiarizer(SUMMARY, "none"), [
    { id: "b", mean: 0.2, min: 0.1, max: 0.5, n: 3 },
    { id: "a", mean: 0.3, min: 0.2, max: 0.4, n: 3 },
    { id: "c", mean: null, min: null, max: null, n: 0 },
  ]);
  assert.deepEqual(derByDiarizer(SUMMARY, "dn").map((r) => r.id), ["b", "a"]);
  // only "a" ran on every denoiser: each denoiser is compared on it
  assert.deepEqual(derByDenoiser(SUMMARY), [
    { id: "xx", mean: 0.1, min: 0.1, max: 0.1, n: 1 },
    { id: "none", mean: 0.3, min: 0.3, max: 0.3, n: 1 },
    { id: "dn", mean: 0.5, min: 0.5, max: 0.5, n: 1 },
  ]);
  const two = { models: SUMMARY.models.filter((m) => !m.id.startsWith("xx")) };
  const rows = derByDenoiser(two);
  assert.deepEqual(rows.map((r) => r.id), ["none", "dn"]);
  assert.ok(Math.abs(rows[0].mean - 0.25) < 1e-12);
  assert.deepEqual([rows[1].min, rows[1].max, rows[1].n], [0.3, 0.5, 2]);
  assert.deepEqual(derByDenoiser({ models: [] }), []);
  assert.deepEqual(derByDiarizer(null), []);
});

const RESULTS = {
  tracks: [
    { id: "none+a", denoiser: "none", diarizer: "a", agreement_with_consensus: 0.8 },
    { id: "none+b", denoiser: "none", diarizer: "b", agreement_with_consensus: 0.9 },
    { id: "dn+a", denoiser: "dn", diarizer: "a", agreement_with_consensus: 0.6 },
    { id: "dn+b", denoiser: "dn", diarizer: "b", agreement_with_consensus: null },
  ],
  reference: {
    tracks: [
      { track: "none+a", denoiser: "none", diarizer: "a", der: 0.3 },
      { track: "none+b", denoiser: "none", diarizer: "b", der: 0.2 },
      { track: "dn+a", der: 0.5 },
    ],
  },
};

test("agreement per diarizer and the figures beside each sample", () => {
  assert.deepEqual(agreementByDiarizer(RESULTS).map((r) => [r.id, r.mean]), [["b", 0.9], ["a", 0.8]]);
  const rows = sampleRows(RESULTS);
  assert.deepEqual(rows.map((r) => r.denoiser), ["none", "dn"]);
  assert.ok(Math.abs(rows[0].der - 0.25) < 1e-12);
  assert.deepEqual([rows[0].bestDer, rows[0].bestDiarizer], [0.2, "b"]);
  // a corpus recording: the consensus names nobody, so no agreement is shown
  assert.equal(rows[0].agreement, null);
  assert.deepEqual([rows[1].der, rows[1].bestDiarizer, rows[1].agreement], [0.5, "a", null]);
  const own = sampleRows({ tracks: RESULTS.tracks, reference: null });
  assert.equal(own[0].der, null);
  assert.equal(own[0].bestDiarizer, null);
  assert.ok(Math.abs(own[0].agreement - 0.85) < 1e-12);
  assert.equal(own[1].agreement, 0.6);
});

test("the excerpt goes where people overlap or take turns, else 10 % in", () => {
  const spans = [
    { start: 0, end: 40, speakers: [] },
    { start: 40, end: 50, speakers: ["A"] },
    { start: 50, end: 52, speakers: ["B"] },
    { start: 52, end: 60, speakers: ["A"] },
    { start: 100, end: 104, speakers: ["A", "B"] },
    { start: 104, end: 110, speakers: ["A"] },
  ];
  assert.deepEqual(chooseExcerpt(spans, 300, 20), { start: 100, end: 120, reason: "overlap" });
  const turns = spans.filter((s) => s.speakers.length < 2);
  assert.deepEqual(chooseExcerpt(turns, 300, 20), { start: 40, end: 60, reason: "changes" });
  // no consensus, or one voice all along: 10 % in
  assert.deepEqual(chooseExcerpt([], 300, 20), { start: 30, end: 50, reason: "default" });
  assert.deepEqual(chooseExcerpt([{ start: 0, end: 300, speakers: ["A"] }], 300), { start: 30, end: 50, reason: "default" });
  // the window never runs past the end, nor is longer than the recording
  assert.deepEqual(chooseExcerpt([{ start: 290, end: 295, speakers: ["A", "B"] }], 300, 20), { start: 280, end: 300, reason: "overlap" });
  assert.deepEqual(chooseExcerpt(null, 12, 20), { start: 0, end: 12, reason: "default" });
  assert.deepEqual(chooseExcerpt([], 100, 95), { start: 5, end: 100, reason: "default" });
});

test("reference turns become spans with every voice at each moment", () => {
  const spans = turnsToSpans([
    { start: 0, end: 4, speaker: "A" }, { start: 3, end: 8, speaker: "B" }, { start: 10, end: 12, speaker: "A" },
  ]);
  assert.deepEqual(spans, [
    { start: 0, end: 3, speakers: ["A"] }, { start: 3, end: 4, speakers: ["A", "B"] },
    { start: 4, end: 8, speakers: ["B"] }, { start: 10, end: 12, speakers: ["A"] },
  ]);
  assert.deepEqual(turnsToSpans(null), []);
  assert.deepEqual(chooseExcerpt(spans, 100, 5), { start: 0, end: 5, reason: "overlap" });
});

test("recordings without speech are left out and counted per reason", () => {
  const entries = [
    { session_id: "callhome-data-0001", corpus: "callhome", excluded: null },
    { session_id: "callhome-data-0002", corpus: "callhome", excluded: "no speech in the audio" },
    { session_id: "callhome-data-0003", corpus: "callhome", excluded: "no speech in the audio" },
    { session_id: "rec-1", corpus: null },
  ];
  const { kept, excluded } = splitExcluded(entries);
  assert.deepEqual(kept.map((e) => e.session_id), ["callhome-data-0001", "rec-1"]);
  assert.deepEqual(excluded, [{ reason: "no speech in the audio", n: 2 }]);
  assert.deepEqual(splitExcluded([]), { kept: [], excluded: [] });
});
