import test from "node:test";
import assert from "node:assert/strict";
import {
  groupCatalogue, defaultSelection, runRequest, planCounts, groupAnswers, textVariants,
  speakerColours, allSpeakers, timeScale, ticks, clampWindow, zoomWindow, scrollWindow, visible,
  overlaps, playStart, sortRows, idPart, pairMatrix, heatClass, denoiserTable, bestBy,
  coveringLabel, placeLabel, clearSpan, makeLabel, labelsBody, labelProgress, LabelQueue,
  keyAction, exportPayload, isUnmatched, answerKey,
} from "../../transcriber/adapters/inbound/web/js/lib/lab.js";

const MODELS = [
  { id: "none", stage: "denoise", state: "ready" },
  { id: "dn", stage: "denoise", state: "ready" },
  { id: "builtin", stage: "diarize", state: "ready" },
  { id: "gated", stage: "diarize", state: "needs_token" },
  { id: "z", stage: "diarize", state: "ready" },
  { id: "r", stage: "recognise", state: "ready" },
  { id: "absent", stage: "recognise", state: "not_installed" },
];

test("catalogue grouped by stage and the default selection", () => {
  const groups = groupCatalogue(MODELS);
  assert.deepEqual(groups.diarize.map((m) => m.id), ["builtin", "gated", "z"]);
  assert.deepEqual([...defaultSelection(MODELS)], ["none", "dn", "builtin", "z", "r"]);
});

test("request and counts follow the service's plan", () => {
  const request = runRequest(defaultSelection(MODELS), MODELS);
  assert.deepEqual(request, { denoisers: ["none", "dn"], diarizers: ["builtin", "z"], recognisers: ["r"] });
  // 2 denoisers x 2 diarizers x 1 recogniser; steps: 1 denoise + 2 x (2 + 1)
  assert.deepEqual(planCounts(request), { pipelines: 4, steps: 7, valid: true });
  const bare = runRequest(new Set(["r"]), MODELS);
  assert.deepEqual(bare.denoisers, ["none"]);
  assert.deepEqual(planCounts(bare), { pipelines: 1, steps: 1, valid: true });
  assert.equal(planCounts(runRequest(new Set(["dn"]), MODELS)).valid, false);
});

test("identical answers are grouped, most common first", () => {
  const groups = groupAnswers({
    "none+a+r": { speakers: ["A"] }, "dn+a+r": { speakers: ["A"] },
    "none+b+r": { speakers: ["B", "A"] }, "none+c+r": { speakers: [] }, "dn+b+r": ["A", "B"],
  });
  assert.deepEqual(groups.map((g) => [g.key, g.count]), [["A", 2], ["A+B", 2], ["", 1]]);
  assert.deepEqual(groups[1].pipelines, ["dn+b+r", "none+b+r"]);
  assert.equal(answerKey(["B", "A"]), "A+B");
  assert.deepEqual(groupAnswers(null), []);
});

test("text variants are counted per denoised recogniser", () => {
  const variants = textVariants({
    "none+a+r": { text: "hola  que tal" }, "none+b+r": { text: "hola que tal" },
    "none+a+w": { text: "hola, que tal" }, "none+a+-": { text: "" },
  });
  assert.deepEqual(variants, [
    { text: "hola que tal", recognisers: ["none+r"], pipelines: 2 },
    { text: "hola, que tal", recognisers: ["none+w"], pipelines: 1 },
  ]);
});

test("speaker colours: same person same colour, unmatched voices neutral", () => {
  const speakers = allSpeakers({ t1: [{ speaker: "~x" }, { speaker: "B" }], t2: [{ speaker: "A" }] });
  assert.deepEqual(speakers, ["A", "B", "~x"]);
  const colours = speakerColours(speakers);
  assert.equal(colours.get("A"), "spk-0");
  assert.equal(colours.get("B"), "spk-1");
  assert.equal(colours.get("~x"), "anon-0");
  assert.ok(isUnmatched("~x") && !isUnmatched("A"));
  const many = speakerColours(Array.from({ length: 10 }, (_, i) => `p${i}`));
  assert.equal(many.get("p8"), "spk-0");
});

test("time scale, ticks and the zoom window", () => {
  const scale = timeScale(10, 20, 200);
  assert.equal(scale.x(15), 100);
  assert.equal(scale.time(50), 12.5);
  assert.deepEqual(ticks(0, 60, 6), [0, 10, 20, 30, 40, 50, 60]);
  assert.deepEqual(ticks(0, 3, 6), [0, 0.5, 1, 1.5, 2, 2.5, 3]);
  assert.deepEqual(clampWindow(-5, 10, 100), { start: 0, end: 15 });
  assert.deepEqual(clampWindow(95, 110, 100), { start: 85, end: 100 });
  assert.deepEqual(clampWindow(0, 0.1, 100, 1), { start: 0, end: 1 });
  assert.deepEqual(zoomWindow({ start: 0, end: 100 }, 0.5, 50, 100), { start: 25, end: 75 });
  assert.deepEqual(zoomWindow({ start: 25, end: 75 }, 4, 50, 100), { start: 0, end: 100 });
  assert.deepEqual(scrollWindow({ start: 0, end: 10 }, 0.5, 100), { start: 5, end: 15 });
  assert.deepEqual(scrollWindow({ start: 0, end: 10 }, -1, 100), { start: 0, end: 10 });
});

test("visible segments, overlaps and the play start", () => {
  const segments = [{ start: 0, end: 5, speaker: "A" }, { start: 4, end: 8, speaker: "B" }, { start: 6, end: 7, speaker: "A" }, { start: 9, end: 10, speaker: "A" }];
  assert.equal(visible(segments, 5, 9).length, 2);
  assert.deepEqual(overlaps(segments), [{ start: 4, end: 5 }, { start: 6, end: 7 }]);
  assert.deepEqual(overlaps([]), []);
  assert.equal(playStart(10, 2), 8);
  assert.equal(playStart(1, 2), 0);
  assert.equal(playStart(3, "x"), 3);
});

test("sortRows keeps missing values last in both directions", () => {
  const rows = [{ v: 2 }, { v: null }, { v: 1 }, { v: 3 }];
  assert.deepEqual(sortRows(rows, "v").map((r) => r.v), [1, 2, 3, null]);
  assert.deepEqual(sortRows(rows, "v", true).map((r) => r.v), [3, 2, 1, null]);
  assert.deepEqual(sortRows([{ s: "b" }, { s: "a" }], "s").map((r) => r.s), ["a", "b"]);
});

test("ids, the pairwise matrix and heat classes", () => {
  assert.deepEqual(idPart(["none+a", "dn+a", "none+b"], 0), ["none", "dn"]);
  assert.deepEqual(idPart(["none+a", "dn+a", "none+b"], 1), ["a", "b"]);
  const m = pairMatrix([{ reference: "x", hypothesis: "y", agreement: 0.8 }], ["x", "y", "z"], "agreement");
  assert.equal(m.value("x", "x"), 1);
  assert.equal(m.value("x", "y"), 0.8);
  assert.equal(m.value("y", "x"), 0.8);
  assert.equal(m.value("x", "z"), null);
  assert.equal(heatClass(0.85), "heat-8");
  assert.equal(heatClass(1), "heat-9");
  assert.equal(heatClass(-1), "heat-0");
  assert.equal(heatClass(null), "heat-none");
});

test("denoiser effect table and best choice per group", () => {
  const table = denoiserTable([
    { denoiser: "dn", stage: "recognise", model: "r", wer: 0.1 },
    { denoiser: "dn", stage: "diarize", model: "z", agreement: 0.9 },
    { denoiser: "x", stage: "recognise", model: "w", wer: 0.2 },
  ], "recognise", "wer");
  assert.deepEqual(table.columns, ["r", "w"]);
  assert.deepEqual(table.rows, [{ denoiser: "dn", cells: { r: 0.1 } }, { denoiser: "x", cells: { w: 0.2 } }]);
  const best = bestBy([
    { pipeline: "none+z+r", error_rate: 0.3 }, { pipeline: "dn+z+r", error_rate: 0.2 },
    { pipeline: "none+y+r", error_rate: 0.1 }, { pipeline: "dn+y+r", error_rate: null },
  ], 1, 0, "error_rate");
  assert.deepEqual(best.map((b) => [b.group, b.choice, b.value]), [["y", "none", 0.1], ["z", "dn", 0.2]]);
  assert.equal(bestBy([{ pipeline: "a+b+c", s: 0.5 }, { pipeline: "d+b+c", s: 0.7 }], 1, 0, "s", true)[0].choice, "d");
});

test("labels never overlap: placing cuts what was there", () => {
  let labels = placeLabel([], { start: 0, end: 10, kind: "nobody", speakers: [] });
  labels = placeLabel(labels, { start: 4, end: 6, kind: "person", speakers: ["A"] });
  assert.deepEqual(labels.map((l) => [l.start, l.end, l.kind]), [[0, 4, "nobody"], [4, 6, "person"], [6, 10, "nobody"]]);
  labels = clearSpan(labels, 4, 6);
  assert.deepEqual(labels.map((l) => [l.start, l.end]), [[0, 4], [6, 10]]);
  assert.equal(coveringLabel(labels, 1, 2).kind, "nobody");
  assert.equal(coveringLabel(labels, 4, 6), null);
});

test("makeLabel checks the people a kind needs", () => {
  const f = { start: 1, end: 2 };
  assert.deepEqual(makeLabel(f, "person", ["A"], "  hola "), { start: 1, end: 2, kind: "person", speakers: ["A"], text: "hola" });
  assert.equal(makeLabel(f, "person", []), null);
  assert.equal(makeLabel(f, "both", ["A"]), null);
  assert.deepEqual(makeLabel(f, "both", ["B", "A"]).speakers, ["A", "B"]);
  assert.deepEqual(makeLabel(f, "nobody", ["A"]).speakers, []);
  assert.equal(makeLabel(f, "unsure").text, null);
  const body = labelsBody([{ start: 1, end: 2, kind: "unsure", extra: 1 }]);
  assert.deepEqual(body, { labels: [{ start: 1, end: 2, kind: "unsure", speakers: [], text: null }] });
});

test("label progress counts seconds and covered fragments", () => {
  const spans = [{ start: 0, end: 2 }, { start: 2, end: 4 }, { start: 4, end: 8 }];
  const p = labelProgress([{ start: 0, end: 3.5, kind: "nobody" }], spans, 8);
  assert.deepEqual(p, { seconds: 3.5, duration: 8, fragments: 2, total: 3 });
});

function fakeTimers() {
  const pending = new Map();
  let next = 1;
  return {
    setTimer: (fn, ms) => { pending.set(next, { fn, ms }); return next++; },
    clearTimer: (id) => pending.delete(id),
    pending,
    async fire() {
      const [[id, { fn }]] = pending;
      pending.delete(id);
      await fn();
    },
  };
}

function memoryStorage() {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
    data,
  };
}

test("the label queue debounces, saves and clears the local copy", async () => {
  const timers = fakeTimers();
  const storage = memoryStorage();
  const sent = [];
  const states = [];
  const queue = new LabelQueue({
    save: async (labels) => { sent.push(labels); return labels.map((l) => ({ ...l, updated_at: "t" })); },
    storage, key: "k", ...timers, onChange: (s) => states.push(s),
  });
  assert.equal(queue.load([{ start: 0, end: 1, kind: "nobody" }]), false);
  queue.set([{ start: 0, end: 1, kind: "unsure" }]);
  queue.set([{ start: 0, end: 2, kind: "unsure" }]);
  assert.equal(timers.pending.size, 1, "one debounce timer");
  assert.ok(storage.data.has("k"), "kept locally until saved");
  await timers.fire();
  assert.equal(sent.length, 1);
  assert.equal(sent[0][0].end, 2);
  assert.equal(queue.state, "saved");
  assert.equal(queue.labels[0].updated_at, "t");
  assert.ok(!storage.data.has("k"));
  assert.deepEqual(states.slice(-2), ["saving", "saved"]);
});

test("the label queue keeps failed labels and retries with back-off", async () => {
  const timers = fakeTimers();
  const storage = memoryStorage();
  let fail = true;
  const queue = new LabelQueue({
    save: async (labels) => { if (fail) throw Object.assign(new Error("down"), { kind: "unavailable" }); return labels; },
    storage, key: "k", retry: [100, 200], ...timers,
  });
  queue.load([]);
  queue.set([{ start: 0, end: 1, kind: "nobody" }]);
  await timers.fire();
  assert.equal(queue.state, "failed");
  assert.equal([...timers.pending.values()][0].ms, 100);
  await timers.fire();
  assert.equal([...timers.pending.values()][0].ms, 200);
  assert.ok(storage.data.has("k"), "nothing lost");
  // A reload finds the unsaved labels and sends them.
  const again = new LabelQueue({ save: async (l) => l, storage, key: "k", ...fakeTimers() });
  assert.equal(again.load([]), true);
  assert.equal(again.labels.length, 1);
  assert.equal(again.state, "pending");
  fail = false;
  await timers.fire();
  assert.equal(queue.state, "saved");
});

test("a change during a save is saved afterwards", async () => {
  const timers = fakeTimers();
  let release;
  const sent = [];
  const queue = new LabelQueue({
    save: (labels) => { sent.push(labels); return new Promise((resolve) => { release = () => resolve(labels); }); },
    ...timers,
  });
  queue.load([]);
  queue.set([{ start: 0, end: 1, kind: "nobody" }]);
  const first = timers.fire();
  queue.set([{ start: 0, end: 1, kind: "unsure" }]);
  release();
  await first;
  assert.equal(queue.state, "pending");
  await timers.fire();
  release();
  await new Promise((r) => setImmediate(r));
  assert.equal(sent.length, 2);
  assert.equal(sent[1][0].kind, "unsure");
  assert.equal(queue.state, "saved");
});

test("a refused save (client error) is not retried in a loop", async () => {
  const timers = fakeTimers();
  const queue = new LabelQueue({ save: async () => { throw Object.assign(new Error("bad"), { kind: "client" }); }, ...timers });
  queue.load([]);
  queue.set([]);
  await timers.fire();
  assert.equal(queue.state, "failed");
  assert.equal(timers.pending.size, 0);
});

test("the keyboard map", () => {
  const key = (k, extra = {}) => keyAction({ key: k, target: { tagName: "LI" }, ...extra }, 2);
  assert.deepEqual(key("1"), { action: "person", index: 0 });
  assert.deepEqual(key("2"), { action: "person", index: 1 });
  assert.equal(key("3"), null);
  assert.deepEqual(key("B"), { action: "both" });
  assert.deepEqual(key("n"), { action: "nobody" });
  assert.deepEqual(key("u"), { action: "unsure" });
  assert.deepEqual(key("x"), { action: "clear" });
  assert.deepEqual(key(" "), { action: "play" });
  assert.deepEqual(key("j"), { action: "next" });
  assert.deepEqual(key("k"), { action: "previous" });
  assert.equal(key("1", { ctrlKey: true }), null);
  assert.equal(keyAction({ key: "1", target: { tagName: "INPUT" } }), null);
  assert.equal(keyAction({ key: "1", target: { tagName: "TEXTAREA" } }), null);
  assert.equal(keyAction(null), null);
});

test("the export holds labels and scores", () => {
  const out = exportPayload("s", "r", [{ start: 0, end: 1, kind: "nobody" }], { scores: [] }, new Date("2026-01-01T00:00:00Z"));
  assert.equal(out.schema_name, "lab.review.v1");
  assert.equal(out.exported_at, "2026-01-01T00:00:00.000Z");
  assert.equal(out.labels[0].kind, "nobody");
  assert.deepEqual(out.scores, { scores: [] });
});
