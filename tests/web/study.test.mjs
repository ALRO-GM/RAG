// The study page's pure logic, checked against the committed data file
// (study/data.json, built by transcriber.adapters.inbound.study_data).

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  SCHEMA, isStudyData, linear, ticks, denoiserName, diarizerName, comboLabel, rowById, barKind,
  points, layoutBars, status, bestJudge, judgeHeadline, sourceLabel, eventTotals,
} from "../../transcriber/adapters/inbound/web/js/lib/study.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const data = JSON.parse(readFileSync(
  new URL("../../transcriber/adapters/inbound/web/study/data.json", import.meta.url), "utf8"));
const view = readFileSync(
  new URL("../../transcriber/adapters/inbound/web/js/views/study.js", import.meta.url), "utf8");

test("the committed data file is one the page can draw", () => {
  assert.equal(data.schema, SCHEMA);
  assert.ok(isStudyData(data));
  assert.equal(isStudyData(null), false);
  assert.equal(isStudyData({ ...data, schema: "study.v0" }), false);
  assert.equal(isStudyData({ ...data, separation: { corpora: [] } }), false);
  assert.match(data.updated, /^\d{4}-\d{2}-\d{2}$/);
});

test("each corpus has its 24 combinations, best first, the best highlighted once", () => {
  for (const corpus of data.separation.corpora) {
    assert.equal(corpus.rows.length, corpus.combinations);
    const ders = corpus.rows.map((r) => r.der);
    assert.deepEqual(ders, [...ders].sort((a, b) => a - b));
    assert.equal(corpus.rows.filter((r) => barKind(r) === "best").length, 1);
    assert.equal(corpus.rows.filter((r) => barKind(r) === "builtin").length, 1);
    assert.equal(corpus.headline.best, corpus.rows[0].id);
    for (const r of corpus.rows) {
      assert.ok(Math.abs(r.success - (1 - r.der)) < 1e-3, r.id);
      assert.ok(r.gain.points >= 0, r.id);
    }
    assert.ok(data.sources[corpus.source], corpus.source);
  }
});

test("the headline matches the rows it names", () => {
  for (const corpus of data.separation.corpora) {
    const best = rowById(corpus, corpus.headline.best);
    for (const key of ["builtin", "next"]) {
      const other = rowById(corpus, corpus.headline[key].id);
      assert.ok(Math.abs(corpus.headline[key].points - (other.der - best.der)) < 1e-3);
      assert.ok(Math.abs(corpus.headline[key].error_reduction - (other.der - best.der) / other.der) < 1e-3);
    }
    assert.equal(rowById(corpus, corpus.headline.builtin.id).builtin, true);
    assert.equal(corpus.headline.next.id, corpus.rows[1].id);
  }
});

test("names of a combination: diarizer, denoiser, licence", () => {
  const row = { diarizer: "diarizen-x", diarizer_name: "DiariZen", denoiser: "none", denoiser_name: "none", non_commercial: true };
  assert.equal(denoiserName(row, "sin limpiar"), "sin limpiar");
  assert.equal(comboLabel(row, "sin limpiar"), "DiariZen · sin limpiar (NC)");
  const own = { diarizer: "builtin", diarizer_name: "Built-in", denoiser: "deepfilternet3", denoiser_name: "DeepFilterNet3", non_commercial: false };
  assert.equal(diarizerName(own, "Propio"), "Propio");
  assert.equal(diarizerName(own), "Built-in");
  assert.equal(comboLabel(own, "x", "Propio"), "Propio · DeepFilterNet3");
  assert.equal(barKind({ best: false, builtin: false }), "other");
  assert.equal(barKind({ best: true, builtin: true }), "best");
});

test("scales, ticks and the bar layout", () => {
  assert.equal(linear([0, 1], [0, 200])(0.5), 100);
  assert.equal(linear([2, 2], [0, 10])(2), 0);
  assert.deepEqual(ticks(0, 1, 0.25), [0, 0.25, 0.5, 0.75, 1]);
  assert.ok(Math.abs(points(0.0402) - 4.02) < 1e-9);
  const { rows, height } = layoutBars(3, { top: 10, labelH: 16, barH: 14, gap: 10 });
  assert.deepEqual(rows.map((r) => r.barY), [26, 66, 106]);
  assert.ok(rows.every((r) => r.labelY < r.barY));
  assert.equal(height, 130);
  assert.deepEqual(layoutBars(0).rows, []);
});

test("work not measured yet is pending, until its results are there", () => {
  assert.equal(status(undefined), "pending");
  assert.equal(status({ status: "pending" }), "pending");
  assert.equal(status({ status: "done" }), "done");
  assert.equal(status(data.separation.rule), data.separation.rule.status);
  assert.equal(status(data.labelling.tagger), data.labelling.tagger.status);
  if (status(data.labelling.tagger) === "pending") assert.deepEqual(data.labelling.tagger.approaches, []);
});

test("a small sample never wins and the judge headline compares like with like", () => {
  const judges = [
    { model: "a", accuracy: 0.3, small_sample: false },
    { model: "b", accuracy: 0.6, small_sample: false },
    { model: "c", accuracy: 0.5, small_sample: false },
    { model: "d", accuracy: 0.9, small_sample: true },
  ];
  assert.equal(bestJudge(judges).model, "b");
  const head = judgeHeadline(judges);
  assert.equal(head.best.model, "b");
  assert.equal(head.next.model, "c");
  assert.equal(head.first.model, "a");
  assert.deepEqual(judgeHeadline([{ model: "x", accuracy: 1, small_sample: true }]), { best: null, next: null, first: null });
  assert.equal(judgeHeadline([judges[1]]).first, null);
  const real = judgeHeadline(data.labelling.judges);
  assert.equal(real.best.small_sample, false);
  assert.ok(data.labelling.judges.some((j) => j.small_sample), "the Claude smoke test is flagged");
});

test("every source of the page links to a section of a document in the repository", () => {
  assert.equal(sourceLabel({ path: "docs/transcriber/LAB.md", section: "13.8.1 AMI (4 meetings)" }), "LAB.md §13.8.1");
  assert.equal(sourceLabel({ path: "docs/x/E.md", section: "6. Evaluation" }), "E.md §6");
  assert.equal(sourceLabel({ path: "a/README.md", section: "Contents (measured with x)" }), "README.md, Contents");
  for (const [id, source] of Object.entries(data.sources)) {
    assert.match(source.url, /^https:\/\/github\.com\/[^#]+\.md#[\w-]+$/, id);
  }
  const used = [...view.matchAll(/sourceLink\(data, "([\w]+)"\)/g)].map((m) => m[1]);
  for (const id of used) assert.ok(data.sources[id], id);
});

test("the event totals add up to the rows", () => {
  const ev = data.labelling.events;
  for (const mode of ["given_links", "inferred_links"]) {
    const totals = eventTotals(ev[mode].rows);
    assert.equal(totals.expected, ev[mode].expected);
    assert.equal(totals.derived, ev[mode].derived);
    assert.ok(Math.abs(ev[mode].precision - totals.matched / totals.derived) < 1e-3);
  }
});

test("every key the study view asks for exists in both languages", () => {
  const keys = new Set([...view.matchAll(/"(study\.[\w.]+)"/g)].map((m) => m[1]));
  const prefixes = [...view.matchAll(/t\(`(study\.[\w.]+)\.\$\{/g)].map((m) => m[1]);
  for (const key of keys) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  const dynamic = {
    "study.corpus": data.separation.corpora.map((c) => c.id),
    "study.corpus.title": data.separation.corpora.map((c) => c.id),
    "study.event": data.labelling.events.given_links.rows.map((r) => r.type),
    "study.step": ["denoise", "diarize", "words", "label"],
    "study.h": ["summary", "bench", "separate", "label"],
    "study.metric": ["success", "missed", "fa", "conf", "speakers", "points", "acts", "events"],
    "study.limit": ["small", "domain", "words", "synthetic"],
    "study.bench.corpus": data.separation.corpora.map((c) => c.id),
    "study.col": ["rank", "combo", "success", "der", "missed", "fa", "conf", "spk", "points", "reduction", "n",
      "model", "confirmed", "rejected", "event", "expected", "given", "inferred"],
  };
  for (const prefix of prefixes) assert.ok(dynamic[prefix], `no list of keys for ${prefix}`);
  for (const [prefix, names] of Object.entries(dynamic)) {
    for (const name of names) {
      assert.ok(es[`${prefix}.${name}`], `es ${prefix}.${name}`);
      assert.ok(en[`${prefix}.${name}`], `en ${prefix}.${name}`);
    }
  }
});
