import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  PANEL_KINDS, linear, ticks, fmt, panelsOf, figuresOf, mixedStatistics, groupBy,
  verifiedClass, singleFromAggregate, spearmanBrown, situationsFor, layoutRows,
  noteLine, figureDetail, entryDetail, fold, filterSources, familiesOf, entriesOf, clip,
} from "../../transcriber/adapters/inbound/web/js/research/charts.js";

const data = JSON.parse(readFileSync(
  new URL("../../transcriber/adapters/inbound/web/research/data.json", import.meta.url), "utf8"));

test("linear maps a domain onto a range", () => {
  const x = linear([0, 1], [10, 110]);
  assert.equal(x(0), 10);
  assert.equal(x(0.5), 60);
  assert.equal(x(1), 110);
  const y = linear([0, 100], [200, 0]);
  assert.equal(y(25), 150);
  assert.equal(linear([3, 3], [0, 10])(3), 0);
});

test("ticks are evenly spaced and exact", () => {
  assert.deepEqual(ticks(0, 1, 0.25), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(ticks(0, 0.5, 0.1), [0, 0.1, 0.2, 0.3, 0.4, 0.5]);
});

test("fmt writes coefficients the APA way", () => {
  assert.equal(fmt(0.8), ".80");
  assert.equal(fmt(0.286), ".286");
  assert.equal(fmt(0.54), ".54");
  assert.equal(fmt(-0.26), "−.26");
  assert.equal(fmt(0), "0");
  assert.equal(fmt(1), "1");
  assert.equal(fmt(Number.NaN), "");
});

test("every figure of the data sits in a panel of its own statistic kind", () => {
  assert.deepEqual(mixedStatistics(data), []);
  for (const p of data.panels) assert.ok(PANEL_KINDS[p.kind], p.kind);
  const bad = { panels: [{ id: "k", kind: "kappa" }], figures: [{ panel: "k", statistic: "r" }] };
  assert.equal(mixedStatistics(bad).length, 1);
});

test("panels and figures are grouped per chart in file order", () => {
  assert.deepEqual(panelsOf(data, "reliability").map((p) => p.id), ["kappa", "icc", "g", "channel", "machine", "auroc"]);
  assert.deepEqual(panelsOf(data, "situations").map((p) => p.id), ["exercises", "roleplays"]);
  assert.equal(figuresOf(data, "channel").length, 15);
  assert.equal(figuresOf(data, "validity").length, 19);
  const cited = new Set(data.sources.map((s) => s.id));
  for (const f of data.figures) assert.ok(cited.has(f.source), f.id);
});

test("groupBy keeps the first-seen order", () => {
  const groups = groupBy([{ t: "b" }, { t: "a" }, { t: "b" }], (x) => x.t);
  assert.deepEqual(groups.map((g) => [g.name, g.members.length]), [["b", 2], ["a", 1]]);
});

test("verification level picks the mark style", () => {
  assert.equal(verifiedClass("full-text"), "solid");
  assert.equal(verifiedClass("abstract-only"), "weak abstract");
  assert.equal(verifiedClass("secondary"), "weak secondary");
});

test("Spearman-Brown projection matches the published data", () => {
  const single = singleFromAggregate(0.65, 3);
  assert.ok(Math.abs(single - 0.3824) < 1e-3);
  assert.ok(Math.abs(spearmanBrown(single, 3) - 0.65) < 1e-9);
  assert.equal(situationsFor(single, 0.7), 4);
  assert.equal(situationsFor(single, 0.8), 7);
  assert.equal(situationsFor(0.01, 0.99, 5), null);
  assert.equal(data.projection.derived, true);
  assert.ok(Math.abs(data.projection.single - single) < 1e-3);
  for (const p of data.projection.points) assert.ok(Math.abs(p.value - spearmanBrown(single, p.situations)) < 1e-3);
});

test("layoutRows stacks groups and bars, and drops the note when asked", () => {
  const f = { value: 0.5 };
  const { items, height } = layoutRows([
    { group: "A", figure: f }, { group: "A", figure: f, note: false }, { group: "B", figure: f },
  ]);
  assert.deepEqual(items.map((i) => i.kind), ["group", "bar", "bar", "group", "bar"]);
  assert.ok(items[1].noteY > items[1].barY);
  assert.equal(items[2].noteY, null);
  assert.ok(items[2].barY - items[1].barY === 18 + 14 + 16 + 10);
  assert.ok(height > items[4].noteY);
});

test("detail and note texts carry statistic, sample and source", () => {
  const f = data.figures.find((x) => x.id === "k_negotiact");
  assert.match(noteLine(f), /κ de Cohen · 2 codificadores · 18 negociaciones.*Jäckel et al\. \(2024\)/);
  const d = figureDetail(f, data.labels.verified);
  assert.match(d, /: \.80 \(κ de Cohen\)/);
  assert.match(d, /Fuente: Jäckel et al\. \(2024\), PDF p\. 23-24/);
  assert.match(d, /Leído: texto completo/);
  const e = data.entries.find((x) => x.id === "assessment_centre");
  assert.match(entryDetail(e, data.labels), /A 69, B 74, C 93\. Confianza alta\. Veredicto: usar\./);
});

test("the reference filter folds accents and combines conditions", () => {
  assert.equal(fold("Jäckel ÁÉ"), "jackel ae");
  const all = filterSources(data.sources);
  assert.equal(all.length, data.sources.length);
  assert.ok(filterSources(data.sources, { query: "jackel negotiact" }).includes("jaeckel2024_negotiact"));
  const secondary = filterSources(data.sources, { verified: "secondary" });
  assert.ok(secondary.length > 0);
  for (const id of secondary) assert.equal(data.sources.find((s) => s.id === id).verified, "secondary");
  const fam = filterSources(data.sources, { family: "automatic-method" });
  assert.ok(fam.includes("pugh2021"));
  assert.deepEqual(filterSources(data.sources, { query: "zzzz-nothing" }), []);
});

test("families and entries for the map", () => {
  assert.deepEqual(familiesOf(data.sources, data.labels.family), Object.keys(data.labels.family));
  const sel = entriesOf(data.entries, "selection-method");
  assert.equal(sel.length, 9);
  assert.equal(sel[0].C, 100);
  assert.equal(data.entries.length, 70);
});

test("clip shortens on a word boundary", () => {
  assert.equal(clip("short", 10), "short");
  assert.equal(clip("Assessment centre interpersonal exercises", 20), "Assessment centre…");
  assert.equal(clip("abcdefghijklmnop", 6), "abcde…");
});
