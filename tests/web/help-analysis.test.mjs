// The analysis help page: its strings exist in both languages, its
// competencies are the rubric's own, its routes and links resolve.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  SECTIONS, anchorOf, competencies, helpHref, helpKeys, scrollTarget,
} from "../../transcriber/adapters/inbound/web/js/lib/help-analysis.js";
import { HELP_RUBRIC } from "../../transcriber/adapters/inbound/web/js/lib/help-rubric.js";
import { STEPS } from "../../transcriber/adapters/inbound/web/js/lib/session-analysis.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const WEB = new URL("../../transcriber/adapters/inbound/web/js/", import.meta.url);
const read = (path) => readFileSync(new URL(path, WEB), "utf8");
const rubric = (lang) => JSON.parse(readFileSync(
  new URL(`../../../../../knowledge/rubrics/core.${lang}.json`, WEB), "utf8"));

test("every string the help page reads exists in Spanish and English", () => {
  for (const key of helpKeys()) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
});

test("the help keys of the two dictionaries are the same", () => {
  const keys = (d) => Object.keys(d).filter((k) => k.startsWith("help.")).sort();
  assert.deepEqual(keys(es), keys(en));
});

test("the two languages are written apart, not copied", () => {
  const same = helpKeys().filter((k) => es[k] === en[k] && /[a-z]{4}/.test(es[k]) && !k.startsWith("help.act."));
  assert.deepEqual(same, []);
});

test("examples are invented: no timestamps of recordings, no file names", () => {
  for (const key of Object.keys(es).filter((k) => k.startsWith("help."))) {
    assert.doesNotMatch(es[key], /\.(wav|mp3|m4a)\b|session-\d/i, key);
  }
});

test("each competency of the rubric is on the page, in both languages", () => {
  for (const lang of ["es", "en"]) {
    const source = rubric(lang).competencies;
    const shown = competencies(lang);
    assert.deepEqual(shown.map((c) => c.id), source.map((c) => c.id));
    for (const [i, c] of source.entries()) {
      assert.equal(shown[i].name, c.name);
      assert.equal(shown[i].definition, c.definition);
      assert.equal(shown[i].limits, c.limits);
      assert.deepEqual(shown[i].helps, c.practice.helps);
      assert.deepEqual(shown[i].hurts, c.practice.hurts);
      assert.ok(shown[i].positive && shown[i].negative, c.id);
      const positives = c.indicators.filter((x) => x.polarity === "positive").flatMap((x) => x.examples);
      assert.ok(positives.includes(shown[i].positive), c.id);
      assert.ok(c.indicators.some((x) => x.polarity === "negative" && x.text === shown[i].negative), c.id);
    }
  }
});

test("an unknown language falls back to Spanish", () => {
  assert.equal(competencies("fr"), HELP_RUBRIC.es);
});

test("links: a step or a section has an anchor, anything else none", () => {
  assert.equal(helpHref(), "#/help/analysis");
  assert.equal(helpHref("acts"), "#/help/analysis/acts");
  for (const step of STEPS) assert.equal(scrollTarget(step), `step-${step}`);
  for (const section of SECTIONS) assert.equal(scrollTarget(section), anchorOf(section));
  assert.equal(scrollTarget("nope"), null);
  assert.equal(scrollTarget(undefined), null);
});

test("the router sends #/help/analysis[/x] to the help view", () => {
  const main = read("main.js");
  const line = main.split("\n").find((l) => l.includes("helpAnalysisView]"));
  assert.ok(line);
  const pattern = new RegExp(line.match(/\[\/(.*)\/, helpAnalysisView\]/)[1]);
  assert.ok(pattern.test("#/help/analysis"));
  assert.equal(pattern.exec("#/help/analysis/acts")[1], "acts");
  assert.ok(!pattern.test("#/help/other"));
});

test("the analysis page links to the help, from the top and from each step", () => {
  const view = read("views/analysis.js");
  assert.match(view, /helpHref\(\)/);
  assert.match(view, /helpHref\(step\.name\)/);
  assert.match(view, /t\("help\.hint"\)/);
});
