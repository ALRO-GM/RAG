// The live demo's prepared result (#232) read by the summary-first
// result page (#230): the stored run counts as having results, only the
// candidate is a person, and her summary carries the ring, the
// strengths, the improvements and the bars with the required level.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { doneResultSteps, hasResults, peopleOf, sessionFacts } from "../../transcriber/adapters/inbound/web/js/lib/analysis-results.js";
import { personSummary } from "../../transcriber/adapters/inbound/web/js/lib/summary-first.js";

const BUNDLE = JSON.parse(readFileSync(new URL("../../examples/demo/entrevista-lucia.analysis.json", import.meta.url), "utf8"));
const STEPS = ["sentences", "interaction", "candidates", "acts", "events", "confirmation", "assessment", "reports", "synthesis"];
// The run the RAG stores for it (DemoAnalysis.run): the steps given are done.
const given = new Set([...Object.keys(BUNDLE.files), "synthesis"]);
const RUN = {
  session_id: BUNDLE.session_id, status: "done", synthetic: true, started_at: BUNDLE.started_at, finished_at: BUNDLE.started_at,
  steps: STEPS.map((name) => ({ name, status: given.has(name) ? "done" : "unavailable" })),
};
const DATA = { ...BUNDLE.files, synthesis: BUNDLE.synthesis };

test("the demo run has results the page can show", () => {
  assert.equal(hasResults(RUN), true);
  assert.deepEqual(doneResultSteps(RUN), ["sentences", "assessment", "synthesis"]);
  const facts = sessionFacts(BUNDLE.session_id, RUN, DATA);
  assert.equal(facts.synthetic, true);
  assert.equal(facts.people, 2);
  assert.equal(facts.type, "entrevista");
});

test("only the candidate is a person, with the ring, points and bars", () => {
  const people = peopleOf(DATA);
  assert.deepEqual(people.map((p) => p.name), ["Lucía Marín"]);
  const summary = personSummary(people[0]);
  assert.equal(summary.overall.percent, 88.7);
  assert.equal(summary.overall.verdict, "strong");
  assert.deepEqual(summary.strengths.map((s) => s.name), ["Resuelve con datos", "Comunicación clara", "Asume responsabilidad"]);
  assert.deepEqual(summary.improvements.map((s) => s.name), ["Pruebas automáticas", "Delegar"]);
  const testing = summary.bars.find((b) => b.id === "pruebas_automaticas");
  assert.ok(testing.percent < testing.required);
  assert.equal(summary.shownBars.length, 5);
});
