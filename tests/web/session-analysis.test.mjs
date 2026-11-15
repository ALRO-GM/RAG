import test from "node:test";
import assert from "node:assert/strict";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import {
  ACTS, COUNT_KEYS, OPEN_ISSUES, STATUSES, STEPS, actRows, byPerson, clock, doubtfulCount, eventRows,
  isActive, orderedSteps, ratingRows, reportFileName, startMode, stepStatus,
  tagChip, tagged, verdict,
} from "../../transcriber/adapters/inbound/web/js/lib/session-analysis.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

function recorder(answer, status = 200, type = "application/json") {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    const body = typeof answer === "string" ? answer : JSON.stringify(answer);
    return new Response(body, { status, headers: { "Content-Type": type } });
  });
  return { api, calls };
}

test("the analysis calls go through the transcriber with the code", async () => {
  const { api, calls } = recorder({ ok: true });
  await api.publish("s 1");
  await api.startAnalysis("s 1", { synthetic: true });
  await api.analysisStatus("s 1");
  await api.analysisStep("s 1", "candidates");
  assert.deepEqual(calls.map((c) => [c.init.method, c.url]), [
    ["POST", "/api/sessions/s%201/publish"],
    ["POST", "/api/analysis/sessions/s%201"],
    ["GET", "/api/analysis/sessions/s%201"],
    ["GET", "/api/analysis/sessions/s%201/steps/candidates"],
  ]);
  assert.deepEqual(JSON.parse(calls[1].init.body), { restart: false, synthetic: true });
  for (const call of calls) assert.equal(call.init.headers["X-Access-Code"], "code-123");
});

test("a report comes back as the page's text, not parsed", async () => {
  const { api, calls } = recorder("<!doctype html><p>Hola</p>", 200, "text/html");
  assert.equal(await api.analysisReport("s1", "2"), "<!doctype html><p>Hola</p>");
  assert.equal(calls[0].url, "/api/analysis/sessions/s1/reports/2");
});

test("a run is active while the RAG says so or a step is running", () => {
  assert.equal(isActive(null), false);
  assert.equal(isActive({ active: true, status: "done" }), true);
  assert.equal(isActive({ active: false, status: "running" }), true);
  assert.equal(isActive({ active: false, status: "failed" }), false);
  assert.equal(startMode(null), "start");
  assert.equal(startMode({ active: true }), "running");
  assert.equal(startMode({ status: "failed" }), "resume");
  assert.equal(startMode({ status: "interrupted", active: false }), "resume");
  assert.equal(startMode({ status: "done" }), "again");
});

test("steps come in running order, filled in when missing", () => {
  const steps = orderedSteps({ steps: [{ name: "reports", status: "done" }, { name: "sentences", status: "weird" }] });
  assert.deepEqual(steps.map((s) => s.name), STEPS);
  assert.equal(stepStatus(steps[0]), "pending");
  assert.equal(stepStatus(steps[7]), "done");
  assert.equal(steps[3].status, "pending");
});

test("every step, status, count, issue and event has words in both languages", () => {
  const keys = [
    ...STEPS.map((s) => `sa.step_${s}`), ...STATUSES.map((s) => `sa.status_${s}`),
    ...COUNT_KEYS.map((k) => `sa.count_${k}`), ...OPEN_ISSUES.map((k) => `sa.issue_${k}`),
    ...["start", "running", "resume", "again"].map((m) => `sa.start_${m}`),
    ...["running", "done", "failed"].map((m) => `sa.run_${m}`),
    ...["rejected", "unparsed", "not_sent", "none"].map((v) => `sa.verdict_${v}`),
    ...["acts", "events"].map((s) => `sa.unavailable_${s}`),
    ...["candidates", "confirmation", "assessment", "reports"].map((s) => `sa.help_${s}`),
    ...["question_answered", "question_unanswered", "proposal_accepted", "proposal_rejected",
      "proposal_contested", "proposal_ignored", "decision", "action_item", "open_issue"].map((e) => `sa.event_${e}`),
    "nav.analysis", "sa.acts_help", "sa.doubtful", "sa.run_interrupted",
    ...ACTS.map((a) => `act.${a}`),
  ];
  for (const key of keys) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
});

test("clock, grouping by person and the verdict of a tagged sentence", () => {
  assert.equal(clock(65.7), "1:05");
  assert.equal(clock(null), "");
  assert.equal(clock(-1), "");
  const groups = byPerson([{ speaker: "Ana", text: "a" }, { speaker: "Luis", text: "b" }, { speaker: "Ana", text: "c" }]);
  assert.deepEqual(groups.map((g) => [g.speaker, g.items.length]), [["Ana", 2], ["Luis", 1]]);
  assert.deepEqual(byPerson(undefined), []);
  const tag = { indicator_id: "LISTEN.1", polarity: "positive", similarity: 0.6234 };
  assert.deepEqual(verdict({ tags: [tag], decision: { status: "confirmed", indicator_id: "LISTEN.2", polarity: "positive" } }),
    { kind: "confirmed", indicator: "LISTEN.2", polarity: "positive" });
  assert.equal(verdict({ tags: [tag], decision: { status: "rejected" } }).kind, "rejected");
  assert.equal(verdict({ tags: [tag], decision: { status: "unparsed" } }).kind, "unparsed");
  assert.equal(verdict({ tags: [tag] }).kind, "not_sent");
  assert.equal(verdict({ tags: [] }).kind, "none");
  assert.equal(tagChip(tag), "+LISTEN.1 0.62");
  assert.equal(tagChip({ indicator_id: "CONFL.3", polarity: "negative" }), "−CONFL.3");
  assert.equal(tagged({ sentences: [{ tags: [tag] }, { tags: [] }] }).length, 1);
});

test("ratings, acts, events and the report's file name", () => {
  const rows = ratingRows({ scorecards: [
    { participant: "Ana", assessments: [{ competency_id: "A", rating: 4 }, { competency_id: "B", rating: null }] },
    { participant: "Luis", assessments: [{ competency_id: "B", rating: 2 }] },
  ] });
  assert.deepEqual(rows.competencies, ["A", "B"]);
  assert.deepEqual(rows.people, [{ name: "Ana", ratings: [4, null] }, { name: "Luis", ratings: [null, 2] }]);
  const acts = { utterances: [
    { id: "t0.s0", speaker: "A", text: "x", by: "encoder", labels: [{ label: "otro", weight: 60, doubtful: true }, { label: "opinion", weight: 30 }] },
    { id: "t1.s0", speaker: "B", text: "y", act: "pregunta" },
  ] };
  assert.deepEqual(actRows(acts)[0], {
    id: "t0.s0", speaker: "A", text: "x", start: undefined, by: "encoder",
    labels: [{ label: "otro", weight: 60, doubtful: true }, { label: "opinion", weight: 30, doubtful: false }],
  });
  assert.deepEqual(actRows(acts)[1].labels, [{ label: "pregunta", weight: 100, doubtful: false }]);
  assert.equal(doubtfulCount(acts), 1);
  assert.equal(doubtfulCount(undefined), 0);
  assert.deepEqual(eventRows({ events: [{ order: 5 }, { order: 1 }] }).map((e) => e.order), [1, 5]);
  assert.equal(reportFileName("sim-1", { position: 2, name: "Joaquín Salas" }), "sim-1-2-Joaquin-Salas.html");
  assert.equal(reportFileName("sim-1", { position: 1, name: "../" }), "sim-1-1.html");
});
