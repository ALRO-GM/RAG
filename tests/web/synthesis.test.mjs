import test from "node:test";
import assert from "node:assert/strict";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import {
  MAX_JOB, SYNTHESIS_CONFIDENCE, asksForClaude, canRedoSynthesis, claudeAllowed, jobForm, jobOf, startBody,
  synthesisPeople,
} from "../../transcriber/adapters/inbound/web/js/lib/session-analysis.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

test("a start sends the job and the steps only when given", async () => {
  assert.deepEqual(startBody(), { restart: false, synthetic: false });
  assert.deepEqual(startBody({ steps: ["synthesis"], job: "Jefe de turno", claudeSynthesis: 1 }), {
    restart: false, synthetic: false, steps: ["synthesis"], job: "Jefe de turno", claude_synthesis: true,
  });
  assert.equal(startBody({ job: "x".repeat(MAX_JOB + 5) }).job.length, MAX_JOB);
  assert.deepEqual(startBody({ job: "" }), { restart: false, synthetic: false, job: "", claude_synthesis: false });
  const calls = [];
  const api = new Api(() => "c", async (url, init) => {
    calls.push(init);
    return new Response("{}", { status: 202, headers: { "Content-Type": "application/json" } });
  });
  await api.startAnalysis("s1", { steps: ["synthesis"], job: "A" });
  assert.deepEqual(JSON.parse(calls[0].body), {
    restart: false, synthetic: false, steps: ["synthesis"], job: "A", claude_synthesis: false,
  });
});

test("the stored job, the leave to use Claude and when to ask for it", () => {
  assert.equal(jobOf(null), "");
  assert.equal(jobOf({ job: { text: "Puesto" } }), "Puesto");
  assert.equal(claudeAllowed({ job: { use_claude: true } }), true);
  assert.equal(claudeAllowed({}), false);
  assert.equal(asksForClaude({ mode: "local" }), true);
  assert.equal(asksForClaude({ mode: "claude_code" }), false);
  assert.equal(asksForClaude(null), false);
});

test("the synthesis can be redone once the reports exist and nothing runs", () => {
  assert.equal(canRedoSynthesis(null), false);
  assert.equal(canRedoSynthesis({ status: "done", steps: [{ name: "reports", status: "done" }] }), true);
  assert.equal(canRedoSynthesis({ status: "done", active: true, steps: [{ name: "reports", status: "done" }] }), false);
  assert.equal(canRedoSynthesis({ status: "failed", steps: [{ name: "reports", status: "failed" }] }), false);
});

test("people of a synthesis: quotes and events apart, unknown values made safe", () => {
  const people = synthesisPeople({ people: [
    {
      participant: "Lucía", status: "done", model: "claude-code:opus", confidence: "high",
      strengths: [{ text: "Cierra acuerdos.", basis: "firm", cites: [
        { id: "Q1", kind: "firm", text: "¿Para el jueves?", start: 418 },
        { id: "E2", kind: "event", text: "Lucía: tarea" },
      ] }],
      fit: [{ text: "Coordinar." }], unobserved: ["Feedback"], questions: ["¿Cómo sigues las tareas?"],
      confidence_basis: { firm: 6 }, dropped: { no_citation: 2, verdict: 1 },
    },
    { participant: "Joaquín", status: "skipped", reason: "no_evidence", confidence: "weird" },
  ] });
  assert.equal(people.length, 2);
  const [lucia, joaquin] = people;
  assert.equal(lucia.strengths[0].quotes.length, 1);
  assert.equal(lucia.strengths[0].events[0].id, "E2");
  assert.deepEqual(lucia.fit[0], { text: "Coordinar.", basis: "observation", quotes: [], events: [] });
  assert.equal(lucia.dropped, 3);
  assert.equal(lucia.confidence, "high");
  assert.equal(joaquin.status, "skipped");
  assert.equal(joaquin.confidence, "low");
  assert.deepEqual(joaquin.strengths, []);
  assert.deepEqual(synthesisPeople(null), []);
});

test("every synthesis text has words in both languages", () => {
  const keys = [
    "sa.step_synthesis", "sa.unavailable_synthesis", "sa.job_title", "sa.job_hint", "sa.job_placeholder",
    "sa.claude_synthesis", "sa.redo_synthesis", "sa.syn_note", "sa.syn_job", "sa.syn_strengths",
    "sa.syn_unobserved", "sa.syn_unobserved_none", "sa.syn_fit", "sa.syn_tasks", "sa.syn_questions",
    "sa.syn_confidence", "sa.syn_conf_reason", "sa.syn_none", "sa.syn_skipped", "sa.syn_no_evidence",
    "sa.syn_dropped", "sa.syn_model",
    ...SYNTHESIS_CONFIDENCE.map((c) => `sa.syn_conf_${c}`),
    ...["firm", "provisional", "observation"].map((b) => `sa.syn_basis_${b}`),
  ];
  for (const key of keys) {
    assert.ok(es[key], `es: ${key}`);
    assert.ok(en[key], `en: ${key}`);
  }
  assert.match(es["sa.syn_note"], /no una decisión/);
});

test("a refresh keeps what the person typed or ticked since the last start (#115)", () => {
  const run = { job: { text: "Jefe de turno", use_claude: true } };
  assert.deepEqual(jobForm(run, false), { job: "Jefe de turno", useClaude: true });
  assert.equal(jobForm(run, true), null);
  assert.equal(jobForm(null, false), null);
  assert.deepEqual(jobForm({}, false), { job: "", useClaude: false });
});
