// The AI interviewer's pure logic (#208): the topics, editing a plan, the
// live coverage from the feed and the report's figures (the comparison of
// candidates moved to the offer page, #271: offers.test.mjs).
// Nothing here depends on the browser's language: the logic has no text.
import test from "node:test";
import assert from "node:assert/strict";
import {
  INTERVIEW, InterviewState, addSlot, clonePlan, editSlot, planCounts, planErrors,
  planForSave, removeSlot, reportFigures, scoreClass, setEssential, speakerProducts,
} from "../../transcriber/adapters/inbound/web/js/lib/interview.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

function plan() {
  return {
    schema: "interview.plan.v1", id: "plan-1", revision: 2, title: "Dependiente/a", minutes: 20,
    slots: [
      { id: "s1", label: "Caja", kind: "requirement", competency: null, level: "minimo", phase: "experience", questions: ["¿Dónde?"], evidence: "", weight: 4 },
      { id: "s2", label: "Comunicación", kind: "competency", competency: "COMM", level: "deseable", phase: "behavioural", questions: ["Cuéntame"], evidence: "", weight: 3 },
    ],
  };
}

test("only the products that speak are topics", () => {
  const found = speakerProducts([{ product: "voicebot", kind: "speaker" }, { product: INTERVIEW, kind: "speaker" },
    { product: "live_assist.support", kind: "live_page" }]);
  assert.deepEqual(found.map((p) => p.product), ["voicebot", INTERVIEW]);
  assert.deepEqual(speakerProducts(undefined), []);
});

test("a plan is edited without touching the saved copy", () => {
  const saved = plan();
  const draft = clonePlan(saved);
  const id = addSlot(draft);
  assert.equal(id, "s3");
  editSlot(draft, "s3", { label: "Turnos", questions: [" ¿Sábados? ", "", "¿Noches?"], weight: 9 });
  editSlot(draft, "s1", { competency: "SOLVE" });
  editSlot(draft, "s2", { competency: "nope" });
  editSlot(draft, "zz", { label: "x" });
  assert.deepEqual(draft.slots[2].questions, ["¿Sábados?", "¿Noches?"]);
  assert.equal(draft.slots[2].weight, 5);
  assert.equal(draft.slots[0].kind, "competency");
  assert.equal(draft.slots[0].phase, "behavioural");
  assert.equal(draft.slots[1].kind, "requirement");
  assert.equal(saved.slots.length, 2);
  setEssential(draft, "s2", true);
  setEssential(draft, "s1", false);
  setEssential(draft, "s3", false);
  assert.deepEqual(draft.slots.map((s) => s.level), ["deseable", "minimo", "deseable"]);
  assert.deepEqual(planCounts(draft), { minimo: 1, deseable: 2, opcional: 0, total: 3 });
  removeSlot(draft, "s2");
  assert.deepEqual(planErrors(draft), ["iv.err.no_minimum"]);
  draft.slots[0].questions = [" "];
  draft.slots[0].label = "";
  draft.minutes = 200;
  draft.title = " ";
  assert.deepEqual(planErrors(draft), ["iv.err.title", "iv.err.minutes", "iv.err.no_minimum", "iv.err.label", "iv.err.question"]);
  assert.deepEqual(planForSave({ ...plan(), slots: [{ ...plan().slots[0], questions: ["a", " "] }] }).slots[0].questions, ["a"]);
  const full = plan();
  for (let i = 0; i < 20; i += 1) addSlot(full);
  assert.equal(full.slots.length, 12);
  assert.equal(addSlot(full), null);
  assert.deepEqual(planErrors({ ...plan(), slots: [] }), ["iv.err.no_slots", "iv.err.no_minimum"]);
});

test("the live coverage follows the feed", () => {
  const state = new InterviewState();
  assert.deepEqual(state.slots(), []);
  state.apply({ type: "speech" });
  state.apply({ type: "coverage", action: "probe", coverage: { target: "s1", minutes: 20, minutes_left: 12,
    slots: [{ id: "s1", status: "partial" }, { id: "s2", status: "empty" }, { id: "s3", status: "unknown" }] } });
  assert.equal(state.action, "probe");
  assert.deepEqual(state.slots().map((s) => s.current), [true, false, false]);
  assert.deepEqual(state.counts(), { filled: 0, partial: 1, empty: 1, skipped: 0, unknown: 1 });
  state.apply({ type: "interview", state: "closed", closed_by: "time" });
  state.apply({ type: "interview", state: "reporting" });
  assert.ok(state.closed && state.reporting && state.closedBy === "time");
  state.apply({ type: "interview_result", sheet: { slots: [] } });
  assert.ok(!state.reporting && state.result.sheet);
});

test("the report's figures and score classes", () => {
  assert.deepEqual([scoreClass(null), scoreClass(85), scoreClass(70), scoreClass(45), scoreClass(10)],
    ["none", "good", "good", "mid", "low"]);
  const figures = reportFigures({ suitability: 55.5, slots: [
    { id: "s1", label: "a", level: "minimo", weight: 2, score: 90, above: true, status: "filled" },
    { id: "s2", label: "b", level: "minimo", weight: 5, score: 0, status: "unknown" },
    { id: "s3", label: "c", level: "deseable", weight: 2, score: null, status: "empty" },
    { id: "s4", label: "d", level: "opcional", weight: 1, score: 10, status: "skipped" },
  ] });
  assert.equal(figures.suitability, 55.5);
  assert.equal(figures.meets, 1);
  assert.deepEqual(figures.gaps.map((s) => s.id), ["s2", "s3"]);
  assert.deepEqual(figures.above.map((s) => s.id), ["s1"]);
  assert.equal(figures.unknown, 1);
  assert.equal(figures.skipped, 1);
  assert.deepEqual(figures.byWeight.map((s) => s.id), ["s2", "s1", "s3", "s4"]);
  assert.equal(reportFigures(null).total, 0);
});

test("every error, level, status and competency has a text in both languages", () => {
  const keys = [
    "iv.err.title", "iv.err.minutes", "iv.err.no_slots", "iv.err.no_minimum", "iv.err.label", "iv.err.question",
    ...["minimo", "deseable", "opcional"].map((l) => `iv.level.${l}`),
    ...["filled", "partial", "empty", "skipped", "unknown"].map((s) => `iv.status.${s}`),
    ...["COMM", "LISTEN", "INFLU", "COLLAB", "INCL", "CONFL", "FEEDB", "COORD", "SOLVE", "INIT", "ADAPT"].map((c) => `iv.comp.${c}`),
    "voicebot.topic.voicebot", `voicebot.topic.${INTERVIEW}`,
  ];
  for (const key of keys) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
});
