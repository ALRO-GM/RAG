// The concepts of each point of an interview plan (#258): edited one per
// line in the plan review, followed treated / not yet / does not know it in
// the AI interviewer's coverage and report, and on the live page's topics
// with the hint of what is still to cover. The views are drawn in a
// stand-in DOM with the language pinned (CI runs Node 22, any locale).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  InterviewState, MAX_POINTS, POINT_CHARS, addSlot, editSlot, planForSave, pointCounts, pointsFromLines,
} from "../../transcriber/adapters/inbound/web/js/lib/interview.js";
import { LiveState } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const DEMO = JSON.parse(readFileSync(new URL("../../examples/demo/entrevista-lucia.demo.json", import.meta.url), "utf8"));

const RUNNING = [
  { id: "p1", text: "Tipos de pisada", status: "treated", quote: "miro la pisada", utterance: "t1.s0" },
  { id: "p2", text: "Amortiguación y drop", status: "unknown", quote: "el drop no lo conozco", utterance: "t3.s0" },
  { id: "p3", text: "Cómo elegir la talla", status: "pending", quote: "", utterance: null },
];

test("the review's lines become concepts that keep their ids", () => {
  const before = [{ id: "p1", text: "Tipos de pisada" }, { id: "p2", text: "Drop" }];
  const found = pointsFromLines(before, ["Drop", "  Talla  ", "", "talla", "Tipos de pisada", "x".repeat(200), "a", "b", "c"]);
  assert.deepEqual(found.map((p) => p.id), ["p2", "p3", "p1", "p4", "p5", "p6"]);
  assert.deepEqual(found.slice(0, 3).map((p) => p.text), ["Drop", "Talla", "Tipos de pisada"]);
  assert.equal(found[3].text.length, POINT_CHARS);
  assert.equal(found.length, MAX_POINTS);
  assert.deepEqual(pointsFromLines(undefined, undefined), []);
  const plan = { slots: [{ id: "s1", label: "Running", questions: ["¿Qué sabes?"] }] };
  const id = addSlot(plan);
  assert.deepEqual(plan.slots.find((s) => s.id === id).points, []);
  editSlot(plan, "s1", { points: ["Pisada", "Drop"] });
  assert.deepEqual(plan.slots[0].points, [{ id: "p1", text: "Pisada" }, { id: "p2", text: "Drop" }]);
  assert.deepEqual(planForSave({ slots: [{ id: "s9", questions: ["a"] }] }).slots[0].points, [], "an old plan saves no concepts");
  assert.deepEqual(pointCounts(RUNNING), { treated: 1, pending: 1, unknown: 1, total: 3 });
  assert.deepEqual(pointCounts(undefined), { treated: 0, pending: 0, unknown: 0, total: 0 });
});

test("the live topics keep their concepts and say what is still to cover", () => {
  const state = new LiveState();
  assert.equal(state.topicFocus(), null);
  state.apply({ type: "topics", topics: [
    { id: "s1", name: "Running", done: false, current: true, points: [...RUNNING, { id: "p4", text: "Raro", status: "great" }, { text: "sin id" }] },
    { id: "s2", name: "Caja", done: false },
  ] });
  const chips = state.topicChips();
  assert.deepEqual(chips[0].points.map((p) => p.status), ["treated", "unknown", "pending", "pending"]);
  assert.equal(chips[0].current, true);
  assert.ok(!("points" in chips[1]) && !("current" in chips[1]));
  const focus = state.topicFocus();
  assert.equal(focus.id, "s1");
  assert.deepEqual(focus.pending.map((p) => p.text), ["Cómo elegir la talla", "Raro"]);
  state.apply({ type: "topics", topics: [{ id: "s2", name: "Caja", done: false, current: true }] });
  assert.equal(state.topicFocus(), null, "a topic without concepts gives no hint");
});

test("the scripted interview's topics carry concepts in both languages, lines untouched", () => {
  for (const topic of DEMO.topics) {
    assert.ok(topic.asked_by && topic.points.length >= 2, topic.id);
    for (const point of topic.points) assert.ok(point.text && point.en?.text, `${topic.id}.${point.id}`);
  }
  const said = new Set(DEMO.transcript.utterances.map((u) => u.id));
  for (const topic of DEMO.topics) for (const point of topic.points) if (point.covered_by) assert.ok(said.has(point.covered_by));
  const pending = DEMO.topics.flatMap((t) => t.points.filter((p) => !p.covered_by).map((p) => p.text));
  assert.deepEqual(pending, ["Pruebas de integración"]);
});

test("every new text exists in Spanish and English", () => {
  const keys = ["iv.points", "iv.points_hint", "iv.points_line", "iv.points_note", "iv.point.treated", "iv.point.pending",
    "iv.point.unknown", "live.points_missing", "live.points_done", "live.points_title"];
  for (const key of keys) assert.ok(es[key] && en[key] && es[key] !== en[key], key);
});

// ---- the views in a stand-in DOM -------------------------------------------

class FakeNode {}
class FakeText extends FakeNode {
  constructor(text) { super(); this.textContent = text; }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.open = false;
    this.hidden = false;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.children.map((c) => (c.text ? c.text() : c.textContent)).join(""); }
}

async function withDom(language, fn) {
  const saved = { document: globalThis.document, Node: globalThis.Node };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  try {
    const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
    setLanguage(language);
    return await fn(await import("../../transcriber/adapters/inbound/web/js/views/interview.js"));
  } finally {
    Object.assign(globalThis, saved);
  }
}

function coverageState() {
  const state = new InterviewState();
  state.apply({ type: "coverage", action: "probe", coverage: {
    plan_id: "p", title: "Tienda", minutes: 20, minutes_left: 15, target: "s1", turns: 2,
    slots: [
      { id: "s1", label: "Material de running", level: "minimo", status: "partial", asked: 1, probes: 1,
        answers: [{ utterance: "u1", quote: "miro la pisada", status: "partial" }], points: RUNNING },
      { id: "s2", label: "Caja", level: "deseable", status: "empty", asked: 0, probes: 0, answers: [], points: [] },
    ],
  } });
  return state;
}

test("the coverage panel marks each concept with a sign and a word", async () => {
  await withDom("es", ({ coveragePanel, pointList }) => {
    const panel = coveragePanel(coverageState());
    const text = panel.text();
    assert.ok(text.includes("Conceptos: 1 de 3 tratados"));
    for (const word of [es["iv.point.treated"], es["iv.point.pending"], es["iv.point.unknown"]]) assert.ok(text.includes(word), word);
    assert.ok(text.includes(es["iv.points_note"]), "treated is said to be no judgement");
    const items = panel.all().filter((n) => n.tagName === "LI" && n.className.startsWith("iv-point "));
    assert.deepEqual(items.map((n) => n.className), ["iv-point is-treated", "iv-point is-unknown", "iv-point is-pending"]);
    const marks = panel.all().filter((n) => n.className === "iv-point-mark");
    assert.ok(marks.every((n) => n.attributes["aria-hidden"] === "true"), "the sign is decoration; the word is read");
    assert.equal(pointList([]), null);
    assert.ok(!pointList(RUNNING).text().includes("miro la pisada"), "the live panel shows no quotes per concept");
    assert.ok(pointList(RUNNING, { quotes: true }).text().includes("el drop no lo conozco"));
  });
  await withDom("en", ({ coveragePanel }) => {
    const text = coveragePanel(coverageState()).text();
    assert.ok(text.includes("Concepts: 1 of 3 covered") && text.includes(en["iv.point.unknown"]));
  });
});

test("the report lists each point's concepts with the words that settled them", async () => {
  await withDom("es", ({ reportCard }) => {
    const sheet = {
      schema: "interview.sheet.v1", session_id: "vb-1", title: "Tienda", suitability: 55, summary: "", turns: 4,
      elapsed_s: 300, filler: "haiku", scorer: "haiku", other_skills: [],
      slots: [{ id: "s1", label: "Material de running", level: "minimo", weight: 4, status: "partial", score: 50,
        answers: [{ utterance: "t1.s0", quote: "miro la pisada", status: "partial" }], points: RUNNING }],
    };
    const card = reportCard({ sheet, handoff: {} });
    assert.ok(card.text().includes("Conceptos: 1 de 3 tratados"), "the summary says how many were treated");
    const more = card.all().find((n) => n.tagName === "DETAILS");
    more.open = true;
    for (const fn of more.listeners.toggle ?? []) fn();
    const inside = card.text();
    assert.ok(inside.includes("el drop no lo conozco") && inside.includes(es["iv.point.unknown"]));
  });
});
