// Coverage of a live interview (#286): each topic's level from the server
// (a ring that fills and a green that deepens, never colour alone), each
// concept asked and treated apart, «Ya lo he preguntado», the next concept to
// ask, the end summary and the report's block. Views are drawn in a stand-in
// DOM with the language pinned (CI runs Node 22, any locale). Invented data.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { LiveState } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import {
  FILL, counted, coverageCounts, coveredCount, levelChanges, levelOf, lowerFirst, pointCovered, pointState,
} from "../../transcriber/adapters/inbound/web/js/lib/coverage.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const DEMO = JSON.parse(readFileSync(new URL("../../examples/demo/entrevista-lucia.demo.json", import.meta.url), "utf8"));

const TOPICS = [
  { id: "s1", name: "Bases de datos", en: { name: "Databases" }, done: false, current: true,
    coverage: { level: "treated", concepts: 3, asked: 2, treated: 1, covered: 2 },
    points: [
      { id: "p1", text: "Índices", en: { text: "Indexes" }, status: "treated", asked: { by: "rules", at: 12.5 } },
      { id: "p2", text: "Caché", status: "pending", asked: { by: "hand", at: 30 } },
      { id: "p3", text: "Trazas", status: "pending", asked: null },
    ] },
  { id: "s2", name: "Inglés", done: false, coverage: { level: "none", concepts: 0, asked: 0, treated: 0, covered: 0 } },
  { id: "s3", name: "Pruebas", done: true, coverage: { level: "well_treated", concepts: 2, asked: 2, treated: 2, covered: 2 } },
];
const NEXT = { topic: "s1", topic_name: "Bases de datos", point: "p3", text: "Trazas",
  question: "Sobre bases de datos: ¿qué me puedes contar de trazas?", again: false };

test("the live state keeps each topic's level, the asked marks and the next concept", () => {
  const state = new LiveState();
  state.apply({ type: "topics", next: NEXT, topics: [
    ...TOPICS,
    { id: "s4", name: "Raro", coverage: { level: "excellent", asked: -1 },
      points: [{ id: "p1", text: "X", status: "pending", asked: { by: "magic", at: 1 } }] },
  ] });
  const chips = state.topicChips();
  assert.deepEqual(chips.map(levelOf), ["treated", "none", "well_treated", null]);
  assert.deepEqual(chips[0].points.map((p) => p.asked?.by ?? null), ["rules", "hand", null]);
  assert.equal(chips[3].points[0].asked, undefined, "an unknown author is no mark");
  assert.deepEqual(state.next, NEXT);
  // A concept asked is not pending any more, whatever the answer was.
  assert.deepEqual(state.topicFocus().pending.map((p) => p.id), ["p3"]);
  state.apply({ type: "topics", topics: TOPICS, next: { topic: "s1" } });
  assert.equal(state.next, null, "a proposal without a question is dropped");
});

test("the level rule's words: what counts as covered, asked apart from treated", () => {
  assert.deepEqual(["none", "named", "treated", "well_treated"].map(counted), [false, false, true, true]);
  assert.equal(coveredCount(TOPICS), 2);
  assert.equal(coveredCount([{ id: "x", done: true }, { id: "y", done: false }]), 1, "no level: the old done");
  assert.deepEqual(Object.values(FILL), [0, 1 / 3, 2 / 3, 1], "each level its own fill");
  assert.deepEqual(TOPICS[0].points.map(pointState), ["asked_answered", "asked", "pending"]);
  assert.equal(pointState({ status: "treated" }), "treated");
  assert.equal(pointState({ status: "unknown", asked: { by: "rules" } }), "asked_unknown");
  assert.deepEqual(TOPICS[0].points.map(pointCovered), [true, true, false]);
  assert.equal(lowerFirst("Índices"), "índices");
  assert.equal(lowerFirst("API REST"), "API REST");
});

test("a level change is announced once, never on the first reading", () => {
  const seen = new Map();
  assert.deepEqual(levelChanges(seen, TOPICS), []);
  const later = [{ ...TOPICS[0], coverage: { ...TOPICS[0].coverage, level: "well_treated" } }, TOPICS[1]];
  assert.deepEqual(levelChanges(seen, later).map((c) => [c.chip.id, c.level]), [["s1", "well_treated"]]);
  assert.deepEqual(levelChanges(seen, later), [], "said once");
});

test("the hand-off's coverage is counted per level and asked", () => {
  const coverage = [
    { id: "s1", level: "treated", points: [{ id: "p1", asked: { by: "hand" } }, { id: "p2", asked: null }] },
    { id: "s2", level: "none", points: [] }, { id: "s3", level: "bogus" },
  ];
  assert.deepEqual(coverageCounts(coverage), { none: 1, named: 0, treated: 1, well_treated: 0, topics: 2, asked: 1, hand: 1, concepts: 2 });
  assert.equal(coverageCounts(undefined).topics, 0);
});

test("every new text exists in Spanish and English, and the count says which level counts", () => {
  const keys = Object.keys(es).filter((k) => k.startsWith("cov."));
  assert.ok(keys.length >= 30);
  for (const key of [...keys, "of.asked", "of.asked_no_answer", "of.never_asked"]) {
    assert.ok(es[key] && en[key], key);
  }
  assert.ok(es["live.topics_count"].includes("tratados o bien tratados"));
  assert.ok(en["live.topics_count"].includes("treated or well treated"));
  // No verdict wording: the level is coverage, never the answers' quality.
  assert.ok(es["cov.level_note"].includes("no si las respuestas son buenas"));
  for (const key of keys) assert.ok(!/candidato bueno|buen candidato|good candidate/i.test(es[key] + en[key]), key);
});

test("the scripted interview names which line asks each concept", () => {
  const said = new Set(DEMO.transcript.utterances.map((u) => u.id));
  const asked = DEMO.topics.flatMap((t) => t.points.filter((p) => p.asked_by));
  assert.ok(asked.length >= 5);
  for (const p of asked) assert.ok(said.has(p.asked_by), p.id);
  const integration = DEMO.topics.find((t) => t.id === "testing").points.find((p) => p.id === "integration");
  assert.ok(integration.asked_by && !integration.covered_by, "one concept asked and never answered");
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
    return await fn(await import("../../transcriber/adapters/inbound/web/js/views/coverage.js"));
  } finally {
    Object.assign(globalThis, saved);
  }
}

test("a topic chip shows its level by fill and in words, for hover, focus and screen readers", async () => {
  await withDom("es", ({ topicChip, levelRing }) => {
    const chip = topicChip(TOPICS[0], "Bases de datos");
    assert.equal(chip.attributes["data-level"], "treated");
    assert.equal(chip.attributes.tabindex, "0", "it can take focus to show its words");
    assert.equal(chip.title, "Bases de datos: tratado · 1 de 3 conceptos tratados, 2 preguntados");
    const spoken = chip.all().find((n) => n.className === "sr-only");
    assert.equal(spoken.text(), chip.title);
    const ring = chip.all().find((n) => n.tagName === "SVG");
    assert.equal(ring.attributes["aria-hidden"], "true", "the ring is decoration; the words are read");
    const parts = (level) => levelRing(level).children.map((c) => c.attributes.class);
    assert.deepEqual(parts("none"), ["cov-ring-track"]);
    assert.deepEqual(parts("named"), ["cov-ring-track", "cov-ring-fill"]);
    assert.deepEqual(parts("well_treated"), ["cov-ring-track", "cov-ring-full"]);
    const third = levelRing("named").children[1].attributes["stroke-dasharray"];
    const two = levelRing("treated").children[1].attributes["stroke-dasharray"];
    assert.notEqual(third, two, "named and treated fill differently");
    assert.ok(topicChip(TOPICS[1], "Inglés").title === "Inglés: sin mencionar");
  });
  await withDom("en", ({ topicChip }) => {
    assert.equal(topicChip(TOPICS[2], "Testing").title, "Testing: well treated · 2 of 2 concepts treated, 2 asked");
  });
});

test("a concept says asked and treated apart, and can be marked by hand while live", async () => {
  await withDom("es", ({ pointRow }) => {
    const marked = [];
    const onMark = (point, asked) => marked.push([point.id, asked]);
    const rows = TOPICS[0].points.map((p) => pointRow(p, { running: true, onMark }));
    assert.deepEqual(rows.map((r) => r.className), ["iv-point cov-point is-asked_answered", "iv-point cov-point is-asked",
      "iv-point cov-point is-pending"]);
    assert.ok(rows[0].text().includes("preguntado y respondido"));
    assert.ok(rows[1].text().includes("preguntado, sin respuesta aún") && rows[1].text().includes("marcado a mano"));
    assert.ok(rows[2].text().includes("aún no preguntado"));
    const buttons = rows.map((r) => r.all().find((n) => n.tagName === "BUTTON") ?? null);
    assert.equal(buttons[0], null, "a mark by the rules is not undone by hand");
    assert.equal(buttons[1].attributes["aria-pressed"], "true");
    assert.equal(buttons[1].text(), "Deshacer «preguntado»");
    assert.equal(buttons[2].text(), "Ya lo he preguntado");
    buttons[2].listeners.click[0]();
    buttons[1].listeners.click[0]();
    assert.deepEqual(marked, [["p3", true], ["p2", false]]);
    const ended = pointRow(TOPICS[0].points[2], { running: false, onMark, ended: true });
    assert.ok(!ended.all().some((n) => n.tagName === "BUTTON"), "no marks once the session ended");
    assert.ok(ended.text().includes("no preguntado") && !ended.text().includes("aún"), "no «aún» after the end");
    assert.ok(pointRow(TOPICS[0].points[1], { ended: true }).text().includes("preguntado, sin respuesta"));
  });
  await withDom("en", ({ pointRow }) => {
    const row = pointRow(TOPICS[0].points[0], { lang: "en" });
    assert.ok(row.text().includes("Indexes asked and answered"));
    assert.ok(row.all().some((n) => n.tagName === "SVG"), "the check mark is the icon");
  });
});

test("the next concept to ask is named with its question, in the page's language", async () => {
  await withDom("es", ({ nextLine }) => {
    const box = nextLine(NEXT, TOPICS, "es");
    assert.equal(box.attributes.role, "status");
    assert.equal(box.text(), "Siguiente: Trazas (Bases de datos)Sobre bases de datos: ¿qué me puedes contar de trazas?");
    const again = nextLine({ ...NEXT, again: true }, TOPICS, "es").text();
    assert.ok(again.startsWith("Otra vez, si hay tiempo: Trazas") && again.includes("Volviendo a trazas"));
    const topicOnly = nextLine({ topic: "s2", topic_name: "Inglés", point: null, text: null,
      question: "¿Qué nivel de inglés tienes?", again: false }, TOPICS, "es");
    assert.ok(topicOnly.text().endsWith("¿Qué nivel de inglés tienes?"));
    assert.equal(nextLine(null, TOPICS, "es"), null);
  });
  await withDom("en", ({ nextLine }) => {
    const text = nextLine({ ...NEXT, point: "p1", text: "Índices" }, TOPICS, "en").text();
    assert.equal(text, "Next: Indexes (Databases)On databases: what can you tell me about indexes?");
  });
});

test("the report tells «preguntado, sin respuesta» from «no preguntado»", async () => {
  const sheet = { kind: "live", coverage: [
    { id: "s1", label: "Bases de datos", level: "named", concepts: 2, asked: 1, treated: 0, covered: 1, words: 0, points: [
      { id: "p1", text: "Índices", status: "pending", asked: { by: "after_session", at: 3, utterance: "q1", quote: "¿Y los índices?" } },
      { id: "p2", text: "Caché", status: "pending", asked: null }] },
    { id: "s2", label: "Inglés", level: "none", concepts: 0, asked: 0, treated: 0, covered: 0, words: 0, points: [] },
  ] };
  await withDom("es", ({ coverageBlock, summaryLine }) => {
    const block = coverageBlock(sheet);
    const text = block.text();
    assert.ok(text.includes("Índices preguntado, sin respuesta por el modelo, al terminar"));
    assert.ok(text.includes("¿Y los índices?"), "the interviewer's words");
    assert.ok(text.includes("Caché no preguntado"));
    assert.ok(text.includes("no de la persona candidata"), "not asked is the interview's gap");
    assert.equal(block.attributes["aria-label"], "Cobertura de la entrevista");
    assert.equal(coverageBlock({ kind: "live" }), null, "older reports: nothing");
    assert.equal(summaryLine(sheet.coverage).text(),
      "Temas: 0 bien tratados, 0 tratados, 1 nombrados, 1 sin mencionar · Conceptos preguntados: 1 de 2");
    assert.equal(summaryLine([]), null);
  });
});
