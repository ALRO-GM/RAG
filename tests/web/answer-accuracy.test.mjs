// How accurate the answers were (#259): the live state keeps a round's draft
// judgement, and the block drawn on the AI interviewer's report and at the
// end of a live round, in a stand-in DOM with the language pinned (CI runs
// Node 22 in any locale). Plus the candidate notice @2 and its wording.
import test from "node:test";
import assert from "node:assert/strict";
import { LiveState } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const RESULT = {
  state: "ready",
  slots: [
    { id: "s1", label: "Material de running", level: "minimo", status: "filled",
      judgement: { label: "wrong", reason: "El drop no es el peso de la zapatilla.", quote: "el drop es lo que pesa", utterance: "t3.s0", basis: "documents", cites: ["tienda:c1"] },
      accuracy: { wrong: 1, not_treated: 1 },
      points: [
        { id: "p1", text: "Amortiguación y drop", status: "treated", quote: "el drop es lo que pesa", utterance: "t3.s0",
          judgement: { label: "wrong", reason: "Confunde drop con peso.", quote: "el drop es lo que pesa", utterance: "t3.s0", basis: "general", cites: [] } },
        { id: "p2", text: "Tipos de pisada", status: "pending" },
        { id: "p3", text: "Cómo elegir la talla", status: "unknown" },
      ] },
    { id: "s2", label: "Experiencia en tienda", level: "minimo", status: "filled",
      judgement: { label: "to_verify", reason: "Es experiencia propia.", quote: "trabajé tres años", utterance: "t1.s0", basis: "general", cites: [] },
      accuracy: { to_verify: 1 }, points: [] },
  ],
  to_check: [{ slot: "s1", quote: "el drop es lo que pesa", utterance: "t3.s0", why: "wrong", note: "Preguntar qué entiende por drop." }],
};

test("the live state keeps a round's judgement from the feed or the hand-off", () => {
  const state = new LiveState();
  state.apply({ type: "accuracy", state: "reporting" });
  assert.equal(state.accuracy.state, "reporting");
  state.apply({ type: "accuracy", ...RESULT });
  assert.equal(state.accuracy.slots.length, 2);
  state.apply({ type: "accuracy", state: "nonsense" });
  assert.equal(state.accuracy.state, "ready", "an unknown state is ignored");
  const late = new LiveState();
  late.apply({ type: "handoff", handoff: { session_id: "r1", accuracy: RESULT } });
  assert.equal(late.accuracy.slots[0].id, "s1");
});

test("the notice @2 says an analysis may be drafted and reviewed; no wording judges the person", () => {
  for (const dict of [es, en]) {
    const notice = dict["live.banner.interview_ai_notice@2"];
    assert.ok(notice, "notice @2 exists");
    assert.ok(!/no puntúa|scores or rates no one/i.test(notice));
  }
  assert.match(es["live.banner.interview_ai_notice@2"], /borrador de análisis de las respuestas/);
  assert.match(es["live.banner.interview_ai_notice@2"], /revisa una persona/);
  assert.match(en["live.banner.interview_ai_notice@2"], /a person reviews it|which a person reviews/);
  const keys = Object.keys(es).filter((k) => k.startsWith("iv.judge") || k.startsWith("iv.check.") || k.startsWith("iv.to_check")
    || k.startsWith("iv.basis.") || k.startsWith("live.accuracy"));
  assert.ok(keys.length >= 18);
  for (const key of keys) {
    assert.ok(en[key], `${key} in English`);
    for (const text of [es[key], en[key]]) {
      assert.ok(!/miente|mentir|invent|\bapto\b|no apto|contratar|lying|liar|made it up|suitable|\bhire/i.test(text), `${key}: ${text}`);
    }
  }
});

// ---- the block in a stand-in DOM ----------------------------------------

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

test("the judgement block: per point, concepts, reasons, basis, what to check, the draft note", async () => {
  await withDom("es", ({ accuracyBlock, reportCard }) => {
    const block = accuracyBlock(RESULT);
    const text = block.text();
    assert.ok(text.includes(es["iv.judge_title"]) && text.includes(es["iv.judge_note"]));
    for (const key of ["wrong", "to_verify", "not_treated", "does_not_know"]) assert.ok(text.includes(es[`iv.judge.${key}`]), key);
    assert.ok(text.includes("El drop no es el peso de la zapatilla.") && text.includes("Confunde drop con peso."));
    assert.ok(text.includes("base: documentos del dominio tienda:c1") && text.includes(es["iv.basis.general"]));
    assert.ok(text.includes(es["iv.to_check_title"]) && text.includes(es["iv.to_check_hint"]) && text.includes(es["iv.check.wrong"]));
    assert.ok(text.includes("contiene un error: 1"), "the counts per point");
    const note = block.all().find((n) => n.className === "draft-notice");
    assert.equal(note.attributes.role, "note");
    assert.equal(accuracyBlock({ slots: [], to_check: [] }), null);
    assert.equal(accuracyBlock(null), null);
    const sheet = { schema: "interview.sheet.v1", session_id: "vb-1", title: "Tienda", suitability: 30, summary: "", turns: 3,
      elapsed_s: 200, filler: "haiku", scorer: "haiku", other_skills: [], ...RESULT,
      slots: RESULT.slots.map((s) => ({ ...s, weight: 3, score: 30, answers: [] })) };
    const card = reportCard({ sheet, handoff: {} }).text();
    assert.ok(card.includes(es["iv.judge_note"]), "the note is visible on the report, not behind «Más información»");
    assert.ok(card.includes(es["iv.to_check_title"]));
  });
  await withDom("en", ({ accuracyBlock }) => {
    const text = accuracyBlock(RESULT).text();
    assert.ok(text.includes("Statements worth checking") && text.includes(en["iv.judge.wrong"]) && text.includes(en["iv.judge_note"]));
  });
});

test("on the voice page the report waits behind a button for the interviewer", async () => {
  await withDom("es", ({ reportBehindClick }) => {
    const sheet = { schema: "interview.sheet.v1", session_id: "vb-1", title: "Tienda", suitability: 30, summary: "", turns: 3,
      elapsed_s: 200, filler: "haiku", scorer: "haiku", judge: "interview.report@2", other_skills: [], ...RESULT,
      slots: RESULT.slots.map((s) => ({ ...s, weight: 3, score: 30, answers: [] })) };
    const gate = reportBehindClick({ sheet, handoff: {} });
    const before = gate.text();
    assert.ok(before.includes(es["iv.report_ended"]) && before.includes(es["iv.report_private"]));
    assert.ok(!before.includes(es["iv.judge_title"]) && !before.includes("30") && !before.includes(es["iv.suitability"]),
      "nothing of the report before the click");
    const button = gate.all().find((n) => n.tagName === "BUTTON");
    assert.equal(button.text(), es["iv.report_open"]);
    button.listeners.click[0]();
    const after = gate.text();
    assert.ok(after.includes(es["iv.judge_title"]) && after.includes(es["iv.suitability"]));
    assert.ok(!es["iv.judge_note"].includes("no lo ve") && !en["iv.judge_note"].includes("does not see"));
  });
});

test("an old or status-only sheet draws no empty judgement section", async () => {
  await withDom("es", ({ accuracyBlock }) => {
    const old = { slots: [{ id: "s1", label: "Running", points: [{ id: "p1", text: "Drop", status: "treated" }] }], to_check: [] };
    assert.equal(accuracyBlock(old), null);
    assert.ok(accuracyBlock({ ...old, judge: "interview.report@2" }), "judged sheets still show their concepts");
  });
});

test("a report the model could not write says why, in the summary", async () => {
  assert.equal(es["iv.report_error"], "El modelo no pudo puntuar ni valorar esta entrevista ({reason}). Las puntuaciones son una estimación por el estado de cada punto y no hay valoración de las respuestas.");
  await withDom("es", ({ reportCard }) => {
    const base = { schema: "interview.sheet.v1", session_id: "vb-1", title: "Tienda", suitability: 40, summary: "", turns: 3,
      elapsed_s: 200, filler: "haiku", scorer: "status", other_skills: [], to_check: [],
      slots: [{ id: "s1", label: "Caja", level: "minimo", weight: 3, status: "partial", score: 40, answers: [], points: [] }] };
    const failed = reportCard({ sheet: { ...base, report_error: "timeout: timeout" }, handoff: {} }).text();
    assert.ok(failed.includes("El modelo no pudo puntuar ni valorar esta entrevista (timeout: timeout)."));
    assert.ok(!reportCard({ sheet: base, handoff: {} }).text().includes("El modelo no pudo"));
  });
});

test("who attributed the answers and decided each concept is said in words (#268)", async () => {
  const byModel = {
    ...RESULT, attribution: "after_session",
    slots: [{ ...RESULT.slots[0], points: [
      { ...RESULT.slots[0].points[0], by: "after_session" },
      { id: "p2", text: "Tipos de pisada", status: "pending" },
      { id: "p3", text: "Cómo elegir la talla", status: "unknown", by: "live_model" },
    ] }],
  };
  await withDom("es", ({ accuracyBlock }) => {
    const text = accuracyBlock(byModel).text();
    assert.ok(text.includes(es["iv.attribution.after_session"]));
    assert.ok(text.includes(es["iv.by.after_session"]) && text.includes(es["iv.by.live_model"]));
    assert.ok(!text.includes(es["iv.by.rules"]), "a pending concept says nobody decided it");
    const fallback = accuracyBlock({ ...RESULT, attribution: "rules", attribution_error: "timeout: timeout" }).text();
    assert.ok(fallback.includes(es["iv.attribution.rules"]) && fallback.includes("El modelo no pudo hacerlo (timeout: timeout)."));
    assert.ok(!accuracyBlock(RESULT).text().includes(es["iv.attribution.rules"]), "older results say nothing");
  });
  await withDom("en", ({ accuracyBlock }) => {
    assert.ok(accuracyBlock(byModel).text().includes(en["iv.by.after_session"]));
  });
  for (const key of Object.keys(es).filter((k) => k.startsWith("iv.by.") || k.startsWith("iv.attribution"))) {
    assert.ok(en[key], `${key} in English`);
  }
  const state = new LiveState();
  state.apply({ type: "accuracy", state: "reporting" });
  state.apply({ type: "accuracy", ...byModel, attribution_error: null });
  assert.equal(state.accuracy.attribution, "after_session");
  state.apply({ type: "accuracy", state: "empty" });
  assert.equal(state.accuracy, null, "nothing answered: no report, no wait");
});
