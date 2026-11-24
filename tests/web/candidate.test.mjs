// The candidate across rounds (#235): the answers in words, the rows of the
// «Datos de la candidata» block and the live row, the `core` feed message,
// the job reading only for the candidate, and the block drawn in a
// stand-in DOM (compact first, corrections and deletion inside).
import test from "node:test";
import assert from "node:assert/strict";
import { CORE_IDS, answerText, coreCount, coreRows, grouped, questionLabel } from "../../transcriber/adapters/inbound/web/js/lib/candidate.js";
import { jobReading, rolesOf } from "../../transcriber/adapters/inbound/web/js/lib/analysis-results.js";
import { LiveState } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const tEs = (key, params = {}) => String(es[key] ?? key).replace(/\{(\w+)\}/g, (_, k) => String(params[k] ?? ""));

const question = (id, order, answer = null, proposal = null) => ({
  id, order, phase: "opening", question: { es: `¿${id}?`, en: `${id}?` }, fields: [], answer, proposal,
});
const answer = (fields, extra = {}) => ({ question: "x", fields, summary: "", source: "confirmed", session_id: "r1", at: "2026-11-24T10:00:00", ...extra });

const PROFILE = {
  schema: "interview.candidate.v1", id: "lucia-marin", plan_id: "dev", name: "Lucía Marín", answered: 3, proposed: 1,
  questions: [
    question("salary_expectation", 5, null, answer({ "salary.expectation_min": 28000, "salary.expectation_max": 32000 }, { source: "proposed" })),
    question("work_mode_location", 1, answer({ "availability.work_mode_ok": false, "availability.location_note": null })),
    question("work_authorization", 2, answer({ "eligibility.work_authorization": true }, { source: "corrected" })),
    question("start_date_notice", 3, answer({ "availability.earliest_start": "en dos semanas", "availability.notice_days": 15 })),
    question("motivation", 4),
  ],
  rounds: [{ session_id: "r1" }, { session_id: "r2" }],
};

test("every core question has a short label in both languages, and every new text exists in both", () => {
  for (const id of CORE_IDS) {
    assert.ok(es[`cand.q.${id}`] && en[`cand.q.${id}`], id);
  }
  for (const key of Object.keys(es).filter((k) => k.startsWith("cand.") || k.startsWith("live.core") || k.startsWith("live.candidate") || k.startsWith("live.plan"))) {
    assert.ok(en[key], `${key} in English`);
  }
  assert.ok(es["live.notice.core_not_kept"] && en["live.notice.core_not_kept"]);
});

test("answers in words: yes/no from the dictionary, euros grouped, start and notice, kept words", () => {
  const rows = coreRows(tEs, PROFILE, "es");
  assert.deepEqual(rows.map((r) => r.id), ["work_mode_location", "work_authorization", "start_date_notice", "motivation", "salary_expectation"]);
  assert.deepEqual(rows.map((r) => r.text), ["No", "Sí", "en dos semanas · preaviso de 15 días", "", ""]);
  // A proposal never counts as answered.
  assert.deepEqual(rows.map((r) => r.answered), [true, true, true, false, false]);
  assert.deepEqual(rows.map((r) => r.proposed), [false, false, false, false, true]);
  assert.equal(rows[4].proposal, "28.000–32.000 € al año");
  assert.equal(rows[1].source, "corrected");
  assert.equal(rows[0].source, "confirmed");
  assert.equal(rows[3].source, "");
  assert.equal(rows[0].label, es["cand.q.work_mode_location"]);
  assert.deepEqual(coreCount(PROFILE), { done: 3, proposed: 1, all: 5 });
  assert.deepEqual(coreCount(null), { done: 0, proposed: 0, all: 0 });
  assert.equal(grouped(30000, "en"), "30,000");
  assert.equal(grouped("x"), "");
  const one = { id: "salary_expectation", answer: answer({ "salary.expectation_min": 30000, "salary.expectation_max": 30000 }) };
  assert.equal(answerText(tEs, one), "30.000 € al año");
  const none = { id: "start_date_notice", answer: answer({ "availability.earliest_start": null, "availability.notice_days": 0 }) };
  assert.equal(answerText(tEs, none), "sin preaviso");
  const typed = { id: "start_date_notice", answer: answer({}, { summary: "Tras Semana Santa", source: "corrected" }) };
  assert.equal(answerText(tEs, typed), "Tras Semana Santa");
  assert.equal(answerText(tEs, { id: "x" }), "");
  assert.equal(questionLabel(tEs, { id: "other", question: { es: "¿Otra?" } }), "¿Otra?");
});

test("the live state keeps the newest profile and marks what was captured", () => {
  const state = new LiveState();
  state.apply({ type: "core", profile: PROFILE });
  assert.equal(state.core.id, "lucia-marin");
  state.apply({ type: "core", profile: { ...PROFILE, answered: 5 }, captured: "motivation" });
  assert.equal(state.core.answered, 5);
  assert.ok(state.arrived.has("core|motivation"));
  state.apply({ type: "core", profile: { broken: true } });
  assert.equal(state.core.answered, 5, "a malformed profile is ignored");
});

test("with roles only the candidate is read against the job; without roles everyone, as before", () => {
  const data = { sentences: { speakers: [{ name: "Ana", role: "interviewer" }, { name: "Lucía", role: "candidate" }, { name: "Eco" }] } };
  const roles = rolesOf(data);
  assert.deepEqual(roles, { Ana: "interviewer", Lucía: "candidate" });
  assert.equal(jobReading(roles, "Ana"), false);
  assert.equal(jobReading(roles, "Lucía"), true);
  assert.equal(jobReading(roles, "Eco"), true);
  assert.deepEqual(rolesOf({}), {});
  assert.equal(jobReading({}, "Ana"), true);
});

// ---- the block in a stand-in DOM -------------------------------------------

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
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.open = false;
    this.hidden = false;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  querySelector(selector) {
    const [tag, cls] = selector.split(".");
    return this.all().find((n) => n !== this && n.tagName === tag.toUpperCase() && (!cls || n.className.split(" ").includes(cls))) ?? null;
  }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  remove() { this.removed = true; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.children.map((c) => (c.text ? c.text() : c.textContent)).join(""); }
  shown() {
    return this.children.map((c) => {
      if (!c.shown) return c.textContent;
      if (c.tagName === "DETAILS" && !c.open) return c.children[0]?.text?.() ?? "";
      return c.shown();
    }).join("");
  }
}

async function withDom(fn) {
  const saved = { document: globalThis.document, Node: globalThis.Node };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  try {
    return await fn();
  } finally {
    Object.assign(globalThis, saved);
  }
}

test("«Datos de la candidata»: compact first, «Confirmar datos» for proposals, corrections inside «Más información»", async () => {
  await withDom(async () => {
    const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
    setLanguage("es");
    const { candidateBlock } = await import("../../transcriber/adapters/inbound/web/js/views/candidate.js");
    const calls = [];
    const confirmed = { ...PROFILE, questions: PROFILE.questions.map((q) => (q.id === "salary_expectation" ? { ...q, answer: { ...q.proposal, source: "confirmed" }, proposal: null } : q)) };
    const api = {
      correctAnswer: async (...args) => { calls.push(["correct", ...args]); return { ...PROFILE, questions: PROFILE.questions.map((q) => (q.id === "motivation" ? { ...q, answer: answer({ "motivation.summary": "Crecer" }, { summary: "Crecer", source: "corrected" }) } : q)) }; },
      clearAnswer: async (...args) => { calls.push(["clear", ...args]); return PROFILE; },
      confirmAnswers: async (...args) => { calls.push(["confirm", ...args]); return confirmed; },
      discardProposal: async (...args) => { calls.push(["discard", ...args]); return PROFILE; },
    };
    const block = candidateBlock(PROFILE, { api });
    const shown = block.shown();
    assert.ok(shown.includes(es["cand.title"]));
    assert.ok(shown.includes("Lucía Marín") && shown.includes("3 de 5 preguntas comunes") && shown.includes("2 rondas"));
    assert.ok(shown.includes("Por confirmar: 28.000–32.000 € al año"), "a proposal is said as such");
    assert.ok(shown.includes(es["cand.confirm_title"]) && shown.includes(es["cand.confirm_hint"]));
    assert.ok(shown.includes(es["cand.pending"]), "the open question says so");
    assert.ok(!shown.includes(es["cand.source_corrected"]));
    // Confirm the one proposal: it is sent and the block draws the answer.
    const confirm = block.all().find((n) => n.tagName === "BUTTON" && n.text() === es["cand.confirm"]);
    confirm.listeners.click[0]();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(calls[0], ["confirm", "dev", "lucia-marin", ["salary_expectation"]]);
    assert.ok(!block.shown().includes(es["cand.confirm_title"]), "nothing left to confirm");
    assert.ok(block.shown().includes("4 de 5 preguntas comunes"));
    const more = block.querySelector("details.sf-more");
    more.open = true;
    more.listeners.toggle[0]();
    const inside = block.text();
    assert.ok(inside.includes(es["cand.source_corrected"]) && inside.includes(es["cand.source_confirmed"]));
    assert.ok(inside.includes("#151"), "the pending owner decisions are said");
    // Correct the open motivation: the typed words are sent and the block draws the answer.
    const details = block.querySelector("ul.sf-cand-details");
    const inputs = details.all().filter((n) => n.tagName === "INPUT");
    inputs[3].value = "Crecer";
    const saves = details.all().filter((n) => n.tagName === "BUTTON" && n.text() === es["cand.save"]);
    saves[3].listeners.click[0]();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(calls[1], ["correct", "dev", "lucia-marin", "motivation", "Crecer"]);
    assert.ok(block.shown().includes("Crecer"));
    assert.ok(block.querySelector("details.sf-more").open, "the details stay open after a correction");
    // Discard and confirm all, on a fresh block.
    const again = candidateBlock({ ...PROFILE, questions: [...PROFILE.questions, question("extra", 6, null, answer({}, { summary: "x", source: "proposed" }))] }, { api });
    const all = again.all().find((n) => n.tagName === "BUTTON" && n.text() === "Confirmar los 2");
    all.listeners.click[0]();
    again.all().find((n) => n.tagName === "BUTTON" && n.text() === es["cand.discard"]).listeners.click[0]();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(calls.slice(2).map((c) => c[0]), ["confirm", "discard"]);
    assert.equal(calls[2].length, 3, "confirm all sends no list");
    // A read-only block has no edit fields, no confirmation and no delete button.
    const quiet = candidateBlock(PROFILE);
    const box = quiet.querySelector("details.sf-more");
    box.open = true;
    box.listeners.toggle[0]();
    assert.ok(!quiet.all().some((n) => n.tagName === "INPUT" || n.tagName === "BUTTON"));
  });
});

test("the talk balance: each side's share of the final sentences' time, null before anyone spoke", () => {
  const state = new LiveState();
  assert.equal(state.talkBalance(), null);
  state.apply({ type: "session", session: { page: { roles: { S1: "interviewer", S2: "candidate" } } } });
  const final = (id, speaker, start, end) => ({ type: "event", event: { type: "final", utterance: { id, speaker, start, end, text: "x", revision: 0 } } });
  state.apply(final("t0.s0", "S1", 0, 6));
  state.apply(final("t1.s0", "S2", 6, 20));
  state.apply(final("t2.s0", "S2", 20, 20)); // no length: ignored
  assert.deepEqual(state.talkBalance(), { employee: 30, other: 70, seconds: 20 });
});

test("a later round's topics keep what earlier rounds left", () => {
  const state = new LiveState();
  state.apply({ type: "topics", topics: [
    { id: "s2", name: "Pruebas", done: false, earlier: "weak" },
    { id: "s1", name: "Python", done: true, earlier: "covered" },
    { id: "s3", name: "Datos", done: false, earlier: "nonsense" },
    { id: "s4", name: "Inglés", done: false, earlier: "revisit" },
  ] });
  assert.deepEqual(state.topicChips().map((c) => [c.id, c.earlier ?? null, c.done]), [["s2", "weak", false], ["s1", "covered", true], ["s3", null, false], ["s4", "revisit", false]]);
  // A live round's «revisit» is a prompt, never a rating.
  assert.doesNotMatch(es["live.earlier.revisit"], /nivel/);
  assert.match(es["live.earlier.revisit"], /no es una valoración/);
  for (const key of ["live.topics_next", "live.earlier.weak", "live.earlier.revisit", "live.earlier.open", "live.earlier.covered", "live.talk_label", "live.talk_you", "live.talk_other", "live.talk_balance"]) {
    assert.ok(es[key] && en[key], key);
  }
});
