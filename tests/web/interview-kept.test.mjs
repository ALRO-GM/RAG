// Interview reports kept in the candidate's file and positions closed (#262):
// the comparison lists both kinds and opens each report, the stored report
// page, the candidate's list of reports, the live end summary's link, and
// the texts of closing (which deletes nothing). Language pinned; stand-in DOM.
import test from "node:test";
import assert from "node:assert/strict";
import { comparisonRows, reportHref } from "../../transcriber/adapters/inbound/web/js/lib/interview.js";
import { LiveState } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const LIVE = {
  schema: "interview.sheet.v1", kind: "live", session_id: "r1", plan_id: "dev", title: "Backend", candidate: "Ana Ruiz",
  candidate_id: "ana-ruiz", finished_at: "2026-11-25T10:30:00", judge: "interview.report@2", suitability: null,
  slots: [{ id: "s1", label: "Bases de datos", level: "minimo", status: "partial", score: null, answers: [],
    judgement: { label: "correct_basic", reason: "Correcto pero breve.", quote: "uso índices", utterance: "t1.s0", basis: "general", cites: [] },
    accuracy: { correct_basic: 1 }, points: [] }],
  to_check: [],
};

test("the comparison knows the kind of each report and where it opens", () => {
  const rows = comparisonRows({ plan: { id: "dev", revision: 0, slots: [{ id: "s1" }] }, candidates: [
    { session_id: "r1", kind: "live", candidate: "Ana", finished_at: "2", suitability: null, slots: [{ id: "s1", status: "partial", score: null }] },
    { session_id: "vb 2", candidate: "Luis", finished_at: "1", suitability: 70, slots: [] },
  ] });
  assert.deepEqual(rows.map((r) => r.kind), ["live", "ai"]);
  assert.equal(reportHref("dev", "vb 2"), "#/interview-report/dev/vb%202");
});

test("the live state keeps where a round's report was saved", () => {
  const state = new LiveState();
  state.apply({ type: "accuracy", state: "ready", saved: true, plan_id: "dev", session_id: "r1", slots: [], to_check: [] });
  assert.deepEqual([state.accuracy.saved, state.accuracy.plan_id, state.accuracy.session_id], [true, "dev", "r1"]);
  state.apply({ type: "accuracy", state: "failed" });
  assert.equal(state.accuracy.saved, false);
});

test("closing says it deletes nothing; the retention line says what is true", () => {
  assert.equal(es["iv.close_body"], "No se podrán empezar entrevistas nuevas con este plan, ni con IA ni en directo, y aparecerá como cerrado. No se borra nada: los informes siguen en la ficha de cada persona hasta que la borres. Puedes reabrirlo cuando quieras.");
  assert.equal(es["iv.retention"], "Se conserva en la ficha de la persona hasta que la borres.");
  assert.match(es["cand.delete_body"], /sus informes de entrevista guardados \({reports}\)/);
  assert.match(es["cand.privacy"], /se conserva con ellos hasta que borres la ficha/);
  assert.match(es["cand.privacy"], /#151/);
  assert.match(es["live.accuracy_failed"], /no se ha guardado ningún informe/);
  for (const dict of [es, en]) {
    for (const key of ["iv.close_position", "iv.reopen_position", "iv.close_title", "iv.closed_notice", "iv.kind.ai", "iv.kind.live",
      "iv.open_report", "rep.iv_title", "rep.iv_missing", "cand.reports_title", "live.accuracy_saved", "live.accuracy_retry", "live.plan_closed"]) {
      assert.ok(dict[key], key);
    }
  }
});

// ---- views in a stand-in DOM ----------------------------------------------

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
    this.classList = { add: () => {} };
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.open = false;
    this.hidden = false;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelector() { return null; }
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
    return await fn();
  } finally {
    Object.assign(globalThis, saved);
  }
}

test("a stored live report opens with its judgement, kind, date and retention", async () => {
  await withDom("es", async () => {
    const { reportPage } = await import("../../transcriber/adapters/inbound/web/js/views/interview-report.js");
    const page = reportPage(LIVE).text();
    assert.ok(page.includes("Ana Ruiz") && page.includes(es["iv.kind.live"]) && page.includes(es["iv.retention"]));
    assert.ok(page.includes(es["iv.judge_title"]) && page.includes("Correcto pero breve."));
    assert.ok(!page.includes(es["iv.suitability"]), "a live report has no score");
    const ai = reportPage({ ...LIVE, kind: "ai", suitability: 60, slots: LIVE.slots.map((s) => ({ ...s, score: 60, weight: 3 })) }).text();
    assert.ok(ai.includes(es["iv.kind.ai"]) && ai.includes(es["iv.suitability"]));
  });
});

test("the comparison opens every report; the candidate's file lists theirs", async () => {
  await withDom("es", async () => {
    const { comparisonCard } = await import("../../transcriber/adapters/inbound/web/js/views/interview.js");
    const { reportList } = await import("../../transcriber/adapters/inbound/web/js/views/candidate.js");
    const card = comparisonCard({ plan: { id: "dev", revision: 0, slots: [{ id: "s1", label: "Bases de datos" }] }, candidates: [
      { session_id: "r1", kind: "live", candidate: "Ana Ruiz", finished_at: "2026-11-25T10:30:00", suitability: null,
        plan_revision: 0, slots: [{ id: "s1", status: "partial", score: null }] }] });
    const text = card.text();
    assert.ok(text.includes(es["iv.kind.live"]) && text.includes(es["iv.open_report"]) && text.includes(es["iv.no_score_live"]));
    const link = card.all().find((n) => n.tagName === "A");
    assert.equal(link.href, "#/interview-report/dev/r1");
    assert.equal(reportList([]), null);
    const list = reportList([{ session_id: "r1", kind: "live", plan_id: "dev", finished_at: "2026-11-25T10:30:00" },
      { session_id: "vb-1", kind: "ai", plan_id: "dev", finished_at: "2026-11-20T09:00:00" }]);
    assert.ok(list.text().includes(es["cand.reports_title"]) && list.text().includes(es["iv.retention"]));
    assert.deepEqual(list.all().filter((n) => n.tagName === "A").map((n) => n.href),
      ["#/interview-report/dev/r1", "#/interview-report/dev/vb-1"]);
  });
});

test("a report without a person's file says so and can be deleted from the comparison", async () => {
  assert.equal(es["iv.retention_no_file"], "Sin ficha de persona: se conserva con el puesto hasta que lo borres aquí.");
  assert.equal(es["iv.delete_report_body"], "Se borra el informe de esta entrevista, con sus citas y valoraciones. No se puede deshacer.");
  await withDom("es", async () => {
    const { comparisonCard } = await import("../../transcriber/adapters/inbound/web/js/views/interview.js");
    const { reportPage } = await import("../../transcriber/adapters/inbound/web/js/views/interview-report.js");
    const asked = [];
    const results = { plan: { id: "dev", revision: 0, slots: [] }, candidates: [
      { session_id: "vb-1", candidate: "Sin ficha", finished_at: "2026-11-20T09:00:00", suitability: 50, plan_revision: 0, slots: [] },
      { session_id: "r1", kind: "live", candidate: "Ana Ruiz", candidate_id: "ana-ruiz", plan_revision: 0, slots: [] }] };
    const card = comparisonCard(results, { onDelete: (row) => asked.push(row.session) });
    const text = card.text();
    assert.ok(text.includes(es["iv.retention"]) && text.includes(es["iv.retention_no_file"]) && text.includes(es["iv.no_file"]));
    const buttons = card.all().filter((n) => n.tagName === "BUTTON");
    assert.equal(buttons.length, 1, "only the report without a file has a delete button");
    buttons[0].listeners.click[0]();
    assert.deepEqual(asked, ["vb-1"]);
    const onlyFiled = comparisonCard({ ...results, candidates: [results.candidates[1]] }).text();
    assert.ok(!onlyFiled.includes(es["iv.retention_no_file"]));
    assert.ok(reportPage({ ...LIVE, candidate_id: null }).text().includes(es["iv.retention_no_file"]));
    assert.ok(!reportPage({ ...LIVE, candidate_id: null }).text().includes(es["iv.retention"]));
  });
});
