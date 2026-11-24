// Summary first (#230): what a result page shows before «Más información».
// The pure helpers (ring, verdict words, short summary, strengths,
// improvements, bars, legacy results), the home's percentage, and the
// analysis overview drawn in a stand-in DOM for a v2 and a v1 synthesis.
import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_IMPROVEMENTS, MAX_STRENGTHS, RING_LENGTH, anyLegacy, barsOf, caveatKey, caveatsOf, clampPercent, isLegacy,
  personSummary, ringDash, scalePercent, sentencesOf, shortSummary, verdictFamily, verdictKey,
} from "../../transcriber/adapters/inbound/web/js/lib/summary-first.js";
import { peopleOf } from "../../transcriber/adapters/inbound/web/js/lib/analysis-results.js";
import { analysisFacts, conclusionOf } from "../../transcriber/adapters/inbound/web/js/lib/home.js";
import { splitActions } from "../../transcriber/adapters/inbound/web/js/lib/sessions.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

// ---- a v2 synthesis (docs/skills/DESIGN.md §14.1) -----------------------------

const skill = (id, score, verdict, conclusion, extra = {}) => ({
  competency_id: id, name: id, score, verdict, conclusion, basis: "firm", confidence: "medium", caveats: [], source: "judge", cites: [], ...extra,
});

const V2 = {
  format_version: 2, conversation_type: "entrevista", job: "Desarrolladora backend",
  people: [{
    participant: "Lucía", status: "done", format_version: 2, conversation_type: "entrevista", confidence: "medium",
    strengths: [{ text: "Explica cómo resolvió un problema real. Lo midió con datos.", basis: "firm", cites: [] }],
    fit: [{ text: "Sus ejemplos cubren las tareas de API del puesto, p. ej. la caché. Falta ver pruebas.", basis: "firm", cites: [] }],
    unobserved: [], questions: [],
    overall: {
      score: 4.5, maximum: 5, minimum: 1, percent: 87.5, verdict: "strong",
      line: "Muestra con ejemplos propios lo que pide el puesto en APIs y datos. Las pruebas, menos.",
      frame: "q", confidence: "medium", caveats: ["provisional"], source: "judge",
    },
    skills: [
      skill("COMM", 5, "strong", "Respuestas ordenadas y con ejemplos. Mucho detalle."),
      skill("SOLVE", 4.6, "strong", "Resuelve con datos."),
      skill("COORD", 4, "strong", "Coordina bien."),
      skill("LISTEN", 3.9, "strong", "Escucha."),
      skill("INIT", 2, "weak", "Pocas propuestas propias.", { caveats: ["no_quotes"], required: 3 }),
      skill("COLLAB", 3, "adequate", "Delegar le cuesta.", { caveats: ["provisional"] }),
      skill("FEEDB", 3, "adequate", "Sin más.", { caveats: [] }),
    ],
    top_strengths: ["COMM", "SOLVE", "COORD", "LISTEN"],
    top_improvements: ["INIT", "COLLAB", "FEEDB"],
  }],
};

const V1 = {
  people: [{
    participant: "Lucía", status: "done", confidence: "low",
    strengths: [{ text: "Escucha antes de proponer.", basis: "observation", cites: [] }],
    fit: [], unobserved: [], questions: [],
  }],
};

const assessment = {
  conversation_type: "entrevista",
  scorecards: [{
    participant: "Lucía",
    assessments: [
      { competency_id: "COMM", status: "rated", rating: 4, evidence: [{ quote: "Lo medimos.", start: 12, polarity: "positive" }] },
      { competency_id: "LISTEN", status: "provisional", rating: null, estimate: { rating: 2, confidence: "low" } },
      { competency_id: "FEEDB", status: "insufficient_evidence", rating: null },
    ],
  }],
};

const reports = { people: [{ position: 1, name: "Lucía", strengths: ["Comunicación", "Escucha", "Iniciativa", "Otra"], to_improve: ["Feedback", "A", "B"], why: [] }] };

const v2Person = () => peopleOf({ assessment, reports, synthesis: V2 })[0];
const v1Person = () => peopleOf({ assessment, reports, synthesis: V1 })[0];

test("percentages: kept in 0–100, a scale point mapped with the lowest score at 0 %", () => {
  assert.equal(clampPercent(120), 100);
  assert.equal(clampPercent(-3), 0);
  assert.equal(clampPercent("x"), null);
  assert.equal(scalePercent(1), 0);
  assert.equal(scalePercent(5), 100);
  assert.equal(scalePercent(4.5), 87.5);
  assert.equal(scalePercent(3, 0, 4), 75);
  assert.equal(scalePercent(3, 5, 5), null);
});

test("the ring's arc is the percentage of its circle; nothing for no number", () => {
  const full = ringDash(100);
  assert.equal(full.filled, full.total);
  assert.equal(full.rest, 0);
  assert.equal(ringDash(50).filled, Math.round(RING_LENGTH * 50) / 100);
  assert.equal(ringDash(null).filled, 0);
  assert.equal(ringDash(250).filled, full.total);
});

test("the verdict is said in descriptive words by conversation type, never as a decision", () => {
  assert.equal(verdictFamily("entrevista"), "job");
  assert.equal(verdictFamily("coaching"), "coaching");
  assert.equal(verdictFamily("venta"), "sale");
  assert.equal(verdictFamily("reunion_decision"), "general");
  assert.equal(verdictFamily(""), "general");
  assert.equal(verdictKey("strong", "entrevista"), "sf.verdict.job.strong");
  assert.equal(verdictKey("odd", "venta"), "sf.verdict.sale.adequate");
  assert.equal(es["sf.verdict.job.strong"], "Encaje alto con lo que pide el puesto");
  for (const family of ["job", "coaching", "sale", "general"]) {
    for (const verdict of ["strong", "adequate", "weak"]) {
      for (const dict of [es, en]) {
        const text = dict[`sf.verdict.${family}.${verdict}`];
        assert.ok(text, `${family}.${verdict}`);
        assert.doesNotMatch(text, /fase|recomend|contrat|apto|descart|next stage|hire|recommend|reject|proceed/i);
      }
    }
  }
});

test("sentences: an abbreviation does not end one; a short summary skips repeats", () => {
  assert.deepEqual(sentencesOf("Uno, p. ej. la caché. Dos. "), ["Uno, p. ej. la caché.", "Dos."]);
  assert.deepEqual(sentencesOf(""), []);
  assert.equal(shortSummary(["A uno. B.", "C dos. D.", "E tres."]), "A uno. C dos.");
  assert.equal(shortSummary(["Igual.", "igual", "Otra."], { max: 3 }), "Igual. Otra.");
  assert.equal(shortSummary(["Ya dicho. X.", "Nuevo."], { skip: ["Ya dicho."] }), "Nuevo.");
  const long = shortSummary([`${"palabra ".repeat(40)}fin.`]);
  assert.ok(long.length <= 180 && long.endsWith("…"));
});

test("a v2 person: overall, a two-sentence summary, at most 3 strengths and 2 improvements, a bar per skill", () => {
  const s = personSummary(v2Person());
  assert.equal(s.overall.percent, 87.5);
  assert.equal(s.overall.verdict, "strong");
  assert.equal(s.overall.line, "Muestra con ejemplos propios lo que pide el puesto en APIs y datos.");
  assert.equal(s.summary, "Sus ejemplos cubren las tareas de API del puesto, p. ej. la caché. Explica cómo resolvió un problema real.");
  assert.equal(s.strengths.length, MAX_STRENGTHS);
  assert.deepEqual(s.strengths[0], { id: "COMM", name: "COMM", text: "Respuestas ordenadas y con ejemplos." });
  assert.equal(s.improvements.length, MAX_IMPROVEMENTS);
  assert.deepEqual(s.improvements.map((i) => i.id), ["INIT", "COLLAB"]);
  assert.equal(s.legacy, false);
  assert.equal(s.bars.length, 7);
  assert.deepEqual(s.bars[0], { id: "COMM", name: s.bars[0].name, percent: 100, verdict: "strong", required: null });
  const init = s.bars.find((b) => b.id === "INIT");
  assert.equal(init.percent, 25);
  assert.equal(init.required, 50);
  assert.deepEqual(s.more.caveats, ["provisional", "no_quotes"]);
  assert.equal(s.more.confidence, "medium");
  assert.equal(s.more.quotes, 1);
  assert.ok(s.more.notices >= 2);
});

test("a v1 person: no overall, legacy, the reports' names and the stored levels", () => {
  const p = v1Person();
  assert.equal(isLegacy(p), true);
  const s = personSummary(p);
  assert.equal(s.overall, null);
  assert.equal(s.legacy, true);
  assert.equal(s.summary, "Escucha antes de proponer.");
  assert.deepEqual(s.strengths.map((x) => x.name), ["Comunicación", "Escucha", "Iniciativa"]);
  assert.deepEqual(s.improvements.map((x) => x.name), ["Feedback", "A"]);
  assert.deepEqual(barsOf(p).map((b) => [b.id, b.percent, b.verdict]), [["COMM", 75, "strong"], ["LISTEN", 25, "weak"]]);
  assert.equal(anyLegacy([s]), true);
  // No synthesis at all, or a skipped one: nothing to redo from here.
  assert.equal(isLegacy(peopleOf({ assessment, reports })[0]), false);
  assert.equal(isLegacy(peopleOf({ assessment, reports, synthesis: { people: [{ participant: "Lucía", status: "skipped" }] } })[0]), false);
});

test("caveats: overall then each skill's, once; known codes have words in both languages", () => {
  assert.deepEqual(caveatsOf({ overall: { caveats: ["a", "b"] }, skills: [{ caveats: ["b", "c"] }, {}] }), ["a", "b", "c"]);
  assert.deepEqual(caveatsOf(null), []);
  assert.equal(caveatKey("odd"), null);
  for (const code of ["provisional", "observations_only", "no_quotes", "short_participation", "no_firm_levels", "judge_unavailable"]) {
    assert.ok(es[caveatKey(code)] && en[caveatKey(code)], code);
  }
});

test("the home's conclusion is the overall line with its percentage when the synthesis has one", async () => {
  assert.deepEqual(conclusionOf(V2), {
    person: "Lucía", text: "Muestra con ejemplos propios lo que pide el puesto en APIs y datos.", percent: 87.5,
  });
  assert.deepEqual(conclusionOf(V1), { person: "Lucía", text: "Escucha antes de proponer." });
  const computed = conclusionOf({ people: [{ participant: "Eva", overall: { score: 3.45, maximum: 5, minimum: 1, line: "" }, fit: [{ text: "Bien." }] }] });
  assert.deepEqual(computed, { person: "Eva", text: "Bien.", percent: 61.3 });
  const api = {
    analysisStatus: async () => ({ steps: [{ name: "assessment", status: "done" }, { name: "synthesis", status: "done" }] }),
    analysisStep: async () => V2,
  };
  assert.equal((await analysisFacts(api, { id: "x", state: "done" })).conclusion.percent, 87.5);
});

test("a session card shows the actions that move it forward; the rest go under «Más opciones»", () => {
  assert.deepEqual(splitActions({ state: "done" }), { main: ["open"], more: ["transcribe", "describe", "delete"] });
  assert.deepEqual(splitActions({ state: "uploaded" }), { main: ["transcribe"], more: ["describe", "delete"] });
  assert.deepEqual(splitActions({ state: "recording" }), { main: ["resume", "finish"], more: ["delete"] });
  assert.deepEqual(splitActions({ state: "running" }), { main: [], more: ["describe"] });
});

// ---- the overview in a stand-in DOM ---------------------------------------------

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
  setAttribute(key, value) { this.attributes[key] = value; if (key === "class") this.className = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.children.map((c) => (c.text ? c.text() : c.textContent)).join(""); }
  // The text outside every closed <details> (what shows before «Más información»).
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

const RUN = {
  session_id: "2026-11-24-100000-entrevista", status: "done", active: false, finished_at: "2026-11-24T10:40:00Z",
  steps: ["sentences", "interaction", "assessment", "reports", "synthesis"].map((name) => ({ name, status: "done" })),
};
const SENTENCES = { conversation_type: "entrevista", sentences: [{ id: "s0", speaker: "Lucía", text: "Lo medimos.", start: 12, end: 14 }] };

async function panelFor(synthesis, onRedo = null) {
  const { resultsPanel } = await import("../../transcriber/adapters/inbound/web/js/views/analysis-results.js");
  const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
  setLanguage("es");
  return resultsPanel({ id: RUN.session_id, run: RUN, data: { sentences: SENTENCES, assessment, reports, synthesis }, onRedo });
}

test("v2 overview: ring, descriptive verdict, line and lists first; confidence and caveats only in «Más información»", async () => {
  await withDom(async () => {
    const panel = await panelFor(V2);
    const shown = panel.shown();
    const nodes = panel.all();
    assert.ok(shown.includes(es["sf.verdict.job.strong"]));
    assert.ok(shown.includes("87,5"));
    assert.ok(shown.includes("Muestra con ejemplos propios"));
    assert.ok(shown.includes(es["sf.strengths"]) && shown.includes(es["sf.improvements"]));
    assert.ok(shown.includes(es["sf.bars_job"]));
    assert.ok(shown.includes(es["sf.min_key"]), "INIT has a minimum: the key is shown");
    const ring = nodes.find((n) => n.tagName === "SVG" && n.attributes.role === "img" && /87,5/.test(n.attributes["aria-label"] ?? ""));
    assert.ok(ring, "the ring names its value");
    const minimum = nodes.filter((n) => n.attributes.class === "sf-bar-min");
    assert.equal(minimum.length, 1);
    // Nothing tentative before «Más información»; all of it inside.
    for (const key of ["sf.caveat.provisional", "sf.caveat.no_quotes", "sf.reliability_note", "res.draft_text", "res.fairness"]) {
      assert.ok(!shown.includes(es[key]), `${key} hidden`);
      assert.ok(panel.text().includes(es[key]), `${key} kept`);
    }
    assert.doesNotMatch(shown, /provisional|orientativ|insuficiente|siguiente fase/i);
    assert.ok(!nodes.some((n) => n.tagName === "BUTTON" && n.text?.() === es["sa.redo_synthesis"]));
  });
});

test("v1 overview: no ring, the earlier version said once, the synthesis can be redone from the summary", async () => {
  await withDom(async () => {
    let redone = 0;
    const panel = await panelFor(V1, () => { redone += 1; });
    const shown = panel.shown();
    const nodes = panel.all();
    assert.ok(!nodes.some((n) => n.className === "sf-ring"));
    assert.equal(shown.split(es["sf.legacy"]).length - 1, 1, "said once");
    assert.ok(shown.includes("Comunicación") && shown.includes("Feedback"));
    const redo = nodes.find((n) => n.tagName === "BUTTON" && n.text() === es["sa.redo_synthesis"]);
    redo.listeners.click[0]();
    assert.equal(redone, 1);
    // Without the action (a run in progress), the sentence stays, the button does not.
    const quiet = await panelFor(V1, null);
    assert.ok(!quiet.all().some((n) => n.tagName === "BUTTON" && n.text() === es["sa.redo_synthesis"]));
  });
});
