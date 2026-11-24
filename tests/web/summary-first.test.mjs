// Summary first (#230): what a result page shows before «Más información».
// The pure helpers (ring, verdict words, short summary, strengths,
// improvements, bars, legacy results), the home's percentage, and the
// analysis overview drawn in a stand-in DOM for a v2 and a v1 synthesis.
import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_BARS, MAX_IMPROVEMENTS, MAX_STRENGTHS, RING_LENGTH, splitBars, anyLegacy, barsOf, caveatKey, caveatsOf, clampPercent, isLegacy,
  personSummary, ringDash, scalePercent, sentencesOf, shortSummary, verdictFamily, verdictKey,
} from "../../transcriber/adapters/inbound/web/js/lib/summary-first.js";
import { peopleOf } from "../../transcriber/adapters/inbound/web/js/lib/analysis-results.js";
import { analysisFacts, clauseCut, conclusionOf } from "../../transcriber/adapters/inbound/web/js/lib/home.js";
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
  // Whole sentences only: a long one is cut at a clause, or left out; never «…».
  const clause = shortSummary([`${"uno ".repeat(20)}, ${"dos ".repeat(15)}; ${"tres ".repeat(30)}fin.`]);
  assert.ok(clause.length <= 180 && clause.endsWith(".") && !clause.includes("…"), clause);
  assert.equal(shortSummary([`${"palabra ".repeat(40)}fin.`, "Corta."]), "Corta.");
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

test("a cut keeps whole clauses, closed with a full stop, never «…»; nothing when there is no clause to keep", () => {
  assert.equal(clauseCut("Corta. Otra.", 50), "Corta.");
  assert.equal(clauseCut("Primera parte larga, segunda parte que no cabe en el límite.", 30), "Primera parte larga.");
  assert.equal(clauseCut("Sin comas ni pausas en toda esta frase tan larga.", 20), "");
  assert.equal(clauseCut("A, b c d e f g h i j k l m n o p.", 20), "", "a boundary too early is no clause");
  assert.equal(clauseCut("", 20), "");
  assert.equal(clauseCut("Propone los pasos siguientes, cuando hay dudas (preguntar hoy, asignar el resto) y sigue luego.", 70),
    "Propone los pasos siguientes.",
    "never inside an open bracket");
});

test("up front only the bars of the strengths and improvements named, in that order, at most five", () => {
  const bars = ["A", "B", "C", "D", "E", "F", "G"].map((id) => ({ id, name: `n${id}` }));
  const { shown, all } = splitBars(bars, [{ id: "C" }, { id: "A" }, { id: "Z" }, { id: "B" }, { id: "D" }, { id: "E" }, { id: "F" }]);
  assert.deepEqual(shown.map((b) => b.id), ["C", "A", "B", "D", "E"]);
  assert.equal(shown.length, MAX_BARS);
  assert.equal(all.length, 7);
  assert.deepEqual(splitBars(bars, [{ id: "", name: "nG" }]).shown.map((b) => b.id), ["G"], "older results match by name");
  const s = personSummary(v2Person());
  assert.deepEqual(s.shownBars.map((b) => b.id), ["COMM", "SOLVE", "COORD", "INIT", "COLLAB"]);
  assert.equal(s.bars.length, 7);
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
    this.style = {};
  }
  setAttribute(key, value) { this.attributes[key] = value; if (key === "class") this.className = value; }
  querySelector(selector) { return this.all().find((n) => n !== this && n.tagName === selector.toUpperCase()) ?? null; }
  focus() { globalThis.focused = this; }
  scrollIntoView() {}
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

async function panelFor(synthesis, onRedo = null, extra = {}) {
  const { resultsPanel } = await import("../../transcriber/adapters/inbound/web/js/views/analysis-results.js");
  const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
  setLanguage("es");
  return resultsPanel({ id: RUN.session_id, run: RUN, data: { sentences: SENTENCES, assessment, reports, synthesis, ...extra }, onRedo });
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
    assert.ok(minimum.length >= 1);
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

test("v2 «Más información» keeps everything: the question, the whole line, every skill's full conclusion and basis", async () => {
  await withDom(async () => {
    const panel = await panelFor(V2);
    const more = panel.all().find((n) => n.tagName === "DETAILS" && n.className === "sf-more");
    const inside = more.text();
    assert.ok(inside.includes("Muestra con ejemplos propios lo que pide el puesto en APIs y datos. Las pruebas, menos."));
    assert.ok(inside.includes(es["sf.frame"]) && inside.includes("q"));
    for (const k of V2.people[0].skills) assert.ok(inside.includes(k.conclusion), k.competency_id);
    assert.ok(inside.includes(es["sf.basis.firm"]) && inside.includes(es["sf.source.judge"]));
    // The bars of all seven competencies sit behind «Ver las 7 competencias».
    const all = panel.all().find((n) => n.tagName === "DETAILS" && n.className === "sf-all-bars");
    assert.ok(all.text().includes(es["sf.all_bars"].replace("{count}", "7")));
    assert.equal(all.all().filter((n) => n.tagName === "LI").length, 7);
  });
});

test("only the first person's card is open; a row of the overview opens another and moves the focus to it", async () => {
  await withDom(async () => {
    const two = { ...V2, people: [V2.people[0], { ...V2.people[0], participant: "Pablo" }] };
    const people = { people: [...reports.people, { position: 2, name: "Pablo", strengths: [], to_improve: [], why: [] }] };
    const panel = await panelFor(two, null, { reports: people });
    const cards = panel.all().filter((n) => n.tagName === "DETAILS" && n.className.startsWith("sf-person"));
    assert.deepEqual(cards.map((c) => c.open), [true, false]);
    // A closed card still shows its name, percentage, verdict words and line.
    const head = cards[1].children[0];
    assert.equal(head.tagName, "SUMMARY");
    for (const bit of ["Pablo", "87,5", es["sf.verdict.job.strong"], "Muestra con ejemplos propios"]) assert.ok(head.text().includes(bit), bit);
    const row = panel.all().filter((n) => n.className === "sf-glance-row")[1];
    row.listeners.click[0]();
    assert.equal(cards[1].open, true);
    assert.equal(globalThis.focused, head);
  });
});

test("the home reads its percentage out with words, not only a tooltip", async () => {
  await withDom(async () => {
    const { homeView } = await import("../../transcriber/adapters/inbound/web/js/views/home.js");
    const root = new FakeElement("main");
    root.classList = { add() {}, remove() {} };
    const api = {
      listSessions: async () => ({ sessions: [{ id: "x", state: "done", title: "T", started_at: "2026-11-24T10:00:00Z" }] }),
      analysisStatus: async () => ({ steps: [{ name: "assessment", status: "done" }, { name: "synthesis", status: "done" }] }),
      analysisStep: async () => V2,
    };
    await homeView({ api }, root);
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    const score = root.all().find((n) => n.className === "recent-score");
    assert.ok(score, "the row has a score");
    const hidden = score.children.find((c) => c.className === "sr-only");
    assert.equal(hidden.text(), es["sf.home_score"].replace("{value}", "87,5 %"));
    assert.equal(score.children[0].attributes["aria-hidden"], "true");
  });
});

test("the interviewer's report: suitability, summary and bars first; tiles, notes, quotes and gaps behind «Más información»", async () => {
  await withDom(async () => {
    const { reportCard } = await import("../../transcriber/adapters/inbound/web/js/views/interview.js");
    const result = {
      sheet: {
        title: "Backend", session_id: "iv-1", suitability: 72.4, summary: "Cubre lo pedido en APIs.",
        turns: 18, elapsed_s: 900, turns_to_minimums: 9, filler: 2, scorer: "judge",
        slots: [
          { label: "APIs", level: "minimo", weight: 5, score: 85, status: "filled", notes: "Nota APIs", confidence: 0.8, scored_by: "judge",
            answers: [{ quote: "Diseñé la API", utterance: "u3" }] },
          { label: "Tests", level: "minimo", weight: 4, score: 30, status: "partial", notes: "Nota tests", confidence: 0.5, answers: [] },
        ],
        other_skills: [{ label: "Inglés", quote: "Hablo inglés" }],
      },
      handoff: { transcript: true, analysis: true },
    };
    const card = reportCard(result);
    const more = card.all().find((n) => n.tagName === "DETAILS" && n.className === "sf-more");
    assert.equal(more.open, false);
    const shown = card.shown();
    for (const bit of ["72", "Cubre lo pedido en APIs.", "APIs", "Tests", es["sf.iv_draft_short"]]) assert.ok(shown.includes(bit), bit);
    for (const bit of ["Nota APIs", "Diseñé la API", "Hablo inglés", es["iv.draft"], es["iv.tile.gaps"]]) {
      assert.ok(!shown.includes(bit), `${bit} hidden`);
      assert.ok(more.text().includes(bit), `${bit} kept`);
    }
    assert.ok(card.all().some((n) => n.tagName === "A" && n.href === "#/analysis/iv-1"));
  });
});
