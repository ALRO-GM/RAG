// «Ofertas» (#271): the list of offers and one offer's table of people. The
// pure part (the address's state, sorting, searching, filtering) and the page
// in a stand-in DOM with the language pinned: real table semantics, aria-sort
// on the sorted column, keyboard-operable buttons and selects, words for every
// state (never the colour alone), the draft notice and «Cómo se ordena» where
// the order is shown, and no verdict wording. Invented people only.
import test from "node:test";
import assert from "node:assert/strict";
import {
  POINT_FILTERS, ariaSort, boardRows, closedCount, filtering, nextSort, offerHref, offerList, optionalPoints,
  parseQuery, personHref, pointColumns, queryString, visibleRows,
} from "../../transcriber/adapters/inbound/web/js/lib/offers.js";
import { crumbsFor, sectionOf } from "../../transcriber/adapters/inbound/web/js/lib/shell.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const SLOTS = [
  { id: "s1", label: "Tipos de café", level: "minimo", weight: 4, questions: ["¿Qué cafés conoces?"], points: [] },
  { id: "s2", label: "Atención en barra", level: "minimo", weight: 4, questions: ["¿Cómo atiendes la barra?"], points: [] },
  { id: "s3", label: "Inglés", level: "deseable", weight: 2, questions: ["¿Qué nivel tienes?"], points: [] },
];

function point(id, state, value, extra = {}) {
  return { id, state, value, basis: "labels", score: null, label: null, treated: 0, concepts: 0, quote: "", reason: "", ...extra };
}

/** A board as GET …/board returns it, with invented people. */
function board(state = "open") {
  const person = (i, name, latest, others = []) => ({
    person: `file:${name}`, name, candidate_id: name.toLowerCase().replace(/\s+/g, "-"), position: i,
    latest: { kind: "ai", owned: true, accuracy: {}, to_check: 0, missing_required: [], not_asked: [], plan_changed: false,
      less_reliable: false, figure_basis: "scores", candidate: name, ...latest },
    others,
  });
  return {
    schema: "interview.board.v1",
    plan: { id: "cafe", title: "Barista", revision: 1, minutes: 20, summary: "Cafetería inventada.", state, slots: SLOTS },
    required: ["s1", "s2"], desirable: ["s3"],
    people: [
      person(1, "Ana Ruiz", { session_id: "vb-1", figure: 82.5, finished_at: "2026-11-20T10:00:00",
        points: [point("s1", "solid", 90, { score: 90, basis: "score", quote: "uso arábica y robusta", reason: "Correcto y concreto." }),
          point("s2", "correct_basic", 75, { score: 75, basis: "score" }), point("s3", "answered", 60, { score: 60 })],
        accuracy: { solid: 2, correct_basic: 1 }, to_check: 0 },
      [{ session_id: "r-old", kind: "live", finished_at: "2026-11-01T10:00:00", figure: 40 }]),
      person(2, "Ángel Soto", { session_id: "r-2", kind: "live", figure: 55, figure_basis: "labels", finished_at: "2026-11-24T10:00:00",
        points: [point("s1", "unclear", 40), point("s2", "correct_basic", 70, { treated: 1, concepts: 2 }), point("s3", "not_treated", null)],
        accuracy: { correct_basic: 1, not_treated: 1 }, to_check: 2, less_reliable: true }),
      person(3, "Bea Gil", { session_id: "r-3", kind: "live", figure: 55, figure_basis: "labels", finished_at: "2026-11-25T10:00:00",
        points: [point("s1", "wrong", 0), point("s2", "solid", 100), point("s3", "not_asked", null)],
        missing_required: ["s1"], not_asked: ["s3"], plan_changed: true, accuracy: { wrong: 1, solid: 1 }, to_check: 1, owned: false }),
    ],
    interviews: 4, truncated: false,
  };
}

// ---- the address and the order ---------------------------------------------

test("the table's state goes to the address and back; defaults are left out", () => {
  assert.deepEqual(parseQuery(""), { sort: "fit", dir: "desc", q: "", kind: "all", point: "", state: "any", cols: [] });
  const state = { sort: "s1", dir: "asc", q: "ana", kind: "live", point: "s2", state: "solid_basic", cols: ["s3"] };
  assert.equal(queryString(state), "sort=s1&dir=asc&q=ana&kind=live&point=s2&state=solid_basic&cols=s3");
  assert.deepEqual(parseQuery(queryString(state)), state);
  assert.equal(queryString(parseQuery("")), "");
  assert.equal(queryString({ sort: "name", dir: "asc" }), "sort=name", "names start A to Z");
  assert.deepEqual(parseQuery("sort=../x&dir=up&kind=robot&state=apto&point=zz&cols=s1,../x"),
    { sort: "fit", dir: "desc", q: "", kind: "all", point: "", state: "any", cols: ["s1"] });
  assert.equal(offerHref("caf é", { ...state, q: "" }), "#/offers/caf%20%C3%A9?sort=s1&dir=asc&kind=live&point=s2&state=solid_basic&cols=s3");
  assert.equal(offerHref("cafe"), "#/offers/cafe");
  assert.equal(personHref("cafe", "ana ruiz"), "#/offers/cafe/person/ana%20ruiz");
});

test("a header click sorts by its column, a second click flips it; aria-sort says which", () => {
  let state = parseQuery("");
  assert.equal(ariaSort(state, "fit"), "descending");
  assert.equal(ariaSort(state, "name"), "none");
  state = nextSort(state, "name");
  assert.deepEqual([state.sort, state.dir, ariaSort(state, "name")], ["name", "asc", "ascending"]);
  state = nextSort(state, "name");
  assert.equal(ariaSort(state, "name"), "descending");
  state = nextSort(state, "s1");
  assert.deepEqual([state.sort, state.dir], ["s1", "desc"]);
});

test("the default order is the server's; other columns sort with missing values last", () => {
  const rows = boardRows(board());
  const names = (state) => visibleRows(rows, { ...parseQuery(""), ...state }).map((r) => r.name);
  assert.deepEqual(names({}), ["Ana Ruiz", "Ángel Soto", "Bea Gil"]);
  assert.deepEqual(names({ dir: "asc" }), ["Bea Gil", "Ángel Soto", "Ana Ruiz"]);
  assert.deepEqual(names({ sort: "s1" }), ["Ana Ruiz", "Ángel Soto", "Bea Gil"]);
  assert.deepEqual(names({ sort: "s2", dir: "asc" }), ["Ángel Soto", "Ana Ruiz", "Bea Gil"]);
  assert.deepEqual(names({ sort: "s3" }), ["Ana Ruiz", "Ángel Soto", "Bea Gil"], "no value: last");
  assert.deepEqual(names({ sort: "s3", dir: "asc" }), ["Ana Ruiz", "Ángel Soto", "Bea Gil"], "no value: last both ways");
  assert.deepEqual(names({ sort: "name", dir: "asc" }), ["Ana Ruiz", "Ángel Soto", "Bea Gil"]);
  assert.deepEqual(names({ sort: "date" }), ["Bea Gil", "Ángel Soto", "Ana Ruiz"]);
  assert.deepEqual(names({ sort: "to_check" }), ["Ángel Soto", "Bea Gil", "Ana Ruiz"]);
});

test("search, kind and a required point's state filter; nobody is hidden without a filter", () => {
  const rows = boardRows(board());
  const names = (state) => visibleRows(rows, { ...parseQuery(""), ...state }).map((r) => r.name);
  assert.equal(names({}).length, 3);
  assert.equal(filtering(parseQuery("")), false);
  assert.deepEqual(names({ q: "angel" }), ["Ángel Soto"], "accents do not matter");
  assert.deepEqual(names({ kind: "live" }), ["Ángel Soto", "Bea Gil"]);
  assert.deepEqual(names({ kind: "ai" }), ["Ana Ruiz"]);
  assert.deepEqual(names({ point: "s1", state: "solid_basic" }), ["Ana Ruiz"]);
  assert.deepEqual(names({ point: "s1", state: "covered" }), ["Ana Ruiz", "Ángel Soto"]);
  assert.deepEqual(names({ point: "s1", state: "doubtful" }), ["Ángel Soto"]);
  assert.deepEqual(names({ point: "s1", state: "wrong_off" }), ["Bea Gil"]);
  assert.deepEqual(names({ point: "s3", state: "not_covered" }), ["Ángel Soto", "Bea Gil"]);
  assert.deepEqual(names({ point: "s1", state: "any" }), ["Ana Ruiz", "Ángel Soto", "Bea Gil"]);
  assert.ok(filtering({ q: "a" }) && filtering({ kind: "ai" }) && filtering({ point: "s1", state: "covered" }));
  assert.ok(!filtering({ point: "s1", state: "any" }));
  assert.ok(Object.keys(POINT_FILTERS).every((k) => es[`of.filter.${k}`] && en[`of.filter.${k}`]));
});

test("the point columns are the required ones plus the desirable ones chosen", () => {
  const b = board();
  assert.deepEqual(pointColumns(b).map((c) => [c.id, c.required]), [["s1", true], ["s2", true]]);
  assert.deepEqual(pointColumns(b, ["s3"]).map((c) => c.id), ["s1", "s2", "s3"]);
  assert.deepEqual(optionalPoints(b).map((s) => s.id), ["s3"]);
});

test("the list shows open offers first and closed ones only on request", () => {
  const plans = [
    { id: "a", state: "closed", updated_at: "2026-11-25" },
    { id: "b", state: "open", updated_at: "2026-11-01", last_interview: "2026-11-20" },
    { id: "c", state: "open", updated_at: "2026-11-10" },
  ];
  assert.deepEqual(offerList(plans).map((p) => p.id), ["b", "c"]);
  assert.deepEqual(offerList(plans, { closed: true }).map((p) => p.id), ["b", "c", "a"]);
  assert.equal(closedCount(plans), 1);
});

test("the offers have their own navigation entry and breadcrumbs", () => {
  const t = (key) => key;
  assert.equal(sectionOf("#/offers/cafe?sort=s1"), "#/offers");
  assert.deepEqual(crumbsFor("#/offers?closed=1", t), [{ label: "nav.offers", href: null }]);
  assert.deepEqual(crumbsFor("#/offers/cafe?sort=s1&dir=asc", t), [{ label: "nav.offers", href: "#/offers" }, { label: "cafe", href: null }]);
  assert.deepEqual(crumbsFor("#/offers/cafe/person/ana-ruiz", t),
    [{ label: "nav.offers", href: "#/offers" }, { label: "cafe", href: "#/offers/cafe" }, { label: "ana-ruiz", href: null }]);
});

// ---- the wording -------------------------------------------------------------

test("the texts say it is a draft for a person and never give a verdict", () => {
  assert.equal(es["of.title"], "Ofertas");
  assert.equal(es["of.how_title"], "Cómo se ordena");
  assert.match(es["of.draft_note"], /para que lo revise una persona antes de cualquier decisión/);
  assert.match(es["of.draft_note"], /no deja fuera a nadie/);
  assert.match(es["of.draft_note"], /no se muestra a las personas candidatas/);
  assert.match(en["of.draft_note"], /for a person to review before any decision/);
  assert.match(es["of.legal"], /alto riesgo/);
  assert.match(es["of.legal"], /#151/);
  const keys = Object.keys(es).filter((k) => k.startsWith("of.") || k === "nav.offers" || k === "iv.compare_offer" || k === "iv.all_offers");
  assert.ok(keys.length >= 90, `${keys.length} keys`);
  for (const key of keys) {
    assert.ok(en[key], `${key} in English`);
    for (const text of [es[key], en[key]]) {
      assert.ok(!/\bapt[oa]s?\b|recomendad|descartad|rechazad|contrat|ranking|\bmejor candidat|suitable\b|\bhire|\breject|recommend|\bbest candidate/i.test(text),
        `${key}: ${text}`);
    }
  }
  const enKeys = Object.keys(en).filter((k) => k.startsWith("of."));
  assert.deepEqual(enKeys.sort(), keys.filter((k) => k.startsWith("of.")).sort());
});

// ---- the page in a stand-in DOM ------------------------------------------------

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
    this.dataset = {};
    this.classList = { add: () => {} };
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.open = false;
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
  }
  get firstChild() { return this.children[0] ?? null; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelector() { return null; }
  remove() {}
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return [this.textContent, ...this.children.map((c) => (c.text ? c.text() : c.textContent))].join(""); }
  fire(type) { for (const fn of this.listeners[type] ?? []) fn({ target: this }); }
}

async function withDom(language, fn) {
  const saved = { document: globalThis.document, Node: globalThis.Node };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
    getElementById: () => null,
    body: new FakeElement("body"),
  };
  globalThis.Node = FakeNode;
  try {
    const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
    setLanguage(language);
    return await fn(await import("../../transcriber/adapters/inbound/web/js/views/offers.js"));
  } finally {
    Object.assign(globalThis, saved);
  }
}

const byTag = (node, tag) => node.all().filter((n) => n.tagName === tag);
const bodyRows = (node) => byTag(node, "TBODY")[0].children;

test("the table is a real table: caption, column headers, row headers, aria-sort", async () => {
  await withDom("es", async ({ peopleSection }) => {
    const section = peopleSection({}, board(), parseQuery(""));
    const table = byTag(section, "TABLE")[0];
    assert.ok(table, "a <table>");
    assert.ok(byTag(table, "CAPTION")[0].text().includes("Barista"));
    const heads = byTag(table, "TH").filter((th) => th.attributes.scope === "col");
    assert.ok(heads.length >= 8);
    assert.deepEqual(heads.filter((th) => th.attributes["aria-sort"] === "descending").map((th) => th.attributes["data-key"]), ["fit"]);
    assert.ok(heads.filter((th) => th.attributes["data-key"]).every((th) => byTag(th, "BUTTON").length === 1), "sorting is a button");
    const rowHeads = byTag(table, "TH").filter((th) => th.attributes.scope === "row");
    assert.deepEqual(rowHeads.map((th) => byTag(th, "SPAN")[0].text()), ["Ana Ruiz", "Ángel Soto", "Bea Gil"]);
    const text = section.text();
    assert.ok(text.includes(es["of.how_title"]) && text.includes(es["of.draft_note"]));
    assert.ok(text.includes(es["of.how.ties"]));
    assert.ok(text.includes("Tipos de café") && !text.includes("Inglés · "), "desirable columns are hidden until chosen");
  });
});

test("every state is said in words; gaps, word-rule reports and edited plans are marked", async () => {
  await withDom("es", async ({ peopleSection }) => {
    const section = peopleSection({}, board(), parseQuery(""));
    const rows = bodyRows(byTag(section, "TABLE")[0]);
    const [ana, angel, bea] = rows.map((r) => r.text());
    assert.ok(ana.includes(es["iv.judge.solid"]) && ana.includes("puntuación 90") && ana.includes(es["of.covers_all"]));
    assert.ok(ana.includes(es["of.basis.scores"]) && ana.includes("82,5"));
    assert.ok(ana.includes("1 entrevistas más") || ana.includes(es["of.others"].replace("{n}", "1")));
    assert.ok(angel.includes(es["iv.judge.unclear"]) && angel.includes("1 de 2 conceptos") && angel.includes(es["of.less_reliable"]));
    assert.ok(angel.includes(es["of.basis.labels"]));
    assert.ok(bea.includes(es["iv.judge.wrong"]) && bea.includes("Imprescindibles sin cubrir: 1") && bea.includes(es["of.plan_changed"]));
    assert.ok(bea.includes(es["iv.no_file"]));
    assert.ok(rows[2].className.includes("is-missing"));
    const evidence = byTag(rows[0], "DETAILS").find((d) => d.text().includes(es["of.evidence"]));
    assert.ok(evidence.text().includes("uso arábica y robusta") && evidence.text().includes("Correcto y concreto."));
    assert.ok(byTag(evidence, "A")[0].href === "#/interview-report/cafe/vb-1");
    const files = byTag(section, "A").filter((a) => a.href.includes("/person/")).map((a) => a.href);
    assert.deepEqual(files, ["#/offers/cafe/person/ana-ruiz", "#/offers/cafe/person/ángel-soto"].map((h) => h.replace("ángel", "%C3%A1ngel")));
  });
});

test("sorting, searching and filtering redraw the table, announce it and keep the state in the address", async () => {
  await withDom("es", async ({ peopleSection }) => {
    const states = [];
    const section = peopleSection({}, board(), parseQuery(""), { onState: (s) => states.push(queryString(s)) });
    const status = section.all().find((n) => n.attributes?.role === "status");
    assert.match(status.textContent, /Se ven 3 de 3 personas\. Ordenado por «Cifra \(borrador\)», de mayor a menor\./);
    const sortBy = (key) => byTag(section, "TH").find((th) => th.attributes["data-key"] === key);
    byTag(sortBy("name"), "BUTTON")[0].fire("click");
    assert.equal(sortBy("name").attributes["aria-sort"], "ascending");
    assert.equal(sortBy("fit").attributes["aria-sort"], "none");
    assert.match(status.textContent, /Ordenado por «Persona», de menor a mayor/);
    byTag(sortBy("name"), "BUTTON")[0].fire("click");
    assert.equal(sortBy("name").attributes["aria-sort"], "descending");
    assert.deepEqual(byTag(section, "TH").filter((th) => th.attributes.scope === "row").map((th) => byTag(th, "SPAN")[0].text()),
      ["Bea Gil", "Ángel Soto", "Ana Ruiz"]);
    const search = section.all().find((n) => n.id === "of-search");
    search.value = "bea";
    search.fire("input");
    assert.match(status.textContent, /Se ven 1 de 3 personas/);
    const kind = section.all().find((n) => n.id === "of-kind");
    search.value = "";
    search.fire("input");
    kind.value = "live";
    kind.fire("change");
    assert.equal(bodyRows(byTag(section, "TABLE")[0]).length, 2);
    const point = section.all().find((n) => n.id === "of-point");
    const pointState = section.all().find((n) => n.id === "of-point-state");
    assert.ok(pointState.disabled, "the state needs a point first");
    point.value = "s1";
    point.fire("change");
    assert.ok(!pointState.disabled);
    pointState.value = "wrong_off";
    pointState.fire("change");
    assert.equal(bodyRows(byTag(section, "TABLE")[0]).length, 1);
    assert.equal(states.at(-1), "sort=name&dir=desc&kind=live&point=s1&state=wrong_off");
    const clear = byTag(section, "BUTTON").find((b) => b.text() === es["of.clear"]);
    assert.ok(!clear.hidden);
    clear.fire("click");
    assert.equal(bodyRows(byTag(section, "TABLE")[0]).length, 3);
    assert.equal(states.at(-1), "sort=name&dir=desc");
    pointState.value = "covered";
    point.value = "s9";
    const column = section.all().find((n) => n.id === "of-col-s3");
    column.checked = true;
    column.fire("change");
    assert.ok(byTag(section, "TH").some((th) => th.attributes["data-key"] === "s3"), "the chosen column shows");
    assert.match(states.at(-1), /cols=s3/);
  });
});

test("a filter that matches nobody says so; an offer without interviews explains it", async () => {
  await withDom("es", async ({ peopleSection }) => {
    const none = peopleSection({}, board(), parseQuery("q=zzz"));
    assert.ok(none.text().includes(es["of.no_match"]));
    const empty = peopleSection({}, { ...board(), people: [] }, parseQuery(""));
    assert.ok(empty.text().includes(es["of.no_people_title"]));
    assert.equal(byTag(empty, "TABLE").length, 0);
  });
});

test("the list of offers: cards, the closed switch and the first-time explanation", async () => {
  await withDom("es", async ({ offersPage }) => {
    const plans = [
      { id: "cafe", title: "Barista", state: "open", slots: 3, minimums: 2, desirables: 1, interviews: 4, interviews_ai: 2,
        interviews_live: 2, last_interview: "2026-11-25T10:00:00", created_at: "2026-11-02T10:00:00" },
      { id: "tienda", title: "Dependienta", state: "closed", slots: 2, minimums: 1, desirables: 1, interviews: 0 },
    ];
    let asked = null;
    const page = offersPage(plans, { onClosed: (on) => { asked = on; } });
    const text = page.text();
    assert.ok(text.includes("Barista") && !text.includes("Dependienta"));
    assert.ok(text.includes("3 puntos: 2 imprescindibles, 1 deseables") && text.includes("4 entrevistas: 2 con IA, 2 en directo"));
    assert.ok(text.includes(es["of.state.open"]) && text.includes(es["of.show_closed"].replace("{n}", "1")));
    assert.equal(byTag(page, "A").find((a) => a.text() === "Barista").href, "#/offers/cafe");
    assert.equal(byTag(page, "A").find((a) => a.text().includes(es["of.new"])).href, "#/voicebot/interview");
    const toggle = page.all().find((n) => n.id === "of-closed");
    toggle.checked = true;
    toggle.fire("change");
    assert.equal(asked, true);
    assert.ok(page.text().includes("Dependienta") && page.text().includes(es["of.state.closed"]));
    const first = offersPage([]).text();
    assert.ok(first.includes(es["of.empty_title"]) && first.includes(es["of.empty_body"]));
  });
});

test("the offer page edits the questions with the plan editor and saves a new revision", async () => {
  await withDom("es", async ({ offerPage }) => {
    const saved = [];
    let reloaded = 0;
    const ctx = { api: { savePlan: async (id, plan) => { saved.push([id, plan]); return { ...plan, revision: 2 }; } } };
    const parts = offerPage(ctx, board(), parseQuery(""), () => { reloaded += 1; });
    const page = new FakeElement("div");
    page.append(...parts.filter(Boolean));
    assert.ok(page.text().includes("Barista") && page.text().includes(es["of.state.open"]));
    assert.ok(page.text().includes(es["of.start_ai"]));
    const edit = byTag(page, "BUTTON").find((b) => b.text().includes(es["of.edit"]));
    assert.equal(edit.attributes["aria-expanded"], "false");
    edit.fire("click");
    assert.equal(edit.attributes["aria-expanded"], "true");
    assert.ok(page.text().includes(es["of.edit_hint"]));
    const save = byTag(page, "BUTTON").find((b) => b.text() === es["of.save"]);
    assert.ok(save.disabled, "nothing to save yet");
    const title = page.all().find((n) => n.id === "iv-plan-title");
    title.value = "Barista de mañana";
    title.fire("change");
    const save2 = byTag(page, "BUTTON").find((b) => b.text() === es["of.save"]);
    assert.ok(!save2.disabled);
    save2.fire("click");
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(saved[0][0], "cafe");
    assert.equal(saved[0][1].title, "Barista de mañana");
    assert.equal(saved[0][1].revision, 1, "the revision it was based on");
    assert.equal(reloaded, 1);
    const closed = new FakeElement("div");
    closed.append(...offerPage(ctx, board("closed"), parseQuery("")).filter(Boolean));
    assert.ok(closed.text().includes(es["iv.closed_notice"]) && closed.text().includes(es["iv.reopen_position"]));
    assert.ok(!closed.text().includes(es["of.start_ai"]), "a closed position starts no interview");
  });
});

test("the English page reads in English", async () => {
  await withDom("en", async ({ peopleSection }) => {
    const text = peopleSection({}, board(), parseQuery("")).text();
    assert.ok(text.includes(en["of.how_title"]) && text.includes(en["of.col.fit"]) && text.includes("82.5"));
  });
});

test("the voice-bot page's saved plans lead to «Ofertas» instead of a comparison of its own", async () => {
  await withDom("es", async () => {
    const { planSetup } = await import("../../transcriber/adapters/inbound/web/js/views/interview.js");
    const opened = [];
    const plan = { ...board().plan, planner: { status: "ok" } };
    const api = {
      interviewPlans: async () => ({ plans: [{ id: "cafe", title: "Barista", slots: 3, interviews: 4, state: "open" }] }),
      interviewPlan: async (id) => { opened.push(id); return plan; },
    };
    const root = planSetup({ api, fail: () => {} }, { onReady: () => {}, initial: "cafe" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(opened, ["cafe"], "opened from the offer page with its plan");
    const links = byTag(root, "A").map((a) => a.href);
    assert.ok(links.includes("#/offers"), "the saved plans point to «Ofertas»");
    assert.ok(links.includes("#/offers/cafe"), "the review links to the offer's people");
    assert.ok(!root.text().includes("Comparar candidatos"));
  });
});
