// «Lo que hay que decir» lists (#294): the pure part of the editor and of the
// live start form's choice, the dictionaries (both languages, calm wording, no
// verdict words, the heuristic and legal notes), and the pages in a stand-in
// DOM with the language pinned: real buttons with names that say which item
// they act on, one status line that says a thing once, the save and the review
// going to the API with the editor's body. Invented lists only.
import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_LIMITS, LAST_LIST_KEY, MOMENTS, draftOf, generalId, isDirty, isGeneral, listErrors, listForSave, listUsedText,
  listsOf, moveItem, newItem, offersUsing, rememberList, rememberedList, warningLines,
} from "../../transcriber/adapters/inbound/web/js/lib/mandatory.js";
import { ADMIN_PARTS, crumbsFor } from "../../transcriber/adapters/inbound/web/js/lib/shell.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const SAVED = {
  id: "verano", name: "Campaña Verano", kind: "support", mode: "add", version: 2, general: false,
  items: [
    { id: "precio", text: "Precio de la tarifa Verano", phase: "during",
      must_say: ["la tarifa Verano cuesta veinte euros al mes", "son veinte euros al mes"] },
    { id: "permanencia", text: "Sin permanencia", phase: "closing", must_say: ["no tiene permanencia ninguna"] },
  ],
};
const OVERVIEW = {
  lists: [
    { id: "general.support", general: true, kind: "support", items: 4, version: 1, domain: "support.demo" },
    { id: "general.interview", general: true, kind: "interview", items: 3, version: 1 },
    { id: "verano", name: "Campaña Verano", kind: "support", mode: "add", version: 2, items: 2, general: false },
    { id: "panaderia", name: "Panadería", kind: "interview", mode: "replace", version: 1, items: 3, general: false },
  ],
  offers: { "plan-1": "panaderia", "plan-2": "panaderia" },
  moments: MOMENTS,
  limits: DEFAULT_LIMITS,
};

/** A memory stand-in for localStorage. */
function memoryStorage(failing = false) {
  const kept = new Map();
  return {
    getItem: (k) => (failing ? (() => { throw new Error("no"); })() : kept.get(k) ?? null),
    setItem: (k, v) => { if (failing) throw new Error("no"); kept.set(k, v); },
    kept,
  };
}

const tr = (key, params = {}) => Object.entries(params).reduce((s, [k, v]) => s.replace(`{${k}}`, v), es[key] ?? key);

// ---- the pure part -------------------------------------------------------------

test("a draft goes to the server with its texts on one line and its ids kept", () => {
  const draft = draftOf(SAVED);
  assert.equal(draft.mode, "add");
  draft.items[0].text = "  Precio   de la tarifa ";
  draft.items[0].must_say.push("   ");
  draft.items.push({ ...newItem(), text: "Nuevo", must_say: ["una frase nueva de ejemplo"] });
  const body = listForSave(draft);
  assert.equal(body.version, 2);
  assert.equal(body.items[0].text, "Precio de la tarifa");
  assert.equal(body.items[0].must_say.length, 2);
  assert.equal(body.items[0].id, "precio");
  assert.ok(!("id" in body.items[2]), "a new item gets its id from the server");
  assert.ok(!("version" in listForSave(draftOf(null))), "a new list has no version");
  assert.equal(draftOf({ ...SAVED, mode: "nope", items: [{ text: "x", phase: "resolution" }] }).items[0].phase, null);
});

test("what stops a save is said per item", () => {
  const draft = draftOf(null);
  assert.deepEqual(listErrors(draft).map((e) => e.key), ["name", "item_text", "no_example"]);
  draft.name = "x".repeat(DEFAULT_LIMITS.name + 1);
  draft.items = [{ ...newItem(), text: "y".repeat(301), must_say: ["a", "A", "b".repeat(301)] }];
  const keys = listErrors(draft).map((e) => e.key);
  assert.deepEqual(keys, ["name_long", "item_long", "example_long", "example_twice"]);
  draft.items = [];
  assert.ok(listErrors(draft).some((e) => e.key === "no_items"));
  draft.items = Array.from({ length: 3 }, () => ({ ...newItem(), text: "t", must_say: ["1", "2", "3"] }));
  const tight = listErrors(draft, { ...DEFAULT_LIMITS, items: 2, examples: 2 }).map((e) => e.key);
  assert.ok(tight.includes("too_many") && tight.includes("examples_many"));
  assert.deepEqual(listErrors(draftOf(SAVED)), []);
});

test("dirty, move, the lists of a kind and the offers using a list", () => {
  const draft = draftOf(SAVED);
  assert.equal(isDirty(draft, SAVED), false);
  draft.items = moveItem(draft.items, 1, 0);
  assert.equal(isDirty(draft, SAVED), true);
  assert.deepEqual(draft.items.map((i) => i.id), ["permanencia", "precio"]);
  assert.deepEqual(moveItem(["a", "b"], 5, 0), ["a", "b"]);
  assert.deepEqual(moveItem(["a", "b", "c"], 0, 9), ["b", "c", "a"]);
  const support = listsOf(OVERVIEW, "support");
  assert.equal(support.general.id, "general.support");
  assert.deepEqual(support.own.map((l) => l.id), ["verano"]);
  assert.deepEqual(offersUsing(OVERVIEW, "panaderia"), ["plan-1", "plan-2"]);
  assert.equal(generalId("interview"), "general.interview");
  assert.ok(isGeneral("general.support") && !isGeneral("verano"));
});

test("the last list chosen is remembered while it is still offered", () => {
  const storage = memoryStorage();
  const choices = [{ id: "general.support" }, { id: "verano" }];
  assert.equal(rememberedList("support", choices, storage), "general.support");
  assert.equal(rememberList("support", "verano", storage), true);
  assert.equal(rememberList("interview", "panaderia", storage), true);
  assert.equal(rememberedList("support", choices, storage), "verano");
  assert.deepEqual(JSON.parse(storage.kept.get(LAST_LIST_KEY)), { support: "verano", interview: "panaderia" });
  assert.equal(rememberedList("support", [{ id: "general.support" }], storage), "general.support", "a deleted list is not chosen");
  const broken = memoryStorage(true);
  assert.equal(rememberList("support", "verano", broken), false);
  assert.equal(rememberedList("support", choices, broken), "general.support");
  storage.kept.set(LAST_LIST_KEY, "{not json");
  assert.equal(rememberedList("support", choices, storage), "general.support");
});

test("a session names the list and version it used", () => {
  assert.equal(listUsedText(null, tr), null);
  assert.equal(listUsedText({ id: "general.support", general: true, version: 1 }, tr), "Lista general · versión 1");
  assert.equal(listUsedText({ id: "verano", name: "Campaña Verano", version: 3 }, tr), "Lista «Campaña Verano» · versión 3");
  const lines = warningLines({
    close: [{ item: "precio", other: "permanencia", example: "son veinte euros" }],
    short: [{ item: "permanencia", example: "sin permanencia", words: 2 }],
  }, (id) => id.toUpperCase());
  assert.deepEqual(lines.map((l) => [l.key, l.params.item]), [["close", "PRECIO"], ["short", "PERMANENCIA"]]);
  assert.deepEqual(warningLines(null), []);
});

// ---- the dictionaries -------------------------------------------------------------

test("every text is in both languages, calm, with no verdict wording", () => {
  const keys = Object.keys(es).filter((k) => k.startsWith("mi.") || k.startsWith("live.list") || k === "live.plan_list"
    || k === "live.summary_list" || k === "nav.mandatory" || k === "admin.mandatory");
  assert.ok(keys.length >= 90, `${keys.length} keys`);
  for (const key of keys) {
    assert.ok(en[key], `${key} in English`);
    for (const text of [es[key], en[key]]) {
      assert.ok(!/\bapt[oa]s?\b|recomendad|descartad|rechazad|contrat|ranking|\bmejor candidat|suitable\b|\bhire|\breject|recommend|!/i.test(text),
        `${key}: ${text}`);
      const params = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
      assert.deepEqual(params(en[key]), params(es[key]), `${key}: the same placeholders`);
    }
  }
  assert.match(es["mi.legal"], /no son asesoramiento legal/);
  assert.match(es["mi.legal"], /textos legales/);
  assert.match(es["mi.warn_heuristic"], /no una medición/);
  assert.match(es["mi.mode_hint.add"], /protección de datos/);
  assert.match(es["mi.how"], /dos o tres formas distintas/);
  assert.match(en["mi.how"], /two or three different ways/);
});

test("the page is in the Administración area with its crumbs", () => {
  assert.ok(ADMIN_PARTS.has("mandatory"));
  const t = (k) => `<${k}>`;
  assert.deepEqual(crumbsFor("#/mandatory", t).map((c) => c.label), ["<nav.admin>", "<nav.mandatory>"]);
  assert.deepEqual(crumbsFor("#/mandatory/verano", t).map((c) => c.label), ["<nav.admin>", "<nav.mandatory>", "verano"]);
});

// ---- the pages in a stand-in DOM -------------------------------------------------

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
    this.classList = { add: () => {}, toggle: () => {} };
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title", "name"]) this[key] = "";
    this.open = false;
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.selected = false;
    this.maxLength = 0;
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
  async fire(type) { for (const fn of this.listeners[type] ?? []) await fn({ target: this }); }
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
    return await fn(await import("../../transcriber/adapters/inbound/web/js/views/mandatory.js"));
  } finally {
    Object.assign(globalThis, saved);
  }
}

const byTag = (node, tag) => node.all().filter((n) => n.tagName === tag);
const buttons = (node) => byTag(node, "BUTTON");
const named = (node, words) => buttons(node).find((b) => (b.attributes["aria-label"] ?? b.text()).includes(words));
const statusOf = (node) => node.all().find((n) => n.attributes?.role === "status");
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test("the lists page shows both kinds, the general lists read-only and the legal note", async () => {
  await withDom("es", async ({ listsPage }) => {
    const page = listsPage({ api: {} }, OVERVIEW);
    const text = page.text();
    for (const key of ["mi.kind.support", "mi.kind.interview", "mi.read_only", "mi.legal", "mi.how"]) assert.ok(text.includes(es[key]), key);
    assert.ok(text.includes("Campaña Verano") && text.includes("Panadería"));
    assert.ok(text.includes("ofertas que la usan: 2"), "an interview list says how many offers use it");
    assert.ok(text.includes(es["mi.mode.replace"]) && text.includes(es["mi.mode.add"]));
    const links = byTag(page, "A").map((a) => a.href);
    assert.ok(links.includes("#/mandatory/new-support") && links.includes("#/mandatory/new-interview"));
    assert.ok(links.includes("#/mandatory/general.support"));
    // The general list has no delete; each own list has one.
    assert.equal(buttons(page).filter((b) => b.text().includes(es["mi.delete"])).length, 2);
  });
  await withDom("en", async ({ listsPage }) => {
    const text = listsPage({ api: {} }, OVERVIEW).text();
    assert.ok(text.includes(en["mi.kind.support"]) && text.includes(en["mi.general_name"]));
  });
});

test("the editor is buttons and labelled fields; moving says once where the item went", async () => {
  await withDom("es", async ({ listEditor }) => {
    const box = listEditor({ api: {} }, SAVED);
    const status = statusOf(box);
    assert.equal(status.attributes["aria-live"], "polite");
    const up = named(box, "Subir «Sin permanencia»");
    const down = named(box, "Bajar «Precio de la tarifa Verano»");
    assert.ok(up && down, "move buttons name their item");
    assert.ok(named(box, "Subir «Precio de la tarifa Verano»").disabled, "the first item cannot go up");
    assert.ok(named(box, "Quitar la forma 1 de «Sin permanencia»").disabled, "the only example cannot be removed");
    const save = buttons(box).find((b) => b.text() === es["mi.save"]);
    assert.equal(save.disabled, true, "nothing to save yet");
    await up.fire("click");
    assert.equal(status.textContent, "«Sin permanencia» ahora es el punto 1");
    assert.equal(save.disabled, false);
    const labels = byTag(box, "LABEL").map((l) => l.htmlFor);
    assert.ok(labels.includes("mi-name") && labels.includes("mi-item-0-text") && labels.includes("mi-item-0-phase"));
    const legends = byTag(box, "LEGEND").map((l) => l.text());
    assert.ok(legends.includes("Punto 1") && legends.includes("Punto 2") && legends.includes(es["mi.mode_title"]));
  });
});

test("typing keeps the draft, a save sends it and shows the warnings, a review too", async () => {
  const calls = [];
  const api = {
    saveMandatoryList: async (id, body) => {
      calls.push(["save", id, body]);
      return { ...SAVED, ...body, version: 3, warnings: { checked: true, close: [], short: [{ item: "permanencia", example: "no", words: 1 }] } };
    },
    reviewMandatoryList: async (body) => {
      calls.push(["review", body]);
      return { checked: false, close: [{ item: "precio", other: "permanencia", example: "son veinte euros al mes" }], short: [] };
    },
  };
  await withDom("es", async ({ listEditor }) => {
    const saved = [];
    const box = listEditor({ api }, SAVED, { onSaved: (s) => saved.push(s) });
    const name = byTag(box, "INPUT").find((i) => i.id === "mi-name");
    name.value = "Campaña Verano 2026";
    await name.fire("input");
    const save = buttons(box).find((b) => b.text() === es["mi.save"]);
    assert.equal(save.disabled, false);
    assert.equal(calls[0][0], "review", "a saved list shows its warnings when it opens");
    await save.fire("click");
    await settle();
    assert.equal(calls[1][0], "save");
    assert.equal(calls[1][1], "verano");
    assert.equal(calls[1][2].name, "Campaña Verano 2026");
    assert.equal(calls[1][2].version, 2, "the edit names the version it was based on");
    assert.equal(saved[0].version, 3);
    const text = box.text();
    assert.ok(text.includes("Avisos (1)") && text.includes(es["mi.warn_heuristic"]));
    assert.equal(statusOf(box).textContent, "«Campaña Verano 2026» guardada (versión 3)");
    const review = buttons(box).find((b) => b.text() === es["mi.review"]);
    await review.fire("click");
    await settle();
    assert.equal(calls[2][0], "review");
    assert.ok(box.text().includes(es["mi.warn_unchecked"]));
    assert.ok(box.text().includes("Una forma de decir «Precio de la tarifa Verano» se parece mucho a una de «Sin permanencia»"));
  });
});

test("a conflict on save says someone saved first", async () => {
  const api = { saveMandatoryList: async () => { throw Object.assign(new Error("x"), { code: "conflict" }); } };
  await withDom("es", async ({ listEditor }) => {
    const box = listEditor({ api }, SAVED);
    const name = byTag(box, "INPUT").find((i) => i.id === "mi-name");
    name.value = "Otro nombre";
    await name.fire("input");
    await buttons(box).find((b) => b.text() === es["mi.save"]).fire("click");
    await settle();
    assert.equal(statusOf(box).textContent, es["mi.err.conflict"]);
  });
});

test("a new list starts with one empty item and says what is missing", async () => {
  await withDom("es", async ({ listEditor }) => {
    const box = listEditor({ api: {} }, null, { kind: "interview" });
    const text = box.text();
    assert.ok(text.includes(es["mi.err.name"]) && text.includes("Al punto 1 le falta el texto."));
    const add = buttons(box).find((b) => b.text().includes(es["mi.item_add"]));
    await add.fire("click");
    assert.ok(byTag(box, "LEGEND").some((l) => l.text() === "Punto 2"));
    const more = buttons(box).filter((b) => b.text().includes(es["mi.example_add"]))[0];
    await more.fire("click");
    assert.ok(named(box, "Forma 2 de decir") === undefined, "examples are inputs, not buttons");
    assert.ok(byTag(box, "INPUT").some((i) => i.attributes["aria-label"] === "Forma 2 de decir «Punto 1»"));
  });
});

test("the offer card needs the API, and lists the interview lists", async () => {
  await withDom("es", async ({ offerListCard }) => {
    assert.equal(offerListCard({}, { id: "plan-1", title: "Panadería" }), null);
    const set = [];
    const api = {
      mandatoryLists: async () => OVERVIEW,
      offerMandatory: async () => ({ plan: "plan-1", list: "panaderia", name: "Panadería" }),
      setOfferMandatory: async (plan, list) => { set.push([plan, list]); return { plan, list, name: null }; },
    };
    const card = offerListCard({ api }, { id: "plan-1", title: "Panadería" });
    await settle();
    const select = byTag(card, "SELECT")[0];
    const options = byTag(select, "OPTION");
    assert.deepEqual(options.map((o) => o.value), ["general.interview", "panaderia"]);
    assert.ok(options[1].selected, "the offer's list is chosen");
    assert.ok(card.text().includes(es["mi.offer_hint"]));
    select.value = "general.interview";
    await select.fire("change");
    await settle();
    assert.deepEqual(set, [["plan-1", null]], "the general list is sent as null");
  });
});

test("the offer table keeps the list of each live round", async () => {
  const { boardRows } = await import("../../transcriber/adapters/inbound/web/js/lib/offers.js");
  const rows = boardRows({ people: [
    { person: "file:a", latest: { session_id: "r1", kind: "live", mandatory_list: { id: "panaderia", name: "Panadería", version: 2 } } },
    { person: "file:b", latest: { session_id: "r2", kind: "ai" } },
  ] });
  assert.deepEqual(rows.map((r) => r.mandatoryList?.id ?? null), ["panaderia", null]);
  assert.equal(listUsedText(rows[0].mandatoryList, tr), "Lista «Panadería» · versión 2");
});
