// The «Competencia» page (#291): its data file (every external figure
// with its source and the date it was seen, both languages, nothing
// private), the with/without retrieval estimate computed from the
// measured inputs, its texts in both dictionaries and its place in the
// navigation of every page.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BASES, KINDS, SCHEMA, WHERE, callCost, citedSources, competitorsOf, digitsFor, isBilingual,
  isCompetitionData, maximumLibraries, promptBase, ragMaxRows, ragRows, textOf,
} from "../../transcriber/adapters/inbound/web/js/lib/competition.js";
import { ADMIN_TOOLS } from "../../transcriber/adapters/inbound/web/js/lib/home.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const WEB = new URL("../../transcriber/adapters/inbound/web/", import.meta.url).pathname;
const RAW = readFileSync(`${WEB}competition/data.json`, "utf8");
const data = JSON.parse(RAW);
const VIEW = readFileSync(`${WEB}js/views/competition.js`, "utf8");

test("the data file is a competition.v1 file the page can draw", () => {
  assert.equal(data.schema, SCHEMA);
  assert.ok(isCompetitionData(data));
  assert.ok(!isCompetitionData(null));
  assert.ok(!isCompetitionData({ ...data, schema: "competition.v0" }));
  assert.ok(!isCompetitionData({ ...data, competitors: [] }));
  assert.match(data.updated, /^\d{4}-\d{2}-\d{2}$/);
});

test("every source the file cites exists and is a public https address", () => {
  for (const { id, where } of citedSources(data)) {
    const source = data.sources[id];
    assert.ok(source, `${where} cites ${id}`);
    assert.match(source.url, /^https:\/\/[^\s]+$/, id);
    assert.ok(source.label, id);
  }
  for (const [id, source] of Object.entries(data.sources)) {
    assert.match(source.url, /^https:\/\//, id);
    if (!source.url.startsWith("https://github.com/ALRO-GM/RAG/")) {
      assert.match(source.seen ?? "", /^\d{4}-\d{2}-\d{2}$/, `${id}: an outside source says when it was seen`);
    }
  }
});

test("every competitor has a purpose, a price with its basis, its limits, where it runs and its source", () => {
  assert.ok(data.competitors.length >= 6);
  for (const c of data.competitors) {
    assert.ok(KINDS.includes(c.kind), c.name);
    assert.ok(BASES.includes(c.basis), c.name);
    assert.equal(c.runs, "cloud", c.name);
    for (const field of ["price", "users"]) assert.ok(isBilingual(c[field]), `${c.name}.${field}`);
    const source = data.sources[c.source];
    assert.ok(source?.seen, `${c.name}: a price is shown with the date it was seen`);
    assert.doesNotMatch(source.url, /github\.com/, `${c.name}: a vendor page or a marketplace, never our notes`);
    if (c.basis === "marketplace") assert.match(source.url, /aws\.amazon\.com\/marketplace/, c.name);
    if (c.basis === "official") assert.match(c.price.es, /\d/, `${c.name}: an official price has a figure`);
    if (c.basis === "none") {
      assert.match(c.price.es, /^Sin precio público/, c.name);
      assert.match(c.price.en, /^No public price/, c.name);
      assert.doesNotMatch(c.price.es, /\d/, `${c.name}: no estimate published as a price`);
    }
  }
  for (const kind of KINDS) assert.ok(competitorsOf(data.competitors, kind).length > 0, kind);
  const byName = Object.fromEntries(data.competitors.map((c) => [c.name, c]));
  assert.match(byName.Metaview.price.es, /precio no comprobado/, "a tab the tool could not read is not «sin precio público»");
  assert.match(byName.Five9.price.es, /usuario simultáneo.*cargos por uso/);
  assert.match(byName.PolyAI.price.es, /telefonía no incluida/);
  assert.match(byName.Twilio.price.es, /puede cambiar/);
  for (const key of ["comp.table.note"]) {
    assert.match(es[key], /Sin precio público/);
    assert.match(es[key], /Precio no comprobado/);
    assert.match(es[key], /EE\. UU\..*no se comprobaron en euros ni para España/);
  }
});

test("no third-party estimate or aggregator figure is published as a price", () => {
  for (const pattern of [/vendr/i, /\bg2\b/i, /capterra/i, /mediana/i, /median/i, /Genesys/i]) {
    assert.doesNotMatch(RAW, pattern);
  }
});

test("every text of the file is in both languages, and every figure about us has its source", () => {
  const lists = [data.steps.map((s) => s.title), data.steps.map((s) => s.body), data.rag_uses.yes, data.rag_uses.no,
    data.models.map((m) => m.phase), data.models.map((m) => m.model), data.strengths.map((s) => s.title),
    data.strengths.map((s) => s.body), data.gaps, data.content_today, data.content_planned,
    data.roadmap.map((r) => r.title), data.roadmap.map((r) => r.body), data.rag.libraries.map((l) => l.label)];
  for (const list of lists) for (const item of list) assert.ok(isBilingual(item), JSON.stringify(item));
  for (const item of [data.privacy, data.content_automatic, data.ours.name, data.ours.price, data.ours.users]) {
    assert.ok(isBilingual(item), JSON.stringify(item));
  }
  for (const m of data.models) {
    assert.ok(WHERE.includes(m.where), m.id);
    assert.ok(data.sources[m.source], `model ${m.id} has a source`);
  }
  for (const item of [...data.strengths, ...data.roadmap]) {
    if (/\d/.test(item.body.es)) assert.ok(data.sources[item.source], `a figure needs its source: ${item.body.es}`);
  }
  for (const item of data.content_today) assert.ok(data.sources[item.source], item.es);
});

test("about our system: the audio stays, the text goes out; one live session; synthetic figures", () => {
  assert.match(data.privacy.es, /Claude \(Anthropic\)/);
  assert.match(data.privacy.en, /Claude \(Anthropic\)/);
  assert.doesNotMatch(RAW, /todo (es |funciona en )?local|everything (is |runs )?local/i);
  const allEs = [RAW, ...Object.entries(es).filter(([k]) => k.startsWith("comp.")).map(([, v]) => v)].join(" ");
  assert.doesNotMatch(allEs, /funcionan en la nube|«En la nube»|"En la nube"|runs in the cloud/i, "no blanket claim about where others run");
  assert.doesNotMatch(allEs, /servicios externos|outside services|not sent outside/i, "outside services are AI services, said so");
  assert.doesNotMatch(allEs, /idioma secundario|secondary language/, "no unverified claim about competitors' Spanish");
  assert.match(es["comp.col.runs"], /según su página/);
  assert.equal(data.ours.runs, "mixed");
  assert.match(data.ours.users.es, /Una sesión en directo a la vez/);
  assert.ok(data.models.some((m) => m.where === "external"), "the outside model is named");
  assert.ok(data.gaps.some((g) => /sintéticas/.test(g.es)));
  assert.match(es["comp.synthetic"], /sintéticas/);
  assert.match(data.content_automatic.es, /^Hoy nada aprende solo/);
  assert.match(es["comp.next.lead"], /todavía no existe/);
});

test("nothing private or personal is published", () => {
  assert.doesNotMatch(RAW, /\/home\/|file:\/\/|data\/research|research_notes|competencia\/reports|@[\w-]+\.\w/);
  assert.doesNotMatch(VIEW, /\/home\/|data\/research/);
});

test("a call costs the list price, five times more above the long-prompt step", () => {
  const price = data.rag.price;
  const near = (a, b) => Math.abs(a - b) < 1e-12;
  assert.ok(near(callCost(price, 0, 1_000_000), 0.50), "output alone: 0.50 per million");
  assert.ok(near(callCost(price, 100_000, 0), 0.01), "100,000 input tokens: 0.10 per million");
  assert.ok(near(callCost(price, 100_000, 1000), 0.0105));
  assert.ok(near(callCost(price, 100_001, 1000), (100_001 * 0.5 + 1000 * 2.5) / 1e6), "one token more: everything at 5 times");
});

test("the estimate on the sample libraries: the measured prompt against the whole library (docs/web/COMPETITION_PAGE.md)", () => {
  const rows = ragRows(data.rag);
  assert.deepEqual(rows.map((r) => r.id), ["with", "support", "interview"]);
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  const [withRag, support, interview] = rows;
  assert.equal(promptBase(data.rag), 1500);
  assert.equal(withRag.tokens, 1750);
  assert.ok(near(withRag.cost, 0.22), withRag.cost);
  assert.equal(support.tokens, 4600);
  assert.ok(near(support.cost, 0.505), support.cost);
  assert.equal(interview.tokens, 13800);
  assert.ok(near(interview.cost, 1.425), interview.cost);
  for (const row of rows.slice(1)) assert.ok(row.tokens > withRag.tokens && row.cost > withRag.cost, row.id);
  assert.equal(digitsFor(0.22), 2);
  assert.equal(digitsFor(12.3), 1);
  assert.equal(digitsFor(129.7), 0);
});

test("the large library is a maximum, with the same piece size on both sides and on its own scale", () => {
  const libs = maximumLibraries(data.rag);
  assert.deepEqual(libs.map((l) => l.id), ["large"]);
  assert.equal(data.rag.top_k, 4, "config/products/live_assist.*.yaml: top_k 4");
  const [withRag, without] = ragMaxRows(data.rag, libs[0]);
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  assert.equal(withRag.tokens, 1500 + 4 * 1000 / 4);
  assert.ok(near(withRag.cost, 0.295), withRag.cost);
  assert.equal(without.tokens, 259000);
  assert.ok(without.long);
  assert.ok(near(without.cost, 129.725), without.cost);
  assert.ok(near(without.ratio, 259000 / 2500));
  assert.match(es["comp.rag.max"], /Como máximo: si cada trozo fuese de \{chars\} caracteres/);
  assert.match(en["comp.rag.max"], /At most/);
});

test("the estimate is labelled as such and says what is not measured", () => {
  assert.match(VIEW, /t\("comp\.estimate"\)/);
  assert.equal(es["comp.estimate"], "Estimación");
  assert.match(es["comp.rag.assume"], /no está medida/);
  assert.match(es["comp.rag.assume"], /unos \{n\} tokens de respuesta/);
  assert.doesNotMatch(es["comp.rag.assume"], /hasta/);
  assert.match(es["comp.rag.with"], /medida en la prueba de latencia/);
  assert.match(es["comp.rag.acts"], /\{model\}/);
  assert.doesNotMatch(es["comp.rag.acts"], /[Úú]nico/);
  assert.match(es["comp.rag.acts"], /no usa hoy/);
  assert.match(en["comp.rag.assume"], /not measured/);
  assert.match(es["comp.rag.without_time"], /no está medido/);
});

test("a {es, en} text is taken in the page's language", () => {
  assert.equal(textOf({ es: "hola", en: "hello" }, "en"), "hello");
  assert.equal(textOf({ es: "hola" }, "en"), "hola");
  assert.equal(textOf("tal cual", "en"), "tal cual");
  assert.equal(textOf(null, "es"), "");
  assert.ok(!isBilingual({ es: "solo" }));
});

test("every dictionary key the page uses exists in both languages and differs where it should", () => {
  const keys = new Set([...VIEW.matchAll(/t\(\s*[`"](comp\.[\w.]+)[`"]/g)].map((m) => m[1]));
  for (const kind of KINDS) keys.add(`comp.kind.${kind}`);
  for (const key of ["comp.where.local", "comp.where.external", "comp.where.both", "comp.runs.cloud", "comp.runs.mixed",
    "comp.title", "comp.loading", "comp.error", "nav.competition", "admin.competition"]) keys.add(key);
  assert.ok(keys.size > 40);
  for (const key of keys) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  for (const key of ["comp.title", "comp.lead", "nav.competition", "comp.h.market"]) assert.notEqual(es[key], en[key], key);
});

/** The hrefs of a page's sidebar and its Administración group. */
function sidebar(page) {
  const html = readFileSync(`${WEB}${page}`, "utf8");
  const nav = html.slice(html.indexOf('<nav data-i18n-label="nav.main"'), html.indexOf("</nav>"));
  const admin = nav.slice(nav.indexOf('<div class="nav-admin"'));
  return [...admin.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1]);
}

test("«Competencia» is the last entry of Administración, under «Investigación», on every page and in the overview", () => {
  for (const page of ["index.html", "study.html", "research.html", "competition.html"]) {
    const admin = sidebar(page);
    assert.equal(admin.at(-1), "competition.html", page);
    assert.equal(admin.at(-2), "research.html", page);
  }
  assert.deepEqual(ADMIN_TOOLS.slice(-2).map((tool) => tool.href), ["research.html", "competition.html"]);
  const html = readFileSync(`${WEB}competition.html`, "utf8");
  assert.match(html, /<a href="competition\.html" aria-current="page" data-i18n="nav\.competition">/);
  assert.match(html, /<span aria-current="page" data-i18n="nav\.competition">/);
  assert.match(html, /<button type="button" id="nav-more"/, "the phone's «Más» opens the sidebar that lists it");
  const icons = readFileSync(`${WEB}icons.svg`, "utf8");
  assert.match(icons, /<symbol id="i-competition"/);
});
