// The «Info» page («Competencia» in #291, renamed and reordered in #295):
// its data file (every external figure with its source and the date it was
// seen, both languages, nothing private), the with/without retrieval
// estimate, the cost of the same service per scenario, the task-by-task
// comparison and us against one company, its texts in both dictionaries,
// its place in the navigation of every page and the old address.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  BASES, KINDS, LEVELS, SCHEMA, WHERE, callCost, cellOf, citedInfoSources, citedSources, companies, competitorCost, costScale,
  competitorsOf, digitsFor, haveIt, isBilingual, isCompetitionData, maximumLibraries, oursCost, oursPerHour, planCost,
  promptBase, ragMaxRows, ragRows, scenarioOf, textOf, versus,
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
  assert.ok(!isCompetitionData({ ...data, tasks: [] }), "#295: the task charts are required");
  assert.ok(!isCompetitionData({ ...data, cost: undefined }), "#295: the cost scenarios are required");
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
    assert.ok(["cloud", "unpublished"].includes(c.runs), `${c.name}: cloud only where its page says so (#295)`);
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
  for (const level of LEVELS) keys.add(`comp.level.${level}`);
  for (const sc of data.cost.scenarios) keys.add(`comp.cost.scenario.${sc.id}`);
  for (const kind of KINDS) keys.add(`comp.cost.other.${kind}`);
  assert.ok(keys.size > 40);
  for (const key of keys) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  for (const key of ["comp.title", "comp.lead", "comp.h.market", "comp.h.cost", "comp.h.tasks", "comp.h.vs"]) assert.notEqual(es[key], en[key], key);
});

/** The hrefs of a page's sidebar and its Administración group. */
function sidebar(page) {
  const html = readFileSync(`${WEB}${page}`, "utf8");
  const nav = html.slice(html.indexOf('<nav data-i18n-label="nav.main"'), html.indexOf("</nav>"));
  const admin = nav.slice(nav.indexOf('<div class="nav-admin"'));
  return [...admin.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1]);
}

test("«Info» is the last entry of Administración, under «Investigación», on every page and in the overview", () => {
  for (const page of ["index.html", "study.html", "research.html", "info.html"]) {
    const admin = sidebar(page);
    assert.equal(admin.at(-1), "info.html", page);
    assert.equal(admin.at(-2), "research.html", page);
    assert.match(readFileSync(`${WEB}${page}`, "utf8"), /<span class="label">Info<\/span>/, page);
  }
  assert.deepEqual(ADMIN_TOOLS.slice(-2).map((tool) => tool.href), ["research.html", "info.html"]);
  const html = readFileSync(`${WEB}info.html`, "utf8");
  assert.match(html, /<a href="info\.html" aria-current="page" data-i18n="nav\.competition">/);
  assert.match(html, /<span aria-current="page" data-i18n="nav\.competition">Info<\/span>/);
  assert.match(html, /<title data-i18n="comp\.title">Info: /);
  assert.match(html, /<button type="button" id="nav-more"/, "the phone's «Más» opens the sidebar that lists it");
  assert.equal(es["nav.competition"], "Info");
  assert.equal(en["nav.competition"], "Info");
  assert.match(es["comp.title"], /^Info: /);
  assert.match(en["comp.title"], /^Info: /);
  const icons = readFileSync(`${WEB}icons.svg`, "utf8");
  assert.match(icons, /<symbol id="i-competition"/);
});

test("the old address competition.html still works: it sends the browser to info.html with the section", () => {
  const old = readFileSync(`${WEB}competition.html`, "utf8");
  assert.match(old, /<meta http-equiv="refresh" content="0; url=info\.html">/);
  assert.match(old, /<script src="js\/info-redirect\.js"><\/script>/);
  assert.match(old, /<a href="info\.html">Info<\/a>/, "a plain link when nothing else works");
  assert.doesNotMatch(old, /<script>/, "no inline script");
  const script = readFileSync(`${WEB}js/info-redirect.js`, "utf8");
  assert.match(script, /location\.replace\(`info\.html\$\{location\.hash\}`\)/);
});

// ---- #295: tasks, the cost of the same service, us against one company ----

const byId = Object.fromEntries(data.competitors.map((c) => [c.id, c]));
const taskOf = (id) => data.tasks.find((task) => task.id === id);

test("every chart has the same rows in the same order: us first, then the companies in file order", () => {
  const rows = companies(data);
  assert.equal(rows[0].id, "ours");
  assert.ok(rows[0].ours);
  assert.deepEqual(rows.slice(1).map((r) => r.id), data.competitors.map((c) => c.id));
  assert.equal(new Set(data.competitors.map((c) => c.id)).size, data.competitors.length, "ids are unique");
  for (const c of data.competitors) assert.match(c.id, /^[a-z0-9]+$/, c.name);
});

test("the tasks the issue names are all charted, our strengths first and our limits last", () => {
  assert.deepEqual(data.tasks.map((task) => task.id),
    ["interviews", "where", "must_say", "live_docs", "content", "transcribe", "after", "voicebot", "limits"]);
  for (const task of data.tasks) {
    assert.ok(isBilingual(task.title) && isBilingual(task.question), task.id);
    assert.ok(LEVELS.includes(task.ours.level), task.id);
    assert.ok(isBilingual(task.ours.text), task.id);
    assert.ok(data.sources[task.ours.source], `${task.id}: our cell has its source`);
  }
});

test("every cell about a company is backed: a level, its words in both languages and a source with the date it was seen", () => {
  for (const task of data.tasks) {
    for (const [id, cell] of Object.entries(task.cells)) {
      assert.ok(byId[id], `${task.id}: ${id} is a company of the file`);
      assert.ok(LEVELS.includes(cell.level), `${task.id}.${id}`);
      if (cell.level !== "unknown" || cell.text) {
        assert.ok(isBilingual(cell.text), `${task.id}.${id}: what it says`);
        const source = data.sources[cell.source];
        assert.ok(source, `${task.id}.${id}: a claim about a company has its source`);
        assert.match(source.seen ?? "", /^\d{4}-\d{2}-\d{2}$/, `${task.id}.${id}: seen on a date`);
        assert.doesNotMatch(source.url, /github\.com/, `${task.id}.${id}: the company's page, never our notes`);
      }
    }
    for (const c of data.competitors) assert.ok(LEVELS.includes(cellOf(task, c.id).level));
    assert.equal(cellOf(taskOf("must_say"), "hirevue").level, "unknown", "a cell the file does not fill is «no publicado»");
  }
  for (const { id, where } of citedInfoSources(data)) {
    assert.ok(data.sources[id], `${where} cites ${id}`);
    assert.match(data.sources[id].url, /^https:\/\//, id);
  }
});

test("«No» about a company only where the cited page says so: its own quote, per cell", () => {
  for (const task of data.tasks) {
    for (const [id, cell] of Object.entries(task.cells)) {
      if (cell.level !== "no") continue;
      assert.equal(task.id, "where", `${task.id}.${id}: a «no» elsewhere would need its own source`);
      assert.ok(data.sources[cell.source]?.seen, `${id}: the page that says it`);
      assert.ok(typeof cell.quote === "string" && cell.quote.length > 10, `${id}: the words of that page`);
      assert.match(cell.quote, /cloud|SaaS|hosting/i, `${id}: the quote says where it runs`);
      assert.equal(byId[id].runs, "cloud", `${id}: the badge says the same`);
      assert.match(cell.text.es, /según su web/);
    }
  }
  const where = taskOf("where");
  for (const c of data.competitors) {
    if (c.runs === "cloud") assert.equal(cellOf(where, c.id).level, "no", `${c.id}: a cloud badge needs its cell`);
    else assert.equal(cellOf(where, c.id).level, "unknown", `${c.id}: no quote, «No publicado»`);
  }
  assert.deepEqual(data.competitors.filter((c) => c.runs === "cloud").map((c) => c.id), ["nice", "five9", "cresta", "observe", "vapi"]);
  assert.match(VIEW, /runsBadge\(c\.runs\)/, "the badge follows the data, never a fixed text");
  assert.doesNotMatch(VIEW, /h\("span", \{ class: "comp-runs is-cloud" \}/);
  assert.match(es["comp.level.unknown"], /^No publicado$/);
  assert.match(es["comp.level.unknown_means"], /no quiere decir que no lo tenga/);
  assert.match(en["comp.level.unknown_means"], /does not mean/);
});

test("cells the review checked against the companies' pages (#295)", () => {
  assert.equal(cellOf(taskOf("voicebot"), "talkdesk").level, "yes");
  assert.match(cellOf(taskOf("voicebot"), "talkdesk").quote, /virtual assistant for voice/);
  assert.equal(cellOf(taskOf("voicebot"), "five9").level, "yes");
  assert.match(cellOf(taskOf("voicebot"), "five9").quote, /voice/);
  for (const id of ["nice", "cresta"]) assert.equal(cellOf(taskOf("voicebot"), id).level, "unknown", id);
  for (const id of ["five9", "cresta", "talkdesk"]) assert.equal(cellOf(taskOf("must_say"), id).level, "partial", id);
  assert.equal(cellOf(taskOf("must_say"), "twilio").level, "yes");
  assert.equal(cellOf(taskOf("interviews"), "metaview").source, "metaview_screening");
  assert.equal(data.sources.metaview_screening.url, "https://www.metaview.ai/screening");
  assert.equal(cellOf(taskOf("transcribe"), "metaview").level, "yes");
  assert.equal(cellOf(taskOf("transcribe"), "brighthire").level, "unknown");
  for (const task of data.tasks) for (const cell of Object.values(task.cells)) if (cell.level === "partial") assert.ok(cell.text, task.id);
});

test("what we have not got is not claimed: no phone, one live conversation, the text goes to Claude", () => {
  assert.equal(taskOf("voicebot").ours.level, "partial");
  assert.match(taskOf("voicebot").ours.text.es, /sin teléfono/);
  assert.equal(taskOf("limits").ours.level, "no");
  assert.match(taskOf("limits").ours.text.es, /Una conversación en directo a la vez/);
  assert.match(taskOf("where").ours.text.es, /el texto sí va a Claude \(Anthropic\)/);
  assert.equal(data.strengths[0].title.es, "La voz no va a ningún servicio de IA externo", "#295: the exact wording stays");
  assert.match(data.strengths[1].body.es, /No aprende solo de las llamadas/);
  assert.match(es["comp.cost.ours_note"], /hoy hace falta un servidor por cada conversación en directo a la vez/);
  assert.doesNotMatch(data.strengths[0].body.es, /nube/, "no blanket claim about where the others run");
});

test("the scenarios: people, hours, calls and minutes of a month", () => {
  const [small, large] = data.cost.scenarios.map((sc) => scenarioOf(data.cost, sc));
  assert.deepEqual(small, { id: "small", agents: 5, hours: 400, minutes: 24000, calls: 2400 });
  assert.deepEqual(large, { id: "large", agents: 50, hours: 4000, minutes: 240000, calls: 24000 });
});

test("our cost is an estimate from the measured tokens and session cost at list price (docs/web/COMPETITION_PAGE.md §3c)", () => {
  const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;
  const haiku = oursPerHour(data.cost, data.rag, "haiku");
  assert.ok(near(haiku.answersPerHour, 13 / 108 * 3600), "13 model calls in the 108 s test call");
  assert.ok(near(haiku.live, haiku.answersPerHour * 0.00022), "1,750 in + 90 out at 0.10 / 0.50");
  assert.ok(near(haiku.session, (300 * 0.1 + 287729 * 0.2 + 4585 * 0.01 + 88498 * 0.5) / 1e6));
  assert.equal(Math.round(haiku.session * 1000) / 1000, data.cost.ours.analysis_usd_reported, "the 0.102 USD MODEL_COMPARISON.md reports");
  assert.ok(near(haiku.analysis, haiku.session * 3600 / 881.8), "the 881.8 s test meeting, scaled by duration");
  const opus = oursPerHour(data.cost, data.rag, "opus");
  assert.ok(near(opus.session, (300 * 4 + 287729 * 8 + 4585 * 0.2 + 88498 * 20) / 1e6), "Opus 5.5 cache hits at 0.20 (0.05x)");
  const [small, large] = data.cost.scenarios.map((sc) => oursCost(data.cost, data.rag, scenarioOf(data.cost, sc)));
  assert.deepEqual([small.haiku, small.opus, large.haiku, large.opus].map(Math.round), [204, 6691, 2045, 66909]);
  assert.equal(es["comp.cost.estimate"], "Estimación de coste, no es un precio");
  assert.match(es["comp.cost.ours_opus"], /Opus.*opción por defecto hoy/);
  assert.match(es["comp.cost.ours_haiku"], /Haiku/);
  assert.match(es["comp.cost.assume"], /llamadas al modelo por hora \(calculadas a partir de una llamada de prueba de 108 s\)/);
  assert.doesNotMatch(es["comp.cost.assume"], /sugerencias por hora \(medidas\)/);
  assert.match(es["comp.cost.assume"], /sin el servidor con tarjeta gráfica ni la luz/i);
  assert.match(es["comp.cost.assume"], /medido con Haiku; con Opus, los mismos tokens a precio de Opus \(no medido\)/);
  assert.match(VIEW, /t\("comp\.cost\.ours_haiku"\)/);
  assert.match(VIEW, /t\("comp\.cost\.ours_opus"\)/);
  assert.match(data.ours.price.es, /Sin precio definido/, "a cost, never presented as our price");
});

test("each company's published price applied to the scenario, or none", () => {
  const [small, large] = data.cost.scenarios.map((sc) => scenarioOf(data.cost, sc));
  const range = (id, sc) => { const r = competitorCost(data.cost, id, sc); return r && [Math.round(r.low), Math.round(r.high)]; };
  assert.deepEqual(range("talkdesk", small), [525, 1125]);
  assert.deepEqual(range("nice", small), [550, 1845], "the top plan adds 0.25 USD per session (a call)");
  assert.deepEqual(range("five9", small), [7950, 7950], "Core, the cheapest plan with voice, minimum of 50 seats");
  assert.deepEqual(range("five9", large), [7950, 7950]);
  assert.deepEqual(range("twilio", small), [1240, 1590], "0.035 per minute plus 1 per hour or 150 per user");
  assert.deepEqual(range("cresta", small), [12500, 12500], "the yearly contract; 1.50 per call is only additional usage on top of it");
  assert.match(data.cost.competitors.cresta.note.es, /1,50 USD por llamada adicional/);
  assert.deepEqual(range("cresta", large), [36000, 36000], "288,000 calls a year: 188,000 extra at 1.50");
  assert.deepEqual(range("observe", small), [345, 4140], "828 per agent, a year or a month: the listing is unclear");
  for (const id of ["brighthire", "hirevue", "metaview", "retell", "vapi", "polyai"]) assert.equal(competitorCost(data.cost, id, small), null, id);
  assert.equal(planCost({ seat: 10 }, small, 50), 500);
  for (const [id, spec] of Object.entries(data.cost.competitors)) {
    assert.ok(byId[id], id);
    assert.equal(byId[id].kind, "callcentre", `${id}: the scenario is a call centre`);
    assert.notEqual(byId[id].basis, "none", `${id}: a figure only from a published price`);
    assert.ok(isBilingual(spec.note), id);
  }
  assert.ok(competitorCost(data.cost, "observe", small).unclear);
});

test("the cost chart's scale: the largest row but our Opus row, which is drawn cut and says «fuera de escala»", () => {
  const rows = [{ id: "ours-haiku", high: 2045 }, { id: "ours-opus", high: 66909 }, { id: "talkdesk", high: 11250 },
    { id: "observe", high: 41400 }, { id: "brighthire", high: null }];
  assert.deepEqual(costScale(rows), { max: 41400, out: ["ours-opus"] });
  assert.deepEqual(costScale(rows.map((r) => (r.id === "ours-opus" ? { ...r, high: 6691 } : r))), { max: 41400, out: [] }, "not cut when it fits");
  assert.equal(costScale([{ id: "a", high: 5 }, { id: "ours-opus", high: 9 }], []).out.length, 0, "with no outlier the largest row sets the scale");
  assert.match(es["comp.cost.off_scale"], /^Fuera de escala/);
  assert.match(en["comp.cost.off_scale"], /^Off the scale/);
  assert.match(VIEW, /t\("comp\.cost\.off_scale"\)/, "said in the text, not only drawn");
  assert.match(VIEW, /class: "break"/);
});

test("a task chart says how many companies publish nothing about it", () => {
  assert.deepEqual(haveIt(taskOf("where"), data.competitors), { n: 0, unknown: 7, of: 12 });
  assert.match(es["comp.task.count_unknown"], /\(\{u\} no lo publican\)/);
  assert.match(VIEW, /comp\.task\.count_unknown/);
});

test("us against one company: where we do more, where it publishes nothing, where it does more, what both do", () => {
  assert.deepEqual(versus(data, "talkdesk"), {
    weMore: ["must_say"], unknown: ["interviews", "where", "content"], theyMore: ["voicebot", "limits"],
    both: ["live_docs", "transcribe", "after"],
  });
  assert.deepEqual(versus(data, "retell").theyMore, ["voicebot", "limits"]);
  assert.match(es["comp.vs.unknown"], /^Lo que \{name\} no publica$/, "no «Nosotros sí» where we are only in part");
  assert.doesNotMatch(es["comp.vs.we_more"], /Nosotros sí/);
  assert.match(VIEW, /mark\(cellOf\(task, "ours"\)\.level\)/, "our own mark next to each task");
  for (const c of data.competitors) {
    const v = versus(data, c.id);
    for (const id of v.weMore) assert.notEqual(cellOf(taskOf(id), c.id).level, "unknown", `${c.id}.${id}: «no publicado» is never «no»`);
    for (const offer of c.offers) {
      assert.ok(isBilingual(offer), c.name);
      assert.ok(data.sources[offer.source]?.seen, `${c.name}: what it offers has its source`);
    }
  }
  assert.match(VIEW, /h\("span", \{ id: "competencia"/, "competition.html#competencia lands on the prices");
  assert.deepEqual(haveIt(taskOf("limits"), data.competitors), { n: 10, unknown: 2, of: 12 });
});

test("every chart is a table with its words: marks carry a word, bars sit next to their figure", () => {
  assert.match(VIEW, /h\("table", \{ role: "table", class: "comp-chart comp-matrix" \}/);
  assert.match(VIEW, /h\("table", \{ role: "table", class: "comp-chart comp-cost-chart" \}/);
  assert.match(VIEW, /comp-mark-word/);
  assert.match(VIEW, /"aria-hidden": "true"/);
  const css = readFileSync(`${WEB}style.css`, "utf8");
  assert.match(css, /\.comp-chart tr\.is-ours > th \{ box-shadow: inset 4px 0 0 var\(--accent\); \}/);
  assert.match(css, /@media \(max-width: 47\.99rem\) \{\n  \.comp-chart, /);
});
