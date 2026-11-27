// The «Info» page's body (#291, reordered in #295): our strengths first,
// then the cost of the same service against each company (two
// scenarios), one chart per task with us first, us against one company
// at a time, the published prices and where we are behind, and then how
// it works (steps, with/without retrieval, models, content, roadmap).
// Every figure and every sentence about a company or our system comes
// from competition/data.json; headings and labels from the dictionaries.
// Every chart is a table (rows and cells are its text equivalent); its
// bars and marks are drawn next to words, never colour alone.
// Colours come from classes in style.css (the CSP allows no inline
// styles); text always goes in as text nodes.

import { h } from "../dom.js";
import { decimal, lang, t } from "../i18n.js";
import {
  KINDS, cellOf, companies, competitorCost, competitorsOf, digitsFor, haveIt, maximumLibraries, oursCost,
  ragMaxRows, ragRows, scenarioOf, textOf, versus,
} from "../lib/competition.js";

const NS = "http://www.w3.org/2000/svg";

/** An SVG element with attributes and children (strings become text). */
function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null && v !== false) el.setAttribute(k, String(v));
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

/** A text of the data file in the page's language. */
const tx = (value) => textOf(value, lang);

/** An integer with the language's thousands separator. */
const whole = (n) => decimal(Math.round(n), 0);

/** A dollar amount with sensible decimals. */
const usd = (v) => `${decimal(v, digitsFor(v))} USD`;

/** "27/11/2026" or "11/27/2026" from an ISO date. */
function day(iso) {
  const [y, m, d] = String(iso).split("-").map(Number);
  if (!y || !m || !d) return String(iso ?? "");
  return new Intl.DateTimeFormat(lang, { day: "numeric", month: "numeric", year: "numeric", timeZone: "UTC" })
    .format(new Date(Date.UTC(y, m - 1, d)));
}

/** A link to a source of the data file (outbound, never fetched). */
function sourceLink(data, id, text = null) {
  const source = data.sources[id];
  if (!source) return null;
  return h("a", { href: source.url, rel: "noopener noreferrer" }, text ?? source.label);
}

/** "Fuente: …" under a block. */
function sourceLine(data, ids) {
  const links = ids.map((id) => sourceLink(data, id)).filter(Boolean);
  if (links.length === 0) return null;
  const parts = [];
  links.forEach((link, i) => { if (i) parts.push(" · "); parts.push(link); });
  return h("p", { class: "hint comp-source" }, t("comp.source"), " ", parts);
}

/** A small icon from the sprite. */
function icon(name) {
  return s("svg", { class: "icon", "aria-hidden": "true", focusable: "false" }, s("use", { href: `icons.svg#i-${name}` }));
}

/** A section with its heading and an optional lead sentence. */
function section(id, titleKey, leadKey, ...children) {
  return h("section", { id, "aria-labelledby": `h-${id}` },
    h("h2", { id: `h-${id}` }, t(titleKey)),
    leadKey ? h("p", { class: "comp-lead" }, t(leadKey)) : null,
    ...children);
}

/** The «RAG» tag on a step that looks up the customer's documents. */
function ragTag() {
  return h("span", { class: "comp-rag" }, t("comp.rag.tag"));
}

function howItWorks(data) {
  const steps = h("ol", { class: "comp-flow" }, data.steps.map((step) => h("li", { class: step.rag ? "comp-step uses-rag" : "comp-step" },
    h("h3", {}, tx(step.title)),
    h("p", {}, tx(step.body)),
    step.rag ? ragTag() : null)));
  const uses = (key, items, cls) => h("div", { class: `comp-uses ${cls}` },
    h("h3", {}, t(key)),
    h("ul", {}, items.map((item) => h("li", {}, tx(item)))));
  return section("como", "comp.h.how", "comp.how.lead",
    steps,
    h("p", { class: "comp-explain" }, ragTag(), " ", t("comp.rag.what")),
    h("div", { class: "comp-two" },
      uses("comp.rag.yes", data.rag_uses.yes, "is-yes"),
      uses("comp.rag.no", data.rag_uses.no, "is-no")));
}

/** One bar on a 0-100 scale that stretches with its box (no redraw). */
function bar(fraction, cls) {
  const w = Math.max(0.6, Math.min(100, fraction * 100));
  return s("svg", { class: "comp-bar", viewBox: "0 0 100 10", preserveAspectRatio: "none", "aria-hidden": "true" },
    s("rect", { x: 0, y: 0, width: 100, height: 10, rx: 0, class: "track" }),
    s("rect", { x: 0, y: 0, width: w, height: 10, class: cls }));
}

/** A list of bars on one scale: [{ label, row, first }]. */
function barList(rag, items, title) {
  const max = Math.max(...items.map((item) => item.row.tokens));
  return h("ul", { class: "comp-bars", "aria-label": title }, items.map(({ label, row, first }) => {
    const times = first ? null : t("comp.rag.times", { n: decimal(row.ratio, row.ratio < 10 ? 1 : 0) });
    return h("li", { class: first ? "is-with" : "is-without" },
      h("p", { class: "comp-bar-label" }, h("strong", {}, label)),
      bar(row.tokens / max, first ? "fill-with" : "fill-without"),
      h("p", { class: "comp-bar-values" },
        h("span", {}, t("comp.rag.tokens", { n: whole(row.tokens) })),
        times ? h("span", { class: "comp-times" }, times) : null,
        h("span", {}, t("comp.rag.cost", { cost: usd(row.cost), n: whole(rag.answers) })),
        row.long ? h("span", { class: "comp-long" }, t("comp.rag.long", { n: whole(rag.price.long_prompt_tokens) })) : null));
  }));
}

function withWithout(data) {
  const rag = data.rag;
  const rows = ragRows(rag);
  const samples = barList(rag, rows.map((row, i) => ({
    row, first: i === 0,
    label: i === 0 ? t("comp.rag.with") : t("comp.rag.without", { what: tx(row.label) }),
  })), t("comp.rag.chart"));
  const large = maximumLibraries(rag).map((lib) => {
    const [withRow, withoutRow] = ragMaxRows(rag, lib);
    return h("div", { class: "comp-max" },
      h("h3", {}, tx(lib.label)),
      h("p", { class: "hint" }, t("comp.rag.max", { chars: whole(rag.piece_chars), k: rag.top_k })),
      barList(rag, [
        { row: withRow, first: true, label: t("comp.rag.max_with", { k: rag.top_k }) },
        { row: withoutRow, first: false, label: t("comp.rag.max_without") },
      ], tx(lib.label)));
  });
  const tiles = h("div", { class: "comp-tiles" },
    h("div", { class: "comp-tile" },
      h("p", { class: "comp-big" }, t("comp.rag.search_ms", { lo: rag.search_ms[0], hi: rag.search_ms[1] })),
      h("p", {}, t("comp.rag.search"))),
    h("div", { class: "comp-tile" },
      h("p", { class: "comp-big" }, t("comp.rag.model_s", { lo: decimal(rag.model_s[0], 1), hi: decimal(rag.model_s[1], 1) })),
      h("p", {}, t("comp.rag.model"))),
    h("div", { class: "comp-tile is-unknown" },
      h("p", { class: "comp-big" }, t("comp.rag.unmeasured")),
      h("p", {}, t("comp.rag.without_time"))));
  const acts = rag.acts;
  return section("rag", "comp.h.rag", "comp.rag.lead",
    h("p", {}, h("span", { class: "badge warn" }, t("comp.estimate"))),
    h("h3", {}, t("comp.rag.h.samples")),
    samples,
    large,
    tiles,
    h("p", { class: "hint comp-assume" }, t("comp.rag.assume", {
      model: rag.price.model, in: decimal(rag.price.input_per_mtok, 2), out: decimal(rag.price.output_per_mtok, 2),
      n: whole(rag.output_tokens), long: whole(rag.price.long_prompt_tokens),
    })),
    acts ? h("p", { class: "hint" }, t("comp.rag.acts", {
      model: acts.model, n: acts.sentences, without: decimal(acts.without * 100, 0), with: decimal(acts.with * 100, 0),
      used: decimal(acts.in_use * 100, 0), m: whole(acts.in_use_sentences),
    })) : null,
    sourceLine(data, [rag.price.source]));
}

const WHERE_KEY = { local: "comp.where.local", external: "comp.where.external", both: "comp.where.both" };

function whereBadge(where) {
  const cls = where === "local" ? "is-local" : where === "external" ? "is-external" : "is-both";
  return h("span", { class: `comp-where ${cls}` }, t(WHERE_KEY[where] ?? "comp.where.both"));
}

function models(data) {
  return section("modelos", "comp.h.models", "comp.models.lead",
    h("p", { class: "notice info comp-privacy" }, tx(data.privacy)),
    h("ul", { class: "comp-models" }, data.models.map((m) => h("li", {},
      h("h3", {}, tx(m.phase)),
      h("p", {}, tx(m.model)),
      whereBadge(m.where)))));
}

const RUNS_KEY = { cloud: "comp.runs.cloud", mixed: "comp.runs.mixed", unpublished: "comp.runs.unpublished" };

/** Where a company runs, as its page says it (c.runs), or «No publicado». */
function runsBadge(runs) {
  const cls = runs === "cloud" ? "is-cloud" : runs === "mixed" ? "is-mixed" : "is-unpublished";
  return h("span", { class: `comp-runs ${cls}` }, t(RUNS_KEY[runs] ?? "comp.runs.unpublished"));
}

function competitorRow(data, c) {
  const source = data.sources[c.source];
  const name = h("th", { scope: "row", role: "rowheader" },
    h("span", { class: "comp-name" }, c.name),
    source ? h("span", { class: "comp-seen" },
      sourceLink(data, c.source, c.basis === "marketplace" ? t("comp.seen.market") : t("comp.seen.page")),
      " ", t("comp.seen.on", { date: day(source.seen) })) : null);
  return h("tr", { role: "row", class: c.basis === "none" ? "is-unpriced" : null },
    name,
    h("td", { role: "cell", "data-label": t("comp.col.price") }, tx(c.price)),
    h("td", { role: "cell", "data-label": t("comp.col.users") }, tx(c.users)),
    h("td", { role: "cell", "data-label": t("comp.col.runs") }, runsBadge(c.runs)));
}

function market(data) {
  const groups = KINDS.map((kind) => {
    const rows = competitorsOf(data.competitors, kind);
    if (rows.length === 0) return null;
    return h("tbody", { role: "rowgroup" },
      h("tr", { role: "row", class: "comp-group" }, h("th", { role: "rowheader", colspan: 4, scope: "colgroup" }, t(`comp.kind.${kind}`))),
      rows.map((c) => competitorRow(data, c)));
  });
  const ours = data.ours;
  const oursBody = h("tbody", { role: "rowgroup", class: "comp-ours" },
    h("tr", { role: "row" },
      h("th", { scope: "row", role: "rowheader" }, h("span", { class: "comp-name" }, tx(ours.name)), h("span", { class: "comp-seen" }, t("comp.ours.what"))),
      h("td", { role: "cell", "data-label": t("comp.col.price") }, tx(ours.price)),
      h("td", { role: "cell", "data-label": t("comp.col.users") }, tx(ours.users)),
      h("td", { role: "cell", "data-label": t("comp.col.runs") }, h("span", { class: "comp-runs is-mixed" }, t(RUNS_KEY[ours.runs] ?? "comp.runs.mixed")))));
  const table = h("div", { class: "comp-table-wrap" },
    h("table", { role: "table", class: "comp-table" },
      h("caption", {}, t("comp.table.caption")),
      h("thead", { role: "rowgroup" }, h("tr", { role: "row" },
        h("th", { role: "columnheader", scope: "col" }, t("comp.col.company")),
        h("th", { role: "columnheader", scope: "col" }, t("comp.col.price")),
        h("th", { role: "columnheader", scope: "col" }, t("comp.col.users")),
        h("th", { role: "columnheader", scope: "col" }, t("comp.col.runs")))),
      groups, oursBody));
  const gaps = h("ul", { class: "comp-gaps" }, data.gaps.map((item) => h("li", {}, tx(item))));
  return section("precios", "comp.h.market", "comp.market.lead",
    h("span", { id: "competencia", class: "comp-old-anchor" }),
    table,
    h("p", { class: "hint comp-assume" }, t("comp.table.note", { date: day(data.updated) })),
    h("h3", { id: "detras" }, t("comp.h.gaps")),
    gaps);
}

function content(data) {
  return section("contenido", "comp.h.content", "comp.content.lead",
    h("div", { class: "comp-two" },
      h("div", { class: "comp-uses is-yes" },
        h("h3", {}, t("comp.content.today")),
        h("ul", {}, data.content_today.map((item) => h("li", {}, tx(item))))),
      h("div", { class: "comp-uses is-planned" },
        h("h3", {}, t("comp.content.planned")),
        h("ul", {}, data.content_planned.map((item) => h("li", {}, tx(item)))))),
    h("p", { class: "notice comp-auto" }, tx(data.content_automatic)));
}

function roadmap(data) {
  return section("mejoras", "comp.h.next", "comp.next.lead",
    h("ol", { class: "comp-road" }, data.roadmap.map((item) => h("li", {},
      h("h3", {}, tx(item.title)),
      h("p", {}, tx(item.body))))));
}

function strengths(data) {
  return section("destacamos", "comp.h.strengths", "comp.strengths.lead",
    h("ul", { class: "comp-cards comp-strengths" }, data.strengths.map((item) => h("li", { class: "comp-card" },
      h("h3", {}, icon("check"), tx(item.title)),
      h("p", {}, tx(item.body))))));
}

// ---- Marks and rows shared by every chart (#295) ----

/** A level's mark: a shape and a word, so it never rests on colour alone. */
function mark(level) {
  const glyph = {
    yes: s("path", { d: "M6 10.4l2.7 2.7L14.2 7.4", class: "glyph" }),
    partial: s("path", { d: "M10 1.5a8.5 8.5 0 0 1 0 17z", class: "half" }),
    no: s("path", { d: "M6.4 10h7.2", class: "glyph" }),
    unknown: s("text", { x: 10, y: 14.2, "text-anchor": "middle", class: "q" }, "?"),
  }[level];
  return h("span", { class: `comp-mark is-${level}` },
    s("svg", { class: "comp-mark-icon", viewBox: "0 0 20 20", "aria-hidden": "true", focusable: "false" },
      s("circle", { cx: 10, cy: 10, r: 8.5, class: "ring" }), glyph),
    h("span", { class: "comp-mark-word" }, t(`comp.level.${level}`)));
}

/** The company cell that starts every row of every chart; ours carries «Nosotros». */
function companyCell(row) {
  return h("th", { scope: "row", role: "rowheader", class: "comp-co" },
    h("span", { class: "comp-co-name" }, row.label ?? (row.ours ? tx(row.name) : row.name)),
    row.ours ? h("span", { class: "comp-us" }, t("comp.us")) : null);
}

/** The column headers of a chart table, for screen readers (the chart shows them as layout). */
function chartHead(keys) {
  return h("thead", { role: "rowgroup", class: "sr-only" },
    h("tr", { role: "row" }, keys.map((key) => h("th", { role: "columnheader", scope: "col" }, t(key)))));
}

/** «Fuentes: …» with each source once, in order of first use. */
function sourcesOf(data, ids) {
  const line = sourceLine(data, [...new Set(ids.filter(Boolean))]);
  if (!line) return null;
  return h("details", { class: "comp-sources" }, h("summary", {}, t("comp.sources")), line);
}

// ---- Cost of the same service ----

let scenarioId = null;

/** Dollars with the thousands separator always (5.950 next to 12.500 on one chart). */
const grouped = (n) => new Intl.NumberFormat(lang, { maximumFractionDigits: 0, useGrouping: "always" }).format(Math.round(n));

/** A whole number of dollars a month, or a range. */
function money(low, high) {
  const lo = grouped(low);
  const hi = grouped(high);
  return lo === hi ? t("comp.cost.month", { v: lo }) : t("comp.cost.month", { v: `${lo}–${hi}` });
}

/** A range bar on a 0-100 scale: solid to `low`, lighter to `high`. */
function rangeBar(low, high, max, ours) {
  const a = Math.max(0.8, (low / max) * 100);
  const b = Math.max(a, (high / max) * 100);
  return s("svg", { class: `comp-bar comp-range${ours ? " is-ours" : ""}`, viewBox: "0 0 100 10", preserveAspectRatio: "none", "aria-hidden": "true" },
    s("rect", { x: 0, y: 0, width: 100, height: 10, class: "track" }),
    b > a ? s("rect", { x: a, y: 0, width: b - a, height: 10, class: "fill-range" }) : null,
    s("rect", { x: 0, y: 0, width: a, height: 10, class: "fill-low" }));
}

function costRows(data, sc) {
  const cost = data.cost;
  const ours = oursCost(cost, data.rag, sc);
  const oursNote = t("comp.cost.ours_note");
  const rows = [
    { row: { id: "ours-haiku", ours: true, label: t("comp.cost.ours_haiku") }, range: { low: ours.haiku, high: ours.haiku }, note: oursNote, sources: [] },
    { row: { id: "ours-opus", ours: true, label: t("comp.cost.ours_opus") }, range: { low: ours.opus, high: ours.opus }, note: oursNote, sources: [] },
  ];
  for (const row of companies(data).slice(1)) {
    const c = data.competitors.find((item) => item.id === row.id);
    const range = competitorCost(cost, row.id, sc);
    const spec = cost.competitors[row.id];
    if (range) rows.push({ row, range, note: tx(spec.note), sources: [c.source, ...(spec.also ?? [])] });
    else rows.push({ row, range: null, note: t(`comp.cost.other.${c.kind}`), price: c.basis === "none" ? t("comp.cost.unpublished") : null, sources: [c.source] });
  }
  return { ours, rows };
}

function costTable(data, sc) {
  const { ours, rows } = costRows(data, sc);
  const max = Math.max(...rows.filter((r) => r.range).map((r) => r.range.high));
  const table = h("table", { role: "table", class: "comp-chart comp-cost-chart" },
    h("caption", { class: "sr-only" }, t("comp.cost.caption", { n: sc.agents, hours: grouped(sc.hours) })),
    chartHead(["comp.col.company", "comp.cost.col.bar", "comp.cost.col.value"]),
    h("tbody", { role: "rowgroup" }, rows.map(({ row, range, note, price }) => h("tr", {
      role: "row", class: `${row.ours ? "is-ours" : ""}${range ? "" : " is-none"}`.trim() || null,
    },
    companyCell(row),
    h("td", { role: "cell", class: "comp-chart-bar" }, range ? rangeBar(range.low, range.high, max, row.ours) : h("span", { class: "comp-nobar" })),
    h("td", { role: "cell", class: "comp-chart-value" },
      range ? h("strong", { class: "comp-value" }, money(range.low, range.high)) : h("strong", { class: "comp-value is-muted" }, price ?? t("comp.cost.not_applicable")),
      row.ours ? h("span", { class: "badge warn comp-estimate" }, t("comp.cost.estimate")) : null,
      h("span", { class: "comp-note" }, note))))));
  return { table, ours, rows };
}

function costBody(data, sc) {
  const cost = data.cost;
  const { table, ours, rows } = costTable(data, sc);
  return [
    h("p", { class: "comp-scenario" }, icon("sessions"), t("comp.cost.line", { n: sc.agents, h: cost.hours_per_agent, hours: grouped(sc.hours), min: cost.call_minutes })),
    table,
    h("p", { class: "hint comp-assume" }, t("comp.cost.legend"), " ", t("comp.cost.assume", { answers: whole(ours.perHour.haiku.answersPerHour) }), " ", t("comp.cost.fair")),
    sourcesOf(data, [data.rag.price.source, cost.ours.live_source, cost.ours.analysis_source, cost.ours.analysis_seconds_source, ...rows.flatMap((r) => r.sources)]),
  ];
}

function costSection(data) {
  const cost = data.cost;
  if (!cost.scenarios.some((sc) => sc.id === scenarioId)) scenarioId = cost.scenarios[0].id;
  const body = h("div", { class: "comp-cost-body" });
  const buttons = cost.scenarios.map((sc) => h("button", {
    type: "button", class: "comp-switch-button", "data-scenario": sc.id,
    onclick: () => { scenarioId = sc.id; draw(); },
  }, icon("check"), t(`comp.cost.scenario.${sc.id}`, { n: sc.agents })));
  function draw() {
    const sc = cost.scenarios.find((item) => item.id === scenarioId);
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.scenario === scenarioId));
    body.replaceChildren(...costBody(data, scenarioOf(cost, sc)));
  }
  draw();
  return section("coste", "comp.h.cost", "comp.cost.lead",
    h("div", { class: "comp-switch", role: "group", "aria-label": t("comp.cost.switch") }, buttons),
    body);
}

// ---- One chart per task ----

function legend() {
  return h("div", { class: "comp-legend" },
    h("ul", { "aria-label": t("comp.legend") }, ["yes", "partial", "no", "unknown"].map((level) => h("li", {}, mark(level)))),
    h("p", { class: "hint" }, t("comp.level.unknown_means")));
}

function taskChart(data, task, rows) {
  const { n, of } = haveIt(task, data.competitors);
  const cells = rows.map((row) => ({ row, cell: cellOf(task, row.id) }));
  return h("article", { class: "comp-task", id: `tarea-${task.id}`, "aria-labelledby": `h-tarea-${task.id}` },
    h("header", { class: "comp-task-head" },
      h("h3", { id: `h-tarea-${task.id}` }, tx(task.title)),
      h("p", { class: "comp-task-q" }, tx(task.question)),
      h("p", { class: "comp-task-count" }, t("comp.task.count", { n, of }))),
    h("table", { role: "table", class: "comp-chart comp-matrix" },
      h("caption", { class: "sr-only" }, tx(task.title)),
      chartHead(["comp.col.company", "comp.task.col.answer", "comp.task.col.detail"]),
      h("tbody", { role: "rowgroup" }, cells.map(({ row, cell }) => h("tr", { role: "row", class: `${row.ours ? "is-ours " : ""}is-${cell.level}` },
        companyCell(row),
        h("td", { role: "cell", class: "comp-chart-mark" }, mark(cell.level)),
        h("td", { role: "cell", class: "comp-detail" }, cell.text ? tx(cell.text) : ""))))),
    sourcesOf(data, cells.map(({ cell }) => cell.source)));
}

function tasksSection(data) {
  const rows = companies(data);
  return section("tareas", "comp.h.tasks", "comp.tasks.lead",
    legend(),
    h("div", { class: "comp-tasks" }, data.tasks.map((task) => taskChart(data, task, rows))));
}

// ---- Us against one company ----

let pickedId = null;

function versusList(titleKey, name, items, cls) {
  if (items.length === 0) return null;
  return h("div", { class: `comp-vs-col ${cls}` },
    h("h4", {}, t(titleKey, { name })),
    h("ul", {}, items));
}

function versusPanel(data, c) {
  const v = versus(data, c.id);
  const byId = Object.fromEntries(data.tasks.map((task) => [task.id, task]));
  const item = (id) => {
    const task = byId[id];
    const theirs = cellOf(task, c.id);
    return h("li", {}, h("strong", {}, tx(task.title)),
      h("span", { class: "comp-vs-marks" },
        h("span", {}, t("comp.us"), " ", mark(cellOf(task, "ours").level)),
        h("span", {}, c.name, " ", mark(theirs.level))),
      theirs.text ? h("span", { class: "comp-vs-note" }, `${c.name}: ${tx(theirs.text)}`) : null);
  };
  const source = data.sources[c.source];
  const facts = h("dl", { class: "comp-vs-facts" },
    h("div", {}, h("dt", {}, t("comp.col.price")), h("dd", {}, tx(c.price))),
    h("div", {}, h("dt", {}, t("comp.col.users")), h("dd", {}, tx(c.users))),
    h("div", {}, h("dt", {}, t("comp.vs.where")), h("dd", {}, runsBadge(c.runs))));
  const theyMore = [...v.theyMore.map(item), ...(c.offers ?? []).map((offer) => h("li", {}, h("strong", {}, tx(offer))))];
  return [
    h("div", { class: "comp-vs-head" },
      h("h3", {}, c.name, " ", h("span", { class: "comp-vs-kind" }, t(`comp.kind.${c.kind}`))),
      facts,
      source ? h("p", { class: "hint comp-source" }, sourceLink(data, c.source), " ", t("comp.seen.on", { date: day(source.seen) })) : null),
    h("div", { class: "comp-vs-grid" },
      versusList("comp.vs.we_more", c.name, v.weMore.map(item), "is-yes"),
      versusList("comp.vs.unknown", c.name, v.unknown.map(item), "is-unknown"),
      versusList("comp.vs.they_more", c.name, theyMore, "is-they"),
      versusList("comp.vs.both", c.name, v.both.map(item), "is-both")),
    sourcesOf(data, [...data.tasks.map((task) => cellOf(task, c.id).source), ...(c.offers ?? []).map((offer) => offer.source)]),
  ];
}

function versusSection(data) {
  if (!data.competitors.some((c) => c.id === pickedId)) pickedId = data.competitors[0].id;
  const panel = h("div", { class: "comp-vs-panel", "aria-live": "polite" });
  const draw = () => panel.replaceChildren(...versusPanel(data, data.competitors.find((c) => c.id === pickedId)));
  const select = h("select", { id: "comp-pick", onchange: (event) => { pickedId = event.target.value; draw(); } },
    data.competitors.map((c) => h("option", { value: c.id, selected: c.id === pickedId }, c.name)));
  draw();
  return section("frente", "comp.h.vs", "comp.vs.lead",
    h("div", { class: "comp-pick" }, h("label", { for: "comp-pick" }, t("comp.vs.pick")), select),
    panel);
}

/** The page's contents: a row of links to its sections. */
function contents() {
  const items = [["destacamos", "comp.h.strengths"], ["coste", "comp.h.cost"], ["tareas", "comp.h.tasks"], ["frente", "comp.h.vs"],
    ["precios", "comp.h.market"], ["como", "comp.h.how"], ["rag", "comp.h.rag"], ["modelos", "comp.h.models"],
    ["contenido", "comp.h.content"], ["mejoras", "comp.h.next"]];
  return h("nav", { class: "comp-toc", "aria-label": t("comp.toc") },
    h("ul", {}, items.map(([id, key]) => h("li", {}, h("a", { href: `#${id}` }, t(key))))));
}

/** Draw the whole page: our strengths and the comparisons first, then how it works. */
export function renderCompetition(main, data) {
  main.replaceChildren(
    h("header", { class: "comp-hero" },
      h("h1", {}, t("comp.title")),
      h("p", { class: "comp-lead" }, t("comp.lead")),
      h("p", { class: "hint" }, t("comp.synthetic"))),
    contents(),
    strengths(data),
    costSection(data),
    tasksSection(data),
    versusSection(data),
    market(data),
    howItWorks(data),
    withWithout(data),
    models(data),
    content(data),
    roadmap(data),
    h("footer", { class: "comp-foot" },
      h("p", { class: "hint" }, t("comp.updated", { date: day(data.updated) }), " ", sourceLink(data, "page_doc", t("comp.how_updated")))));
}

/** A message in place of the page (loading, or the data could not be read). */
export function renderMessage(main, key, error = false) {
  main.replaceChildren(h("h1", {}, t("comp.title")), h("p", { class: error ? "notice error" : "hint" }, t(key)));
}
