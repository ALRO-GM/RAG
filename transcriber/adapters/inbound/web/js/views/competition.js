// The «Competencia» page's body (#291): how the system works (steps, where
// retrieval acts), the with/without retrieval estimate, which model does
// what, the market comparison, where we stand out and behind, how the
// content is kept up to date, and what comes next. Every figure and
// every sentence about a company or our system comes from
// competition/data.json; headings and labels from the dictionaries.
// Colours come from classes in style.css (the CSP allows no inline
// styles); text always goes in as text nodes.

import { h } from "../dom.js";
import { decimal, lang, t } from "../i18n.js";
import {
  KINDS, competitorsOf, digitsFor, maximumLibraries, ragMaxRows, ragRows, textOf,
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

const RUNS_KEY = { cloud: "comp.runs.cloud", mixed: "comp.runs.mixed" };

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
    h("td", { role: "cell", "data-label": t("comp.col.runs") }, h("span", { class: "comp-runs is-cloud" }, t(RUNS_KEY[c.runs] ?? "comp.runs.cloud"))));
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
  const strengths = h("ul", { class: "comp-cards" }, data.strengths.map((item) => h("li", { class: "comp-card" },
    h("h4", {}, icon("check"), tx(item.title)),
    h("p", {}, tx(item.body)))));
  const gaps = h("ul", { class: "comp-gaps" }, data.gaps.map((item) => h("li", {}, tx(item))));
  return section("competencia", "comp.h.market", "comp.market.lead",
    table,
    h("p", { class: "hint comp-assume" }, t("comp.table.note", { date: day(data.updated) })),
    h("h3", { id: "destacamos" }, t("comp.h.strengths")),
    strengths,
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

/** The page's contents: a row of links to its sections. */
function contents() {
  const items = [["como", "comp.h.how"], ["rag", "comp.h.rag"], ["modelos", "comp.h.models"],
    ["competencia", "comp.h.market"], ["contenido", "comp.h.content"], ["mejoras", "comp.h.next"]];
  return h("nav", { class: "comp-toc", "aria-label": t("comp.toc") },
    h("ul", {}, items.map(([id, key]) => h("li", {}, h("a", { href: `#${id}` }, t(key))))));
}

/** Draw the whole page. */
export function renderCompetition(main, data) {
  main.replaceChildren(
    h("header", { class: "comp-hero" },
      h("h1", {}, t("comp.title")),
      h("p", { class: "comp-lead" }, t("comp.lead")),
      h("p", { class: "hint" }, t("comp.synthetic"))),
    contents(),
    howItWorks(data),
    withWithout(data),
    models(data),
    market(data),
    content(data),
    roadmap(data),
    h("footer", { class: "comp-foot" },
      h("p", { class: "hint" }, t("comp.updated", { date: day(data.updated) }), " ", sourceLink(data, "page_doc", t("comp.how_updated")))));
}

/** A message in place of the page (loading, or the data could not be read). */
export function renderMessage(main, key, error = false) {
  main.replaceChildren(h("h1", {}, t("comp.title")), h("p", { class: error ? "notice error" : "hint" }, t(key)));
}
