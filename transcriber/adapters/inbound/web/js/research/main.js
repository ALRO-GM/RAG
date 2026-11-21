// The research page: draws the charts from research/data.json as inline
// SVG and adds the filter to the reference list. Without this script (or
// without the data) the page still shows every figure in its tables.
// Colours and dashes come from classes in style.css (the CSP allows no
// inline styles); text always goes in as text nodes.

import {
  linear, ticks, fmt, panelsOf, figuresOf, groupBy, verifiedClass, layoutRows,
  noteLine, figureDetail, entryDetail, filterSources, familiesOf, entriesOf, clip,
  spearmanBrown, situationsFor,
} from "./charts.js";
import { lang, translateStatic } from "../i18n.js";
import { installLanguageSwitch } from "../lang-switch.js";
import { installShell } from "../shell.js";

const NS = "http://www.w3.org/2000/svg";

/** An SVG element with attributes and children (strings become text). */
function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null && v !== false) el.setAttribute(k, String(v));
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

/** An HTML element with attributes and children (strings become text). */
function e(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null && v !== false) el.setAttribute(k, String(v));
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

/** Make a mark keyboard-focusable and show its detail on hover or focus. */
function interactive(mark, text, detail) {
  mark.setAttribute("tabindex", "0");
  mark.setAttribute("role", "img");
  mark.setAttribute("aria-label", text);
  mark.append(s("title", {}, text));
  const show = () => { detail.textContent = text; };
  mark.addEventListener("mouseenter", show);
  mark.addEventListener("focus", show);
  mark.addEventListener("click", show);
}

function widthOf(el) {
  return Math.max(280, Math.floor(el.clientWidth || 320));
}

/** The legend of the verification levels (solid, dashed, dotted). */
function verificationLegend(levels) {
  const item = (cls, text) => e("li", {}, s("svg", { width: 26, height: 12, "aria-hidden": "true" },
    s("rect", { x: 1, y: 1, width: 24, height: 10, rx: 2, class: `bar s0 ${cls}` })), text);
  return e("ul", { class: "legend" },
    item("solid", levels["full-text"]),
    item("weak abstract", levels["abstract-only"]),
    item("weak secondary", levels.secondary));
}

/** A series legend: [{ cls, text }]. */
function seriesLegend(entries) {
  return e("ul", { class: "legend" }, entries.map(({ cls, text }) => e("li", {},
    s("svg", { width: 14, height: 14, "aria-hidden": "true" }, s("rect", { x: 1, y: 1, width: 12, height: 12, rx: 2, class: `bar ${cls} solid` })),
    text)));
}

/**
 * Horizontal bars, label above each bar, value at its end, note below.
 * rows: [{ group?, figure, series }]; domain [0, max].
 */
function barList(host, rows, { max = 1, labels, detail, step = 0.25 }) {
  const W = widthOf(host);
  const right = 44;
  const x = linear([0, max], [0, W - right]);
  const { items, height } = layoutRows(rows);
  const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, class: "plot", role: "group" });
  for (const t of ticks(0, max, step)) {
    const px = Math.round(x(t)) + 0.5;
    svg.append(s("line", { x1: px, x2: px, y1: 16, y2: height - 4, class: "grid" }));
    svg.append(s("text", { x: Math.min(Math.max(px, 6), W - right), y: 11, class: "tick", "text-anchor": "middle" }, fmt(t)));
  }
  const chars = Math.floor((W - 8) / 6.6);
  for (const it of items) {
    if (it.kind === "group") {
      svg.append(s("text", { x: 0, y: it.y, class: "group-label" }, clip(it.text, chars)));
      continue;
    }
    const f = it.figure;
    svg.append(s("text", { x: 0, y: it.labelY, class: "bar-label" }, clip(f.label ?? f.method, chars)));
    const w = Math.max(2, x(f.value) - x(0));
    const bar = s("rect", { x: 0.5, y: it.barY, width: w, height: 14, rx: 3, class: `bar ${it.series ?? "s0"} ${verifiedClass(f.verified)}` });
    interactive(bar, figureDetail(f, labels.verified), detail);
    svg.append(bar);
    svg.append(s("text", { x: w + 6, y: it.barY + 11.5, class: "value" }, fmt(f.value)));
    if (it.noteY !== null) svg.append(s("text", { x: 0, y: it.noteY, class: "note" }, clip(typeof it.note === "string" ? it.note : noteLine(f), chars + 6)));
  }
  return svg;
}

/** A panel: its heading, how to read it, the plot and a legend. */
function panel(title, how, plot, ...legends) {
  return e("div", { class: "panel" }, e("h4", {}, title), how ? e("p", { class: "hint" }, how) : null, plot, legends);
}

function detailBox() {
  return e("p", { class: "detail", role: "status", "aria-live": "polite" }, "Seleccione una barra o un punto para ver su detalle.");
}

// ---------------------------------------------------------------- chart 1

const CHANNEL_SERIES = { text: "s1", audio: "s2", av: "s3" };

function drawReliability(host, data) {
  const labels = data.labels;
  const out = [];
  for (const p of panelsOf(data, "reliability")) {
    const figs = figuresOf(data, p.id);
    const box = e("div", { class: "plot-host" });
    const detail = detailBox();
    const parts = [];
    let rows;
    if (p.id === "channel") {
      rows = groupBy(figs, (f) => f.trait).flatMap((g) => g.members.map((f) => ({
        group: g.name, series: CHANNEL_SERIES[f.channel], note: `${f.sample} · ${f.cite}`,
        figure: { ...f, label: labels.channel[f.channel] },
      })));
      parts.push(seriesLegend(Object.entries(CHANNEL_SERIES).map(([k, cls]) => ({ cls, text: labels.channel[k] }))));
    } else if (p.id === "machine") {
      rows = groupBy(figs, (f) => f.study).flatMap((g) => g.members.map((f) => ({
        group: g.name, series: f.who === "machine" ? "s2" : "s1", figure: f,
      })));
      parts.push(seriesLegend([{ cls: "s1", text: "Humanos entre sí" }, { cls: "s2", text: "Máquina frente a humanos" }]));
    } else {
      rows = figs.map((f) => ({ series: "s1", figure: f }));
    }
    out.push({ p, box, rows, parts, detail });
  }
  const render = () => {
    for (const { box, rows, detail } of out) box.replaceChildren(barList(box, rows, { labels, detail }));
  };
  host.replaceChildren(
    verificationLegend(labels.verified),
    ...out.map(({ p, box, parts, detail }) => panel(`${p.title} · ${p.statistic}`, p.how, box, ...parts, detail)),
  );
  return render;
}

// ---------------------------------------------------------------- chart 2

const MODE_SERIES = { behaviour: "s1", interview: "s2", simulation: "s3", test: "s0", "self-report": "s0" };

function drawValidity(host, data) {
  const detail = detailBox();
  const labels = data.labels;
  const figs = [...figuresOf(data, "validity")].sort((a, b) => b.value - a.value);
  const rows = figs.map((f) => ({
    series: MODE_SERIES[f.mode], note: `${labels.mode[f.mode]} · DT de ρ ${fmt(f.sd)}`,
    figure: { ...f, label: f.method },
  }));
  const box = e("div", { class: "plot-host" });
  host.replaceChildren(
    seriesLegend([
      { cls: "s1", text: "Observa conducta (muestra de trabajo, assessment centre)" },
      { cls: "s2", text: "Entrevista" },
      { cls: "s3", text: "Simulación escrita o en vídeo" },
      { cls: "s0", text: "Test o autoinforme" },
    ]),
    verificationLegend(labels.verified),
    e("p", { class: "hint" }, "Todas las barras: ρ operativa, Sackett et al. (2022), Tabla 3."),
    box, detail,
  );
  return () => box.replaceChildren(barList(box, rows, { max: 0.5, step: 0.1, labels, detail }));
}

// ---------------------------------------------------------------- chart 3

const FAMILY_SERIES = ["s1", "s2", "s3", "s4", "s5", "s6"];

function marker(cx, cy, confidence, cls) {
  if (confidence === "High") return s("circle", { cx, cy, r: 5.5, class: cls });
  if (confidence === "Medium") return s("rect", { x: cx - 5, y: cy - 5, width: 10, height: 10, class: cls });
  return s("path", { d: `M${cx} ${cy - 6.5} L${cx + 6} ${cy + 4.5} L${cx - 6} ${cy + 4.5} Z`, class: cls });
}

function scatter(host, entries, series, labels, detail) {
  const W = widthOf(host);
  const H = Math.round(Math.min(W * 0.8, 320));
  const m = { l: 34, r: 12, t: 12, b: 34 };
  const x = linear([0, 100], [m.l, W - m.r]);
  const y = linear([0, 100], [H - m.b, m.t]);
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "plot", role: "group" });
  for (const t of [0, 25, 50, 75, 100]) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "grid" }));
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "grid" }));
    svg.append(s("text", { x: x(t), y: H - m.b + 14, class: "tick", "text-anchor": "middle" }, t));
    svg.append(s("text", { x: m.l - 6, y: y(t) + 4, class: "tick", "text-anchor": "end" }, t));
  }
  svg.append(s("text", { x: (m.l + W - m.r) / 2, y: H - 4, class: "axis-label", "text-anchor": "middle" }, "A: credibilidad de la evidencia (0-100)"));
  svg.append(s("text", { x: 10, y: (m.t + H - m.b) / 2, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 10 ${(m.t + H - m.b) / 2})` }, "C: encaje (0-100)"));
  const chars = Math.floor(W / 7);
  for (const en of entries) {
    const g = s("g", {});
    const mark = marker(x(en.A), y(en.C), en.confidence, `pt ${series}`);
    g.append(mark);
    interactive(g, entryDetail(en, labels), detail);
    svg.append(g);
    if (en.C >= 93) {
      const left = x(en.A) > W / 2;
      svg.append(s("text", { x: x(en.A) + (left ? -9 : 9), y: y(en.C) + 4, class: "direct", "text-anchor": left ? "end" : "start" }, clip(en.name, Math.min(chars, 28))));
    }
  }
  return svg;
}

function drawMap(host, data) {
  const detail = detailBox();
  const labels = data.labels;
  const families = Object.keys(labels.family);
  const facets = families.map((fam, i) => ({ fam, series: FAMILY_SERIES[i], box: e("div", { class: "plot-host" }) }));
  const shapes = e("ul", { class: "legend" },
    [["High", "Confianza alta"], ["Medium", "Confianza media"], ["Low", "Confianza baja"]].map(([c, text]) =>
      e("li", {}, s("svg", { width: 16, height: 16, "aria-hidden": "true" }, marker(8, 8.5, c, "pt s0")), text)));
  host.replaceChildren(shapes,
    e("div", { class: "facets" }, facets.map(({ fam, box }) => {
      const n = data.entries.filter((x) => x.family === fam).length;
      return panel(`${labels.family[fam]} (${n})`, null, box);
    })),
    detail);
  return () => {
    for (const { fam, series, box } of facets) box.replaceChildren(scatter(box, entriesOf(data.entries, fam), series, labels, detail));
  };
}

// ---------------------------------------------------------------- chart 4

function situationsPlot(host, data, detail) {
  const W = widthOf(host);
  const H = Math.round(Math.min(W * 0.75, 300));
  const m = { l: 34, r: 14, t: 12, b: 36 };
  const proj = data.projection;
  const maxN = proj.points.length;
  const x = linear([1, maxN], [m.l + 6, W - m.r]);
  const y = linear([0, 1], [H - m.b, m.t]);
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "plot", role: "group" });
  for (const t of ticks(0, 1, 0.2)) {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "grid" }));
    svg.append(s("text", { x: m.l - 6, y: y(t) + 4, class: "tick", "text-anchor": "end" }, fmt(t)));
  }
  for (let n = 1; n <= maxN; n += 1) svg.append(s("text", { x: x(n), y: H - m.b + 14, class: "tick", "text-anchor": "middle" }, n));
  svg.append(s("text", { x: (m.l + W - m.r) / 2, y: H - 4, class: "axis-label", "text-anchor": "middle" }, "Situaciones (ejercicios) promediadas"));
  for (const target of [0.7, 0.8]) {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(target), y2: y(target), class: "target" }));
    const n = situationsFor(proj.single, target);
    svg.append(s("text", { x: m.l + 4, y: y(target) - 4, class: "note" }, `${fmt(target)}: unas ${n} situaciones (proyección)`));
  }
  const pts = proj.points.map((p) => `${x(p.situations)},${y(spearmanBrown(proj.single, p.situations))}`).join(" ");
  const line = s("polyline", { points: pts, class: "projection" });
  interactive(line, `Proyección propia con Spearman-Brown a partir de .65 con 3 ejercicios: una situación ${fmt(proj.single)}, 4 situaciones ${fmt(spearmanBrown(proj.single, 4))}, 7 situaciones ${fmt(spearmanBrown(proj.single, 7))}. No es un dato publicado.`, detail);
  svg.append(line);
  for (const f of figuresOf(data, "exercises")) {
    const g = s("g", {});
    g.append(s("circle", { cx: x(f.situations), cy: y(f.value), r: 6, class: `pt s1 ${verifiedClass(f.verified)}` }));
    interactive(g, figureDetail(f, data.labels.verified), detail);
    svg.append(g);
    svg.append(s("text", { x: x(f.situations) + 10, y: y(f.value) + 4, class: "direct" }, `${fmt(f.value)} ${clip(f.label, Math.floor(W / 8))}`));
  }
  return svg;
}

function drawSituations(host, data) {
  const detail = detailBox();
  const labels = data.labels;
  const left = e("div", { class: "plot-host" });
  const right = e("div", { class: "plot-host" });
  const [pEx, pRp] = panelsOf(data, "situations");
  const rows = groupBy(figuresOf(data, "roleplays"), (f) => (f.situations === 1 ? "1 juego de rol (media de los 18)" : "18 juegos de rol agregados"))
    .flatMap((g) => g.members.map((f) => ({ group: g.name, series: f.who === "machine" ? "s2" : "s1", figure: f })));
  host.replaceChildren(
    e("div", { class: "facets" },
      panel(`${pEx.title} · ${pEx.statistic}`, pEx.how, left,
        e("ul", { class: "legend" },
          e("li", {}, s("svg", { width: 16, height: 16, "aria-hidden": "true" }, s("circle", { cx: 8, cy: 8, r: 6, class: "pt s1 solid" })), "Dato publicado"),
          e("li", {}, s("svg", { width: 26, height: 12, "aria-hidden": "true" }, s("line", { x1: 1, x2: 25, y1: 6, y2: 6, class: "projection" })), "Proyección propia, no es un dato"))),
      panel(`${pRp.title} · ${pRp.statistic}`, pRp.how, right,
        seriesLegend([{ cls: "s1", text: "Evaluadores humanos" }, { cls: "s2", text: "Máquina o evaluador + máquina" }]),
        verificationLegend(labels.verified))),
    detail);
  return () => {
    left.replaceChildren(situationsPlot(left, data, detail));
    right.replaceChildren(barList(right, rows, { labels, detail }));
  };
}

// ---------------------------------------------------------------- references

function enableFilters(data) {
  const box = document.getElementById("ref-filters");
  const list = document.getElementById("ref-list");
  if (!box || !list) return;
  const search = document.getElementById("ref-search");
  const family = document.getElementById("ref-family");
  const level = document.getElementById("ref-verified");
  const count = document.getElementById("ref-count");
  for (const f of familiesOf(data.sources, data.labels.family)) family.append(e("option", { value: f }, data.labels.family[f]));
  const apply = () => {
    const keep = new Set(filterSources(data.sources, { query: search.value, family: family.value, verified: level.value }));
    for (const li of list.children) li.hidden = !keep.has(li.id.replace(/^ref-/, ""));
    count.textContent = `${keep.size} de ${data.sources.length} fuentes`;
  };
  for (const el of [search, family, level]) el.addEventListener("input", apply);
  box.hidden = false;
  apply();
}

// ---------------------------------------------------------------- start

const DRAW = { reliability: drawReliability, validity: drawValidity, map: drawMap, situations: drawSituations };

async function start() {
  let data;
  try {
    const response = await fetch("research/data.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(String(response.status));
    data = await response.json();
  } catch {
    const status = document.getElementById("charts-status");
    status.textContent = "No se pudieron cargar los datos de los gráficos. Las tablas de debajo de cada gráfico tienen las mismas cifras.";
    status.hidden = false;
    return;
  }
  const renders = [];
  for (const host of document.querySelectorAll("[data-chart]")) {
    const draw = DRAW[host.dataset.chart];
    if (!draw) continue;
    renders.push(draw(host, data));
    const table = host.parentElement.querySelector("details.data-table");
    if (table) table.open = false;
  }
  const renderAll = () => { for (const r of renders) r(); };
  renderAll();
  let lastWidth = document.documentElement.clientWidth;
  let timer = null;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const w = document.documentElement.clientWidth;
      if (w !== lastWidth) { lastWidth = w; renderAll(); }
    }, 150);
  });
  enableFilters(data);
}

/** The header follows the chosen language; the article itself is only in Spanish. */
function applyLanguage() {
  translateStatic(document);
  const note = document.getElementById("only-spanish");
  if (note) note.hidden = lang !== "en";
}

applyLanguage();
installLanguageSwitch(document.querySelector(".top .lang"), applyLanguage);
installShell({ onLanguage: applyLanguage });
start();
