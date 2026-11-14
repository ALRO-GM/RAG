// One lab run: the comparison overview, and the shell of the timeline,
// fragments and scores sections. The results (5-7 MB on a full run) are
// fetched once per run and kept while the tab stays on that run.

import { fill, h, svg } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  denoiserTable, heatClass, idPart, pairMatrix, sortRows,
} from "../lib/lab.js";
import { crumbs } from "./lab.js";
import { timelineSection } from "./lab-timeline.js";
import { fragmentsSection, scoresSection } from "./lab-review.js";

const SECTIONS = ["overview", "timeline", "fragments", "scores"];
const ENDED = new Set(["done", "partial"]);
const cache = { key: null, results: null, tracks: null };

/** Yield to the browser so a long build does not freeze the page. */
export function breathe() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function pct(value) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)} %`;
}

export function num(value, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

/** The run's data, fetched once: { results, tracks() }. */
export async function runData(ctx, session, run) {
  const key = `${session}/${run}`;
  if (cache.key !== key) {
    cache.key = key;
    cache.results = null;
    cache.tracks = null;
  }
  if (!cache.results) cache.results = ctx.api.labResults(session, run).catch((error) => { cache.results = null; throw error; });
  return {
    results: await cache.results,
    tracks() {
      if (!cache.tracks) cache.tracks = ctx.api.labTracks(session, run).catch((error) => { cache.tracks = null; throw error; });
      return cache.tracks;
    },
  };
}

export async function runView(ctx, root, session, run, section) {
  if (!SECTIONS.includes(section)) section = "overview";
  root.classList.add("wide");
  const base = `#/lab/${encodeURIComponent(session)}/${encodeURIComponent(run)}`;
  root.append(crumbs(session, run), h("h1", {}, t("lab.run_heading", { id: run.slice(0, 8) })),
    h("nav", { class: "tabs", "aria-label": t("lab.sections") }, SECTIONS.map((s) => h("a", {
      href: `${base}/${s}`, class: s === section ? "tab active" : "tab", "aria-current": s === section ? "page" : null,
    }, t(`lab.section.${s}`)))));
  const status = await ctx.api.labStatus(session, run);
  if (!ENDED.has(status.state)) {
    root.append(h("p", { class: "notice", role: "status" }, ["queued", "running"].includes(status.state)
      ? t("lab.in_progress", { done: status.done_steps, total: status.total_steps })
      : t("lab.no_results", { state: t(`lab.state.${status.state}`) })),
    h("a", { class: "button", href: `#/lab/${encodeURIComponent(session)}` }, t("lab.back_to_runs")));
    return () => root.classList.remove("wide");
  }
  const body = h("div", { class: "stack" }, h("p", { class: "notice", role: "status" }, t("lab.loading")));
  root.append(body);
  let data;
  try {
    data = await runData(ctx, session, run);
  } catch (error) {
    if (error.kind === "unauthorized" || error.kind === "unavailable") throw error;
    body.replaceChildren(h("p", { class: "error", role: "alert" }, describeError(error)));
    return () => root.classList.remove("wide");
  }
  let done = null;
  const env = { ctx, session, run, results: data.results, tracks: data.tracks, status };
  if (section === "overview") await overview(body, env);
  else if (section === "timeline") done = await timelineSection(body, env);
  else if (section === "fragments") done = await fragmentsSection(body, env);
  else done = await scoresSection(body, env);
  return () => {
    root.classList.remove("wide");
    done?.();
  };
}

// ---- overview -----------------------------------------------------------

const PIPELINE_COLUMNS = [
  ["id", "lab.col.pipeline", (v) => v],
  ["seconds", "lab.col.seconds", (v) => num(v)],
  ["peak_gpu_mb", "lab.col.gpu", (v) => num(v, 0)],
  ["speaker_count", "lab.col.speakers", (v) => (v ?? "—")],
  ["speech_seconds", "lab.col.speech", (v) => num(v)],
  ["overlap_seconds", "lab.col.overlap", (v) => num(v)],
  ["agreement_with_consensus", "lab.col.agreement", pct],
  ["words_on_consensus", "lab.col.words_agreement", pct],
];

/** A sortable table: columns [key, label key, format]; rows rendered lazily in chunks. */
export function sortableTable(rows, columns, { initial, descending = true, rowClass = () => null, caption = null } = {}) {
  let key = initial ?? columns[0][0];
  let desc = descending;
  const tbody = h("tbody");
  const heads = columns.map(([k, label]) => {
    const button = h("button", { type: "button", class: "sort" }, t(label));
    button.addEventListener("click", () => {
      desc = key === k ? !desc : k !== "id";
      key = k;
      fill();
    });
    return h("th", { scope: "col", "data-key": k }, button);
  });
  let generation = 0;
  async function fill() {
    const mine = ++generation;
    heads.forEach((th) => th.setAttribute("aria-sort", th.dataset.key === key ? (desc ? "descending" : "ascending") : "none"));
    const sorted = sortRows(rows, key, desc);
    tbody.replaceChildren();
    for (let i = 0; i < sorted.length; i += 60) {
      if (mine !== generation) return;
      tbody.append(...sorted.slice(i, i + 60).map((row) => h("tr", { class: rowClass(row) },
        columns.map(([k, , format], c) => (c === 0 ? h("th", { scope: "row" }, format(row[k], row)) : h("td", {}, format(row[k], row)))))));
      await breathe();
    }
  }
  fill();
  return h("div", { class: "table-wrap" }, h("table", { class: "data" },
    caption ? h("caption", {}, caption) : null, h("thead", {}, h("tr", {}, heads)), tbody));
}

function selector(id, label, values, current, onChange, show = (v) => v) {
  const select = h("select", { id }, values.map((v) => h("option", { value: v, selected: v === current }, show(v))));
  select.addEventListener("change", () => onChange(select.value));
  return h("span", { class: "row tight" }, h("label", { for: id }, label), select);
}

async function overview(body, env) {
  const { results } = env;
  const rows = results.pipelines.map((p) => ({ ...p, speaker_count: p.speakers?.length ?? null }));
  const nonCommercial = (row) => (row.licence_commercial === false ? "flag-nc" : null);
  const pipelines = sortableTable(rows, PIPELINE_COLUMNS, {
    initial: "agreement_with_consensus", rowClass: nonCommercial,
    caption: t("lab.pipelines_caption", { count: rows.length }),
  });
  fill(body,
    h("p", { class: "notice warn", role: "note" }, t("lab.agreement_note")),
    h("p", { class: "meta" }, t("lab.overview_meta", {
      duration: num(results.duration), pipelines: results.pipelines.length,
      tracks: results.tracks.length, fragments: results.fragment_count,
      coverage: pct(results.consensus?.coverage),
    })),
    h("h2", {}, t("lab.pipelines")), h("p", { class: "hint" }, t("lab.nc_hint")), pipelines);
  await breathe();
  body.append(heatSection(results), recognisersSection(results), effectSection(results));
}

// ---- heat map -------------------------------------------------------------

function heatSection(results) {
  const tracks = results.tracks.map((tr) => tr.id);
  const denoisers = idPart(tracks, 0);
  const diarizers = idPart(tracks, 1);
  let mode = "denoiser";
  let denoiser = denoisers.includes("none") ? "none" : denoisers[0];
  let diarizer = diarizers[0];
  let field = "agreement";
  const out = h("div", { class: "stack" });
  const readout = h("p", { class: "meta", role: "status", "aria-live": "polite" }, t("lab.heat_hint"));
  const controls = h("div", { class: "row wrap" });

  function draw() {
    const ids = mode === "denoiser"
      ? tracks.filter((id) => id.startsWith(`${denoiser}+`))
      : tracks.filter((id) => id.endsWith(`+${diarizer}`));
    const short = (id) => (mode === "denoiser" ? id.slice(denoiser.length + 1) : id.slice(0, id.length - diarizer.length - 1));
    const matrix = pairMatrix(results.diarization, ids, field, field === "error_rate" ? 0 : 1);
    const cell = 40;
    const pad = 150;
    const top = 110;
    const width = pad + ids.length * cell + 90;
    const height = top + ids.length * cell;
    const map = svg("svg", { class: "heat", viewBox: `0 0 ${width} ${height}`, width, height, role: "group", "aria-label": t("lab.heat_title") });
    ids.forEach((a, i) => {
      map.append(svg("text", { x: pad - 6, y: top + i * cell + cell * 0.6, "text-anchor": "end", class: "axis" }, short(a)));
      map.append(svg("text", { x: pad + i * cell + cell * 0.65, y: top - 6, transform: `rotate(-45 ${pad + i * cell + cell * 0.65} ${top - 6})`, class: "axis" }, short(a)));
      ids.forEach((b, j) => {
        const value = matrix.value(a, b);
        const shown = field === "error_rate" ? num(value, 2) : pct(value);
        const text = t("lab.heat_cell", { a, b, value: shown });
        const heat = field === "error_rate" && value !== null ? 1 - Math.min(1, value) : value;
        map.append(svg("rect", {
          x: pad + j * cell, y: top + i * cell, width: cell - 2, height: cell - 2, class: `cell ${heatClass(heat)}`,
          tabindex: 0, "aria-label": text,
          onFocus: () => { readout.textContent = text; }, onMouseenter: () => { readout.textContent = text; },
        }, svg("title", {}, text)));
      });
    });
    const table = h("table", { class: "data compact" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, t("lab.heat_ref")), ids.map((b) => h("th", { scope: "col" }, short(b))))),
      h("tbody", {}, ids.map((a) => h("tr", {}, h("th", { scope: "row" }, short(a)),
        ids.map((b) => h("td", {}, field === "error_rate" ? num(matrix.value(a, b), 2) : pct(matrix.value(a, b))))))));
    out.replaceChildren(h("div", { class: "heat-wrap" }, map), readout,
      h("details", {}, h("summary", {}, t("lab.as_table")), h("div", { class: "table-wrap" }, table)));
  }

  function renderControls() {
    controls.replaceChildren(
      selector("lab-heat-mode", t("lab.heat_mode"), ["denoiser", "diarizer"], mode, (v) => { mode = v; renderControls(); draw(); }, (v) => t(`lab.heat_mode.${v}`)),
      mode === "denoiser"
        ? selector("lab-heat-dn", t("lab.denoiser"), denoisers, denoiser, (v) => { denoiser = v; draw(); })
        : selector("lab-heat-dz", t("lab.diarizer"), diarizers, diarizer, (v) => { diarizer = v; draw(); }),
      selector("lab-heat-field", t("lab.heat_field"), ["agreement", "name_agreement", "error_rate"], field, (v) => { field = v; draw(); }, (v) => t(`lab.field.${v}`)));
  }
  renderControls();
  draw();
  return h("section", { class: "stack" }, h("h2", {}, t("lab.heat_title")), h("p", { class: "hint" }, t("lab.heat_about")), controls, out);
}

// ---- recognisers -----------------------------------------------------------

function recognisersSection(results) {
  const ids = results.transcripts.map((tr) => tr.id);
  const denoisers = idPart(ids, 0);
  let denoiser = denoisers.includes("none") ? "none" : denoisers[0];
  const out = h("div", { class: "table-wrap" });
  const words = new Map(results.transcripts.map((tr) => [tr.id, tr.words]));

  function draw() {
    const chosen = ids.filter((id) => id.startsWith(`${denoiser}+`));
    const short = (id) => id.slice(denoiser.length + 1);
    out.replaceChildren(h("table", { class: "data compact" },
      h("caption", {}, t("lab.wer_caption")),
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, t("lab.heat_ref")), h("th", { scope: "col" }, t("lab.col.words")),
        chosen.map((b) => h("th", { scope: "col" }, short(b))))),
      h("tbody", {}, chosen.map((a) => h("tr", {}, h("th", { scope: "row" }, short(a)), h("td", {}, words.get(a) ?? "—"),
        chosen.map((b) => h("td", {}, a === b ? "—" : pct(directed(results.recognition, a, b)))))))));
  }
  if (!ids.length) return h("section", {}, h("h2", {}, t("lab.recognisers")), h("p", { class: "hint" }, t("lab.no_recognisers")));
  draw();
  return h("section", { class: "stack" }, h("h2", {}, t("lab.recognisers")), h("p", { class: "hint" }, t("lab.wer_about")),
    h("div", { class: "row wrap" }, selector("lab-rec-dn", t("lab.denoiser"), denoisers, denoiser, (v) => { denoiser = v; draw(); })), out);
}

/** WER of b with a as reference (the pairs are directed). */
function directed(rows, a, b) {
  const row = rows.find((r) => r.reference === a && r.hypothesis === b);
  return row ? row.wer : null;
}

// ---- denoiser effect ---------------------------------------------------------

function effectTable(table, format, caption) {
  if (!table.rows.length) return null;
  return h("div", { class: "table-wrap" }, h("table", { class: "data compact" }, h("caption", {}, caption),
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, t("lab.denoiser")), table.columns.map((c) => h("th", { scope: "col" }, c)))),
    h("tbody", {}, table.rows.map((row) => h("tr", {}, h("th", { scope: "row" }, row.denoiser),
      table.columns.map((c) => h("td", {}, format(row.cells[c]))))))));
}

function effectSection(results) {
  const effects = results.denoiser_effect ?? [];
  return h("section", { class: "stack" }, h("h2", {}, t("lab.effect")), h("p", { class: "hint" }, t("lab.effect_about")),
    effectTable(denoiserTable(effects, "diarize", "agreement"), pct, t("lab.effect_diarize")),
    effectTable(denoiserTable(effects, "diarize", "consensus_change"), (v) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}`), t("lab.effect_consensus")),
    effectTable(denoiserTable(effects, "recognise", "wer"), pct, t("lab.effect_recognise")),
    effects.length ? null : h("p", { class: "hint" }, t("lab.no_effect")));
}
