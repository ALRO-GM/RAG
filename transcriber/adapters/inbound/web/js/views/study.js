// The study page's body: summary, test bench, separating (one bar list
// per corpus, best highlighted, with its table), labelling (judges,
// tagger, conversation events). Every number comes from study/data.json
// and every sentence from the dictionaries; colours come from classes in
// style.css (the CSP allows no inline styles).

import { h } from "../dom.js";
import { decimal, lang, percent, t } from "../i18n.js";
import {
  barKind, comboLabel, denoiserName, layoutBars, linear, points, rowById, sourceLabel,
  judgeHeadline, status, ticks,
} from "../lib/study.js";

const NS = "http://www.w3.org/2000/svg";

/** An SVG element with attributes and children (strings become text). */
function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null && v !== false) el.setAttribute(k, String(v));
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

/** A value that may be missing, formatted, or "n/a". */
function maybe(value, format) {
  return value === null || value === undefined ? t("study.na") : format(value);
}

const pct = (v) => percent(v, 1);
const pts = (fraction) => decimal(points(fraction), 1);
const signed = (v) => `${v > 0 ? "+" : ""}${decimal(v, 2)}`;

/** "Fuente: LAB.md §13.8.1" with a link to the section. */
function sourceLink(data, id) {
  const source = data.sources[id];
  if (!source) return null;
  return h("p", { class: "hint source" }, t("study.source"), " ",
    h("a", { href: source.url, rel: "noopener noreferrer" }, sourceLabel(source)));
}

/** A link to an issue of the repository, when the data file has its address. */
function issueLink(data, number) {
  const issue = data.issues?.[String(number)];
  const text = t("study.issue", { n: number });
  return issue ? h("a", { href: issue.url, rel: "noopener noreferrer" }, text) : text;
}

function corpusName(id) {
  return t(`study.corpus.${id}`);
}

/** The detail line under a chart: what the focused bar says. */
function detailBox() {
  return h("p", { class: "detail", role: "status", "aria-live": "polite" }, t("study.detail.hint"));
}

/** Make a bar keyboard-focusable and show its detail on hover, focus or tap. */
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

/**
 * A horizontal bar list on a 0-100 % scale: label above each bar, value
 * at its end. items: [{ label, value, kind, weak?, strong?, detail }].
 */
function barList(host, items, title, detail) {
  const W = Math.max(280, Math.floor(host.clientWidth || 320));
  const right = 54;
  const x = linear([0, 1], [0, W - right]);
  const { rows, height } = layoutBars(items.length);
  const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, class: "plot", role: "group", "aria-label": title });
  for (const tick of ticks(0, 1, 0.25)) {
    const px = Math.round(x(tick)) + 0.5;
    svg.append(s("line", { x1: px, x2: px, y1: 16, y2: height - 6, class: "grid" }));
    const anchor = tick === 0 ? "start" : "middle";
    svg.append(s("text", { x: px, y: 11, class: "tick", "text-anchor": anchor }, percent(tick, 0)));
  }
  items.forEach((item, i) => {
    const { labelY, barY } = rows[i];
    svg.append(s("text", { x: 0, y: labelY, class: item.strong ? "bar-label strong" : "bar-label" }, item.label));
    const w = Math.max(2, x(item.value) - x(0));
    const bar = s("rect", { x: 0.5, y: barY, width: w, height: 14, rx: 3, class: `bar study-${item.kind}${item.weak ? " weak small" : ""}` });
    interactive(bar, item.detail, detail);
    svg.append(bar);
    svg.append(s("text", { x: w + 6, y: barY + 11.5, class: "value" }, pct(item.value)));
  });
  host.replaceChildren(svg);
}

/** A legend: [{ kind, text, weak? }]. */
function legend(entries) {
  return h("ul", { class: "legend" }, entries.map(({ kind, text, weak }) => h("li", {},
    s("svg", { width: 14, height: 14, "aria-hidden": "true" },
      s("rect", { x: 1, y: 1, width: 12, height: 12, rx: 2, class: `bar study-${kind}${weak ? " weak small" : ""}` })),
    text)));
}

/** A table inside a closed <details>, scrolling sideways on a phone. */
function tableBox(caption, head, body) {
  return h("details", { class: "data-table" },
    h("summary", {}, t("study.table.show")),
    h("div", { class: "table-wrap" },
      h("table", { class: "data" },
        h("caption", {}, caption),
        h("thead", {}, h("tr", {}, head.map((c, i) => h("th", { scope: "col", class: i < 2 ? "left" : null }, c)))),
        h("tbody", {}, body))));
}

/** A chart host that redraws with the page's width. */
function chart(draws, draw) {
  const host = h("div", { class: "plot-host" });
  draws.push(() => draw(host));
  return host;
}

// ---------------------------------------------------------------- summary

function headlineCard(data, corpus) {
  const none = t("study.none");
  const head = corpus.headline;
  const best = rowById(corpus, head.best);
  const name = (id) => comboLabel(rowById(corpus, id), none, t("study.builtin"));
  const lines = [
    h("p", { class: "big-figure" }, pct(head.success)),
    h("p", {}, t("study.head.best", { name: comboLabel(best, none, t("study.builtin")), success: pct(head.success) })),
  ];
  if (head.builtin) {
    lines.push(h("p", {}, t("study.head.builtin", {
      pp: pts(head.builtin.points), red: percent(head.builtin.error_reduction, 0),
      name: name(head.builtin.id), success: pct(head.builtin.success),
    })));
  }
  if (head.next) {
    lines.push(h("p", {}, t("study.head.next", {
      pp: pts(head.next.points), red: percent(head.next.error_reduction, 0),
      name: name(head.next.id), success: pct(head.next.success),
    })));
  }
  if (head.best_commercial) {
    lines.push(h("p", { class: "hint" }, t("study.head.commercial", {
      name: name(head.best_commercial.id), success: pct(head.best_commercial.success),
    })));
  }
  return h("div", { class: "card headline" },
    h("h3", {}, t(`study.corpus.title.${corpus.id}`)), lines, sourceLink(data, corpus.source));
}

function judgeLine(judges) {
  const { best, next, first } = judgeHeadline(judges);
  if (!best) return null;
  const hosted = judges.find((j) => j.small_sample);
  const parts = [t("study.head.label", { model: best.model, acc: pct(best.accuracy), n: best.items })];
  if (next) parts.push(t("study.head.judge.next", { pp: pts(best.accuracy - next.accuracy), model: next.model, acc: pct(next.accuracy) }));
  if (first && first !== next) parts.push(t("study.head.judge.first", { pp: pts(best.accuracy - first.accuracy), model: first.model, acc: pct(first.accuracy) }));
  if (hosted) parts.push(t("study.head.judge.hosted", { acc: pct(hosted.accuracy), n: hosted.items }));
  return h("p", {}, parts.join(" "));
}

function summary(data) {
  const judges = data.labelling.judges;
  return h("section", { id: "resumen", "aria-labelledby": "h-resumen" },
    h("h2", { id: "h-resumen" }, t("study.h.summary")),
    h("p", {}, t("study.summary.what")),
    h("ol", { class: "pipeline" },
      ["denoise", "diarize", "words", "label"].map((k) => h("li", {}, t(`study.step.${k}`)))),
    h("p", {}, t("study.summary.why")),
    h("h3", {}, t("study.h.separating.head")),
    h("div", { class: "headlines" }, data.separation.corpora.map((c) => headlineCard(data, c))),
    h("h3", {}, t("study.h.labelling.head")),
    judgeLine(judges),
    sourceLink(data, "judges_local"),
    h("p", { class: "hint" }, t("study.summary.caveat")));
}

// ---------------------------------------------------------------- bench

function bench(data) {
  const b = data.bench;
  const corpora = data.separation.corpora;
  const combos = corpora[0]?.combinations ?? 0;
  const own = b.own_recording;
  return h("section", { id: "banco", "aria-labelledby": "h-banco" },
    h("h2", { id: "h-banco" }, t("study.h.bench")),
    h("h3", {}, t("study.h.corpora")),
    h("ul", {}, corpora.map((c) => h("li", {},
      t(`study.bench.corpus.${c.id}`, {
        n: c.recordings, speakers: c.speakers, minutes: Math.round(c.seconds / 60), licence: c.licence,
      }),
      c.non_commercial ? ` ${t("study.bench.nc")}` : "",
      " ",
      c.excluded_no_speech || c.excluded_invalid
        ? t("study.bench.excluded", { noise: c.excluded_no_speech, invalid: c.excluded_invalid })
        : t("study.bench.excluded.none")))),
    sourceLink(data, "lab_results"),
    h("p", {}, t("study.bench.suite", { denoisers: b.denoisers, diarizers: b.diarizers, combos })),
    h("p", {}, t("study.bench.own", {
      seconds: own.seconds, people: own.people, pipelines: own.pipelines,
      mean: decimal(own.mean_agreement, 3), best: decimal(own.best_agreement, 3),
    })),
    sourceLink(data, "lab_against_r1"),
    h("p", {}, t("study.bench.acts", { dialogues: b.acts_set.dialogues, sentences: b.acts_set.sentences.toLocaleString(lang) })),
    sourceLink(data, b.acts_set.source),
    h("h3", {}, t("study.h.metrics")),
    h("ul", {}, ["success", "missed", "fa", "conf", "speakers", "points", "acts", "events"]
      .map((k) => h("li", {}, t(`study.metric.${k}`)))),
    sourceLink(data, "lab_scores"),
    h("h3", {}, t("study.h.licences")),
    h("p", {}, t("study.licence.text")),
    sourceLink(data, "lab_licences"),
    h("h3", {}, t("study.h.limits")),
    h("ul", {}, ["small", "domain", "words", "synthetic"].map((k) => h("li", {}, t(`study.limit.${k}`, {
      meetings: corpora.find((c) => c.id === "ami")?.recordings ?? 0,
      calls: corpora.find((c) => c.id === "callhome")?.recordings ?? 0,
    })))),
    sourceLink(data, "lab_limits"));
}

// ---------------------------------------------------------------- separating

function rowDetail(row) {
  const name = comboLabel(row, t("study.none"), t("study.builtin"));
  let text = t("study.detail.row", {
    name, success: pct(row.success), der: decimal(row.der, 3), sd: maybe(row.der_sd, (v) => decimal(v, 3)), n: row.n,
    missed: maybe(row.missed, pct), fa: maybe(row.false_alarm, pct), conf: maybe(row.confusion, pct),
    spk: maybe(row.speaker_error, signed),
  });
  if (!row.best) text += ` ${t("study.detail.gain", { pp: pts(row.gain.points), red: maybe(row.gain.error_reduction, (v) => percent(v, 0)) })}`;
  return text;
}

function corpusTable(corpus) {
  const none = t("study.none");
  const body = corpus.rows.map((r) => h("tr", { class: `kind-${barKind(r)}` },
    h("td", { class: "left" }, String(r.rank)),
    h("th", { scope: "row" }, comboLabel(r, none, t("study.builtin"))),
    h("td", {}, pct(r.success)),
    h("td", {}, `${decimal(r.der, 3)} ± ${maybe(r.der_sd, (v) => decimal(v, 3))}`),
    h("td", {}, maybe(r.missed, pct)),
    h("td", {}, maybe(r.false_alarm, pct)),
    h("td", {}, maybe(r.confusion, pct)),
    h("td", {}, `${maybe(r.speaker_error, signed)} (${maybe(r.speaker_abs_error, (v) => decimal(v, 2))})`),
    h("td", {}, r.best ? "–" : pts(r.gain.points)),
    h("td", {}, r.best ? "–" : maybe(r.gain.error_reduction, (v) => percent(v, 0))),
    h("td", {}, String(r.n))));
  return tableBox(t("study.table.caption", { corpus: corpusName(corpus.id) }),
    ["rank", "combo", "success", "der", "missed", "fa", "conf", "spk", "points", "reduction", "n"].map((c) => t(`study.col.${c}`)),
    body);
}

function corpusChart(data, corpus, draws) {
  const none = t("study.none");
  const detail = detailBox();
  const title = t("study.chart.label", { corpus: corpusName(corpus.id) });
  const items = corpus.rows.map((r) => ({
    label: `${r.rank}. ${comboLabel(r, none, t("study.builtin"))}`, value: r.success, kind: barKind(r), strong: r.best, detail: rowDetail(r),
  }));
  return h("div", { class: "panel" },
    h("h3", {}, t(`study.corpus.title.${corpus.id}`)),
    legend([
      { kind: "best", text: t("study.legend.best") },
      { kind: "builtin", text: t("study.legend.builtin") },
      { kind: "other", text: t("study.legend.other") },
    ]),
    chart(draws, (host) => barList(host, items, title, detail)),
    detail,
    corpusTable(corpus),
    sourceLink(data, corpus.source));
}

function denoiserChart(data, draws) {
  const none = t("study.none");
  return h("div", { class: "panel" },
    h("h3", {}, t("study.h.denoiser")),
    h("p", {}, t("study.denoiser.intro", { n: data.bench.diarizers })),
    h("div", { class: "facets" }, data.separation.corpora.map((corpus) => {
      const detail = detailBox();
      const title = t("study.denoiser.chart", { corpus: corpusName(corpus.id) });
      const items = corpus.denoisers.map((d, i) => {
        const label = denoiserName(d, none);
        return {
          label, value: d.success, kind: i === 0 ? "best" : "other", strong: i === 0,
          detail: t("study.denoiser.detail", { name: label, success: pct(d.success), n: d.diarizers }),
        };
      });
      return h("div", {}, h("h4", {}, title), chart(draws, (host) => barList(host, items, title, detail)), detail);
    })),
    sourceLink(data, "lab_denoiser"));
}

function rule(data) {
  const block = data.separation.rule;
  const done = status(block) === "done";
  return h("div", { class: "panel" },
    h("h3", {}, t("study.h.rule")),
    h("p", {}, t("study.rule.text")),
    h("p", { class: done ? "callout" : "notice" },
      h("span", { class: `badge ${done ? "" : "warn"}` }, t(done ? "study.status.done" : "study.status.pending")), " ",
      issueLink(data, block.issue), " ",
      done ? t("study.rule.done") : t("study.rule.pending")),
    done ? h("ul", {}, Object.entries(block.corpora).map(([id, v]) => h("li", {},
      t("study.rule.row", { corpus: corpusName(id), success: pct(v.success), n: v.n })))) : null);
}

function separation(data, draws) {
  return h("section", { id: "separar", "aria-labelledby": "h-separar" },
    h("h2", { id: "h-separar" }, t("study.h.separate")),
    h("p", {}, t("study.sep.intro")),
    data.separation.corpora.map((c) => corpusChart(data, c, draws)),
    denoiserChart(data, draws),
    rule(data));
}

// ---------------------------------------------------------------- labelling

function judgeName(j) {
  return j.kind === "hosted" ? t("study.judge.hosted", { model: j.model }) : t("study.judge.local", { model: j.model, bits: j.bits });
}

function judgesPanel(data, draws) {
  const judges = data.labelling.judges;
  const top = judgeHeadline(judges).best;
  const detail = detailBox();
  const title = t("study.judges.chart");
  const items = judges.map((j) => ({
    label: j.small_sample ? `${judgeName(j)}: ${t("study.judge.small", { n: j.items })}` : judgeName(j),
    value: j.accuracy, kind: j === top ? "best" : "other", strong: j === top, weak: j.small_sample,
    detail: t("study.judge.detail", { name: judgeName(j), acc: pct(j.accuracy), n: j.items }),
  }));
  const confirm = judges.filter((j) => j.tags_confirmed !== null);
  return h("div", { class: "panel" },
    h("h3", {}, t("study.h.judges")),
    h("p", {}, t("study.label.intro")),
    legend([
      { kind: "best", text: t("study.legend.bestjudge", { n: top?.items ?? 0 }) },
      { kind: "other", text: t("study.legend.other") },
      { kind: "other", weak: true, text: t("study.legend.small") },
    ]),
    chart(draws, (host) => barList(host, items, title, detail)),
    detail,
    sourceLink(data, "judges_local"),
    sourceLink(data, "judges_both"),
    h("h3", {}, t("study.h.tags")),
    h("p", {}, t("study.tags.intro")),
    h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ["model", "confirmed", "rejected"].map((c, i) => h("th", { scope: "col", class: i === 0 ? "left" : null }, t(`study.col.${c}`))))),
      h("tbody", {}, confirm.map((j) => h("tr", {},
        h("th", { scope: "row" }, judgeName(j)),
        h("td", {}, String(j.tags_confirmed)),
        h("td", {}, String(j.tags_rejected))))))),
    sourceLink(data, "judges_local"));
}

function taggerPanel(data, draws) {
  const block = data.labelling.tagger;
  const done = status(block) === "done";
  const parts = [
    h("h3", {}, t("study.h.tagger")),
    h("p", { class: done ? "callout" : "notice" },
      h("span", { class: `badge ${done ? "" : "warn"}` }, t(done ? "study.status.done" : "study.status.pending")), " ",
      issueLink(data, block.issue), " ",
      t(done ? "study.tagger.done" : "study.tagger.pending")),
  ];
  if (done) {
    const detail = detailBox();
    const title = t("study.tagger.chart");
    const items = block.approaches.map((a, i) => ({
      label: a.approach, value: a.accuracy, kind: i === 0 ? "best" : "other", strong: i === 0,
      detail: t("study.judge.detail", { name: a.approach, acc: pct(a.accuracy), n: a.items }),
    }));
    parts.push(chart(draws, (host) => barList(host, items, title, detail)), detail);
  }
  return h("div", { class: "panel" }, parts);
}

function eventsPanel(data) {
  const ev = data.labelling.events;
  const g = ev.given_links;
  const inf = ev.inferred_links;
  const cell = (r) => `${r.derived} / ${maybe(r.precision, pct)} / ${maybe(r.recall, pct)}`;
  const byType = new Map(inf.rows.map((r) => [r.type, r]));
  return h("div", { class: "panel" },
    h("h3", {}, t("study.h.events")),
    h("p", {}, t("study.events.intro", { expected: g.expected, dialogues: ev.dialogues })),
    h("p", {}, t("study.events.summary", {
      p: maybe(g.precision, pct), r: maybe(g.recall, pct), pi: maybe(inf.precision, pct), ri: maybe(inf.recall, pct),
    })),
    tableBox(t("study.events.caption"),
      ["event", "expected", "given", "inferred"].map((c) => t(`study.col.${c}`)),
      [
        ...g.rows.map((r) => h("tr", {},
          h("th", { scope: "row" }, t(`study.event.${r.type}`)),
          h("td", {}, String(r.expected)),
          h("td", {}, cell(r)),
          h("td", {}, byType.has(r.type) ? cell(byType.get(r.type)) : t("study.na")))),
        h("tr", { class: "total" },
          h("th", { scope: "row" }, t("study.events.total")),
          h("td", {}, String(g.expected)),
          h("td", {}, cell(g)),
          h("td", {}, cell(inf))),
      ]),
    h("p", { class: "hint" }, t("study.events.limit", { n: ev.dialogues })),
    sourceLink(data, ev.source));
}

function labelling(data, draws) {
  return h("section", { id: "etiquetar", "aria-labelledby": "h-etiquetar" },
    h("h2", { id: "h-etiquetar" }, t("study.h.label")),
    judgesPanel(data, draws),
    taggerPanel(data, draws),
    eventsPanel(data));
}

// ---------------------------------------------------------------- page

/**
 * Draw the whole study into `main`. Returns a function that redraws the
 * charts (for a change of width).
 */
export function renderStudy(main, data) {
  const draws = [];
  main.replaceChildren(
    h("h1", {}, t("study.title")),
    h("p", { class: "meta" }, t("study.updated", { date: new Date(`${data.updated}T12:00:00`).toLocaleDateString(lang, { year: "numeric", month: "long", day: "numeric" }) })),
    h("nav", { class: "toc", "aria-label": t("study.toc.label") }, h("ol", {},
      [["resumen", "summary"], ["banco", "bench"], ["separar", "separate"], ["etiquetar", "label"]]
        .map(([id, key]) => h("li", {}, h("a", { href: `#${id}` }, t(`study.h.${key}`)))))),
    summary(data),
    bench(data),
    separation(data, draws),
    labelling(data, draws));
  const redraw = () => { for (const draw of draws) draw(); };
  redraw();
  return redraw;
}

/** Show a message instead of the study (loading, or the data is missing). */
export function renderMessage(main, key, error = false) {
  main.replaceChildren(h("h1", {}, t("study.title")), h("p", { class: error ? "notice error" : "hint" }, t(key)));
}
