// The start of "Pruebas": how the tests are going across every session (a
// stacked bar of each newest run's steps, grouped by corpus), the results so
// far (DER over the public corpora, agreement on own recordings), what is
// pending with the reason, and short audio samples beside their figures.
// Refreshes every 10 s while a run is running and the page is visible.

import { fill, h, svg } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import {
  ACTIVE_RUNS, EXCERPT_SECONDS, OWN, POLL_MS, agreementByDiarizer, anyActive, axisTicks, barSegments, chooseExcerpt,
  corpusOf, derByDenoiser, derByDiarizer, groupByCorpus, linearScale, niceMax, overviewTotals, pendingGroups,
  sampleRows, stepTotal, summaryDenoisers, tallyOf, turnsToSpans,
} from "../lib/lab-progress.js";
import { num, pct } from "./lab-run.js";
import { stepLabel } from "./lab.js";

function corpusName(corpus) {
  return ["own", "ami", "callhome"].includes(corpus) ? t(`lab.corpus.${corpus}`) : corpus;
}

function sessionLink(entry) {
  return h("a", { href: `#/lab/${encodeURIComponent(entry.session_id)}` }, entry.title || entry.session_id);
}

function segmentLabel(segment) {
  if (segment.kind !== "failed") return t(`lab.progress.legend.${segment.kind}`);
  // "failed" is the runners' generic code: say "other error" rather than "failed: failed"
  return ["failed", "unknown"].includes(segment.code) ? t("lab.progress.legend.failed_other") : t("lab.progress.legend.failed", { code: segment.code });
}

function segmentClass(segment) {
  return segment.kind === "failed" ? `seg seg-failed${segment.code === "out_of_memory" ? " seg-oom" : ""}` : `seg seg-${segment.kind}`;
}

// ---- progress -----------------------------------------------------------------

function stepsBar(entry) {
  const tally = tallyOf(entry);
  const label = t("lab.progress.bar", { session: entry.title || entry.session_id, done: tally.done, running: tally.running, pending: tally.pending, failed: tally.failed });
  const bar = svg("svg", { class: "stepbar", viewBox: "0 0 100 6", preserveAspectRatio: "none", role: "img", "aria-label": label });
  bar.append(svg("rect", { x: 0, y: 0, width: 100, height: 6, class: "seg seg-track" }));
  for (const segment of barSegments(tally)) {
    bar.append(svg("rect", { x: segment.start * 100, y: 0, width: segment.width * 100, height: 6, class: segmentClass(segment) },
      svg("title", {}, `${segmentLabel(segment)}: ${segment.count}`)));
  }
  return bar;
}

function legend(entries) {
  const kinds = [{ kind: "done" }, { kind: "running" }, { kind: "pending" }];
  const codes = new Map();
  for (const entry of entries) for (const s of barSegments(tallyOf(entry))) if (s.kind === "failed") codes.set(s.code, s);
  const items = [...kinds, ...[...codes.values()].sort((a, b) => a.code.localeCompare(b.code))];
  return h("ul", { class: "legend" }, items.map((item) => h("li", {},
    svg("svg", { width: 14, height: 14, viewBox: "0 0 14 14", "aria-hidden": "true" }, svg("rect", { x: 0, y: 0, width: 14, height: 14, class: segmentClass(item) })),
    segmentLabel(item))));
}

function progressBlock(entries, updated) {
  if (!entries.length) return [h("p", { class: "hint" }, t("lab.progress.empty"))];
  const totals = overviewTotals(entries);
  const active = anyActive(entries);
  return [
    h("p", { class: "totals" }, t("lab.progress.totals", { done: totals.done, left: totals.left, failed: totals.failed, running: totals.runsRunning })),
    h("p", { class: "hint" }, t("lab.progress.updated", { time: updated.toLocaleTimeString(lang) }), active ? ` · ${t("lab.progress.auto")}` : ""),
    legend(entries),
    groupByCorpus(entries).map(({ corpus, sessions }) => h("section", { class: "stack tight" },
      h("h3", {}, corpusName(corpus)),
      h("ul", { class: "plain progress-rows" }, sessions.map((entry) => {
        const tally = tallyOf(entry);
        return h("li", { class: `progress-row state-${entry.run_state || "none"}` },
          h("div", { class: "row wrap tight" }, sessionLink(entry),
            entry.run_state ? h("span", { class: "badge" }, t(`lab.state.${entry.run_state}`)) : h("span", { class: "badge muted" }, t("lab.progress.not_started")),
            entry.run_id ? h("span", { class: "meta" }, t("lab.progress.counts", { done: tally.done, total: stepTotal(tally) })) : null),
          entry.run_id ? stepsBar(entry) : null);
      })))),
  ];
}

// ---- results -------------------------------------------------------------------

/** A horizontal bar chart: rows {id, mean, min, max, n}; whiskers from min to max when given. */
export function rangeChart(rows, { title, format, max = null }) {
  const shown = rows.filter((r) => Number.isFinite(r.mean));
  if (!shown.length) return h("p", { class: "hint" }, t("lab.results.none"));
  const top = max ?? niceMax(Math.max(...shown.map((r) => (Number.isFinite(r.max) ? r.max : r.mean))));
  const width = 340;
  const left = 8;
  const right = width - 12;
  const row = 34;
  const head = 4;
  const height = head + shown.length * row + 22;
  const x = linearScale(top, left, right);
  const chart = svg("svg", { class: "plot rangechart", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title });
  for (const tick of axisTicks(top, 4)) {
    chart.append(svg("line", { x1: x(tick), x2: x(tick), y1: head, y2: height - 18, class: "grid" }),
      svg("text", { x: x(tick), y: height - 4, "text-anchor": "middle", class: "tick" }, format(tick)));
  }
  shown.forEach((r, i) => {
    const y = head + i * row;
    const text = t("lab.results.bar", { id: r.id, mean: format(r.mean), min: format(r.min), max: format(r.max), n: r.n });
    chart.append(svg("text", { x: left, y: y + 11, class: "label" }, `${r.id}  ${format(r.mean)}`),
      svg("rect", { x: left, y: y + 15, width: Math.max(1, x(r.mean) - left), height: 9, class: "bar s1" }, svg("title", {}, text)));
    if (Number.isFinite(r.min) && Number.isFinite(r.max) && r.max > r.min) {
      chart.append(svg("line", { x1: x(r.min), x2: x(r.max), y1: y + 19.5, y2: y + 19.5, class: "whisker" }),
        svg("line", { x1: x(r.min), x2: x(r.min), y1: y + 15, y2: y + 24, class: "whisker" }),
        svg("line", { x1: x(r.max), x2: x(r.max), y1: y + 15, y2: y + 24, class: "whisker" }));
    }
  });
  const table = h("table", { class: "data compact" },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, ""), h("th", { scope: "col" }, t("lab.results.mean")),
      h("th", { scope: "col" }, t("lab.results.min")), h("th", { scope: "col" }, t("lab.results.max")), h("th", { scope: "col" }, "n"))),
    h("tbody", {}, rows.map((r) => h("tr", {}, h("th", { scope: "row" }, r.id), h("td", {}, format(r.mean)),
      h("td", {}, format(r.min)), h("td", {}, format(r.max)), h("td", {}, String(r.n))))));
  return h("figure", { class: "chart-box" }, h("figcaption", {}, title), chart,
    h("details", {}, h("summary", {}, t("lab.as_table")), h("div", { class: "table-wrap" }, table)));
}

const der = (v) => num(v, 3);

function corpusResults(corpus, summary) {
  const denoisers = summaryDenoisers(summary);
  let denoiser = denoisers.includes("none") ? "none" : denoisers[0];
  const out = h("div", { class: "stack tight" });
  const scored = summary.recordings.length - summary.unscored.length;
  function draw() {
    const byDenoiser = derByDenoiser(summary);
    out.replaceChildren(
      rangeChart(derByDiarizer(summary, denoiser), { title: t("lab.results.der_diarizer", { denoiser: denoiser === "none" ? t("lab.original") : denoiser }), format: der }),
      denoisers.length > 1 ? rangeChart(byDenoiser, { title: t("lab.results.der_denoiser", { n: byDenoiser[0]?.n ?? 0 }), format: der }) : null);
  }
  const select = h("select", { id: `lab-res-${corpus}` }, denoisers.map((d) => h("option", { value: d, selected: d === denoiser }, d === "none" ? t("lab.original") : d)));
  select.addEventListener("change", () => { denoiser = select.value; draw(); });
  draw();
  return h("section", { class: "stack tight" }, h("h3", {}, corpusName(corpus)),
    h("p", { class: "meta" }, t("lab.results.corpus_meta", { scored, recordings: summary.recordings.length })),
    summary.non_commercial ? h("p", { class: "notice warn", role: "note" }, t("lab.results.note_nc")) : null,
    denoisers.length > 1 ? h("span", { class: "row tight" }, h("label", { for: `lab-res-${corpus}` }, t("lab.denoiser")), select) : null,
    denoisers.length ? out : h("p", { class: "hint" }, t("lab.results.none")));
}

// ---- pending -------------------------------------------------------------------

function pendingBlock(entries) {
  const groups = pendingGroups(entries);
  if (!groups.length) return [h("p", { class: "hint" }, t("lab.pending.none"))];
  return [
    h("p", { class: "notice", role: "note" }, t("lab.pending.resume")),
    h("ul", { class: "plain stack tight" }, groups.map((group) => h("li", {}, h("details", { class: "pending" },
      h("summary", {}, t("lab.pending.summary", { session: group.entry.title || group.entry.session_id, failed: group.failed, left: group.steps.length })),
      group.active ? h("p", { class: "hint" }, t("lab.pending.running_note")) : null,
      h("ol", { class: "steps" }, group.steps.map((step) => h("li", { class: `step state-${step.state}` },
        h("span", { class: "badge" }, t(`lab.step_state.${step.state}`)), " ", stepLabel(step),
        step.code ? h("p", { class: "error" }, `${step.code}${step.message ? `: ${step.message}` : ""}`) : null))),
      h("p", {}, h("a", { class: "button", href: `#/lab/${encodeURIComponent(group.entry.session_id)}` }, t("lab.pending.open"))))))),
  ];
}

// ---- samples -------------------------------------------------------------------

/** The excerpt of a session: from its reference turns when it has a reference, else from the consensus. */
export async function sampleCut(ctx, entry, results) {
  const duration = results.duration || entry.duration || 0;
  let spans = results.consensus?.spans;
  if (entry.corpus) {
    try {
      spans = turnsToSpans((await ctx.api.labReferenceTurns(entry.session_id)).turns);
    } catch (error) {
      if (error.kind === "unauthorized") throw error;
    }
  }
  return chooseExcerpt(spans, duration, EXCERPT_SECONDS);
}

function samplesFor(ctx, entry, results, cookie, cut) {
  const seconds = Math.max(0.5, cut.end - cut.start);
  const rows = sampleRows(results);
  return [
    h("p", { class: "meta" }, t("lab.samples.where", { from: formatTime(cut.start), to: formatTime(cut.end), reason: t(`lab.samples.reason.${cut.reason}`) })),
    cookie ? null : h("p", { class: "error", role: "alert" }, t("player.no_audio")),
    h("ul", { class: "plain samples" }, rows.map((row) => h("li", { class: "sample" },
      h("strong", {}, row.denoiser === "none" ? t("lab.original") : row.denoiser),
      cookie ? h("audio", { controls: true, preload: "none", src: ctx.api.labExcerptUrl(entry.session_id, entry.results_run, row.denoiser, cut.start, seconds) }) : null,
      h("p", { class: "meta" }, [
        row.der !== null ? t("lab.samples.der", { mean: der(row.der), best: der(row.bestDer), diarizer: row.bestDiarizer }) : null,
        row.agreement !== null ? t("lab.samples.agreement", { value: pct(row.agreement) }) : null,
      ].filter(Boolean).join(" · ") || "—")))),
  ];
}

// ---- the panel -------------------------------------------------------------------

/** The overview panel: { element, start(), stop() }. */
export function progressPanel(ctx) {
  const results = new Map(); // run id -> promise of its results
  const summaries = new Map(); // corpus -> summary
  let entries = [];
  let timer = null;
  let stopped = false;
  let summaryKey = null;
  let cookie = null;

  const message = h("p", { class: "error", role: "alert", hidden: true });
  const progress = h("div", { class: "stack", "aria-live": "polite" });
  const resultsBody = h("div", { class: "stack" });
  const pending = h("div", { class: "stack" });
  const samples = h("div", { class: "stack" });

  function runResults(entry) {
    const key = entry.results_run;
    if (!results.has(key)) {
      results.set(key, ctx.api.labResults(entry.session_id, key).catch((error) => { results.delete(key); throw error; }));
    }
    return results.get(key);
  }

  function ownResults(own) {
    const scored = own.filter((e) => e.results_run);
    if (!scored.length) return null;
    return h("section", { class: "stack tight" }, h("h3", {}, corpusName(OWN)),
      h("p", { class: "notice", role: "note" }, t("lab.results.note_own")),
      h("ul", { class: "plain stack tight" }, scored.map((entry) => {
        const out = h("div", {});
        const button = h("button", { type: "button" }, t("lab.results.show_agreement"));
        button.addEventListener("click", async () => {
          button.disabled = true;
          try {
            const data = await runResults(entry);
            out.replaceChildren(rangeChart(agreementByDiarizer(data), { title: t("lab.results.agreement"), format: pct, max: 1 }));
            button.hidden = true;
          } catch (error) {
            if (error.kind === "unauthorized") return ctx.fail(error);
            out.replaceChildren(h("p", { class: "error" }, describeError(error)));
            button.disabled = false;
          }
        });
        return h("li", { class: "stack tight" }, h("div", { class: "row wrap tight" }, sessionLink(entry), button), out);
      })));
  }

  async function drawResults() {
    const corpora = [...new Set(entries.filter((e) => e.corpus).map((e) => e.corpus))].sort();
    const key = JSON.stringify(entries.map((e) => [e.session_id, e.results_run]));
    if (key === summaryKey) return;
    summaryKey = key;
    const blocks = [];
    if (corpora.length) blocks.push(h("p", { class: "notice", role: "note" }, t("lab.results.note_reference")));
    for (const corpus of corpora) {
      try {
        summaries.set(corpus, await ctx.api.labReferenceSummary(corpus));
      } catch (error) {
        if (error.kind === "unauthorized") return ctx.fail(error);
        blocks.push(h("p", { class: "error" }, describeError(error)));
        continue;
      }
      blocks.push(corpusResults(corpus, summaries.get(corpus)));
    }
    blocks.push(ownResults(entries.filter((e) => corpusOf(e) === OWN)));
    const shown = blocks.filter(Boolean);
    resultsBody.replaceChildren(...(shown.length ? shown : [h("p", { class: "hint" }, t("lab.results.none"))]));
  }

  function drawSamples() {
    const ready = entries.filter((e) => e.results_run);
    if (!ready.length) {
      samples.replaceChildren(h("p", { class: "hint" }, t("lab.samples.none")));
      return;
    }
    const open = new Set([...samples.querySelectorAll("details[open]")].map((d) => d.dataset.key));
    samples.replaceChildren(h("p", { class: "hint" }, t("lab.samples.about", { seconds: EXCERPT_SECONDS })),
      ...groupByCorpus(ready).map(({ corpus, sessions }) => h("section", { class: "stack tight" }, h("h3", {}, corpusName(corpus)),
        sessions.map((entry) => {
          const key = `${entry.session_id}/${entry.results_run}`;
          const body = h("div", { class: "stack tight" });
          const box = h("details", { class: "samples-box", "data-key": key }, h("summary", {}, entry.title || entry.session_id), body);
          let loaded = false;
          box.addEventListener("toggle", async () => {
            if (!box.open || loaded) return;
            loaded = true;
            body.replaceChildren(h("p", { class: "hint" }, t("lab.loading")));
            try {
              if (cookie === null) {
                try {
                  await ctx.api.audioCookie();
                  cookie = true;
                } catch (error) {
                  if (error.kind === "unauthorized") throw error;
                  cookie = false;
                }
              }
              const data = await runResults(entry);
              body.replaceChildren(...samplesFor(ctx, entry, data, cookie, await sampleCut(ctx, entry, data)));
            } catch (error) {
              loaded = false;
              if (error.kind === "unauthorized") return ctx.fail(error);
              body.replaceChildren(h("p", { class: "error" }, describeError(error)));
            }
          });
          if (open.has(key)) box.open = true;
          return box;
        }))));
  }

  async function refresh() {
    if (stopped) return;
    clearTimeout(timer);
    timer = null;
    try {
      entries = (await ctx.api.labOverview()).sessions;
      message.hidden = true;
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error);
      message.hidden = false;
      message.textContent = describeError(error);
      schedule(true);
      return;
    }
    if (stopped) return;
    fill(progress, progressBlock(entries, new Date()));
    fill(pending, pendingBlock(entries));
    const samplesKey = JSON.stringify(entries.map((e) => e.results_run));
    if (samples.dataset.key !== samplesKey) {
      samples.dataset.key = samplesKey;
      drawSamples();
    }
    await drawResults();
    schedule(false);
  }

  function schedule(always) {
    if (stopped || document.hidden) return;
    if (always || anyActive(entries)) timer = setTimeout(refresh, POLL_MS);
  }

  function visibility() {
    if (document.hidden) {
      clearTimeout(timer);
      timer = null;
    } else if (!timer && entries.some((e) => ACTIVE_RUNS.has(e.run_state))) {
      refresh();
    }
  }

  const element = h("div", { class: "stack lab-progress" }, message,
    h("section", { class: "stack" }, h("h2", {}, t("lab.progress.title")), progress),
    h("section", { class: "stack" }, h("h2", {}, t("lab.results.title")), resultsBody),
    h("section", { class: "stack" }, h("h2", {}, t("lab.pending.title")), pending),
    h("section", { class: "stack" }, h("h2", {}, t("lab.samples.title")), samples));

  return {
    element,
    async start() {
      document.addEventListener("visibilitychange", visibility);
      await refresh();
    },
    stop() {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibility);
    },
  };
}
