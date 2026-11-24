// Summary first (#230): the parts every result page draws before its
// details. A person's summary (score ring, descriptive verdict, short
// summary, at most three strengths and two improvements, one bar per
// competency) and the «Más información» disclosure that holds the rest.
// The ring and the bars are SVG drawn with presentation attributes, so
// the page's CSP (no inline styles) holds. Pure parts: lib/summary-first.js.

import { h, svg } from "../dom.js";
import { t, decimal } from "../i18n.js";
import { RING_RADIUS, caveatKey, ringDash, verdictKey } from "../lib/summary-first.js";

/** "88,7 %" in Spanish, "88.7%" in English (one decimal). */
export function percentText(value) {
  return t("sf.percent", { value: decimal(value, 1) });
}

/** The score ring: the percentage in the middle, the arc around it. */
export function scoreRing(percent, { label = "" } = {}) {
  const dash = ringDash(percent);
  const name = label ? t("sf.ring_label", { what: label, value: percentText(percent) }) : percentText(percent);
  return h("div", { class: "sf-ring" },
    svg("svg", { viewBox: "0 0 120 120", role: "img", "aria-label": name, focusable: "false" },
      svg("circle", { class: "sf-ring-track", cx: 60, cy: 60, r: RING_RADIUS, fill: "none", "stroke-width": 10 }),
      svg("circle", {
        class: "sf-ring-fill", cx: 60, cy: 60, r: RING_RADIUS, fill: "none", "stroke-width": 10,
        "stroke-linecap": "round", "stroke-dasharray": `${dash.filled} ${dash.total}`, transform: "rotate(-90 60 60)",
      })),
    h("div", { class: "sf-ring-value", "aria-hidden": "true" }, h("span", {},
      h("span", { class: "sf-ring-number" }, decimal(percent, 1)),
      h("span", { class: "sf-ring-unit" }, t("sf.percent_sign")))));
}

/** Results older than the opinion: said once, with «Rehacer solo la síntesis» when it can run. */
export function legacyNote(onRedo) {
  return h("p", { class: "sf-legacy" }, t("sf.legacy"), " ",
    onRedo ? h("button", { type: "button", class: "small no-print", onClick: onRedo }, t("sa.redo_synthesis")) : null);
}

/** The descriptive label of a verdict, as a quiet pill with a dot. */
export function verdictPill(verdict, conversationType) {
  return h("span", { class: `sf-verdict is-${verdict}` },
    h("span", { class: "sf-dot", "aria-hidden": "true" }), t(verdictKey(verdict, conversationType)));
}

/** One competency bar: name, track with the level (and the minimum asked for, when known), value. */
export function competencyBar(bar) {
  const width = bar.percent ?? 0;
  const parts = [
    svg("rect", { class: "sf-bar-track", x: 0, y: 3, width: 100, height: 8, rx: 4 }),
    svg("rect", { class: `sf-bar-fill is-${bar.verdict}`, x: 0, y: 3, width: Math.max(width, 0.5), height: 8, rx: 4 }),
  ];
  if (bar.required !== null && bar.required !== undefined) {
    parts.push(svg("rect", { class: "sf-bar-min", x: Math.min(99, Math.max(0, bar.required - 0.5)), y: 0, width: 1, height: 14 }));
  }
  const value = bar.percent === null ? "–" : String(Math.round(bar.percent));
  const label = bar.required === null || bar.required === undefined
    ? t("sf.bar_label", { name: bar.name, value })
    : t("sf.bar_label_min", { name: bar.name, value, min: String(Math.round(bar.required)) });
  return h("li", { class: `sf-bar is-${bar.verdict}` },
    h("span", { class: "sf-bar-name" }, bar.name),
    svg("svg", { class: "sf-bar-svg", viewBox: "0 0 100 14", preserveAspectRatio: "none", role: "img", "aria-label": label, focusable: "false" }, parts),
    h("span", { class: "sf-bar-value", "aria-hidden": "true" }, value));
}

/** A list of strengths or improvements: the competency in bold, then its first sentence. */
function pointList(items, kind) {
  return h("ul", { class: `sf-points is-${kind}` }, items.map((item) => h("li", {},
    h("span", { class: "sf-dot", "aria-hidden": "true" }),
    h("span", {}, h("strong", {}, item.text ? `${item.name}.` : item.name), item.text ? ` ${item.text}` : ""))));
}

/**
 * The «Más información» disclosure: closed, with a line saying what is
 * inside. `children` are drawn when it is opened the first time (or at
 * once when `eager`), so long pages stay light.
 */
export function moreInfo(hint, children, { eager = true, id = null, extraClass = "" } = {}) {
  const body = h("div", { class: "sf-more-body stack" });
  const box = h("details", { class: `sf-more ${extraClass}`.trim(), id },
    h("summary", {}, h("span", { class: "sf-more-title" }, t("sf.more")), hint ? h("span", { class: "sf-more-hint" }, hint) : null),
    body);
  let drawn = false;
  const draw = () => {
    if (drawn) return;
    drawn = true;
    const made = typeof children === "function" ? children() : children;
    body.append(...[made].flat(Infinity).filter(Boolean));
  };
  if (eager) draw();
  else box.addEventListener("toggle", () => { if (box.open) draw(); });
  return box;
}

/** The line of a person's «Más información»: supporting sentences · reliability · notices. */
export function moreHint(more) {
  const parts = [t(more.quotes === 1 ? "sf.more_quotes_one" : "sf.more_quotes", { count: more.quotes }), t("sf.more_reliability")];
  if (more.notices) parts.push(t(more.notices === 1 ? "sf.more_notices_one" : "sf.more_notices", { count: more.notices }));
  return parts.join(" · ");
}

/** How firm the opinion is and its caveats, in words: the first thing inside «Más información». */
export function reliabilityBlock(summary) {
  const { confidence, caveats } = summary.more;
  return h("div", { class: "sf-reliability stack tight" },
    h("h4", {}, t("sf.reliability_title")),
    confidence ? h("p", {}, t("sf.confidence", { level: t(`sa.syn_conf_${confidence}`) })) : null,
    caveats.length
      ? h("ul", {}, caveats.map((code) => h("li", {}, caveatKey(code) ? t(caveatKey(code)) : code)))
      : null,
    h("p", { class: "hint" }, t("sf.reliability_note")));
}

/**
 * The summary of one person, as a disclosure (#230): its header — name,
 * percentage, verdict words and the overall line — is always visible;
 * opening it shows the ring with the short summary, the strengths and
 * improvements, the bars of those (the full list behind «Ver las N
 * competencias») and the «Más información» element `more`. Only the
 * card given `open` starts open. Results that predate the opinion say so
 * once, in the overview.
 */
export function personSummaryCard(summary, { conversationType = "", withJob = false, more = null, id = null, open = false } = {}) {
  const o = summary.overall;
  const head = h("summary", { class: "sf-person-head" },
    h("span", { class: "sf-head-row" },
      h("span", { class: "sf-name" }, h("span", { class: `dot who-${summary.colour}`, "aria-hidden": "true" }), summary.name),
      o && o.percent !== null ? h("span", { class: "sf-head-score" }, percentText(o.percent)) : null,
      o ? verdictPill(o.verdict, conversationType) : null),
    o?.line ? h("span", { class: "sf-line" }, o.line) : null);
  const hero = (o && o.percent !== null) || summary.summary
    ? h("div", { class: "sf-hero" },
      o && o.percent !== null ? scoreRing(o.percent, { label: t(withJob ? "sf.ring_job" : "sf.ring_general") }) : null,
      summary.summary ? h("p", { class: "sf-summary" }, summary.summary) : null)
    : null;
  const columns = [];
  if (summary.strengths.length) {
    columns.push(h("section", { class: "sf-col" }, h("h4", {}, t("sf.strengths")), pointList(summary.strengths, "strength")));
  }
  if (summary.improvements.length) {
    columns.push(h("section", { class: "sf-col" }, h("h4", {}, t("sf.improvements")), pointList(summary.improvements, "improve")));
  }
  const hasMin = summary.bars.some((b) => b.required !== null && b.required !== undefined);
  const allBars = summary.bars.length
    ? h("details", { class: "sf-all-bars" },
      h("summary", {}, t("sf.all_bars", { count: summary.bars.length })),
      h("ul", { class: "sf-bar-list" }, summary.bars.map(competencyBar)))
    : null;
  return h("details", { class: `sf-person who-${summary.colour}`, id, open },
    head,
    h("div", { class: "sf-person-body" },
      hero,
      columns.length ? h("div", { class: "sf-columns" }, columns) : null,
      summary.bars.length
        ? h("section", { class: "sf-bars" },
          h("div", { class: "sf-bars-head" },
            h("h4", {}, t(withJob ? "sf.bars_job" : "sf.bars_general")),
            hasMin ? h("span", { class: "sf-min-key" }, h("span", { class: "sf-min-mark", "aria-hidden": "true" }), t("sf.min_key")) : null),
          summary.shownBars.length ? h("ul", { class: "sf-bar-list" }, summary.shownBars.map(competencyBar)) : null,
          allBars)
        : null,
      more));
}

/**
 * The whole opinion, for «Más información»: the question it answers
 * (`frame`), the full overall line, and every competency's verdict,
 * score, full conclusion, what it rests on, who wrote it and the
 * sentences it cites (`moment` draws a quote's minute).
 */
export function opinionBlock(person, moment) {
  const opinion = person?.opinion;
  if (!opinion?.overall && !(opinion?.skills ?? []).length) return null;
  const o = opinion.overall;
  const names = new Map((person.skills ?? []).map((k) => [k.id, k.name]));
  return h("div", { class: "sf-opinion stack tight" },
    h("h4", {}, t("sf.opinion_title")),
    o?.frame ? h("p", {}, h("strong", {}, t("sf.frame")), " ", o.frame) : null,
    o?.line ? h("p", {}, h("strong", {}, t("sf.full_line")), " ", o.line) : null,
    o ? h("p", { class: "hint" }, t(`sf.source.${o.source === "judge" ? "judge" : "rules"}`)) : null,
    (opinion.skills ?? []).length
      ? h("ul", { class: "sf-opinion-skills" }, opinion.skills.map((k) => h("li", { class: "stack tight" },
        h("p", {}, h("strong", {}, names.get(k.id) ?? (k.name || k.id)), " · ",
          t(`sf.skill_verdict.${k.verdict}`), " · ",
          t("sf.skill_score", { score: decimal(k.score, 1) })),
        k.conclusion ? h("p", {}, k.conclusion) : null,
        h("p", { class: "hint" }, [
          t(`sf.basis.${["firm", "provisional", "observation", "participation"].includes(k.basis) ? k.basis : "participation"}`),
          t(`sf.source.${k.source === "judge" ? "judge" : "rules"}`),
        ].join(" · ")),
        k.cites.length
          ? h("ul", { class: "quotes" }, k.cites.map((c) => h("li", { class: "quote" }, moment(c.start), " ", h("q", {}, c.text))))
          : null)))
      : null);
}
