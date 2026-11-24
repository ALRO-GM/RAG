// «Calidad de la conexión» (#247): the card of the live page's start screen
// and the quiet indicator of a running session. One verdict first (text,
// icon and a small dot: never colour alone), one sentence on what to do
// when something is wrong, the services as a short list, and the figures
// behind «Más información». The requests are js/connection.js, the
// figures and the verdict lib/connection.js (docs/live/CONNECTION.md).

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { formatDecimal } from "../lib/i18n.js";
import { CHECK, SERVICE_ROWS, assess, rounded } from "../lib/connection.js";
import { runCheck } from "../connection.js";
import { icon } from "../ui.js";
import { moreInfo } from "./summary-first.js";

const ICON = { good: "check", fair: "alert", poor: "alert", lost: "alert" };
const ROW_DOT = { available: "done", unavailable: "avoid", not_set_up: "", unknown: "", checking: "" };

/** The last check of this page load, shown again while it is young. */
let last = null; // { result, at }

/** The level of the last check still young enough to show (null otherwise). */
export function lastLevel(options = {}) {
  if (!last || Date.now() - last.at > CHECK.maxAgeMs) return null;
  return assess(last.result, options).level;
}

function clock(at) {
  return new Date(at).toLocaleTimeString(lang === "en" ? "en-GB" : "es-ES", { hour: "2-digit", minute: "2-digit" });
}

/** A rate in kbit/s as words: «640 kbit/s», «2,1 Mbit/s». */
export function rateText(kbps) {
  if (kbps === null || kbps === undefined) return t("conn.fig.none");
  return kbps >= 1000 ? t("conn.unit.mbps", { value: formatDecimal(lang, kbps / 1000, 1) }) : t("conn.unit.kbps", { value: Math.round(kbps) });
}

function ms(value) {
  const v = rounded(value);
  return v === null ? t("conn.fig.none") : t("conn.unit.ms", { value: v });
}

/**
 * The start screen's card. `options()` gives { needs, upload } for the
 * session about to start (read again on every drawing, so a change of the
 * audio source updates the verdict without a new check).
 * Returns { node, check(force), verdict(), redraw() }.
 */
export function connectionCard({ getCode, options = () => ({}), run = runCheck }) {
  const verdict = h("p", { class: "conn-verdict", role: "status" });
  const advice = h("p", { class: "conn-advice", hidden: true });
  const rows = h("ul", { class: "conn-rows", "aria-label": t("conn.services") });
  const figures = h("dl", { class: "conn-figures" });
  const at = h("p", { class: "hint conn-at" });
  const again = h("button", { type: "button", class: "ghost small", onClick: () => check(true) }, t("conn.again"));
  const more = moreInfo(t("conn.more_summary"), [figures, at, h("p", { class: "hint" }, t("conn.how"))], { extraClass: "conn-more" });
  const node = h("section", { class: "card conn-card", "aria-labelledby": "conn-title" },
    h("div", { class: "section-head" }, h("h2", { id: "conn-title" }, t("conn.title")), again),
    verdict, advice, rows, more);
  let busy = false;
  let current = null; // the assessment shown

  function row(name, state) {
    return h("li", { class: "conn-row" },
      h("span", { class: `state-dot ${ROW_DOT[state] ?? ""}`.trim(), "aria-hidden": "true" }),
      h("span", { class: "conn-row-name" }, t(`conn.row.${name}`)),
      h("span", { class: `conn-row-state is-${state}` }, t(`conn.state.${state}`)));
  }

  function draw() {
    if (busy || !last) {
      node.dataset.level = "checking";
      fill(verdict, h("span", { class: "conn-icon", "aria-hidden": "true" }, h("span", { class: "state-dot lv-pulse" })), h("strong", {}, t("conn.checking")));
      advice.hidden = true;
      fill(rows, SERVICE_ROWS.map((name) => row(name, "checking")));
      fill(figures);
      fill(at);
      return;
    }
    current = assess(last.result, options());
    const { level, figures: f } = current;
    node.dataset.level = level;
    fill(verdict,
      h("span", { class: `conn-icon is-${level}` }, icon(ICON[level])),
      h("strong", {}, t(`conn.verdict.${level}`)));
    advice.hidden = !current.advice;
    advice.textContent = current.advice ? t(`conn.advice.${current.advice}`) : "";
    const needs = new Set(["web", "link", ...(options().needs ?? ["transcription", "assistant", "knowledge"])]);
    fill(rows, SERVICE_ROWS.filter((name) => needs.has(name)).map((name) => row(name, current.rows[name])));
    const pair = (key, value) => [h("dt", {}, t(key)), h("dd", {}, value)];
    fill(figures,
      pair("conn.fig.rtt", ms(f.rttMs)),
      pair("conn.fig.jitter", f.rttMs === null ? t("conn.fig.none") : ms(f.jitterMs)),
      pair("conn.fig.loss", f.count ? t("conn.fig.loss_value", { lost: f.lost, count: f.count }) : t("conn.fig.none")),
      pair("conn.fig.web_rtt", ms(f.webRttMs)),
      options().upload ? pair("conn.fig.up", f.upKbps === null ? t("conn.fig.none") : t("conn.fig.up_need", { rate: rateText(f.upKbps) })) : null);
    at.textContent = t("conn.fig.at", { time: clock(last.at) });
  }

  async function check(force = false) {
    if (busy) return;
    if (!force && last && Date.now() - last.at <= CHECK.maxAgeMs) {
      draw();
      return;
    }
    busy = true;
    again.disabled = true;
    draw();
    try {
      const result = await run({ getCode, upload: true });
      last = { result, at: Date.now() };
    } finally {
      busy = false;
      again.disabled = false;
      draw();
    }
  }

  return {
    node,
    check,
    verdict: () => (busy || !last ? null : (current ?? assess(last.result, options()))),
    redraw: draw,
  };
}

/**
 * The live header's indicator: an icon and a short word, its longer
 * sentence as the title. `set(state)` with "checking" | "good" | "fair" |
 * "poor" | "lost".
 */
export function connectionChip() {
  const node = h("span", { class: "lv-conn", role: "status" });
  let shown = null;
  function set(state) {
    if (state === shown) return;
    shown = state;
    node.dataset.state = state;
    node.title = t(`conn.chip_title.${state}`);
    fill(node,
      ICON[state] ? icon(ICON[state]) : null,
      h("span", { class: "lv-conn-text" }, t(`conn.chip.${state}`)));
  }
  set("checking");
  return { node, set };
}

export { clock as connectionClock };
