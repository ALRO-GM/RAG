// Coverage of a live interview (#286), drawn: a topic's ring that fills by
// level with a green that deepens (never colour alone: each level has its own
// fill fraction and its words, for screen readers and on hover / focus), a
// concept's state with asked and treated apart, «Ya lo he preguntado», the
// next concept to ask, the end summary line and the report's block. The
// facts come from the server (lib/coverage.js reads them); nothing here
// rates the candidate.

import { h, svg } from "../dom.js";
import { t } from "../i18n.js";
import { localized } from "../lib/live.js";
import { FILL, coverageCounts, lowerFirst, pointState } from "../lib/coverage.js";
import { icon } from "../ui.js";

const RADIUS = 7;
const AROUND = 2 * Math.PI * RADIUS;

/** The ring of a level: an empty circle, a third, two thirds, full (decoration). */
export function levelRing(level) {
  const part = FILL[level] ?? 0;
  return svg("svg", { class: `cov-ring is-${level}`, viewBox: "0 0 18 18", width: 18, height: 18, "aria-hidden": "true", focusable: "false" },
    svg("circle", { class: "cov-ring-track", cx: 9, cy: 9, r: RADIUS }),
    part > 0 && part < 1
      ? svg("circle", { class: "cov-ring-fill", cx: 9, cy: 9, r: RADIUS, "stroke-dasharray": `${(AROUND * part).toFixed(2)} ${AROUND.toFixed(2)}`, transform: "rotate(-90 9 9)" })
      : null,
    part >= 1 ? svg("circle", { class: "cov-ring-full", cx: 9, cy: 9, r: RADIUS }) : null);
}

/** The words of a topic's level and its concepts: «Bases de datos: tratado · 2 de 3 conceptos tratados, 1 preguntado». */
export function topicWords(name, coverage) {
  const level = t(`cov.level.${coverage.level}`);
  return coverage.concepts
    ? t("cov.topic_label_concepts", { name, level, treated: coverage.treated, all: coverage.concepts, asked: coverage.asked })
    : t("cov.topic_label", { name, level });
}

/** A topic chip with its level: ring, name, the level in words (read and shown on hover / focus). */
export function topicChip(chip, name, extra = "") {
  const words = topicWords(name, chip.coverage);
  const classes = ["lv-chip", "cov-chip", `is-${chip.coverage.level}`, extra].filter(Boolean).join(" ");
  return h("span", {
    class: classes, title: words, tabindex: "0",
    "data-level": chip.coverage.level,
  }, levelRing(chip.coverage.level), h("span", { "aria-hidden": "true" }, name), h("span", { class: "sr-only" }, words));
}

/** A concept's line on the live page: its mark, words, state and, while running, «Ya lo he preguntado». */
export function pointRow(point, { lang = "es", running = false, onMark = null } = {}) {
  const state = pointState(point);
  const button = running && onMark && (!point.asked || point.asked.by === "hand")
    ? h("button", {
      type: "button", class: "small cov-mark", "aria-pressed": point.asked ? "true" : "false",
      onClick: () => onMark(point, !point.asked),
    }, point.asked ? t("cov.unmark") : t("cov.mark"))
    : null;
  return h("li", { class: `iv-point cov-point is-${state}` },
    h("span", { class: "iv-point-mark", "aria-hidden": "true" }, mark(state)),
    h("span", { class: "iv-point-text" }, localized(point, "text", lang)),
    " ", h("span", { class: "iv-point-status" }, t(`cov.point.${state}`)),
    point.asked?.by === "hand" ? [" ", h("span", { class: "meta" }, t("cov.by.hand"))] : null,
    button ? [" ", button] : null);
}

function mark(state) {
  if (state === "asked_answered" || state === "treated") return icon("check");
  if (state === "asked_unknown" || state === "unknown") return "?";
  if (state === "asked") return "•";
  return h("span", { class: "lv-ring" });
}

/**
 * The next concept to ask (#286), in the page's language: «Siguiente:
 * Índices (Bases de datos)» and the proposed question; null when none.
 */
export function nextLine(next, chips, lang = "es") {
  if (!next) return null;
  const topic = chips.find((c) => c.id === next.topic);
  const point = topic?.points?.find((p) => p.id === next.point);
  const topicName = topic ? localized(topic, "name", lang) : next.topic_name;
  const pointName = point ? localized(point, "text", lang) : next.text;
  const question = !pointName ? next.question
    : next.again ? t("cov.question_again", { point: lowerFirst(pointName) })
      : t("cov.question", { topic: lowerFirst(topicName), point: lowerFirst(pointName) });
  return h("div", { class: "cov-next", role: "status" },
    h("p", { class: "cov-next-head" }, t(next.again ? "cov.next_again" : "cov.next", { point: pointName || topicName, topic: topicName })),
    h("p", { class: "cov-next-question" }, question));
}

/** The end summary's line: topics per level, concepts asked (by hand); null without coverage. */
export function summaryLine(coverage) {
  const c = coverageCounts(coverage);
  if (!c.topics) return null;
  return h("p", { class: "hint cov-summary" },
    t("cov.summary", { well: c.well_treated, treated: c.treated, named: c.named, none: c.none, asked: c.asked, all: c.concepts }),
    c.hand ? t("cov.summary_hand", { n: c.hand }) : "");
}

/**
 * The report's coverage (#286): every topic with its level, each concept
 * asked or not and treated or not, with who said it was asked. «No
 * preguntado» is the interview's gap, not the candidate's. Null for a
 * report without coverage (AI interviews, older live reports).
 */
export function coverageBlock(sheet) {
  const coverage = Array.isArray(sheet?.coverage) ? sheet.coverage : null;
  if (!coverage?.length) return null;
  return h("section", { class: "cov-report", "aria-label": t("cov.title") },
    h("h3", {}, t("cov.title")),
    h("p", { class: "hint" }, t("cov.report_note")),
    summaryLine(coverage),
    h("ol", { class: "iv-skills" }, coverage.map((topic) => h("li", { class: "iv-skill" },
      h("div", { class: "row wrap iv-skill-head" }, levelRing(topic.level),
        h("span", { class: "iv-skill-label" }, topic.label || topic.id),
        h("span", { class: "meta" }, t(`cov.level.${topic.level}`))),
      topic.points?.length ? h("ul", { class: "iv-point-list" }, topic.points.map((p) => {
        const state = pointState(p);
        return h("li", { class: `iv-point cov-point is-${state}` },
          h("span", { class: "iv-point-mark", "aria-hidden": "true" }, mark(state)),
          h("span", { class: "iv-point-text" }, p.text || p.id), " ",
          h("span", { class: "iv-point-status" }, t(`cov.report.${state}`)),
          p.asked ? [" ", h("span", { class: "meta" }, t(`cov.by.${p.asked.by}`))] : null,
          p.asked?.quote ? [" ", h("q", {}, p.asked.quote)] : null);
      })) : null))));
}
