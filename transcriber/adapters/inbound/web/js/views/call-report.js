// «Informe de la llamada» (#/call-report/<demo>, #250): the report of the
// scripted support call, for the supervisor or the HR person who reads it.
// Summary first; then what each line of the agent did (judged by the
// customer's next reaction), how the call followed the reference script,
// how other agents word the same moments (illustrative sample), and
// whether anything was left unsaid. Every part points at lines of the
// call («Línea 5»), which open in the full conversation at the bottom.
// Everything here is fictitious and the page says so. Pure parts:
// lib/call-report.js. Coaching material: it never ranks people.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { localized } from "../lib/live.js";
import {
  callLines, effectCounts, isDemoReport, itemText, lineOf, outcomeCounts, stepCounts,
} from "../lib/call-report.js";
import { emptyState, icon, pageHead } from "../ui.js";

const EFFECT_ICON = { helped: "check", neutral: "info", hurt: "alert" };
const STEP_ICON = { followed: "check", reordered: "info", skipped: "alert" };
const OUTCOME_ICON = { said: "check", said_late: "info", never: "alert" };

export async function callReportView(ctx, root, [demoId] = []) {
  root.classList.add("wide");
  let data = null;
  try {
    data = await ctx.api.liveDemoReport(demoId);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("report.title"), { lead: t("report.lead") }),
      emptyState({
        iconName: "analysis", title: t("report.unavailable"),
        body: error?.code === "not_found" ? t("report.unavailable_body") : describeError(error),
      }));
    return null;
  }
  if (!isDemoReport(data)) {
    root.append(pageHead(t("report.title")), emptyState({ iconName: "analysis", title: t("report.unavailable"), body: t("report.unavailable_body") }));
    return null;
  }
  draw(root, data);
  return null;
}

function draw(root, data) {
  const call = callLines(data);
  const full = h("details", { class: "card cr-call", id: "cr-call" },
    h("summary", {}, t("report.call_title")),
    h("p", { class: "hint" }, t("report.call_hint")),
    h("ol", { class: "cr-lines" }, call.lines.map((l) => h("li", {
      class: `cr-line is-${l.speaker === call.agent ? "agent" : "customer"}`, id: `cr-line-${l.n}`, tabindex: "-1",
    }, h("span", { class: "cr-line-head" },
      h("span", { class: "cr-line-n" }, t("report.line", { n: l.n })), " · ", h("span", {}, l.name), " · ", h("span", { class: "meta" }, l.time)),
    h("span", { class: "cr-line-text" }, l.text)))));

  /** A chip that points at a line of the call and opens it in the conversation below. */
  function chip(id) {
    const l = lineOf(call, id);
    if (!l) return null;
    return h("button", {
      type: "button", class: "lv-chip is-button cr-chip",
      "aria-label": t("report.line_open", { n: l.n }),
      onClick: () => {
        full.open = true;
        const target = root.querySelector(`#cr-line-${l.n}`);
        for (const old of root.querySelectorAll(".cr-line.is-pointed")) old.classList.remove("is-pointed");
        target?.classList.add("is-pointed");
        target?.scrollIntoView?.({ block: "center", behavior: "smooth" });
        target?.focus?.({ preventScroll: true });
      },
    }, t("report.line", { n: l.n }));
  }
  const chips = (ids) => h("span", { class: "cr-chips" }, (ids ?? []).map(chip));
  const quote = (id) => {
    const l = lineOf(call, id);
    return l ? h("q", { class: "cr-quote" }, l.text) : null;
  };

  fill(root,
    pageHead(t("report.title"), { lead: t("report.lead") }),
    h("p", { class: "notice info cr-demo", role: "note" }, icon("info"), " ", localized(data, "note", lang) || t("report.demo_note")),
    h("p", { class: "cr-call-name" }, h("strong", {}, localized(data, "title", lang)), " · ", h("span", { class: "type-chip" }, t("live.scripted.badge"))),
    summarySection(data, chips),
    linesSection(data, call, chips, quote),
    scriptSection(data, chips),
    topSection(data, chips, quote),
    leftSection(data, chips, quote),
    full);
}

function section(id, title, hint, ...body) {
  return h("section", { class: "card cr-section", "aria-labelledby": `cr-${id}` },
    h("h2", { id: `cr-${id}` }, title), hint ? h("p", { class: "hint" }, hint) : null, ...body);
}

function summarySection(data, chips) {
  const s = data.summary;
  const counts = effectCounts(data);
  const point = (p, kind) => h("li", { class: `cr-point is-${kind}` },
    h("strong", {}, localized(p, "title", lang)), h("p", {}, localized(p, "text", lang)), chips(p.about));
  return h("section", { class: "card cr-section", "aria-labelledby": "cr-summary" },
    h("h2", { id: "cr-summary" }, t("report.summary")),
    h("p", { class: "cr-result" }, localized(s, "result", lang)),
    h("p", { class: "hint" }, t("report.counts", { helped: counts.helped, neutral: counts.neutral, hurt: counts.hurt })),
    h("div", { class: "cr-two" },
      h("div", {}, h("h3", {}, t("report.strengths")), h("ul", { class: "cr-points" }, s.strengths.map((p) => point(p, "strength")))),
      h("div", {}, h("h3", {}, t("report.improve")), h("ul", { class: "cr-points" }, s.improve.map((p) => point(p, "improve"))))));
}

function linesSection(data, call, chips, quote) {
  return section("lines", t("report.lines_title"), t("report.lines_hint"),
    h("ol", { class: "cr-cards" }, data.lines_review.map((r) => {
      const l = lineOf(call, r.utterance);
      return h("li", { class: `cr-card is-${r.effect}` },
        h("div", { class: "cr-card-head" },
          h("span", { class: `cr-effect is-${r.effect}` }, icon(EFFECT_ICON[r.effect]), t(`report.effect.${r.effect}`)),
          chips([r.utterance])),
        h("p", { class: "cr-said" }, l ? h("q", {}, l.text) : null),
        h("p", { class: "cr-react" }, h("span", { class: "cr-label" }, t("report.reaction")), " ", quote(r.reaction), " ", chips([r.reaction])),
        h("p", {}, localized(r, "why", lang)),
        r.better ? h("p", { class: "cr-better" }, h("span", { class: "cr-label" }, t("report.better")), " ", h("q", {}, localized(r, "better", lang))) : null);
    })));
}

function scriptSection(data, chips) {
  const script = data.script;
  const counts = stepCounts(data);
  const verdict = script.verdict;
  return section("script", t("report.script_title"), t("report.script_hint", { name: localized(script, "name", lang) }),
    h("p", { class: "hint" }, t("report.script_counts", { followed: counts.followed, reordered: counts.reordered, skipped: counts.skipped })),
    h("ol", { class: "cr-steps" }, script.steps.map((s) => h("li", { class: `cr-step is-${s.status}` },
      h("span", { class: `cr-state is-${s.status}` }, icon(STEP_ICON[s.status]), t(`report.step.${s.status}`)),
      h("span", { class: "cr-step-text" }, h("strong", {}, localized(s, "name", lang)),
        s.note ? h("span", { class: "hint" }, " · ", localized(s, "note", lang)) : null),
      chips(s.about)))),
    h("div", { class: `cr-verdict${verdict.improvable ? " is-improvable" : ""}`, role: "note" },
      h("h3", {}, verdict.improvable ? t("report.script_improvable") : t("report.script_fine")),
      h("p", {}, localized(verdict, "text", lang)), chips(verdict.about),
      h("p", { class: "hint" }, localized(verdict, "caution", lang))));
}

function topSection(data, chips, quote) {
  const top = data.top_scorers;
  return section("top", t("report.top_title"), null,
    h("p", { class: "notice cr-illustrative", role: "note" }, localized(top, "note", lang)),
    h("ul", { class: "cr-moments" }, top.moments.map((m) => h("li", { class: "cr-moment" },
      h("h3", {}, localized(m, "moment", lang), " ", chips([m.utterance])),
      h("div", { class: "cr-pair" },
        h("div", {}, h("span", { class: "cr-label" }, t("report.top_yours")), quote(m.utterance)),
        h("div", {}, h("span", { class: "cr-label" }, t("report.top_theirs")), h("q", { class: "cr-quote" }, localized(m, "top", lang)))),
      h("p", { class: "hint" }, localized(m, "why", lang))))));
}

function leftSection(data, chips, quote) {
  const counts = outcomeCounts(data);
  const gaps = data.unanswered ?? [];
  return section("left", t("report.left_title"), t("report.left_hint"),
    h("p", { class: "hint" }, t("report.left_counts", { said: counts.said, late: counts.said_late, never: counts.never })),
    h("ul", { class: "cr-items" }, data.mandatory.map((m) => h("li", { class: `cr-item is-${m.outcome}` },
      h("span", { class: `cr-state is-${m.outcome}` }, icon(OUTCOME_ICON[m.outcome]), t(`report.outcome.${m.outcome}`)),
      h("span", { class: "cr-step-text" }, h("strong", {}, itemText(m.item, data.mandatory_items)),
        h("span", { class: "hint" }, " · ", localized(m, "note", lang))),
      chips(m.utterance ? [m.utterance] : [])))),
    h("h3", {}, t("report.unanswered_title")),
    gaps.length
      ? h("ul", { class: "cr-items" }, gaps.map((g) => h("li", { class: "cr-item is-never" },
        h("span", { class: "cr-state is-never" }, icon("alert"), t(`report.gap.${g.kind}`)),
        h("span", { class: "cr-step-text" }, quote(g.utterance), h("span", { class: "hint" }, " ", localized(g, "text", lang))),
        chips([g.utterance]))))
      : h("p", { class: "hint" }, t("report.unanswered_none")));
}
