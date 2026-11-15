// "Ayuda" for the analysis (#/help/analysis[/<step or section>]): what each
// value means with small invented examples, what each competency of the
// rubric is and where it is useful, and the limits. Text comes from the
// dictionaries (help.*) and the rubric (lib/help-rubric.js).

import { h } from "../dom.js";
import { lang, t } from "../i18n.js";
import { ACTS, STEPS } from "../lib/session-analysis.js";
import {
  ACT_EXTRAS, CONFIRMATION, EVENTS, INTERACTION, LIMITS, RATING, SECTIONS,
  anchorOf, competencies, scrollTarget,
} from "../lib/help-analysis.js";

/** One entry: term, what it is, example. */
function entry(id, term, what, example) {
  return h("div", { class: "help-item", id },
    h("dt", {}, term),
    h("dd", {}, what),
    h("dd", { class: "help-example" }, h("span", { class: "badge" }, t("help.example")), " ", example));
}

function section(name, ...items) {
  return h("section", { class: "stack help-section", id: anchorOf(name) },
    h("h2", {}, t(`help.sec.${name}`)),
    h("p", { class: "hint" }, t(`help.sec.${name}_intro`)),
    ...items);
}

function list(items) {
  return h("dl", { class: "help-list" }, items);
}

function sentences(texts) {
  return h("ul", { class: "plain" }, texts.map((x) => h("li", {}, x)));
}

function competency(c) {
  return h("details", { class: "help-competency", id: `help-comp-${c.id}` },
    h("summary", {}, h("strong", {}, c.name)),
    h("p", {}, c.definition),
    h("p", {}, h("strong", {}, t("help.comp.positive")), ": «", c.positive, "»"),
    h("p", {}, h("strong", {}, t("help.comp.negative")), ": ", c.negative),
    h("p", {}, h("strong", {}, t("help.comp.helps"))),
    sentences(c.helps),
    h("p", {}, h("strong", {}, t("help.comp.hurts"))),
    sentences(c.hurts),
    h("p", { class: "hint" }, h("strong", {}, t("help.comp.limits")), ": ", c.limits));
}

export async function helpAnalysisView(ctx, root, [param]) {
  const toc = h("ul", { class: "plain help-toc" }, SECTIONS.map((s) =>
    h("li", {}, h("a", { href: `#/help/analysis/${s}` }, t(`help.sec.${s}`)))));
  root.append(
    h("h1", {}, t("help.title")),
    h("p", { class: "meta" }, h("a", { href: "#/" }, t("nav.sessions"))),
    h("p", {}, t("help.intro")),
    h("nav", { "aria-label": t("help.toc") }, h("strong", {}, t("help.toc")), toc),
    section("steps", list(STEPS.map((id) => entry(`step-${id}`,
      t(`help.step.${id}.t`), t(`help.step.${id}.d`), t(`help.step.${id}.e`))))),
    section("interaction", list(INTERACTION.map((id) => entry(`help-interaction-${id}`,
      t(`help.interaction.${id}.t`), t(`help.interaction.${id}.d`), t(`help.interaction.${id}.e`))))),
    section("acts",
      list(ACT_EXTRAS.map((id) => entry(`help-acts-${id}`,
        t(`help.acts.${id}.t`), t(`help.acts.${id}.d`), t(`help.acts.${id}.e`)))),
      list(ACTS.map((id) => h("div", { class: "help-item", id: `help-act-${id}` },
        h("dt", {}, t(`act.${id}`)), h("dd", { class: "help-example" }, t(`help.act.${id}.e`)))))),
    section("events", list(EVENTS.map((id) => entry(`help-event-${id}`,
      t(`sa.event_${id}`), t(`help.event.${id}.d`), t(`help.event.${id}.e`))))),
    section("confirmation", list(CONFIRMATION.map((id) => entry(`help-confirmation-${id}`,
      t(`help.confirmation.${id}.t`), t(`help.confirmation.${id}.d`), t(`help.confirmation.${id}.e`))))),
    section("rating",
      list(RATING.map((id) => entry(`help-rating-${id}`,
        t(`help.rating.${id}.t`), t(`help.rating.${id}.d`), t(`help.rating.${id}.e`)))),
      h("p", { class: "notice" }, t("help.rating.tiers_note"))),
    section("competencies", ...competencies(lang).map(competency)),
    section("limits", h("ol", {}, LIMITS.map((n) => h("li", {}, t(`help.limit.${n}`))))));
  const id = scrollTarget(param);
  if (id) document.getElementById(id)?.scrollIntoView();
}
