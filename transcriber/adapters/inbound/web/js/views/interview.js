// The AI interviewer's parts of the voice-bot page (#208,
// docs/live/VOICEBOT_INTERVIEWER.md): the job offer and the plan review
// before the interview, the live coverage panel for the recruiter, the
// visual report after it and the comparison of the candidates of one offer.
// Each point can carry its concepts (#258): edited one per line in the
// review, shown treated / not yet / does not know it in the coverage and
// the report (an icon and a word, never the colour alone).
// The state and the figures live in lib/interview.js; this module draws them.
// Scores are for the recruiter only: the bot never says them.

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  COMPETENCIES, LEVELS, MAX_MINUTES, MAX_POINTS, MIN_MINUTES, MIN_OFFER, ROLE_TYPES,
  addSlot, clonePlan, comparisonRows, editSlot, planCounts, planErrors, planForSave,
  pointCounts, removeSlot, reportFigures, scoreClass, setEssential,
} from "../lib/interview.js";
import { icon, toast } from "../ui.js";
import { moreInfo } from "./summary-first.js";

const WEIGHTS = [1, 2, 3, 4, 5];

function option(value, label, selected) {
  return h("option", { value, selected: value === selected }, label);
}

function select(id, values, selected, label, onChange) {
  const el = h("select", { id, "aria-label": label }, values.map(([v, text]) => option(v, text, selected)));
  el.addEventListener("change", () => onChange(el.value));
  return el;
}

/** A 0–100 bar with its colour class; `null` draws an empty track. */
export function scoreBar(score, { label = null, small = false } = {}) {
  const fillEl = h("span", { class: `iv-bar-fill is-${scoreClass(score)}` });
  fillEl.style.width = `${Math.max(0, Math.min(100, score ?? 0))}%`;
  return h("span", {
    class: small ? "iv-bar small" : "iv-bar", role: "img",
    "aria-label": label ?? (score === null || score === undefined ? t("iv.no_score") : t("iv.score_of", { score })),
  }, fillEl);
}

function statusBadge(status) {
  return h("span", { class: `badge iv-status is-${status}` }, t(`iv.status.${status}`));
}

function levelTag(level) {
  return h("span", { class: `tag iv-level is-${level}` }, t(`iv.level.${level}`));
}

/** A concept's sign: the check icon, an empty ring or a question mark (decoration; the word says it). */
export function pointMark(status) {
  if (status === "treated") return h("span", { class: "iv-point-mark", "aria-hidden": "true" }, icon("check"));
  if (status === "unknown") return h("span", { class: "iv-point-mark", "aria-hidden": "true" }, "?");
  return h("span", { class: "iv-point-mark", "aria-hidden": "true" }, h("span", { class: "lv-ring" }));
}

/**
 * A slot's concepts, each with a mark and its status in words (#258);
 * `quotes` adds the words that settled each one. Null when it has none.
 */
export function pointList(points, { quotes = false } = {}) {
  if (!points?.length) return null;
  const counts = pointCounts(points);
  return h("div", { class: "iv-points" },
    h("p", { class: "meta" }, t("iv.points_line", { done: counts.treated, all: counts.total })),
    h("ul", { class: "iv-point-list" }, points.map((p) => {
      const status = p.status ?? "pending";
      return h("li", { class: `iv-point is-${status}` },
        pointMark(status),
        h("span", { class: "iv-point-text" }, p.text),
        " ", h("span", { class: "iv-point-status" }, t(`iv.point.${status}`)),
        quotes && p.quote && status !== "pending" ? [" ", h("q", {}, p.quote)] : null);
    })));
}

// ---- before: the offer and the plan ------------------------------------------

/**
 * The job offer form and the plan review. `onReady(plan, candidate)` is
 * called when the recruiter starts the interview with a saved plan.
 */
export function planSetup(ctx, { onReady }) {
  const root = h("div", { class: "stack iv-setup" });
  let plan = null;
  let saved = null;

  const source = select("iv-source", [["text", t("iv.source.text")], ["document", t("iv.source.document")]], "text",
    t("iv.source"), () => syncSource());
  const offer = h("textarea", { id: "iv-offer", rows: 8, placeholder: t("iv.offer_placeholder") });
  const domains = h("select", { id: "iv-domain", "aria-label": t("iv.domain") });
  const docs = h("select", { id: "iv-document", "aria-label": t("iv.document") });
  const docHint = h("p", { class: "hint" }, t("iv.document_hint"));
  const docBox = h("div", { class: "stack tight", hidden: true },
    h("div", { class: "field" }, h("label", { for: "iv-domain" }, t("iv.domain")), domains),
    h("div", { class: "field" }, h("label", { for: "iv-document" }, t("iv.document")), docs), docHint);
  const textBox = h("div", { class: "field" }, h("label", { for: "iv-offer" }, t("iv.offer")), offer);
  const title = h("input", { id: "iv-title", type: "text", maxLength: 120, placeholder: t("iv.title_placeholder") });
  const role = select("iv-role", [["", t("iv.role.auto")], ...ROLE_TYPES.map((r) => [r, t(`iv.role.${r}`)])], "",
    t("iv.role"), () => {});
  const minutes = h("input", { id: "iv-minutes", type: "number", min: MIN_MINUTES, max: MAX_MINUTES, step: 5, value: 20 });
  const make = h("button", { type: "button", class: "primary", onClick: () => generate() }, icon("edit"), t("iv.generate"));
  const saved_plans = h("select", { id: "iv-saved", "aria-label": t("iv.saved") });
  const open = h("button", { type: "button", onClick: () => openSaved() }, t("iv.open"));
  const formError = h("p", { class: "error", role: "alert", hidden: true });
  const review = h("section", { class: "card iv-review", hidden: true, "aria-label": t("iv.review") });

  function syncSource() {
    const doc = source.value === "document";
    docBox.hidden = !doc;
    textBox.hidden = doc;
    if (doc && !domains.options.length) loadDomains();
  }

  async function loadDomains() {
    try {
      const rows = await ctx.api.knowledgeDomains();
      fill(domains, rows.map((d) => option(d.id, `${d.title} (${d.id})`, null)));
      domains.onchange = () => loadDocuments();
      await loadDocuments();
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      fill(domains);
      toast(describeError(error), { error: true });
    }
    return null;
  }

  async function loadDocuments() {
    if (!domains.value) return;
    try {
      const rows = await ctx.api.knowledgeDocuments(domains.value);
      const ready = rows.filter((d) => d.status === "indexed");
      fill(docs, ready.length ? ready.map((d) => option(d.id, d.name, null)) : option("", t("iv.no_documents"), ""));
    } catch (error) {
      fill(docs, option("", t("iv.no_documents"), ""));
    }
  }

  async function loadSaved() {
    try {
      const { plans } = await ctx.api.interviewPlans();
      fill(saved_plans, plans.length
        ? plans.map((p) => option(p.id, t("iv.saved_row", { title: p.title, slots: p.slots, interviews: p.interviews }), null))
        : option("", t("iv.no_saved"), ""));
      open.disabled = !plans.length;
    } catch {
      fill(saved_plans, option("", t("iv.no_saved"), ""));
      open.disabled = true;
    }
  }

  async function openSaved() {
    if (!saved_plans.value) return;
    try {
      show(await ctx.api.interviewPlan(saved_plans.value));
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  }

  async function generate() {
    formError.hidden = true;
    const body = { minutes: Number(minutes.value) || 20 };
    if (title.value.trim()) body.title = title.value.trim();
    if (role.value) body.role_type = role.value;
    if (source.value === "document") {
      if (!docs.value) {
        formError.textContent = t("iv.err.document");
        formError.hidden = false;
        return;
      }
      body.document = { domain: domains.value, id: docs.value };
    } else {
      if (offer.value.trim().length < MIN_OFFER) {
        formError.textContent = t("iv.err.offer", { n: MIN_OFFER });
        formError.hidden = false;
        return;
      }
      body.offer = offer.value;
    }
    make.disabled = true;
    make.textContent = t("iv.generating");
    try {
      show(await ctx.api.createPlan(body));
    } catch (error) {
      if (error?.kind === "unauthorized") ctx.fail(error);
      formError.textContent = describeError(error);
      formError.hidden = false;
    }
    make.disabled = false;
    make.replaceChildren(icon("edit"), t("iv.generate"));
  }

  function show(document) {
    saved = document;
    plan = clonePlan(document);
    review.hidden = false;
    drawReview();
    review.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }

  function drawReview() {
    const counts = planCounts(plan);
    const errors = planErrors(plan);
    const dirty = JSON.stringify(planForSave(plan)) !== JSON.stringify(planForSave(saved));
    const planTitle = h("input", { id: "iv-plan-title", type: "text", value: plan.title, maxLength: 120 });
    planTitle.addEventListener("change", () => { plan.title = planTitle.value; drawReview(); });
    const planMinutes = h("input", { id: "iv-plan-minutes", type: "number", min: MIN_MINUTES, max: MAX_MINUTES, step: 5, value: plan.minutes });
    planMinutes.addEventListener("change", () => { plan.minutes = Number(planMinutes.value); drawReview(); });
    const candidate = h("input", { id: "iv-candidate", type: "text", maxLength: 80, placeholder: t("iv.candidate_placeholder") });
    const save = h("button", { type: "button", disabled: !dirty || errors.length > 0, onClick: () => persist() }, t("iv.save"));
    const start = h("button", {
      type: "button", class: "primary big", disabled: dirty || errors.length > 0,
      onClick: () => onReady(saved, candidate.value.trim()),
    }, icon("voicebot"), t("iv.start"));
    const compare = h("button", { type: "button", onClick: () => showComparison() }, t("iv.compare"));
    const comparison = h("div", { class: "iv-comparison" });

    async function showComparison() {
      try {
        fill(comparison, comparisonCard(await ctx.api.planResults(saved.id)));
      } catch (error) {
        toast(describeError(error), { error: true });
      }
    }

    async function persist() {
      save.disabled = true;
      try {
        show(await ctx.api.savePlan(saved.id, planForSave(plan)));
        toast(t("iv.saved_ok"));
      } catch (error) {
        toast(error?.code === "conflict" ? t("iv.err.conflict") : describeError(error), { error: true });
        save.disabled = false;
      }
    }

    fill(review,
      h("h2", {}, t("iv.review")),
      plan.planner?.status === "fallback" ? h("p", { class: "notice", role: "note" }, t("iv.fallback", { error: plan.planner.error ?? "" })) : null,
      h("p", { class: "hint" }, t("iv.review_hint")),
      h("div", { class: "row wrap iv-plan-head" },
        h("div", { class: "field grow" }, h("label", { for: "iv-plan-title" }, t("iv.plan_title")), planTitle),
        h("div", { class: "field" }, h("label", { for: "iv-plan-minutes" }, t("iv.minutes")), planMinutes)),
      plan.summary ? h("p", { class: "meta" }, plan.summary) : null,
      h("p", { class: "meta" }, t("iv.counts", { total: counts.total, minimo: counts.minimo, deseable: counts.deseable, opcional: counts.opcional })),
      h("ol", { class: "iv-slots" }, plan.slots.map((slot) => slotEditor(slot))),
      h("div", { class: "row wrap" },
        h("button", { type: "button", disabled: plan.slots.length >= 12, onClick: () => { addSlot(plan); drawReview(); } }, icon("plus"), t("iv.add_slot"))),
      errors.length ? h("ul", { class: "error", role: "alert" }, errors.map((e) => h("li", {}, t(e)))) : null,
      h("div", { class: "row wrap iv-actions" }, save, compare,
        dirty ? h("span", { class: "hint" }, t("iv.unsaved")) : null),
      h("div", { class: "field" }, h("label", { for: "iv-candidate" }, t("iv.candidate")), candidate,
        h("p", { class: "hint" }, t("iv.candidate_hint"))),
      start, comparison);
  }

  function slotEditor(slot) {
    const key = `iv-${slot.id}`;
    const essential = h("input", { type: "checkbox", id: `${key}-min`, checked: slot.level === "minimo" });
    essential.addEventListener("change", () => { setEssential(plan, slot.id, essential.checked); drawReview(); });
    const label = h("input", { id: `${key}-label`, type: "text", value: slot.label, maxLength: 120, "aria-label": t("iv.slot_label") });
    label.addEventListener("change", () => { editSlot(plan, slot.id, { label: label.value }); drawReview(); });
    const level = select(`${key}-level`, LEVELS.map((l) => [l, t(`iv.level.${l}`)]), slot.level, t("iv.level"),
      (v) => { editSlot(plan, slot.id, { level: v }); drawReview(); });
    const weight = select(`${key}-weight`, WEIGHTS.map((w) => [String(w), t(`iv.weight.${w}`)]), String(slot.weight ?? 3),
      t("iv.weight"), (v) => { editSlot(plan, slot.id, { weight: Number(v) }); drawReview(); });
    const competency = select(`${key}-comp`, [["", t("iv.comp.none")], ...COMPETENCIES.map((c) => [c, t(`iv.comp.${c}`)])],
      slot.competency ?? "", t("iv.competency"), (v) => { editSlot(plan, slot.id, { competency: v || null }); drawReview(); });
    const questions = h("textarea", { id: `${key}-q`, rows: 2, value: slot.questions.join("\n"), "aria-label": t("iv.questions") });
    questions.addEventListener("change", () => { editSlot(plan, slot.id, { questions: questions.value.split("\n") }); drawReview(); });
    const evidence = h("input", { id: `${key}-ev`, type: "text", value: slot.evidence ?? "", maxLength: 300, "aria-label": t("iv.evidence") });
    evidence.addEventListener("change", () => { editSlot(plan, slot.id, { evidence: evidence.value }); drawReview(); });
    const points = h("textarea", {
      id: `${key}-pts`, rows: Math.max(2, Math.min(MAX_POINTS, (slot.points ?? []).length + 1)),
      value: (slot.points ?? []).map((p) => p.text).join("\n"), "aria-label": t("iv.points"),
      "aria-describedby": `${key}-pts-hint`,
    });
    points.addEventListener("change", () => { editSlot(plan, slot.id, { points: points.value.split("\n") }); drawReview(); });
    const remove = h("button", { type: "button", class: "ghost", "aria-label": t("iv.remove"), onClick: () => { removeSlot(plan, slot.id); drawReview(); } }, icon("trash"));
    return h("li", { class: `iv-slot is-${slot.level}` },
      h("div", { class: "row wrap iv-slot-head" },
        h("label", { class: "check", for: `${key}-min` }, essential, " ", t("iv.essential")),
        label, remove),
      h("div", { class: "row wrap iv-slot-meta" },
        h("label", { class: "field" }, t("iv.level"), level),
        h("label", { class: "field" }, t("iv.weight"), weight),
        h("label", { class: "field" }, t("iv.competency"), competency)),
      h("label", { class: "field" }, t("iv.questions"), questions),
      h("label", { class: "field" }, t("iv.evidence"), evidence),
      h("label", { class: "field" }, t("iv.points"), points, h("span", { class: "hint", id: `${key}-pts-hint` }, t("iv.points_hint", { n: MAX_POINTS }))));
  }

  fill(root,
    h("section", { class: "card iv-offer" },
      h("h2", {}, t("iv.offer_title")),
      h("p", { class: "hint" }, t("iv.offer_hint")),
      h("div", { class: "field" }, h("label", { for: "iv-source" }, t("iv.source")), source),
      textBox, docBox,
      h("div", { class: "row wrap" },
        h("div", { class: "field grow" }, h("label", { for: "iv-title" }, t("iv.plan_title")), title),
        h("div", { class: "field" }, h("label", { for: "iv-role" }, t("iv.role")), role),
        h("div", { class: "field" }, h("label", { for: "iv-minutes" }, t("iv.minutes")), minutes)),
      make, formError,
      h("div", { class: "row wrap iv-saved" },
        h("label", { for: "iv-saved" }, t("iv.saved")), saved_plans, open)),
    review);
  loadSaved();
  return root;
}

// ---- during: the live coverage -------------------------------------------------

/** The recruiter's live coverage panel (filled / partial / empty, quotes). */
export function coveragePanel(state) {
  const slots = state.slots();
  const counts = state.counts();
  const coverage = state.coverage;
  return h("section", { class: "vb-pane iv-coverage", "aria-label": t("iv.coverage") },
    h("h2", {}, t("iv.coverage")),
    coverage ? h("p", { class: "meta" }, t("iv.coverage_line", {
      filled: counts.filled, partial: counts.partial, empty: counts.empty, left: coverage.minutes_left, minutes: coverage.minutes,
    })) : h("p", { class: "hint" }, t("iv.coverage_wait")),
    state.closed ? h("p", { class: "notice info", role: "status" }, t(`iv.closed.${state.closedBy ?? "covered"}`)) : null,
    state.reporting ? h("p", { class: "notice info", role: "status" }, t("iv.reporting")) : null,
    h("ol", { class: "iv-cover-list" }, slots.map((s) => h("li", { class: `iv-cover is-${s.status}${s.current ? " is-current" : ""}` },
      h("div", { class: "row wrap" }, statusBadge(s.status), levelTag(s.level),
        h("span", { class: "iv-cover-label" }, s.label),
        s.current ? h("span", { class: "tag accent" }, t("iv.asking")) : null),
      s.asked ? h("p", { class: "meta" }, t("iv.attempts", { asked: s.asked, probes: s.probes })) : null,
      pointList(s.points),
      s.answers?.length ? h("ul", { class: "iv-quotes" }, s.answers.map((a) => h("li", {}, h("q", {}, a.quote)))) : null))),
    slots.some((s) => s.points?.length) ? h("p", { class: "hint" }, t("iv.points_note")) : null);
}

// ---- the accuracy of the answers (#259) -----------------------------------------

/** A judgement's label as a tag (a word, never the colour alone). */
export function judgeTag(label) {
  return h("span", { class: `tag iv-judge is-${label}` }, t(`iv.judge.${label}`));
}

function judgementText(j) {
  if (!j) return null;
  const basis = j.basis === "documents" ? t("iv.basis.documents", { ids: (j.cites ?? []).join(", ") }) : t("iv.basis.general");
  return h("p", { class: "meta iv-judge-why" },
    j.quote ? [h("q", {}, j.quote), " "] : null,
    j.reason ? [j.reason, " "] : null,
    h("span", { class: "iv-basis" }, `(${basis})`));
}

function countsLine(accuracy) {
  const parts = Object.entries(accuracy ?? {}).filter(([, n]) => n > 0)
    .map(([key, n]) => t("iv.judge_count", { n, label: t(`iv.judge.${key}`) }));
  return parts.length ? h("p", { class: "meta" }, parts.join(" · ")) : null;
}

/**
 * The draft judgement of how accurate the answers were: per point its
 * judgement and its concepts' (quote, reason, basis), then «Afirmaciones que
 * conviene comprobar», under a visible note that it is a model's draft.
 * `result`: { slots, to_check } of a sheet (AI interviewer) or of a live round.
 */
export function accuracyBlock(result) {
  const judged = (result?.slots ?? []).some((s) => s.judgement || s.points?.some((p) => p.judgement));
  const checks = result?.to_check ?? [];
  // Older sheets and status-only scores were never judged: no empty section.
  if (!result?.judge && !judged && !checks.length) return null;
  const slots = (result?.slots ?? []).filter((s) => s.judgement || s.points?.some((p) => p.judgement || p.status !== "pending"));
  if (!slots.length && !checks.length) return null;
  return h("section", { class: "iv-accuracy", "aria-label": t("iv.judge_title") },
    h("h3", {}, t("iv.judge_title")),
    h("p", { class: "draft-notice", role: "note" }, t("iv.judge_note")),
    h("ol", { class: "iv-skills" }, slots.map((s) => h("li", { class: "iv-skill" },
      h("div", { class: "row wrap iv-skill-head" }, h("span", { class: "iv-skill-label" }, s.label),
        s.judgement ? judgeTag(s.judgement.label) : null),
      countsLine(s.accuracy),
      judgementText(s.judgement),
      s.points?.length ? h("ul", { class: "iv-point-list" }, s.points.map((p) => {
        const label = p.judgement ? p.judgement.label : p.status === "pending" ? "not_treated" : p.status === "unknown" ? "does_not_know" : null;
        return h("li", { class: "iv-point" },
          h("span", { class: "iv-point-text" }, p.text), " ",
          label ? judgeTag(label) : h("span", { class: "iv-point-status" }, t("iv.point.treated")),
          judgementText(p.judgement));
      })) : null))),
    checks.length ? h("div", { class: "iv-checks" },
      h("h3", {}, t("iv.to_check_title")),
      h("p", { class: "hint" }, t("iv.to_check_hint")),
      h("ul", {}, checks.map((c) => h("li", {},
        h("q", {}, c.quote), " ", h("span", { class: "tag" }, t(`iv.check.${c.why}`)),
        c.note ? [" ", h("span", { class: "meta" }, c.note)] : null)))) : null);
}

// ---- after: the report ----------------------------------------------------------

/**
 * The visual report of one interview (a draft for the recruiter), summary
 * first (#230): the suitability, the summary, one bar per point of the
 * plan and where the transcript went; the tiles, notes, confidence,
 * quotes, gaps and facts behind «Más información».
 */
export function reportCard(result) {
  const sheet = result.sheet;
  const figures = reportFigures(sheet);
  const handoff = result.handoff ?? {};
  const suitability = figures.suitability;
  const quotes = figures.byWeight.reduce((n, s) => n + Math.min(3, s.answers?.length ?? 0), 0);
  const more = moreInfo(t("sf.iv_more_hint", { quotes, gaps: figures.gaps.length }), () => [
    h("div", { class: "draft-notice", role: "note" }, h("strong", {}, t("res.draft_title")), " ", t("iv.draft")),
    h("div", { class: "iv-tiles" },
      tile(figures.meets, t("iv.tile.meets"), "good"),
      tile(figures.gaps.length, t("iv.tile.gaps"), "low"),
      tile(figures.above.length, t("iv.tile.above"), "accent"),
      tile(figures.unknown, t("iv.tile.unknown"), "mid"),
      tile(figures.skipped, t("iv.tile.skipped"), "mid")),
    h("ol", { class: "iv-skills" }, figures.byWeight.map((s) => h("li", { class: "iv-skill" },
      h("div", { class: "row wrap iv-skill-head" },
        h("span", { class: "iv-skill-label" }, s.label), levelTag(s.level),
        h("span", { class: "tag" }, t("iv.weight_short", { w: s.weight ?? 3 })),
        s.status === "unknown" || s.status === "skipped" || s.status === "empty" ? statusBadge(s.status) : null,
        s.above ? h("span", { class: "tag accent" }, t("iv.above")) : null),
      h("p", { class: "meta" }, [
        s.notes || null,
        s.confidence !== null && s.confidence !== undefined
          ? t("iv.confidence", { pct: Math.round(s.confidence * 100), by: t(`iv.scored_by.${s.scored_by ?? "status"}`) }) : null,
      ].filter(Boolean).join(" · ")),
      pointList(s.points, { quotes: true }),
      s.answers?.length ? h("ul", { class: "iv-quotes" }, s.answers.slice(0, 3).map((a) => h("li", {}, h("q", {}, a.quote),
        " ", h("span", { class: "meta" }, a.utterance)))) : null))),
    figures.byWeight.some((s) => s.points?.length) ? h("p", { class: "hint" }, t("iv.points_note")) : null,
    figures.gaps.length ? h("div", {}, h("h3", {}, t("iv.gaps")),
      h("ul", { class: "iv-gaps" }, figures.gaps.map((s) => h("li", {}, s.label, " — ", t(`iv.status.${s.status}`))))) : null,
    sheet.other_skills?.length ? h("div", {}, h("h3", {}, t("iv.other_skills")),
      h("ul", { class: "iv-gaps" }, sheet.other_skills.map((o) => h("li", {}, h("strong", {}, o.label), o.quote ? [" ", h("q", {}, o.quote)] : null)))) : null,
    h("p", { class: "meta" }, t("iv.facts", {
      turns: sheet.turns, minutes: Math.round((sheet.elapsed_s ?? 0) / 60), to_min: sheet.turns_to_minimums ?? "–",
      filler: sheet.filler, scorer: sheet.scorer ?? "status",
    })),
  ]);
  return h("section", { class: "card iv-report", "aria-label": t("iv.report") },
    h("h2", {}, t("iv.report_title", { title: sheet.title })),
    h("div", { class: "iv-headline" },
      h("div", { class: `iv-suit is-${scoreClass(suitability)}` },
        h("span", { class: "iv-suit-value" }, suitability === null ? "–" : String(Math.round(suitability))),
        h("span", { class: "iv-suit-label" }, t("iv.suitability"))),
      sheet.summary ? h("p", { class: "iv-summary" }, sheet.summary) : null),
    h("h3", {}, t("iv.by_skill")),
    h("ol", { class: "iv-skills" }, figures.byWeight.map((s) => h("li", { class: "iv-skill" },
      h("div", { class: "row wrap iv-skill-head" }, h("span", { class: "iv-skill-label" }, s.label), levelTag(s.level),
        s.judgement ? judgeTag(s.judgement.label) : null),
      h("div", { class: "row iv-skill-bar" }, scoreBar(s.score),
        h("span", { class: "iv-score" }, s.score === null || s.score === undefined ? "–" : String(s.score))),
      s.points?.length ? h("p", { class: "meta" }, t("iv.points_line", { done: pointCounts(s.points).treated, all: s.points.length })) : null))),
    h("p", { class: "meta" }, handoff.transcript
      ? [t(handoff.analysis ? "iv.handoff.analysis" : "iv.handoff.transcript"), " ",
        h("a", { href: `#/analysis/${encodeURIComponent(sheet.session_id)}` }, t("iv.open_analysis"))]
      : t("iv.handoff.none", { error: handoff.error ?? "" })),
    h("p", { class: "sf-draft" }, t("sf.iv_draft_short")),
    accuracyBlock(sheet),
    more);
}

/**
 * The end of an AI interview on the page where the candidate just spoke
 * (#259 review): only that it ended and a button for the interviewer; the
 * report (scores and the judgement of the answers) is drawn after the click.
 */
export function reportBehindClick(result) {
  const box = h("section", { class: "card iv-report-gate", "aria-label": t("iv.report") });
  const open = h("button", { type: "button", class: "primary", onClick: () => fill(box, reportCard(result)) },
    t("iv.report_open"));
  fill(box, h("h2", {}, t("iv.report_ended")), h("p", { class: "hint" }, t("iv.report_private")), open);
  return box;
}

function tile(value, label, kind) {
  return h("div", { class: `iv-tile is-${kind}` }, h("span", { class: "iv-tile-value" }, String(value)), h("span", { class: "iv-tile-label" }, label));
}

/** The candidates of one plan side by side: suitability and a bar per skill. */
export function comparisonCard(results) {
  const rows = comparisonRows(results);
  const slots = results.plan?.slots ?? [];
  if (!rows.length) return h("p", { class: "hint" }, t("iv.compare_none"));
  return h("div", { class: "iv-compare-wrap" },
    h("table", { class: "iv-compare" },
      h("thead", {}, h("tr", {}, h("th", {}, t("iv.candidate")), h("th", {}, t("iv.suitability")),
        slots.map((s) => h("th", { title: s.label }, h("span", { class: "iv-th" }, s.label), " ", h("span", { class: "tag" }, t("iv.weight_short", { w: s.weight ?? 3 })))))),
      h("tbody", {}, rows.map((r) => h("tr", {},
        h("th", { scope: "row" }, r.candidate || r.session, r.stale ? h("span", { class: "tag" }, t("iv.older_plan")) : null),
        h("td", {}, scoreBar(r.suitability), " ", r.suitability === null ? "–" : String(Math.round(r.suitability))),
        r.slots.map((s) => h("td", {}, scoreBar(s.score, { small: true }), " ", s.score === null || s.score === undefined ? t(`iv.status.${s.status ?? "empty"}`) : String(s.score))))))));
}
