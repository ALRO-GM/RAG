// The "Ayuda" page of the analysis: what it explains, in order, as data
// (no DOM, tested in node). Sentences live in the dictionaries under
// help.*; the competencies come from the rubric (lib/help-rubric.js,
// built from knowledge/rubrics).

import { ACTS, STEPS } from "./session-analysis.js";
import { HELP_RUBRIC } from "./help-rubric.js";

export const INTERACTION = ["turns", "words", "share", "questions", "equality", "dominant", "response"];
export const ACT_EXTRAS = ["weight", "doubtful"];
export const EVENTS = [
  "question_answered", "question_unanswered", "proposal_accepted", "proposal_rejected",
  "proposal_contested", "proposal_ignored", "decision", "action_item", "open_issue",
];
export const CONFIRMATION = ["plus", "minus", "rejected"];
export const RATING = ["scale", "insufficient", "similarity", "firm", "provisional", "observation"];
export const LIMITS = [1, 2, 3, 4, 5, 6, 7];

/** The sections of the page, in order; each id is also its anchor. */
export const SECTIONS = ["steps", "interaction", "acts", "events", "confirmation", "rating", "competencies", "limits"];

/** The element id a «?» link points to: a step, or a section. */
export function anchorOf(target) {
  return STEPS.includes(target) ? `step-${target}` : `help-${target}`;
}

/** The #/help/analysis address, optionally to a step or a section. */
export function helpHref(target) {
  return target ? `#/help/analysis/${target}` : "#/help/analysis";
}

/** The element id to scroll to for a route parameter, or null if unknown. */
export function scrollTarget(param) {
  if (!param) return null;
  if (STEPS.includes(param) || SECTIONS.includes(param)) return anchorOf(param);
  return null;
}

/** The competencies of the rubric in `language` (Spanish when unknown). */
export function competencies(language) {
  return HELP_RUBRIC[language] ?? HELP_RUBRIC.es;
}

/** Every dictionary key the page reads, so a test can check both languages. */
export function helpKeys() {
  const keys = ["help.title", "help.back", "help.link", "help.hint", "help.intro", "help.toc", "help.example"];
  for (const s of SECTIONS) keys.push(`help.sec.${s}`, `help.sec.${s}_intro`);
  for (const id of STEPS) keys.push(`help.step.${id}.t`, `help.step.${id}.d`, `help.step.${id}.e`);
  for (const id of INTERACTION) keys.push(`help.interaction.${id}.t`, `help.interaction.${id}.d`, `help.interaction.${id}.e`);
  for (const id of ACTS) keys.push(`act.${id}`, `help.act.${id}.e`);
  for (const id of ACT_EXTRAS) keys.push(`help.acts.${id}.t`, `help.acts.${id}.d`, `help.acts.${id}.e`);
  for (const id of EVENTS) keys.push(`sa.event_${id}`, `help.event.${id}.d`, `help.event.${id}.e`);
  for (const id of CONFIRMATION) keys.push(`help.confirmation.${id}.t`, `help.confirmation.${id}.d`, `help.confirmation.${id}.e`);
  for (const id of RATING) keys.push(`help.rating.${id}.t`, `help.rating.${id}.d`, `help.rating.${id}.e`);
  keys.push("help.rating.tiers_note");
  for (const id of ["positive", "negative", "helps", "hurts", "limits"]) keys.push(`help.comp.${id}`);
  for (const n of LIMITS) keys.push(`help.limit.${n}`);
  return keys;
}
