// Summary first (#230): what a worker reads before anything else on a
// session's result. Pure helpers (no DOM), tested in node. They pick and
// shorten what the analysis stored (synthesis v2 `overall`, `skills`,
// `top_strengths`, `top_improvements`; docs/skills/DESIGN.md §14.1) and
// keep the rest (confidence, caveats, quotes) for «Más información».
// Nothing here ranks people: they stay in the order they first spoke.

import { clauseCut, firstSentence } from "./home.js";

/** At most this many strengths and improvements on the summary. */
export const MAX_STRENGTHS = 3;
export const MAX_IMPROVEMENTS = 2;

/** At most this many sentences in a person's short summary. */
export const SUMMARY_SENTENCES = 2;

/** Longest sentence of that summary, in characters (cut at a word with an ellipsis). */
export const SENTENCE_MAX = 180;

/** Longest strength or improvement sentence on the summary, in characters. */
export const POINT_MAX = 120;

/** Radius of the score ring (SVG user units; the viewBox is 0 0 120 120). */
export const RING_RADIUS = 52;

/** Length of the ring's circle, in the same units. */
export const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

/** The verdicts of synthesis v2, highest first. */
export const VERDICTS = ["strong", "adequate", "weak"];

/** Caveat codes the dictionaries explain (others are shown as they are). */
export const CAVEATS = [
  "provisional", "observations_only", "no_quotes", "short_participation", "no_firm_levels", "judge_unavailable",
];

/** A percentage kept between 0 and 100 (null when it is not a number). */
export function clampPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.min(100, Math.max(0, value));
}

/** A score on a scale as a percentage: the lowest score is 0 %, one decimal. */
export function scalePercent(score, minimum = 1, maximum = 5) {
  if (typeof score !== "number" || !Number.isFinite(score) || !(maximum > minimum)) return null;
  return clampPercent(Math.round(((score - minimum) / (maximum - minimum)) * 1000) / 10);
}

/**
 * The two dash lengths of the ring's coloured arc for `percent` (SVG
 * `stroke-dasharray`: filled, then the rest of the circle), two decimals.
 */
export function ringDash(percent) {
  const share = (clampPercent(percent) ?? 0) / 100;
  const filled = Math.round(RING_LENGTH * share * 100) / 100;
  const total = Math.round(RING_LENGTH * 100) / 100;
  return { filled, rest: Math.round((total - filled) * 100) / 100, total };
}

/**
 * Which words describe a verdict, by the kind of conversation: a job
 * interview speaks of the role, coaching of what was learnt, a sale of
 * the customer, anything else of the conversation.
 */
export function verdictFamily(conversationType) {
  if (conversationType === "entrevista") return "job";
  if (conversationType === "coaching") return "coaching";
  if (conversationType === "venta") return "sale";
  return "general";
}

/** The dictionary key of the descriptive label of a verdict (never a decision). */
export function verdictKey(verdict, conversationType) {
  const known = VERDICTS.includes(verdict) ? verdict : "adequate";
  return `sf.verdict.${verdictFamily(conversationType)}.${known}`;
}

/** The sentences of a text, in order (a stop after «Sr.» or «p. ej.» does not end one). */
export function sentencesOf(text) {
  let rest = String(text ?? "").replace(/\s+/g, " ").trim();
  const found = [];
  while (rest) {
    const sentence = firstSentence(rest).trim();
    if (!sentence) break;
    found.push(sentence);
    rest = rest.slice(sentence.length).trim();
  }
  return found;
}

/**
 * A short summary from several texts: the first sentence of each, in
 * order, without repeating one already said (nor `skip`), at most `max`.
 * Whole sentences only: a sentence over SENTENCE_MAX is cut at a clause
 * boundary (clauseCut) or left out; the full texts stay under «Más
 * información».
 */
export function shortSummary(texts, { max = SUMMARY_SENTENCES, skip = [] } = {}) {
  const seen = new Set(skip.map((s) => normalise(s)).filter(Boolean));
  const picked = [];
  for (const text of texts) {
    const sentence = clauseCut(text, SENTENCE_MAX);
    const key = normalise(sentence);
    if (!sentence || seen.has(key) || [...seen].some((s) => s.includes(key) || key.includes(s))) continue;
    seen.add(key);
    picked.push(sentence);
    if (picked.length >= max) break;
  }
  return picked.join(" ");
}

function normalise(text) {
  return String(text ?? "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

function pointTexts(list) {
  return (list ?? []).map((p) => (typeof p === "string" ? p : p?.text ?? "")).filter((s) => String(s).trim());
}

/** Whether a person's stored synthesis predates the opinion (synthesis v1): a profile, no `overall`. */
export function isLegacy(person) {
  return Boolean(person?.profile) && person.profile.status !== "skipped" && !person.opinion?.overall;
}

/**
 * One bar per competency: its level as a percentage of the scale and,
 * when the data names one, the minimum the role asks for (`required`,
 * on the same scale). v2 reads the opinion of every skill; older results
 * the firm or provisional level of the assessment (nothing for "no level").
 */
export function barsOf(person) {
  const overall = person?.opinion?.overall;
  const minimum = overall?.minimum ?? 1;
  const maximum = overall?.maximum ?? 5;
  const names = new Map((person?.skills ?? []).map((s) => [s.id, s.name]));
  const opinions = person?.opinion?.skills ?? [];
  if (opinions.length) {
    return opinions.map((o) => ({
      id: o.id,
      name: names.get(o.id) ?? (o.name || o.id),
      percent: scalePercent(o.score, minimum, maximum),
      verdict: o.verdict,
      required: typeof o.required === "number" ? scalePercent(o.required, minimum, maximum) : null,
    }));
  }
  return (person?.skills ?? []).filter((s) => typeof s.rating === "number").map((s) => ({
    id: s.id,
    name: s.name,
    percent: scalePercent(s.rating, 1, 5),
    verdict: s.rating >= 3.75 ? "strong" : s.rating >= 2.5 ? "adequate" : "weak",
    required: null,
  }));
}

/**
 * Strengths or improvements as `{ name, text }`: the competency and the
 * first sentence of its conclusion when it fits (or a whole clause of it);
 * otherwise the name alone. The whole conclusion is in «Más información».
 */
function picked(list, names) {
  return list.map((o) => ({ id: o.id, name: names.get(o.id) ?? (o.name || o.id), text: clauseCut(o.conclusion, POINT_MAX) }));
}

/** At most this many bars before «Ver las N competencias». */
export const MAX_BARS = 5;

/**
 * The bars shown up front and the full list: those of the strengths and
 * improvements named on the summary, in that order (matched by id, or by
 * name for older results), at most MAX_BARS. `all` keeps every bar.
 */
export function splitBars(bars, points) {
  const shown = [];
  for (const point of points) {
    const bar = bars.find((b) => (point.id && b.id === point.id) || (!point.id && b.name === point.name));
    if (bar && !shown.includes(bar)) shown.push(bar);
    if (shown.length >= MAX_BARS) break;
  }
  return { shown, all: bars };
}

/** The caveats of a person's opinion: the overall ones, then each skill's, once each. */
export function caveatsOf(opinion) {
  const codes = [...(opinion?.overall?.caveats ?? [])];
  for (const skill of opinion?.skills ?? []) for (const code of skill.caveats ?? []) codes.push(code);
  return [...new Set(codes.filter((c) => typeof c === "string" && c))];
}

/**
 * What the summary shows of one person (an entry of `peopleOf`), and the
 * counts behind «Más información». `overall` is null for older results
 * (`legacy` then says whether the synthesis could be redone to get one).
 */
export function personSummary(person) {
  const opinion = person?.opinion ?? {};
  const overall = opinion.overall ?? null;
  const names = new Map((person?.skills ?? []).map((s) => [s.id, s.name]));
  const profile = person?.profile && person.profile.status !== "skipped" ? person.profile : null;
  const line = overall?.line ? sentencesOf(overall.line)[0] ?? overall.line : "";
  const summary = shortSummary([...pointTexts(profile?.fit), ...pointTexts(profile?.strengths)], { skip: [line] });
  const strengths = (opinion.topStrengths ?? []).length
    ? picked(opinion.topStrengths, names).slice(0, MAX_STRENGTHS)
    : (person?.strengths ?? []).slice(0, MAX_STRENGTHS).map((name) => ({ id: "", name, text: "" }));
  const improvements = (opinion.topImprovements ?? []).length
    ? picked(opinion.topImprovements, names).slice(0, MAX_IMPROVEMENTS)
    : (person?.toDevelop ?? []).slice(0, MAX_IMPROVEMENTS).map((name) => ({ id: "", name, text: "" }));
  const quotes = (person?.quotes ?? 0) + (person?.skills ?? []).reduce((n, s) => n + (s.observations?.length ?? 0), 0);
  const caveats = caveatsOf(opinion);
  const bars = barsOf(person);
  return {
    name: person?.name ?? "",
    colour: person?.colour ?? 0,
    overall: overall ? {
      percent: clampPercent(overall.percent),
      verdict: overall.verdict,
      line,
      confidence: overall.confidence,
    } : null,
    legacy: isLegacy(person),
    summary,
    strengths,
    improvements,
    bars,
    shownBars: splitBars(bars, [...strengths, ...improvements]).shown,
    more: {
      quotes,
      caveats,
      confidence: overall?.confidence || person?.confidence || "",
      notices: caveats.length + (person?.certainty?.provisional ?? 0) + (person?.why?.length ? 1 : 0),
    },
  };
}

/** Whether any person's results predate the opinion (offer to redo the synthesis). */
export function anyLegacy(summaries) {
  return (summaries ?? []).some((s) => s.legacy);
}

/** The dictionary key of a caveat code, or null for one the dictionaries do not know. */
export function caveatKey(code) {
  return CAVEATS.includes(code) ? `sf.caveat.${code}` : null;
}
