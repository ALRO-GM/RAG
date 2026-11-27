// Coverage of a live interview (#286), pure: no DOM here, so node --test runs it.
//
// The server computes each topic's level (none → named → treated → well
// treated; it only grows) and, per concept, whether its question was asked
// apart from whether the candidate treated it (orchestrator/domain/
// interview_coverage.py). This module turns those facts into what the page,
// the end summary and the report say: the ring's fill, the words, which
// level counts as covered, the changes to announce once. A level says how
// much a topic was talked about, never how good the answers were.

import { LEVELS } from "./live.js";

/** Fill of the ring per level: its own fraction each, never colour alone. */
export const FILL = { none: 0, named: 1 / 3, treated: 2 / 3, well_treated: 1 };
/** The lowest level counted in «{done} de {all}». */
export const COVERED_LEVEL = "treated";

/** The level of a topic chip, or null when the server sent none (script phases, older feeds). */
export function levelOf(chip) {
  return LEVELS.includes(chip?.coverage?.level) ? chip.coverage.level : null;
}

/** Whether a level counts as covered (treated or well treated). */
export function counted(level) {
  return LEVELS.indexOf(level) >= LEVELS.indexOf(COVERED_LEVEL);
}

/** How many chips count as covered: by level when there is one, else the old `done`. */
export function coveredCount(chips) {
  return (chips ?? []).filter((c) => (levelOf(c) ? counted(levelOf(c)) : c.done)).length;
}

/**
 * A concept's state in words (#286), asked and treated apart:
 * asked_answered, asked_unknown, asked (no answer yet), treated (not asked),
 * unknown (not asked), pending (neither).
 */
export function pointState(point) {
  const status = ["treated", "unknown"].includes(point?.status) ? point.status : "pending";
  if (point?.asked) return status === "treated" ? "asked_answered" : status === "unknown" ? "asked_unknown" : "asked";
  return status;
}

/** Whether a concept is covered: asked, or treated / said unknown. */
export function pointCovered(point) {
  return Boolean(point?.asked) || ["treated", "unknown"].includes(point?.status);
}

/**
 * The levels that changed since `before` (Map id -> level), to announce
 * once; updates `before`. The first reading announces nothing.
 */
export function levelChanges(before, chips) {
  const first = before.size === 0;
  const changed = [];
  for (const chip of chips ?? []) {
    const level = levelOf(chip);
    if (!level) continue;
    const was = before.get(chip.id);
    before.set(chip.id, level);
    if (!first && was !== undefined && was !== level) changed.push({ chip, level });
  }
  return changed;
}

/** First letter in lower case, unless it reads as an acronym («API»). */
export function lowerFirst(text) {
  const s = String(text ?? "");
  return /^[A-ZÁÉÍÓÚÑ]{2}/.test(s) ? s : s.charAt(0).toLowerCase() + s.slice(1);
}

/**
 * Counts of a round's coverage (the hand-off's or a report's `coverage`):
 * topics per level, concepts asked (and by hand) of all.
 */
export function coverageCounts(coverage) {
  const out = { none: 0, named: 0, treated: 0, well_treated: 0, topics: 0, asked: 0, hand: 0, concepts: 0 };
  for (const topic of Array.isArray(coverage) ? coverage : []) {
    if (!LEVELS.includes(topic?.level)) continue;
    out[topic.level] += 1;
    out.topics += 1;
    for (const point of Array.isArray(topic.points) ? topic.points : []) {
      out.concepts += 1;
      if (point?.asked) out.asked += 1;
      if (point?.asked?.by === "hand") out.hand += 1;
    }
  }
  return out;
}
