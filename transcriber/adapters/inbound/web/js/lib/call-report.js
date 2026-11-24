// Pure logic of the call report page (#/call-report/<demo>, #250): the
// report of the scripted support call, joined to the sentences it points
// at so every part can say «línea 5» and quote it. No DOM here, so node
// --test can run it. In the demo the report is written by hand; nothing
// here measures anybody.

import { lineNumbers } from "./call-demo.js";

export const EFFECTS = ["helped", "neutral", "hurt"];
export const STEP_STATUSES = ["followed", "reordered", "skipped"];
export const OUTCOMES = ["said", "said_late", "never"];

/** `m:ss` of a time in seconds. */
export function mmss(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The report's view of the call: the numbered, named sentences. */
export function callLines(data) {
  const numbers = lineNumbers(data?.utterances);
  const names = new Map((data?.speakers ?? []).map((s) => [s.id, s.name || s.id]));
  const lines = (data?.utterances ?? []).map((u) => ({
    id: u.id, n: numbers.get(u.id), speaker: u.speaker, name: names.get(u.speaker) ?? u.speaker,
    start: u.start, time: mmss(u.start), text: u.text,
  }));
  // The first speaker of a call is the agent (the product's first channel).
  return { lines, byId: new Map(lines.map((l) => [l.id, l])), agent: data?.speakers?.[0]?.id ?? null };
}

/** The line of an id, or null: the reference a chip points at. */
export function lineOf(call, id) {
  return call.byId.get(id) ?? null;
}

/** Count of the lines by what they did: { helped, neutral, hurt }. */
export function effectCounts(report) {
  const counts = { helped: 0, neutral: 0, hurt: 0 };
  for (const review of report?.lines_review ?? []) if (review.effect in counts) counts[review.effect] += 1;
  return counts;
}

/** Count of the script's steps by status: { followed, reordered, skipped }. */
export function stepCounts(report) {
  const counts = { followed: 0, reordered: 0, skipped: 0 };
  for (const step of report?.script?.steps ?? []) if (step.status in counts) counts[step.status] += 1;
  return counts;
}

/** Count of the mandatory items by what became of them. */
export function outcomeCounts(report) {
  const counts = { said: 0, said_late: 0, never: 0 };
  for (const item of report?.mandatory ?? []) if (item.outcome in counts) counts[item.outcome] += 1;
  return counts;
}

/**
 * The text of a mandatory item: from the demo's list of items when it is there
 * (the one the live page showed), else the id.
 */
export function itemText(id, items, language = "es") {
  const item = (items ?? []).find((i) => i.id === id);
  return (language === "en" && item?.en?.text) || item?.text || id;
}

/** Whether the report may be drawn: its demo flag is set and it has the five parts. */
export function isDemoReport(data) {
  return Boolean(data?.demo === true && data.summary && data.lines_review?.length && data.script && data.top_scorers && data.mandatory);
}

/**
 * A count sentence with the right singular or plural of each part:
 * `parts` are [key, n] and the dictionary has `<key>_one` and `<key>_other`
 * («1 perjudicó», «2 perjudicaron»); the sentence is built by `join`.
 */
export function countParts(t, parts) {
  return parts.map(([key, n]) => t(`${key}_${n === 1 ? "one" : "other"}`, { n }));
}
