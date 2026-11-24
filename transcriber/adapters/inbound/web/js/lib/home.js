// Pure logic of the home page (#/) and of the Administración overview
// (#/admin): the three services, the admin tools, the latest sessions and
// the one-line conclusion read from a session's stored synthesis. No DOM
// here, so node --test can run it. Nothing is scored or ranked here: a
// conclusion is a sentence the analysis already wrote.

import { sortSessions } from "./sessions.js";
import { stepStatus } from "./session-analysis.js";
import { hasResults, opinionOf } from "./analysis-results.js";

/**
 * The three services of the home, in the order they are shown; `demo` is a sample to watch (#232)
 * and `more` other samples of the same service (#250: the sample call-centre call).
 */
export const SERVICES = [
  { id: "analyse", href: "#/sessions", icon: "analysis", demo: "#/call-report/llamada-lumbre" },
  { id: "live", href: "#/live", icon: "live", demo: "#/live/demo", more: [{ href: "#/live/demo/support", key: "home.live.demo_support" }] },
  { id: "voicebot", href: "#/voicebot", icon: "voicebot", demo: "#/voicebot/demo" },
];

/** The tools of the Administración area, in the order they are shown. */
export const ADMIN_TOOLS = [
  { id: "settings", href: "#/settings", icon: "settings" },
  { id: "enrol", href: "#/enrol", icon: "enrol" },
  { id: "knowledge", href: "#/knowledge", icon: "knowledge" },
  { id: "lab", href: "#/lab", icon: "lab" },
  { id: "study", href: "study.html", icon: "study" },
  { id: "research", href: "research.html", icon: "research" },
];

/** How many sessions the home lists. */
export const LATEST = 4;

/** Longest conclusion shown on one line, in characters. */
export const CONCLUSION_MAX = 140;

/** The `n` most recent sessions, newest first. */
export function latestSessions(sessions, n = LATEST) {
  return sortSessions(Array.isArray(sessions) ? sessions : []).slice(0, Math.max(0, n));
}

/** Whether a session may have an analysis worth asking for (its transcript is done). */
export function mayHaveAnalysis(session) {
  return session?.state === "done";
}

/** Whether a stored analysis run has results a person can read (levels or reports). */
export function isAnalysed(run) {
  return hasResults(run);
}

/** Whether the stored run of an analysis has its synthesis written. */
export function hasSynthesis(run) {
  return (run?.steps ?? []).some((s) => s.name === "synthesis" && stepStatus(s) === "done");
}

/** Where a row of the latest sessions leads: the analysis when there is one, else the session. */
export function sessionHref(session, analysed) {
  const id = encodeURIComponent(session?.id ?? "");
  return analysed ? `#/analysis/${id}` : `#/s/${id}`;
}

/** Words after which a full stop does not end the sentence (ES and EN, lower case, no dot). */
export const ABBREVIATIONS = new Set([
  "sr", "sra", "srta", "dr", "dra", "d", "dña", "ud", "uds", "vd", "etc", "ej", "p", "pág", "núm", "nº", "aprox", "dpto", "tel", "vs",
  "mr", "mrs", "ms", "prof", "st", "jr", "e.g", "i.e", "approx", "dept", "fig",
]);

/** The text up to the first full stop, question or exclamation mark that ends a sentence. */
export function firstSentence(text) {
  const pattern = /[.!?](?=\s|$)/g;
  let m;
  while ((m = pattern.exec(text)) !== null) {
    if (m[0] === ".") {
      const word = text.slice(0, m.index).split(/\s/).pop().replace(/^[«"'(¿¡]+/, "");
      if (word.length === 1 || ABBREVIATIONS.has(word.toLowerCase())) continue;
    }
    return text.slice(0, m.index + 1);
  }
  return text;
}

/** The first sentence of `text`, cut at `max` characters with an ellipsis. */
export function oneLine(text, max = CONCLUSION_MAX) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  if (!flat) return "";
  const sentence = firstSentence(flat);
  if (sentence.length <= max) return sentence;
  const cut = sentence.slice(0, max - 1);
  const space = cut.lastIndexOf(" ");
  return `${(space > max / 2 ? cut.slice(0, space) : cut).replace(/[\s,;:.]+$/, "")}…`;
}

/**
 * The first sentence of `text`, whole when it fits in `max` characters;
 * else cut at its last clause boundary (, ; : —) within `max` and closed
 * with a full stop, never in the middle of a clause (nor inside an open
 * bracket or quote) and with no ellipsis; "" when no boundary falls in the second two thirds (#230).
 */
export function clauseCut(text, max = CONCLUSION_MAX) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  if (!flat) return "";
  const sentence = firstSentence(flat);
  if (sentence.length <= max) return sentence;
  const head = sentence.slice(0, max);
  const marks = [...head.matchAll(/\s?[,;:\u2014\u2013](?=\s)/g)];
  // A clause that leaves a bracket or a quote open is not whole.
  const closed = (text) => [["(", ")"], ["«", "»"], ["“", "”"], ["[", "]"]]
    .every(([a, b]) => text.split(a).length === text.split(b).length) && (text.split('"').length - 1) % 2 === 0;
  const last = marks.filter((m) => m.index >= max / 3 && closed(head.slice(0, m.index))).pop();
  if (!last) return "";
  return `${head.slice(0, last.index).replace(/[\s,;:.]+$/, "")}.`;
}

/**
 * The one-line conclusion (whole sentence or clause, never cut mid-clause)
 * of a synthesis result for the first person
 * profiled: their overall line (synthesis v2) with its percentage, one
 * decimal (`percent`, only when stored), else the first point on the job
 * fit, else their first strength; null when there is none.
 * `{ person, text[, percent] }`.
 */
export function conclusionOf(synthesis) {
  for (const person of synthesis?.people ?? []) {
    if (person?.status === "skipped") continue;
    const overall = opinionOf(person).overall;
    const point = [...(person?.fit ?? []), ...(person?.strengths ?? [])].find((p) => String(p?.text ?? "").trim());
    // Whole sentences only: a clause when the sentence is too long, else the sentence.
    const short = (value) => clauseCut(value) || firstSentence(String(value).replace(/\s+/g, " ").trim());
    const text = (overall?.line ? short(overall.line) : "") || (point ? short(point.text) : "");
    if (!text && !overall) continue;
    const found = { person: String(person.participant ?? ""), text };
    if (overall && Number.isFinite(overall.percent)) found.percent = Math.round(overall.percent * 10) / 10;
    return found;
  }
  return null;
}

/**
 * What the home shows of `session`'s analysis: `{ analysed, conclusion }`,
 * or null when there is none to ask for or the RAG did not answer.
 * `api` has analysisStatus(id) and analysisStep(id, step). A synthesis
 * that cannot be read keeps the row analysed, without a conclusion.
 */
export async function analysisFacts(api, session) {
  if (!mayHaveAnalysis(session)) return null;
  let run;
  try {
    run = await api.analysisStatus(session.id);
  } catch {
    return null; // no analysis, or the RAG is away: the row shows the session alone
  }
  const analysed = isAnalysed(run);
  if (!analysed || !hasSynthesis(run)) return { analysed, conclusion: null };
  try {
    return { analysed, conclusion: conclusionOf(await api.analysisStep(session.id, "synthesis")) };
  } catch {
    return { analysed, conclusion: null };
  }
}

/** The state of the service from a health check: "ok" when it answered, else "down". */
export function serviceState(answer) {
  return answer?.status === "ok" ? "ok" : "down";
}
