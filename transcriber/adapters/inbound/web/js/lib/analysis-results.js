// The results of a session's analysis as a person in HR reads them: pure
// helpers (no DOM), tested in node. They only rearrange what the RAG
// stored (interaction, events, assessment, reports, synthesis); no score
// is computed or changed here. People are always listed in the order
// they first spoke: the page never ranks them.

import { STEPS, isActive, stepStatus } from "./session-analysis.js";

/** The steps whose results the overview reads. */
export const RESULT_STEPS = ["sentences", "interaction", "events", "assessment", "reports", "synthesis"];

/** How sure a skill's level is, in the page's words. */
export const CERTAINTY = ["firm", "provisional", "insufficient"];

/** Number of distinct person colours (the CSS defines .who-0 … .who-7). */
export const COLOURS = 8;

/** Event types shown in the timeline, in the order of the filter. */
export const TIMELINE_TYPES = [
  "decision", "action_item", "proposal_accepted", "proposal_rejected", "proposal_contested",
  "proposal_ignored", "question_answered", "question_unanswered", "open_issue",
];

/** Conversation types the dictionaries name (the transcript contract's). */
export const CONVERSATION_TYPES = [
  "reunion_decision", "lluvia_ideas", "resolucion_problemas", "negociacion", "entrevista",
  "feedback", "conflicto", "coaching", "venta", "conversacion_libre", "otro",
];

/** The result steps a run has finished, in the order they are read. */
export function doneResultSteps(run) {
  const done = new Set((run?.steps ?? []).filter((s) => stepStatus(s) === "done").map((s) => s.name));
  return RESULT_STEPS.filter((name) => done.has(name));
}

/** Whether there is anything to show a person: the levels or the reports exist. */
export function hasResults(run) {
  const done = doneResultSteps(run);
  return done.includes("assessment") || done.includes("reports");
}

/** A key that changes whenever the stored results may have changed. */
export function resultsKey(run) {
  if (!run) return "";
  const steps = (run.steps ?? []).map((s) => `${s.name}:${s.status}:${s.started_at ?? ""}`).join(",");
  return `${run.finished_at ?? ""}|${steps}`;
}

/** Progress of a run for a plain progress bar: steps done, total, the step running now. */
export function progressOf(run) {
  const steps = run?.steps ?? [];
  const finished = steps.filter((s) => ["done", "unavailable"].includes(stepStatus(s))).length;
  const current = steps.find((s) => stepStatus(s) === "running")?.name ?? null;
  return { done: finished, total: STEPS.length, current, active: isActive(run) };
}

/**
 * When the conversation was recorded, read from the session id the
 * transcriber gives (`2026-11-15-143902-…`): { date: "2026-11-15",
 * time: "14:39" }, or null for ids without a date (simulations).
 */
export function recordedAt(id) {
  const m = /^(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})(?:-|$)/.exec(String(id ?? ""));
  if (!m) return null;
  const [, y, mo, d, hh, mm] = m;
  if (Number(mo) < 1 || Number(mo) > 12 || Number(d) < 1 || Number(d) > 31 || Number(hh) > 23 || Number(mm) > 59) return null;
  return { date: `${y}-${mo}-${d}`, time: `${hh}:${mm}` };
}

/** The date part (YYYY-MM-DD) of an ISO timestamp, or "". */
export function isoDate(value) {
  const m = /^(\d{4}-\d{2}-\d{2})/.exec(String(value ?? ""));
  return m ? m[1] : "";
}

/** How long the conversation lasted (seconds): the end of its last sentence, or null. */
export function sessionLength(sentences) {
  let longest = null;
  for (const s of sentences?.sentences ?? []) {
    const end = typeof s.end === "number" ? s.end : s.start;
    if (typeof end === "number" && Number.isFinite(end) && (longest === null || end > longest)) longest = end;
  }
  return longest;
}

/** "14 min 27 s", "45 s" or "" (the units come from the dictionary). */
export function duration(t, seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "";
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  return minutes ? t("res.minutes_seconds", { m: minutes, s: whole % 60 }) : t("res.seconds", { s: whole });
}

/** Counts of the events by type (every type of TIMELINE_TYPES, 0 when none). */
export function eventCounts(events) {
  const counts = Object.fromEntries(TIMELINE_TYPES.map((type) => [type, 0]));
  for (const e of events?.events ?? []) if (e.type in counts) counts[e.type] += 1;
  return counts;
}

/** The kind of conversation declared when it was recorded, or "" when unknown. */
export function conversationType(data) {
  const kind = data?.assessment?.conversation_type ?? data?.sentences?.conversation_type ?? "";
  return CONVERSATION_TYPES.includes(kind) ? kind : "";
}

/** The facts of the session header. */
export function sessionFacts(id, run, data) {
  const interaction = data?.interaction ?? {};
  const people = (interaction.per_speaker ?? []).length || (data?.reports?.people ?? []).length;
  return {
    recorded: recordedAt(id),
    analysed: isoDate(run?.finished_at ?? run?.started_at),
    seconds: sessionLength(data?.sentences),
    people,
    turns: typeof interaction.turns === "number" ? interaction.turns : null,
    type: conversationType(data),
    goal: typeof data?.sentences?.conversation_goal === "string" ? data.sentences.conversation_goal : "",
    synthetic: Boolean(run?.synthetic ?? data?.reports?.synthetic ?? data?.synthesis?.synthetic),
    status: run ? (isActive(run) ? "running" : run.status ?? "done") : "never",
    job: typeof data?.synthesis?.job === "string" ? data.synthesis.job : (typeof run?.job?.text === "string" ? run.job.text : ""),
    events: eventCounts(data?.events),
  };
}

/**
 * The one-line, plain summary of the session: who took part, how long,
 * and what came out of it (decisions, tasks, open questions). Facts
 * only, never a judgement of anyone.
 */
export function plainSummary(t, facts) {
  const counted = (key, count) => t(count === 1 ? `${key}_one` : key, { count });
  const parts = [counted("res.sum_people", facts.people)];
  const length = duration(t, facts.seconds);
  if (length) parts.push(t("res.sum_length", { length }));
  const e = facts.events;
  const outcome = [];
  if (e.decision) outcome.push(counted("res.sum_decisions", e.decision));
  if (e.action_item) outcome.push(counted("res.sum_actions", e.action_item));
  const open = e.question_unanswered + e.open_issue;
  if (open) outcome.push(counted("res.sum_open", open));
  const head = parts.join(" ");
  return outcome.length ? t("res.sum_with", { head, outcome: outcome.join(", ") }) : t("res.sum_plain", { head });
}

/** How sure the level of one competency is: firm, provisional or insufficient. */
export function certaintyOf(item) {
  if (typeof item?.rating === "number") return "firm";
  if (item?.status === "provisional" && typeof item.estimate?.rating === "number") return "provisional";
  return "insufficient";
}

/** The rubric's 1–5 scale point a level falls on (null when there is none). */
export function scalePoint(rating) {
  if (typeof rating !== "number" || !Number.isFinite(rating)) return null;
  return Math.min(5, Math.max(1, Math.round(rating)));
}

/** A quote as the page shows it: text, moment, sign and indicator. */
function quoteOf(e, source = "") {
  return {
    text: e.quote ?? e.text ?? "",
    start: typeof e.start === "number" ? e.start : null,
    turn: typeof e.turn === "number" ? e.turn : null,
    positive: e.polarity !== "negative",
    indicator: e.indicator_id ?? "",
    source,
  };
}

/**
 * One person's competencies in the rubric's order: the level and how
 * sure it is, the judge's explanation, the quotes that support it and
 * the observations to check. `names` maps a competency id to its name
 * and definition in the page's language.
 */
export function skillsOf(card, names = {}) {
  return (card?.assessments ?? []).map((a) => {
    const certainty = certaintyOf(a);
    const rating = certainty === "firm" ? a.rating : certainty === "provisional" ? a.estimate.rating : null;
    const known = names[a.competency_id] ?? {};
    return {
      id: a.competency_id,
      name: known.name ?? a.competency_id,
      definition: known.definition ?? "",
      certainty,
      rating,
      point: scalePoint(rating),
      confidence: certainty === "provisional" ? a.estimate?.confidence ?? "low" : "",
      rationale: a.rationale ?? "",
      quotes: (a.evidence ?? []).map((e) => quoteOf(e)).sort(byMoment),
      observations: (a.observations ?? []).map((o) => quoteOf(o, o.source ?? "")).sort(byMoment),
    };
  });
}

function byMoment(a, b) {
  return (a.start ?? a.turn ?? 0) - (b.start ?? b.turn ?? 0);
}

/** How many skills of each certainty a person has. */
export function certaintyCounts(skills) {
  const counts = { firm: 0, provisional: 0, insufficient: 0 };
  for (const s of skills) counts[s.certainty] += 1;
  return counts;
}

const VERDICTS = ["strong", "adequate", "weak"];

/** One skill opinion as the page reads it (or null when the data has none). */
function skillOpinionOf(o) {
  if (!o || typeof o.score !== "number") return null;
  return {
    id: o.competency_id ?? "",
    name: o.name ?? "",
    score: o.score,
    verdict: VERDICTS.includes(o.verdict) ? o.verdict : "adequate",
    conclusion: o.conclusion ?? "",
    basis: o.basis ?? "participation",
    confidence: ["low", "medium", "high"].includes(o.confidence) ? o.confidence : "low",
    caveats: Array.isArray(o.caveats) ? o.caveats : [],
    cites: (o.cites ?? []).map((c) => quoteOf(c)),
    source: o.source ?? "rules",
  };
}

/**
 * The opinion on one person (synthesis format_version 2, additive): an
 * overall score and verdict line, the opinion on every skill and the
 * main strengths (at most 3) and improvements (at most 2). The headline
 * is `overall.verdict` and `overall.line`; `confidence` and `caveats`
 * are for the «Más información» section, never the headline. Older
 * stored results have none of it: `overall` is then null and the lists
 * are empty.
 */
export function opinionOf(profile) {
  const skills = (profile?.skills ?? []).map(skillOpinionOf).filter(Boolean);
  const byId = new Map(skills.map((k) => [k.id, k]));
  const pick = (ids) => (ids ?? []).map((id) => byId.get(id)).filter(Boolean);
  const o = profile?.overall;
  const maximum = typeof o?.maximum === "number" ? o.maximum : 5;
  const minimum = typeof o?.minimum === "number" ? o.minimum : 1;
  const overall = o && typeof o.score === "number" ? {
    score: o.score,
    maximum,
    minimum,
    percent: typeof o.percent === "number" ? o.percent : Math.round(((o.score - minimum) / (maximum - minimum)) * 1000) / 10,
    verdict: VERDICTS.includes(o.verdict) ? o.verdict : "adequate",
    line: o.line ?? "",
    frame: o.frame ?? "",
    confidence: ["low", "medium", "high"].includes(o.confidence) ? o.confidence : "low",
    caveats: Array.isArray(o.caveats) ? o.caveats : [],
    source: o.source ?? "rules",
  } : null;
  return {
    overall,
    skills,
    byId,
    topStrengths: pick(profile?.top_strengths).slice(0, 3),
    topImprovements: pick(profile?.top_improvements).slice(0, 2),
    conversationType: typeof profile?.conversation_type === "string" ? profile.conversation_type : "",
  };
}

/**
 * Everything the page knows about each person, in the order they first
 * spoke: participation, what the reports list as strengths and areas to
 * develop, the competencies, the profile and their part in the events.
 */
export function peopleOf(data, names = {}) {
  const interaction = data?.interaction?.per_speaker ?? [];
  const order = interaction.map((p) => p.speaker);
  for (const p of data?.reports?.people ?? []) if (!order.includes(p.name)) order.push(p.name);
  for (const c of data?.assessment?.scorecards ?? []) if (!order.includes(c.participant)) order.push(c.participant);
  const byName = (list, key) => new Map((list ?? []).map((x) => [x[key], x]));
  const stats = byName(interaction, "speaker");
  const reports = byName(data?.reports?.people, "name");
  const cards = byName(data?.assessment?.scorecards, "participant");
  const profiles = byName(data?.synthesis?.people, "participant");
  const events = byName(data?.events?.summary?.per_person, "speaker");
  return order.map((name, index) => {
    const s = stats.get(name) ?? {};
    const report = reports.get(name) ?? {};
    const skills = skillsOf(cards.get(name), names);
    const profile = profiles.get(name) ?? null;
    const opinion = opinionOf(profile);
    for (const skill of skills) skill.opinion = opinion.byId.get(skill.id) ?? null;
    return {
      name,
      index,
      colour: index % COLOURS,
      position: report.position ?? index + 1,
      share: typeof s.word_share === "number" ? s.word_share : null,
      turns: typeof s.turns === "number" ? s.turns : null,
      words: typeof s.words === "number" ? s.words : null,
      seconds: typeof s.seconds === "number" ? s.seconds : null,
      questions: typeof s.questions === "number" ? s.questions : null,
      overlaps: typeof s.overlapping_starts === "number" ? s.overlapping_starts : null,
      strengths: report.strengths ?? [],
      toDevelop: report.to_improve ?? [],
      why: report.why ?? [],
      skills,
      certainty: certaintyCounts(skills),
      quotes: skills.reduce((n, k) => n + k.quotes.length, 0),
      profile,
      opinion,
      confidence: ["low", "medium", "high"].includes(profile?.confidence) ? profile.confidence : "",
      events: events.get(name) ?? null,
    };
  });
}

/** The events in the order they happened, each with its moment and who took part. */
export function timeline(events, types = TIMELINE_TYPES) {
  return (events?.events ?? [])
    .filter((e) => types.includes(e.type))
    .map((e) => ({
      id: e.id ?? "",
      type: e.type,
      start: typeof e.start === "number" ? e.start : null,
      order: typeof e.order === "number" ? e.order : 0,
      by: e.by ?? "",
      people: e.people ?? [],
      owner: e.owner ?? "",
      text: e.what || e.text || "",
      deadline: e.deadline ?? "",
    }))
    .sort((a, b) => (a.start ?? 0) - (b.start ?? 0) || a.order - b.order);
}

/**
 * The sentence of the transcript a quote's moment points to: the last
 * one starting at or before `seconds` (null when there are none).
 */
export function sentenceAt(sentences, seconds) {
  let found = null;
  for (const s of sentences?.sentences ?? []) {
    if (typeof s.start !== "number") continue;
    if (s.start <= seconds + 0.5) found = s;
    else break;
  }
  return found;
}

/** The DOM id of a transcript sentence on the page. */
export function sentenceAnchor(sentence) {
  return `line-${String(sentence?.id ?? "").replace(/[^\w-]/g, "_")}`;
}
