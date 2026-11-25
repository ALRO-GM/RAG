// Pure logic of the AI interviewer on the voice-bot page (#208,
// docs/live/VOICEBOT_INTERVIEWER.md): the topics the page offers, editing an
// interview plan (interview.plan.v1) before the interview, the live coverage
// from the orchestrator's feed ("coverage", "interview", "interview_result"
// messages), the report's figures and the comparison of candidates of one
// offer; each point of the plan can carry the concrete concepts a complete
// answer talks about, followed one by one (#258). No DOM and no text here (every text is in the dictionaries), so
// node --test runs it.

export const INTERVIEW = "voicebot.interview";
export const LEVELS = ["minimo", "deseable", "opcional"];
export const PHASES = ["experience", "technical", "behavioural"];
export const ROLE_TYPES = ["tecnico", "comercio", "atencion_cliente", "hosteleria", "almacen", "oficina", "otro"];
export const COMPETENCIES = ["COMM", "LISTEN", "INFLU", "COLLAB", "INCL", "CONFL", "FEEDB", "COORD", "SOLVE", "INIT", "ADAPT"];
export const STATUSES = ["filled", "partial", "empty", "skipped", "unknown"];
export const MIN_MINUTES = 5;
export const MAX_MINUTES = 90;
export const MAX_SLOTS = 12;
export const MIN_OFFER = 40;
export const MAX_POINTS = 6;
export const POINT_CHARS = 80;
export const POINT_STATUSES = ["treated", "pending", "unknown"];

/** The products the voice-bot page can start (the ones that speak). */
export function speakerProducts(products) {
  return (products ?? []).filter((p) => p.kind === "speaker");
}

/** A deep copy of a plan, for editing without touching the saved one. */
export function clonePlan(plan) {
  return JSON.parse(JSON.stringify(plan));
}

function nextId(plan) {
  const used = new Set(plan.slots.map((s) => s.id));
  for (let i = 1; ; i += 1) if (!used.has(`s${i}`)) return `s${i}`;
}

/** Add an empty slot (deseable, weight 3); returns its id, or null when full. */
export function addSlot(plan) {
  if (plan.slots.length >= MAX_SLOTS) return null;
  const id = nextId(plan);
  plan.slots.push({
    id, label: "", kind: "requirement", competency: null, level: "deseable",
    phase: "experience", questions: [""], evidence: "", weight: 3, points: [],
  });
  return id;
}

/** Remove a slot. */
export function removeSlot(plan, id) {
  plan.slots = plan.slots.filter((s) => s.id !== id);
}

// The lowest `p<n>` not taken (ids go up to p99 in the contract).
function lowestFree(current, kept) {
  const taken = new Set([...(current ?? []).map((p) => p.id), ...kept.filter(Boolean)]);
  for (let i = 1; ; i += 1) if (!taken.has(`p${i}`)) return `p${i}`;
}

/**
 * The concepts of a slot from the editor's lines (one per line): trimmed,
 * cut to POINT_CHARS, empty and repeated ones dropped, at most MAX_POINTS.
 * A concept whose text did not change keeps its id; new ones get ids after
 * the highest id the slot has now, so within one edit a removed concept's
 * id is not given to a new one. Across saves an id freed earlier can be
 * given again (the plan keeps no history of ids); a sheet keeps each
 * concept's text, so an older interview still reads right.
 */
export function pointsFromLines(current, lines) {
  const before = new Map((current ?? []).map((p) => [p.text, p.id]));
  const highest = Math.max(0, ...(current ?? []).map((p) => Number(String(p.id).slice(1)) || 0));
  const texts = [];
  for (const line of lines ?? []) {
    const text = String(line).replace(/\s+/g, " ").trim().slice(0, POINT_CHARS).trim();
    if (text && !texts.some((x) => x.toLowerCase() === text.toLowerCase())) texts.push(text);
  }
  let next = highest;
  const given = texts.map((x) => before.get(x));
  return texts.slice(0, MAX_POINTS).map((text) => {
    let id = before.get(text);
    if (!id) {
      next += 1;
      id = next > 99 ? lowestFree(current, given) : `p${next}`;
      given.push(id);
    }
    return { id, text };
  });
}

/** Change fields of a slot; a competency makes it a competency slot. */
export function editSlot(plan, id, fields) {
  const slot = plan.slots.find((s) => s.id === id);
  if (!slot) return;
  Object.assign(slot, fields);
  if ("competency" in fields) {
    slot.competency = COMPETENCIES.includes(fields.competency) ? fields.competency : null;
    slot.kind = slot.competency ? "competency" : "requirement";
    if (slot.competency && slot.phase === "experience") slot.phase = "behavioural";
  }
  if ("weight" in fields) slot.weight = Math.max(1, Math.min(5, Math.round(Number(fields.weight) || 3)));
  if ("questions" in fields) slot.questions = fields.questions.map((q) => q.trim()).filter(Boolean).slice(0, 4);
  if ("points" in fields) slot.points = pointsFromLines(slot.points, fields.points);
}

/** Tick or untick "imprescindible": minimo, or back to deseable. */
export function setEssential(plan, id, on) {
  const slot = plan.slots.find((s) => s.id === id);
  if (!slot) return;
  if (on) slot.level = "minimo";
  else if (slot.level === "minimo") slot.level = "deseable";
}

/** Why a plan cannot be saved, as dictionary keys (empty: it can). */
export function planErrors(plan) {
  const errors = [];
  if (!plan.title?.trim()) errors.push("iv.err.title");
  if (!(plan.minutes >= MIN_MINUTES && plan.minutes <= MAX_MINUTES)) errors.push("iv.err.minutes");
  if (!plan.slots.length) errors.push("iv.err.no_slots");
  if (!plan.slots.some((s) => s.level === "minimo")) errors.push("iv.err.no_minimum");
  if (plan.slots.some((s) => !s.label?.trim())) errors.push("iv.err.label");
  if (plan.slots.some((s) => !s.questions?.some((q) => q.trim()))) errors.push("iv.err.question");
  return errors;
}

/** The plan as sent back to the server (empty questions dropped; no concepts: []). */
export function planForSave(plan) {
  const copy = clonePlan(plan);
  copy.slots = copy.slots.map((s) => ({ ...s, questions: s.questions.map((q) => q.trim()).filter(Boolean), points: s.points ?? [] }));
  return copy;
}

/** How many of a slot's concepts are treated, pending or not known, and in all. */
export function pointCounts(points) {
  const out = Object.fromEntries(POINT_STATUSES.map((s) => [s, 0]));
  for (const point of points ?? []) out[point.status ?? "pending"] = (out[point.status ?? "pending"] ?? 0) + 1;
  return { ...out, total: (points ?? []).length };
}

/** The time budget's sum of minimums, for the review's summary line. */
export function planCounts(plan) {
  const by = Object.fromEntries(LEVELS.map((l) => [l, 0]));
  for (const slot of plan.slots) by[slot.level] = (by[slot.level] ?? 0) + 1;
  return { ...by, total: plan.slots.length };
}

/** Score class of a 0–100 score: good (≥ 70), mid (≥ 40), low, none. */
export function scoreClass(score) {
  if (score === null || score === undefined) return "none";
  if (score >= 70) return "good";
  if (score >= 40) return "mid";
  return "low";
}

/**
 * The report's figures at a glance: suitability, counts, gaps (the slots
 * that are minimo or deseable and under 50, unscored, unknown or skipped),
 * the slots by weight (heaviest first).
 */
export function reportFigures(sheet) {
  const slots = sheet?.slots ?? [];
  const gaps = slots.filter((s) => s.level !== "opcional"
    && (s.score === null || s.score === undefined || s.score < 50 || s.status === "unknown" || s.status === "skipped"));
  return {
    suitability: sheet?.suitability ?? null,
    meets: slots.filter((s) => (s.score ?? 0) >= 70).length,
    gaps,
    above: slots.filter((s) => s.above),
    unknown: slots.filter((s) => s.status === "unknown").length,
    skipped: slots.filter((s) => s.status === "skipped").length,
    total: slots.length,
    byWeight: [...slots].sort((a, b) => (b.weight ?? 3) - (a.weight ?? 3) || LEVELS.indexOf(a.level) - LEVELS.indexOf(b.level)),
  };
}

/** The live interview from the feed: coverage, closing, the final report. */
export class InterviewState {
  constructor() {
    this.coverage = null;
    this.action = null;
    this.closed = false;
    this.closedBy = null;
    this.reporting = false;
    this.result = null;
  }

  /** Apply one feed message (others are ignored). */
  apply(message) {
    const type = message?.type;
    if (type === "coverage") {
      this.coverage = message.coverage ?? this.coverage;
      this.action = message.action ?? null;
    } else if (type === "interview") {
      if (message.state === "closed") {
        this.closed = true;
        this.closedBy = message.closed_by ?? null;
      }
      if (message.state === "reporting") this.reporting = true;
    } else if (type === "interview_result") {
      this.result = message;
      this.reporting = false;
    }
  }

  /** The slots of the live coverage, with the one being asked marked. */
  slots() {
    const coverage = this.coverage;
    if (!coverage) return [];
    return (coverage.slots ?? []).map((s) => ({ ...s, current: s.id === coverage.target }));
  }

  /** filled / partial / empty… counts of the live coverage. */
  counts() {
    const out = Object.fromEntries(STATUSES.map((s) => [s, 0]));
    for (const slot of this.slots()) out[slot.status] = (out[slot.status] ?? 0) + 1;
    return out;
  }
}

/** Where a stored interview report opens (#262). */
export function reportHref(planId, sessionId) {
  return `#/interview-report/${encodeURIComponent(planId)}/${encodeURIComponent(sessionId)}`;
}

/** The comparison of the candidates of one plan: one row each, slots in plan order. */
export function comparisonRows(results) {
  const plan = results?.plan ?? { slots: [] };
  const order = plan.slots.map((s) => s.id);
  return (results?.candidates ?? []).map((row) => {
    const byId = new Map((row.slots ?? []).map((s) => [s.id, s]));
    return {
      session: row.session_id, kind: row.kind === "live" ? "live" : "ai", candidate: row.candidate, suitability: row.suitability ?? null,
      // #262: a report without a candidate's file is kept with the position and deleted from here.
      candidateId: row.candidate_id ?? null,
      finished: row.finished_at, closedBy: row.closed_by, stale: (row.plan_revision ?? 0) !== (plan.revision ?? 0),
      slots: order.map((id) => byId.get(id) ?? { id, score: null, status: "empty" }),
    };
  });
}
