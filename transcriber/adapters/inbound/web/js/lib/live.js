// Pure logic of the live page (#/live): the Server-Sent Events parser, the
// session's state built from the orchestrator's feed (live.event.v1 events
// and suggestion records), the script-phase coverage of interviews and the
// customer signals of support calls. No DOM here, so node --test runs it.
//
// Rule (docs/platform/ARCHITECTURE.md §8): signals are shown only in the
// support product, as what the customer *says* with the quoted sentence;
// an interview never shows signals, ratings or labels about the candidate.
//
// LV-16 (#206): the employee and the other party, names given during the
// session, which side's lines are shown, how many suggestions stay open,
// the assistance mode and the mandatory items (docs/live/EMPLOYEE_UX.md).
//
// LV-17 (#232): the calm page shows one thing at a time. `now()` picks the
// single «Ahora» card (a line better avoided first, then the newest
// suggestion not yet used); `used()` tells when the employee asked what a
// suggestion proposed («Hecho»); the topics come from a scripted demo or
// from the script phases; lines better avoided arrive as `avoid` flags
// (a hint about the question, never about the candidate).

export const SUPPORT = "live_assist.support";
export const INTERVIEW = "live_assist.interview";
export const AUDIO = ["stereo", "mic"];
export const SPEEDS = [1, 2, 4];
export const MODES = ["pincelada", "guiado"];
export const MAX_SUGGESTIONS = [1, 2, 3, 5];
/** Roles of the person the page helps (the rest is the other party). */
export const EMPLOYEE_ROLES = ["agent", "interviewer"];
/** Customer signals that mean the customer is not comfortable. */
export const DISCOMFORT = ["complaint", "stated_frustration", "churn_risk", "urgency"];
/** Where the page remembers this browser's view choices. */
export const PREFS_KEY = "ragputin.live.view";
/** Share of a suggestion's example words the employee must say for «Hecho». */
export const ECHO_SHARE = 0.6;
/** Words this short do not count when comparing an example with a line. */
const SHORT_WORD = 4;

/**
 * The view's choices: { showEmployee, showOther, max, mode }. The saved
 * settings («Ajustes», settings.v1 choices) give the defaults; what this
 * browser chose last (`stored`, JSON) wins. Anything unreadable is ignored.
 */
export function viewPrefs(choices = {}, stored = null) {
  const out = {
    showEmployee: choices?.["live.show_employee"] !== "off",
    showOther: choices?.["live.show_other"] !== "off",
    max: MAX_SUGGESTIONS.includes(Number(choices?.["live.max_suggestions"])) ? Number(choices["live.max_suggestions"]) : 3,
    mode: MODES.includes(choices?.["live.assist_mode"]) ? choices["live.assist_mode"] : MODES[0],
  };
  let kept = null;
  try {
    kept = stored ? JSON.parse(stored) : null;
  } catch {
    kept = null;
  }
  if (kept && typeof kept === "object") {
    if (typeof kept.showEmployee === "boolean") out.showEmployee = kept.showEmployee;
    if (typeof kept.showOther === "boolean") out.showOther = kept.showOther;
    if (MAX_SUGGESTIONS.includes(kept.max)) out.max = kept.max;
  }
  return out;
}

/** Read this browser's stored view choices (null when storage is unavailable). */
export function readPrefs(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(PREFS_KEY) ?? null;
  } catch {
    return null;
  }
}

/** Keep the view choices in this browser; false when storage refuses. */
export function writePrefs(prefs, storage = globalThis.localStorage) {
  try {
    storage.setItem(PREFS_KEY, JSON.stringify({ showEmployee: prefs.showEmployee, showOther: prefs.showOther, max: prefs.max }));
    return true;
  } catch {
    return false;
  }
}

/** The suggestion cards split into those shown open and the older, collapsed ones. */
export function splitCards(cards, max = 3) {
  const open = Math.max(1, Number(max) || 3);
  return { shown: cards.slice(0, open), older: cards.slice(open) };
}

/** Whether pending mandatory items are due a reminder: every `seconds` since `last`. */
export function reminderDue(pending, seconds, last, now) {
  if (!pending || !seconds) return false;
  return now - last >= seconds * 1000;
}

/** Lower case, accents and punctuation removed, words of `SHORT_WORD` letters or more. */
export function contentWords(text) {
  return String(text ?? "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= SHORT_WORD);
}

/**
 * Whether `line` says what `example` proposed: at least `share` of the
 * example's content words appear in the line. Words, not meaning: a
 * rephrased question with other words is not caught.
 */
export function echoes(example, line, share = ECHO_SHARE) {
  const wanted = [...new Set(contentWords(example))];
  if (!wanted.length) return false;
  const said = new Set(contentWords(line));
  return wanted.filter((w) => said.has(w)).length / wanted.length >= share;
}

/** Seconds as "mm:ss" (minutes keep counting past the hour). */
export function clockText(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${two(Math.floor(total / 60))}:${two(total % 60)}`;
}

/** Words of a hint, to tell a pincelada (≤ 8) from a guided sentence. */
export function wordCount(text) {
  return String(text ?? "").split(/\s+/).filter(Boolean).length;
}

/** "support" or "interview" from a product name; null for others. */
export function productKind(product) {
  if (product === SUPPORT) return "support";
  if (product === INTERVIEW) return "interview";
  return null;
}

/** Whether the page may show customer signals for `page` (the session's page info). */
export function showsSignals(page) {
  return Boolean(page && page.signals === true && productKind(page.product) === "support");
}

function two(n) {
  return String(n).padStart(2, "0");
}

/** A session id the contract accepts: "<prefix>-YYYYMMDD-HHMMSS". */
export function newSessionId(prefix = "demo", now = new Date()) {
  const date = `${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}`;
  const time = `${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}`;
  return `${prefix}-${date}-${time}`;
}

/**
 * Split an SSE text buffer into complete items and the unfinished rest.
 * Items are { id, event, data } (data lines joined by "\n"); comments
 * (lines starting with ":") and the "retry" field are ignored.
 */
export function parseSse(buffer) {
  const text = String(buffer ?? "").replace(/\r\n?/g, "\n");
  const blocks = text.split("\n\n");
  const rest = blocks.pop() ?? "";
  const items = [];
  for (const block of blocks) {
    const item = { id: null, event: "message", data: [] };
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const at = line.indexOf(":");
      const field = at < 0 ? line : line.slice(0, at);
      let value = at < 0 ? "" : line.slice(at + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "id") item.id = value;
      else if (field === "event") item.event = value;
      else if (field === "data") item.data.push(value);
    }
    if (item.data.length || item.event === "end") items.push({ ...item, data: item.data.join("\n") });
  }
  return { items, rest };
}

/** Milliseconds rounded for people: { knowledge, reasoner, total } (null when unknown). */
export function latency(stages = {}) {
  const ms = (v) => (typeof v === "number" && Number.isFinite(v) ? Math.round(v) : null);
  return { knowledge: ms(stages.knowledge_ms), reasoner: ms(stages.reasoner_ms), total: ms(stages.total_ms) };
}

/** The quoted signals of one suggestion, or [] when the page must not show them. */
export function signalLines(suggestion, page) {
  if (!showsSignals(page)) return [];
  return (suggestion?.expressed ?? []).filter((e) => e && typeof e.signal === "string" && typeof e.quote === "string" && e.quote);
}

/** The chunks a suggestion cites: [{ id, text, metadata }] (text null when not retrieved). */
export function citations(suggestion, record) {
  const retrieved = new Map((record?.retrieved ?? []).map((c) => [c.id, c]));
  return (suggestion?.cites ?? []).map((id) => {
    const chunk = retrieved.get(id);
    return { id, text: chunk?.text ?? null, metadata: chunk?.metadata ?? {} };
  });
}

/** The phases each record touched: the suggestions' script_phase, else the top chunk's. */
function phasesOf(record) {
  const found = [];
  for (const s of record?.suggestions ?? []) if (s?.script_phase) found.push(s.script_phase);
  if (!found.length) {
    const top = record?.retrieved?.[0]?.metadata?.script_phase;
    if (top) found.push(top);
  }
  return found;
}

/**
 * Coverage of the script's phases: [{ phase, count, state }] with state
 * "current" (the latest phase talked about), "covered" or "pending".
 * Only records with a status of "ok" or "no_suggestion" count.
 */
export function coverage(phases, records) {
  const counts = new Map((phases ?? []).map((p) => [p, 0]));
  let current = null;
  for (const record of records ?? []) {
    if (record?.status !== "ok" && record?.status !== "no_suggestion") continue;
    for (const phase of phasesOf(record)) {
      if (!counts.has(phase)) continue;
      counts.set(phase, counts.get(phase) + 1);
      current = phase;
    }
  }
  return [...counts].map(([phase, count]) => ({
    phase, count, state: phase === current ? "current" : count > 0 ? "covered" : "pending",
  }));
}

const HALTING = new Set(["quota", "auth"]);

/**
 * The live page's state, fed one message of the orchestrator's stream at
 * a time (`apply`). Utterances keep their last revision; partials are
 * kept per utterance until its final arrives.
 */
export class LiveState {
  constructor() {
    this.session = null; // the orchestrator's session view
    this.page = null; // what to draw: product, banner, signals, phases, roles
    this.speakers = new Map(); // id -> { role, name, channel }
    this.utterances = new Map(); // id -> { id, speaker, text, start, end, revision }
    this.partials = new Map(); // id -> { id, speaker, text }
    this.records = []; // suggestion records, oldest first
    this.warnings = []; // { code, message }
    this.halted = null; // "quota" | "auth" once the reasoner stopped
    this.state = "starting"; // starting | running | ended | failed
    this.error = null;
    this.lastId = 0;
    this.employee = null; // { id, name, role, speaker } chosen at the start
    this.names = new Map(); // speaker -> name given (start or rename)
    this.mode = null; // assistance mode: pincelada | guiado
    this.checklist = null; // mandatory items: { items: [{ id, text, state }], remind_seconds }
    this.handoff = null; // the end-of-session record
    this.notices = []; // { code, message } from the orchestrator
    this.avoid = []; // lines better avoided: { utterance, category, title, why, alternative, state }
    this.topics = null; // a demo's topics: [{ id, name, done }], else the script phases
    this.demo = null; // { id, title, candidate, result_session, note } of a scripted demo
    this.streamTime = 0; // the latest event's stream time (seconds)
    this.seen = 0; // messages applied so far: the order things arrived in
    this.arrived = new Map(); // record or flag key -> `seen` when it arrived or changed
  }

  /** Apply one feed message ({ type, ... }); `id` is its SSE number. */
  apply(message, id = null) {
    const number = Number.parseInt(id ?? "", 10);
    if (Number.isFinite(number) && number > this.lastId) this.lastId = number;
    this.seen += 1;
    switch (message?.type) {
      case "session":
        this._session(message.session);
        if (this.state === "starting") this.state = "running";
        break;
      case "status":
        this._session(message.session);
        this.state = message.session?.state === "failed" ? "failed" : "ended";
        this.error = message.session?.error ?? null;
        break;
      case "event":
        this._event(message.event);
        break;
      case "suggestion":
        this._record(message.record);
        break;
      case "mandatory":
        if (message.checklist) this.checklist = message.checklist;
        break;
      case "rename":
        if (message.speaker && message.name) this.names.set(message.speaker, message.name);
        break;
      case "mode":
        if (MODES.includes(message.mode)) this.mode = message.mode;
        break;
      case "handoff":
        this.handoff = message.handoff ?? null;
        break;
      case "notice":
        this.notices.push({ code: message.code, message: message.message });
        break;
      case "avoid":
        this._flags(message.flags);
        break;
      case "topics":
        if (Array.isArray(message.topics)) this.topics = message.topics.filter((x) => x && x.id && x.name);
        break;
      default:
        break;
    }
  }

  _flags(flags) {
    if (!Array.isArray(flags)) return;
    const before = new Map(this.avoid.map((f) => [flagKey(f), f.state]));
    this.avoid = flags.filter((f) => f && f.utterance && f.category && f.title);
    for (const f of this.avoid) {
      if (before.get(flagKey(f)) !== f.state) this.arrived.set(flagKey(f), this.seen);
    }
  }

  _session(view) {
    if (!view) return;
    this.session = view;
    if (view.employee) this.employee = view.employee;
    for (const [speaker, name] of Object.entries(view.names ?? {})) this.names.set(speaker, name);
    if (MODES.includes(view.assist_mode) && !this.mode) this.mode = view.assist_mode;
    if (view.mandatory) this.checklist = view.mandatory;
    if (view.handoff) this.handoff = view.handoff;
    if (view.demo) this.demo = view.demo;
    if (Array.isArray(view.avoid) && !this.avoid.length) this._flags(view.avoid);
    if (view.page) {
      this.page = view.page;
      for (const [speaker, role] of Object.entries(view.page.roles ?? {})) {
        if (!this.speakers.has(speaker)) this.speakers.set(speaker, { role });
      }
    }
  }

  _event(event) {
    if (!event) return;
    if (typeof event.stream_time === "number" && event.stream_time > this.streamTime) this.streamTime = event.stream_time;
    const u = event.utterance;
    switch (event.type) {
      case "session_started":
        for (const s of event.speakers ?? []) this.speakers.set(s.id, { role: s.role, name: s.name ?? null, channel: s.channel ?? null });
        break;
      case "partial":
        if (u && !this.utterances.has(u.id)) this.partials.set(u.id, { id: u.id, speaker: u.speaker, text: u.text });
        break;
      case "final":
        if (u) {
          const kept = this.utterances.get(u.id);
          if (!kept || (u.revision ?? 0) >= (kept.revision ?? 0)) {
            this.utterances.set(u.id, { id: u.id, speaker: u.speaker, text: u.text, start: u.start, end: u.end, revision: u.revision ?? 0 });
          }
          this.partials.delete(u.id);
        }
        break;
      case "speaker_update":
        if (u && this.utterances.has(u.id)) this.utterances.get(u.id).speaker = u.speaker;
        break;
      case "warning":
        if (event.warning) this.warnings.push({ code: event.warning.code, message: event.warning.message });
        break;
      default:
        break;
    }
  }

  _record(record) {
    if (!record) return;
    this.records.push(record);
    this.arrived.set(record, this.seen);
    if (HALTING.has(record.status)) this.halted = record.status;
  }

  /** The role of a speaker ("unknown" when not known); the employee's id is theirs. */
  roleOf(speaker) {
    if (this.employee && speaker === this.employee.speaker) return this.employee.role;
    const known = this.speakers.get(speaker)?.role;
    if (known && known !== "unknown") return known;
    return this.page?.roles?.[speaker] ?? known ?? "unknown";
  }

  /** "employee" for the person the page helps, "other" for everyone else. */
  sideOf(speaker) {
    return EMPLOYEE_ROLES.includes(this.roleOf(speaker)) ? "employee" : "other";
  }

  /** The name of a speaker: given during the session, else known, else null. */
  nameOf(speaker) {
    return this.names.get(speaker) ?? this.speakers.get(speaker)?.name ?? null;
  }

  /** The lines to draw with the view's side choices (partials included). */
  visibleLines({ showEmployee = true, showOther = true } = {}) {
    return this.lines().filter((u) => (this.sideOf(u.speaker) === "employee" ? showEmployee : showOther));
  }

  /**
   * The customer's discomfort to highlight (support only): the newest
   * answered call whose suggestion quotes a complaint, frustration, churn
   * or urgency signal. { signals, utterance, quote, record } or null.
   */
  discomfort() {
    if (!showsSignals(this.page)) return null;
    const newest = [...this.records].reverse().find((r) => r.status === "ok" && (r.suggestions ?? []).length);
    if (!newest) return null;
    const found = (newest.suggestions ?? []).flatMap((s) => signalLines(s, this.page)).filter((e) => DISCOMFORT.includes(e.signal));
    if (!found.length) return null;
    return { signals: [...new Set(found.map((e) => e.signal))], utterance: found[0].utterance ?? null, quote: found[0].quote, record: newest };
  }

  /** The mandatory items not said nor ticked yet. */
  pendingItems() {
    return (this.checklist?.items ?? []).filter((i) => i.state === "pending");
  }

  /** Finals in time order, then the partials still open. */
  lines() {
    const finals = [...this.utterances.values()].sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
    const open = [...this.partials.values()].map((p) => ({ ...p, partial: true }));
    return [...finals, ...open];
  }

  /**
   * Suggestion cards, newest call first: { record, suggestion, signals,
   * cites, about } for every suggestion of an answered call, in the
   * order the orchestrator ranked them. Signals are empty unless the
   * page is a support page.
   */
  cards() {
    const out = [];
    for (const record of [...this.records].reverse()) {
      if (record.status !== "ok") continue;
      for (const suggestion of record.suggestions ?? []) {
        out.push({
          record, suggestion,
          signals: signalLines(suggestion, this.page),
          cites: citations(suggestion, record),
          about: (suggestion.about ?? []).map((id) => this.utterances.get(id) ?? { id, speaker: null, text: null }),
        });
      }
    }
    return out;
  }

  /** The flags of one line (better avoided), [] when it has none. */
  flagsOf(utterance) {
    return this.avoid.filter((f) => f.utterance === utterance);
  }

  /** Whether the employee already said what a card proposed (its example, or a guided sentence). */
  used(card) {
    const proposal = card?.suggestion?.example ?? (card?.record?.mode === "guiado" ? card.suggestion.text : null);
    if (!proposal) return false;
    const after = card.record?.trigger?.stream_time ?? 0;
    return [...this.utterances.values()].some((u) => u.end >= after && this.sideOf(u.speaker) === "employee"
      && echoes(proposal, u.text));
  }

  /**
   * The one thing the «Ahora» card shows:
   * - a line better avoided not rephrased yet ({ kind: "avoid", flag, done: false });
   * - a rephrased one, until the other party has answered after the rephrasing
   *   or a newer suggestion arrived ({ kind: "avoid", flag, done: true });
   * - else the newest suggestion not used yet, or the newest one
   *   ({ kind: "suggestion", card, used });
   * - else { kind: "idle" }.
   */
  now() {
    const cards = this.cards();
    const open = [...this.avoid].reverse().find((f) => f.state !== "rephrased");
    if (open) return { kind: "avoid", flag: open, done: false };
    const rephrased = [...this.avoid].reverse().find((f) => f.state === "rephrased");
    if (rephrased) {
      const by = this.utterances.get(rephrased.rephrased_by);
      const answered = by && [...this.utterances.values()].some((u) => u.start > by.start && this.sideOf(u.speaker) === "other");
      const since = this.arrived.get(flagKey(rephrased)) ?? 0;
      const newer = cards.some((c) => (this.arrived.get(c.record) ?? 0) > since);
      if (!answered && !newer) return { kind: "avoid", flag: rephrased, done: true };
    }
    if (!cards.length) return { kind: "idle" };
    const card = cards.find((c) => !this.used(c)) ?? cards[0];
    return { kind: "suggestion", card, used: this.used(card) };
  }

  /** The suggestions not in the «Ahora» card, newest first (for «Sugerencias anteriores»). */
  earlier() {
    const shown = this.now();
    return this.cards().filter((c) => shown.kind !== "suggestion" || c.suggestion !== shown.card.suggestion);
  }

  /**
   * The chips of the topics: a demo's own topics ({ id, name, done }), or
   * the script phases of an interview ({ id: phase, phase, done }), where a
   * phase is done once a suggestion was about it.
   */
  topicChips() {
    if (this.topics) return this.topics.map((x) => ({ id: x.id, name: x.name, done: Boolean(x.done) }));
    return this.coverage().map((c) => ({ id: c.phase, phase: c.phase, done: c.state !== "pending" }));
  }

  /** Which side is speaking now (an open partial): "employee", "other" or null. */
  speaking() {
    const open = [...this.partials.values()].at(-1);
    return open ? this.sideOf(open.speaker) : null;
  }

  /** Calls by status ({ ok: 3, timeout: 1 }). */
  counts() {
    const counts = {};
    for (const r of this.records) counts[r.status] = (counts[r.status] ?? 0) + 1;
    return counts;
  }

  /** The script phases' coverage (interviews), [] when the product has none. */
  coverage() {
    return coverage(this.page?.script_phases ?? [], this.records);
  }
}

function flagKey(flag) {
  return `${flag?.utterance}|${flag?.category}`;
}
