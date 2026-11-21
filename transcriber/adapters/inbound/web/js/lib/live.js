// Pure logic of the live page (#/live): the Server-Sent Events parser, the
// session's state built from the orchestrator's feed (live.event.v1 events
// and suggestion records), the script-phase coverage of interviews and the
// customer signals of support calls. No DOM here, so node --test runs it.
//
// Rule (docs/platform/ARCHITECTURE.md §8): signals are shown only in the
// support product, as what the customer *says* with the quoted sentence;
// an interview never shows signals, ratings or labels about the candidate.

export const SUPPORT = "live_assist.support";
export const INTERVIEW = "live_assist.interview";
export const AUDIO = ["stereo", "mic"];
export const SPEEDS = [1, 2, 4];

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
  }

  /** Apply one feed message ({ type, ... }); `id` is its SSE number. */
  apply(message, id = null) {
    const number = Number.parseInt(id ?? "", 10);
    if (Number.isFinite(number) && number > this.lastId) this.lastId = number;
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
      default:
        break;
    }
  }

  _session(view) {
    if (!view) return;
    this.session = view;
    if (view.page) {
      this.page = view.page;
      for (const [speaker, role] of Object.entries(view.page.roles ?? {})) {
        if (!this.speakers.has(speaker)) this.speakers.set(speaker, { role });
      }
    }
  }

  _event(event) {
    if (!event) return;
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
    if (HALTING.has(record.status)) this.halted = record.status;
  }

  /** The role of a speaker ("unknown" when not known). */
  roleOf(speaker) {
    return this.speakers.get(speaker)?.role ?? this.page?.roles?.[speaker] ?? "unknown";
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
