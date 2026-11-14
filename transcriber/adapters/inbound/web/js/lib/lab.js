// Pure logic of the "Pruebas" tab (the model lab): no DOM, no network.
// Ids follow docs/transcriber/LAB.md: a track or transcript is
// "denoiser+model", a pipeline "denoiser+diarizer+recogniser" ("-" for a
// missing stage); a speaker nobody was found for is "~label".

export const STAGES = ["denoise", "diarize", "recognise"];
export const NO_DENOISER = "none";
export const SEP = "+";
export const PALETTE_SIZE = 8;

// ---- catalogue and selection -------------------------------------------

/** Models by stage, in catalogue order: { denoise: [...], diarize: [...], recognise: [...] }. */
export function groupCatalogue(models) {
  const groups = Object.fromEntries(STAGES.map((s) => [s, []]));
  for (const model of models ?? []) {
    if (groups[model.stage]) groups[model.stage].push(model);
  }
  return groups;
}

/** The default selection: every model that is ready on this machine. */
export function defaultSelection(models) {
  return new Set((models ?? []).filter((m) => m.state === "ready").map((m) => m.id));
}

/** The request body for the chosen ids (catalogue order); denoisers default to "none". */
export function runRequest(selected, models) {
  const groups = groupCatalogue(models);
  const pick = (stage) => groups[stage].filter((m) => selected.has(m.id)).map((m) => m.id);
  const denoisers = pick("denoise");
  return {
    denoisers: denoisers.length ? denoisers : [NO_DENOISER],
    diarizers: pick("diarize"),
    recognisers: pick("recognise"),
  };
}

/** How many pipelines and steps a request implies (as the service plans them). */
export function planCounts(request) {
  const d = request.denoisers.length;
  const z = request.diarizers.length;
  const r = request.recognisers.length;
  const denoiseSteps = request.denoisers.filter((x) => x !== NO_DENOISER).length;
  return {
    pipelines: d * Math.max(1, z) * Math.max(1, r),
    steps: denoiseSteps + d * (z + r),
    valid: z + r > 0,
  };
}

// ---- ids --------------------------------------------------------------

/** Split "a+b+c" into its parts. */
export function splitId(id) {
  return String(id ?? "").split(SEP);
}

/** A speaker no enrolled person was found for ("~label"). */
export function isUnmatched(speaker) {
  return String(speaker ?? "").startsWith("~");
}

// ---- answers of a fragment ---------------------------------------------

/** A set of speakers as a stable key: sorted names joined by "+", "" for nobody. */
export function answerKey(speakers) {
  return [...(speakers ?? [])].sort().join(SEP);
}

/**
 * Group identical answers: [{ key, speakers, pipelines: [ids], count }],
 * most common first (ties by key). `byPipeline` maps an id to { speakers }
 * or directly to a speakers array.
 */
export function groupAnswers(byPipeline) {
  const groups = new Map();
  for (const [id, value] of Object.entries(byPipeline ?? {})) {
    const speakers = Array.isArray(value) ? value : value?.speakers ?? [];
    const key = answerKey(speakers);
    if (!groups.has(key)) groups.set(key, { key, speakers: [...speakers].sort(), pipelines: [] });
    groups.get(key).pipelines.push(id);
  }
  return [...groups.values()]
    .map((g) => ({ ...g, pipelines: g.pipelines.sort(), count: g.pipelines.length }))
    .sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
}

/**
 * The text variants of a fragment: [{ text, recognisers: [ids], pipelines: n }],
 * one per distinct trimmed text, most common first. Pipelines without a
 * recogniser ("-") are left out; an empty text counts as a variant.
 */
export function textVariants(byPipeline) {
  const groups = new Map();
  for (const [id, value] of Object.entries(byPipeline ?? {})) {
    const [denoiser, , recogniser] = splitId(id);
    if (!recogniser || recogniser === "-") continue;
    const text = String(value?.text ?? "").replace(/\s+/g, " ").trim();
    if (!groups.has(text)) groups.set(text, { text, sources: new Set(), pipelines: 0 });
    const group = groups.get(text);
    group.sources.add(`${denoiser}${SEP}${recogniser}`);
    group.pipelines += 1;
  }
  return [...groups.values()]
    .map((g) => ({ text: g.text, recognisers: [...g.sources].sort(), pipelines: g.pipelines }))
    .sort((a, b) => b.pipelines - a.pipelines || a.text.localeCompare(b.text));
}

// ---- colours ----------------------------------------------------------

/**
 * A colour slot per speaker: enrolled ids get "spk-0".."spk-7" in the
 * order given (the same person keeps the same colour in every row);
 * unmatched "~label" voices get "anon-0".."anon-3" (neutral, hatched).
 */
export function speakerColours(speakers) {
  const colours = new Map();
  let named = 0;
  let anonymous = 0;
  for (const speaker of speakers ?? []) {
    if (colours.has(speaker)) continue;
    if (isUnmatched(speaker)) colours.set(speaker, `anon-${anonymous++ % 4}`);
    else colours.set(speaker, `spk-${named++ % PALETTE_SIZE}`);
  }
  return colours;
}

/** Every speaker of every track, enrolled first (sorted), then the unmatched ones (sorted). */
export function allSpeakers(tracks) {
  const seen = new Set();
  for (const segments of Object.values(tracks ?? {})) {
    for (const s of segments) seen.add(s.speaker);
  }
  const list = [...seen];
  return [...list.filter((s) => !isUnmatched(s)).sort(), ...list.filter(isUnmatched).sort()];
}

// ---- time and zoom ------------------------------------------------------

/** A linear scale from the window [start, end] (seconds) to [0, width] pixels. */
export function timeScale(start, end, width) {
  const span = Math.max(1e-6, end - start);
  return {
    x: (time) => ((time - start) / span) * width,
    time: (x) => start + (x / Math.max(1e-6, width)) * span,
  };
}

/** Round tick times (seconds) for a window, about `count` of them. */
export function ticks(start, end, count = 6) {
  const span = Math.max(1e-6, end - start);
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find((s) => span / s <= count) ?? steps[steps.length - 1];
  const out = [];
  for (let t = Math.ceil(start / step) * step; t <= end + 1e-9; t += step) out.push(Number(t.toFixed(3)));
  return out;
}

/** Keep a window inside [0, duration] with a width of at least `minSpan`. */
export function clampWindow(start, end, duration, minSpan = 1) {
  let span = Math.min(Math.max(end - start, Math.min(minSpan, duration)), duration);
  let from = Math.min(Math.max(0, start), Math.max(0, duration - span));
  if (!(span > 0)) span = duration;
  if (!Number.isFinite(from)) from = 0;
  return { start: from, end: from + span };
}

/** Zoom the window by `factor` (<1 zooms in) around `centre` (seconds). */
export function zoomWindow(view, factor, centre, duration, minSpan = 1) {
  const span = (view.end - view.start) * factor;
  const ratio = (centre - view.start) / Math.max(1e-6, view.end - view.start);
  const start = centre - span * ratio;
  return clampWindow(start, start + span, duration, minSpan);
}

/** Move the window by `fraction` of its width (negative: earlier). */
export function scrollWindow(view, fraction, duration) {
  const span = view.end - view.start;
  return clampWindow(view.start + span * fraction, view.end + span * fraction, duration, span);
}

/** Segments that show in the window (any overlap). */
export function visible(segments, start, end) {
  return (segments ?? []).filter((s) => s.end > start && s.start < end);
}

/** Stretches where two or more speakers of one track talk at once. */
export function overlaps(segments) {
  const sorted = [...(segments ?? [])].sort((a, b) => a.start - b.start);
  const out = [];
  let reach = -Infinity;
  for (const s of sorted) {
    if (s.start < reach) {
      const end = Math.min(reach, s.end);
      const last = out[out.length - 1];
      if (last && s.start <= last.end) last.end = Math.max(last.end, end);
      else out.push({ start: s.start, end });
    }
    reach = Math.max(reach, s.end);
  }
  return out;
}

/** The playback start for a span: `lead` seconds earlier, not before 0. */
export function playStart(start, lead) {
  return Math.max(0, Number(start) - Math.max(0, Number(lead) || 0));
}

// ---- tables -------------------------------------------------------------

/**
 * The overview's pipeline rows with their scores against a corpus
 * reference (results.reference), when the session has one: ref_der,
 * ref_jer, ref_wer (null where not measured). A non-commercial corpus
 * marks every row non-commercial.
 */
export function withReference(pipelines, reference) {
  const scored = new Map((reference?.pipelines ?? []).map((p) => [p.id, p]));
  return (pipelines ?? []).map((p) => {
    const row = { ...p, speaker_count: p.speakers?.length ?? null };
    if (!reference) return row;
    const score = scored.get(p.id);
    return {
      ...row,
      ref_der: score?.der ?? null,
      ref_jer: score?.jer ?? null,
      ref_wer: score?.wer ?? null,
      licence_commercial: p.licence_commercial !== false && reference.licence_commercial !== false,
    };
  });
}

/** Sort rows by a numeric or text field; missing values always last. */
export function sortRows(rows, key, descending = false) {
  const sign = descending ? -1 : 1;
  return [...(rows ?? [])].sort((a, b) => {
    const x = a?.[key];
    const y = b?.[key];
    const xMissing = x === null || x === undefined || Number.isNaN(x);
    const yMissing = y === null || y === undefined || Number.isNaN(y);
    if (xMissing || yMissing) return xMissing === yMissing ? 0 : xMissing ? 1 : -1;
    if (typeof x === "number" && typeof y === "number") return sign * (x - y);
    return sign * String(x).localeCompare(String(y));
  });
}

/** Distinct values of one part of ids ("a+b+c" part 0, 1 or 2), in first-seen order. */
export function idPart(ids, part) {
  const out = [];
  for (const id of ids ?? []) {
    const value = splitId(id)[part];
    if (value !== undefined && !out.includes(value)) out.push(value);
  }
  return out;
}

/**
 * The pairwise matrix of tracks for one value (e.g. "agreement"):
 * { ids, value(a, b) } where value(a, a) is 1 for agreement-like fields
 * and null when the pair was not compared.
 */
export function pairMatrix(rows, ids, field, diagonal = 1) {
  const cells = new Map();
  for (const row of rows ?? []) cells.set(`${row.reference}|${row.hypothesis}`, row[field]);
  return {
    ids,
    value(a, b) {
      if (a === b) return diagonal;
      const direct = cells.get(`${a}|${b}`);
      if (direct !== undefined && direct !== null) return direct;
      const reverse = cells.get(`${b}|${a}`);
      return reverse === undefined ? null : reverse;
    },
  };
}

/** A heat class "heat-0".."heat-9" for a share in [0, 1]; "heat-none" when missing. */
export function heatClass(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "heat-none";
  return `heat-${Math.min(9, Math.max(0, Math.floor(value * 10)))}`;
}

/**
 * The effect of each denoiser, as a small table:
 * { columns: [models], rows: [{ denoiser, cells: { model: value } }] }
 * for one stage ("diarize": agreement; "recognise": wer).
 */
export function denoiserTable(effects, stage, field) {
  const columns = [];
  const rows = new Map();
  for (const e of effects ?? []) {
    if (e.stage !== stage) continue;
    if (!columns.includes(e.model)) columns.push(e.model);
    if (!rows.has(e.denoiser)) rows.set(e.denoiser, { denoiser: e.denoiser, cells: {} });
    rows.get(e.denoiser).cells[e.model] = e[field];
  }
  return { columns, rows: [...rows.values()] };
}

/**
 * The best value of `field` per group: for every value of part `by` of the
 * pipeline ids (e.g. 1, the diarizer), the row with the best score over
 * part `over` (e.g. 0, the denoiser). Lower is better unless `higher`.
 */
export function bestBy(rows, by, over, field, higher = false) {
  const best = new Map();
  for (const row of rows ?? []) {
    const value = row[field];
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    const parts = splitId(row.pipeline ?? row.id);
    const key = parts[by];
    const current = best.get(key);
    if (!current || (higher ? value > current.value : value < current.value)) {
      best.set(key, { group: key, choice: parts[over], value, row });
    }
  }
  return [...best.values()].sort((a, b) => (higher ? b.value - a.value : a.value - b.value));
}

// ---- labels -------------------------------------------------------------

/** The label holding the middle of [start, end), if any. */
export function coveringLabel(labels, start, end) {
  const middle = (start + end) / 2;
  return (labels ?? []).find((l) => l.start <= middle && middle < l.end) ?? null;
}

/**
 * Put `label` in the list: labels overlapping it are cut (the parts outside
 * it stay) so labels never overlap. Returns a new list in time order.
 */
export function placeLabel(labels, label) {
  const out = [];
  for (const l of labels ?? []) {
    if (l.end <= label.start || l.start >= label.end) {
      out.push(l);
      continue;
    }
    if (l.start < label.start) out.push({ ...l, end: label.start });
    if (l.end > label.end) out.push({ ...l, start: label.end });
  }
  out.push(label);
  return out.filter((l) => l.end - l.start > 1e-6).sort((a, b) => a.start - b.start);
}

/** Remove whatever labels cover [start, end) (parts outside it stay). */
export function clearSpan(labels, start, end) {
  return placeLabel(labels, { start, end, kind: "unsure" }).filter((l) => !(l.start === start && l.end === end));
}

/** A label of `kind` for a fragment; `both` needs two or more people. */
export function makeLabel(fragment, kind, speakers = [], text = null) {
  const start = Number(fragment.start);
  const end = Number(fragment.end);
  const label = { start, end, kind, speakers: [], text: text && String(text).trim() ? String(text).trim() : null };
  if (kind === "person") label.speakers = speakers.slice(0, 1);
  if (kind === "both") {
    label.speakers = [...new Set(speakers)].sort();
    if (label.speakers.length < 2) return null;
  }
  if (kind === "person" && label.speakers.length !== 1) return null;
  return label;
}

/** The labels as the API stores them (no fields it does not know). */
export function labelsBody(labels) {
  return {
    labels: (labels ?? []).map((l) => {
      const out = { start: l.start, end: l.end, kind: l.kind, speakers: l.speakers ?? [], text: l.text ?? null };
      if (l.updated_at) out.updated_at = l.updated_at;
      return out;
    }),
  };
}

/** Labelled seconds and fragments: { seconds, duration, fragments, total }. */
export function labelProgress(labels, spans, duration) {
  const seconds = (labels ?? []).reduce((sum, l) => sum + Math.max(0, l.end - l.start), 0);
  const fragments = (spans ?? []).filter((s) => coveringLabel(labels, s.start, s.end)).length;
  return { seconds, duration: duration ?? 0, fragments, total: (spans ?? []).length };
}

/**
 * Saving labels: the whole list goes in one PUT, so the queue only keeps the
 * latest list and whether it is saved. States: "saved", "pending" (waiting
 * for the debounce), "saving", "failed" (kept locally, retried later).
 * Timers, storage and the clock are injected so the logic can be tested.
 */
export class LabelQueue {
  constructor({ save, storage = null, key = "lab.labels", delay = 800, retry = [2000, 5000, 15000, 30000],
    setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (id) => clearTimeout(id), onChange = () => {} }) {
    Object.assign(this, { save, storage, key, delay, retry, setTimer, clearTimer, onChange });
    this.labels = [];
    this.state = "saved";
    this.failures = 0;
    this.timer = null;
    this.version = 0;
    this.error = null;
  }

  /** Start from the server's labels, or from unsaved local ones if there are any. */
  load(serverLabels) {
    const local = this.readLocal();
    if (local) {
      this.labels = local;
      this.version += 1;
      this.setState("pending");
      this.schedule(this.delay);
      return true;
    }
    this.labels = serverLabels ?? [];
    this.setState("saved");
    return false;
  }

  /** Replace the list (after a change); it is saved after the debounce. */
  set(labels) {
    this.labels = labels;
    this.version += 1;
    this.writeLocal();
    this.setState("pending");
    this.schedule(this.delay);
  }

  /** Save now (e.g. when leaving the page). */
  flush() {
    if (this.state === "saved" || this.state === "saving") return Promise.resolve();
    return this.run();
  }

  schedule(ms) {
    if (this.timer !== null) this.clearTimer(this.timer);
    this.timer = this.setTimer(() => { this.timer = null; this.run(); }, ms);
  }

  async run() {
    if (this.timer !== null) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    if (this.state === "saving") return;
    const version = this.version;
    this.setState("saving");
    try {
      const saved = await this.save(this.labels);
      this.failures = 0;
      this.error = null;
      if (version === this.version) {
        if (Array.isArray(saved)) this.labels = saved;
        this.clearLocal();
        this.setState("saved");
      } else {
        this.setState("pending");
        this.schedule(this.delay);
      }
    } catch (error) {
      this.error = error;
      const wait = this.retry[Math.min(this.failures, this.retry.length - 1)];
      this.failures += 1;
      this.setState("failed");
      if (error?.kind !== "unauthorized" && error?.kind !== "client") this.schedule(wait);
    }
  }

  setState(state) {
    this.state = state;
    this.onChange(state);
  }

  readLocal() {
    try {
      const text = this.storage?.getItem(this.key);
      const parsed = text ? JSON.parse(text) : null;
      return Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  writeLocal() {
    try {
      this.storage?.setItem(this.key, JSON.stringify(this.labels));
    } catch {
      // kept in memory only
    }
  }

  clearLocal() {
    try {
      this.storage?.removeItem(this.key);
    } catch {
      // nothing stored
    }
  }
}

// ---- keyboard -------------------------------------------------------------

/**
 * What a key does in the review list, or null: { action: "person", index },
 * "both", "nobody", "unsure", "clear", "play", "next", "previous".
 * Keys typed in a text field, or with Ctrl/Alt/Meta, do nothing.
 */
export function keyAction(event, people = 2) {
  if (!event || event.ctrlKey || event.altKey || event.metaKey) return null;
  const tag = String(event.target?.tagName ?? "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target?.isContentEditable) return null;
  const key = String(event.key ?? "");
  if (/^[1-9]$/.test(key)) {
    const index = Number(key) - 1;
    return index < people ? { action: "person", index } : null;
  }
  switch (key.toLowerCase()) {
    case "b": return { action: "both" };
    case "n": return { action: "nobody" };
    case "u": return { action: "unsure" };
    case "x": return { action: "clear" };
    case " ": return { action: "play" };
    case "j": return { action: "next" };
    case "k": return { action: "previous" };
    default: return null;
  }
}

// ---- export -------------------------------------------------------------

/** What the export download holds (labels and, when there are any, the scores). */
export function exportPayload(sessionId, runId, labels, scores, now = new Date()) {
  return {
    schema_name: "lab.review.v1",
    exported_at: now.toISOString(),
    session_id: sessionId,
    run_id: runId,
    labels: labelsBody(labels).labels,
    scores: scores ?? null,
  };
}
