// Pure logic of the "Pruebas" progress overview (no DOM): sessions grouped by
// corpus, step counts and bar segments, chart scales, the DER and agreement
// rows of the results charts and the choice of an audio excerpt.

export const ACTIVE_RUNS = new Set(["queued", "running"]);
export const OWN = "own";
export const CORPUS_ORDER = [OWN, "ami", "callhome"];
export const EXCERPT_SECONDS = 20;
export const POLL_MS = 10000;
export const SEP = "+";

const EMPTY_TALLY = Object.freeze({ done: 0, running: 0, pending: 0, failed: 0, failed_codes: {}, open: [] });

/** The corpus of an overview entry: its reference's, or "own". */
export function corpusOf(entry) {
  return entry.corpus || OWN;
}

/** The step counts of an entry's newest run (zeros without a run). */
export function tallyOf(entry) {
  return entry.steps || EMPTY_TALLY;
}

/** Number of steps of a tally. */
export function stepTotal(tally) {
  return tally.done + tally.running + tally.pending + tally.failed;
}

/** Entries grouped by corpus: own recordings first, then AMI, CALLHOME, others by name. */
export function groupByCorpus(entries) {
  const groups = new Map();
  for (const entry of entries) {
    const corpus = corpusOf(entry);
    if (!groups.has(corpus)) groups.set(corpus, []);
    groups.get(corpus).push(entry);
  }
  const rank = (c) => (CORPUS_ORDER.includes(c) ? CORPUS_ORDER.indexOf(c) : CORPUS_ORDER.length);
  return [...groups.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([corpus, sessions]) => ({
      corpus,
      sessions: [...sessions].sort((a, b) => a.session_id.localeCompare(b.session_id)),
    }));
}

/** Totals over every entry: steps done and left, runs running, sessions without a run. */
export function overviewTotals(entries) {
  const totals = { done: 0, running: 0, pending: 0, failed: 0, left: 0, steps: 0, runsRunning: 0, notStarted: 0, sessions: entries.length };
  for (const entry of entries) {
    const tally = tallyOf(entry);
    totals.done += tally.done;
    totals.running += tally.running;
    totals.pending += tally.pending;
    totals.failed += tally.failed;
    if (ACTIVE_RUNS.has(entry.run_state)) totals.runsRunning += 1;
    if (!entry.run_id) totals.notStarted += 1;
  }
  totals.left = totals.running + totals.pending + totals.failed;
  totals.steps = totals.done + totals.left;
  return totals;
}

/** Whether the page should keep refreshing: some run is queued or running. */
export function anyActive(entries) {
  return entries.some((entry) => ACTIVE_RUNS.has(entry.run_state));
}

/**
 * The stacked bar of a tally as segments in [0, 1]: done, running, pending,
 * then failed split by error code (most frequent first, then by code).
 */
export function barSegments(tally) {
  const total = stepTotal(tally);
  if (!total) return [];
  const parts = [["done", null, tally.done], ["running", null, tally.running], ["pending", null, tally.pending]];
  const codes = Object.entries(tally.failed_codes || {}).sort(([a, x], [b, y]) => y - x || a.localeCompare(b));
  let counted = 0;
  for (const [code, count] of codes) {
    parts.push(["failed", code, count]);
    counted += count;
  }
  if (tally.failed > counted) parts.push(["failed", "unknown", tally.failed - counted]);
  const segments = [];
  let at = 0;
  for (const [kind, code, count] of parts) {
    if (!count) continue;
    segments.push({ kind, code, count, start: at / total, width: count / total });
    at += count;
  }
  return segments;
}

/** Every failure code over the entries with its number of steps, most frequent first. */
export function failureCodes(entries) {
  const codes = new Map();
  for (const entry of entries) {
    for (const [code, count] of Object.entries(tallyOf(entry).failed_codes || {})) codes.set(code, (codes.get(code) || 0) + count);
  }
  return [...codes.entries()].sort(([a, x], [b, y]) => y - x || a.localeCompare(b)).map(([code, count]) => ({ code, count }));
}

/** The sessions with steps left, failed ones first; each with its open steps (failed, then the rest). */
export function pendingGroups(entries) {
  const order = { failed: 0, running: 1, pending: 2 };
  return entries
    .filter((entry) => tallyOf(entry).open.length)
    .map((entry) => ({
      entry,
      failed: tallyOf(entry).failed,
      active: ACTIVE_RUNS.has(entry.run_state),
      steps: [...tallyOf(entry).open].sort((a, b) => (order[a.state] ?? 3) - (order[b.state] ?? 3)),
    }))
    .sort((a, b) => b.failed - a.failed || a.entry.session_id.localeCompare(b.entry.session_id));
}

/** A round upper bound for an axis that must reach `value` (1, 2, 2.5 or 5 times a power of ten). */
export function niceMax(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * power + 1e-12) return step * power;
  }
  return 10 * power;
}

/** A linear map of [0, max] onto [from, to] (values clamped to the range). */
export function linearScale(max, from, to) {
  const top = max > 0 ? max : 1;
  return (value) => from + (Math.min(Math.max(value, 0), top) / top) * (to - from);
}

/** `count` + 1 evenly spaced ticks from 0 to `max`, rounded to avoid float noise. */
export function axisTicks(max, count = 4) {
  const out = [];
  for (let i = 0; i <= count; i += 1) out.push(Math.round((max * i / count) * 1e6) / 1e6);
  return out;
}

/** The denoisers of the summary's diarization tracks ("none" first). */
export function summaryDenoisers(summary) {
  const found = new Set();
  for (const model of summary?.models || []) {
    if (model.kind === "diarization") found.add(model.id.split(SEP)[0]);
  }
  return [...found].sort((a, b) => (a === "none" ? -1 : b === "none" ? 1 : a.localeCompare(b)));
}

function spreadRow(id, spread) {
  return { id, mean: spread?.mean ?? null, min: spread?.min ?? null, max: spread?.max ?? null, n: spread?.n ?? 0 };
}

function byMean(a, b) {
  if (a.mean === null && b.mean === null) return a.id.localeCompare(b.id);
  if (a.mean === null) return 1;
  if (b.mean === null) return -1;
  return a.mean - b.mean || a.id.localeCompare(b.id);
}

/** DER of each diarizer on one denoiser's audio over the corpus: mean, min, max, n; best first. */
export function derByDiarizer(summary, denoiser = "none", metric = "der") {
  const prefix = `${denoiser}${SEP}`;
  return (summary?.models || [])
    .filter((m) => m.kind === "diarization" && m.id.startsWith(prefix))
    .map((m) => spreadRow(m.id.slice(prefix.length), m.metrics?.[metric]))
    .sort(byMean);
}

/**
 * DER of each denoiser: the mean over the diarizers that ran on every
 * denoiser (so denoisers are compared on the same diarizers), with the range
 * from the best to the worst of those diarizers' means.
 */
export function derByDenoiser(summary, metric = "der") {
  const tracks = (summary?.models || []).filter((m) => m.kind === "diarization");
  const byDenoiser = new Map();
  for (const m of tracks) {
    const [denoiser, diarizer] = m.id.split(SEP);
    const mean = m.metrics?.[metric]?.mean;
    if (!Number.isFinite(mean)) continue;
    if (!byDenoiser.has(denoiser)) byDenoiser.set(denoiser, new Map());
    byDenoiser.get(denoiser).set(diarizer, mean);
  }
  const sets = [...byDenoiser.values()].map((m) => new Set(m.keys()));
  const common = sets.length ? [...sets[0]].filter((d) => sets.every((s) => s.has(d))) : [];
  return [...byDenoiser.entries()].map(([denoiser, means]) => {
    const values = common.map((d) => means.get(d));
    return {
      id: denoiser,
      mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null,
      min: values.length ? Math.min(...values) : null,
      max: values.length ? Math.max(...values) : null,
      n: values.length,
    };
  }).sort(byMean);
}

/** Agreement with the consensus of each diarizer on one denoiser's audio (one run's results), best first. */
export function agreementByDiarizer(results, denoiser = "none") {
  return (results?.tracks || [])
    .filter((tr) => tr.denoiser === denoiser && Number.isFinite(tr.agreement_with_consensus))
    .map((tr) => ({ id: tr.diarizer, mean: tr.agreement_with_consensus, min: null, max: null, n: 1 }))
    .sort((a, b) => b.mean - a.mean || a.id.localeCompare(b.id));
}

/**
 * Speaker turns that may overlap ({start, end, speaker}, as a corpus
 * reference gives them) as consecutive spans {start, end, speakers} that
 * chooseExcerpt reads like the consensus; stretches with nobody are left out.
 */
export function turnsToSpans(turns) {
  const edges = new Set();
  for (const turn of turns || []) {
    if (turn.end > turn.start) {
      edges.add(turn.start);
      edges.add(turn.end);
    }
  }
  const points = [...edges].sort((a, b) => a - b);
  const spans = [];
  for (let i = 0; i + 1 < points.length; i += 1) {
    const [start, end] = [points[i], points[i + 1]];
    const speakers = [...new Set((turns || []).filter((tr) => tr.start <= start && tr.end >= end).map((tr) => tr.speaker))].sort();
    if (speakers.length) spans.push({ start, end, speakers });
  }
  return spans;
}

function speakerKey(span) {
  return [...(span.speakers || [])].sort().join(SEP);
}

/**
 * Where to take an excerpt of `seconds` from a recording of `duration`: the
 * window with the most overlapped speech and speaker changes of the consensus
 * `spans` ({start, end, speakers}), else 10 % in. Returns {start, end, reason}
 * with reason "overlap", "changes" or "default".
 */
export function chooseExcerpt(spans, duration, seconds = EXCERPT_SECONDS) {
  const length = Math.max(0, Math.min(seconds, duration));
  const latest = Math.max(0, duration - length);
  const fallback = { start: Math.min(duration * 0.1, latest), end: 0, reason: "default" };
  fallback.end = fallback.start + length;
  const speech = (spans || []).filter((s) => (s.speakers || []).length && s.end > s.start).sort((a, b) => a.start - b.start);
  if (!speech.length || !(length > 0)) return fallback;
  const changes = [];
  for (let i = 1; i < speech.length; i += 1) {
    if (speakerKey(speech[i]) !== speakerKey(speech[i - 1])) changes.push(speech[i].start);
  }
  let best = null;
  for (const span of speech) {
    const start = Math.min(span.start, latest);
    const end = start + length;
    let overlap = 0;
    for (const other of speech) {
      if (other.start >= end) break;
      if (other.speakers.length > 1) overlap += Math.max(0, Math.min(other.end, end) - Math.max(other.start, start));
    }
    const switched = changes.filter((at) => at > start && at < end).length;
    const score = overlap * 2 + switched;
    if (score > 0 && (!best || score > best.score + 1e-9)) best = { start, end, score, overlap, switched };
  }
  if (!best) return fallback;
  return { start: round3(best.start), end: round3(best.end), reason: best.overlap > 0 ? "overlap" : "changes" };
}

function round3(value) {
  return Math.round(value * 1000) / 1000;
}

/**
 * One row per audio of a run with results (the original and each denoiser
 * whose tracks exist): its DER against the reference (mean over the
 * diarizers and the best one) and its mean agreement with the consensus.
 */
export function sampleRows(results) {
  const reference = new Map();
  for (const row of results?.reference?.tracks || []) {
    if (!Number.isFinite(row.der)) continue;
    const denoiser = row.denoiser ?? row.track.split(SEP)[0];
    if (!reference.has(denoiser)) reference.set(denoiser, []);
    reference.get(denoiser).push({ diarizer: row.diarizer ?? row.track.split(SEP)[1], der: row.der });
  }
  // on a corpus recording nobody is enrolled, so the consensus names nobody
  // and agreement with it says nothing: only the DER is shown there
  const agreement = new Map();
  const scored = Boolean(results?.reference);
  for (const tr of results?.tracks || []) {
    if (scored) {
      if (!agreement.has(tr.denoiser)) agreement.set(tr.denoiser, []);
      continue;
    }
    if (!agreement.has(tr.denoiser)) agreement.set(tr.denoiser, []);
    if (Number.isFinite(tr.agreement_with_consensus)) agreement.get(tr.denoiser).push(tr.agreement_with_consensus);
  }
  const denoisers = [...new Set([...agreement.keys(), ...reference.keys()])]
    .sort((a, b) => (a === "none" ? -1 : b === "none" ? 1 : a.localeCompare(b)));
  return denoisers.map((denoiser) => {
    const ders = reference.get(denoiser) || [];
    const best = ders.reduce((low, r) => (!low || r.der < low.der ? r : low), null);
    const agreements = agreement.get(denoiser) || [];
    return {
      denoiser,
      der: ders.length ? ders.reduce((a, r) => a + r.der, 0) / ders.length : null,
      bestDer: best ? best.der : null,
      bestDiarizer: best ? best.diarizer : null,
      agreement: agreements.length ? agreements.reduce((a, b) => a + b, 0) / agreements.length : null,
    };
  });
}
