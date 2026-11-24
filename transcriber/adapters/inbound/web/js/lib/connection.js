// The connection check of the live page (#247), pure parts: the figures
// of a check (latency, its variation, lost requests, the upload rate),
// the one verdict they give with what to do about it, and the quiet
// in-session watch that changes state only after a sustained problem.
// The requests themselves are made by js/connection.js; the drawing is
// js/views/connection.js. Thresholds and their reasons:
// docs/live/CONNECTION.md (chosen from what the live path sends and
// receives, not measured on calls).

/** Verdicts from best to worst. */
export const LEVELS = ["good", "fair", "poor"];

/**
 * Thresholds: a figure up to `good` is good, up to `fair` is fair, above is
 * poor (the upload rate the other way round: at least `good` is good).
 *  - rttMs: median round trip of the processing server's health route.
 *    The live path's budget is 3 s from the end of a sentence to its
 *    suggestion with 220 ms of margin (docs/platform/ARCHITECTURE.md §6);
 *    the network adds about one round trip (audio up, suggestion down).
 *    150 ms fits in the margin; 400 ms makes suggestions noticeably later.
 *  - jitterMs: mean change between consecutive round trips. Partial lines
 *    arrive every 800 ms; up to 50 ms is not seen, above 150 ms (a fifth
 *    of that step) lines and suggestions come in bursts.
 *  - lossPct: requests that failed or took more than CHECK.timeoutMs. One
 *    lost request is a stall of 2 s, most of the 3 s budget.
 *  - upKbps: the browser's microphone is sent as 16 kHz 16-bit mono PCM,
 *    256 kbit/s. Good from four times that, fair from 1.5 times.
 */
export const THRESHOLDS = {
  rttMs: { good: 150, fair: 400 },
  jitterMs: { good: 50, fair: 150 },
  lossPct: { good: 0, fair: 20 },
  upKbps: { good: 1024, fair: 384 },
};

/** What one check sends: a few small requests, done in a couple of seconds. */
export const CHECK = {
  linkPings: 6, // to /api/health; the first one only opens the connection
  webPings: 3, // to /healthz, answered by the public server itself
  timeoutMs: 2000,
  uploadBytes: 64 * 1024, // two seconds of the microphone's audio
  maxAgeMs: 60000, // a check younger than this is shown again, not repeated
};

/** The in-session watch: one ping every everyMs (retryMs after a failure). */
export const WATCH = {
  everyMs: 15000,
  retryMs: 5000,
  timeoutMs: 4000,
  worse: 3, // samples in a row needed to get worse
  better: 2, // samples in a row needed to get better
  lostAfter: 2, // failed pings in a row: the connection is lost
  window: 8, // samples kept (two minutes)
  maxLost: 1, // more lost pings than this in the window: at most fair
  servicesEvery: 4, // pings between two checks of the services
};

/** The services a session shows, in order; the link comes first. */
export const SERVICE_ROWS = ["web", "link", "transcription", "assistant", "knowledge"];

/** What a session depends on, by how it gets its conversation. */
export function needsOf(mode) {
  if (mode === "demo") return ["assistant"];
  if (mode === "replay") return ["assistant", "knowledge"];
  return ["transcription", "assistant", "knowledge"];
}

/** A service whose absence makes the session impossible (the others make it worse). */
const ESSENTIAL = new Set(["transcription", "assistant"]);

export function rank(level) {
  const i = LEVELS.indexOf(level);
  return i < 0 ? 0 : i;
}

/** The worst of some levels (unknown ones count as good). */
export function worst(...levels) {
  return LEVELS[Math.max(0, ...levels.map(rank))];
}

/** The best of some levels. */
export function best(...levels) {
  return levels.length ? LEVELS[Math.min(...levels.map(rank))] : "good";
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Mean absolute change between consecutive values (0 with fewer than two). */
export function jitter(values) {
  if (values.length < 2) return 0;
  let sum = 0;
  for (let i = 1; i < values.length; i += 1) sum += Math.abs(values[i] - values[i - 1]);
  return sum / (values.length - 1);
}

/** The figures of some samples `{ok, ms}`: rttMs and jitterMs over the answered ones. */
export function summarise(samples) {
  const answered = samples.filter((s) => s.ok).map((s) => s.ms);
  const lost = samples.length - answered.length;
  return {
    count: samples.length,
    lost,
    lossPct: samples.length ? (100 * lost) / samples.length : 0,
    rttMs: median(answered),
    jitterMs: jitter(answered),
  };
}

/** The level of one figure; `higher` when more is better. Null figures are good (not measured). */
export function grade(value, { good, fair }, higher = false) {
  if (value === null || value === undefined || Number.isNaN(value)) return "good";
  if (higher) return value >= good ? "good" : value >= fair ? "fair" : "poor";
  return value <= good ? "good" : value <= fair ? "fair" : "poor";
}

/**
 * The upload rate in kbit/s of `bytes` sent in `ms`, less one round trip
 * (the answer's way back). A small body never leaves TCP's slow start, so
 * this is a lower bound, never a full speed test.
 */
export function uploadKbps(bytes, ms, rttMs = 0) {
  if (!bytes || !(ms > 0)) return null;
  const sending = Math.max(ms - (rttMs ?? 0), ms / 4, 1);
  return (bytes * 8) / sending;
}

/** The grades of the figures of the link's samples. */
function quality(stats) {
  return {
    rtt: grade(stats.rttMs, THRESHOLDS.rttMs),
    jitter: grade(stats.jitterMs, THRESHOLDS.jitterMs),
    loss: grade(stats.lossPct, THRESHOLDS.lossPct),
  };
}

/**
 * The verdict of a check.
 * `result`: { web: {reachable, samples}, link: {reachable, samples},
 *             upload: {bytes, ms} | null, services: {name: word} | null }
 * `options`: { needs: service names, upload: whether the browser sends audio }
 * Returns { level, advice, figures, rows }: advice is the key of the one
 * thing to do (null when all is well); rows the state of every service row
 * ("available" | "unavailable" | "not_set_up" | "unknown").
 */
export function assess(result, { needs = needsOf("live"), upload = false } = {}) {
  const web = result?.web ?? { reachable: false, samples: [] };
  const link = result?.link ?? { reachable: false, samples: [] };
  const services = result?.services ?? null;
  const linkStats = summarise(link.samples ?? []);
  const webStats = summarise(web.samples ?? []);
  const kbps = result?.upload ? uploadKbps(result.upload.bytes, result.upload.ms, linkStats.rttMs) : null;
  const figures = { ...linkStats, webRttMs: webStats.rttMs, upKbps: kbps };
  const rows = {
    web: web.reachable ? "available" : "unavailable",
    link: link.reachable ? "available" : "unavailable",
  };
  for (const name of ["transcription", "assistant", "knowledge"]) {
    rows[name] = !link.reachable ? "unavailable" : services?.[name] ?? "unknown";
  }
  if (!web.reachable && !link.reachable) return { level: "poor", advice: "offline", figures, rows };
  if (!link.reachable) return { level: "poor", advice: "server", figures, rows };

  const down = needs.filter((n) => rows[n] === "unavailable");
  const essential = down.find((n) => ESSENTIAL.has(n));
  if (essential) return { level: "poor", advice: `service_${essential}`, figures, rows };

  const q = quality(linkStats);
  const network = worst(q.rtt, q.jitter, q.loss);
  const up = upload ? grade(kbps, THRESHOLDS.upKbps, true) : "good";
  const knowledge = down.length ? "fair" : "good";
  const level = worst(network, up, knowledge);
  let advice = null;
  if (rank(network) > 0 && rank(network) >= rank(up)) {
    const w = quality(webStats);
    // The public server answers slowly too: the problem is this device's network.
    advice = rank(worst(w.rtt, w.jitter, w.loss)) > 0 ? "local" : "link";
  } else if (rank(up) > 0) {
    advice = "upload";
  } else if (down.length) {
    advice = `service_${down[0]}`;
  }
  return { level, advice, figures, rows };
}

/** Round a figure for the page: whole numbers, null as null. */
export function rounded(value) {
  return value === null || value === undefined || Number.isNaN(value) ? null : Math.round(value);
}

/**
 * The quiet indicator of a running session. It is fed one ping at a time
 * (`ping({ok, ms})`) and, now and then, the services (`services(map)`); it
 * gets worse only after WATCH.worse samples in a row are worse, better
 * after WATCH.better in a row are better, and says "lost" after
 * WATCH.lostAfter failed pings in a row. Each call returns
 * { state, changed, lost, back } (lost/back: the connection was just lost
 * or just came back).
 */
export class ConnectionWatch {
  constructor({ initial = null, needs = needsOf("live"), watch = WATCH } = {}) {
    this.w = watch;
    this.needs = needs;
    this.link = initial && LEVELS.includes(initial) ? initial : null; // null: not known yet
    this.samples = []; // {ok, level}
    this.failures = 0;
    this.lost = false;
    this.downCount = new Map(); // service -> checks in a row unavailable
    this.servicesLevel = "good";
  }

  /** The state shown: "checking" | "good" | "fair" | "poor" | "lost". */
  get state() {
    if (this.lost) return "lost";
    if (this.link === null) return "checking";
    const lostInWindow = this.samples.filter((s) => !s.ok).length;
    const capped = lostInWindow > this.w.maxLost ? "fair" : "good";
    return worst(this.link, capped, this.servicesLevel);
  }

  ping({ ok, ms }) {
    const before = this.state;
    const level = ok ? grade(ms, THRESHOLDS.rttMs) : "poor";
    this.samples.push({ ok, level });
    if (this.samples.length > this.w.window) this.samples.shift();
    let back = false;
    let lostNow = false;
    if (!ok) {
      this.failures += 1;
      if (this.failures >= this.w.lostAfter && !this.lost) {
        this.lost = true;
        lostNow = true;
      }
    } else {
      this.failures = 0;
      if (this.lost) {
        // Back: start again from this answer; the failures stay out of the window.
        this.lost = false;
        back = true;
        this.samples = [{ ok, level }];
        this.link = level;
      }
    }
    if (!back) this.link = this.next(this.link);
    const state = this.state;
    return { state, changed: state !== before, lost: lostNow, back };
  }

  next(current) {
    const levels = this.samples.map((s) => s.level);
    if (current === null) return levels.length ? levels[levels.length - 1] : null;
    const lastWorse = levels.slice(-this.w.worse);
    if (lastWorse.length >= this.w.worse && rank(best(...lastWorse)) > rank(current)) return best(...lastWorse);
    const lastBetter = levels.slice(-this.w.better);
    if (lastBetter.length >= this.w.better && rank(worst(...lastBetter)) < rank(current)) return worst(...lastBetter);
    return current;
  }

  /** Feed the services' words (`null`: they could not be asked; nothing changes). */
  services(map) {
    const before = this.state;
    if (map) {
      for (const name of this.needs) {
        const down = map[name] === "unavailable";
        this.downCount.set(name, down ? (this.downCount.get(name) ?? 0) + 1 : 0);
      }
      // Two checks in a row (about two minutes) before a service counts as down.
      const down = this.needs.filter((n) => (this.downCount.get(n) ?? 0) >= 2);
      this.servicesLevel = down.some((n) => ESSENTIAL.has(n)) ? "poor" : down.length ? "fair" : "good";
    }
    const state = this.state;
    return { state, changed: state !== before, lost: false, back: false };
  }
}
