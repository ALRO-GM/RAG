// A short connection loss must cost nothing (#263): the pure parts of the
// live page's links, tested without a browser.
//
// - Outbox: the microphone's PCM not yet acknowledged by the transcriber,
//   with absolute sample offsets. It lives only in this page's memory and
//   is forgotten as soon as the server acknowledges it; bounded (30 min),
//   the oldest audio goes first and becomes a gap the server fills with
//   silence.
// - backoffMs: how long to wait before the next reconnection.
// - linkLine: the one calm status line the interviewer sees.
// - rememberSession / rememberedSession / forgetSession: the running
//   session kept in sessionStorage, so a reload reattaches to it.
// - feedOutcome: what a failed read of the session's stream means.

export const SAMPLE_RATE = 16000;
export const MAX_OUTBOX_SECONDS = 30 * 60;
export const BACKOFF_FIRST_MS = 500;
export const BACKOFF_MAX_MS = 8000;
/** Pending audio above this many seconds still says «enviando lo pendiente». */
export const CATCHING_UP_SECONDS = 2;
export const SESSION_KEY = "ragputin.live.running";

/** Unacknowledged PCM (16-bit, `channels` interleaved) with absolute offsets. */
export class Outbox {
  constructor({ maxSeconds = MAX_OUTBOX_SECONDS, channels = 1 } = {}) {
    this.maxSamples = Math.round(maxSeconds * SAMPLE_RATE);
    this.frameBytes = 2 * channels;
    this.pieces = []; // { offset, samples, bytes }, in order, contiguous
    this.end = 0; // the offset after the last sample captured
    this.acked = 0; // everything before it was taken by the server
    this.lost = 0; // samples dropped because the outbox was full
    this.held = 0; // samples kept
  }

  /** Keep one piece of captured audio (whole frames). */
  push(bytes) {
    const samples = Math.floor(bytes.length / this.frameBytes);
    if (!samples) return;
    this.pieces.push({ offset: this.end, samples, bytes });
    this.end += samples;
    this.held += samples;
    while (this.held > this.maxSamples && this.pieces.length > 1) {
      const oldest = this.pieces.shift();
      this.held -= oldest.samples;
      this.lost += oldest.samples;
    }
  }

  /** The server took everything before `samples`: forget it. */
  ack(samples) {
    if (!Number.isFinite(samples) || samples <= this.acked) return;
    this.acked = samples;
    while (this.pieces.length && this.pieces[0].offset + this.pieces[0].samples <= this.acked) {
      this.held -= this.pieces.shift().samples;
    }
  }

  /** The first offset still held (`end` when empty). */
  get start() {
    return this.pieces.length ? this.pieces[0].offset : this.end;
  }

  /** Seconds captured and not yet acknowledged. */
  get pendingSeconds() {
    return Math.max(this.end - this.acked, 0) / SAMPLE_RATE;
  }

  /**
   * What to send from `offset` on: `gap` samples the outbox no longer has
   * (they became silence on the server), then the pieces, the first one
   * cut so that it starts exactly at `offset`.
   */
  from(offset) {
    const gap = Math.max(this.start - offset, 0);
    const at = offset + gap;
    const pieces = [];
    for (const piece of this.pieces) {
      const last = piece.offset + piece.samples;
      if (last <= at) continue;
      if (piece.offset < at) {
        const skip = (at - piece.offset) * this.frameBytes;
        pieces.push({ offset: at, samples: last - at, bytes: piece.bytes.subarray(skip) });
      } else {
        pieces.push(piece);
      }
    }
    return { gap, pieces };
  }

  /** A page that reattached: its audio starts at the server's count. */
  rebase(base) {
    const shift = base - this.acked;
    if (shift <= 0) return;
    for (const piece of this.pieces) piece.offset += shift;
    this.end += shift;
    this.acked = base;
  }
}

/** The wait before reconnection `attempt` (0, 1, …): 0.5 → 8 s, ±20 % jitter. */
export function backoffMs(attempt, random = Math.random) {
  const base = Math.min(BACKOFF_FIRST_MS * 2 ** Math.max(attempt, 0), BACKOFF_MAX_MS);
  return Math.round(base * (0.8 + 0.4 * random()));
}

/**
 * The one status line of the running page, or null (all is well).
 *   audio  { state: "connecting" | "open" | "lost" | "ended", pendingSeconds, everOpen } | null
 *   feed   true while the session's stream is reconnecting
 * Returns { key, params } for t(); the audio of this page comes first.
 */
export function linkLine({ audio = null, feed = false } = {}) {
  const pending = Math.max(Math.round(audio?.pendingSeconds ?? 0), 0);
  if (audio && audio.everOpen && (audio.state === "lost" || audio.state === "connecting")) {
    return { key: "live.link.lost", params: { n: pending } };
  }
  if (audio && audio.state === "open" && (audio.pendingSeconds ?? 0) > CATCHING_UP_SECONDS) {
    return { key: "live.link.sending", params: { n: pending } };
  }
  if (feed) return { key: "live.reconnecting", params: {} };
  return null;
}

/** What a failed read of the session's stream means: "ended", "unauthorized" or "retry". */
export function feedOutcome(error) {
  if (error?.kind === "unauthorized") return "unauthorized";
  if (error?.status === 404 || error?.code === "not_found") return "ended";
  return "retry";
}

function storageOf(storage) {
  if (storage !== undefined) return storage;
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

/** Remember the running live session of this tab (id and audio kind). */
export function rememberSession(session, audio, storage) {
  const store = storageOf(storage);
  if (!store || !session?.id) return;
  try {
    store.setItem(SESSION_KEY, JSON.stringify({ id: session.id, audio: audio ?? null }));
  } catch {
    // private mode or full: a reload then shows the form, as before
  }
}

/** The remembered session ({ id, audio }), or null. */
export function rememberedSession(storage) {
  const store = storageOf(storage);
  if (!store) return null;
  try {
    const kept = JSON.parse(store.getItem(SESSION_KEY) ?? "null");
    return kept && typeof kept.id === "string" && kept.id ? { id: kept.id, audio: kept.audio ?? null } : null;
  } catch {
    return null;
  }
}

/** Forget the remembered session (only `id`'s, when given). */
export function forgetSession(id, storage) {
  const store = storageOf(storage);
  if (!store) return;
  try {
    if (id && rememberedSession(store)?.id !== id) return;
    store.removeItem(SESSION_KEY);
  } catch {
    // nothing to forget
  }
}
