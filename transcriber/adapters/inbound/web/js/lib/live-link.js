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

import { productKind } from "./live.js";

export const SAMPLE_RATE = 16000;
export const MAX_OUTBOX_SECONDS = 30 * 60;
export const BACKOFF_FIRST_MS = 500;
export const BACKOFF_MAX_MS = 8000;
/** Pending audio above this many seconds still says «enviando lo pendiente». */
export const CATCHING_UP_SECONDS = 2;
export const SESSION_KEY = "ragputin.live.running";

/**
 * Unacknowledged PCM (16-bit, `channels` interleaved) with absolute offsets.
 * A piece without bytes is silence: time the microphone produced nothing
 * (a suspended tab, a laptop asleep), sent as a gap so the stream stays on
 * the wall clock.
 */
export class Outbox {
  constructor({ maxSeconds = MAX_OUTBOX_SECONDS, channels = 1 } = {}) {
    this.maxSamples = Math.round(maxSeconds * SAMPLE_RATE);
    this.frameBytes = 2 * channels;
    this.pieces = []; // { offset, samples, bytes | null }, in order, contiguous
    this.end = 0; // the offset after the last sample captured
    this.acked = 0; // everything before it was taken by the server
    this.lost = 0; // samples of audio dropped because the outbox was full
    this.held = 0; // samples kept (audio and silence)
    this.audio = 0; // samples of audio kept
  }

  _add(piece) {
    this.pieces.push(piece);
    this.end += piece.samples;
    this.held += piece.samples;
    if (piece.bytes) this.audio += piece.samples;
    while (this.held > this.maxSamples && this.pieces.length > 1) this._drop(true);
  }

  _drop(lost) {
    const oldest = this.pieces.shift();
    this.held -= oldest.samples;
    if (oldest.bytes) {
      this.audio -= oldest.samples;
      if (lost) this.lost += oldest.samples;
    }
  }

  /** Keep one piece of captured audio (whole frames). */
  push(bytes) {
    const samples = Math.floor(bytes.length / this.frameBytes);
    if (samples) this._add({ offset: this.end, samples, bytes });
  }

  /** Keep `samples` of silence (the microphone paused). */
  skip(samples) {
    const count = Math.floor(samples);
    if (count > 0) this._add({ offset: this.end, samples: count, bytes: null });
  }

  /** The server took everything before `samples`: forget it. */
  ack(samples) {
    if (!Number.isFinite(samples) || samples <= this.acked) return;
    this.acked = samples;
    while (this.pieces.length && this.pieces[0].offset + this.pieces[0].samples <= this.acked) this._drop(false);
  }

  /** The first offset still held (`end` when empty). */
  get start() {
    return this.pieces.length ? this.pieces[0].offset : this.end;
  }

  /** Seconds of audio captured and not yet acknowledged (not silence, not what was dropped). */
  get pendingSeconds() {
    const first = this.pieces[0];
    const taken = first?.bytes ? Math.max(Math.min(this.acked - first.offset, first.samples), 0) : 0;
    return Math.max(this.audio - taken, 0) / SAMPLE_RATE;
  }

  /**
   * What to send from `offset` on: `gap` samples the outbox no longer has
   * (they became silence on the server), then the pieces, the first one
   * cut so that it starts exactly at `offset`; a piece with `bytes: null`
   * is silence, sent as a gap too.
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
        pieces.push({ offset: at, samples: last - at, bytes: piece.bytes ? piece.bytes.subarray(skip) : null });
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
 * Returns { key, count, params } for t(): `key` is the sentence (it changes
 * only with the state, so a screen reader hears it once) and `count` the
 * seconds still waiting, drawn outside the live region. The audio of this
 * page comes first.
 */
export function linkLine({ audio = null, feed = false } = {}) {
  const n = Math.max(Math.round(audio?.pendingSeconds ?? 0), 0);
  if (audio && audio.everOpen && (audio.state === "lost" || audio.state === "connecting")) {
    return { key: "live.link.lost", count: "live.link.pending", params: { n } };
  }
  if (audio && audio.state === "open" && (audio.pendingSeconds ?? 0) > CATCHING_UP_SECONDS) {
    return { key: "live.link.sending", count: "live.link.pending", params: { n } };
  }
  if (feed) return { key: "live.reconnecting", count: null, params: {} };
  return null;
}

/**
 * The line shown while «Terminar» waits for what is still held (#263),
 * or null when nothing waits. `count` is drawn outside the live region.
 */
export function finishLine(link) {
  if (!link || (link.pendingSeconds ?? 0) <= 0) return null;
  const n = Math.max(Math.ceil(link.pendingSeconds), 1);
  const offline = link.state !== "open";
  return { key: offline ? "live.finish.offline" : "live.finish.sending", count: "live.link.pending", button: "live.finish.now", params: { n } };
}

/**
 * The page that shows a session: "voicebot" for the voice assistant's
 * conversations, "live" for everything else (support, interview, replays).
 */
export function sessionPage(session) {
  if (!session) return "live";
  if (productKind(session.product) || ["demo", "replay"].includes(session.mode)) return "live";
  return "voicebot";
}

/** The first running session of a list of the orchestrator's sessions, or null. */
export function runningSession(sessions) {
  return (sessions ?? []).find((s) => s?.state === "running") ?? null;
}

/**
 * Where to reattach a remembered session: { show: view } when it still
 * runs, else { form: true, forget, error } (forget it unless the server
 * could not be reached; `error` to show).
 */
export async function reattachPlan(liveSession, kept) {
  try {
    const session = await liveSession(kept.id);
    if (session?.state === "running") return { show: session };
    return { form: true, forget: true, error: null };
  } catch (error) {
    if (error?.kind === "unauthorized") return { unauthorized: error };
    const ended = feedOutcome(error) === "ended";
    return { form: true, forget: ended, error: ended ? null : error };
  }
}

/**
 * Follow a session's stream until it ends (#263): `follow()` resolves
 * "end" when the session is over, anything else to read again; a 404
 * ends the following (no endless retry), a refused code returns at once.
 * Resolves { how: "end" | "ended" | "unauthorized" | "aborted", error }.
 */
export async function followUntilEnd({ follow, wait, retryMs, stopped = () => false, onRetry = () => {} }) {
  while (!stopped()) {
    try {
      if (await follow() === "end") return { how: "end" };
    } catch (error) {
      const outcome = feedOutcome(error);
      if (outcome === "unauthorized") return { how: "unauthorized", error };
      if (outcome === "ended") return { how: "ended" };
    }
    if (stopped()) break;
    onRetry();
    await wait(retryMs);
  }
  return { how: "aborted" };
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

/** Remember the running session of this tab: its id, audio kind and page ("live" | "voicebot"). */
export function rememberSession(session, audio, page = "live", storage = undefined) {
  const store = storageOf(storage);
  if (!store || !session?.id) return;
  try {
    store.setItem(SESSION_KEY, JSON.stringify({ id: session.id, audio: audio ?? null, page }));
  } catch {
    // private mode or full: a reload then shows the form, as before
  }
}

/** The remembered session ({ id, audio, page }), or null. */
export function rememberedSession(storage) {
  const store = storageOf(storage);
  if (!store) return null;
  try {
    const kept = JSON.parse(store.getItem(SESSION_KEY) ?? "null");
    return kept && typeof kept.id === "string" && kept.id
      ? { id: kept.id, audio: kept.audio ?? null, page: kept.page === "voicebot" ? "voicebot" : "live" } : null;
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
