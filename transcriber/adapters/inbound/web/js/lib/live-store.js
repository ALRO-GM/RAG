// Nothing lost on a long outage (#265): the live session's audio the server
// has not acknowledged is mirrored to this browser's IndexedDB, so a closed
// or crashed tab, a connection that only returns after «Terminar», or an
// outage longer than the page's 30 min memory still has it. Only when the
// product allows it (privacy.buffer_audio_on_outage), only until the server
// has it, at most `buffer_max_hours` after the session started.
//
// Format: database "ragputin-live-audio", two stores:
//   sessions  { id, product, plan, candidate_id, employee, channels,
//               started_at, expires_at, ended_locally, ended_at, chunks,
//               samples }  — no audio, no words, no names
//   chunks    { key: "<id>|<seq, 8 digits>", session, seq, offset,
//               samples, bytes }  — 16-bit PCM at 16 kHz, `channels`
//               interleaved; one chunk is at most 5 s (160 KB mono), and
//               only the part not yet acknowledged is written
// A chunk is deleted as soon as the server acknowledges everything in it,
// unless part of it became silence on the server (the page could not send
// it in time): that one is kept for the recovery upload. Expired sessions
// and their chunks are swept at every page load. A write the browser
// refuses (no room) is reported, never swallowed.

export const DB_NAME = "ragputin-live-audio";
export const SAMPLE_RATE = 16000;
export const CHUNK_SECONDS = 5;
export const DEFAULT_MAX_HOURS = 48;
const CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE;

/** The key of chunk `seq` of session `id` (sorts by session, then order). */
export function chunkKey(id, seq) {
  return `${id}|${String(seq).padStart(8, "0")}`;
}

function request(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** IndexedDB through a small promise interface (`factory` is `indexedDB`). */
export class IdbBackend {
  constructor(factory = globalThis.indexedDB) {
    this.factory = factory;
    this._db = null;
  }

  async db() {
    if (this._db) return this._db;
    if (!this.factory) throw Object.assign(new Error("no IndexedDB"), { name: "NotSupportedError" });
    const open = this.factory.open(DB_NAME, 1);
    open.onupgradeneeded = () => {
      const db = open.result;
      if (!db.objectStoreNames.contains("sessions")) db.createObjectStore("sessions", { keyPath: "id" });
      if (!db.objectStoreNames.contains("chunks")) db.createObjectStore("chunks", { keyPath: "key" });
    };
    this._db = await request(open);
    return this._db;
  }

  async _store(name, mode) {
    return (await this.db()).transaction(name, mode).objectStore(name);
  }

  async put(name, value) {
    return request((await this._store(name, "readwrite")).put(value));
  }

  async get(name, key) {
    return request((await this._store(name, "readonly")).get(key));
  }

  async delete(name, key) {
    return request((await this._store(name, "readwrite")).delete(key));
  }

  async keys(name) {
    return request((await this._store(name, "readonly")).getAllKeys());
  }

  async all(name) {
    return request((await this._store(name, "readonly")).getAll());
  }
}

/** What a failed write means for the page: "quota" (no room) or "storage". */
export function storageProblem(error) {
  return error?.name === "QuotaExceededError" || /quota/i.test(String(error?.message ?? "")) ? "quota" : "storage";
}

/** The session record kept with the audio (no words, no names). */
export function sessionMeta(session, { now = Date.now(), maxHours = DEFAULT_MAX_HOURS, channels = 1 } = {}) {
  const started = Number.isFinite(now) ? now : Date.now();
  return {
    id: session.id,
    product: session.product ?? null,
    plan: session.plan ?? null,
    candidate_id: session.candidate_id ?? null,
    employee: session.employee?.id ?? null,
    channels,
    started_at: new Date(started).toISOString(),
    expires_at: new Date(started + maxHours * 3600 * 1000).toISOString(),
    ended_locally: false,
    ended_at: null,
    chunks: 0,
    samples: 0,
  };
}

/**
 * The persistent outbox of one running session. `mirror(outbox, limit)`
 * writes what the in-memory Outbox holds and the server has not
 * acknowledged; `ack`, `keep` and `shift` follow the socket. Every write
 * goes through one promise chain (`flushed()` waits for it).
 */
export class LocalRecording {
  constructor({ backend, meta, onProblem = () => {} }) {
    this.backend = backend;
    this.meta = { ...meta };
    this.onProblem = onProblem;
    this.index = new Map(); // seq -> { offset, samples, kept }
    this.seq = 0;
    this.persistedEnd = null; // offsets before it were looked at
    this.failed = null; // "quota" | "storage" once a write failed
    this._chain = Promise.resolve();
  }

  /** Write the session record; chunks of an earlier page keep their numbers. */
  async open() {
    try {
      const known = (await this.backend.keys("chunks")).filter((k) => String(k).startsWith(`${this.meta.id}|`));
      this.seq = known.reduce((top, k) => Math.max(top, Number(String(k).split("|")[1]) + 1), 0);
      const earlier = await this.backend.get("sessions", this.meta.id);
      if (earlier) this.meta = { ...this.meta, started_at: earlier.started_at, expires_at: earlier.expires_at, chunks: earlier.chunks ?? 0, samples: earlier.samples ?? 0 };
      await this.backend.put("sessions", this.meta);
    } catch (error) {
      this._fail(error);
    }
    return this;
  }

  _fail(error) {
    const kind = storageProblem(error);
    if (this.failed !== kind) {
      this.failed = kind;
      this.onProblem(kind);
    }
  }

  _then(work) {
    this._chain = this._chain.then(work).catch((error) => this._fail(error));
    return this._chain;
  }

  /** Wait until every write asked so far has been tried. */
  flushed() {
    return this._chain;
  }

  /** Seconds of audio kept in this browser for the session. */
  get keptSeconds() {
    return (this.meta.samples ?? 0) / SAMPLE_RATE;
  }

  /**
   * Write what `outbox` holds before `limit` (a suspected pause is not
   * final yet) and the server has not acknowledged, in chunks of at most
   * 5 s; only once 5 s are waiting, unless `force`.
   */
  mirror(outbox, { limit = Infinity, force = false } = {}) {
    const end = Math.min(outbox.end, limit);
    if (this.persistedEnd === null) this.persistedEnd = Math.max(outbox.acked, outbox.start);
    if (end - this.persistedEnd < (force ? 1 : CHUNK_SAMPLES)) return this._chain;
    const from = this.persistedEnd;
    const groups = [];
    let group = null;
    for (const piece of outbox.pieces) {
      const last = piece.offset + piece.samples;
      if (piece.offset < from) continue; // looked at before
      if (last > end) break; // not final yet
      this.persistedEnd = last;
      if (!piece.bytes || last <= outbox.acked) {
        group = null; // silence, or already on the server
        continue;
      }
      if (!group || group.offset + group.samples !== piece.offset || group.samples + piece.samples > CHUNK_SAMPLES) {
        group = { offset: piece.offset, samples: 0, parts: [] };
        groups.push(group);
      }
      group.samples += piece.samples;
      group.parts.push(piece.bytes);
    }
    for (const g of groups) {
      const bytes = new Uint8Array(g.parts.reduce((n, p) => n + p.length, 0));
      let at = 0;
      for (const part of g.parts) {
        bytes.set(part, at);
        at += part.length;
      }
      const seq = this.seq;
      this.seq += 1;
      this.index.set(seq, { offset: g.offset, samples: g.samples, kept: false });
      this.meta.chunks += 1;
      this.meta.samples += g.samples;
      this._then(() => this.backend.put("chunks", {
        key: chunkKey(this.meta.id, seq), session: this.meta.id, seq, offset: g.offset, samples: g.samples, bytes,
      }));
    }
    if (groups.length) this._then(() => this.backend.put("sessions", { ...this.meta }));
    return this._chain;
  }

  /** The server took everything before `samples`: its chunks go. */
  ack(samples) {
    const gone = [];
    for (const [seq, chunk] of this.index) {
      if (!chunk.kept && chunk.offset + chunk.samples <= samples) gone.push(seq);
    }
    if (!gone.length) return this._chain;
    for (const seq of gone) {
      const chunk = this.index.get(seq);
      this.index.delete(seq);
      this.meta.chunks -= 1;
      this.meta.samples -= chunk.samples;
      this._then(() => this.backend.delete("chunks", chunkKey(this.meta.id, seq)));
    }
    this._then(() => this.backend.put("sessions", { ...this.meta }));
    return this._chain;
  }

  /** [from, to) became silence on the server: its chunks stay for the recovery. */
  keep(from, to) {
    for (const chunk of this.index.values()) {
      if (chunk.offset < to && chunk.offset + chunk.samples > from) chunk.kept = true;
    }
  }

  /** The server took less silence than asked: what follows moves by `delta`. */
  shift(from, delta) {
    if (!delta) return this._chain;
    if (this.persistedEnd !== null && this.persistedEnd >= from) this.persistedEnd += delta;
    for (const [seq, chunk] of this.index) {
      if (chunk.offset < from) continue;
      chunk.offset += delta;
      const key = chunkKey(this.meta.id, seq);
      this._then(async () => {
        const stored = await this.backend.get("chunks", key);
        if (stored) await this.backend.put("chunks", { ...stored, offset: chunk.offset });
      });
    }
    return this._chain;
  }

  /** «Terminar» without connection: the session ended here, its audio waits. */
  endLocally(now = Date.now()) {
    this.meta.ended_locally = true;
    this.meta.ended_at = new Date(now).toISOString();
    return this._then(() => this.backend.put("sessions", { ...this.meta }));
  }

  /** The session ended with everything acknowledged: nothing is kept. */
  async forgetIfEmpty() {
    await this.flushed();
    if ((this.meta.chunks ?? 0) > 0) return false;
    await discardRecording(this.backend, this.meta.id).catch(() => {});
    return true;
  }
}

/** The sessions whose audio waits in this browser, oldest first. */
export async function pendingRecordings(backend, now = Date.now()) {
  const sessions = await backend.all("sessions");
  return sessions
    .filter((s) => (s.chunks ?? 0) > 0 && Date.parse(s.expires_at) > now)
    .sort((a, b) => String(a.started_at).localeCompare(String(b.started_at)))
    .map((s) => ({ ...s, minutes: Math.max(Math.round((s.samples ?? 0) / SAMPLE_RATE / 60), 1) }));
}

/** Delete everything kept for session `id`. */
export async function discardRecording(backend, id) {
  for (const key of await backend.keys("chunks")) {
    if (String(key).startsWith(`${id}|`)) await backend.delete("chunks", key);
  }
  await backend.delete("sessions", id);
}

/**
 * Delete what is past its expiry, and chunks without a session (at every
 * page load). Returns { sessions, chunks } deleted (counts only).
 */
export async function sweepExpired(backend, now = Date.now()) {
  const sessions = await backend.all("sessions");
  const alive = new Set();
  let gone = 0;
  for (const s of sessions) {
    if (!(Date.parse(s.expires_at) > now)) {
      await backend.delete("sessions", s.id);
      gone += 1;
    } else {
      alive.add(s.id);
    }
  }
  let chunks = 0;
  for (const key of await backend.keys("chunks")) {
    if (!alive.has(String(key).split("|")[0])) {
      await backend.delete("chunks", key);
      chunks += 1;
    }
  }
  return { sessions: gone, chunks };
}

/** The sweep at page load: never throws (a browser without IndexedDB has nothing). */
export async function sweepAtLoad(backend = new IdbBackend(), now = Date.now()) {
  try {
    return await sweepExpired(backend, now);
  } catch {
    return { sessions: 0, chunks: 0 };
  }
}

const RETRY_MS = [1000, 2000, 4000];

/**
 * Send session `id`'s kept audio to the server (resumable: the chunks the
 * server already has are skipped), close the upload, then delete it here.
 *   api       { recoveryStatus(id), putRecoveryChunk(id, seq, offset, bytes, channels), closeRecovery(id, body) }
 *   onProgress(fraction)
 * Resolves the server's job view. A refusal for room (413) or anything
 * after three tries is thrown, the audio stays here.
 */
export async function uploadRecording({ backend, api, id, onProgress = () => {}, wait = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
  const meta = await backend.get("sessions", id);
  if (!meta) throw Object.assign(new Error("nothing kept"), { code: "not_found" });
  let have = new Set();
  try {
    const status = await api.recoveryStatus(id);
    if (status?.state && status.state !== "uploading") {
      await discardRecording(backend, id);
      return status;
    }
    have = new Set(status?.chunks ?? []);
  } catch (error) {
    if (error?.status !== 404) throw error;
  }
  const keys = (await backend.keys("chunks")).filter((k) => String(k).startsWith(`${id}|`)).sort();
  let done = 0;
  for (const key of keys) {
    const chunk = await backend.get("chunks", key);
    if (chunk && !have.has(chunk.seq)) {
      for (let attempt = 0; ; attempt += 1) {
        try {
          await api.putRecoveryChunk(id, chunk.seq, chunk.offset, chunk.bytes, meta.channels ?? 1);
          break;
        } catch (error) {
          if (error?.status === 413 || error?.status === 404 || error?.status === 409 || error?.kind === "unauthorized" || attempt >= RETRY_MS.length) throw error;
          await wait(RETRY_MS[attempt]);
        }
      }
    }
    done += 1;
    onProgress(done / keys.length);
  }
  const view = await api.closeRecovery(id, { count: keys.length, expires_at: meta.expires_at });
  await discardRecording(backend, id);
  return view;
}

/**
 * The line of a recovery job for the interviewer: { key, params } for t(),
 * or null. «Pendiente de analizar» → «Analizando (n %)» → «Completado
 * (n min recuperados)» / «No se pudo recuperar».
 */
export function recoveryLine(view) {
  if (!view?.state) return null;
  if (view.state === "done") return { key: "live.recovery.done", params: { n: formatMinutes(view.seconds) } };
  if (view.state === "failed") return { key: "live.recovery.failed", params: {} };
  if (view.state === "uploading" || view.state === "queued") return { key: "live.recovery.pending", params: {} };
  const pct = Math.round(Math.min(Math.max(view.state === "transcribing" ? (view.progress ?? 0) * 0.8 : 0.9, 0), 0.99) * 100);
  return { key: "live.recovery.analysing", params: { n: pct } };
}

function formatMinutes(seconds) {
  const minutes = (seconds ?? 0) / 60;
  return minutes >= 10 ? Math.round(minutes) : Math.max(Math.round(minutes * 10) / 10, 0.1);
}

/** Whether a recovery is still moving (worth asking again). */
export function recoveryMoving(view) {
  return Boolean(view?.state) && !["done", "failed"].includes(view.state);
}
