// Nothing lost on a long outage (#265): the live session's audio the server
// has not acknowledged is mirrored to this browser's IndexedDB, so a closed
// or crashed tab, a connection that only returns after «Terminar», or an
// outage longer than the page's 30 min memory still has it. Only when the
// product allows it (privacy.buffer_audio_on_outage), only while the link
// is down or more than a few seconds wait for their ack (a healthy session
// writes nothing), only until the server has it, at most
// `buffer_max_hours` after the session started.
//
// Format: database "ragputin-live-audio" (version 2), three stores:
//   sessions  { id, product, plan, candidate_id, employee, channels,
//               started_at, expires_at, ended_locally, ended_at }
//               — no audio, no words, no names
//   chunks    { key: "<id>|<seq, 8 digits>", bytes } — 16-bit PCM at
//               16 kHz, `channels` interleaved; at most 5 s (160 KB mono)
//   index     { key, session, seq, offset, samples } — the same rows
//               without the audio, written in the same transaction, so
//               what waits is counted from the rows themselves
// A chunk is deleted as soon as the server acknowledges everything in it,
// unless part of it became silence on the server (the page could not send
// it in time): that one is kept for the recovery upload. Expired sessions
// and stray rows are swept when any page of the app loads, every time it
// becomes visible and every 10 minutes while it is open. A write the
// browser refuses (no room) is reported, never swallowed.

export const DB_NAME = "ragputin-live-audio";
export const DB_VERSION = 2;
export const SAMPLE_RATE = 16000;
export const CHUNK_SECONDS = 5;
export const DEFAULT_MAX_HOURS = 48;
export const SWEEP_EVERY_MS = 10 * 60 * 1000;
const CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE;
const STORES = { sessions: "id", chunks: "key", index: "key" };

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

function done(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? Object.assign(new Error("aborted"), { name: "AbortError" }));
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
    const open = this.factory.open(DB_NAME, DB_VERSION);
    open.onupgradeneeded = () => {
      const db = open.result;
      for (const [name, keyPath] of Object.entries(STORES)) {
        if (!db.objectStoreNames.contains(name)) db.createObjectStore(name, { keyPath });
      }
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

  /** Several writes, all or none: [["put", store, value] | ["delete", store, key]]. */
  async write(operations) {
    const names = [...new Set(operations.map((o) => o[1]))];
    const tx = (await this.db()).transaction(names, "readwrite");
    const finished = done(tx);
    for (const [kind, name, value] of operations) {
      if (kind === "put") tx.objectStore(name).put(value);
      else tx.objectStore(name).delete(value);
    }
    return finished;
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
  };
}

/** The index rows of session `id`. */
async function rowsOf(backend, id) {
  return (await backend.all("index")).filter((row) => row.session === id);
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
    this.opened = false; // the session record was written
    this._chain = Promise.resolve();
  }

  /** Read what an earlier page of the session left; the record is written with the first chunk. */
  async open() {
    try {
      const rows = await rowsOf(this.backend, this.meta.id);
      this.seq = rows.reduce((top, r) => Math.max(top, r.seq + 1), 0);
      const earlier = await this.backend.get("sessions", this.meta.id);
      if (earlier) {
        this.meta = { ...this.meta, started_at: earlier.started_at, expires_at: earlier.expires_at };
        this.opened = true;
      }
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

  /** Seconds of audio this page wrote and still keeps. */
  get keptSeconds() {
    let samples = 0;
    for (const chunk of this.index.values()) samples += chunk.samples;
    return samples / SAMPLE_RATE;
  }

  /**
   * Write what `outbox` holds before `limit` (a suspected pause is not
   * final yet) and the server has not acknowledged, in chunks of at most
   * 5 s; only once 5 s are waiting, unless `force`. The caller decides
   * when there is anything to keep (a link down, a backlog, «Terminar»).
   */
  mirror(outbox, { limit = Infinity, force = false } = {}) {
    const end = Math.min(outbox.end, limit);
    if (this.persistedEnd === null || this.persistedEnd < outbox.acked) {
      this.persistedEnd = Math.max(this.persistedEnd ?? 0, outbox.acked, outbox.start);
    }
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
    if (!groups.length) return this._chain;
    const operations = [];
    if (!this.opened) {
      operations.push(["put", "sessions", { ...this.meta }]);
      this.opened = true;
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
      const key = chunkKey(this.meta.id, seq);
      operations.push(["put", "chunks", { key, bytes }],
        ["put", "index", { key, session: this.meta.id, seq, offset: g.offset, samples: g.samples }]);
    }
    return this._then(() => this.backend.write(operations));
  }

  /** The server took everything before `samples`: its chunks go. */
  ack(samples) {
    const operations = [];
    for (const [seq, chunk] of this.index) {
      if (chunk.kept || chunk.offset + chunk.samples > samples) continue;
      this.index.delete(seq);
      const key = chunkKey(this.meta.id, seq);
      operations.push(["delete", "chunks", key], ["delete", "index", key]);
    }
    if (!operations.length) return this._chain;
    return this._then(() => this.backend.write(operations));
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
        const stored = await this.backend.get("index", key);
        if (stored) await this.backend.put("index", { ...stored, offset: chunk.offset });
      });
    }
    return this._chain;
  }

  /** «Terminar» without connection: the session ended here, its audio waits. */
  endLocally(now = Date.now()) {
    this.meta.ended_locally = true;
    this.meta.ended_at = new Date(now).toISOString();
    if (!this.opened) return this._chain; // nothing was written: nothing waits
    return this._then(() => this.backend.put("sessions", { ...this.meta }));
  }

  /** Nothing of the session waits here (all acknowledged): nothing is kept. */
  async forgetIfEmpty() {
    await this.flushed();
    try {
      if ((await rowsOf(this.backend, this.meta.id)).length) return false;
      await discardRecording(this.backend, this.meta.id);
    } catch {
      return true;
    }
    return true;
  }
}

/** The sessions whose audio waits in this browser, oldest first, counted from the rows. */
export async function pendingRecordings(backend, now = Date.now()) {
  const rows = await backend.all("index");
  const samples = new Map();
  for (const row of rows) samples.set(row.session, (samples.get(row.session) ?? 0) + (row.samples ?? 0));
  const sessions = await backend.all("sessions");
  return sessions
    .filter((s) => samples.get(s.id) > 0 && Date.parse(s.expires_at) > now)
    .sort((a, b) => String(a.started_at).localeCompare(String(b.started_at)))
    .map((s) => ({
      ...s, samples: samples.get(s.id), chunks: rows.filter((r) => r.session === s.id).length,
      minutes: Math.max(Math.round(samples.get(s.id) / SAMPLE_RATE / 60), 1),
    }));
}

/** Delete everything kept for session `id` (or only chunks `keys`). */
export async function discardRecording(backend, id, keys = null) {
  const rows = keys ?? (await rowsOf(backend, id)).map((r) => r.key);
  const operations = rows.flatMap((key) => [["delete", "chunks", key], ["delete", "index", key]]);
  if (!keys) operations.push(["delete", "sessions", id]);
  if (operations.length) await backend.write(operations);
}

/**
 * Delete what is past its expiry, and rows of no session. The rows are
 * read before the sessions, so a session another tab is creating at the
 * same time is never taken for a stray. Returns { sessions, chunks }
 * deleted (counts only).
 */
export async function sweepExpired(backend, now = Date.now()) {
  const keys = [...new Set([...(await backend.keys("index")), ...(await backend.keys("chunks"))])];
  const sessions = await backend.all("sessions");
  const alive = new Set();
  let gone = 0;
  for (const s of sessions) {
    if (Date.parse(s.expires_at) > now) {
      alive.add(s.id);
    } else {
      await backend.delete("sessions", s.id);
      gone += 1;
    }
  }
  const stray = keys.filter((key) => !alive.has(String(key).split("|")[0]));
  if (stray.length) await backend.write(stray.flatMap((key) => [["delete", "chunks", key], ["delete", "index", key]]));
  return { sessions: gone, chunks: stray.length };
}

/** One sweep that never throws (a browser without IndexedDB has nothing). */
export async function sweepAtLoad(backend = new IdbBackend(), now = Date.now()) {
  try {
    return await sweepExpired(backend, now);
  } catch {
    return { sessions: 0, chunks: 0 };
  }
}

/**
 * Sweep now, whenever the page becomes visible again, and every 10 min
 * while it is open (any page of the app). Returns a function that stops.
 */
export function installSweeps({ backend = new IdbBackend(), target = globalThis, everyMs = SWEEP_EVERY_MS, now = () => Date.now() } = {}) {
  const sweep = () => sweepAtLoad(backend, now());
  sweep();
  const onVisible = () => {
    if (target.document?.visibilityState !== "hidden") sweep();
  };
  target.addEventListener?.("visibilitychange", onVisible);
  const timer = target.setInterval?.(sweep, everyMs);
  return () => {
    target.removeEventListener?.("visibilitychange", onVisible);
    if (timer !== undefined) target.clearInterval?.(timer);
  };
}

const RETRY_MS = [1000, 2000, 4000];

/** Why the server refused a recovery upload, for the page: a code. */
function refusal(error) {
  if (error?.code === "session_running") return "running";
  if (error?.code === "not_buffered") return "not_buffered";
  if (error?.status === 413) return "no_room";
  return null;
}

/**
 * Send session `id`'s kept audio to the server (resumable: the chunks the
 * server already has are skipped), close the upload, then delete here
 * exactly the chunks the server holds.
 *   api       { recoveryStatus(id), putRecoveryChunk(id, seq, offset, bytes, channels),
 *               closeRecovery(id, body), stopLive?(id) }
 *   onProgress(fraction)
 * A chunk the server refuses as such (400: outside the time the
 * session's audio may be kept) is skipped, the rest is sent and closed,
 * and the refused ones are deleted here: the view says how many seconds
 * (`refused_seconds`). A session this browser ended offline that the
 * server still holds is stopped first (`stopLive`), then sent. Thrown,
 * with the audio kept here: no room on the server (code "no_room"), a
 * session still on ("running"), a product that keeps no audio
 * ("not_buffered"), a closed upload missing some of these chunks
 * ("closed"), anything else after three tries.
 */
export async function uploadRecording({ backend, api, id, onProgress = () => {}, wait = (ms) => new Promise((r) => setTimeout(r, ms)) }) {
  const meta = await backend.get("sessions", id);
  if (!meta) throw Object.assign(new Error("nothing kept"), { code: "not_found" });
  const rows = (await rowsOf(backend, id)).sort((a, b) => a.seq - b.seq);
  let have = new Set();
  try {
    const status = await api.recoveryStatus(id);
    have = new Set(status?.chunks ?? []);
    if (status?.state && status.state !== "uploading") {
      // Closed already: only what it holds may go from here.
      const held = rows.filter((r) => have.has(r.seq)).map((r) => r.key);
      await discardRecording(backend, id, held);
      if (held.length === rows.length) {
        await discardRecording(backend, id);
        return status;
      }
      throw Object.assign(new Error("closed"), { code: "closed" });
    }
  } catch (error) {
    if (error?.code === "closed" || error?.status !== 404) throw error;
  }
  let stopped = false;
  let sent = 0;
  const refused = [];
  for (const row of rows) {
    if (!have.has(row.seq)) {
      const chunk = await backend.get("chunks", row.key);
      if (!chunk) throw Object.assign(new Error("chunk gone"), { code: "storage" });
      for (let attempt = 0; ; attempt += 1) {
        try {
          await api.putRecoveryChunk(id, row.seq, row.offset, chunk.bytes, meta.channels ?? 1);
          break;
        } catch (error) {
          const why = refusal(error);
          if (why === "running" && meta.ended_locally && api.stopLive && !stopped) {
            // Ended here offline while the server still waits for it:
            // stop it now that there is a connection, then send.
            stopped = true;
            await api.stopLive(id).catch(() => {});
            attempt = -1;
            continue;
          }
          if (why === "running" && stopped && attempt < RETRY_MS.length) {
            await wait(RETRY_MS[attempt]); // its hand-over is finishing
            continue;
          }
          if (why) throw Object.assign(error, { code: why });
          if (error?.status === 400) {
            refused.push(row);
            break;
          }
          if (error?.status === 404 || error?.kind === "unauthorized" || attempt >= RETRY_MS.length) throw error;
          await wait(RETRY_MS[attempt]);
        }
      }
    }
    sent += 1;
    onProgress(sent / rows.length);
  }
  const accepted = rows.filter((r) => !refused.includes(r));
  let view = { state: "refused" };
  if (accepted.length) {
    try {
      view = await api.closeRecovery(id, { count: accepted.length, expires_at: meta.expires_at });
    } catch (error) {
      const why = refusal(error);
      if (why) throw Object.assign(error, { code: why });
      throw error;
    }
  }
  await discardRecording(backend, id, rows.map((r) => r.key));
  if (!(await rowsOf(backend, id)).length) await backend.delete("sessions", id);
  const refusedSeconds = refused.reduce((n, r) => n + r.samples, 0) / SAMPLE_RATE;
  return refused.length ? { ...view, refused_seconds: refusedSeconds } : view;
}

/**
 * The line of a recovery job for the interviewer: { key, params } for t(),
 * or null. «Pendiente de analizar» → «Analizando (n %)» → «Completado
 * (n min recuperados)» (or «…, sin identificar quién habla») / «No se pudo
 * recuperar».
 */
export function recoveryLine(view) {
  if (!view?.state) return null;
  if (view.state === "done" && !(view.added > 0)) return { key: "live.recovery.done_nothing", params: {} };
  if (view.state === "done") {
    return { key: view.unidentified ? "live.recovery.done_unidentified" : "live.recovery.done", params: { n: formatMinutes(view.seconds) } };
  }
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
