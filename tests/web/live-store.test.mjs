// #265: nothing lost on a long outage, and nothing written when nothing is
// lost. The audio the server has not acknowledged is mirrored to IndexedDB
// (a fake one here, with a quota and all-or-none transactions) only while
// the link is down or a backlog waits; deleted as soon as it is
// acknowledged, kept when it became silence on the server, shifted when the
// server clamps a gap; «Terminar» without connection ends locally and keeps
// it; the next visit sends it (resumable, retried, a full server reported, a
// session still on sent back to, a closed upload never losing late chunks
// silently) or discards it; expired audio is swept at load, on return and
// on a timer, with no race with another tab. Invented bytes only.
import test from "node:test";
import assert from "node:assert/strict";
import { LiveAudio } from "../../transcriber/adapters/inbound/web/js/live-audio.js";
import { finishLine } from "../../transcriber/adapters/inbound/web/js/lib/live-link.js";
import {
  CHUNK_SECONDS, IdbBackend, LocalRecording, chunkKey, discardRecording, installSweeps, pendingRecordings,
  recoveryLine, recoveryMoving, sessionMeta, storageProblem, sweepAtLoad, sweepExpired, uploadRecording,
} from "../../transcriber/adapters/inbound/web/js/lib/live-store.js";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import { setLanguage, t } from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

// ---- a fake IndexedDB: the calls the store makes, answered asynchronously

class FakeIndexedDB {
  constructor({ quota = Infinity } = {}) {
    this.stores = new Map();
    this.quota = quota;
    this.writes = 0;
  }

  used() {
    let bytes = 0;
    for (const store of this.stores.values()) for (const v of store.rows.values()) bytes += v.bytes?.length ?? 64;
    return bytes;
  }

  open() {
    const self = this;
    const db = {
      objectStoreNames: { contains: (name) => self.stores.has(name) },
      createObjectStore: (name, { keyPath }) => self.stores.set(name, { keyPath, rows: new Map() }),
      transaction: (names) => self._transaction(Array.isArray(names) ? names : [names]),
    };
    const req = { result: db };
    setTimeout(() => {
      req.onupgradeneeded?.();
      req.onsuccess?.();
    }, 0);
    return req;
  }

  _transaction(names) {
    const self = this;
    const staged = names.map((n) => [n, new Map(this.stores.get(n).rows)]);
    const copies = new Map(staged);
    const tx = { error: null };
    let failed = null;
    const pending = [];
    const finish = () => {
      if (failed) {
        tx.error = failed;
        tx.onerror?.();
        return;
      }
      for (const [name, rows] of copies) self.stores.get(name).rows = rows;
      tx.oncomplete?.();
    };
    setTimeout(() => setTimeout(finish, 0), 0);
    tx.objectStore = (name) => {
      const { keyPath } = self.stores.get(name);
      const rows = copies.get(name);
      const req = (work) => {
        const r = {};
        pending.push(r);
        setTimeout(() => {
          try {
            r.result = work();
            r.onsuccess?.();
          } catch (error) {
            failed = error;
            r.error = error;
            r.onerror?.();
          }
        }, 0);
        return r;
      };
      return {
        put: (value) => req(() => {
          self.writes += 1;
          if (self.used() + (value.bytes?.length ?? 64) > self.quota) {
            throw Object.assign(new Error("full"), { name: "QuotaExceededError" });
          }
          rows.set(value[keyPath], structuredClone(value));
        }),
        get: (key) => req(() => (rows.has(key) ? structuredClone(rows.get(key)) : undefined)),
        delete: (key) => req(() => { rows.delete(key); }),
        getAllKeys: () => req(() => [...rows.keys()].sort()),
        getAll: () => req(() => [...rows.values()].map((v) => structuredClone(v))),
      };
    };
    return tx;
  }
}

// ---- the socket, the microphone and the clock (as in live-reconnect)

class Clock {
  constructor() {
    this.t = 0;
    this.jobs = [];
    this.next = 1;
  }

  setTimeout = (fn, ms) => {
    const id = this.next++;
    this.jobs.push({ id, at: this.t + ms, fn });
    return id;
  };

  clearTimeout = (id) => {
    this.jobs = this.jobs.filter((j) => j.id !== id);
  };

  now = () => this.t;

  advance(ms) {
    const end = this.t + ms;
    for (;;) {
      this.jobs.sort((a, b) => a.at - b.at);
      const job = this.jobs[0];
      if (!job || job.at > end) break;
      this.jobs.shift();
      this.t = job.at;
      job.fn();
    }
    this.t = end;
  }
}

function rig({ quota } = {}) {
  const clock = new Clock();
  const made = [];
  class FakeSocket {
    constructor() {
      this.readyState = 0;
      this.bufferedAmount = 0;
      this.sent = [];
      this.listeners = {};
      made.push(this);
    }

    addEventListener(kind, fn) { (this.listeners[kind] ??= []).push(fn); }

    emit(kind, event = {}) { for (const fn of this.listeners[kind] ?? []) fn(event); }

    send(data) { this.sent.push(data); }

    open() {
      this.readyState = 1;
      this.emit("open");
    }

    server(message) { this.emit("message", { data: JSON.stringify(message) }); }

    close() {
      if (this.readyState === 3) return;
      this.readyState = 3;
      this.emit("close", { code: 1006 });
    }

    texts() { return this.sent.filter((d) => typeof d === "string").map((d) => JSON.parse(d)); }
  }
  const box = { onPcm: null };
  class FakeCapture {
    constructor({ onPcm }) { box.onPcm = onPcm; }

    async start() {}

    async stop() {}
  }
  const idb = new FakeIndexedDB({ quota });
  const backend = new IdbBackend(idb);
  const problems = [];
  const recording = new LocalRecording({
    backend, meta: sessionMeta({ id: "iv-1", product: "live_assist.interview", plan: "dev", candidate_id: "c1" },
      { now: Date.parse("2026-11-26T10:00:00Z") }),
    onProblem: (kind) => problems.push(kind),
  });
  const audio = new LiveAudio({
    id: "iv-1", code: "c0de", recording,
    deps: {
      openSocket: () => new FakeSocket(), setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout,
      now: clock.now, random: () => 0.5, Capture: FakeCapture, location: { protocol: "https:", host: "x.example" },
    },
  });
  return { clock, made, box, audio, recording, backend, idb, problems };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
const piece = (n) => new Uint8Array(3200).fill(n); // 100 ms

async function started(r, received = 0) {
  await r.recording.open();
  const going = r.audio.start();
  await settle();
  const socket = r.made.at(-1);
  socket.open();
  socket.server({ type: "ready", received_samples: received });
  await going;
  return socket;
}

/** «Terminar» with no connection, then «Terminar ya» once it waits. */
async function finishOffline(r) {
  let done = null;
  r.audio.finish().then((result) => { done = result; });
  for (let i = 0; i < 200 && !r.audio._finish && done === null; i += 1) await settle();
  r.audio.endNow();
  for (let i = 0; i < 200 && done === null; i += 1) await settle();
  return done;
}

async function rows(backend) {
  return (await backend.all("index")).map((c) => [c.seq, c.offset, c.samples]);
}

test("a healthy session writes nothing to the browser's disk", async () => {
  const r = rig();
  const socket = await started(r);
  for (let n = 0; n < 200; n += 1) {
    r.box.onPcm(piece(n));
    if (n % 10 === 9) socket.server({ type: "ack", samples: (n - 4) * 1600 }); // ~0.5 s behind
  }
  await r.recording.flushed();
  await settle();
  assert.equal(r.idb.writes, 0, "20 s connected: not one write");
  assert.deepEqual(await r.backend.all("sessions"), []);
  socket.server({ type: "ack", samples: 200 * 1600 });
  await r.audio.keepNow(); // the tab is hidden with nothing pending
  assert.equal(r.idb.writes, 0);
  r.box.onPcm(piece(1));
  await r.audio.keepNow(); // hidden with 0.1 s pending: that is kept
  assert.deepEqual(await rows(r.backend), [[0, 320000, 1600]]);
});

test("a backlog of more than a few seconds is kept, and deleted on its ack", async () => {
  const r = rig();
  const socket = await started(r);
  for (let n = 0; n < 50; n += 1) r.box.onPcm(piece(n)); // 5 s, nothing acknowledged
  await r.recording.flushed();
  await settle();
  assert.deepEqual(await rows(r.backend), [[0, 0, 80000]]);
  const kept = await r.backend.get("sessions", "iv-1");
  assert.equal(kept.product, "live_assist.interview");
  assert.ok(!("name" in kept) && !("candidate" in kept), "no names kept");
  socket.server({ type: "ack", samples: 80000 });
  await r.recording.flushed();
  await settle();
  assert.deepEqual(await rows(r.backend), []);
  assert.deepEqual(await r.backend.keys("chunks"), []);
});

test("an outage is written in 5 s chunks; «Terminar» offline ends here and loses nothing", async () => {
  const r = rig();
  const socket = await started(r);
  socket.close(); // the link drops for good
  for (let n = 0; n < 120; n += 1) r.box.onPcm(piece(n)); // 12 s
  await r.recording.flushed();
  await settle();
  assert.deepEqual((await rows(r.backend)).map((c) => c[2]), [16000 * CHUNK_SECONDS, 16000 * CHUNK_SECONDS]);
  assert.equal(finishLine(r.audio.link).key, "live.finish.offline_kept");
  assert.equal(finishLine(r.audio.link).button, "live.finish.now_kept");
  const done = await finishOffline(r);
  assert.deepEqual(done, { lostSeconds: 0, keptSeconds: 12 });
  assert.equal((await rows(r.backend)).reduce((n, c) => n + c[2], 0), 16000 * 12, "all 12 s kept");
  assert.equal((await r.backend.get("sessions", "iv-1")).ended_locally, true);
  const pending = await pendingRecordings(r.backend, Date.parse("2026-11-26T12:00:00Z"));
  assert.deepEqual(pending.map((p) => [p.id, p.minutes, p.chunks]), [["iv-1", 1, 3]]);
});

test("audio the full outbox let go stays here for the recovery; a clamp shifts it", async () => {
  const r = rig();
  const socket = await started(r);
  r.audio.outbox.maxSamples = 16000 * 6; // a small memory for the test
  socket.close();
  for (let n = 0; n < 100; n += 1) r.box.onPcm(piece(n)); // 10 s: 4 s let go in memory
  await r.recording.flushed();
  r.clock.advance(500);
  const back = r.made.at(-1);
  back.open();
  back.server({ type: "ready", received_samples: 0 });
  assert.equal(back.texts()[1].type, "gap", "the 4 s became silence there");
  back.server({ type: "ack", samples: 16000 * 10 });
  await r.recording.flushed();
  await settle();
  assert.deepEqual((await rows(r.backend)).map((c) => c[1]), [0], "the chunk that became silence is kept");
  await r.recording.shift(0, -1600);
  await settle();
  assert.equal((await r.backend.get("index", chunkKey("iv-1", 0))).offset, -1600);
});

test("a browser out of room says so (once), writes nothing half, and the audio counts as lost", async () => {
  const r = rig({ quota: 200 });
  const socket = await started(r);
  socket.close();
  for (let n = 0; n < 60; n += 1) r.box.onPcm(piece(n));
  await r.recording.flushed();
  await settle();
  assert.deepEqual(r.problems, ["quota"]);
  assert.deepEqual(await r.backend.all("index"), [], "the transaction wrote nothing");
  assert.equal(r.audio.link.keeping, false);
  assert.equal(finishLine(r.audio.link).button, "live.finish.now");
  const done = await finishOffline(r);
  assert.equal(done.keptSeconds, 0);
  assert.equal(done.lostSeconds, 6);
  assert.equal(storageProblem({ name: "Other" }), "storage");
});

async function keptSession(backend, n = 3, id = "iv-1") {
  await backend.put("sessions", { id, channels: 1, started_at: "2026-11-26T10:00:00.000Z",
    expires_at: "2026-11-28T10:00:00.000Z" });
  for (let seq = 0; seq < n; seq += 1) {
    const key = chunkKey(id, seq);
    await backend.write([["put", "chunks", { key, bytes: new Uint8Array(32000) }],
      ["put", "index", { key, session: id, seq, offset: seq * 16000, samples: 16000 }]]);
  }
}

test("a reattached page numbers its chunks after the earlier page's", async () => {
  const backend = new IdbBackend(new FakeIndexedDB());
  await keptSession(backend, 5);
  const again = await new LocalRecording({ backend, meta: sessionMeta({ id: "iv-1" }) }).open();
  assert.equal(again.seq, 5);
  assert.equal(again.meta.started_at, "2026-11-26T10:00:00.000Z", "the first page's start and expiry stay");
  assert.equal(await again.forgetIfEmpty(), false);
  const empty = await new LocalRecording({ backend, meta: sessionMeta({ id: "iv-2" }) }).open();
  await empty.endLocally();
  assert.equal(await empty.forgetIfEmpty(), true);
  assert.equal(await backend.get("sessions", "iv-2"), undefined);
});

function fakeApi({ have = [], fail = {}, state = "uploading", closeError = null } = {}) {
  const calls = [];
  return {
    calls,
    recoveryStatus: async () => {
      calls.push(["status"]);
      if (state === null) throw Object.assign(new Error("none"), { status: 404 });
      return { state, chunks: have };
    },
    putRecoveryChunk: async (id, seq, offset, bytes, channels) => {
      calls.push(["put", seq, offset, bytes.length, channels]);
      if (fail[seq]) {
        const error = fail[seq].shift();
        if (error) throw error;
      }
      return {};
    },
    closeRecovery: async (id, body) => {
      calls.push(["close", body]);
      if (closeError) throw closeError;
      return { session_id: id, state: "queued" };
    },
  };
}

test("the next visit sends what waits: resumable, retried, then deleted here", async () => {
  const backend = new IdbBackend(new FakeIndexedDB());
  await keptSession(backend);
  const api = fakeApi({ have: [1], fail: { 2: [Object.assign(new Error("net"), { status: null })] } });
  const progress = [];
  const waits = [];
  const view = await uploadRecording({ backend, api, id: "iv-1", onProgress: (f) => progress.push(f), wait: async (ms) => waits.push(ms) });
  assert.equal(view.state, "queued");
  assert.deepEqual(api.calls.filter((c) => c[0] === "put").map((c) => c[1]), [0, 2, 2], "chunk 1 was there; 2 retried");
  assert.deepEqual(waits, [1000]);
  assert.deepEqual(api.calls.at(-1), ["close", { count: 3, expires_at: "2026-11-28T10:00:00.000Z" }]);
  assert.deepEqual(progress.map((p) => Math.round(p * 3)), [1, 2, 3]);
  assert.deepEqual(await backend.all("index"), []);
  assert.deepEqual(await backend.keys("chunks"), []);
  assert.equal(await backend.get("sessions", "iv-1"), undefined);
});

test("a full server or a session still on is said, not retried, and the audio stays here", async () => {
  const backend = new IdbBackend(new FakeIndexedDB());
  await keptSession(backend, 2);
  const full = fakeApi({ state: null, fail: { 0: [Object.assign(new Error("full"), { status: 413 })] } });
  await assert.rejects(uploadRecording({ backend, api: full, id: "iv-1", wait: async () => {} }), (e) => e.status === 413);
  const on = fakeApi({ state: null, fail: { 0: [Object.assign(new Error("on"), { status: 409, code: "wrong_state" })] } });
  await assert.rejects(uploadRecording({ backend, api: on, id: "iv-1", wait: async () => {} }), (e) => e.code === "running");
  const closing = fakeApi({ state: null, closeError: Object.assign(new Error("on"), { status: 409, code: "wrong_state" }) });
  await assert.rejects(uploadRecording({ backend, api: closing, id: "iv-1" }), (e) => e.code === "running");
  assert.equal((await backend.all("index")).length, 2);
});

test("a closed upload deletes here only what the server holds; late chunks are said", async () => {
  const backend = new IdbBackend(new FakeIndexedDB());
  await keptSession(backend, 3);
  const closed = fakeApi({ state: "done", have: [0, 1] });
  await assert.rejects(uploadRecording({ backend, api: closed, id: "iv-1" }), (e) => e.code === "closed");
  assert.deepEqual((await backend.all("index")).map((r) => r.seq), [2], "the late one stays");
  const all = fakeApi({ state: "done", have: [2] });
  assert.equal((await uploadRecording({ backend, api: all, id: "iv-1" })).state, "done");
  assert.deepEqual(await backend.all("sessions"), []);
  await assert.rejects(uploadRecording({ backend, api: all, id: "iv-1" }), (e) => e.code === "not_found");
});

test("expired audio is swept; a session another tab is creating is not a stray", async () => {
  const backend = new IdbBackend(new FakeIndexedDB());
  await keptSession(backend, 2);
  await backend.write([["put", "chunks", { key: chunkKey("ghost", 0), bytes: new Uint8Array(2) }]]);
  assert.deepEqual(await sweepExpired(backend, Date.parse("2026-11-27T10:00:00Z")), { sessions: 0, chunks: 1 });
  // Another tab writes its session and chunks between the reads: kept.
  const keys = backend.keys.bind(backend);
  backend.keys = async (name) => {
    const found = await keys(name);
    if (name === "chunks") await keptSession(backend, 1, "iv-new");
    return found;
  };
  assert.deepEqual(await sweepExpired(backend, Date.parse("2026-11-27T10:00:00Z")), { sessions: 0, chunks: 0 });
  backend.keys = keys;
  assert.equal((await pendingRecordings(backend, Date.parse("2026-11-27T10:00:00Z"))).length, 2);
  assert.deepEqual(await sweepExpired(backend, Date.parse("2026-11-28T10:00:00Z")), { sessions: 2, chunks: 3 });
  await keptSession(backend, 1);
  await discardRecording(backend, "iv-1");
  assert.deepEqual(await backend.all("index"), []);
  assert.deepEqual(await sweepAtLoad(new IdbBackend(null)), { sessions: 0, chunks: 0 }, "no IndexedDB: nothing");
});

test("the sweep runs at load, when the page comes back and on a timer", async () => {
  const listeners = {};
  const timers = [];
  const target = {
    document: { visibilityState: "visible" },
    addEventListener: (kind, fn) => { listeners[kind] = fn; },
    removeEventListener: (kind) => { delete listeners[kind]; },
    setInterval: (fn, ms) => { timers.push([fn, ms]); return 7; },
    clearInterval: (id) => { timers.push(["cleared", id]); },
  };
  let sweeps = 0;
  const backend = { keys: async () => { sweeps += 1; return []; }, all: async () => [], write: async () => {}, delete: async () => {} };
  const stop = installSweeps({ backend, target });
  await settle();
  listeners.visibilitychange();
  target.document.visibilityState = "hidden";
  listeners.visibilitychange();
  timers[0][0]();
  await settle();
  assert.equal(sweeps, 2 * 3, "load, return and timer (two key reads each)");
  assert.equal(timers[0][1], 10 * 60 * 1000);
  stop();
  assert.deepEqual(timers.at(-1), ["cleared", 7]);
  assert.equal(listeners.visibilitychange, undefined);
});

test("the interviewer's lines: pending, analysing, completed (and how), failed", () => {
  setLanguage("es");
  const line = (v) => { const l = recoveryLine(v); return l && t(l.key, l.params); };
  assert.equal(line({ state: "queued" }), "Pendiente de analizar");
  assert.equal(line({ state: "uploading" }), "Pendiente de analizar");
  assert.equal(line({ state: "transcribing", progress: 0.5 }), "Analizando (40 %)");
  assert.equal(line({ state: "rejudging" }), "Analizando (90 %)");
  assert.equal(line({ state: "done", seconds: 180, added: 4 }), "Completado (3 min recuperados)");
  assert.equal(line({ state: "done", seconds: 1500, added: 9 }), "Completado (25 min recuperados)");
  assert.equal(line({ state: "done", seconds: 120, added: 2, unidentified: true }),
    "Completado (2 min recuperados, sin identificar quién habla)");
  assert.equal(line({ state: "done", seconds: 0, added: 0 }), "Completado: el audio enviado no tenía conversación que añadir");
  assert.equal(line({ state: "failed" }), "No se pudo recuperar");
  assert.equal(recoveryLine(null), null);
  assert.ok(recoveryMoving({ state: "merging" }) && !recoveryMoving({ state: "done" }));
  setLanguage("en");
  assert.equal(line({ state: "done", seconds: 30, added: 1 }), "Completed (0.5 min recovered)");
  setLanguage("es");
});

test("every recovery text exists in both languages and keeps the dictionary's words", () => {
  const keys = Object.keys(es).filter((k) => k.startsWith("live.recovery.") || k.startsWith("live.store.")
    || ["live.finish.offline_kept", "live.finish.now_kept", "live.finish.kept"].includes(k));
  assert.ok(keys.length >= 18);
  for (const key of keys) {
    assert.ok(en[key], key);
    assert.doesNotMatch(es[key], /equipo|ordenador|\bPC\b/, key);
    assert.doesNotMatch(en[key], /computer|workstation|machine/, key);
  }
  assert.equal(es["live.recovery.banner"], "Hay {min} min de la entrevista del {date} sin enviar.");
});

test("the API sends a chunk with its offset and closes, deletes and lists", async () => {
  const seen = [];
  const api = new Api(() => "code", async (url, init) => {
    seen.push([init.method, url, init.headers["Content-Type"] ?? null]);
    return { ok: true, status: init.method === "DELETE" ? 204 : 200, json: async () => ({}) };
  });
  await api.putRecoveryChunk("iv-1", 3, 48000, new Uint8Array(4), 1);
  await api.closeRecovery("iv-1", { count: 4 });
  await api.recoveryStatus("iv-1");
  await api.discardRecovery("iv-1");
  await api.liveRecoveries();
  assert.deepEqual(seen, [
    ["PUT", "/api/live/sessions/iv-1/recovery/chunks/3?offset=48000&channels=1", "application/octet-stream"],
    ["POST", "/api/live/sessions/iv-1/recovery/close", "application/json"],
    ["GET", "/api/live/sessions/iv-1/recovery", null],
    ["DELETE", "/api/live/sessions/iv-1/recovery", null],
    ["GET", "/api/live/recoveries", null],
  ]);
});

test("the new notices change only the sentence about the audio; the old ids stay", () => {
  const pairs = [["interview_ai_notice@2", "interview_ai_notice@3"], ["support_ai_notice@1", "support_ai_notice@2"]];
  for (const [old, now] of pairs) {
    assert.ok(es[`live.banner.${old}`] && en[`live.banner.${old}`], `${old} stays`);
    assert.equal(es[`live.banner.${now}`],
      es[`live.banner.${old}`].replace("No se guarda el audio.", "El audio se borra en cuanto se transcribe."));
    assert.equal(en[`live.banner.${now}`],
      en[`live.banner.${old}`].replace("No audio is kept.", "The audio is deleted as soon as it is transcribed."));
  }
  for (const dict of [es, en]) {
    for (const [key, text] of Object.entries(dict)) {
      if (!key.startsWith("live.banner.") || /@1$|interview_ai_notice@2$/.test(key)) continue;
      assert.doesNotMatch(text, /No se guarda el audio|No audio is kept/, key);
    }
  }
  assert.match(es["voicebot.notice"], /El audio se borra en cuanto se transcribe;/);
  assert.match(en["voicebot.notice"], /The audio is deleted as soon as it is transcribed;/);
});
