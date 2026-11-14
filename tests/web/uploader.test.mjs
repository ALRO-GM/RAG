import test from "node:test";
import assert from "node:assert/strict";
import { ChunkQueue } from "../../transcriber/adapters/inbound/web/js/lib/chunks.js";
import { ChunkUploader } from "../../transcriber/adapters/inbound/web/js/lib/uploader.js";
import { Api, ApiFailure } from "../../transcriber/adapters/inbound/web/js/api.js";

const bytes = (n, v = 1) => new Uint8Array(n).fill(v);
const failure = (kind, extra = {}) => new ApiFailure({ status: 0, kind, ...extra });

/** A clock and timers the test moves by hand. */
function fakeTime() {
  const t = { now: 0, timers: [] };
  t.setTimer = (fn, ms) => { const h = { fn, at: t.now + ms }; t.timers.push(h); return h; };
  t.clearTimer = (h) => { t.timers = t.timers.filter((x) => x !== h); };
  t.advance = async (ms) => {
    t.now += ms;
    const due = t.timers.filter((h) => h.at <= t.now);
    t.timers = t.timers.filter((h) => h.at > t.now);
    for (const h of due) h.fn();
    await new Promise((r) => setImmediate(r));
  };
  return t;
}

function setup(putChunk, getSession = async () => ({ chunks: 0 }), queueOptions = {}) {
  const time = fakeTime();
  const api = { putChunk, getSession };
  const queue = new ChunkQueue({ chunkBytes: 4, ...queueOptions });
  const states = [];
  const up = new ChunkUploader(api, "s1", queue, {
    now: () => time.now, setTimer: time.setTimer, clearTimer: time.clearTimer,
    onState: (s) => states.push(s),
  });
  return { time, queue, up, states };
}

const settle = () => new Promise((r) => setImmediate(r));

test("chunks go in order, one at a time, and whenDone resolves", async () => {
  const sent = [];
  let busy = 0;
  const { up } = setup(async (id, n, data) => {
    busy += 1;
    assert.equal(busy, 1, "never two at once");
    await settle();
    sent.push([id, n, data.length]);
    busy -= 1;
    return { stored: true };
  });
  up.push(bytes(10));
  up.finish();
  assert.equal(await up.whenDone(), null);
  assert.deepEqual(sent, [["s1", 0, 4], ["s1", 1, 4], ["s1", 2, 2]]);
});

test("a network failure retries the same chunk after the back-off, keeping later audio", async () => {
  const attempts = [];
  let fail = 2;
  const { up, time, queue } = setup(async (id, n) => {
    attempts.push(n);
    if (fail-- > 0) throw failure("unavailable");
    return { stored: true };
  });
  up.push(bytes(8));
  await settle();
  assert.deepEqual(attempts, [0]);
  assert.equal(up.state().lastError, "unavailable");
  assert.equal(up.state().pendingBytes, 8, "nothing dropped");
  await time.advance(999);
  assert.deepEqual(attempts, [0], "still backing off");
  await time.advance(1);
  assert.deepEqual(attempts, [0, 0]);
  await time.advance(2000);
  assert.deepEqual(attempts, [0, 0, 0, 1]);
  assert.ok(queue.drained);
  assert.equal(up.state().lastError, null);
});

test("429 waits at least Retry-After", async () => {
  let first = true;
  const { up, time } = setup(async () => {
    if (first) { first = false; throw failure("rate", { waitSeconds: 20 }); }
    return {};
  });
  up.push(bytes(4));
  await settle();
  await time.advance(19000);
  assert.equal(up.state().drained, false);
  await time.advance(1000);
  assert.equal(up.state().drained, true);
});

test("retryNow skips the wait", async () => {
  let first = true;
  const { up } = setup(async () => {
    if (first) { first = false; throw failure("unavailable"); }
    return {};
  });
  up.push(bytes(4));
  await settle();
  assert.equal(up.state().drained, false);
  up.retryNow();
  await settle();
  assert.equal(up.state().drained, true);
});

test("a refused chunk resyncs with the server once, then goes on", async () => {
  const calls = [];
  const { up } = setup(async (id, n) => {
    calls.push(n);
    if (n === 0 && calls.length === 1) throw failure("client", { code: "wrong_state" });
    return {};
  }, async () => ({ chunks: 1 }));
  up.push(bytes(8));
  assert.equal(await up.whenDone(), null);
  assert.deepEqual(calls, [0, 1], "the server already had chunk 0");
});

test("a chunk refused again after the resync is final", async () => {
  const { up } = setup(async () => { throw failure("client", { code: "invalid_input" }); },
    async () => ({ chunks: 0 }));
  up.push(bytes(4));
  assert.equal(await up.whenDone(), "rejected");
});

test("a server that lost chunks makes the recording unrecoverable, not silently wrong", async () => {
  let n = 0;
  const { up } = setup(async (id, number) => {
    n += 1;
    if (number === 1) throw failure("client");
    return {};
  }, async () => ({ chunks: 0 }));
  up.push(bytes(8));
  assert.equal(await up.whenDone(), "lost");
  assert.equal(n, 2);
});

test("a wrong access code stops the upload", async () => {
  const { up } = setup(async () => { throw failure("unauthorized"); });
  up.push(bytes(4));
  assert.equal(await up.whenDone(), "unauthorized");
});

test("state reports the backlog and warns when it is large", async () => {
  const { up, states } = setup(async () => { throw failure("unavailable"); },
    async () => ({ chunks: 0 }), { warnBytes: 6 });
  up.push(bytes(8));
  await settle();
  const last = states.at(-1);
  assert.equal(last.large, true);
  assert.equal(last.pendingChunks, 2);
  assert.equal(last.pendingBytes, 8);
});

test("a failed resync read counts as a network failure and retries later", async () => {
  let first = true;
  const { up, time } = setup(async () => {
    if (first) { first = false; throw failure("client"); }
    return {};
  }, async () => { throw failure("unavailable"); });
  up.push(bytes(4));
  await settle();
  assert.equal(up.state().lastError, "unavailable");
  await time.advance(1000);
  assert.equal(up.state().drained, true);
});

// ---- Api -----------------------------------------------------------

function fakeFetch(answer) {
  const calls = [];
  const fn = async (url, init) => { calls.push([url, init]); return answer(url, init); };
  fn.calls = calls;
  return fn;
}
const res = (status, body, headers = {}) => ({
  ok: status >= 200 && status < 300, status,
  headers: { get: (k) => headers[k] ?? null },
  json: async () => { if (body === undefined) throw new Error("no json"); return body; },
});

test("Api sends the code in the header, never in the URL", async () => {
  const f = fakeFetch(() => res(200, { sessions: [] }));
  const api = new Api(() => "secret-code-123", f);
  await api.listSessions();
  await api.getSession("a b");
  await api.enrol("ana", { name: "Ana", language: "es", textVersion: "es-1", pcm: bytes(4) });
  for (const [url, init] of f.calls) {
    assert.ok(!url.includes("secret"), url);
    assert.equal(init.headers["X-Access-Code"], "secret-code-123");
  }
  assert.equal(f.calls[1][0], "/api/sessions/a%20b");
  assert.equal(f.calls[2][0], "/api/speakers/ana/enrolment?name=Ana&language=es&consent=true&text_version=es-1");
  assert.equal(f.calls[2][1].headers["Content-Type"], "application/octet-stream");
});

test("Api health needs no code; other calls without one are 401 and send nothing", async () => {
  const f = fakeFetch(() => res(200, { status: "ok" }));
  const api = new Api(() => null, f);
  assert.deepEqual(await api.health(), { status: "ok" });
  assert.equal(f.calls[0][1].headers["X-Access-Code"], undefined);
  await assert.rejects(api.listSessions(), (e) => e.kind === "unauthorized");
  assert.equal(f.calls.length, 1);
});

test("Api maps failures", async () => {
  const code = () => "c";
  const kinds = async (answer) => {
    const api = new Api(code, fakeFetch(answer));
    try { await api.listSessions(); } catch (e) { return e; }
    return null;
  };
  assert.equal((await kinds(() => res(502, undefined))).kind, "unavailable");
  assert.equal((await kinds(() => res(503, undefined))).kind, "unavailable");
  assert.equal((await kinds(() => { throw new TypeError("offline"); })).kind, "unavailable");
  assert.equal((await kinds(() => res(401, { code: "unauthorized", message: "x" }))).kind, "unauthorized");
  const limited = await kinds(() => res(429, { code: "too_many_attempts" }, { "Retry-After": "42" }));
  assert.deepEqual([limited.kind, limited.waitSeconds], ["rate", 42]);
  const rejected = await kinds(() => res(422, { code: "sample_rejected", message: "m", reason: "too_short" }));
  assert.deepEqual([rejected.kind, rejected.code, rejected.reason], ["client", "sample_rejected", "too_short"]);
});

test("Api: 204 gives null, JSON bodies are sent with their type, bad JSON is a failure", async () => {
  const f = fakeFetch(() => res(204));
  const api = new Api(() => "c", f);
  assert.equal(await api.audioCookie(), null);
  await api.createSession("s1", "es", "Reparto");
  assert.equal(f.calls[1][1].body, JSON.stringify({ id: "s1", language: "es", title: "Reparto" }));
  assert.equal(f.calls[1][1].headers["Content-Type"], "application/json");
  await api.queueTranscription("s1", { language: "es", expected_speakers: ["ana"] });
  await api.renameSpeaker("s1", "S1", { name: "Luis" });
  await api.reassignUtterance("s1", "t1.s0", "ana");
  await api.deleteSession("s1");
  await api.closeSession("s1");
  await api.putChunk("s1", 3, bytes(2));
  await api.uploadRecording("s1", { name: "a.m4a" }, "es", "T");
  await api.enrolmentText("es", "Ana");
  await api.speakers();
  await api.deleteSpeaker("ana");
  assert.deepEqual(f.calls.map((c) => `${c[1].method} ${c[0]}`).slice(2), [
    "POST /api/sessions/s1/transcription",
    "PATCH /api/sessions/s1/speakers/S1",
    "PATCH /api/sessions/s1/utterances/t1.s0",
    "DELETE /api/sessions/s1",
    "POST /api/sessions/s1/close",
    "PUT /api/sessions/s1/chunks/3",
    "POST /api/sessions/s1/upload?filename=a.m4a&language=es&title=T",
    "GET /api/enrolment/text?language=es&name=Ana",
    "GET /api/speakers",
    "DELETE /api/speakers/ana",
  ]);
  const broken = new Api(() => "c", fakeFetch(() => res(200, undefined)));
  await assert.rejects(broken.listSessions(), (e) => e.kind === "unavailable");
  assert.equal(f.calls[0][0], "/api/access/cookie");
});
