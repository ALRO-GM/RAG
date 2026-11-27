// Large uploads of the «Conocimiento» page (#282): the server's limits by
// kind, a file sent in pieces with progress, a calm retry after a cut that
// goes on from what the server already holds, cancel, and the messages in
// both languages. The backend is an in-memory fake with a manual clock.
import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_PIECE_BYTES, KnowledgeModel, MAX_TRIES, PieceUpload, RETRY_MS, UploadCancelled, limitKind, limitText, percentOf,
} from "../../transcriber/adapters/inbound/web/js/lib/knowledge.js";
import { Api, ApiFailure } from "../../transcriber/adapters/inbound/web/js/api.js";
import { setLanguage, t } from "../../transcriber/adapters/inbound/web/js/i18n.js";

const LIMITS = { pdf: 500, video: 1000, audio: 200, document: 100, piece_bytes: 64, max_media_seconds: 7200 };
const CUT = () => new ApiFailure({ status: null, kind: "unavailable", code: "network" });

/** A File stand-in: a Blob of `size` bytes with a name. */
function blobFile(name, size) {
  const bytes = new Uint8Array(size).map((_, i) => i % 251);
  return Object.assign(new Blob([bytes]), { name });
}

/** A timer the test fires by hand (records the waits asked for). */
function manualClock() {
  const pending = new Map();
  const waits = [];
  let id = 0;
  return {
    waits,
    set(fn, ms) { id += 1; pending.set(id, fn); waits.push(ms); return id; },
    clear(handle) { pending.delete(handle); },
    count() { return pending.size; },
    async fire() {
      const all = [...pending.values()];
      pending.clear();
      for (const fn of all) await fn();
    },
  };
}

/** Let the pending promises run. */
const settle = () => new Promise((resolve) => setImmediate(resolve));

/**
 * The /knowledge upload routes in memory. `script` lists what the next
 * piece calls do: "cut" (the network drops, nothing arrives), "lost" (it
 * arrives but the answer is lost) or "ok".
 */
class FakeServer {
  constructor() {
    this.uploads = new Map();
    this.script = [];
    this.calls = [];
    this.next = 0;
    this.docs = [];
  }

  async knowledgeLimits() { this.calls.push("limits"); return { ...LIMITS }; }

  async knowledgeDomains() { return [{ id: "hr", title: "RR. HH.", language: "es" }]; }

  async knowledgeDocuments() { return structuredClone(this.docs); }

  async uploadKnowledgeDocument(domain, file) {
    this.calls.push(`single ${file.name}`);
    const doc = { id: `d-${file.name}`, name: file.name, size: file.size, status: "queued" };
    this.docs.push(doc);
    return doc;
  }

  async startKnowledgeUpload(domain, { name, size, metadata }) {
    this.calls.push(`start ${name}`);
    if (size > (this.limits ?? LIMITS)[limitKind(name)]) {
      throw new ApiFailure({ status: 413, kind: "client", code: "payload_too_large", message: "too large" });
    }
    this.next += 1;
    const id = String(this.next).padStart(32, "0");
    this.uploads.set(id, { name, size, metadata, bytes: [], received: 0 });
    return { id, received: 0, size };
  }

  async knowledgeUploadStatus(domain, id) {
    this.calls.push("status");
    const upload = this.uploads.get(id);
    if (!upload) throw new ApiFailure({ status: 404, kind: "client", code: "not_found" });
    return { id, received: upload.received, size: upload.size };
  }

  async putKnowledgePiece(domain, id, offset, piece, { signal } = {}) {
    this.calls.push(`put ${offset}`);
    const action = this.script.shift() ?? "ok";
    if (action === "cut") throw CUT();
    if (action === "hang") {
      return new Promise((_, reject) => signal.addEventListener("abort", () => reject(new ApiFailure({ status: null, kind: "aborted", code: "aborted" }))));
    }
    const upload = this.uploads.get(id);
    if (offset > upload.received) throw new ApiFailure({ status: 409, kind: "client", code: "wrong_state", message: `Expected ${upload.received}` });
    const data = new Uint8Array(await piece.arrayBuffer());
    for (let i = 0; i < data.length; i += 1) upload.bytes[offset + i] = data[i];
    upload.received = Math.max(upload.received, offset + data.length);
    if (action === "lost") throw CUT();
    return { id, received: upload.received, size: upload.size };
  }

  async completeKnowledgeUpload(domain, id) {
    this.calls.push("complete");
    const upload = this.uploads.get(id);
    if (upload.received < upload.size) throw new ApiFailure({ status: 409, kind: "client", code: "wrong_state" });
    const doc = { id: `d-${upload.name}`, name: upload.name, size: upload.size, status: "queued" };
    this.docs.push(doc);
    return doc;
  }

  async cancelKnowledgeUpload(domain, id) {
    this.calls.push("cancel");
    this.uploads.delete(id);
    return null;
  }
}

// ---- pure helpers ------------------------------------------------------------

test("the limit of a file follows its kind; limits read as people write them", () => {
  assert.deepEqual(["a.PDF", "b.mp4", "c.webm", "d.mov", "e.3gp", "f.mp3", "g.m4a", "h.docx", "i.png"].map(limitKind),
    ["pdf", "video", "video", "video", "video", "audio", "audio", "document", "document"]);
  assert.equal(limitText(500 * 1024 * 1024, "es"), "500 MB");
  assert.equal(limitText(1024 * 1024 * 1024, "en"), "1 GB");
  assert.equal(limitText(1.5 * 1024 * 1024 * 1024, "es"), "1,5 GB");
  assert.equal(limitText(0), "");
  assert.equal(percentOf(50, 200), 25);
  assert.equal(percentOf(3, 0), 0);
  assert.equal(percentOf(250, 200), 100);
});

// ---- the API client ----------------------------------------------------------

test("the piece calls use the documented routes; a piece is raw bytes and can be cancelled", async () => {
  const calls = [];
  const api = new Api(() => "code-1", async (url, init) => {
    calls.push({ url, init });
    if (init.signal?.aborted) throw new DOMException("aborted", "AbortError");
    return init.method === "DELETE" ? new Response(null, { status: 204 }) : new Response("{}", { status: 200 });
  });
  const piece = new Blob(["abc"]);
  await api.knowledgeLimits();
  await api.startKnowledgeUpload("hr", { name: "a.pdf", size: 3, metadata: {} });
  await api.knowledgeUploadStatus("hr", "u1");
  await api.putKnowledgePiece("hr", "u1", 8, piece);
  await api.completeKnowledgeUpload("hr", "u1");
  await api.cancelKnowledgeUpload("hr", "u1");
  assert.deepEqual(calls.map((c) => `${c.init.method} ${c.url}`), [
    "GET /api/knowledge/limits",
    "POST /api/knowledge/domains/hr/uploads",
    "GET /api/knowledge/domains/hr/uploads/u1",
    "PUT /api/knowledge/domains/hr/uploads/u1/pieces/8",
    "POST /api/knowledge/domains/hr/uploads/u1/complete",
    "DELETE /api/knowledge/domains/hr/uploads/u1",
  ]);
  assert.equal(calls[3].init.body, piece);
  assert.equal(calls[3].init.headers["Content-Type"], "application/octet-stream");
  assert.equal("signal" in calls[0].init, false, "no signal unless asked");
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(api.putKnowledgePiece("hr", "u1", 0, piece, { signal: controller.signal }), (error) => error.kind === "aborted");
});

// ---- PieceUpload ---------------------------------------------------------------

test("a file goes in pieces with its progress, then is completed", async () => {
  const server = new FakeServer();
  const seen = [];
  const upload = new PieceUpload(server, {
    domain: "hr", file: blobFile("big.pdf", 150), metadata: { tags: [] }, pieceBytes: 64, timer: manualClock(),
    onProgress: (p) => seen.push(`${p.sent}/${p.total}${p.finishing ? " finishing" : ""}`),
  });
  const doc = await upload.run();
  assert.equal(doc.id, "d-big.pdf");
  assert.deepEqual(server.calls, ["start big.pdf", "put 0", "put 64", "put 128", "complete"]);
  assert.deepEqual(seen, ["0/150", "64/150", "128/150", "150/150", "150/150 finishing"]);
  const [stored] = server.uploads.values();
  assert.deepEqual(stored.bytes, [...new Uint8Array(150).map((_, i) => i % 251)]);
});

test("after a cut it waits calmly, asks what arrived and goes on from there", async () => {
  const server = new FakeServer();
  server.script = ["ok", "lost", "cut", "ok"];
  const clock = manualClock();
  const states = [];
  const upload = new PieceUpload(server, {
    domain: "hr", file: blobFile("big.pdf", 150), metadata: {}, pieceBytes: 64, timer: clock,
    onProgress: (p) => states.push(p.retrying ? "retrying" : `${p.sent}`),
  });
  const done = upload.run();
  await settle();
  assert.ok(states.includes("retrying"), "the page says it is retrying");
  assert.deepEqual(clock.waits, [RETRY_MS[0]]);
  await clock.fire(); // back: the status says 128 arrived (the answer was lost)
  await settle();
  assert.deepEqual(clock.waits, [RETRY_MS[0], RETRY_MS[1]], "the next cut waits longer");
  await clock.fire();
  const doc = await done;
  assert.equal(doc.name, "big.pdf");
  assert.deepEqual(server.calls, ["start big.pdf", "put 0", "put 64", "status", "put 128", "status", "put 128", "complete"]);
  assert.equal(server.calls.filter((c) => c === "put 0").length, 1, "never starts over");
  assert.equal(states.at(-1), "150");
});

test("a 409 realigns with the server at once; a refusal ends the upload with its reason", async () => {
  const server = new FakeServer();
  const upload = new PieceUpload(server, { domain: "hr", file: blobFile("a.pdf", 100), metadata: {}, pieceBytes: 64, timer: manualClock() });
  await upload.run();
  // a second tab resumed an upload the server holds 100 bytes of
  const again = new PieceUpload(server, { domain: "hr", file: blobFile("a.pdf", 100), metadata: {}, pieceBytes: 64, timer: manualClock(), uploadId: upload.uploadId });
  server.calls = [];
  await again.run();
  assert.deepEqual(server.calls, ["status", "complete"], "a resumed upload asks first");
  const refused = new PieceUpload(server, { domain: "hr", file: blobFile("huge.pdf", 501), metadata: {}, pieceBytes: 64, timer: manualClock() });
  await assert.rejects(refused.run(), (error) => error.status === 413);
  const full = {
    ...server,
    startKnowledgeUpload: async () => { throw new ApiFailure({ status: 507, kind: "server", code: "insufficient_storage", message: "no room" }); },
  };
  const clock = manualClock();
  await assert.rejects(new PieceUpload(full, { domain: "hr", file: blobFile("b.pdf", 100), metadata: {}, pieceBytes: 64, timer: clock }).run(),
    (error) => error.status === 507);
  assert.deepEqual(clock.waits, [], "a full disk is not retried");
});

test("cuts that never end fail after MAX_TRIES; cancel stops at once and deletes what arrived", async () => {
  const server = new FakeServer();
  server.script = Array(MAX_TRIES).fill("cut");
  const clock = manualClock();
  const upload = new PieceUpload(server, { domain: "hr", file: blobFile("a.pdf", 100), metadata: {}, pieceBytes: 64, timer: clock });
  const outcome = upload.run().then(() => null, (error) => error);
  for (let i = 0; i < MAX_TRIES; i += 1) {
    await settle();
    await clock.fire();
  }
  assert.equal((await outcome)?.kind, "unavailable");
  assert.equal(clock.waits.at(-1), RETRY_MS.at(-1));

  server.script = ["ok", "hang"];
  const hanging = new PieceUpload(server, { domain: "hr", file: blobFile("b.pdf", 150), metadata: {}, pieceBytes: 64, timer: manualClock() });
  const running = hanging.run();
  await settle();
  await hanging.cancel();
  await assert.rejects(running, (error) => error instanceof UploadCancelled);
  assert.equal(server.calls.at(-1), "cancel");
  assert.equal(server.uploads.has(hanging.uploadId), false);

  server.script = ["cut"];
  const clock2 = manualClock();
  const waiting = new PieceUpload(server, { domain: "hr", file: blobFile("c.pdf", 100), metadata: {}, pieceBytes: 64, timer: clock2 });
  const pending = waiting.run();
  await settle();
  assert.equal(clock2.count(), 1, "waiting out a cut");
  await waiting.cancel();
  await assert.rejects(pending, (error) => error.kind === "aborted");
  assert.equal(clock2.count(), 0, "the wait is dropped");
  await new PieceUpload(server, { domain: "hr", file: blobFile("d.pdf", 1), metadata: {}, pieceBytes: 64, timer: clock2 }).cancel();
});

// ---- the page model -----------------------------------------------------------

test("the page reads the server's limits; small files go in one request, large ones in pieces", async () => {
  const server = new FakeServer();
  const clock = manualClock();
  const progress = [];
  const model = new KnowledgeModel(server, {
    timer: clock,
    onChange: (s) => { for (const u of s.uploads) if (u.state === "uploading") progress.push(u.sent); },
  });
  await model.load("hr");
  assert.deepEqual(model.state.limits, LIMITS);
  await model.addFiles([blobFile("small.md", 10), blobFile("big.pdf", 150), blobFile("huge.mp4", 1001), blobFile("big.docx", 101)]);
  assert.deepEqual(server.calls.filter((c) => c !== "limits" && !c.startsWith("put")), ["single small.md", "start big.pdf", "complete"]);
  assert.ok(progress.includes(64) && progress.includes(128), "progress while it goes");
  assert.deepEqual(model.state.uploads.map((u) => [u.name, u.state, u.reason, u.kind, u.limit]), [
    ["huge.mp4", "rejected", "size", "video", 1000],
    ["big.docx", "rejected", "size", "document", 100],
  ]);
  assert.deepEqual(model.state.documents.map((d) => d.name).sort(), ["big.pdf", "small.md"]);
});

test("a failed large upload keeps its id: retrying goes on from what arrived", async () => {
  const server = new FakeServer();
  const clock = manualClock();
  const model = new KnowledgeModel(server, { timer: clock });
  await model.load("hr");
  server.script = ["ok", ...Array(MAX_TRIES).fill("cut")];
  const adding = model.addFiles([blobFile("big.pdf", 150)]);
  for (let i = 0; i < MAX_TRIES + 2; i += 1) {
    await settle();
    await clock.fire();
  }
  await adding;
  const [item] = model.state.uploads;
  assert.equal(item.state, "failed");
  assert.ok(item.uploadId);
  server.calls = [];
  await model.retryUpload(item.key);
  assert.deepEqual(server.calls.filter((c) => c !== "limits"), ["status", "put 64", "put 128", "complete"]);
  assert.equal(model.state.uploads.length, 0);
});

test("cancel drops a waiting file, stops one in pieces and deletes what the server holds", async () => {
  const server = new FakeServer();
  const model = new KnowledgeModel(server, { timer: manualClock() });
  await model.load("hr");
  server.script = ["ok", "hang"];
  const adding = model.addFiles([blobFile("big.pdf", 150), blobFile("next.pdf", 150)]);
  await settle();
  const [first, second] = model.state.uploads;
  assert.equal(first.state, "uploading");
  assert.equal(second.state, "waiting");
  await model.cancelUpload(second.key);
  await model.cancelUpload(first.key);
  await adding;
  assert.equal(model.state.uploads.length, 0);
  assert.ok(server.calls.includes("cancel"));
  assert.equal(server.calls.filter((c) => c.startsWith("start")).length, 1, "the cancelled waiting file is never sent");
  await model.cancelUpload(999);
});

test("without the server's limits a large file still goes in pieces (the server checks the size)", async () => {
  const server = new FakeServer();
  let asked = 0;
  server.knowledgeLimits = async () => { asked += 1; throw CUT(); };
  server.limits = { ...LIMITS, pdf: 10 * DEFAULT_PIECE_BYTES };
  const model = new KnowledgeModel(server, { timer: manualClock() });
  await model.load("hr");
  assert.equal(model.state.limits, null);
  const big = blobFile("big.pdf", DEFAULT_PIECE_BYTES + 10);
  await model.addFiles([blobFile("small.md", 10), big]);
  assert.equal(asked, 3, "the limits are asked for again before each file");
  assert.deepEqual(server.calls.filter((c) => !c.startsWith("put")), ["single small.md", "start big.pdf", "complete"]);
  assert.deepEqual(server.calls.filter((c) => c.startsWith("put")), ["put 0", `put ${DEFAULT_PIECE_BYTES}`]);
  assert.equal(model.state.uploads.length, 0);
});

test("an upload the server no longer has (swept, refused) starts again on retry", async () => {
  const server = new FakeServer();
  const clock = manualClock();
  const model = new KnowledgeModel(server, { timer: clock });
  await model.load("hr");
  server.script = ["ok", ...Array(MAX_TRIES).fill("cut")];
  const adding = model.addFiles([blobFile("big.pdf", 150)]);
  for (let i = 0; i < MAX_TRIES + 2; i += 1) {
    await settle();
    await clock.fire();
  }
  await adding;
  const [item] = model.state.uploads;
  const gone = item.uploadId;
  server.uploads.delete(gone); // swept after 24 h
  server.calls = [];
  const seen = [];
  model.onChange = (s) => { for (const u of s.uploads) if (u.restarted) seen.push("restarted"); };
  await model.retryUpload(item.key);
  assert.deepEqual(server.calls.filter((c) => c !== "limits"), ["status", "start big.pdf", "put 0", "put 64", "put 128", "complete"]);
  assert.ok(seen.length, "the row says it starts again");
  assert.equal(model.state.uploads.length, 0);

  // a 404 in the middle (the upload deleted meanwhile) fails, and the retry starts over
  server.putKnowledgePiece = async () => { throw new ApiFailure({ status: 404, kind: "client", code: "not_found" }); };
  await model.addFiles([blobFile("other.pdf", 150)]);
  const [failed] = model.state.uploads;
  assert.equal(failed.state, "failed");
  assert.equal(failed.uploadId, null, "nothing left to resume");
});

// ---- the messages, in both languages -----------------------------------------------

test("the messages say the real limit of the kind, the progress and the retry calmly", () => {
  try {
    setLanguage("es");
    assert.equal(t("kn.reject.size_pdf", { limit: limitText(500 * 1024 ** 2, "es"), size: limitText(612 * 1024 ** 2, "es") }),
      "Los PDF pueden ocupar hasta 500 MB; este ocupa 612 MB.");
    assert.equal(t("kn.reject.size_video", { limit: "1 GB", size: "1,2 GB" }), "Los vídeos pueden ocupar hasta 1 GB; este ocupa 1,2 GB.");
    assert.equal(t("kn.upload.progress", { sent: "64 MB", total: "1 GB", percent: 6 }), "64 MB de 1 GB (6 %)");
    assert.match(t("kn.upload.retrying"), /La subida seguirá desde donde se quedó/);
    assert.equal(t("kn.pages_done", { done: 40, total: 1200 }), "Leídas 40 de 1200 páginas");
    assert.match(t("kn.drop_limits_media", { pdf: "500 MB", video: "1 GB", audio: "200 MB", document: "100 MB", minutes: 120 }),
      /^Hasta 500 MB por PDF, 1 GB por vídeo, 200 MB por audio y 100 MB por cualquier otro documento; las grabaciones, de 120 minutos/);
    setLanguage("en");
    assert.equal(t("kn.reject.size_pdf", { limit: "500 MB", size: "612 MB" }), "PDF files may be up to 500 MB; this one is 612 MB.");
    assert.equal(t("kn.reject.size_audio", { limit: "200 MB", size: "300 MB" }), "Audio files may be up to 200 MB; this one is 300 MB.");
    assert.equal(t("kn.upload.progress", { sent: "64 MB", total: "1 GB", percent: 6 }), "64 MB of 1 GB (6%)");
    assert.match(t("kn.upload.retrying"), /carry on from where it stopped/);
    assert.match(t("kn.upload.restarted"), /from the start/);
    assert.equal(t("kn.upload.finishing"), "Checking the file…");
    assert.equal(t("kn.err_space"), "The server's disk has no room for this file.");
  } finally {
    setLanguage("es");
  }
});
