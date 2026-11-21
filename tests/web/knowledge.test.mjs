// The «Conocimiento» page without a browser: what files are accepted, what
// is sent (metadata, knowledge.query.v1, multipart), and the page's model
// driven against an in-memory fake of the /knowledge API with a manual
// clock (documents go queued -> processing -> indexed while it polls).
import test from "node:test";
import assert from "node:assert/strict";
import {
  ACCEPT, CATEGORIES, KnowledgeModel, POLL_MS, checkFile, cleanMetadata, countByStatus, extensionOf, formatBytes,
  isDomainId, listOf, needsPolling, parseTags, queryBody, requestId, serviceMissing, slugify, sortDocuments,
} from "../../transcriber/adapters/inbound/web/js/lib/knowledge.js";
import { Api, ApiFailure } from "../../transcriber/adapters/inbound/web/js/api.js";

// ---- pure helpers ----------------------------------------------------------

test("the eight document formats and four image formats are accepted, others and empty or huge files are not", () => {
  for (const name of ["a.pdf", "b.DOCX", "c.odt", "d.txt", "e.md", "f.html", "g.csv", "h.xlsx", "i.PNG", "j.jpg", "k.jpeg", "l.tif", "m.TIFF", "n.webp"]) {
    assert.equal(checkFile({ name, size: 10 }), null, name);
  }
  assert.equal(checkFile({ name: "x.exe", size: 10 }), "type");
  assert.equal(checkFile({ name: "noext", size: 10 }), "type");
  assert.equal(checkFile({ name: "a.pdf", size: 0 }), "empty");
  assert.equal(checkFile({ name: "a.pdf", size: 11 }, 10), "size");
  assert.equal(extensionOf("Informe.Final.PDF"), "pdf");
  for (const ext of [".pdf", ".docx", ".odt", ".txt", ".md", ".html", ".csv", ".xlsx", ".png", ".jpg", ".tiff", ".webp"]) assert.ok(ACCEPT.includes(ext), ext);
  for (const type of ["image/png", "image/jpeg", "image/tiff", "image/webp"]) assert.ok(ACCEPT.includes(type), type);
});

test("sizes read as people write them in each language", () => {
  assert.equal(formatBytes(820, "es"), "820 B");
  assert.equal(formatBytes(1536, "en"), "1.5 KB");
  assert.equal(formatBytes(1536, "es"), "1,5 KB");
  assert.equal(formatBytes(25 * 1024 * 1024, "en"), "25 MB");
  assert.equal(formatBytes(-1), "");
});

test("domain ids follow the contract's pattern; a title gives a valid id", () => {
  assert.equal(slugify("Entrevistas Técnicas: Backend"), "entrevistas-tecnicas-backend");
  assert.equal(slugify("  --Ñandú 2026  "), "nandu-2026");
  assert.ok(isDomainId(slugify("Atención al cliente")));
  assert.ok(isDomainId("interview.backend"));
  assert.ok(!isDomainId("Upper"));
  assert.ok(!isDomainId("-start"));
  assert.ok(!isDomainId("a".repeat(65)));
  assert.equal(slugify("a".repeat(80)).length, 64);
});

test("tags are trimmed, unique and non-empty; blank metadata fields become null", () => {
  assert.deepEqual(parseTags(" rrhh, vacaciones ,, rrhh;nóminas"), ["rrhh", "vacaciones", "nóminas"]);
  assert.deepEqual(cleanMetadata({ category: "pregunta", intent: "  ", script_phase: "cierre ", tags: ["a", "a", " b"] }),
    { category: "pregunta", intent: null, script_phase: "cierre", tags: ["a", "b"] });
  assert.equal(cleanMetadata({ category: "nonsense" }).category, null, "only the contract's categories");
  assert.equal(CATEGORIES.length, 13);
});

test("a test search is a valid knowledge.query.v1 body", () => {
  const body = queryBody({ domain: "hr", text: "  vacaciones  ", topK: 99, mode: "odd", language: "es", id: "r-1" });
  assert.deepEqual(body, { schema: "knowledge.query.v1", request_id: "r-1", domain: "hr", text: "vacaciones", top_k: 50, mode: "hybrid", language: "es" });
  assert.equal(queryBody({ domain: "hr", text: "x", topK: 0 }).top_k, 1);
  assert.ok(!("language" in queryBody({ domain: "hr", text: "x" })));
  const id = requestId(1700000000000, 0.5);
  assert.ok(id.length >= 1 && id.length <= 64);
});

test("lists arrive bare or wrapped; pending documents ask for polling; newest first", () => {
  assert.deepEqual(listOf([1], "domains"), [1]);
  assert.deepEqual(listOf({ domains: [2] }, "domains"), [2]);
  assert.deepEqual(listOf(null, "domains"), []);
  assert.ok(needsPolling([{ status: "indexed" }, { status: "processing" }]));
  assert.ok(!needsPolling([{ status: "indexed" }, { status: "failed" }]));
  assert.deepEqual(sortDocuments([{ name: "a", uploaded_at: "2026-01-01" }, { name: "b", uploaded_at: "2026-02-01" }]).map((d) => d.name), ["b", "a"]);
  assert.deepEqual(countByStatus([{ status: "queued" }, { status: "queued" }, { status: "failed" }]), { queued: 2, processing: 0, indexed: 0, failed: 1 });
  assert.ok(serviceMissing(new ApiFailure({ status: 404, kind: "client" })));
  assert.ok(!serviceMissing(new ApiFailure({ status: 503, kind: "unavailable" })));
});

// ---- the API client ----------------------------------------------------------

function recorder(reply) {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return reply(url, init);
  });
  return { api, calls };
}

const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

test("the knowledge calls use the documented routes under /api with the access code", async () => {
  const { api, calls } = recorder((url, init) => (init.method === "DELETE" ? new Response(null, { status: 204 }) : json(200, {})));
  await api.knowledgeDomains();
  await api.createKnowledgeDomain({ id: "hr", title: "RR. HH.", language: "es" });
  await api.deleteKnowledgeDomain("hr x");
  await api.knowledgeDocuments("hr");
  await api.updateKnowledgeDocument("hr", "d/1", { category: null, intent: "x", script_phase: null, tags: [] });
  await api.deleteKnowledgeDocument("hr", "d1");
  await api.knowledgeQuery({ schema: "knowledge.query.v1" });
  assert.deepEqual(calls.map((c) => `${c.init.method} ${c.url}`), [
    "GET /api/knowledge/domains",
    "POST /api/knowledge/domains",
    "DELETE /api/knowledge/domains/hr%20x",
    "GET /api/knowledge/domains/hr/documents",
    "PATCH /api/knowledge/domains/hr/documents/d%2F1",
    "DELETE /api/knowledge/domains/hr/documents/d1",
    "POST /api/knowledge/query",
  ]);
  assert.ok(calls.every((c) => c.init.headers["X-Access-Code"] === "code-123"));
  assert.deepEqual(JSON.parse(calls[1].init.body), { id: "hr", title: "RR. HH.", language: "es" });
  assert.deepEqual(JSON.parse(calls[4].init.body), { metadata: { category: null, intent: "x", script_phase: null, tags: [] } });
});

test("an upload is multipart with the file and the metadata as JSON; the browser sets the boundary", async () => {
  const { api, calls } = recorder(() => json(201, { id: "d1", status: "queued" }));
  // A Blob with a name stands for a File (File is global only from Node 20).
  const file = Object.assign(new Blob(["hola"], { type: "application/pdf" }), { name: "guía.pdf" });
  const answer = await api.uploadKnowledgeDocument("hr", file, { category: "otro", tags: ["a"] });
  assert.deepEqual(answer, { id: "d1", status: "queued" });
  const { url, init } = calls[0];
  assert.equal(url, "/api/knowledge/domains/hr/documents");
  assert.equal(init.method, "POST");
  assert.ok(init.body instanceof FormData);
  assert.equal(init.headers["Content-Type"], undefined, "no hand-written multipart header");
  assert.equal(init.body.get("file").name, "guía.pdf");
  assert.deepEqual(JSON.parse(init.body.get("metadata")), { category: "otro", tags: ["a"] });
  await api.uploadKnowledgeDocument("hr", file);
  assert.equal(calls[1].init.body.has("metadata"), false);
});

// ---- the page model against a fake backend -----------------------------------

/** An in-memory /knowledge backend: each poll moves a document one status on. */
class FakeKnowledge {
  constructor({ missing = false, failing = new Set() } = {}) {
    this.missing = missing;
    this.failing = failing; // file names whose processing fails
    this.domains = [{ id: "hr", title: "RR. HH.", language: "es", documents: 0, chunks: 0, index_version: null, updated_at: null }];
    this.docs = { hr: [] };
    this.calls = [];
    this.next = 0;
    this.queries = [];
  }

  guard(name) {
    this.calls.push(name);
    if (this.missing) throw new ApiFailure({ status: 404, kind: "client", code: "not_found" });
  }

  recount(id) {
    const domain = this.domains.find((d) => d.id === id);
    domain.documents = this.docs[id].length;
    domain.chunks = this.docs[id].reduce((n, d) => n + (d.chunks ?? 0), 0);
  }

  async knowledgeDomains() { this.guard("domains"); return { domains: structuredClone(this.domains) }; }

  async createKnowledgeDomain(domain) {
    this.guard("create");
    if (this.domains.some((d) => d.id === domain.id)) throw new ApiFailure({ status: 409, kind: "client", code: "conflict", message: "exists" });
    this.domains.push({ ...domain, documents: 0, chunks: 0, index_version: null, updated_at: null });
    this.docs[domain.id] = [];
    return domain;
  }

  async deleteKnowledgeDomain(id) {
    this.guard("delete-domain");
    this.domains = this.domains.filter((d) => d.id !== id);
    delete this.docs[id];
    return null;
  }

  async knowledgeDocuments(id) {
    this.guard("documents");
    for (const doc of this.docs[id]) {
      if (doc.status === "processing") {
        if (this.failing.has(doc.name)) Object.assign(doc, { status: "failed", error: "No text could be read" });
        else Object.assign(doc, { status: "indexed", chunks: 4, pages: 2 });
      } else if (doc.status === "queued") {
        doc.status = "processing";
      }
    }
    this.recount(id);
    return structuredClone(this.docs[id]);
  }

  async uploadKnowledgeDocument(id, file, metadata) {
    this.guard("upload");
    if (file.name.startsWith("offline")) throw new ApiFailure({ status: null, kind: "unavailable", code: "network" });
    this.next += 1;
    const doc = {
      id: `d${this.next}`, name: file.name, type: file.name.split(".").pop(), size: file.size, pages: null, chunks: null,
      status: "queued", error: null, metadata, uploaded_at: `2026-11-21T10:00:0${this.next}Z`,
    };
    this.docs[id].push(doc);
    return structuredClone(doc);
  }

  async updateKnowledgeDocument(id, docId, metadata) {
    this.guard("patch");
    const doc = this.docs[id].find((d) => d.id === docId);
    doc.metadata = metadata;
    return structuredClone(doc);
  }

  async deleteKnowledgeDocument(id, docId) {
    this.guard("delete-doc");
    this.docs[id] = this.docs[id].filter((d) => d.id !== docId);
    this.recount(id);
    return null;
  }

  async knowledgeQuery(body) {
    this.guard("query");
    this.queries.push(body);
    return { schema: "knowledge.result.v1", request_id: body.request_id, domain: body.domain, index_version: "v1", chunks: [
      { id: "c1", text: "22 días laborables", score: 0.91, source: { file_path: "guia.pdf", first_character_index: 0, last_character_index: 18 }, metadata: { tags: ["rrhh"] } },
    ] };
  }
}

/** A timer the test fires by hand. */
function manualClock() {
  const pending = new Map();
  let id = 0;
  return {
    set(fn, ms) { id += 1; pending.set(id, { fn, ms }); return id; },
    clear(handle) { pending.delete(handle); },
    count() { return pending.size; },
    delays() { return [...pending.values()].map((p) => p.ms); },
    async fire() {
      const all = [...pending.entries()];
      pending.clear();
      for (const [, { fn }] of all) await fn();
    },
  };
}

const file = (name, size = 100) => ({ name, size });

test("a 404 on the domain list means the backend is not deployed: a friendly state, no polling", async () => {
  const clock = manualClock();
  const model = new KnowledgeModel(new FakeKnowledge({ missing: true }), { timer: clock });
  await model.load();
  assert.equal(model.state.status, "missing");
  assert.equal(clock.count(), 0);
});

test("other failures are errors the page can retry", async () => {
  const api = { knowledgeDomains: async () => { throw new ApiFailure({ status: 503, kind: "unavailable" }); } };
  const model = new KnowledgeModel(api, { timer: manualClock() });
  await model.load();
  assert.equal(model.state.status, "error");
  assert.equal(model.state.error.kind, "unavailable");
});

test("uploads go one by one, then the page polls every 2 s until every document is settled", async () => {
  const fake = new FakeKnowledge({ failing: new Set(["roto.pdf"]) });
  const clock = manualClock();
  const seen = [];
  const model = new KnowledgeModel(fake, { timer: clock, onChange: (s) => seen.push(s.uploads.map((u) => u.state).join(",")) });
  await model.load("hr");
  assert.equal(model.state.domain, "hr");
  assert.equal(clock.count(), 0, "nothing pending, no polling");

  await model.addFiles([file("guia.pdf"), file("virus.exe"), file("roto.pdf")], { category: "informacion", tags: "rrhh, vacaciones" });
  assert.deepEqual(fake.calls.filter((c) => c === "upload").length, 2, "the rejected file is never sent");
  assert.ok(seen.some((s) => s.includes("uploading")), "each file shows while it uploads");
  assert.deepEqual(model.state.uploads.map((u) => [u.name, u.state, u.reason]), [["virus.exe", "rejected", "type"]]);
  assert.deepEqual(fake.docs.hr[0].metadata, { category: "informacion", intent: null, script_phase: null, tags: ["rrhh", "vacaciones"] });
  // The refresh after the uploads moved them to processing; a poll is planned at 2 s.
  assert.deepEqual(model.state.documents.map((d) => d.status), ["processing", "processing"]);
  assert.deepEqual(clock.delays(), [POLL_MS]);

  await clock.fire();
  const byName = Object.fromEntries(model.state.documents.map((d) => [d.name, d]));
  assert.equal(byName["guia.pdf"].status, "indexed");
  assert.equal(byName["guia.pdf"].chunks, 4);
  assert.equal(byName["roto.pdf"].status, "failed");
  assert.equal(byName["roto.pdf"].error, "No text could be read");
  assert.equal(model.selected().documents, 2, "the domain's counts are refreshed too");
  assert.equal(clock.count(), 0, "polling stops once nothing is queued or processing");

  model.dismissUpload(model.state.uploads[0].key);
  assert.equal(model.state.uploads.length, 0);
});

test("a failed upload keeps its error and can be retried", async () => {
  const fake = new FakeKnowledge();
  const model = new KnowledgeModel(fake, { timer: manualClock() });
  await model.load();
  await model.addFiles([file("offline.txt")]);
  const [item] = model.state.uploads;
  assert.equal(item.state, "failed");
  assert.equal(item.error.kind, "unavailable");
  item.file = file("fine.txt");
  await model.retryUpload(item.key);
  assert.equal(model.state.uploads.length, 0);
  assert.deepEqual(model.state.documents.map((d) => d.name), ["fine.txt"]);
});

test("metadata edits and deletions reach the backend and the list", async () => {
  const fake = new FakeKnowledge();
  const clock = manualClock();
  const model = new KnowledgeModel(fake, { timer: clock });
  await model.load();
  await model.addFiles([file("a.md"), file("b.csv")]);
  await clock.fire();
  const a = model.state.documents.find((d) => d.name === "a.md");
  await model.updateDocument(a.id, { category: "propuesta", intent: "ofrecer", script_phase: "cierre", tags: ["x"] });
  assert.deepEqual(fake.docs.hr.find((d) => d.id === a.id).metadata, { category: "propuesta", intent: "ofrecer", script_phase: "cierre", tags: ["x"] });
  assert.equal(model.state.documents.find((d) => d.id === a.id).metadata.intent, "ofrecer");
  await model.deleteDocument(a.id);
  assert.deepEqual(model.state.documents.map((d) => d.name), ["b.csv"]);
  assert.equal(model.selected().documents, 1);
});

test("domains are created, selected and deleted; an existing id is refused with the server's answer", async () => {
  const fake = new FakeKnowledge();
  const model = new KnowledgeModel(fake, { timer: manualClock() });
  await model.load();
  await model.createDomain({ id: "ventas", title: " Ventas ", language: "es" });
  assert.equal(model.state.domain, "ventas");
  assert.equal(fake.domains.at(-1).title, "Ventas");
  await assert.rejects(model.createDomain({ id: "hr", title: "Otra", language: "es" }), (e) => e.code === "conflict");
  await model.deleteDomain("ventas");
  assert.deepEqual(model.state.domains.map((d) => d.id), ["hr"]);
  assert.equal(model.state.domain, "hr", "the first domain left is selected");
  await model.deleteDomain("hr");
  assert.equal(model.state.domain, null);
});

test("the test search sends the domain's language and keeps the chunks", async () => {
  const fake = new FakeKnowledge();
  const model = new KnowledgeModel(fake, { timer: manualClock() });
  await model.load();
  await model.search({ text: "   " });
  assert.equal(fake.queries.length, 0, "an empty sentence is not sent");
  await model.search({ text: "¿cuántas vacaciones?", topK: 3, mode: "semantic" });
  assert.equal(fake.queries[0].domain, "hr");
  assert.equal(fake.queries[0].language, "es");
  assert.equal(fake.queries[0].top_k, 3);
  assert.equal(fake.queries[0].mode, "semantic");
  assert.equal(model.state.search.result.chunks[0].text, "22 días laborables");
  assert.equal(model.state.search.running, false);
});

test("leaving the page stops the polling and the redraws", async () => {
  const fake = new FakeKnowledge();
  const clock = manualClock();
  let redraws = 0;
  const model = new KnowledgeModel(fake, { timer: clock, onChange: () => { redraws += 1; } });
  await model.load();
  await model.addFiles([file("a.txt")]);
  assert.equal(clock.count(), 1);
  model.stop();
  assert.equal(clock.count(), 0);
  const before = redraws;
  await model.refreshDomains();
  assert.equal(redraws, before);
});

test("switching domain drops the other domain's poll and an answer that arrives late", async () => {
  const fake = new FakeKnowledge();
  await fake.createKnowledgeDomain({ id: "b", title: "B", language: "en" });
  const clock = manualClock();
  const model = new KnowledgeModel(fake, { timer: clock });
  await model.load("hr");
  await model.addFiles([file("a.txt")]);
  assert.equal(clock.count(), 1);
  await model.select("b");
  assert.equal(clock.count(), 0);
  assert.deepEqual(model.state.documents, []);
});
