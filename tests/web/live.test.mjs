// The live page (#/live) without a browser: the SSE parser, the session
// state built from the orchestrator's feed, script-phase coverage, the
// customer signals (support only, never in interviews) and the API calls
// through /api/assist, with the stream read from a fake fetch.
// Nothing here depends on the browser's language: the logic has no text.
import test from "node:test";
import assert from "node:assert/strict";
import {
  INTERVIEW, LiveState, SUPPORT, citations, coverage, latency, newSessionId, parseSse, productKind, showsSignals, signalLines,
} from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { Api, ApiFailure } from "../../transcriber/adapters/inbound/web/js/api.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const SUPPORT_PAGE = { product: SUPPORT, signals: true, banner: "support_ai_notice@1", script_phases: [], roles: { S1: "agent", S2: "customer" } };
const INTERVIEW_PAGE = {
  product: INTERVIEW, signals: false, banner: "interview_ai_notice@1",
  script_phases: ["opening", "experience", "technical", "behavioural", "closing"], roles: { S1: "interviewer", S2: "candidate" },
};

function event(type, extra = {}) {
  return { type: "event", event: { schema: "live.event.v1", session_id: "s", seq: 0, type, at: "2026-11-21T10:00:00Z", stream_time: 0, ...extra } };
}

function utterance(id, speaker, text, start, revision = 0) {
  return { id, turn: 0, index: 0, speaker, start, end: start + 1, text, revision };
}

function record(status, suggestions, retrieved = []) {
  return { type: "suggestion", record: { status, suggestions, retrieved, stages: { knowledge_ms: 12.4, reasoner_ms: 980.6, total_ms: 995.2 } } };
}

const SIGNALLED = {
  kind: "answer", text: "Ofrece escalar el caso", cites: ["k1", "k9"], about: ["t1.s0"],
  signals: ["complaint"], expressed: [{ signal: "complaint", utterance: "t1.s0", quote: "Es la tercera vez que llamo." }],
};

// ---- parser and small helpers ----------------------------------------------

test("SSE items are split from a buffer; an unfinished item waits", () => {
  const { items, rest } = parseSse('retry: 2000\n\nid: 1\nevent: message\ndata: {"a":1}\n\n: keepalive\n\nid: 2\ndata: x\r\ndata: y\r\n\r\nevent: end\ndata: {}\n\nid: 3\nda');
  assert.deepEqual(items, [
    { id: "1", event: "message", data: '{"a":1}' },
    { id: "2", event: "message", data: "x\ny" },
    { id: null, event: "end", data: "{}" },
  ]);
  assert.equal(rest, "id: 3\nda");
  assert.deepEqual(parseSse("").items, []);
});

test("products, ids and latencies", () => {
  assert.equal(productKind(SUPPORT), "support");
  assert.equal(productKind(INTERVIEW), "interview");
  assert.equal(productKind("post_call"), null);
  assert.equal(newSessionId("demo", new Date(2026, 10, 21, 9, 5, 7)), "demo-20261121-090507");
  assert.match(newSessionId(), /^demo-\d{8}-\d{6}$/);
  assert.deepEqual(latency({ knowledge_ms: 12.4, reasoner_ms: 980.6, total_ms: null }), { knowledge: 12, reasoner: 981, total: null });
  assert.deepEqual(latency(), { knowledge: null, reasoner: null, total: null });
});

// ---- the session state -----------------------------------------------------

test("finals keep their last revision, partials wait for them, speakers update", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { id: "s", mode: "replay", page: SUPPORT_PAGE } }, "1");
  assert.equal(state.state, "running");
  state.apply(event("session_started", { speakers: [{ id: "S1", role: "agent", name: "Ana" }, { id: "S2", role: "customer" }] }), "2");
  state.apply(event("partial", { utterance: utterance("t0.s0", "S2", "Me cobr", 1) }), "3");
  assert.equal(state.lines()[0].partial, true);
  state.apply(event("final", { utterance: utterance("t0.s0", "S2", "Me cobraron dos veces", 1, 2) }), "4");
  state.apply(event("final", { utterance: utterance("t0.s0", "S2", "Me cobraron", 1, 1) }), "5");
  state.apply(event("final", { utterance: utterance("t1.s0", "S1", "Lo reviso", 0.2) }), "6");
  state.apply(event("speaker_update", { utterance: utterance("t1.s0", "S2", "Lo reviso", 0.2) }), "7");
  state.apply(event("partial", { utterance: utterance("t0.s0", "S2", "late partial", 1) }), "8");
  state.apply(event("warning", { warning: { code: "asr_behind", message: "behind" } }), "9");
  const lines = state.lines();
  assert.deepEqual(lines.map((u) => [u.id, u.speaker, u.text]), [["t1.s0", "S2", "Lo reviso"], ["t0.s0", "S2", "Me cobraron dos veces"]]);
  assert.equal(state.roleOf("S1"), "agent");
  assert.equal(state.roleOf("nobody"), "unknown");
  assert.equal(state.speakers.get("S1").name, "Ana");
  assert.deepEqual(state.warnings, [{ code: "asr_behind", message: "behind" }]);
  assert.equal(state.lastId, 9);
  state.apply({ type: "status", session: { state: "failed", error: "closed 4404" } }, "10");
  assert.equal(state.state, "failed");
  assert.equal(state.error, "closed 4404");
  state.apply({ type: "nothing" }, "x");
  assert.equal(state.lastId, 10);
});

test("cards are newest first with sources, quoted sentences and counts", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: SUPPORT_PAGE } });
  state.apply(event("final", { utterance: utterance("t1.s0", "S2", "Es la tercera vez que llamo.", 1) }));
  state.apply(record("ok", [SIGNALLED], [{ id: "k1", text: "Escala tras dos llamadas", metadata: { intent: "escalation" } }]));
  state.apply(record("timeout", []));
  state.apply(record("ok", [{ kind: "question", text: "¿Número de pedido?", about: ["t9.s9"] }]));
  const cards = state.cards();
  assert.deepEqual(cards.map((c) => c.suggestion.text), ["¿Número de pedido?", "Ofrece escalar el caso"]);
  assert.deepEqual(cards[1].cites, [
    { id: "k1", text: "Escala tras dos llamadas", metadata: { intent: "escalation" } },
    { id: "k9", text: null, metadata: {} },
  ]);
  assert.equal(cards[1].about[0].text, "Es la tercera vez que llamo.");
  assert.deepEqual(cards[0].about, [{ id: "t9.s9", speaker: null, text: null }]);
  assert.deepEqual(state.counts(), { ok: 2, timeout: 1 });
  assert.equal(state.halted, null);
  state.apply(record("quota", []));
  assert.equal(state.halted, "quota");
});

// ---- customer signals (#153) -----------------------------------------------

test("support shows what the customer says, with the quoted sentence", () => {
  assert.equal(showsSignals(SUPPORT_PAGE), true);
  const lines = signalLines(SIGNALLED, SUPPORT_PAGE);
  assert.deepEqual(lines, [{ signal: "complaint", utterance: "t1.s0", quote: "Es la tercera vez que llamo." }]);
  assert.deepEqual(signalLines({ expressed: [{ signal: "complaint", quote: "" }, null] }, SUPPORT_PAGE), []);
  assert.deepEqual(signalLines({}, SUPPORT_PAGE), []);
});

test("an interview never shows signals, even if a suggestion carried them", () => {
  assert.equal(showsSignals(INTERVIEW_PAGE), false);
  assert.equal(showsSignals({ ...INTERVIEW_PAGE, signals: true }), false);
  assert.equal(showsSignals(null), false);
  assert.deepEqual(signalLines(SIGNALLED, INTERVIEW_PAGE), []);
  const state = new LiveState();
  state.apply({ type: "session", session: { page: { ...INTERVIEW_PAGE, signals: true } } });
  state.apply(record("ok", [SIGNALLED]));
  assert.deepEqual(state.cards()[0].signals, []);
});

test("the dictionaries name every signal as something said, in both languages", () => {
  for (const signal of ["complaint", "stated_frustration", "urgency", "churn_risk", "confusion", "satisfaction"]) {
    assert.ok(es[`live.signal.${signal}`], signal);
    assert.ok(en[`live.signal.${signal}`], signal);
  }
  assert.equal(es["live.customer_says"], "El cliente dice:");
  const live = (d) => Object.keys(d).filter((k) => k.startsWith("live.") || k === "nav.live").sort();
  assert.deepEqual(live(es), live(en));
  // The interview notice says no one is scored or rated.
  assert.match(es["live.banner.interview_ai_notice@1"], /No puntúa ni valora/);
});

// ---- script-phase coverage -------------------------------------------------

test("coverage marks covered, current and pending phases", () => {
  const records = [
    { status: "ok", suggestions: [{ script_phase: "opening" }] },
    { status: "ok", suggestions: [{ text: "x" }], retrieved: [{ metadata: { script_phase: "experience" } }] },
    { status: "timeout", suggestions: [{ script_phase: "closing" }] },
    { status: "ok", suggestions: [{ script_phase: "technical" }, { script_phase: "unknown-phase" }] },
    { status: "no_suggestion", suggestions: [], retrieved: [] },
  ];
  assert.deepEqual(coverage(INTERVIEW_PAGE.script_phases, records), [
    { phase: "opening", count: 1, state: "covered" },
    { phase: "experience", count: 1, state: "covered" },
    { phase: "technical", count: 1, state: "current" },
    { phase: "behavioural", count: 0, state: "pending" },
    { phase: "closing", count: 0, state: "pending" },
  ]);
  assert.deepEqual(coverage([], records), []);
  const state = new LiveState();
  assert.deepEqual(state.coverage(), []);
  assert.deepEqual(citations({}, null), []);
});

// ---- API through the transcriber's proxy -----------------------------------

function recorder(reply) {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return reply(url, init);
  });
  return { api, calls };
}

const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

test("live calls go to /api/assist with the code", async () => {
  const { api, calls } = recorder(() => json(201, { id: "d" }));
  await api.liveProducts();
  await api.liveTranscripts();
  await api.startLive({ product: SUPPORT, audio: "mic", session_id: "live-1" });
  await api.startReplay("demo 1", { product: SUPPORT, transcript: "t", speed: 2 });
  await api.stopLive("demo-1");
  await api.liveSession("demo-1");
  assert.deepEqual(calls.map((c) => [c.init.method, c.url]), [
    ["GET", "/api/assist/products"],
    ["GET", "/api/assist/transcripts"],
    ["POST", "/api/assist/sessions"],
    ["POST", "/api/assist/sessions/demo%201/replay"],
    ["POST", "/api/assist/sessions/demo-1/stop"],
    ["GET", "/api/assist/sessions/demo-1"],
  ]);
  assert.equal(JSON.parse(calls[3].init.body).speed, 2);
  assert.ok(calls.every((c) => c.init.headers["X-Access-Code"] === "code-123"));
});

function streamOf(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

test("the stream is read piece by piece until its end", async () => {
  const { api, calls } = recorder(() => new Response(streamOf([
    "retry: 2000\n\nid: 4\nevent: message\ndata: {\"type\":",
    " \"session\"}\n\nid: 5\ndata: {\"type\": \"event\"}\n\n",
    "event: end\ndata: {}\n\n",
  ]), { status: 200, headers: { "Content-Type": "text/event-stream" } }));
  const got = [];
  const how = await api.followLive("d-1", { after: 3, onItem: (item) => got.push(item) });
  assert.equal(how, "end");
  assert.deepEqual(got.map((i) => [i.id, JSON.parse(i.data).type]), [["4", "session"], ["5", "event"]]);
  assert.equal(calls[0].url, "/api/assist/sessions/d-1/suggestions/stream?after=3");
  assert.equal(calls[0].init.headers["X-Access-Code"], "code-123");
});

test("a stream that closes early says so; refusals are failures", async () => {
  const closed = recorder(() => new Response(streamOf(["id: 1\ndata: {}\n\n"]), { status: 200 }));
  assert.equal(await closed.api.followLive("d", { onItem: () => {} }), "closed");
  const missing = recorder(() => json(404, { code: "not_found" }));
  await assert.rejects(missing.api.followLive("d", { onItem: () => {} }), (e) => e instanceof ApiFailure && e.code === "not_found" && e.kind === "client");
  const down = recorder(() => json(502, { code: "orchestrator_unavailable" }));
  await assert.rejects(down.api.followLive("d", { onItem: () => {} }), (e) => e.kind === "unavailable" && e.code === "orchestrator_unavailable");
  const offline = new Api(() => "c", async () => { throw new TypeError("offline"); });
  await assert.rejects(offline.followLive("d", { onItem: () => {} }), (e) => e.kind === "unavailable");
  const nocode = new Api(() => null, async () => json(200, {}));
  await assert.rejects(nocode.followLive("d", { onItem: () => {} }), (e) => e.kind === "unauthorized");
  const broken = recorder(() => new Response(new ReadableStream({
    pull(controller) { controller.error(new Error("reset")); },
  }), { status: 200 }));
  await assert.rejects(broken.api.followLive("d", { onItem: () => {} }), (e) => e.kind === "unavailable");
});
