// The connection check of the live page (#247) without a browser or a
// network: the figures of a check, the one verdict and its advice, the
// in-session watch that only moves after a sustained problem, and the
// requests the check makes (a scripted fetch and clock).
import test from "node:test";
import assert from "node:assert/strict";
import {
  CHECK, ConnectionWatch, THRESHOLDS, WATCH, assess, best, grade, jitter, median, needsOf, summarise, uploadKbps, worst,
} from "../../transcriber/adapters/inbound/web/js/lib/connection.js";
import { pingOnce, runCheck, servicesOnce, timed } from "../../transcriber/adapters/inbound/web/js/connection.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const ALL_UP = { transcription: "available", assistant: "available", knowledge: "available" };

function samples(...ms) {
  return ms.map((m) => (m === null ? { ok: false, ms: CHECK.timeoutMs } : { ok: true, ms: m }));
}

function result({ link = samples(40, 42, 41, 43, 40), web = samples(20, 21, 20), services = ALL_UP, upload = null, reachable = true, webUp = true } = {}) {
  return { web: { reachable: webUp, samples: web }, link: { reachable, samples: reachable ? link : [] }, services, upload };
}

test("figures: median, jitter, loss and a lower bound of the upload rate", () => {
  assert.equal(median([]), null);
  assert.equal(median([30, 10, 20]), 20);
  assert.equal(median([10, 20, 30, 40]), 25);
  assert.equal(jitter([50]), 0);
  assert.equal(jitter([40, 60, 50]), 15);
  const s = summarise(samples(40, null, 60, 50, null));
  assert.deepEqual([s.count, s.lost, s.lossPct, s.rttMs, s.jitterMs], [5, 2, 40, 50, 15]);
  assert.deepEqual(summarise([]), { count: 0, lost: 0, lossPct: 0, rttMs: null, jitterMs: 0 });
  // 64 KiB in 300 ms with a 50 ms round trip: 524288 bits in 250 ms.
  assert.equal(Math.round(uploadKbps(65536, 300, 50)), 2097);
  // The round trip never takes more than three quarters of the time.
  assert.equal(Math.round(uploadKbps(65536, 100, 400)), 20972);
  assert.equal(uploadKbps(0, 100), null);
  assert.equal(uploadKbps(65536, 0), null);
});

test("grades: up to good is good, up to fair is fair; the upload the other way", () => {
  assert.equal(grade(150, THRESHOLDS.rttMs), "good");
  assert.equal(grade(151, THRESHOLDS.rttMs), "fair");
  assert.equal(grade(400, THRESHOLDS.rttMs), "fair");
  assert.equal(grade(401, THRESHOLDS.rttMs), "poor");
  assert.equal(grade(null, THRESHOLDS.rttMs), "good");
  assert.equal(grade(1024, THRESHOLDS.upKbps, true), "good");
  assert.equal(grade(500, THRESHOLDS.upKbps, true), "fair");
  assert.equal(grade(300, THRESHOLDS.upKbps, true), "poor");
  assert.equal(worst("good", "poor", "fair"), "poor");
  assert.equal(best("poor", "fair"), "fair");
  assert.equal(worst(), "good");
});

test("a quick, steady link with every service up is suitable", () => {
  const out = assess(result({ upload: { bytes: 65536, ms: 200 } }), { upload: true });
  assert.equal(out.level, "good");
  assert.equal(out.advice, null);
  assert.deepEqual(out.rows, { web: "available", link: "available", ...ALL_UP });
  assert.equal(out.figures.rttMs, 41);
  assert.equal(out.figures.webRttMs, 20);
});

test("no answer at all is offline; the public server alone is the processing server down", () => {
  const offline = assess(result({ reachable: false, webUp: false, services: null }));
  assert.deepEqual([offline.level, offline.advice], ["poor", "offline"]);
  const server = assess(result({ reachable: false, services: null }));
  assert.deepEqual([server.level, server.advice], ["poor", "server"]);
  assert.equal(server.rows.web, "available");
  assert.equal(server.rows.link, "unavailable");
  assert.equal(server.rows.assistant, "unavailable");
});

test("a slow link: the advice says whose network it is", () => {
  const slowBoth = assess(result({ link: samples(450, 480, 460, 470, 455), web: samples(420, 400, 430) }));
  assert.deepEqual([slowBoth.level, slowBoth.advice], ["poor", "local"]);
  const slowTunnel = assess(result({ link: samples(250, 260, 255, 240, 250) }));
  assert.deepEqual([slowTunnel.level, slowTunnel.advice], ["fair", "link"]);
  const jumpy = assess(result({ link: samples(40, 300, 40, 300, 40) }));
  assert.equal(jumpy.level, "poor"); // jitter 260 ms
  const oneLost = assess(result({ link: samples(40, 41, null, 42, 40) }));
  assert.deepEqual([oneLost.level, oneLost.advice], ["fair", "link"]); // one in five lost
});

test("loss: none is good, one in five is fair, more is poor", () => {
  assert.equal(grade(0, THRESHOLDS.lossPct), "good");
  assert.equal(grade(20, THRESHOLDS.lossPct), "fair");
  assert.equal(grade(40, THRESHOLDS.lossPct), "poor");
});

test("the upload counts only when this browser sends the audio", () => {
  const slowUp = result({ upload: { bytes: 65536, ms: 1800 } }); // about 297 kbit/s
  assert.deepEqual([assess(slowUp, { upload: true }).level, assess(slowUp, { upload: true }).advice], ["poor", "upload"]);
  assert.equal(assess(slowUp, { upload: false }).level, "good");
  const notMeasured = assess(result(), { upload: true });
  assert.equal(notMeasured.level, "good");
});

test("services: transcription or assistant down is poor, knowledge down is fair, unknown is neutral", () => {
  const noAssistant = assess(result({ services: { ...ALL_UP, assistant: "unavailable" } }));
  assert.deepEqual([noAssistant.level, noAssistant.advice], ["poor", "service_assistant"]);
  const noKnowledge = assess(result({ services: { ...ALL_UP, knowledge: "unavailable" } }));
  assert.deepEqual([noKnowledge.level, noKnowledge.advice], ["fair", "service_knowledge"]);
  const unknown = assess(result({ services: null }));
  assert.equal(unknown.level, "good");
  assert.equal(unknown.rows.transcription, "unknown");
  // A demo needs only the assistant: the transcription being down does not matter.
  const demo = assess(result({ services: { ...ALL_UP, transcription: "unavailable" } }), { needs: needsOf("demo") });
  assert.equal(demo.level, "good");
  assert.deepEqual(needsOf("replay"), ["assistant", "knowledge"]);
  assert.deepEqual(needsOf("live"), ["transcription", "assistant", "knowledge"]);
});

test("every advice and verdict has its words in both languages, with the professional term", () => {
  const keys = ["offline", "server", "local", "link", "upload", "service_transcription", "service_assistant", "service_knowledge"]
    .map((a) => `conn.advice.${a}`)
    .concat(["good", "fair", "poor"].map((l) => `conn.verdict.${l}`))
    .concat(["checking", "good", "fair", "poor", "lost"].flatMap((s) => [`conn.chip.${s}`, `conn.chip_title.${s}`]))
    .concat(["web", "link", "transcription", "assistant", "knowledge"].map((r) => `conn.row.${r}`));
  for (const key of keys) {
    assert.ok(es[key], key);
    assert.ok(en[key], key);
  }
  const esTexts = Object.entries(es).filter(([k]) => k.startsWith("conn.")).map(([, v]) => v).join(" ");
  const enTexts = Object.entries(en).filter(([k]) => k.startsWith("conn.")).map(([, v]) => v).join(" ");
  assert.doesNotMatch(esTexts, /\bPC\b|ordenador|de casa/i);
  assert.doesNotMatch(enTexts, /\bPC\b|home computer|workstation/i);
  assert.match(es["conn.advice.server"], /servidor de procesamiento/);
  assert.match(en["conn.advice.server"], /processing server/);
  assert.equal(es["conn.verdict.good"], "Conexión adecuada para la entrevista");
});

test("the watch ignores one slow or lost ping and moves after a sustained problem", () => {
  const watch = new ConnectionWatch({ initial: "good" });
  assert.equal(watch.state, "good");
  assert.equal(watch.ping({ ok: true, ms: 900 }).changed, false); // one slow answer
  assert.equal(watch.ping({ ok: true, ms: 40 }).state, "good");
  assert.equal(watch.ping({ ok: false, ms: 4000 }).state, "good"); // one lost: not yet
  assert.equal(watch.ping({ ok: true, ms: 40 }).state, "good");
  // Three slow answers in a row: fair (WATCH.worse).
  watch.ping({ ok: true, ms: 300 });
  assert.equal(watch.ping({ ok: true, ms: 320 }).state, "good");
  const third = watch.ping({ ok: true, ms: 310 });
  assert.deepEqual([third.state, third.changed], ["fair", true]);
  // One good answer is not enough to go back (WATCH.better = 2).
  assert.equal(watch.ping({ ok: true, ms: 40 }).state, "fair");
  assert.equal(watch.ping({ ok: true, ms: 45 }).state, "good");
});

test("two failed pings in a row are a lost connection; the next answer says it is back", () => {
  const watch = new ConnectionWatch({ initial: "good" });
  assert.equal(watch.ping({ ok: false, ms: 4000 }).lost, false);
  const lost = watch.ping({ ok: false, ms: 4000 });
  assert.deepEqual([lost.state, lost.lost, lost.changed], ["lost", true, true]);
  assert.equal(watch.ping({ ok: false, ms: 4000 }).lost, false); // said once
  const back = watch.ping({ ok: true, ms: 50 });
  assert.deepEqual([back.state, back.back, back.changed], ["good", true, true]);
});

test("pings lost now and then cap the watch at fair", () => {
  const watch = new ConnectionWatch({ initial: "good" });
  for (const ok of [true, false, true, true, false, true]) watch.ping({ ok, ms: 40 });
  assert.equal(watch.state, "fair");
  for (let i = 0; i < WATCH.window; i += 1) watch.ping({ ok: true, ms: 40 });
  assert.equal(watch.state, "good");
});

test("without a start check the watch says checking until the first answer", () => {
  const watch = new ConnectionWatch();
  assert.equal(watch.state, "checking");
  assert.equal(watch.ping({ ok: true, ms: 30 }).state, "good");
  assert.equal(new ConnectionWatch({ initial: "bogus" }).state, "checking");
});

test("a service counts as down only after two checks in a row", () => {
  const watch = new ConnectionWatch({ initial: "good", needs: needsOf("live") });
  const down = { ...ALL_UP, assistant: "unavailable" };
  assert.equal(watch.services(down).state, "good");
  assert.deepEqual([watch.services(down).state], ["poor"]);
  assert.equal(watch.services(ALL_UP).state, "good");
  watch.services({ ...ALL_UP, knowledge: "unavailable" });
  assert.equal(watch.services({ ...ALL_UP, knowledge: "unavailable" }).state, "fair");
  assert.equal(watch.services(null).state, "fair"); // not asked: nothing changes
  const demo = new ConnectionWatch({ initial: "good", needs: needsOf("demo") });
  demo.services({ ...ALL_UP, transcription: "unavailable" });
  assert.equal(demo.services({ ...ALL_UP, transcription: "unavailable" }).state, "good");
});

// ---- the requests, with a scripted fetch and clock -------------------------

function scripted(routes) {
  const calls = [];
  let clock = 0;
  const fetchFn = async (url, init = {}) => {
    calls.push({ url, init });
    const next = routes[url];
    const step = typeof next === "function" ? next(calls.filter((c) => c.url === url).length, init) : next;
    if (!step || step.fail) {
      clock += step?.ms ?? 1;
      throw new TypeError("network");
    }
    clock += step.ms;
    return { status: step.status, json: async () => step.body ?? {} };
  };
  return { calls, fetchFn, now: () => clock };
}

test("a check: pings, the services behind the code and one upload, all on our origin", async () => {
  const net = scripted({
    "/healthz": { status: 200, ms: 20 },
    "/api/health": (n) => ({ status: 200, ms: n === 1 ? 300 : 40 }),
    "/api/connection": { status: 200, ms: 30, body: { services: ALL_UP } },
    "/api/connection/upload": { status: 200, ms: 250, body: { bytes: CHECK.uploadBytes } },
  });
  const out = await runCheck({ getCode: () => "code", fetchFn: net.fetchFn, now: net.now });
  assert.equal(out.web.samples.length, CHECK.webPings);
  // The first ping only opens the connection and is left out.
  assert.equal(out.link.samples.length, CHECK.linkPings - 1);
  assert.ok(out.link.samples.every((s) => s.ms === 40));
  assert.deepEqual(out.services, ALL_UP);
  assert.deepEqual(out.upload, { bytes: CHECK.uploadBytes, ms: 250 });
  assert.ok(net.calls.length <= CHECK.webPings + CHECK.linkPings + 2);
  assert.ok(net.calls.every((c) => c.url.startsWith("/")));
  const health = net.calls.filter((c) => c.url === "/api/health");
  assert.ok(health.every((c) => !c.init.headers)); // the open route: no code sent
  const upload = net.calls.find((c) => c.url === "/api/connection/upload");
  assert.equal(upload.init.method, "POST");
  assert.equal(upload.init.headers["X-Access-Code"], "code");
  assert.equal(upload.init.body.byteLength, CHECK.uploadBytes);
  assert.equal(assess(out, { upload: true }).level, "good");
});

test("a link that is down: 502 from the public server, no services, no upload", async () => {
  const net = scripted({
    "/healthz": { status: 200, ms: 20 },
    "/api/health": { status: 502, ms: 25 },
  });
  const out = await runCheck({ getCode: () => "code", fetchFn: net.fetchFn, now: net.now });
  assert.equal(out.link.reachable, false);
  assert.equal(out.services, null);
  assert.ok(!net.calls.some((c) => c.url.startsWith("/api/connection")));
  assert.equal(assess(out).advice, "server");
});

test("offline: two failures in a row end each series early", async () => {
  const net = scripted({});
  const out = await runCheck({ getCode: () => "code", fetchFn: net.fetchFn, now: net.now });
  assert.equal(net.calls.length, 4); // two to /healthz, two to /api/health
  assert.equal(assess(out).advice, "offline");
});

test("a development server without /healthz still counts as the web service", async () => {
  const net = scripted({
    "/healthz": { status: 404, ms: 5 },
    "/api/health": { status: 200, ms: 6 },
    "/api/connection": { status: 404, ms: 5 },
    "/api/connection/upload": { status: 404, ms: 5 },
  });
  const out = await runCheck({ getCode: () => "code", fetchFn: net.fetchFn, now: net.now });
  assert.equal(out.web.reachable, true);
  assert.equal(out.services, null); // an older server: the rows say «Sin comprobar»
  assert.equal(out.upload, null);
  assert.equal(assess(out, { upload: true }).level, "good");
});

test("a request that takes too long is aborted and counted as lost", async () => {
  const fetchFn = (url, init) => new Promise((resolve, reject) => {
    init.signal.addEventListener("abort", () => reject(new Error("aborted")));
  });
  let clock = 0;
  const answer = await timed(fetchFn, () => (clock += 5), "/api/health", { timeoutMs: 10 });
  assert.equal(answer.ok, false);
  assert.equal(answer.status, null);
});

test("in session: one ping, and the services only with a code", async () => {
  const net = scripted({
    "/api/health": { status: 200, ms: 33 },
    "/api/connection": { status: 200, ms: 10, body: { services: ALL_UP } },
  });
  assert.deepEqual(await pingOnce({ timeoutMs: 100, fetchFn: net.fetchFn, now: net.now }), { ok: true, ms: 33 });
  assert.deepEqual(await servicesOnce({ getCode: () => "c", fetchFn: net.fetchFn, now: net.now }), ALL_UP);
  assert.equal(await servicesOnce({ getCode: () => null, fetchFn: net.fetchFn, now: net.now }), null);
});

test("the pages that draw the check load (no browser needed to parse them)", async () => {
  for (const name of ["connection", "live", "voicebot"]) {
    const loaded = await import(`../../transcriber/adapters/inbound/web/js/views/${name}.js`);
    assert.ok(Object.keys(loaded).length, name);
  }
});
