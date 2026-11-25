// #263: a short connection loss must cost nothing. The microphone's socket
// is driven here with a fake WebSocket, fake timers and a fake capture: it
// drops mid-session, reconnects with backoff, resends from what the server
// acknowledged, pauses when behind, announces audio it had to let go, stops
// for good when the session is over or taken over, and a reloaded page
// reattaches. Also the outbox, the status line and the remembered session.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { LiveAudio } from "../../transcriber/adapters/inbound/web/js/live-audio.js";
import {
  BACKOFF_MAX_MS, Outbox, SESSION_KEY, backoffMs, feedOutcome, finishLine, followUntilEnd, forgetSession, linkLine,
  reattachPlan, rememberSession, rememberedSession, runningSession, sessionPage,
} from "../../transcriber/adapters/inbound/web/js/lib/live-link.js";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import { setLanguage as setLang, t } from "../../transcriber/adapters/inbound/web/js/i18n.js";

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

function fakeSockets() {
  const made = [];
  class FakeSocket {
    constructor(url) {
      this.url = url;
      this.readyState = 0;
      this.bufferedAmount = 0;
      this.sent = [];
      this.listeners = {};
      made.push(this);
    }

    addEventListener(kind, fn) {
      (this.listeners[kind] ??= []).push(fn);
    }

    emit(kind, event = {}) {
      for (const fn of this.listeners[kind] ?? []) fn(event);
    }

    send(data) {
      this.sent.push(data);
    }

    open() {
      this.readyState = 1;
      this.emit("open");
    }

    server(message) {
      this.emit("message", { data: JSON.stringify(message) });
    }

    close() {
      if (this.readyState === 3) return;
      this.readyState = 3;
      this.emit("close", { code: 1006 });
    }

    texts() {
      return this.sent.filter((d) => typeof d === "string").map((d) => JSON.parse(d));
    }

    audio() {
      return this.sent.filter((d) => typeof d !== "string");
    }
  }
  return { FakeSocket, made };
}

function fakeCapture() {
  const box = { onPcm: null, stopped: 0 };
  class FakeCapture {
    constructor({ onPcm }) {
      box.onPcm = onPcm;
    }

    async start() {}

    async stop() {
      box.stopped += 1;
    }
  }
  return { FakeCapture, box };
}

/** One 100 ms piece whose every byte is `n` (so pieces can be told apart). */
const piece = (n) => new Uint8Array(3200).fill(n);

function setup(extra = {}) {
  const clock = new Clock();
  const { FakeSocket, made } = fakeSockets();
  const { FakeCapture, box } = fakeCapture();
  const links = [];
  const problems = [];
  const audio = new LiveAudio({
    id: "entrevista-1", code: "c0de",
    onLink: (l) => links.push(l), onProblem: (k) => problems.push(k),
    deps: {
      openSocket: (url) => new FakeSocket(url), setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout,
      now: clock.now, random: () => 0.5, Capture: FakeCapture, location: { protocol: "https:", host: "x.example" },
    },
    ...extra,
  });
  return { clock, made, box, audio, links, problems };
}

async function started(rig, received = 0) {
  const going = rig.audio.start();
  const socket = rig.made.at(-1);
  socket.open();
  socket.server({ type: "ready", received_samples: received });
  await going;
  return socket;
}

test("the first connection sends the code, then audio only after ready", async () => {
  const rig = setup();
  const going = rig.audio.start();
  const socket = rig.made[0];
  assert.equal(socket.url, "wss://x.example/api/live/entrevista-1/audio");
  socket.open();
  assert.deepEqual(socket.texts(), [{ type: "auth", access_code: "c0de" }]);
  socket.server({ type: "ready", received_samples: 0 });
  await going;
  rig.box.onPcm(piece(1));
  assert.equal(socket.audio().length, 1);
  assert.equal(rig.audio.link.state, "open");
});

test("a dropped socket reconnects with backoff and resends from what the server has", async () => {
  const rig = setup();
  const first = await started(rig);
  for (let n = 1; n <= 5; n += 1) rig.box.onPcm(piece(n));
  assert.equal(first.audio().length, 5);
  first.server({ type: "ack", samples: 3200 }); // pieces 1 and 2 are taken
  assert.equal(rig.audio.outbox.pieces.length, 3);
  first.close(); // the tunnel resets
  assert.equal(rig.audio.link.state, "lost");
  for (let n = 6; n <= 8; n += 1) rig.box.onPcm(piece(n)); // the microphone goes on
  assert.equal(rig.audio.link.pendingSeconds, 0.6);
  rig.clock.advance(499);
  assert.equal(rig.made.length, 1, "waits the first backoff (0.5 s)");
  rig.clock.advance(1);
  const second = rig.made[1];
  second.open();
  second.server({ type: "ready", received_samples: 4800 }); // it had taken piece 3 too
  const resent = second.audio().map((b) => b[0]);
  assert.deepEqual(resent, [4, 5, 6, 7, 8], "nothing lost, nothing twice");
  assert.equal(rig.audio.link.state, "open");
  assert.deepEqual(rig.problems, []);
});

test("a resend that starts inside a piece is cut at the exact sample", async () => {
  const rig = setup();
  const first = await started(rig);
  rig.box.onPcm(piece(1));
  rig.box.onPcm(piece(2));
  first.close();
  rig.clock.advance(500);
  const second = rig.made[1];
  second.open();
  second.server({ type: "ready", received_samples: 2400 }); // half of piece 2
  const [cut] = second.audio();
  assert.equal(cut.length, 1600);
  assert.ok(cut.every((b) => b === 2));
});

test("the backoff doubles from 0.5 s to a cap of 8 s, with jitter, and resets once open", async () => {
  assert.deepEqual([0, 1, 2, 3, 4, 5, 9].map((n) => backoffMs(n, () => 0.5)), [500, 1000, 2000, 4000, 8000, 8000, 8000]);
  assert.equal(backoffMs(0, () => 0), 400);
  assert.equal(backoffMs(10, () => 1), BACKOFF_MAX_MS * 1.2);
  const rig = setup();
  const first = await started(rig);
  first.close();
  const waits = [];
  for (let i = 0; i < 6; i += 1) {
    const before = rig.made.length;
    let waited = 0;
    while (rig.made.length === before) {
      rig.clock.advance(100);
      waited += 100;
    }
    waits.push(waited);
    rig.made.at(-1).close(); // still down
  }
  assert.deepEqual(waits, [500, 1000, 2000, 4000, 8000, 8000]);
  rig.clock.advance(8000);
  const back = rig.made.at(-1);
  back.open();
  back.server({ type: "ready", received_samples: 0 });
  back.close();
  const before = rig.made.length;
  rig.clock.advance(500);
  assert.equal(rig.made.length, before + 1, "the backoff starts again at 0.5 s");
});

test("audio the full outbox let go is announced as a gap, then the rest follows", async () => {
  const rig = setup();
  const first = await started(rig);
  rig.audio.outbox = new Outbox({ maxSeconds: 0.3 });
  first.close();
  for (let n = 1; n <= 6; n += 1) rig.box.onPcm(piece(n)); // 0.6 s, room for 0.3 s
  assert.equal(rig.audio.outbox.lost, 4800);
  rig.clock.advance(500);
  const second = rig.made[1];
  second.open();
  second.server({ type: "ready", received_samples: 0 });
  assert.deepEqual(second.texts().slice(1), [{ type: "gap", samples: 4800 }]);
  assert.deepEqual(second.audio().map((b) => b[0]), [4, 5, 6]);
});

test("when the socket is behind, sending pauses and nothing is dropped", async () => {
  const rig = setup();
  const socket = await started(rig);
  socket.bufferedAmount = 400000;
  for (let n = 1; n <= 4; n += 1) rig.box.onPcm(piece(n));
  assert.equal(socket.audio().length, 0);
  rig.clock.advance(250);
  assert.equal(socket.audio().length, 0, "still behind");
  socket.bufferedAmount = 0;
  rig.clock.advance(250);
  assert.deepEqual(socket.audio().map((b) => b[0]), [1, 2, 3, 4]);
});

test("a session that ended is not retried and stops the microphone", async () => {
  const rig = setup();
  const socket = await started(rig);
  socket.server({ type: "error", code: "wrong_state", message: "ended" });
  socket.close();
  rig.clock.advance(60000);
  assert.equal(rig.made.length, 1);
  assert.deepEqual(rig.problems, ["session_ended"]);
  assert.equal(rig.audio.link.state, "ended");
  assert.equal(rig.box.stopped, 1);
});

test("a socket taken over by another tab, or a refused code, is not retried", async () => {
  for (const [code, kind] of [["replaced", "replaced"], ["unauthorized", "refused"]]) {
    const rig = setup();
    const socket = await started(rig);
    socket.server({ type: "error", code });
    socket.close();
    rig.clock.advance(60000);
    assert.equal(rig.made.length, 1);
    assert.deepEqual(rig.problems, [kind]);
  }
});

test("a first connection refused fails the start; one that drops is retried", async () => {
  const rig = setup();
  const going = rig.audio.start();
  rig.made[0].close(); // network, before ready
  assert.equal(rig.audio.link.state, "connecting");
  rig.clock.advance(500);
  const second = rig.made[1];
  second.open();
  second.server({ type: "error", code: "not_found" });
  second.close();
  await assert.rejects(going, /session_ended/);
});

test("a socket that goes silent while audio waits is closed and reconnected", async () => {
  const rig = setup();
  const socket = await started(rig);
  rig.box.onPcm(piece(1));
  rig.clock.advance(19000);
  assert.equal(socket.readyState, 1);
  rig.clock.advance(6000);
  assert.equal(socket.readyState, 3, "no ack for 20 s: the socket was dead");
  rig.clock.advance(500);
  assert.equal(rig.made.length, 2);
});

test("a reloaded page reattaches: silence for the time missed, then its audio", async () => {
  const rig = setup();
  const socket = await started(rig, 32000);
  assert.deepEqual(socket.texts().slice(1), [{ type: "align", held: 0 }]);
  rig.box.onPcm(piece(9));
  assert.deepEqual(socket.audio().map((b) => b[0]), [9]);
  assert.equal(rig.audio.outbox.pieces[0].offset, 32000);
  socket.server({ type: "ack", samples: 33600 });
  assert.equal(rig.audio.outbox.pieces.length, 0);
});

test("stop sends what is still held, then stop, and never reconnects", async () => {
  const rig = setup();
  const socket = await started(rig);
  socket.bufferedAmount = 400000;
  rig.box.onPcm(piece(1));
  await rig.audio.stop();
  assert.deepEqual(socket.audio().map((b) => b[0]), [1]);
  assert.deepEqual(socket.texts().at(-1), { type: "stop" });
  rig.clock.advance(3000);
  assert.equal(socket.readyState, 3);
  rig.clock.advance(60000);
  assert.equal(rig.made.length, 1);
});

test("the outbox forgets what was acknowledged and counts what waits", () => {
  const box = new Outbox();
  box.push(piece(1));
  box.push(piece(2));
  box.push(new Uint8Array(3)); // a stray byte: whole samples only
  assert.equal(box.end, 3201);
  assert.equal(box.pendingSeconds, 3201 / 16000);
  box.ack(1600);
  assert.equal(box.pieces.length, 2);
  box.ack(100); // an old ack changes nothing
  assert.equal(box.acked, 1600);
  assert.deepEqual(box.from(0).gap, 1600);
  box.ack(Number.NaN);
  assert.equal(box.acked, 1600);
});

test("a stream that answers 404 means the session ended; a network error retries", () => {
  assert.equal(feedOutcome({ status: 404 }), "ended");
  assert.equal(feedOutcome({ code: "not_found" }), "ended");
  assert.equal(feedOutcome({ kind: "unauthorized" }), "unauthorized");
  assert.equal(feedOutcome({ kind: "unavailable", code: "network" }), "retry");
  assert.equal(feedOutcome(null), "retry");
});

test("the stream says when it is open, and gives up after a silence (keepalives count)", async () => {
  const encoder = new TextEncoder();
  let push = null;
  const body = new ReadableStream({ start(controller) { push = controller; } });
  const api = new Api(() => "c", async () => new Response(body, { status: 200 }));
  let opened = 0;
  const going = api.followLive("s", { onItem: () => {}, onOpen: () => { opened += 1; }, idleMs: 60 });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(opened, 1, "open as soon as the stream answers, before any item");
  push.enqueue(encoder.encode(": keepalive\n\n"));
  await new Promise((r) => setTimeout(r, 40));
  push.enqueue(encoder.encode(": keepalive\n\n"));
  assert.equal(await going, "idle", "no byte for 60 ms after the last keepalive");
});


test("the status line: a sentence that changes only with the state, the seconds beside it", () => {
  setLang("es");
  assert.equal(linkLine(), null);
  assert.equal(linkLine({ audio: { state: "connecting", pendingSeconds: 0, everOpen: false } }), null);
  const lost = linkLine({ audio: { state: "lost", pendingSeconds: 12.4, everOpen: true }, feed: true });
  assert.equal(t(lost.key), "Sin conexión: la entrevista se sigue grabando en este navegador");
  assert.equal(t(lost.count, lost.params), "(12 s pendientes)");
  const later = linkLine({ audio: { state: "lost", pendingSeconds: 13.1, everOpen: true } });
  assert.equal(later.key, lost.key, "the live sentence is the same a second later");
  const sending = linkLine({ audio: { state: "open", pendingSeconds: 8, everOpen: true } });
  assert.equal(t(sending.key), "Conexión recuperada: enviando lo pendiente");
  assert.equal(linkLine({ audio: { state: "open", pendingSeconds: 0.4, everOpen: true } }), null);
  assert.deepEqual(linkLine({ feed: true }), { key: "live.reconnecting", count: null, params: {} });
  const ending = finishLine({ state: "open", pendingSeconds: 4.2 });
  assert.equal(t(ending.key), "Enviando lo pendiente antes de terminar…");
  assert.equal(t(ending.button, ending.params), "Terminar ya: se pierden 5 s");
  assert.equal(t(finishLine({ state: "lost", pendingSeconds: 3 }).key),
    "Sin conexión: lo pendiente se enviará en cuanto vuelva la conexión. Puedes esperar o terminar ya.");
  assert.equal(finishLine({ state: "open", pendingSeconds: 0 }), null);
  setLang("en");
  assert.equal(t(lost.key), "No connection: the interview is still being recorded in this browser");
  assert.equal(t(lost.count, lost.params), "(12 s pending)");
  assert.equal(t(sending.key), "Connection restored: sending what is pending");
  setLang("es");
});

test("the running session is remembered with its page and forgotten at its end", () => {
  const data = new Map();
  const store = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v), removeItem: (k) => data.delete(k) };
  assert.equal(rememberedSession(store), null);
  rememberSession({ id: "live-1" }, "mic", "live", store);
  assert.deepEqual(rememberedSession(store), { id: "live-1", audio: "mic", page: "live" });
  forgetSession("other", store);
  assert.deepEqual(rememberedSession(store), { id: "live-1", audio: "mic", page: "live" });
  forgetSession("live-1", store);
  assert.equal(rememberedSession(store), null);
  rememberSession({ id: "voz-1" }, "mic", "voicebot", store);
  assert.equal(rememberedSession(store).page, "voicebot");
  data.set(SESSION_KEY, "{broken");
  assert.equal(rememberedSession(store), null);
  const full = { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("full"); }, removeItem: () => {} };
  rememberSession({ id: "x" }, null, "live", full);
  assert.equal(rememberedSession(full), null);
  assert.equal(rememberedSession(null), null);
});

test("a reload reattaches to a running session, forgets an ended or unknown one, keeps it when unreachable", async () => {
  const kept = { id: "live-1", audio: "mic", page: "live" };
  const running = { id: "live-1", state: "running" };
  assert.deepEqual(await reattachPlan(async () => running, kept), { show: running });
  assert.deepEqual(await reattachPlan(async () => ({ state: "ended" }), kept), { form: true, forget: true, error: null });
  const missing = { status: 404, code: "not_found" };
  assert.deepEqual(await reattachPlan(async () => { throw missing; }, kept), { form: true, forget: true, error: null });
  const down = { kind: "unavailable", code: "network" };
  assert.deepEqual(await reattachPlan(async () => { throw down; }, kept), { form: true, forget: false, error: down });
  const refused = { kind: "unauthorized" };
  assert.deepEqual(await reattachPlan(async () => { throw refused; }, kept), { unauthorized: refused });
});

test("the stream is followed until it ends; a 404 stops the retries; a network error retries", async () => {
  const waits = [];
  const wait = async (ms) => { waits.push(ms); };
  let calls = 0;
  const steps = [() => { throw { kind: "unavailable", code: "network" }; }, () => "closed", () => "idle", () => "end"];
  let retried = 0;
  const done = await followUntilEnd({ follow: async () => steps[calls++](), wait, retryMs: 2000, onRetry: () => { retried += 1; } });
  assert.deepEqual(done, { how: "end" });
  assert.equal(calls, 4);
  assert.deepEqual(waits, [2000, 2000, 2000]);
  assert.equal(retried, 3);
  let tries = 0;
  const gone = await followUntilEnd({ follow: async () => { tries += 1; throw { status: 404, code: "not_found" }; }, wait, retryMs: 1 });
  assert.deepEqual(gone, { how: "ended" });
  assert.equal(tries, 1, "no endless retry after a 404");
  const refused = { kind: "unauthorized" };
  assert.deepEqual(await followUntilEnd({ follow: async () => { throw refused; }, wait, retryMs: 1 }), { how: "unauthorized", error: refused });
  let stop = false;
  const aborted = await followUntilEnd({ follow: async () => { stop = true; return "closed"; }, wait, retryMs: 1, stopped: () => stop });
  assert.deepEqual(aborted, { how: "aborted" });
});

test("which session holds the slot and which page shows it", () => {
  assert.equal(runningSession([{ id: "a", state: "ended" }, { id: "b", state: "running" }]).id, "b");
  assert.equal(runningSession([]), null);
  assert.equal(runningSession(undefined), null);
  assert.equal(sessionPage({ product: "live_assist.interview", mode: "live" }), "live");
  assert.equal(sessionPage({ product: "live_assist.support", mode: "live" }), "live");
  assert.equal(sessionPage({ product: "voicebot.support", mode: "live" }), "voicebot");
  assert.equal(sessionPage({ product: "x", mode: "replay" }), "live");
  assert.equal(sessionPage(null), "live");
});

test("the microphone paused for a while: that time goes as a gap, so the stream stays on the clock", async () => {
  const rig = setup();
  const socket = await started(rig);
  rig.clock.advance(100);
  rig.box.onPcm(piece(1));
  rig.clock.advance(100);
  rig.box.onPcm(piece(2));
  socket.server({ type: "ack", samples: 3200 });
  rig.clock.advance(30000); // the tab was suspended: no samples at all
  rig.clock.advance(100);
  rig.box.onPcm(piece(3));
  const gap = socket.texts().find((m) => m.type === "gap");
  assert.deepEqual(gap, { type: "gap", samples: 30000 * 16 });
  assert.deepEqual(socket.audio().map((b) => b[0]), [1, 2, 3]);
  rig.clock.advance(1500); // a short hiccup is not a pause
  rig.box.onPcm(piece(4));
  assert.equal(socket.texts().filter((m) => m.type === "gap").length, 1);
  assert.equal(rig.audio.link.pendingSeconds, 0.2, "silence is not pending audio");
});

test("after the outbox overflowed, the pending seconds count only what is still held", () => {
  const box = new Outbox({ maxSeconds: 0.3 });
  for (let n = 1; n <= 6; n += 1) box.push(piece(n));
  assert.equal(box.lost, 4800);
  assert.equal(box.pendingSeconds, 0.3);
  box.skip(16000);
  assert.equal(box.pendingSeconds, 0, "a skip larger than the room pushes the audio out; silence is not pending");
  const plain = new Outbox();
  plain.push(piece(1));
  plain.ack(800);
  assert.equal(plain.pendingSeconds, 0.05, "half a piece acknowledged");
});

test("stop while a reconnection is still connecting closes it and leaves nothing running", async () => {
  const rig = setup();
  const first = await started(rig);
  rig.box.onPcm(piece(1));
  first.close();
  rig.clock.advance(500); // socket 2 is connecting
  const second = rig.made[1];
  await rig.audio.stop();
  assert.equal(second.readyState, 3, "closed, not left behind");
  second.open();
  second.server({ type: "ready", received_samples: 0 });
  assert.deepEqual(second.sent, [], "no code, no audio on a socket opened after stop");
  assert.equal(rig.clock.jobs.length, 0, "no timer re-arms itself");
  rig.clock.advance(60000);
  assert.equal(rig.made.length, 2);
});

test("«Terminar» while catching up waits for the server's ack, then says stop", async () => {
  const rig = setup();
  const socket = await started(rig);
  for (let n = 1; n <= 5; n += 1) rig.box.onPcm(piece(n));
  let done = null;
  rig.audio.finish().then((r) => { done = r; });
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(done, null);
  assert.ok(!socket.texts().some((m) => m.type === "stop"), "no stop while audio waits");
  socket.server({ type: "ack", samples: 8000 });
  await Promise.resolve();
  assert.deepEqual(socket.texts().at(-1), { type: "stop" });
  socket.server({ type: "stopped" });
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(done, { lostSeconds: 0 });
});

test("«Terminar» with the socket down waits for it to come back and resends, then stops", async () => {
  const rig = setup();
  const first = await started(rig);
  for (let n = 1; n <= 3; n += 1) rig.box.onPcm(piece(n));
  first.close();
  let done = null;
  rig.audio.finish().then((r) => { done = r; });
  await Promise.resolve();
  rig.clock.advance(120000); // two minutes down: the cap counts connected time only
  assert.equal(done, null);
  assert.equal(rig.audio.link.state, "lost");
  const back = rig.made.at(-1);
  back.open();
  back.server({ type: "ready", received_samples: 1600 });
  assert.deepEqual(back.audio().map((b) => b[0]), [2, 3]);
  back.server({ type: "ack", samples: 4800 });
  await Promise.resolve();
  back.server({ type: "stopped" });
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(done, { lostSeconds: 0 });
});

test("«Terminar ya» while still connecting ends at once and says what was lost", async () => {
  const rig = setup();
  const first = await started(rig);
  for (let n = 1; n <= 10; n += 1) rig.box.onPcm(piece(n));
  first.close();
  rig.clock.advance(500); // reconnecting: socket 2 is connecting
  let done = null;
  rig.audio.finish().then((r) => { done = r; });
  await Promise.resolve();
  rig.audio.endNow();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(done, { lostSeconds: 1 });
  assert.equal(rig.made[1].readyState, 3);
  rig.clock.advance(60000);
  assert.equal(rig.made.length, 2);
});

test("«Terminar» connected but never acknowledged gives up after the cap", async () => {
  const rig = setup();
  const socket = await started(rig);
  rig.box.onPcm(piece(1));
  let done = null;
  rig.audio.finish({ capMs: 1000 }).then((r) => { done = r; });
  await Promise.resolve();
  rig.clock.advance(1000);
  await Promise.resolve();
  assert.deepEqual(socket.texts().at(-1), { type: "stop" });
  rig.clock.advance(5000); // no "stopped": the wait for it ends too
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(done, { lostSeconds: 0.1 });
});

/** A seeded random number generator (mulberry32). */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A server that follows the protocol of transcriber/adapters/inbound/live_ws.py:
 * it takes what reaches it (messages still in flight when a socket drops are
 * lost), counts samples, turns gaps into zeros, acknowledges now and then,
 * and answers ready with what it has.
 */
class ProtocolServer {
  constructor() {
    this.bytes = [];
    this.received = 0;
    this.socket = null;
    this.read = 0;
  }

  attach(socket) {
    this.socket = socket;
    this.read = 0;
    socket.open();
    this.deliver(1); // the code
    socket.server({ type: "ready", received_samples: this.received });
  }

  deliver(count) {
    const socket = this.socket;
    if (!socket || socket.readyState !== 1) return;
    const until = Math.min(this.read + count, socket.sent.length);
    for (; this.read < until; this.read += 1) {
      const data = socket.sent[this.read];
      if (typeof data === "string") {
        const message = JSON.parse(data);
        if (message.type === "gap") this.take(new Uint8Array(message.samples * 2));
        if (message.type === "stop") socket.server({ type: "stopped" });
      } else {
        this.take(data);
      }
    }
  }

  take(bytes) {
    this.bytes.push(bytes);
    this.received += bytes.length / 2;
  }

  ack() {
    if (this.socket?.readyState === 1) this.socket.server({ type: "ack", samples: this.received });
  }

  drop() {
    this.socket?.close(); // whatever was not delivered is lost in the tunnel
    this.socket = null;
  }

  pcm() {
    const out = new Uint8Array(this.bytes.reduce((n, b) => n + b.length, 0));
    let at = 0;
    for (const b of this.bytes) {
      out.set(b, at);
      at += b.length;
    }
    return out;
  }
}

test("property: whatever the drops, pauses and losses in flight, the server ends with exactly the captured PCM", async () => {
  for (let seed = 1; seed <= 60; seed += 1) {
    const random = seeded(seed);
    const rig = setup();
    const server = new ProtocolServer();
    const going = rig.audio.start();
    server.attach(rig.made[0]);
    await going;
    const expected = [];
    let n = 0;
    let paused = 0; // ms the microphone gave nothing; a gap once it gives again
    const ensure = () => {
      if (server.socket && server.socket.readyState !== 3) return;
      const last = rig.made.at(-1);
      if (last.readyState === 0) server.attach(last);
    };
    for (let step = 0; step < 300; step += 1) {
      ensure();
      const r = random();
      if (r < 0.6) {
        rig.clock.advance(100);
        n += 1;
        if (paused && n > 1) expected.push(new Uint8Array(paused * 16 * 2));
        paused = 0;
        const bytes = new Uint8Array(3200);
        for (let i = 0; i < bytes.length; i += 1) bytes[i] = (n * 7 + i) & 0xff;
        rig.box.onPcm(bytes);
        expected.push(bytes);
      } else if (r < 0.8) {
        server.deliver(1 + Math.floor(random() * 4));
      } else if (r < 0.88) {
        server.ack();
      } else if (r < 0.94) {
        server.drop();
        rig.clock.advance(10400); // past any backoff: the retry has fired
        paused += 10400;
        ensure();
      } else {
        const pause = 3000 + Math.floor(random() * 5000); // a suspended tab
        rig.clock.advance(pause);
        paused += pause;
      }
    }
    rig.clock.advance(10400);
    ensure();
    let done = null;
    rig.audio.finish().then((r) => { done = r; });
    for (let i = 0; i < 50 && done === null; i += 1) {
      ensure();
      server.deliver(100000);
      server.ack();
      await new Promise((r) => setTimeout(r, 0));
    }
    assert.deepEqual(done, { lostSeconds: 0 }, `seed ${seed}`);
    const want = new Uint8Array(expected.reduce((k, b) => k + b.length, 0));
    let at = 0;
    for (const b of expected) {
      want.set(b, at);
      at += b.length;
    }
    const got = server.pcm();
    assert.equal(got.length, want.length, `seed ${seed}: length`);
    assert.ok(Buffer.from(got).equals(Buffer.from(want)), `seed ${seed}: same PCM`);
  }
});
