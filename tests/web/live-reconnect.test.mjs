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
  BACKOFF_MAX_MS, Outbox, SESSION_KEY, backoffMs, feedOutcome, forgetSession, linkLine, rememberSession, rememberedSession,
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

test("the status line: lost with its seconds, then sending, then nothing", () => {
  setLang("es");
  assert.equal(linkLine(), null);
  assert.equal(linkLine({ audio: { state: "connecting", pendingSeconds: 0, everOpen: false } }), null);
  const lost = linkLine({ audio: { state: "lost", pendingSeconds: 12.4, everOpen: true }, feed: true });
  assert.deepEqual(lost, { key: "live.link.lost", params: { n: 12 } });
  assert.equal(t(lost.key, lost.params), "Sin conexión: la entrevista se sigue grabando en este navegador (12 s pendientes)");
  const sending = linkLine({ audio: { state: "open", pendingSeconds: 8, everOpen: true } });
  assert.equal(t(sending.key, sending.params), "Conexión recuperada: enviando lo pendiente");
  assert.equal(linkLine({ audio: { state: "open", pendingSeconds: 0.4, everOpen: true } }), null);
  assert.deepEqual(linkLine({ feed: true }), { key: "live.reconnecting", params: {} });
  setLang("en");
  assert.equal(t(lost.key, lost.params), "No connection: the interview is still being recorded in this browser (12 s pending)");
  assert.equal(t(sending.key), "Connection restored: sending what is pending");
  setLang("es");
});

test("a stream that answers 404 means the session ended; a network error retries", () => {
  assert.equal(feedOutcome({ status: 404 }), "ended");
  assert.equal(feedOutcome({ code: "not_found" }), "ended");
  assert.equal(feedOutcome({ kind: "unauthorized" }), "unauthorized");
  assert.equal(feedOutcome({ kind: "unavailable", code: "network" }), "retry");
  assert.equal(feedOutcome(null), "retry");
});

test("the running session is remembered for a reload and forgotten at its end", () => {
  const data = new Map();
  const store = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v), removeItem: (k) => data.delete(k) };
  assert.equal(rememberedSession(store), null);
  rememberSession({ id: "live-1" }, "mic", store);
  assert.deepEqual(rememberedSession(store), { id: "live-1", audio: "mic" });
  forgetSession("other", store);
  assert.deepEqual(rememberedSession(store), { id: "live-1", audio: "mic" });
  forgetSession("live-1", store);
  assert.equal(rememberedSession(store), null);
  data.set(SESSION_KEY, "{broken");
  assert.equal(rememberedSession(store), null);
  const full = { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("full"); }, removeItem: () => {} };
  rememberSession({ id: "x" }, null, full);
  assert.equal(rememberedSession(full), null);
  assert.equal(rememberedSession(null), null);
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

test("the live page wires the calm line, the stream states and the reattach", () => {
  const view = readFileSync(new URL("../../transcriber/adapters/inbound/web/js/views/live.js", import.meta.url), "utf8");
  assert.match(view, /const linkNode = h\("p", \{ class: "notice", role: "status", "aria-live": "polite", hidden: true \}\)/);
  assert.match(view, /onLink: \(link\) => \{\s+audioLink = link;\s+drawLink\(\);/);
  assert.match(view, /idleMs: FEED_IDLE_MS,\s+onOpen: \(\) => \{\s+if \(!reconnecting\) return;\s+reconnecting = false;/);
  assert.match(view, /if \(outcome === "ended"\) break;/);
  assert.match(view, /const kept = sub \? null : rememberedSession\(\);/);
  assert.match(view, /if \(session\.mode === "live"\) rememberSession\(session, audioKind\);/);
  assert.match(view, /forgetSession\(session\.id\);/);
  assert.match(view, /const FEED_IDLE_MS = 40000;/);
});
