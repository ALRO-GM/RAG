// The browser's microphone as the audio of a live session: 16 kHz mono
// PCM s16le sent over the transcriber's audio socket
// (WS /api/live/{id}/audio, docs/live/LIVE_SESSION.md). A browser cannot
// set headers on a socket, so the access code goes in the first message
// ({"type": "auth"}); the server answers {"type": "ready"} with
// received_samples. This is the only module of the pages that opens a
// socket.
//
// A short connection loss costs nothing (#263): every piece of about
// 100 ms goes into an outbox (lib/live-link.js) with its absolute sample
// offset and stays there until the server acknowledges it
// ({"type": "ack", "samples": N}, about every second). When the socket
// closes without a stop the page reconnects with backoff (0.5 → 8 s,
// jitter) until stopped or until the server says the session is over,
// and resends from the server's received_samples. While the socket is
// behind, sending pauses (nothing is dropped); audio the bounded outbox
// had to let go is announced as {"type": "gap"} and becomes silence on
// the server; when the server takes less ({"type": "gap_clamped"}) the
// page shifts its offsets and goes on. A page that reattached after a
// reload sends {"type": "align"} first. When the microphone produced
// nothing for a while (a suspended tab, a laptop asleep; measured on the
// monotonic clock, plus the time it stood still while the computer slept)
// that time goes as a gap too, so the stream stays on the wall clock. Once
// a connection has sent everything held it says {"type": "caught_up"}.
// The microphone is obtained before the socket claims the session's audio,
// so a denied microphone never takes another browser's place. «Terminar» (finish) waits until the
// server has everything before it says stop. The outbox lives in memory;
// when the product allows it (#265) a LocalRecording (lib/live-store.js)
// mirrors what is not acknowledged yet to IndexedDB, so a long outage, a
// closed tab or a «Terminar» without connection loses nothing: it is sent
// after the session and deleted here as soon as the server has it.

import { Capture } from "./capture.js";
import { Outbox, backoffMs } from "./lib/live-link.js";

const PIECE_BYTES = 3200; // 100 ms of 16 kHz 16-bit mono
const MAX_BEHIND = 32000 * 10; // above 10 s waiting in the socket, sending pauses
const PAUSE_MS = 250; // a paused socket is looked at again this often
const READY_TIMEOUT_MS = 30000; // a socket that never says ready is closed and retried
const SILENT_MS = 20000; // no message for this long while audio waits: the socket is dead
const ENDED = new Set(["not_found", "wrong_state"]);
const PAUSE_SAMPLES = 16000 * 2; // the microphone silent for over 2 s of wall time = a pause
const SLEPT_MS = 2000; // the wall clock ran this much past the monotonic one: the computer slept
const SETTLE_MS = 500; // a suspected pause waits this long for a burst of queued audio
const FINISH_CAP_MS = 60000; // «Terminar» waits at most this long while connected
const FINISH_TICK_MS = 250;
const STOPPED_WAIT_MS = 5000; // the server's "stopped" after our stop

/** The socket address of a session's audio, on this page's host. */
export function audioUrl(loc, id) {
  const scheme = loc.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${loc.host}/api/live/${encodeURIComponent(id)}/audio`;
}

export class LiveAudio {
  /**
   * @param {object} options
   *   id         the live session
   *   code       the access code
   *   onLevel    microphone peak, 0..1
   *   onProblem  (kind) "session_ended" | "replaced" | "refused" (the
   *              socket, for good) | "ended" | "suspended" (the microphone)
   *   onLink     ({ state, pendingSeconds, everOpen }) on every change:
   *              state "connecting" | "open" | "lost" | "ended"
   *   echoCancellation  the browser's echo cancellation (voice bot only)
   *   recording  a LocalRecording (#265) that keeps the unacknowledged
   *              audio in this browser, or null
   *   deps       { openSocket, setTimeout, clearTimeout, random, now,
   *              Capture, location } (tests)
   * `muted` (settable) sends silence instead of the microphone, so the
   * transcriber still hears time pass and closes the sentence (push-to-talk).
   */
  constructor({ id, code, onLevel = () => {}, onProblem = () => {}, onLink = () => {}, echoCancellation = false, recording = null, deps = {} }) {
    this.recording = recording;
    this.id = id;
    this.echoCancellation = echoCancellation;
    this.muted = false;
    this.code = code;
    this.onLevel = onLevel;
    this.onProblem = onProblem;
    this.onLink = onLink;
    this.deps = {
      openSocket: (url) => new WebSocket(url),
      setTimeout: (...a) => globalThis.setTimeout(...a),
      clearTimeout: (...a) => globalThis.clearTimeout(...a),
      random: Math.random,
      now: () => Date.now(),
      mono: deps.now ?? (() => globalThis.performance?.now?.() ?? Date.now()),
      Capture,
      location: globalThis.location,
      ...deps,
    };
    this.socket = null;
    this.ready = false; // this socket said ready: audio may go
    this.sendAt = 0; // the offset of the next sample to send on this socket
    this.capture = null;
    this.pending = [];
    this.pendingBytes = 0;
    this.outbox = new Outbox();
    this.stopping = false;
    this.ended = false;
    this.everOpen = false;
    this.attempt = 0;
    this.state = "connecting";
    this.lastHeard = 0;
    this._timers = { retry: null, ready: null, pump: null, watch: null, finish: null, settle: null };
    this._first = null;
    this._clock = null; // { start, slept, captured, mono, wall }: the capture against the clock
    this._hold = null; // { at, timer }: a suspected pause; pieces from `at` wait to be sent
    this.caughtUp = false; // this connection has sent everything held
    this._finish = null; // { resolve, openMs } while «Terminar» waits
    this._stopped = null; // resolves on the server's "stopped"
    this._fresh = true; // nothing reached a server yet: audio before the first ready is dropped
  }

  /** Connect (retrying until a first "ready"), then start the microphone. */
  async start() {
    // The microphone first: if it is refused, no socket has claimed the
    // session's audio from another browser (#263 re-review).
    this.capture = new this.deps.Capture({
      onPcm: (bytes) => this._send(bytes),
      onLevel: this.onLevel,
      onProblem: (kind) => this.onProblem(kind),
      echoCancellation: this.echoCancellation,
    });
    await this.capture.start();
    if (this.stopping) return;
    const first = new Promise((resolve, reject) => {
      this._first = { resolve, reject };
    });
    this._connect();
    await first;
  }

  /** What the page shows of the link. */
  get link() {
    const link = { state: this.state, pendingSeconds: this.outbox.pendingSeconds, everOpen: this.everOpen };
    // #265: what is held is also kept in this browser (nothing is lost by ending now).
    if (this.recording) link.keeping = !this.recording.failed;
    return link;
  }

  _setState(state) {
    this.state = state;
    this.onLink(this.link);
  }

  _clear(name) {
    if (this._timers[name] !== null) this.deps.clearTimeout(this._timers[name]);
    this._timers[name] = null;
  }

  _connect() {
    this._timers.retry = null;
    if (this.stopping || this.ended) return;
    this._open();
  }

  _open() {
    const { openSocket, location: loc } = this.deps;
    let socket;
    try {
      socket = openSocket(audioUrl(loc, this.id));
    } catch {
      this._lost();
      return;
    }
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    this.ready = false;
    this.refusal = null;
    this._clear("ready");
    this._timers.ready = this.deps.setTimeout(() => socket.close(), READY_TIMEOUT_MS);
    socket.addEventListener("open", () => {
      if (this.stopping || socket !== this.socket) return;
      socket.send(JSON.stringify({ type: "auth", access_code: this.code }));
    });
    socket.addEventListener("message", (event) => this._message(socket, event));
    socket.addEventListener("close", () => this._closed(socket));
  }

  _message(socket, event) {
    if (socket !== this.socket) return;
    this.lastHeard = this.deps.now();
    let message = null;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message?.type === "ready") this._ready(socket, message);
    else if (message?.type === "ack") {
      this.outbox.ack(Number(message.samples));
      this.recording?.ack(Number(message.samples));
      this.onLink(this.link);
      this._finishCheck();
    } else if (message?.type === "gap_clamped") this._clamped(message);
    else if (message?.type === "stopped") this._stopped?.();
    else if (message?.type === "error") this.refusal = message.code ?? "refused";
  }

  _ready(socket, message) {
    this._clear("ready");
    if (this.stopping) {
      socket.close();
      return;
    }
    const received = Number(message.received_samples) || 0;
    if (this._fresh && received === 0) {
      // A new session: what the microphone heard while the model loaded
      // is dropped, as before, so the stream starts on time.
      this._clear("settle");
      this._hold = null;
      this._clock = null;
      this.outbox = new Outbox();
    } else {
      this._settle();
    }
    this._fresh = false;
    this.caughtUp = false;
    if (!this.everOpen && received > 0 && this.outbox.acked === 0) {
      // A page that reattached (a reload): its audio goes after what the
      // server has, and the time nobody sent becomes silence.
      this.outbox.rebase(received);
      socket.send(JSON.stringify({ type: "align", held: this.outbox.end - received }));
    }
    this.outbox.ack(received);
    this.sendAt = Math.max(received, 0);
    this.ready = true;
    this.attempt = 0;
    this.everOpen = true;
    this.lastHeard = this.deps.now();
    this._watch();
    this._setState("open");
    this._first?.resolve();
    this._first = null;
    this._pump();
    this._finishCheck();
    this._mirror();
  }

  /** Keep what is not acknowledged in this browser (#265), once the offsets are the server's. */
  _mirror(force = false) {
    if (!this.recording || !this.everOpen) return Promise.resolve();
    return this.recording.mirror(this.outbox, { limit: this._hold?.at ?? Infinity, force });
  }

  /** Close a socket that went silent while audio waits for its ack. */
  _watch() {
    this._clear("watch");
    this._timers.watch = this.deps.setTimeout(() => {
      this._timers.watch = null;
      const socket = this.socket;
      if (!socket || !this.ready) return;
      const waiting = this.outbox.end > this.outbox.acked;
      if (waiting && this.deps.now() - this.lastHeard >= SILENT_MS) {
        socket.close();
        return;
      }
      this._watch();
    }, SILENT_MS / 4);
  }

  _closed(socket) {
    if (socket !== this.socket) return;
    this.socket = null;
    this.ready = false;
    this._clear("ready");
    this._clear("pump");
    this._clear("watch");
    if (this.stopping) return;
    const code = this.refusal;
    if (code) {
      const kind = ENDED.has(code) ? "session_ended" : code === "replaced" ? "replaced" : "refused";
      this._end(kind);
      return;
    }
    this._lost();
  }

  _lost() {
    this._setState(this.everOpen ? "lost" : "connecting");
    this._mirror(true);
    const wait = backoffMs(this.attempt, this.deps.random);
    this.attempt += 1;
    this._timers.retry = this.deps.setTimeout(() => this._connect(), wait);
  }

  _end(kind) {
    this.ended = true;
    for (const name of Object.keys(this._timers)) this._clear(name);
    this._finishDone();
    this._setState("ended");
    this._first?.reject(new Error(kind));
    this._first = null;
    this.onProblem(kind);
    this.capture?.stop().catch?.(() => {});
  }

  _send(bytes) {
    this.pending.push(this.muted ? new Uint8Array(bytes.length) : bytes);
    this.pendingBytes += bytes.length;
    if (this.pendingBytes < PIECE_BYTES) return;
    const piece = new Uint8Array(this.pendingBytes);
    let at = 0;
    for (const part of this.pending) {
      piece.set(part, at);
      at += part.length;
    }
    this.pending = [];
    this.pendingBytes = 0;
    const offset = this.outbox.end;
    this.outbox.push(piece);
    this._keepOnClock(piece.length / 2, offset);
    this._pump();
    this._mirror();
    this.onLink(this.link);
  }

  /**
   * The microphone produced nothing for a while: that time goes as
   * silence. Counted in total: the time due since the first piece (the
   * monotonic clock, plus the time it stood still while the computer
   * slept: the wall clock then ran well past it) against every sample
   * captured or filled. A deficit over 2 s is only suspected at first:
   * a busy page gets the microphone's queued pieces late, in a burst
   * that makes it up. The pieces from there wait `SETTLE_MS`; then what
   * is still missing goes as silence before them (#263, third review).
   */
  _keepOnClock(samples, offset) {
    const mono = this.deps.mono();
    const wall = this.deps.now();
    const clock = this._clock;
    if (!clock) {
      this._clock = { start: mono - samples / 16, slept: 0, captured: samples, mono, wall };
      return;
    }
    const slept = wall - clock.wall - (mono - clock.mono);
    if (slept > SLEPT_MS) clock.slept += slept;
    clock.mono = mono;
    clock.wall = wall;
    clock.captured += samples;
    if (this._hold !== null && this._deficit() <= PAUSE_SAMPLES) {
      this._settle(); // the burst made it up: no pause, nothing to fill
    } else if (this._hold === null && this._deficit() > PAUSE_SAMPLES) {
      this._hold = { at: offset };
      this._timers.settle = this.deps.setTimeout(() => {
        this._timers.settle = null;
        this._settle();
      }, SETTLE_MS);
    }
  }

  /** Samples the capture is behind the clock, as of the last piece. */
  _deficit() {
    const clock = this._clock;
    if (!clock) return 0;
    const due = Math.round((clock.mono - clock.start + clock.slept) * 16);
    return due - clock.captured;
  }

  /** Decide a suspected pause: what is still missing becomes silence. */
  _settle() {
    const hold = this._hold;
    if (!hold) return;
    this._clear("settle");
    this._hold = null;
    const missing = this._deficit();
    if (missing > PAUSE_SAMPLES) {
      this.outbox.insertSilence(hold.at, missing);
      this._clock.captured += missing;
    }
    this._pump();
    this.onLink(this.link);
  }

  /** The server put in less silence than asked: shift what follows. */
  _clamped({ at, asked, samples }) {
    if (![at, asked, samples].every(Number.isFinite)) return;
    const delta = this.outbox.clamp(at, asked, samples);
    this.recording?.shift(at + asked, delta);
    if (this.sendAt >= at + asked) this.sendAt += delta;
    this.onLink(this.link);
  }

  /** Send what the server does not have yet, unless the socket is behind. */
  _pump(force = false) {
    const socket = this.socket;
    if (!socket || !this.ready || socket.readyState !== 1) return;
    const { gap, pieces } = this.outbox.from(this.sendAt);
    const holdAt = this._hold?.at ?? Infinity; // a suspected pause is not sent yet
    if (gap > 0) {
      // Audio the outbox let go becomes silence there: its copy here stays for the recovery (#265).
      this.recording?.keep(this.sendAt, this.sendAt + gap);
      socket.send(JSON.stringify({ type: "gap", samples: gap }));
      this.sendAt += gap;
    }
    for (const piece of pieces) {
      if (piece.offset >= holdAt) return;
      if (!piece.bytes) {
        socket.send(JSON.stringify({ type: "gap", samples: piece.samples }));
        this.sendAt = piece.offset + piece.samples;
        continue;
      }
      if (!force && socket.bufferedAmount > MAX_BEHIND) {
        // Behind: pause (the outbox keeps the audio) and look again soon.
        if (this._timers.pump === null) {
          this._timers.pump = this.deps.setTimeout(() => {
            this._timers.pump = null;
            this._pump();
          }, PAUSE_MS);
        }
        return;
      }
      socket.send(piece.bytes);
      this.sendAt = piece.offset + piece.samples;
    }
    if (!this.caughtUp && this.sendAt >= this.outbox.end) {
      // Everything held is sent: from now on the audio is live (#263 re-review).
      this.caughtUp = true;
      socket.send(JSON.stringify({ type: "caught_up" }));
    }
  }

  /**
   * «Terminar» (#263): stop the microphone, then wait until the server has
   * everything still held (reconnecting meanwhile), at most `capMs` of
   * connected time; `endNow()` stops waiting. Then say stop and wait for
   * the server's "stopped". Resolves { lostSeconds, keptSeconds }: what
   * never arrived, and of it what stays in this browser to be sent after
   * the session (#265; then nothing is lost).
   */
  async finish({ capMs = FINISH_CAP_MS } = {}) {
    try {
      await this.capture?.stop();
    } catch {
      // already stopped
    }
    this._flush();
    this._settle();
    if (this.recording) await this._mirror(true);
    if (!this.ended && this.everOpen && this.outbox.pendingSeconds > 0) {
      await new Promise((resolve) => {
        this._finish = { resolve, openMs: 0, capMs };
        this._finishTick();
        this._finishCheck();
      });
    }
    let lostSeconds = this.outbox.pendingSeconds;
    if (!this.recording) {
      await this._sayStop();
      return { lostSeconds };
    }
    let keptSeconds = 0;
    if (lostSeconds > 0) {
      this._settle();
      await this._mirror(true);
      if (!this.recording.failed) {
        keptSeconds = lostSeconds;
        lostSeconds = 0;
        if (this.state !== "open") await this.recording.endLocally(this.deps.now());
      }
    }
    await this._sayStop();
    return { lostSeconds, keptSeconds };
  }

  /** Write what is held to this browser now (the tab is being hidden or closed). */
  keepNow() {
    return this._mirror(true);
  }

  /** Stop waiting in `finish()`: what is still held is lost (or kept here, #265). */
  endNow() {
    this._finishDone();
  }

  /** The piece being filled goes to the outbox as it is. */
  _flush() {
    if (!this.pendingBytes) return;
    const piece = new Uint8Array(this.pendingBytes);
    let at = 0;
    for (const part of this.pending) {
      piece.set(part, at);
      at += part.length;
    }
    this.pending = [];
    this.pendingBytes = 0;
    this.outbox.push(piece);
    this._pump();
  }

  _finishTick() {
    this._timers.finish = this.deps.setTimeout(() => {
      this._timers.finish = null;
      const wait = this._finish;
      if (!wait) return;
      if (this.state === "open") wait.openMs += FINISH_TICK_MS;
      this.onLink(this.link);
      if (wait.openMs >= wait.capMs) this._finishDone();
      else this._finishTick();
    }, FINISH_TICK_MS);
  }

  _finishCheck() {
    if (this._finish && this.outbox.pendingSeconds <= 0) this._finishDone();
  }

  _finishDone() {
    const wait = this._finish;
    this._finish = null;
    this._clear("finish");
    wait?.resolve();
  }

  async _sayStop() {
    const socket = this.socket;
    this.stopping = true;
    for (const name of ["retry", "pump", "watch", "ready", "finish"]) this._clear(name);
    if (socket && socket.readyState === 1 && this.ready) {
      this._pump(true);
      let timer = null;
      const stopped = new Promise((resolve) => {
        this._stopped = resolve;
        timer = this.deps.setTimeout(resolve, STOPPED_WAIT_MS);
      });
      socket.send(JSON.stringify({ type: "stop" }));
      await stopped;
      this.deps.clearTimeout(timer);
      this._stopped = null;
    }
    socket?.close();
    this._first?.resolve();
    this._first = null;
  }

  /** Stop the microphone and end the session's audio at once ({"type": "stop"}). */
  async stop() {
    try {
      await this.capture?.stop();
    } catch {
      // already stopped
    }
    this._finishDone();
    const socket = this.socket;
    if (socket && socket.readyState === 1 && this.ready) {
      this._pump(true); // what is still held goes before the stop
      socket.send(JSON.stringify({ type: "stop" }));
      this.deps.setTimeout(() => socket.close(), 3000);
    } else {
      socket?.close(); // connecting, or not ready: never left behind
    }
    this.stopping = true;
    for (const name of ["retry", "pump", "watch", "ready", "finish"]) this._clear(name);
    this._first?.resolve(); // a start still connecting ends quietly
    this._first = null;
  }
}
