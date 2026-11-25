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
// the server. A page that reattached after a reload sends
// {"type": "align"} first. The outbox lives only in memory: the audio is
// never stored.

import { Capture } from "./capture.js";
import { Outbox, backoffMs } from "./lib/live-link.js";

const PIECE_BYTES = 3200; // 100 ms of 16 kHz 16-bit mono
const MAX_BEHIND = 32000 * 10; // above 10 s waiting in the socket, sending pauses
const PAUSE_MS = 250; // a paused socket is looked at again this often
const READY_TIMEOUT_MS = 30000; // a socket that never says ready is closed and retried
const SILENT_MS = 20000; // no message for this long while audio waits: the socket is dead
const ENDED = new Set(["not_found", "wrong_state"]);

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
   *   deps       { WebSocket, setTimeout, clearTimeout, random, now,
   *              Capture, location } (tests)
   * `muted` (settable) sends silence instead of the microphone, so the
   * transcriber still hears time pass and closes the sentence (push-to-talk).
   */
  constructor({ id, code, onLevel = () => {}, onProblem = () => {}, onLink = () => {}, echoCancellation = false, deps = {} }) {
    this.id = id;
    this.echoCancellation = echoCancellation;
    this.muted = false;
    this.code = code;
    this.onLevel = onLevel;
    this.onProblem = onProblem;
    this.onLink = onLink;
    this.deps = {
      WebSocket: globalThis.WebSocket,
      setTimeout: (...a) => globalThis.setTimeout(...a),
      clearTimeout: (...a) => globalThis.clearTimeout(...a),
      random: Math.random,
      now: () => Date.now(),
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
    this._timers = { retry: null, ready: null, pump: null, watch: null };
    this._first = null;
  }

  /** Connect (retrying until a first "ready"), then start the microphone. */
  async start() {
    const first = new Promise((resolve, reject) => {
      this._first = { resolve, reject };
    });
    this._connect();
    await first;
    if (this.stopping) return;
    this.capture = new this.deps.Capture({
      onPcm: (bytes) => this._send(bytes),
      onLevel: this.onLevel,
      onProblem: (kind) => this.onProblem(kind),
      echoCancellation: this.echoCancellation,
    });
    await this.capture.start();
  }

  /** What the page shows of the link. */
  get link() {
    return { state: this.state, pendingSeconds: this.outbox.pendingSeconds, everOpen: this.everOpen };
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
    const { WebSocket: Socket, location: loc } = this.deps;
    let socket;
    try {
      socket = new Socket(audioUrl(loc, this.id));
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
    socket.addEventListener("open", () => socket.send(JSON.stringify({ type: "auth", access_code: this.code })));
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
      this.onLink(this.link);
    } else if (message?.type === "error") this.refusal = message.code ?? "refused";
  }

  _ready(socket, message) {
    this._clear("ready");
    const received = Number(message.received_samples) || 0;
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
    const wait = backoffMs(this.attempt, this.deps.random);
    this.attempt += 1;
    this._timers.retry = this.deps.setTimeout(() => this._connect(), wait);
  }

  _end(kind) {
    this.ended = true;
    for (const name of Object.keys(this._timers)) this._clear(name);
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
    this.outbox.push(piece);
    this._pump();
    this.onLink(this.link);
  }

  /** Send what the server does not have yet, unless the socket is behind. */
  _pump(force = false) {
    const socket = this.socket;
    if (!socket || !this.ready || socket.readyState !== 1) return;
    const { gap, pieces } = this.outbox.from(this.sendAt);
    if (gap > 0) {
      socket.send(JSON.stringify({ type: "gap", samples: gap }));
      this.sendAt += gap;
    }
    for (const piece of pieces) {
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
  }

  /** Stop the microphone and end the session's audio ({"type": "stop"}). */
  async stop() {
    try {
      await this.capture?.stop();
    } catch {
      // already stopped
    }
    const socket = this.socket;
    if (socket && socket.readyState === 1 && this.ready) {
      this._pump(true); // what is still held goes before the stop
      socket.send(JSON.stringify({ type: "stop" }));
      this.stopping = true;
      this.deps.setTimeout(() => socket.close(), 3000);
    }
    this.stopping = true;
    for (const name of ["retry", "pump", "watch", "ready"]) this._clear(name);
    this._first?.resolve(); // a start still connecting ends quietly
    this._first = null;
  }
}
