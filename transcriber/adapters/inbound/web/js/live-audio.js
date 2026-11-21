// The browser's microphone as the audio of a live session: 16 kHz mono
// PCM s16le sent over the transcriber's audio socket
// (WS /api/live/{id}/audio, docs/live/LIVE_SESSION.md). A browser cannot
// set headers on a socket, so the access code goes in the first message
// ({"type": "auth"}); the server answers {"type": "ready"}. Audio is sent
// in pieces of about 100 ms; when the socket falls behind (the
// transcriber's queue is full) pieces wait in the browser, up to
// MAX_BEHIND bytes, then the oldest are dropped. This is the only module
// of the pages that opens a socket.

import { Capture } from "./capture.js";

const PIECE_BYTES = 3200; // 100 ms of 16 kHz 16-bit mono
const MAX_BEHIND = 32000 * 10; // 10 s waiting in the socket

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
   *   onProblem  (kind) "socket" | "refused" | "ended" | "suspended"
   */
  constructor({ id, code, onLevel = () => {}, onProblem = () => {} }) {
    this.id = id;
    this.code = code;
    this.onLevel = onLevel;
    this.onProblem = onProblem;
    this.socket = null;
    this.capture = null;
    this.pending = [];
    this.pendingBytes = 0;
    this.stopping = false;
  }

  /** Open the socket, wait for "ready", then start the microphone. */
  async start() {
    const socket = new WebSocket(audioUrl(location, this.id));
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("timeout")), 30000);
      socket.addEventListener("open", () => socket.send(JSON.stringify({ type: "auth", access_code: this.code })));
      socket.addEventListener("message", (event) => {
        let message = null;
        try {
          message = JSON.parse(event.data);
        } catch {
          return;
        }
        if (message?.type === "ready") {
          clearTimeout(timer);
          resolve();
        } else if (message?.type === "error") {
          clearTimeout(timer);
          reject(new Error(message.code ?? "refused"));
        }
      });
      socket.addEventListener("close", () => {
        clearTimeout(timer);
        reject(new Error("closed"));
        if (!this.stopping) this.onProblem("socket");
      });
    });
    this.capture = new Capture({
      onPcm: (bytes) => this._send(bytes),
      onLevel: this.onLevel,
      onProblem: (kind) => this.onProblem(kind),
    });
    await this.capture.start();
  }

  _send(bytes) {
    this.pending.push(bytes);
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
    const socket = this.socket;
    if (!socket || socket.readyState !== 1) return;
    if (socket.bufferedAmount > MAX_BEHIND) return; // drop: the transcriber is behind
    socket.send(piece);
  }

  /** Stop the microphone and end the session's audio ({"type": "stop"}). */
  async stop() {
    this.stopping = true;
    try {
      await this.capture?.stop();
    } catch {
      // already stopped
    }
    const socket = this.socket;
    if (socket && socket.readyState === 1) {
      socket.send(JSON.stringify({ type: "stop" }));
      setTimeout(() => socket.close(), 3000);
    }
  }
}
