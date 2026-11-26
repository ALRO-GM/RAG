// «Compartir pantalla» (#278): the pure parts, tested without a browser.
//
// - randomPin / validPin / validName / deviceLabel: what the forms propose
//   and accept (the server checks the same).
// - ShareLink: the socket to the relay (/api/share) with the access code
//   as first message, then host / join, and — after a dropped link — resume
//   with the token the server gave, with a backoff like #263's but capped
//   at BACKOFF_MAX_MS = 1 s (a presentation cannot wait 8 s; a retry costs
//   one small message):
//   a tunnel reset of a few seconds resynchronises by itself, nobody types
//   the password again. A ping every PING_MS; no message for SILENT_MS means
//   the link is dead even if the socket did not say so (a tunnel can do
//   that) and it is reopened. The socket, the timers and the clock are
//   injected.
// - statusKey: the one line each device shows about the link.

import { backoffMs } from "./live-link.js";

export const BACKOFF_MAX_MS = 1000;

/** The wait before reconnection `attempt` (0, 1, …): 0.5, 1, 1, 1… s, ±20 % jitter. */
export function shareBackoffMs(attempt, random = Math.random) {
  return Math.min(backoffMs(attempt, random), Math.round(BACKOFF_MAX_MS * (0.8 + 0.4 * random())));
}

export const PIN_DIGITS = 4;
export const NAME_MAX = 40;
export const LABEL_MAX = 40;
export const PING_MS = 15000;
export const SILENT_MS = 40000;
/** Close codes after which reconnecting cannot help. */
export const FINAL_CODES = new Set([4400, 4401, 4403, 4404, 4409, 4410, 4413, 4429]);

/** A random 4-digit password ("0000"–"9999"), from crypto when there is one. */
export function randomPin(random = null) {
  let n;
  if (random) n = Math.floor(random() * 10000);
  else {
    const buf = new Uint32Array(1);
    globalThis.crypto.getRandomValues(buf);
    n = buf[0] % 10000;
  }
  return String(Math.min(Math.max(n, 0), 9999)).padStart(PIN_DIGITS, "0");
}

/** Whether `text` is exactly four ASCII digits. */
export function validPin(text) {
  return /^[0-9]{4}$/.test(String(text ?? ""));
}

/** Whether `text` is a name the server accepts (letters, digits, spaces, a few signs). */
export function validName(text) {
  const name = String(text ?? "").trim().split(/\s+/).join(" ");
  return name.length > 0 && name.length <= NAME_MAX && /^[\p{L}\p{N}_ .,'()&+-]+$/u.test(name);
}

/** A short name for this device from its user agent: the kind ("tablet" | "phone" | "computer") for t(). */
export function deviceKind(userAgent = "", maxTouchPoints = 0) {
  const ua = String(userAgent);
  if (/iPad|Tablet|Android(?!.*Mobile)/i.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1)) return "tablet";
  if (/Mobi|iPhone|Android/i.test(ua)) return "phone";
  return "computer";
}

/** Trim a device's name to what the server keeps. */
export function cleanLabel(text) {
  return String(text ?? "").replace(/[\x00-\x1f\x7f]/g, " ").trim().split(/\s+/).join(" ").slice(0, LABEL_MAX);
}

/**
 * The status line of a device, or null when all is well:
 * { key } for t(). `state` is the link's, `hostOnline` whether the sharing
 * device is connected (seen from a joined device).
 */
export function statusKey({ state, host, hostOnline = true }) {
  if (state === "connecting") return "share.status.connecting";
  if (state === "lost") return "share.status.lost";
  if (!host && !hostOnline) return "share.status.host_away";
  return null;
}

/** The i18n key of a refusal (the server's error code). */
export function refusalKey(code) {
  const known = ["no_match", "too_many_attempts", "full", "busy", "name_taken", "invalid_input", "ended", "unauthorized"];
  return `share.error.${known.includes(code) ? code : "other"}`;
}

/** The i18n key of why a shared session ended for this device. */
export function endedKey(reason) {
  const known = ["stopped", "removed", "expired", "host_left", "too_many_attempts", "left", "gone", "replaced"];
  return `share.ended.${known.includes(reason) ? reason : "other"}`;
}

/** A socket address (ws: or wss:) on this origin. */
export function socketUrl(loc, path = "/api/share") {
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}${path}`;
}

/**
 * The link of one device to the relay.
 *   hello     { type: "host" | "join", name, pin, label } (the first time)
 *   onMessage (message) for everything but welcome / error / pong
 *   onState   (state, detail): "connecting" | "open" | "lost" | "ended"
 *             (detail: { code, retryAfter } of a refusal or { reason })
 */
export class ShareLink {
  constructor({ url, code, hello, openSocket, onMessage = () => {}, onState = () => {}, timers = globalThis, now = () => Date.now(), random = Math.random }) {
    this.url = url;
    this.code = code;
    this.hello = hello;
    this.openSocket = openSocket;
    this.onMessage = onMessage;
    this.onState = onState;
    this.timers = timers;
    this.now = now;
    this.random = random;
    this.token = null;
    this.device = null;
    this.attempt = 0;
    this.state = "idle";
    this.socket = null;
    this.stopped = false;
    this.lastHeard = 0;
    this.pinger = null;
    this.retry = null;
    this.sent = 0; // bytes sent, for the measurements of docs/web/SCREEN_SHARE.md
    this.received = 0;
  }

  _set(state, detail = null) {
    this.state = state;
    this.onState(state, detail);
  }

  start() {
    this.stopped = false;
    this._connect();
  }

  _connect() {
    if (this.stopped) return;
    this._set("connecting");
    let socket;
    try {
      socket = this.openSocket(this.url);
    } catch {
      this._lost();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (socket !== this.socket) return;
      this.lastHeard = this.now();
      this._raw({ type: "auth", access_code: this.code });
      this._raw(this.token ? { type: "resume", token: this.token } : this.hello);
    };
    socket.onmessage = (event) => {
      if (socket !== this.socket) return;
      this.lastHeard = this.now();
      const text = String(event.data);
      this.received += text.length;
      let message;
      try {
        message = JSON.parse(text);
      } catch {
        return;
      }
      this._message(message);
    };
    socket.onclose = (event) => {
      if (socket !== this.socket) return;
      this.socket = null;
      this.timers.clearInterval(this.pinger);
      if (this.stopped || this.state === "ended") return;
      if (FINAL_CODES.has(event?.code)) {
        if (this.state !== "ended") this._set("ended", { code: this.refusal?.code ?? "ended", retryAfter: this.refusal?.retry_after ?? 0, reason: this.reason ?? (event.code === 4409 ? "replaced" : null) });
        return;
      }
      this._lost();
    };
    socket.onerror = () => {};
  }

  _message(message) {
    if (message.type === "welcome") {
      if (message.token) this.token = message.token;
      this.device = message.device;
      this.attempt = 0;
      this.refusal = null;
      this._set("open", message);
      this.timers.clearInterval(this.pinger);
      this.pinger = this.timers.setInterval(() => this._ping(), PING_MS);
      return;
    }
    if (message.type === "pong") return;
    if (message.type === "error") {
      this.refusal = message;
      return;
    }
    if (message.type === "ended") this.reason = message.reason;
    this.onMessage(message);
  }

  _ping() {
    if (this.now() - this.lastHeard > SILENT_MS) {
      // The tunnel may have dropped without closing: start again.
      const dead = this.socket;
      this.socket = null;
      this.timers.clearInterval(this.pinger);
      try {
        dead?.close();
      } catch {
        // already gone
      }
      this._lost();
      return;
    }
    this._raw({ type: "ping" });
  }

  _lost() {
    if (this.stopped) return;
    this._set("lost");
    this.timers.clearTimeout(this.retry);
    this.retry = this.timers.setTimeout(() => this._connect(), shareBackoffMs(this.attempt++, this.random));
  }

  _raw(message) {
    const text = JSON.stringify(message);
    try {
      this.socket?.send(text);
      this.sent += text.length;
      return true;
    } catch {
      return false;
    }
  }

  /** Send `message` when the link is open; false otherwise (nothing is queued). */
  send(message) {
    if (this.state !== "open" || !this.socket || this.socket.readyState !== 1) return false;
    return this._raw(message);
  }

  /** Leave for good: tell the relay, close, never reconnect. */
  stop() {
    if (this.state === "open") this.send({ type: "leave" });
    this.stopped = true;
    this.timers.clearInterval(this.pinger);
    this.timers.clearTimeout(this.retry);
    const socket = this.socket;
    this.socket = null;
    try {
      socket?.close(1000);
    } catch {
      // already closed
    }
    this.state = "ended";
  }
}
