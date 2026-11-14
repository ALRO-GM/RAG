// Drives a ChunkQueue against the API: one chunk at a time, in order,
// retries with back-off, resync on doubt. No DOM; time and timers are
// injected so it is tested with fakes.

export class ChunkUploader {
  /**
   * @param {object} api     needs putChunk(id, n, bytes) and getSession(id)
   * @param {string} sessionId
   * @param {import("./chunks.js").ChunkQueue} queue
   * @param {object} [env]   { now, setTimer, clearTimer, onState }
   */
  constructor(api, sessionId, queue, env = {}) {
    this.api = api;
    this.sessionId = sessionId;
    this.queue = queue;
    this.now = env.now ?? (() => Date.now());
    this.setTimer = env.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
    this.clearTimer = env.clearTimer ?? ((h) => clearTimeout(h));
    this.onState = env.onState ?? (() => {});
    this.timer = null;
    this.running = false;
    this.lastError = null; // kind of the last failure, null once a chunk went through
    this.waiting = []; // resolvers of whenDone()
    this.rejectedOnce = false;
  }

  /** Hand over PCM bytes (even count) and start sending. */
  push(bytes) {
    this.queue.push(bytes);
    this.kick();
  }

  /** The recording ended: seal the last chunk and send everything. */
  finish() {
    this.queue.finish();
    this.kick();
  }

  /** The network is back: try now instead of waiting out the back-off. */
  retryNow() {
    this.queue.retryNow();
    this.kick();
  }

  /** Resolves when everything is acknowledged, or with the fatal reason. */
  whenDone() {
    return new Promise((resolve) => {
      this.waiting.push(resolve);
      this._emit();
    });
  }

  state() {
    return {
      pendingBytes: this.queue.pendingBytes,
      pendingSeconds: this.queue.pendingSeconds,
      pendingChunks: this.queue.pendingChunks,
      large: this.queue.isLarge,
      lastError: this.lastError,
      fatal: this.queue.fatal,
      drained: this.queue.drained,
    };
  }

  kick() {
    if (this.timer !== null) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    this._pump();
  }

  _emit() {
    const state = this.state();
    this.onState(state);
    if (state.fatal || state.drained) {
      const done = this.waiting;
      this.waiting = [];
      for (const resolve of done) resolve(state.fatal);
    }
  }

  async _pump() {
    if (this.running) return;
    this.running = true;
    try {
      for (;;) {
        const now = this.now();
        const chunk = this.queue.next(now);
        if (!chunk) break;
        try {
          await this.api.putChunk(this.sessionId, chunk.number, chunk.data);
          this.queue.sent(chunk.number);
          this.lastError = null;
          this.rejectedOnce = false;
          this._emit();
        } catch (error) {
          if (!(await this._handle(chunk, error))) break;
        }
      }
    } finally {
      this.running = false;
      this._schedule();
      this._emit();
    }
  }

  /** Decide what a failure means; false stops the loop. */
  async _handle(chunk, error) {
    const kind = error?.kind ?? "unavailable";
    this.lastError = kind;
    if (kind === "unauthorized") {
      this.queue.fatal = "unauthorized";
      return false;
    }
    if (kind === "client") {
      // The server disagrees with our numbering (or refused the chunk):
      // read what it holds and align, once; a second refusal is final.
      if (this.rejectedOnce) {
        this.queue.fatal = "rejected";
        return false;
      }
      this.rejectedOnce = true;
      try {
        const view = await this.api.getSession(this.sessionId);
        this.queue.resync(view.chunks);
        return !this.queue.fatal;
      } catch (again) {
        this.queue.failed(chunk.number, this.now());
        this.lastError = again?.kind ?? "unavailable";
        return false;
      }
    }
    const minimum = kind === "rate" ? (error.waitSeconds ?? 0) * 1000 : 0;
    this.queue.failed(chunk.number, this.now(), minimum);
    this._emit();
    return false;
  }

  _schedule() {
    if (this.timer !== null) return;
    const wait = this.queue.waitMs(this.now());
    if (wait === null) return;
    this.timer = this.setTimer(() => {
      this.timer = null;
      this._pump();
    }, wait);
  }
}
