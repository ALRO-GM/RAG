// Pure state machine of the chunk upload: numbered chunks, one at a time
// and in order, retried with back-off, resynchronised with the server.
// It never sends anything: the caller asks next(), sends, and reports.

import { concatBytes } from "./audio.js";

export const SECONDS_PER_CHUNK = 5;
export const BYTES_PER_SECOND = 16000 * 2;
export const DEFAULT_CHUNK_BYTES = SECONDS_PER_CHUNK * BYTES_PER_SECOND;
export const DEFAULT_WARN_BYTES = 300 * BYTES_PER_SECOND; // 5 minutes

export class ChunkQueue {
  constructor(options = {}) {
    const size = options.chunkBytes ?? DEFAULT_CHUNK_BYTES;
    this.chunkBytes = Math.max(2, size - (size % 2));
    this.backoffBase = options.backoffBaseMs ?? 1000;
    this.backoffMax = options.backoffMaxMs ?? 30000;
    this.warnBytes = options.warnBytes ?? DEFAULT_WARN_BYTES;
    this.nextNumber = options.firstNumber ?? 0; // number of the next chunk to cut
    this.acked = this.nextNumber; // chunks [0, acked) are on the server
    this.queue = []; // sealed chunks not yet acknowledged, in order
    this.partial = [];
    this.partialBytes = 0;
    this.inFlight = null;
    this.fatal = null; // string reason once nothing can go on
  }

  /** Add PCM bytes (an even count); whole chunks are sealed as they fill. */
  push(bytes) {
    if (bytes.length % 2 !== 0) throw new RangeError("PCM needs an even number of bytes");
    if (bytes.length === 0) return;
    this.partial.push(bytes);
    this.partialBytes += bytes.length;
    if (this.partialBytes < this.chunkBytes) return;
    const all = concatBytes(this.partial);
    let at = 0;
    while (all.length - at >= this.chunkBytes) {
      this._seal(all.slice(at, at + this.chunkBytes));
      at += this.chunkBytes;
    }
    const rest = all.slice(at);
    this.partial = rest.length ? [rest] : [];
    this.partialBytes = rest.length;
  }

  /** Seal what is left as a last, shorter chunk (at the end of the recording). */
  finish() {
    if (this.partialBytes === 0) return;
    this._seal(concatBytes(this.partial));
    this.partial = [];
    this.partialBytes = 0;
  }

  _seal(data) {
    this.queue.push({ number: this.nextNumber, data, attempts: 0, readyAt: 0 });
    this.nextNumber += 1;
  }

  /** The chunk to send now, or null (busy, backing off, empty, fatal). */
  next(now) {
    if (this.fatal || this.inFlight !== null || this.queue.length === 0) return null;
    const head = this.queue[0];
    if (head.readyAt > now) return null;
    this.inFlight = head.number;
    return { number: head.number, data: head.data, attempt: head.attempts + 1 };
  }

  /** Milliseconds until next() may give a chunk; null when there is none. */
  waitMs(now) {
    if (this.fatal || this.inFlight !== null || this.queue.length === 0) return null;
    return Math.max(0, this.queue[0].readyAt - now);
  }

  /** The chunk was stored (or the server already had it as its last). */
  sent(number) {
    if (this.inFlight !== number || this.queue[0]?.number !== number) {
      throw new Error(`chunk ${number} is not the one in flight`);
    }
    this.queue.shift();
    this.inFlight = null;
    this.acked = number + 1;
  }

  /** The upload failed; keep the chunk and wait before the same number again. */
  failed(number, now, minDelayMs = 0) {
    const head = this.queue[0];
    if (this.inFlight !== number || !head || head.number !== number) {
      throw new Error(`chunk ${number} is not the one in flight`);
    }
    head.attempts += 1;
    const backoff = Math.min(this.backoffMax, this.backoffBase * 2 ** (head.attempts - 1));
    const delay = Math.max(backoff, minDelayMs);
    head.readyAt = now + delay;
    this.inFlight = null;
    return delay;
  }

  /**
   * Align with the number of chunks the server holds. Chunks it has are
   * dropped from the queue; a chunk it lacks that we no longer have makes
   * the recording unrecoverable (fatal) because numbers may not be skipped.
   * The queue always starts at `acked`, so what remains is a server behind
   * ("lost") or ahead ("ahead") of us.
   */
  resync(serverChunks) {
    this.inFlight = null;
    if (serverChunks > this.nextNumber) {
      this.fatal = "ahead";
      return false;
    }
    while (this.queue.length && this.queue[0].number < serverChunks) this.queue.shift();
    if (serverChunks < this.acked) {
      this.fatal = "lost";
      return false;
    }
    this.acked = Math.max(this.acked, serverChunks);
    if (this.queue.length) {
      this.queue[0].readyAt = 0;
      this.queue[0].attempts = 0;
    }
    return true;
  }

  /** Forget the waiting time (the network came back): the head may go now. */
  retryNow() {
    if (this.queue.length) this.queue[0].readyAt = 0;
  }

  get pendingChunks() {
    return this.queue.length;
  }

  get pendingBytes() {
    let n = this.partialBytes;
    for (const c of this.queue) n += c.data.length;
    return n;
  }

  get pendingSeconds() {
    return this.pendingBytes / BYTES_PER_SECOND;
  }

  get isLarge() {
    return this.pendingBytes > this.warnBytes;
  }

  /** Everything handed in has been acknowledged. */
  get drained() {
    return this.queue.length === 0 && this.partialBytes === 0 && this.inFlight === null;
  }
}
