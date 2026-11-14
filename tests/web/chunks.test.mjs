import test from "node:test";
import assert from "node:assert/strict";
import { ChunkQueue, DEFAULT_MAX_BYTES } from "../../transcriber/adapters/inbound/web/js/lib/chunks.js";

const bytes = (n, v = 1) => new Uint8Array(n).fill(v);

test("default chunks are 5 s of 16 kHz 16-bit audio, always even", () => {
  assert.equal(new ChunkQueue().chunkBytes, 160000);
  assert.equal(new ChunkQueue({ chunkBytes: 101 }).chunkBytes, 100);
});

test("push cuts numbered chunks of the exact size and keeps the rest", () => {
  const q = new ChunkQueue({ chunkBytes: 10 });
  q.push(bytes(4));
  assert.equal(q.pendingChunks, 0);
  q.push(bytes(18));
  assert.equal(q.pendingChunks, 2);
  assert.equal(q.pendingBytes, 22);
  assert.deepEqual(q.queue.map((c) => [c.number, c.data.length]), [[0, 10], [1, 10]]);
  assert.throws(() => q.push(bytes(3)), RangeError);
  q.finish();
  assert.deepEqual(q.queue.map((c) => [c.number, c.data.length]), [[0, 10], [1, 10], [2, 2]]);
  q.finish(); // nothing left: no empty chunk
  assert.equal(q.pendingChunks, 3);
});

test("chunks keep the audio in order across pushes", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(Uint8Array.of(1, 2));
  q.push(Uint8Array.of(3, 4, 5, 6));
  q.finish();
  assert.deepEqual(q.queue.map((c) => Array.from(c.data)), [[1, 2, 3, 4], [5, 6]]);
});

test("one at a time, in order, success moves on", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(bytes(12));
  const first = q.next(0);
  assert.equal(first.number, 0);
  assert.equal(q.next(0), null, "nothing while one is in flight");
  q.sent(0);
  assert.equal(q.next(0).number, 1);
  q.sent(1);
  assert.equal(q.next(0).number, 2);
  q.sent(2);
  assert.equal(q.next(0), null);
  assert.ok(q.drained);
  assert.equal(q.acked, 3);
});

test("a failure retries the same number with exponential back-off, capped", () => {
  const q = new ChunkQueue({ chunkBytes: 4, backoffBaseMs: 1000, backoffMaxMs: 5000 });
  q.push(bytes(8));
  let now = 0;
  const delays = [];
  for (let i = 0; i < 5; i++) {
    const c = q.next(now);
    assert.equal(c.number, 0, "never skips a number");
    assert.equal(c.attempt, i + 1);
    delays.push(q.failed(0, now));
    assert.equal(q.next(now), null, "waits");
    assert.equal(q.waitMs(now), delays[i]);
    now += delays[i];
  }
  assert.deepEqual(delays, [1000, 2000, 4000, 5000, 5000]);
  assert.equal(q.pendingChunks, 2, "later audio stays queued");
  q.next(now);
  q.sent(0);
  assert.equal(q.next(now).number, 1);
});

test("reporting a chunk that is not in flight is a programming error", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(bytes(4));
  assert.throws(() => q.sent(0));
  assert.throws(() => q.failed(0, 0));
  q.next(0);
  assert.throws(() => q.sent(1));
});

test("pending size and the large-backlog warning", () => {
  const q = new ChunkQueue({ chunkBytes: 4, warnBytes: 10 });
  q.push(bytes(8));
  assert.equal(q.isLarge, false);
  q.push(bytes(4));
  assert.equal(q.isLarge, true);
  assert.equal(q.pendingSeconds, 12 / 32000);
});

test("resync drops what the server already has and goes on", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(bytes(16));
  q.next(0);
  q.sent(0);
  q.next(0); // chunk 1 in flight, answer lost, but the server stored it
  assert.equal(q.resync(2), true);
  assert.equal(q.next(0).number, 2);
});

test("resync: a retried chunk the server never got is sent again", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(bytes(8));
  q.next(0);
  q.failed(0, 0);
  assert.equal(q.resync(0), true);
  assert.equal(q.next(0).number, 0, "back-off cleared by the resync");
});

test("resync refuses to skip a number or to trust an impossible server", () => {
  const gap = new ChunkQueue({ chunkBytes: 4, firstNumber: 0 });
  gap.push(bytes(8));
  gap.next(0);
  gap.sent(0);
  gap.next(0);
  gap.sent(1);
  assert.equal(gap.resync(0), false, "server lost chunks we already dropped");
  assert.equal(gap.fatal, "lost");
  assert.equal(gap.next(0), null);

  const ahead = new ChunkQueue({ chunkBytes: 4 });
  ahead.push(bytes(4));
  assert.equal(ahead.resync(5), false);
  assert.equal(ahead.fatal, "ahead");

  const hole = new ChunkQueue({ chunkBytes: 4, firstNumber: 3 });
  hole.push(bytes(4));
  assert.equal(hole.resync(1), false, "server stops before our first chunk");
  assert.equal(hole.fatal, "lost");
});

test("continuing a session numbers from what the server holds", () => {
  const q = new ChunkQueue({ chunkBytes: 4, firstNumber: 7 });
  q.push(bytes(4));
  assert.equal(q.next(0).number, 7);
  assert.equal(q.resync(7), true);
});

test("push refuses audio past maxBytes and stays full", () => {
  const q = new ChunkQueue({ chunkBytes: 4, maxBytes: 10 });
  assert.equal(q.push(new Uint8Array(8)), true);
  assert.equal(q.full, false);
  assert.equal(q.push(new Uint8Array(4)), false);
  assert.equal(q.full, true);
  assert.equal(q.pendingBytes, 8);
  assert.equal(q.push(new Uint8Array(2)), false, "once full, nothing more");
});

test("release puts the chunk in flight back, ready, without a failure", () => {
  const q = new ChunkQueue({ chunkBytes: 4 });
  q.push(new Uint8Array(4));
  const chunk = q.next(0);
  q.release(chunk.number);
  const again = q.next(0);
  assert.equal(again.number, 0);
  assert.equal(again.attempt, 1);
  assert.throws(() => q.release(5));
});

test("the default cap is 30 minutes of audio", () => {
  assert.equal(DEFAULT_MAX_BYTES, 1800 * 32000);
});
