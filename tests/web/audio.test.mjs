import test from "node:test";
import assert from "node:assert/strict";
import {
  Resampler, floatToPcm16, pcm16ToBytes, peak, concatBytes,
} from "../../transcriber/adapters/inbound/web/js/lib/audio.js";

function sine(hz, rate, seconds, amp = 0.5) {
  const out = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * hz * i) / rate);
  return out;
}

function rms(x, from = 0, to = x.length) {
  let s = 0;
  for (let i = from; i < to; i++) s += x[i] * x[i];
  return Math.sqrt(s / (to - from));
}

function resample(input, inRate, blockSize) {
  const r = new Resampler(inRate, 16000);
  const parts = [];
  for (let at = 0; at < input.length; at += blockSize) {
    parts.push(r.process(input.subarray(at, at + blockSize)));
  }
  parts.push(r.flush());
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float32Array(total);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

test("48 kHz to 16 kHz keeps a 1 kHz tone: right length, level and pitch", () => {
  const out = resample(sine(1000, 48000, 1), 48000, 4096);
  assert.ok(Math.abs(out.length - 16000) <= 4, `length ${out.length}`);
  const level = rms(out, 1000, 15000);
  assert.ok(Math.abs(level - 0.5 / Math.SQRT2) < 0.005, `rms ${level}`);
  const ref = sine(1000, 16000, 1);
  let worst = 0;
  for (let i = 1000; i < 15000; i++) worst = Math.max(worst, Math.abs(out[i] - ref[i]));
  assert.ok(worst < 0.02, `max deviation ${worst}`);
});

test("44.1 kHz works too (non-integer ratio)", () => {
  const out = resample(sine(500, 44100, 1), 44100, 1000);
  assert.ok(Math.abs(out.length - 16000) <= 4);
  assert.ok(Math.abs(rms(out, 1000, 15000) - 0.5 / Math.SQRT2) < 0.005);
});

test("a tone above the new Nyquist is removed, not folded back (no aliasing)", () => {
  // 10 kHz at 48 kHz would alias to 6 kHz if samples were just dropped.
  const out = resample(sine(10000, 48000, 1), 48000, 4096);
  assert.ok(rms(out, 1000, 15000) < 0.005, `alias level ${rms(out, 1000, 15000)}`);
  const dropped = new Float32Array(16000);
  const src = sine(10000, 48000, 1);
  for (let i = 0; i < dropped.length; i++) dropped[i] = src[i * 3];
  assert.ok(rms(dropped) > 0.3, "sample dropping would alias");
});

test("block size does not change the result", () => {
  const input = sine(700, 48000, 0.5);
  const a = resample(input, 48000, 128);
  const b = resample(input, 48000, 5000);
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-6);
});

test("same rate passes through; bad rates are refused", () => {
  const r = new Resampler(16000, 16000);
  const x = Float32Array.from([0.1, -0.2, 0.3]);
  assert.deepEqual(Array.from(r.process(x)), Array.from(x));
  assert.equal(r.flush().length, 0);
  assert.throws(() => new Resampler(0, 16000), RangeError);
});

test("floatToPcm16 scales, clips and survives NaN", () => {
  const ints = floatToPcm16(Float32Array.from([0, 1, -1, 0.5, -0.5, 2, -2, NaN]));
  assert.deepEqual(Array.from(ints), [0, 32767, -32768, 16384, -16384, 32767, -32768, 0]);
});

test("pcm16ToBytes is little-endian and even in length", () => {
  const bytes = pcm16ToBytes(Int16Array.from([1, -2, 256, -32768]));
  assert.deepEqual(Array.from(bytes), [1, 0, 0xfe, 0xff, 0, 1, 0, 0x80]);
  assert.equal(bytes.length % 2, 0);
});

test("peak and concatBytes", () => {
  assert.equal(peak(Float32Array.from([0.1, -0.7, 0.3])), Math.fround(0.7));
  assert.equal(peak(new Float32Array(0)), 0);
  assert.deepEqual(Array.from(concatBytes([Uint8Array.of(1, 2), Uint8Array.of(), Uint8Array.of(3)])), [1, 2, 3]);
});
