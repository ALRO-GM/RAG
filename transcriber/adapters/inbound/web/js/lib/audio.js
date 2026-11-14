// Pure audio helpers: streaming resampler to 16 kHz, float to 16-bit PCM.
// No DOM, no Web Audio: tested with node.

export const TARGET_RATE = 16000;

const LOWPASS = 0.94; // cut-off as a share of the output Nyquist frequency
const ZERO_CROSSINGS = 8; // sinc lobes kept on each side

function blackman(x) {
  // x in [-1, 1]
  if (x <= -1 || x >= 1) return 0;
  const a = Math.PI * (x + 1);
  return 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
}

/**
 * Streaming windowed-sinc resampler (low-pass plus interpolation in one
 * step), mono Float32 in, Float32 out. Keeps its state across calls, so
 * blocks of any size can be fed one after another.
 */
export class Resampler {
  constructor(inRate, outRate = TARGET_RATE) {
    if (!(inRate > 0) || !(outRate > 0)) {
      throw new RangeError("sample rates must be positive");
    }
    this.inRate = inRate;
    this.outRate = outRate;
    this.ratio = inRate / outRate;
    this.passthrough = inRate === outRate;
    const scale = Math.min(1, outRate / inRate);
    this.cutoff = 0.5 * scale * LOWPASS; // cycles per input sample
    this.halfWidth = ZERO_CROSSINGS / (2 * this.cutoff);
    this.buffer = new Float32Array(0);
    this.base = 0; // absolute index of buffer[0]
    this.produced = 0; // outputs made so far
  }

  /** Feed a block; return the output samples that are ready. */
  process(block) {
    if (this.passthrough) return Float32Array.from(block);
    const joined = new Float32Array(this.buffer.length + block.length);
    joined.set(this.buffer);
    joined.set(block, this.buffer.length);
    this.buffer = joined;
    const out = [];
    const w = this.halfWidth;
    const fc2 = 2 * this.cutoff;
    for (;;) {
      const t = this.produced * this.ratio;
      const last = Math.floor(t + w);
      if (last >= this.base + this.buffer.length) break;
      const first = Math.ceil(t - w);
      let sum = 0;
      let acc = 0;
      for (let i = first; i <= last; i++) {
        const d = i - t;
        const x = fc2 * d;
        const s = Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        const weight = fc2 * s * blackman(d / w);
        sum += weight;
        const j = i - this.base;
        if (j >= 0 && i >= 0) acc += weight * this.buffer[j];
      }
      out.push(acc / sum);
      this.produced += 1;
    }
    const keepFrom = Math.max(this.base, Math.ceil(this.produced * this.ratio - w));
    if (keepFrom > this.base) {
      this.buffer = this.buffer.slice(keepFrom - this.base);
      this.base = keepFrom;
    }
    return Float32Array.from(out);
  }

  /** Pad with silence so the last samples come out; use once at the end. */
  flush() {
    if (this.passthrough) return new Float32Array(0);
    return this.process(new Float32Array(Math.ceil(this.halfWidth) + 2));
  }
}

/** Convert floats in [-1, 1] to 16-bit integers, clipping what is out. */
export function floatToPcm16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let v = samples[i];
    if (!(v === v)) v = 0; // NaN
    if (v > 1) v = 1;
    else if (v < -1) v = -1;
    out[i] = Math.round(v < 0 ? v * 32768 : v * 32767);
  }
  return out;
}

/** Bytes of 16-bit little-endian PCM (independent of the host's order). */
export function pcm16ToBytes(ints) {
  const bytes = new Uint8Array(ints.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < ints.length; i++) view.setInt16(i * 2, ints[i], true);
  return bytes;
}

/** Largest absolute value of a block (0 for an empty one). */
export function peak(samples) {
  let m = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > m) m = a;
  }
  return m;
}

/** Concatenate byte blocks. */
export function concatBytes(parts) {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
