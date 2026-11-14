// Microphone capture: getUserMedia -> AudioWorklet -> 16 kHz mono PCM16.

import { Resampler, TARGET_RATE, floatToPcm16, pcm16ToBytes, peak } from "./lib/audio.js";

export class CaptureError extends Error {
  /** reason: unsupported | noworklet | denied | none | failed */
  constructor(reason) {
    super(reason);
    this.reason = reason;
  }
}

const WORKLET_URL = new URL("./worklet/recorder-processor.js", import.meta.url).href;

export class Capture {
  /**
   * @param {object} handlers
   *   onPcm(bytes)      16 kHz mono 16-bit little-endian PCM, even length
   *   onLevel(peak)     0..1 per block
   *   onProblem(kind)   "ended" (microphone taken) | "suspended"
   */
  constructor({ onPcm, onLevel = () => {}, onProblem = () => {} }) {
    this.onPcm = onPcm;
    this.onLevel = onLevel;
    this.onProblem = onProblem;
    this.paused = false;
    this.samples = 0; // 16 kHz samples delivered (pauses excluded)
    this.stopped = false;
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.resampler = null;
  }

  get seconds() {
    return this.samples / TARGET_RATE;
  }

  async start() {
    const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!navigator.mediaDevices?.getUserMedia || !AudioContextClass) throw new CaptureError("unsupported");
    if (!("audioWorklet" in AudioContextClass.prototype)) throw new CaptureError("noworklet");
    // Created and resumed first, still inside the tap that started us (iOS).
    this.ctx = new AudioContextClass();
    this.ctx.resume?.();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: { ideal: 1 },
        },
      });
    } catch (error) {
      await this._close();
      const reason = error?.name === "NotAllowedError" || error?.name === "SecurityError" ? "denied"
        : error?.name === "NotFoundError" || error?.name === "OverconstrainedError" ? "none" : "failed";
      throw new CaptureError(reason);
    }
    try {
      await this.ctx.audioWorklet.addModule(WORKLET_URL);
      this.resampler = new Resampler(this.ctx.sampleRate, TARGET_RATE);
      const source = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, "recorder-processor", {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      });
      this.node.port.onmessage = (event) => this._block(event.data);
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      source.connect(this.node);
      this.node.connect(mute);
      mute.connect(this.ctx.destination); // some browsers only run connected graphs
    } catch {
      await this.stop();
      throw new CaptureError("failed");
    }
    for (const track of this.stream.getAudioTracks()) {
      track.addEventListener("ended", () => !this.stopped && this.onProblem("ended"));
    }
    this.ctx.addEventListener("statechange", () => {
      if (this.stopped || this.ctx.state === "running") return;
      this.onProblem("suspended");
      this.ctx.resume?.().catch(() => {});
    });
  }

  _block(samples) {
    this.onLevel(peak(samples));
    if (this.paused || this.stopped) return;
    this._emit(this.resampler.process(samples));
  }

  _emit(floats) {
    if (floats.length === 0) return;
    this.samples += floats.length;
    this.onPcm(pcm16ToBytes(floatToPcm16(floats)));
  }

  /** Stop the microphone; the audio still inside the pipeline is delivered. */
  async stop() {
    if (this.stopped) return;
    if (this.node && !this.paused) {
      this.node.port.postMessage("flush");
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    this.stopped = true;
    if (this.resampler && !this.paused) this._emit(this.resampler.flush());
    await this._close();
  }

  async _close() {
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    try {
      this.node?.disconnect();
      await this.ctx?.close();
    } catch {
      // already closed
    }
  }
}
