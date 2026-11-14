// AudioWorklet processor: collects the microphone's 128-frame quanta into
// blocks of 4096 mono Float32 samples and posts them to the page. Several
// input channels are averaged. The page resamples and encodes.

const BLOCK = 4096;

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(BLOCK);
    this.filled = 0;
    this.port.onmessage = (event) => {
      if (event.data === "flush" && this.filled > 0) {
        this.port.postMessage(this.block.slice(0, this.filled));
        this.filled = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channels = input.length;
    const frames = input[0].length;
    for (let i = 0; i < frames; i++) {
      let v = input[0][i];
      for (let c = 1; c < channels; c++) v += input[c][i];
      this.block[this.filled++] = v / channels;
      if (this.filled === BLOCK) {
        const full = this.block;
        this.port.postMessage(full, [full.buffer]);
        this.block = new Float32Array(BLOCK);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
