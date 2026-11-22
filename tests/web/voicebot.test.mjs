// The voice-bot page (#/voicebot) without a browser: the barge-in voice
// detector, the playback queue on a stand-in AudioContext, and the
// conversation built from the orchestrator's feed.
// Nothing here depends on the browser's language: the logic has no text.
import test from "node:test";
import assert from "node:assert/strict";
import {
  EnergyVad, LISTENING_MS, MODES, SpeechQueue, VOICEBOT, VoiceBotState, base64ToBytes, playQuiet, turnLatency,
} from "../../transcriber/adapters/inbound/web/js/lib/voicebot.js";

test("the voice detector starts after sustained level and ends after quiet", () => {
  const vad = new EnergyVad({ threshold: 0.1, startMs: 150, quietMs: 500 });
  assert.equal(vad.push(0.5, 0), null);
  assert.equal(vad.push(0.02, 85), null); // a click is not speech
  assert.equal(vad.push(0.5, 170), null);
  assert.equal(vad.push(0.5, 255), null);
  assert.equal(vad.push(0.5, 340), "start");
  assert.equal(vad.push(0.5, 425), null);
  assert.equal(vad.push(0.01, 510), null);
  assert.equal(vad.push(0.3, 600), null); // speech again resets the quiet time
  assert.equal(vad.push(0.01, 700), null);
  assert.equal(vad.push(0.01, 1250), "end");
  assert.equal(vad.speaking, false);
});

test("base64 becomes bytes", () => {
  assert.deepEqual([...base64ToBytes("AAEC/w==")], [0, 1, 2, 255]);
  assert.equal(base64ToBytes("").length, 0);
});

function fakeContext() {
  const ctx = {
    currentTime: 10,
    destination: {},
    started: [],
    async decodeAudioData(buffer) {
      return { duration: buffer.byteLength / 100 };
    },
    createBufferSource() {
      const source = {
        buffer: null, stopped: false, onended: null,
        connect() {},
        start(at) { ctx.started.push({ at, duration: source.buffer.duration, source }); },
        stop() { source.stopped = true; },
      };
      return source;
    },
  };
  return ctx;
}

test("sentences play back to back; a stop drops the turn's late chunks", async () => {
  const ctx = fakeContext();
  const states = [];
  const queue = new SpeechQueue(ctx, { onPlaying: (on) => states.push(on) });
  await queue.enqueue(1, new Uint8Array(100)); // 1 s
  await queue.enqueue(1, new Uint8Array(50)); // 0.5 s
  assert.deepEqual(ctx.started.map((s) => s.at), [10, 11]);
  assert.equal(queue.playing, true);
  queue.stop(1);
  assert.equal(queue.playing, false);
  assert.ok(ctx.started.every((s) => s.source.stopped));
  assert.equal(await queue.enqueue(1, new Uint8Array(100)).then(() => ctx.started.length), 2);
  await queue.enqueue(2, new Uint8Array(100));
  assert.equal(ctx.started.length, 3);
  assert.equal(ctx.started[2].at, 10);
  ctx.currentTime = 12;
  ctx.started[2].source.onended();
  assert.deepEqual(states, [true, true, false, true, false]);
});

test("a chunk that fails to decode does not block the queue", async () => {
  const ctx = fakeContext();
  const decode = ctx.decodeAudioData;
  ctx.decodeAudioData = async () => { throw new Error("bad"); };
  const queue = new SpeechQueue(ctx);
  await queue.enqueue(1, new Uint8Array(10));
  ctx.decodeAudioData = decode;
  await queue.enqueue(1, new Uint8Array(10));
  assert.equal(ctx.started.length, 1);
  queue.stop();
  assert.equal(queue.stoppedTurn, 0);
});

function speech(turn, text) {
  return { type: "speech", turn, seq: 0, text, audio: "AA==", format: "wav", sample_rate: 22050 };
}

test("the conversation follows the feed: user, thinking, speech, interrupt, record", () => {
  const state = new VoiceBotState();
  state.apply({ type: "session", session: { state: "running", page: { product: VOICEBOT } } }, 1);
  const utt = { id: "t0.s0", turn: 0, index: 0, speaker: "S1", start: 0, end: 1, revision: 0 };
  state.apply({ type: "event", event: { type: "partial", utterance: { ...utt, text: "¿Puedo" } } }, 2);
  state.apply({ type: "event", event: { type: "final", utterance: { ...utt, text: "¿Puedo devolverlo?" } } }, 3);
  state.apply({ type: "bot", turn: 1, state: "thinking", utterance: "t0.s0" }, 4);
  assert.equal(state.lines()[1].state, "thinking");
  state.apply(speech(1, "Sí."), 5);
  state.apply(speech(1, ""), 5); // a streamed sentence's later piece (#214)
  state.apply(speech(1, "Tienes 30 días."), 6);
  const record = { turn: 1, status: "ok", stages: { event_lag_ms: 600, first_audio_ms: 900.4, knowledge_ms: 12 } };
  state.apply({ type: "turn", record }, 7);
  state.apply({ type: "event", event: { type: "partial", utterance: { ...utt, id: "t1.s0", text: "" } } }, 8);
  const lines = state.lines();
  assert.deepEqual(lines.map((l) => [l.kind, l.text, l.state ?? null]), [
    ["user", "¿Puedo devolverlo?", null],
    ["bot", "Sí. Tienes 30 días.", "done"],
  ]);
  assert.equal(lines[0].partial, false);
  assert.equal(state.lastId, 8);
  state.apply({ type: "bot", turn: 2, state: "thinking" });
  state.apply(speech(2, "Te cuento."));
  state.apply({ type: "interrupt", turn: 2, by: "vad" });
  state.apply(speech(2, "tarde"));
  state.apply({ type: "turn", record: { turn: 2, status: "cancelled", stages: {} } });
  const bot2 = state.lines().at(-1);
  assert.equal(bot2.state, "interrupted");
  assert.equal(bot2.by, "vad");
  state.apply({ type: "turn", record: { turn: 3, status: "quota", stages: {} } });
  assert.equal(state.halted, "quota");
  state.apply({ type: "event", event: { type: "session_ended" } });
  assert.equal(state.state, "ended");
  state.apply({ type: "status", session: { state: "failed", error: "x" } });
  assert.equal(state.state, "failed");
  assert.equal(state.error, "x");
  assert.deepEqual(MODES, ["push", "hands_free"]);
});

test("the page knows while the bot's voice is loading", () => {
  const state = new VoiceBotState();
  assert.equal(state.preparing, false);
  state.apply({ type: "voice_state", state: "warming" });
  assert.equal(state.preparing, true);
  state.apply({ type: "voice_state", state: "ready" });
  assert.equal(state.preparing, false);
  state.apply({ type: "voice_state", state: "warming" });
  state.apply({ type: "status", session: { state: "ended" } });
  assert.equal(state.preparing, false, "an ended session prepares nothing");
  state.apply({ type: "voice_state", state: "fallback" });
  assert.equal(state.voice, "fallback");
});

test("turn latency adds the transcriber's lag to the orchestrator's first audio", () => {
  assert.deepEqual(turnLatency({ stages: { event_lag_ms: 650, first_audio_ms: 1210.6, knowledge_ms: 15, first_text_ms: 900, first_tts_ms: 40 } }),
    { lag: 650, knowledge: 15, text: 900, tts: 40, first: 1210.6, total: 1861 });
  assert.equal(turnLatency({ stages: { first_audio_ms: 10 } }).total, null);
  assert.equal(turnLatency(null).first, null);
});

test("push-to-talk sends silence of the same length while the button is up", async () => {
  const sent = [];
  globalThis.location = { protocol: "http:", host: "x" };
  const { LiveAudio } = await import("../../transcriber/adapters/inbound/web/js/live-audio.js");
  const audio = new LiveAudio({ id: "voz-1", code: "c", echoCancellation: true });
  assert.equal(audio.echoCancellation, true);
  audio.socket = { readyState: 1, bufferedAmount: 0, send: (piece) => sent.push(piece) };
  audio.muted = true;
  audio._send(new Uint8Array(3200).fill(7));
  audio.muted = false;
  audio._send(new Uint8Array(3200).fill(7));
  assert.equal(sent.length, 2);
  assert.ok(sent[0].every((b) => b === 0));
  assert.ok(sent[1].every((b) => b === 7));
});

test("a backchannel plays once, quietly, outside the bot's speech queue", async () => {
  const ctx = fakeContext();
  const gains = [];
  ctx.createGain = () => {
    const node = { gain: { value: 1 }, connect() {} };
    gains.push(node);
    return node;
  };
  assert.equal(await playQuiet(ctx, new Uint8Array(40), 0.35), true);
  assert.equal(ctx.started.length, 1);
  assert.equal(ctx.started[0].at, 10);
  assert.equal(gains[0].gain.value, 0.35);
  assert.equal(await playQuiet(ctx, new Uint8Array(40), 7), true);
  assert.equal(gains[1].gain.value, 1);
  ctx.decodeAudioData = async () => { throw new Error("bad"); };
  assert.equal(await playQuiet(ctx, new Uint8Array(40)), false);
  assert.ok(LISTENING_MS >= 1000);
});
