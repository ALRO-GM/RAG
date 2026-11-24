// The scripted examples' voices (#254) without a browser: which examples
// can be heard, the remembered sound on/off, the synthetic-voices note,
// the clip player that never hangs (a clip that fails, is refused by the
// autoplay rules or never ends), the live call playing a line's clip when
// it starts, and the voice example keeping the same timing muted.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CLIP_MARGIN_MS, ClipPlayer, SOUND_KEY, clipsOf, createAudioContext, lineToPlay, readSound, voicesNoteKey, webAudioOutput, writeSound,
} from "../../transcriber/adapters/inbound/web/js/lib/demo-voices.js";
import { play, stepsOf } from "../../transcriber/adapters/inbound/web/js/lib/voice-demo.js";
import * as i18n from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const read = (f) => readFileSync(new URL(`../../${f}`, import.meta.url), "utf8");
const WEB = "transcriber/adapters/inbound/web/js/";
const DEMO = JSON.parse(read("examples/demo/llamada-lumbre.demo.json"));

// What GET /demos/{id}/voices answers once the call and the voice example are rendered.
const VIEW = {
  demo: true, id: DEMO.id,
  examples: {
    call: { available: true, clips: Object.fromEntries(DEMO.transcript.utterances.map((u, i) => [u.id, { seconds: 2 + i / 10 }])) },
    voicebot: { available: true, clips: Object.fromEntries(DEMO.voicebot.exchanges.flatMap((_, n) => [[`vb${n}.user`, { seconds: 3 }], [`vb${n}.bot`, { seconds: 4 }]])) },
  },
  voices: { agente: "stock", cliente: "clone", asistente: "stock" },
  kinds: ["clone", "stock"],
};
const NONE = { demo: true, id: DEMO.id, examples: { call: { available: false, clips: {} }, voicebot: { available: false, clips: {} } }, voices: {}, kinds: [] };

function memory() {
  const kept = new Map();
  return { getItem: (k) => kept.get(k) ?? null, setItem: (k, v) => kept.set(k, v), kept };
}

test("the sound is on by default and remembered, even where storage is blocked", () => {
  const storage = memory();
  assert.equal(readSound(storage), true);
  writeSound(false, storage);
  assert.equal(storage.kept.get(SOUND_KEY), "off");
  assert.equal(readSound(storage), false);
  writeSound(true, storage);
  assert.equal(readSound(storage), true);
  const blocked = { getItem() { throw new Error("denied"); }, setItem() { throw new Error("denied"); } };
  assert.equal(readSound(blocked), true);
  writeSound(false, blocked);
  assert.equal(readSound(null), true);
});

test("an example is heard only when the server says so, with every clip's length", () => {
  const call = clipsOf(VIEW, "call");
  assert.equal(call.size, 19);
  assert.equal(call.get("t0.s0"), 2);
  assert.equal(clipsOf(VIEW, "voicebot").get("vb0.bot"), 4);
  assert.equal(clipsOf(NONE, "call"), null);
  assert.equal(clipsOf(null, "call"), null);
  assert.equal(clipsOf({ examples: { call: { available: true, clips: { a: { seconds: "x" } } } } }, "call"), null);
});

test("the page says what the voices are, or that there are none", () => {
  assert.equal(voicesNoteKey(VIEW, "call"), "dvoice.mixed");
  assert.equal(voicesNoteKey({ ...VIEW, voices: { a: "stock" } }, "call"), "dvoice.stock");
  assert.equal(voicesNoteKey({ ...VIEW, voices: { a: "clone" } }, "voicebot"), "dvoice.clone");
  assert.equal(voicesNoteKey(NONE, "call"), "dvoice.none");
  assert.equal(voicesNoteKey(null, "voicebot"), "dvoice.none");
});

test("every voices text exists in both languages, Spanish first, synthetic and no real customer", () => {
  // Node 21+ has navigator.language, which would pick English: pin Spanish.
  const language = i18n.lang;
  i18n.setLanguage("es");
  try {
    const keys = Object.keys(es).filter((k) => k.startsWith("dvoice."));
    assert.deepEqual(keys.sort(), Object.keys(en).filter((k) => k.startsWith("dvoice.")).sort());
    assert.ok(keys.length >= 7);
    for (const key of ["dvoice.clone", "dvoice.stock", "dvoice.mixed"]) {
      assert.match(i18n.t(key), /sintéticas/);
      assert.match(en[key], /synthetic/);
      assert.match(en[key], /Spanish/);
    }
    assert.match(i18n.t("dvoice.clone"), /permiso/);
    assert.match(i18n.t("dvoice.none"), /no están disponibles/);
  } finally {
    i18n.setLanguage(language);
  }
});

/** A Web Audio output whose clips end, fail to decode or stay suspended, as the test says. */
function fakeOutput(behaviour, started) {
  return {
    ready: async () => behaviour !== "suspended",
    decode: async (bytes) => {
      if (behaviour === "undecodable") throw new Error("EncodingError");
      return { bytes };
    },
    start(buffer) {
      const sound = { buffer, stopped: false };
      started.push(sound);
      return {
        ended: behaviour === "ends" ? Promise.resolve() : new Promise(() => {}),
        stop() { sound.stopped = true; },
      };
    },
  };
}

function clock() {
  const timers = [];
  // Like the browser's, they refuse to be called on another object.
  return {
    timers,
    schedule: function (fn, ms) { assert.equal(this, undefined); timers.push({ fn, ms, live: true }); return timers.length; },
    cancel: function (id) { assert.equal(this, undefined); if (timers[id - 1]) timers[id - 1].live = false; },
    fire() { for (const t of timers.filter((x) => x.live)) { t.live = false; t.fn(); } },
  };
}

test("a clip that ends resolves true; one that fails, is refused or never ends resolves false and is said once", async () => {
  for (const [behaviour, expected] of [["ends", true], ["undecodable", false], ["suspended", false]]) {
    const started = [];
    let failures = 0;
    const player = new ClipPlayer({
      load: async (key) => `bytes:${key}`,
      output: fakeOutput(behaviour, started), onFail: () => { failures += 1; },
    });
    assert.equal(await player.play("t0.s0", 2), expected, behaviour);
    if (expected) assert.equal(started[0].buffer.bytes, "bytes:t0.s0");
    else {
      assert.equal(started.length, 0);
      assert.equal(await player.play("t1.s0", 2), false);
      assert.equal(failures, 1, "said once");
    }
    player.dispose();
  }
  const timers = clock();
  const started = [];
  let failures = 0;
  const silent = new ClipPlayer({ load: async () => "x", output: fakeOutput("silent", started), onFail: () => { failures += 1; }, ...timers });
  const pending = silent.play("t0.s0", 3);
  assert.equal(timers.timers[0].ms, 3000 + CLIP_MARGIN_MS);
  await new Promise((r) => setImmediate(r));
  timers.fire();
  assert.equal(await pending, false);
  assert.equal(failures, 1);
  assert.equal(started[0].stopped, true);
  const broken = new ClipPlayer({ load: async () => { throw new Error("404"); }, output: fakeOutput("ends", []), onFail: () => { failures += 1; } });
  assert.equal(await broken.play("t0.s0", 1), false);
  assert.equal(failures, 2);
});

test("muted, nothing is loaded or played; a new line or the switch stops the one playing", async () => {
  const started = [];
  const loaded = [];
  const player = new ClipPlayer({ load: async (key) => { loaded.push(key); return key; }, output: fakeOutput("silent", started) });
  player.setSound(false);
  assert.equal(await player.play("t0.s0", 2), false);
  assert.deepEqual(loaded, []);
  player.setSound(true);
  const first = player.play("t0.s0", 2);
  await new Promise((r) => setImmediate(r));
  const second = player.play("t1.s0", 2);
  assert.equal(await first, false);
  await new Promise((r) => setImmediate(r));
  player.setSound(false);
  assert.equal(await second, false);
  assert.deepEqual(started.map((a) => a.stopped), [true, true]);
  await player.prefetch(["t2.s0", "t2.s0", "t3.s0"]);
  assert.deepEqual(loaded, ["t0.s0", "t1.s0", "t2.s0", "t3.s0"], "each clip loaded once");
  player.dispose();
  await player.prefetch(["t9.s0"]);
  assert.equal(loaded.includes("t9.s0"), false, "a disposed player loads nothing");
});

test("the Web Audio output resumes, decodes and plays a buffer, and says when it cannot", async () => {
  const calls = [];
  let onended = null;
  const context = {
    state: "suspended", destination: "out",
    async resume() { calls.push("resume"); this.state = "running"; },
    async decodeAudioData(bytes) { calls.push(["decode", bytes]); return "buffer"; },
    createBufferSource() {
      const source = { connect: (to) => calls.push(["connect", to]), start: () => calls.push("start"), stop: () => calls.push("stop"),
        set onended(fn) { onended = fn; } };
      return source;
    },
  };
  const output = webAudioOutput(context);
  assert.equal(await output.ready(), true);
  assert.equal(await output.decode("bytes"), "buffer");
  const sound = output.start("buffer");
  onended();
  await sound.ended;
  sound.stop();
  assert.deepEqual(calls, ["resume", ["decode", "bytes"], ["connect", "out"], "start", "stop"]);
  const refused = webAudioOutput({ state: "suspended", resume: async () => { throw new Error("no gesture"); } });
  assert.equal(await refused.ready(), false);
  assert.equal(await webAudioOutput(null).ready(), false);
  await assert.rejects(() => webAudioOutput(null).decode("x"));
  assert.equal(createAudioContext({}), null);
  class Context { constructor() { this.state = "suspended"; } }
  assert.ok(createAudioContext({ AudioContext: Context }) instanceof Context);
  assert.equal(createAudioContext({ AudioContext: class { constructor() { throw new Error("x"); } } }), null);
});

test("the live call plays a line's clip once, when the line starts", () => {
  const clips = clipsOf(VIEW, "call");
  const started = new Set();
  const event = (type, id) => ({ type: "event", event: { type, utterance: { id } } });
  assert.equal(lineToPlay(event("partial", "t0.s0"), clips, started), "t0.s0");
  assert.equal(lineToPlay(event("partial", "t0.s0"), clips, started), null);
  assert.equal(lineToPlay(event("final", "t0.s0"), clips, started), null);
  assert.equal(lineToPlay(event("final", "t1.s0"), clips, started), "t1.s0", "a final with no partial before");
  assert.equal(lineToPlay(event("final", "zz"), clips, started), null);
  assert.equal(lineToPlay({ type: "suggestion" }, clips, started), null);
  assert.equal(lineToPlay(event("partial", "t2.s0"), null, started), null);
  assert.equal(lineToPlay(event("voice_check", "t2.s0"), clips, started), null);
});

test("the voice example takes the clips' lengths, the same muted, and a failing clip does not hang it", async () => {
  const VOICE = { ...DEMO.voicebot, sources: DEMO.sources };
  const steps = stepsOf(VOICE, DEMO.sources, "es");
  assert.deepEqual(steps.slice(0, 3).map((s) => s.key), ["vb0.user", undefined, "vb0.bot"]);
  const clips = clipsOf(VIEW, "voicebot");
  const timeline = async (voice) => {
    const waited = [];
    const played = [];
    const done = await play(steps, {
      show() {}, sleep: async (ms) => { waited.push(ms); }, clips,
      voice: voice ? async (key, seconds) => { played.push([key, seconds]); return voice(key); } : null,
      narrate: async () => { throw new Error("the browser voice is not used with clips"); },
    });
    return { done, waited, played };
  };
  const heard = await timeline(async () => true);
  const muted = await timeline(null);
  const failing = await timeline(async (key) => key !== "vb1.bot");
  assert.equal(heard.done && muted.done && failing.done, true);
  assert.deepEqual(heard.waited, muted.waited, "the same timing with the sound off");
  assert.deepEqual(failing.waited, muted.waited);
  assert.ok(heard.waited.includes(3000) && heard.waited.includes(4000));
  assert.equal(heard.played.length, 8);
  assert.deepEqual(heard.played[1], ["vb0.bot", 4]);
  assert.equal(muted.played.length, 0);
});

test("the pages play only after a press, remember the switch and say the note once", () => {
  const voice = read(`${WEB}views/voice-demo.js`);
  assert.match(voice, /liveDemoVoices/);
  assert.match(voice, /writeSound\(sound\)/);
  assert.match(voice, /voicesNoteKey\(voices, "voicebot"\)/);
  assert.match(voice, /dispose\(/);
  const live = read(`${WEB}views/live.js`);
  assert.match(live, /userActivation\?\.hasBeenActive/);
  assert.match(live, /lineToPlay\(message, clips, startedLines\)/);
  assert.match(live, /dvoice\.press/);
  assert.match(live, /writeSound\(soundBox\.checked\)/);
  for (const file of ["views/voice-demo.js", "views/live.js"]) {
    for (const m of read(WEB + file).matchAll(/\bt\(\s*"(dvoice\.[a-z_]+)"/g)) {
      assert.ok(es[m[1]] && en[m[1]], m[1]);
    }
  }
  const api = read(`${WEB}api.js`);
  assert.match(api, /\/voices\/\$\{encodeURIComponent\(line\)\}\.wav/);
  assert.match(api, /response\.arrayBuffer\(\)/);
});

test("no clip is played from a blob: URL (the site's CSP has media-src 'self')", () => {
  for (const file of ["lib/demo-voices.js", "views/voice-demo.js", "views/live.js"]) {
    const code = read(WEB + file).replace(/\/\/.*$/gm, "");
    assert.doesNotMatch(code, /createObjectURL/, file);
    assert.doesNotMatch(code, /new Audio\(/, file);
  }
  assert.match(read(`${WEB}views/voice-demo.js`), /webAudioOutput\(audio\)/);
  assert.match(read(`${WEB}views/live.js`), /webAudioOutput\(audioContext\)/);
  assert.match(read(`${WEB}views/live.js`), /onClick: \(\) => \{ unlockAudio\(\); startDemo\(supportDemo\); \}/);
});
