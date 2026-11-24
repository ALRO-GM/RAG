// The scripted examples' voices (#254) without a browser: which examples
// can be heard, the remembered sound on/off, the synthetic-voices note,
// the clip player that never hangs (a clip that fails, is refused by the
// autoplay rules or never ends), the live call playing a line's clip when
// it starts, and the voice example keeping the same timing muted.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CLIP_MARGIN_MS, ClipPlayer, SOUND_KEY, clipsOf, lineToPlay, readSound, voicesNoteKey, writeSound,
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

/** An audio element that ends, fails or is refused, as the test says. */
function fakeAudio(behaviour, made) {
  return (url) => {
    const listeners = {};
    const audio = {
      url, paused: false,
      addEventListener(name, fn) { listeners[name] = fn; },
      pause() { audio.paused = true; },
      play() {
        if (behaviour === "refused") return Promise.reject(Object.assign(new Error("no gesture"), { name: "NotAllowedError" }));
        queueMicrotask(() => {
          if (behaviour === "ends") listeners.ended?.();
          if (behaviour === "error") listeners.error?.();
        });
        return Promise.resolve();
      },
    };
    made.push(audio);
    return audio;
  };
}

function clock() {
  const timers = [];
  return {
    timers,
    schedule: (fn, ms) => { timers.push({ fn, ms, live: true }); return timers.length; },
    cancel: (id) => { if (timers[id - 1]) timers[id - 1].live = false; },
    fire() { for (const t of timers.filter((x) => x.live)) { t.live = false; t.fn(); } },
  };
}

test("a clip that ends resolves true; one that fails, is refused or never ends resolves false and is said once", async () => {
  for (const [behaviour, expected] of [["ends", true], ["error", false], ["refused", false]]) {
    const made = [];
    let failures = 0;
    const loaded = [];
    const player = new ClipPlayer({
      load: async (key) => { loaded.push(key); return `blob:${key}`; },
      makeAudio: fakeAudio(behaviour, made), onFail: () => { failures += 1; },
    });
    assert.equal(await player.play("t0.s0", 2), expected, behaviour);
    assert.equal(made[0].url, "blob:t0.s0");
    if (!expected) {
      assert.equal(await player.play("t1.s0", 2), false);
      assert.equal(failures, 1, "said once");
    }
    await player.dispose();
  }
  const timers = clock();
  const made = [];
  let failures = 0;
  const silent = new ClipPlayer({ load: async () => "blob:x", makeAudio: fakeAudio("silent", made), onFail: () => { failures += 1; }, ...timers });
  const pending = silent.play("t0.s0", 3);
  assert.equal(timers.timers[0].ms, 3000 + CLIP_MARGIN_MS);
  await new Promise((r) => setImmediate(r));
  timers.fire();
  assert.equal(await pending, false);
  assert.equal(failures, 1);
  assert.equal(made[0].paused, true);
  const broken = new ClipPlayer({ load: async () => { throw new Error("404"); }, makeAudio: fakeAudio("ends", []), onFail: () => { failures += 1; } });
  assert.equal(await broken.play("t0.s0", 1), false);
  assert.equal(failures, 2);
});

test("muted, nothing is loaded or played; a new line or the switch stops the one playing", async () => {
  const made = [];
  const loaded = [];
  const player = new ClipPlayer({ load: async (key) => { loaded.push(key); return key; }, makeAudio: fakeAudio("silent", made) });
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
  assert.deepEqual(made.map((a) => a.paused), [true, true]);
  await player.prefetch(["t2.s0", "t2.s0", "t3.s0"]);
  assert.deepEqual(loaded, ["t0.s0", "t1.s0", "t2.s0", "t3.s0"], "each clip loaded once");
  const revoked = [];
  await player.dispose((url) => revoked.push(url));
  assert.deepEqual(revoked.sort(), ["t0.s0", "t1.s0", "t2.s0", "t3.s0"]);
  await player.prefetch(["t9.s0"]);
  assert.equal(loaded.includes("t9.s0"), false, "a disposed player loads nothing");
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
});
