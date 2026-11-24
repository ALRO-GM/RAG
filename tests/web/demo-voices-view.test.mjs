// The voice example's page (#254) driven through a minimal stand-in DOM and
// a fake Web Audio context: nothing sounds before «Empezar el ejemplo», the
// context is resumed inside that click, every line's clip is fetched with
// the access code and played in order, the note says what the voices are,
// and with the sound off nothing plays.
import test from "node:test";
import assert from "node:assert/strict";
import * as i18n from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";

class FakeNode {}

class FakeText extends FakeNode {
  constructor(text) {
    super();
    this.textContent = text;
  }
}

class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    for (const key of ["id", "type", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.checked = false;
    this.hidden = false;
    this.disabled = false;
    this.open = false;
    this.classList = { add: (c) => { this.className = `${this.className} ${c}`.trim(); } };
  }

  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  dispatch(type) { for (const listener of this.listeners[type] ?? []) listener({ target: this }); }
  append(...nodes) { for (const n of nodes) { n.parent = this; this.children.push(n); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }
  querySelector() { return null; }
  get lastElementChild() { return this.children.filter((c) => c instanceof FakeElement).at(-1) ?? null; }
  get lastChild() { return this.children.at(-1) ?? null; }
  all(test) {
    const out = test(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeElement) out.push(...c.all(test));
    return out;
  }
}

class FakeContext {
  constructor(log) {
    this.log = log;
    this.state = "suspended";
    this.destination = "speakers";
  }
  async resume() { this.log.push("resume"); this.state = "running"; }
  async close() { this.log.push("close"); }
  async decodeAudioData(bytes) { return { key: bytes.key }; }
  createBufferSource() {
    const log = this.log;
    const source = {
      buffer: null, onended: null,
      connect() {},
      start() { log.push(`start ${source.buffer.key}`); queueMicrotask(() => source.onended?.()); },
      stop() {},
    };
    return source;
  }
}

const VIEW = {
  examples: {
    call: { available: false, clips: {} },
    voicebot: { available: true, clips: { "vb0.user": { seconds: 0.01 }, "vb0.bot": { seconds: 0.01 } } },
  },
  voices: { cliente: "stock", asistente: "stock" }, kinds: ["stock"],
};
const DATA = {
  title: "Ejemplo", intro: "Intro", sources: [{ id: "s1", title: "Fuente", text: "Texto" }],
  exchanges: [{ user: "Hola", bot: "Buenas", cites: ["s1"] }],
};

async function settle(until, rounds = 400) {
  for (let i = 0; i < rounds && !until(); i += 1) await new Promise((r) => setImmediate(r));
}

test("the voice example sounds only after the click, through Web Audio, line by line", async () => {
  const log = [];
  const fetched = [];
  const saved = {
    document: globalThis.document, Node: globalThis.Node, setTimeout: globalThis.setTimeout,
    AudioContext: globalThis.AudioContext, speechSynthesis: globalThis.speechSynthesis,
  };
  const realTimeout = globalThis.setTimeout;
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag), createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  // Short waits run at once; the clip's give-up timer (>= 2.5 s) never fires here.
  globalThis.setTimeout = (fn, ms) => (ms > 2000 ? 0 : realTimeout(fn, 0));
  globalThis.AudioContext = class extends FakeContext { constructor() { super(log); } };
  delete globalThis.speechSynthesis;
  const language = i18n.lang;
  i18n.setLanguage("es");
  try {
    const { voiceDemoView } = await import("../../transcriber/adapters/inbound/web/js/views/voice-demo.js");
    const ctx = {
      api: {
        liveDemoVoice: async () => DATA,
        liveDemoVoices: async () => VIEW,
        liveDemoClip: async (_id, key) => { fetched.push(key); return { key }; },
      },
      fail: () => {},
    };
    const root = new FakeElement("main");
    const close = await voiceDemoView(ctx, root, "llamada-lumbre");
    await settle(() => false, 5);
    const note = root.all((e) => e.className.includes("vd-voices"))[0];
    assert.equal(note.textContent, es["dvoice.stock"]);
    assert.equal(note.attributes.role, "status");
    assert.deepEqual(log, [], "nothing sounds or resumes before the click");
    const [start] = root.all((e) => e.tagName === "BUTTON" && e.className.includes("primary"));
    start.dispatch("click");
    assert.deepEqual(log, ["resume"], "resumed inside the click");
    await settle(() => !start.hidden);
    assert.deepEqual(log, ["resume", "start vb0.user", "start vb0.bot"]);
    assert.deepEqual(fetched, ["vb0.user", "vb0.bot"]);
    const [box] = root.all((e) => e.tagName === "INPUT" && e.type === "checkbox");
    assert.equal(box.checked, true);
    box.checked = false;
    box.dispatch("change");
    start.dispatch("click");
    await settle(() => false, 50);
    await settle(() => !start.hidden);
    assert.equal(log.filter((x) => x.startsWith("start")).length, 2, "muted: nothing more plays");
    close();
    await settle(() => log.includes("close"), 20);
    assert.ok(log.includes("close"));
  } finally {
    i18n.setLanguage(language);
    Object.assign(globalThis, saved);
    if (saved.speechSynthesis === undefined) delete globalThis.speechSynthesis;
  }
});
