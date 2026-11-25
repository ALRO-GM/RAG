// #263 review (B2): a closed tab used to hold the one session slot for its
// grace with no way out. The start forms now say which session runs and
// offer «Volver a la sesión» and «Terminarla». Driven here with a fake DOM.
import test from "node:test";
import assert from "node:assert/strict";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";

class FakeNode {}
class FakeText extends FakeNode {
  constructor(text) { super(); this.textContent = text; }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    this.classList = { add: () => {} };
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.hidden = false;
    this.disabled = false;
    this.removed = false;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  remove() { this.removed = true; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.children.map((c) => (c.text ? c.text() : c.textContent)).join("") + (this.children.length ? "" : this.textContent); }
  async click() { for (const fn of this.listeners.click ?? []) await fn({ preventDefault() {} }); }
}

async function withDom(fn) {
  const saved = { document: globalThis.document, Node: globalThis.Node };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  try {
    const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
    setLanguage("es");
    return await fn((await import("../../transcriber/adapters/inbound/web/js/views/live-busy.js")).busyNotice);
  } finally {
    Object.assign(globalThis, saved);
  }
}

const INTERVIEW = { id: "live-1", state: "running", product: "live_assist.interview", mode: "live", audio: "mic" };
const BOT = { id: "voz-1", state: "running", product: "voicebot.support", mode: "live", audio: "mic" };

function api(sessions, stopped = []) {
  return {
    api: {
      liveSessions: async () => ({ sessions }),
      stopLive: async (id) => { stopped.push(id); return {}; },
    },
  };
}

const buttons = (card) => card.all().filter((n) => n.tagName === "BUTTON");

test("busy: the card names the running session; «Volver a la sesión» reattaches here", async () => {
  await withDom(async (busyNotice) => {
    const back = [];
    const card = await busyNotice(api([{ id: "old", state: "ended" }, INTERVIEW]), { here: "live", onBack: (s) => back.push(s.id), onEnded: () => {} });
    const text = card.text();
    assert.ok(text.includes(es["live.busy.title"]));
    assert.ok(text.includes("Solo puede haber una sesión a la vez. La que está en marcha: Entrevista de trabajo."));
    const [again, end] = buttons(card);
    assert.equal(again.text(), "Volver a la sesión");
    assert.equal(end.text(), "Terminarla");
    await again.click();
    assert.deepEqual(back, ["live-1"]);
    assert.ok(card.removed);
  });
});

test("busy: a session of the other page is remembered and that page opened", async () => {
  await withDom(async (busyNotice) => {
    const data = new Map();
    globalThis.sessionStorage = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v), removeItem: (k) => data.delete(k) };
    const went = [];
    const card = await busyNotice(api([BOT]), { here: "live", onBack: () => assert.fail("not here"), onEnded: () => {}, go: (h) => went.push(h) });
    assert.ok(card.text().includes("Asistente de voz"));
    await buttons(card)[0].click();
    assert.deepEqual(went, ["#/voicebot"]);
    assert.deepEqual(JSON.parse([...data.values()][0]), { id: "voz-1", audio: "mic", page: "voicebot" });
    delete globalThis.sessionStorage;
  });
});

test("busy: «Terminarla» asks, stops the running session and lets the new one start", async () => {
  await withDom(async (busyNotice) => {
    const stopped = [];
    let ended = 0;
    let asked = null;
    const card = await busyNotice(api([INTERVIEW], stopped), {
      here: "live", onBack: () => {}, onEnded: () => { ended += 1; }, confirm: async (o) => { asked = o; return true; },
    });
    await buttons(card)[1].click();
    assert.equal(asked.title, "¿Terminar la sesión en marcha?");
    assert.deepEqual(stopped, ["live-1"]);
    assert.equal(ended, 1);
    assert.ok(card.text().includes("Sesión terminada. Ya puedes empezar la nueva."));
    const kept = [];
    const no = await busyNotice(api([INTERVIEW], kept), { here: "live", onBack: () => {}, onEnded: () => {}, confirm: async () => false });
    await buttons(no)[1].click();
    assert.deepEqual(kept, [], "nothing stopped without the yes");
  });
});

test("busy: nothing runs, or the list cannot be read: no card", async () => {
  await withDom(async (busyNotice) => {
    assert.equal(await busyNotice(api([{ id: "x", state: "ended" }]), { here: "live", onBack() {}, onEnded() {} }), null);
    const down = { api: { liveSessions: async () => { throw new Error("down"); } } };
    assert.equal(await busyNotice(down, { here: "live", onBack() {}, onEnded() {} }), null);
  });
});

test("both start forms show the card when the start answers busy", async () => {
  const { readFileSync } = await import("node:fs");
  for (const file of ["live.js", "voicebot.js"]) {
    const view = readFileSync(new URL(`../../transcriber/adapters/inbound/web/js/views/${file}`, import.meta.url), "utf8");
    assert.match(view, /if \(error\?\.code === "busy"\) showBusy\(\);/, file);
    assert.match(view, /const card = await busyNotice\(ctx, \{/, file);
  }
});
