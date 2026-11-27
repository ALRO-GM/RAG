// «Compartir pantalla» (#278): two simulated devices on a stand-in DOM.
// The sharing device's page is mirrored into another document by
// snapshot + deltas and must stay identical through additions, removals,
// moves, attributes, texts and form values; what must never leave is not
// sent (scripts, the panel, passwords, sensitive parts, the access code);
// an action of a joined device lands on the same node of the sharing one;
// the link authenticates first, resumes with its token after a drop and
// stops on a final refusal. Invented data only.
import test from "node:test";
import assert from "node:assert/strict";
import {
  MASK, MirrorSource, MirrorView, actionable, attributesOf, elementScrollOf, elementScrollTo, pointerAt, propsOf, setChildren, skipped,
} from "../../transcriber/adapters/inbound/web/js/lib/mirror.js";
import {
  BACKOFF_MAX_MS, FINAL_CODES, ShareLink, shareBackoffMs, cleanLabel, deviceKind, endedKey, externalUrl, randomPin, refusalKey, socketUrl, statusKey, validName, validPin,
} from "../../transcriber/adapters/inbound/web/js/lib/share.js";
import { SoundPlayer, installSoundBridge, sameOriginApi } from "../../transcriber/adapters/inbound/web/js/share-sound.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

// ---- a small DOM ---------------------------------------------------------------

class Node {
  constructor(doc) {
    this.ownerDocument = doc;
    this.parentNode = null;
    this.childNodes = [];
  }
  get isConnected() {
    let node = this;
    while (node.parentNode) node = node.parentNode;
    return node === this.ownerDocument;
  }
  get parentElement() {
    return this.parentNode?.nodeType === 1 ? this.parentNode : null;
  }
  appendChild(node) {
    return this.insertBefore(node, null);
  }
  insertBefore(node, ref) {
    node.parentNode?.removeChild(node);
    const at = ref ? this.childNodes.indexOf(ref) : this.childNodes.length;
    this.childNodes.splice(at < 0 ? this.childNodes.length : at, 0, node);
    node.parentNode = this;
    this.ownerDocument.log?.push({ type: "childList", target: this, removedNodes: [] });
    return node;
  }
  removeChild(node) {
    this.childNodes.splice(this.childNodes.indexOf(node), 1);
    node.parentNode = null;
    this.ownerDocument.log?.push({ type: "childList", target: this, removedNodes: [node] });
    return node;
  }
  replaceChildren(...nodes) {
    for (const old of [...this.childNodes]) this.removeChild(old);
    for (const node of nodes) this.appendChild(node);
  }
  remove() {
    this.parentNode?.removeChild(this);
  }
}

class Text extends Node {
  constructor(doc, data) {
    super(doc);
    this.nodeType = 3;
    this._data = data;
  }
  get data() { return this._data; }
  set data(value) {
    this._data = value;
    this.ownerDocument.log?.push({ type: "characterData", target: this });
  }
  get textContent() { return this._data; }
}

class Element extends Node {
  constructor(doc, tag, ns = null) {
    super(doc);
    this.nodeType = 1;
    this.localName = tag;
    this.tagName = tag.toUpperCase();
    this.namespaceURI = ns;
    this.attrs = new Map();
    this.style = { cssText: "" };
    this.listeners = {};
    this.value = "";
    this.checked = false;
    this.open = false;
    this.type = "";
  }
  get attributes() { return [...this.attrs].map(([name, value]) => ({ name, value })); }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  hasAttribute(name) { return this.attrs.has(name); }
  setAttribute(name, value) {
    this.attrs.set(name, String(value));
    if (name === "type") this.type = String(value);
    this.ownerDocument.log?.push({ type: "attributes", target: this });
  }
  removeAttribute(name) {
    this.attrs.delete(name);
    this.ownerDocument.log?.push({ type: "attributes", target: this });
  }
  matches(selector) {
    return selector.split(",").some((one) => {
      const part = one.trim();
      const attr = /^\[([\w-]+)\]$/.exec(part);
      if (attr) return this.hasAttribute(attr[1]);
      return part === this.localName;
    });
  }
  closest(selector) {
    let el = this;
    while (el && el.nodeType === 1) {
      if (el.matches(selector)) return el;
      el = el.parentNode;
    }
    return null;
  }
  querySelectorAll(selector) {
    const out = [];
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (child.nodeType !== 1) continue;
        if (child.matches(selector)) out.push(child);
        walk(child);
      }
    };
    walk(this);
    return out;
  }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  dispatchEvent(event) {
    event.target ??= this;
    let el = this;
    while (el) {
      for (const fn of el.listeners?.[event.type] ?? []) fn(event);
      el = event.bubbles ? el.parentNode : null;
    }
    return !event.defaultPrevented;
  }
  click() { this.dispatchEvent(new FakeEvent("click", { bubbles: true })); }
  focus() { this.ownerDocument.activeElement = this; }
  getBoundingClientRect() { return this.box ?? { left: 0, top: 0, width: 0, height: 0 }; }
}

class Doc {
  constructor() {
    this.log = [];
    this.parentNode = null;
    this.documentElement = new Element(this, "html");
    this.documentElement.parentNode = this;
    this.body = new Element(this, "body");
    this.documentElement.childNodes.push(this.body);
    this.body.parentNode = this.documentElement;
    this.activeElement = this.body;
  }
  createElement(tag) { return new Element(this, tag); }
  createElementNS(ns, tag) { return new Element(this, tag, ns); }
  createTextNode(text) { return new Text(this, text); }
  querySelectorAll(selector) { return this.body.querySelectorAll(selector); }
  take() {
    const out = this.log;
    this.log = [];
    return out;
  }
}

class FakeEvent {
  constructor(type, init = {}) {
    Object.assign(this, init);
    this.type = type;
    this.defaultPrevented = false;
  }
  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
}
const win = { Event: FakeEvent, MouseEvent: FakeEvent, KeyboardEvent: FakeEvent };

function el(doc, tag, attrs = {}, ...kids) {
  const node = doc.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const kid of kids) node.appendChild(typeof kid === "string" ? doc.createTextNode(kid) : kid);
  return node;
}

/** The tree of `node` as plain data, for comparing two devices. */
function shape(node) {
  if (node.nodeType === 3) return node.data;
  return [node.localName, Object.fromEntries(node.attrs), node.childNodes.map(shape)];
}

/** Two devices: a sharing page with something on it, and a joined one that got its snapshot. */
function devices(code = "a-long-enough-access-code") {
  const host = new Doc();
  const scrub = (text) => (text.includes(code) ? text.split(code).join("••••••") : text);
  const page = el(host, "main", { id: "app" },
    el(host, "h1", {}, "Ofertas"),
    el(host, "ul", { class: "list" }, el(host, "li", {}, "Ana Prueba"), el(host, "li", {}, "Bruno Ejemplo")));
  host.body.appendChild(page);
  host.take();
  const source = new MirrorSource({ doc: host, scrub });
  const viewer = new Doc();
  const view = new MirrorView({ doc: viewer });
  view.apply(JSON.parse(JSON.stringify(source.snapshot())));
  return { host, viewer, source, view, page };
}

/** Send what changed on the sharing page to the joined one, as the relay would (JSON). */
function sync({ host, source, view }) {
  source.record(host.take());
  const delta = source.delta();
  if (delta) view.apply(JSON.parse(JSON.stringify(delta)));
  return delta;
}

// ---- the mirror -------------------------------------------------------------------

test("a joined device gets the whole page first, with the same numbers", () => {
  const d = devices();
  assert.deepEqual(shape(d.viewer.body), shape(d.host.body));
  assert.equal(d.view.count, d.source.count);
  const h1 = d.page.childNodes[0];
  assert.equal(d.view.nodeById(d.source.idOf(h1)).localName, "h1");
});

test("additions, removals, moves, attributes and texts keep both pages identical", () => {
  const d = devices();
  const list = d.page.childNodes[1];
  const [ana, bruno] = list.childNodes;
  list.appendChild(el(d.host, "li", { class: "new" }, "Carla Inventada"));
  list.insertBefore(bruno, ana); // a move, not a copy
  ana.childNodes[0].data = "Ana Prueba (revisada)";
  d.page.childNodes[0].setAttribute("class", "title");
  d.page.childNodes[0].removeAttribute("class");
  d.page.childNodes[0].setAttribute("aria-sort", "ascending");
  const delta = sync(d);
  assert.ok(delta.c && delta.a && delta.t, "children, attributes and texts travel");
  assert.deepEqual(shape(d.viewer.body), shape(d.host.body));
  const movedId = d.source.idOf(bruno);
  assert.equal(d.view.nodeById(movedId).childNodes[0].data, "Bruno Ejemplo", "the moved node keeps its number");
  list.removeChild(ana);
  sync(d);
  assert.deepEqual(shape(d.viewer.body), shape(d.host.body));
  assert.equal(d.view.count, d.source.count, "both forget the removed nodes");
  assert.equal(sync(d), null, "nothing changed: nothing is sent");
});

test("only the attributes that changed are sent again", () => {
  const d = devices();
  const h1 = d.page.childNodes[0];
  h1.setAttribute("class", "a");
  h1.setAttribute("data-x", "1");
  sync(d);
  h1.setAttribute("class", "b");
  h1.removeAttribute("data-x");
  const delta = sync(d);
  assert.deepEqual(delta.a, [[d.source.idOf(h1), { class: "b" }, ["data-x"]]]);
  assert.deepEqual(shape(d.viewer.body), shape(d.host.body));
});

test("a slow joined device may miss states: the next snapshot puts it right", () => {
  const d = devices();
  const list = d.page.childNodes[1];
  list.appendChild(el(d.host, "li", {}, "Carla Inventada"));
  d.source.record(d.host.take());
  d.source.delta(); // lost on the way
  list.childNodes[0].childNodes[0].data = "cambiado";
  d.source.record(d.host.take());
  const next = d.source.delta();
  d.view.apply(JSON.parse(JSON.stringify(next)));
  assert.notEqual(d.view.count, next.n, "its count differs from the sharing device's: it asks again");
  list.childNodes[2].childNodes[0].data = "Carla (cambiada)";
  d.source.record(d.host.take());
  const third = d.source.delta();
  assert.throws(() => d.view.apply(JSON.parse(JSON.stringify(third))), RangeError, "a node it never got: it asks again");
  const again = new MirrorView({ doc: new Doc() });
  again.apply(JSON.parse(JSON.stringify(d.source.snapshot())));
  assert.deepEqual(shape(again.doc.body), shape(d.host.body));
  assert.equal(again.count, d.source.count);
});

test("form values, checked boxes and open dialogs travel as props, not as attributes", () => {
  const d = devices();
  const input = el(d.host, "input", { type: "search" });
  const box = el(d.host, "input", { type: "checkbox" });
  const dialog = el(d.host, "dialog", { open: "" });
  d.page.appendChild(input);
  d.page.appendChild(box);
  d.page.appendChild(dialog);
  sync(d);
  input.value = "Bru";
  box.checked = true;
  dialog.open = true;
  dialog.matches = () => true; // :modal
  const delta = sync(d);
  assert.deepEqual(Object.values(delta.p), [{ v: "Bru" }, { c: 1 }, { o: 1, m: 1 }]);
  const mirrored = d.view.nodeById(d.source.idOf(input));
  assert.equal(mirrored.value, "Bru");
  assert.equal(d.view.nodeById(d.source.idOf(box)).checked, true);
  const opened = d.view.nodeById(d.source.idOf(dialog));
  assert.equal(opened.getAttribute("open"), "", "without showModal the dialog is open by attribute");
  assert.ok(!("open" in attributesOf(dialog)), "a dialog's open state is a prop");
});

test("never mirrored: scripts, the share panel, on* handlers, javascript: links, media sources", () => {
  const d = devices();
  d.host.body.appendChild(el(d.host, "script", {}, "alert(1)"));
  d.host.body.appendChild(el(d.host, "div", { "data-share-skip": "" }, "Contraseña: 1234"));
  d.page.appendChild(el(d.host, "a", { href: "javascript:alert(1)", onclick: "x()" }, "enlace"));
  d.page.appendChild(el(d.host, "audio", { src: "/api/sessions/s1/audio", autoplay: "" }));
  sync(d);
  const text = JSON.stringify(shape(d.viewer.body));
  assert.ok(!text.includes("alert") && !text.includes("1234") && !text.includes("onclick"), text);
  assert.ok(!text.includes("/api/sessions/s1/audio") && !text.includes("autoplay"));
  assert.ok(skipped(el(d.host, "iframe")) && skipped(el(d.host, "noscript")));
});

test("passwords, sensitive parts and the access code are masked before they leave", () => {
  const code = "a-long-enough-access-code";
  const d = devices(code);
  const password = el(d.host, "input", { type: "password", value: "secreto" });
  const secret = el(d.host, "p", { "data-share-sensitive": "" }, "dato reservado");
  d.page.appendChild(password);
  d.page.appendChild(secret);
  d.page.appendChild(el(d.host, "p", { title: `#k=${code}` }, `enlace #k=${code}`));
  sync(d);
  password.value = "otro secreto";
  const delta = sync(d);
  const everything = JSON.stringify(d.source.snapshot()) + JSON.stringify(delta);
  for (const leak of ["secreto", "dato reservado", code]) assert.ok(!everything.includes(leak), leak);
  assert.equal(d.view.nodeById(d.source.idOf(password)).value, MASK);
  assert.deepEqual(propsOf(password), { v: MASK });
});

test("an action of a joined device lands on the same node of the sharing device", () => {
  const d = devices();
  const button = el(d.host, "button", {}, "Ordenar");
  const field = el(d.host, "input", { type: "search" });
  const clicks = [];
  const typed = [];
  button.addEventListener("click", () => clicks.push(1));
  field.addEventListener("input", () => typed.push(field.value));
  d.page.appendChild(button);
  d.page.appendChild(field);
  sync(d);
  // the joined device names the node by its number, as its mirror has it
  const id = d.view.idOf(d.view.nodeById(d.source.idOf(button)));
  assert.ok(d.source.applyInput({ k: "click", node: id }, win));
  assert.ok(d.source.applyInput({ k: "val", node: d.source.idOf(field), v: "Ana" }, win));
  assert.deepEqual([clicks, typed, field.value], [[1], ["Ana"], "Ana"]);
  const keys = [];
  d.host.body.addEventListener("keydown", (event) => keys.push(event.key));
  assert.ok(d.source.applyInput({ k: "key", node: d.source.idOf(field), key: "ArrowDown" }, win));
  assert.deepEqual(keys, ["ArrowDown"]);
  assert.equal(d.source.applyInput({ k: "click", node: 99999 }, win), false, "a node that is gone: nothing happens");
  assert.equal(d.source.applyInput({ k: "dance" }, win), false);
});

test("the pointer and the scroll are said so that another layout can place them", () => {
  const doc = new Doc();
  const box = el(doc, "div");
  box.box = { left: 100, top: 50, width: 200, height: 100 };
  const ids = new Map([[box, 7]]);
  const at = pointerAt(box, 150, 75, (n) => ids.get(n) ?? null);
  assert.deepEqual(at, { node: 7, fx: 0.25, fy: 0.25 });
  const list = { scrollTop: 300, scrollLeft: 0, scrollHeight: 1100, clientHeight: 500 };
  assert.deepEqual(elementScrollOf(list, 3), [3, 300, 0, 0.5]);
  const other = { scrollTop: 0, scrollLeft: 0, scrollHeight: 2500, clientHeight: 500 };
  elementScrollTo(other, [3, 300, 0, 0.5], false);
  assert.equal(other.scrollTop, 1000, "another layout: the same fraction");
  elementScrollTo(other, [3, 300, 0, 0.5], true);
  assert.equal(other.scrollTop, 300, "the same layout: the same offset");
});

test("setChildren reorders in place", () => {
  const doc = new Doc();
  const parent = el(doc, "ul");
  const [a, b, c] = ["a", "b", "c"].map((x) => el(doc, "li", {}, x));
  setChildren(parent, [a, b, c]);
  setChildren(parent, [c, a]);
  assert.deepEqual(parent.childNodes.map((n) => n.childNodes[0].data), ["c", "a"]);
  assert.equal(b.parentNode, null);
});

// ---- the link ------------------------------------------------------------------------

class FakeSocket {
  constructor(url) {
    this.url = url;
    this.sent = [];
    this.readyState = 0;
    FakeSocket.all.push(this);
  }
  open() { this.readyState = 1; this.onopen?.(); }
  receive(message) { this.onmessage?.({ data: JSON.stringify(message) }); }
  send(text) { this.sent.push(JSON.parse(text)); }
  close(code = 1006) { this.readyState = 3; this.onclose?.({ code }); }
}
FakeSocket.all = [];

function timers() {
  const queue = [];
  let clock = 0;
  return {
    now: () => clock,
    setTimeout: (fn, ms) => { queue.push({ fn, at: clock + ms, every: 0 }); return queue.length; },
    clearTimeout: () => {},
    setInterval: (fn, ms) => { queue.push({ fn, at: clock + ms, every: ms }); return queue.length; },
    clearInterval: (i) => { if (queue[i - 1]) queue[i - 1].dead = true; },
    advance(ms) {
      const end = clock + ms;
      for (;;) {
        const next = queue.filter((q) => !q.dead && q.at <= end).sort((x, y) => x.at - y.at)[0];
        if (!next) break;
        clock = next.at;
        if (next.every) next.at += next.every;
        else next.dead = true;
        next.fn();
      }
      clock = end;
    },
  };
}

function link(states, messages, clock = timers()) {
  FakeSocket.all = [];
  return new ShareLink({
    url: "wss://site/api/share", code: "a-long-enough-access-code",
    hello: { type: "join", name: "Demo", pin: "1234", label: "Proyector" },
    openSocket: (url) => new FakeSocket(url), onMessage: (m) => messages.push(m),
    onState: (state, detail) => states.push([state, detail?.code ?? detail?.reason ?? null]),
    timers: clock, now: clock.now, random: () => 0.5,
  });
}

test("the link sends the access code first, never in the address, then the hello", () => {
  const states = [];
  const messages = [];
  const l = link(states, messages);
  l.start();
  const socket = FakeSocket.all[0];
  assert.equal(socket.url, "wss://site/api/share");
  socket.open();
  assert.deepEqual(socket.sent, [
    { type: "auth", access_code: "a-long-enough-access-code" },
    { type: "join", name: "Demo", pin: "1234", label: "Proyector" }]);
  assert.equal(l.send({ type: "in" }), false, "nothing is sent before the welcome");
  socket.receive({ type: "welcome", device: "d1", token: "tok", host: false });
  assert.equal(l.state, "open");
  socket.receive({ type: "roster", devices: [] });
  assert.deepEqual(messages, [{ type: "roster", devices: [] }]);
  assert.ok(l.send({ type: "in", body: {} }));
});

test("a dropped link reconnects with backoff and resumes with its token, no password", () => {
  const states = [];
  const clock = timers();
  const l = link(states, [], clock);
  l.start();
  FakeSocket.all[0].open();
  FakeSocket.all[0].receive({ type: "welcome", device: "d1", token: "tok" });
  FakeSocket.all[0].close(1006); // the tunnel's hourly reset
  assert.equal(l.state, "lost");
  clock.advance(400);
  assert.equal(FakeSocket.all.length, 1, "waits ~0.5 s before the first retry");
  clock.advance(200);
  assert.equal(FakeSocket.all.length, 2);
  FakeSocket.all[1].close(1006); // still down
  clock.advance(1300);
  assert.equal(FakeSocket.all.length, 3, "then ~1 s, and never more");
  FakeSocket.all[2].open();
  assert.deepEqual(FakeSocket.all[2].sent[1], { type: "resume", token: "tok" });
  FakeSocket.all[2].receive({ type: "welcome", device: "d1" });
  assert.equal(l.state, "open");
  assert.equal(l.token, "tok");
  assert.deepEqual([0, 1, 2, 3, 9].map((n) => shareBackoffMs(n, () => 0.5)), [500, 1000, 1000, 1000, BACKOFF_MAX_MS],
    "never more than 1 s between tries: a tunnel reset of a few seconds costs a few seconds");
});

test("#284: the next page of the sharing device resumes with the kept token, no password", () => {
  // study.html / research.html are other documents: the page that loads
  // builds a new link with the token its tab kept (sessionStorage).
  const states = [];
  const l = link(states, []);
  l.token = "kept-token";
  l.start();
  FakeSocket.all[0].open();
  assert.deepEqual(FakeSocket.all[0].sent, [
    { type: "auth", access_code: "a-long-enough-access-code" }, { type: "resume", token: "kept-token" }],
    "the hello (name and password) is never sent again");
  FakeSocket.all[0].receive({ type: "welcome", device: "d1", host: true });
  assert.equal(l.state, "open");
  assert.equal(l.token, "kept-token", "a resume keeps the token it had");
});

test("#284: a link to another site is told apart from the pages of this one", () => {
  const base = "https://site.example/research.html#marco";
  for (const own of ["index.html#/", "#resumen", "study.html", "/research.html", "https://site.example/index.html#/live", "", null, "javascript:void(0)"]) {
    assert.equal(externalUrl(own, base), null, String(own));
  }
  assert.equal(externalUrl("https://doi.org/10.1/x", base), "https://doi.org/10.1/x");
  assert.equal(externalUrl("//other.example/a", base), "https://other.example/a");
  assert.equal(externalUrl("http://site.example/", base), "http://site.example/", "another scheme is another origin");
  assert.equal(externalUrl("mailto:someone@example.org", base), "mailto:someone@example.org");
});

test("a silent link is reopened; a final refusal stops it with its reason", () => {
  const states = [];
  const clock = timers();
  const l = link(states, [], clock);
  l.start();
  FakeSocket.all[0].open();
  FakeSocket.all[0].receive({ type: "welcome", device: "d1", token: "tok" });
  clock.advance(15000);
  assert.deepEqual(FakeSocket.all[0].sent.at(-1), { type: "ping" });
  clock.advance(45000); // no answer for longer than SILENT_MS
  assert.equal(l.state === "lost" || FakeSocket.all.length > 1, true);
  clock.advance(1000);
  const last = FakeSocket.all.at(-1);
  last.open();
  last.receive({ type: "error", code: "too_many_attempts", retry_after: 60 });
  last.close(4429);
  assert.deepEqual(states.at(-1), ["ended", "too_many_attempts"]);
  const count = FakeSocket.all.length;
  clock.advance(60000);
  assert.equal(FakeSocket.all.length, count, "no retry after a final refusal");
  for (const code of [4401, 4404, 4410, 4413, 4429]) assert.ok(FINAL_CODES.has(code));
  assert.ok(!FINAL_CODES.has(1006) && !FINAL_CODES.has(1011));
});

test("leaving tells the relay and never reconnects", () => {
  const states = [];
  const clock = timers();
  const l = link(states, [], clock);
  l.start();
  FakeSocket.all[0].open();
  FakeSocket.all[0].receive({ type: "welcome", device: "d1", token: "tok" });
  const socket = FakeSocket.all[0];
  l.stop();
  assert.deepEqual(socket.sent.at(-1), { type: "leave" });
  clock.advance(30000);
  assert.equal(FakeSocket.all.length, 1);
});

// ---- the pure helpers and the texts ------------------------------------------------------

test("the forms' rules match the server's", () => {
  assert.equal(randomPin(() => 0), "0000");
  assert.equal(randomPin(() => 0.99999), "9999");
  assert.match(randomPin(() => 0.0421), /^\d{4}$/);
  assert.ok(validPin("0420") && !validPin("123") && !validPin("12a4") && !validPin("12345"));
  assert.ok(validName("Sala Uno") && validName("Presentación (1)") && !validName("") && !validName("a/b") && !validName("x".repeat(41)));
  assert.equal(cleanLabel(" Tableta\n de  prueba "), "Tableta de prueba");
  assert.equal(deviceKind("Mozilla/5.0 (iPad; CPU OS 17_0)"), "tablet");
  assert.equal(deviceKind("Mozilla/5.0 (Linux; Android 14; Pixel) Mobile"), "phone");
  assert.equal(deviceKind("Mozilla/5.0 (X11; Linux x86_64)"), "computer");
  assert.equal(deviceKind("Mozilla/5.0 (Macintosh)", 5), "tablet");
  assert.equal(socketUrl({ protocol: "https:", host: "rag.example" }), "wss://rag.example/api/share");
  assert.equal(statusKey({ state: "lost", host: true }), "share.status.lost");
  assert.equal(statusKey({ state: "open", host: false, hostOnline: false }), "share.status.host_away");
  assert.equal(statusKey({ state: "open", host: true }), null);
  assert.equal(refusalKey("nope"), "share.error.other");
  assert.equal(endedKey("removed"), "share.ended.removed");
  assert.equal(sameOriginApi("/api/sessions/s1/audio", { href: "https://x.test/", origin: "https://x.test" }), "/api/sessions/s1/audio");
  assert.equal(sameOriginApi("https://other.test/api/a", { href: "https://x.test/", origin: "https://x.test" }), null);
});

test("every text of «Compartir pantalla» exists in both languages and keeps the word rules", () => {
  const keys = Object.keys(es).filter((k) => k.startsWith("share."));
  assert.ok(keys.length > 60, `${keys.length}`);
  assert.deepEqual(Object.keys(en).filter((k) => k.startsWith("share.")).sort(), [...keys].sort());
  for (const key of keys) {
    assert.doesNotMatch(es[key], /\bequipo|ordenador|\bPC\b|vídeo en directo/i, key);
    assert.doesNotMatch(en[key], /computer|workstation|machine/i, key);
  }
  const codes = ["no_match", "too_many_attempts", "full", "busy", "invalid_input", "ended", "unauthorized", "other"];
  for (const code of codes) assert.ok(es[`share.error.${code}`] && en[`share.error.${code}`], code);
  for (const reason of ["stopped", "removed", "expired", "host_left", "too_many_attempts", "left", "gone", "replaced", "other"]) {
    assert.ok(es[`share.ended.${reason}`] && en[`share.ended.${reason}`], reason);
  }
  assert.equal(es["share.error.no_match"], "No hay ninguna sesión compartida con ese nombre y esa contraseña.",
    "one answer for a wrong name and a wrong password");
});

// ---- the sound on one device ----------------------------------------------------------------

test("the sound device plays the clips the sharing device sends, and stops them", async () => {
  const started = [];
  const stopped = [];
  const context = {
    state: "running", currentTime: 10, destination: {},
    decodeAudioData: async (bytes) => ({ bytes: bytes.byteLength }),
    createBufferSource: () => ({
      connect() {}, start: (at, offset) => started.push([at, offset]), stop: () => stopped.push(1),
    }),
  };
  const player = new SoundPlayer({ context: () => context });
  player.handle({ k: "clip", id: 1, data: Buffer.from("RIFF0000").toString("base64") });
  player.handle({ k: "play", id: 1, play: 5, delay: 0.5, offset: 0 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.deepEqual(started, [[10.5, 0]]);
  player.handle({ k: "stop", play: 5 });
  assert.deepEqual(stopped, [1]);
});

// ---- review fixes (#278) ----------------------------------------------------------------------

test("typing on a joined device at 150–300 ms loses nothing: an older echo never overwrites the field", async () => {
  // The sharing device's page: a search field that the joined device types into.
  const host = new Doc();
  const field = el(host, "input", { type: "search" });
  host.body.appendChild(field);
  host.take();
  const source = new MirrorSource({ doc: host });
  const viewer = new Doc();
  let typing = null;
  const HOLD = 1200;
  let clock = 0;
  const view = new MirrorView({ doc: viewer, hold: (id) => id === typing?.node && clock - typing.at < HOLD });
  view.apply(JSON.parse(JSON.stringify(source.snapshot())));
  const id = source.idOf(field);
  const mine = view.nodeById(id);
  // A one-way delay of 75–150 ms, in order (a socket keeps order).
  const wire = [];
  const later = (at, fn) => wire.push({ at, fn });
  const run = (until) => {
    wire.sort((a, b) => a.at - b.at);
    while (wire.length && wire[0].at <= until) {
      const next = wire.shift();
      clock = Math.max(clock, next.at);
      next.fn();
    }
    clock = until;
  };
  let last = 0;
  for (const [i, ch] of [..."abcdef"].entries()) {
    run(i * 60); // a key every 60 ms
    mine.value += ch; // the browser types it here at once
    typing = { node: id, at: clock };
    const value = mine.value;
    const reach = Math.max(last, clock + 75 + (i % 2) * 75);
    last = reach;
    later(reach, () => {
      source.applyInput({ k: "val", node: id, v: value }, win);
      source.record(host.take());
      const delta = source.delta();
      const echo = JSON.parse(JSON.stringify(delta));
      later(clock + 75 + ((i + 1) % 2) * 75, () => view.apply(echo));
    });
  }
  run(5 * 60 + 400);
  assert.equal(mine.value, "abcdef", "no echo while typing went backwards");
  assert.equal(field.value, "abcdef", "the sharing device has every key");
  run(5 * 60 + HOLD + 100);
  view.release(id);
  assert.equal(mine.value, "abcdef");
});

test("a held echo is put back once typing stopped, if the sharing device changed it", () => {
  const d = devices();
  const field = el(d.host, "input", { type: "search" });
  d.page.appendChild(field);
  sync(d);
  const id = d.source.idOf(field);
  let holding = true;
  d.view.hold = () => holding;
  field.value = "otra cosa"; // the page itself changed it (a reset)
  sync(d);
  assert.equal(d.view.nodeById(id).value, "", "held while typing");
  holding = false;
  d.view.release(id);
  assert.equal(d.view.nodeById(id).value, "otra cosa");
});

test("snapshots carry an epoch: a page load numbers nodes anew", () => {
  const first = new MirrorSource({ doc: new Doc() });
  const second = new MirrorSource({ doc: new Doc() });
  assert.ok(first.snapshot().e && first.epoch !== second.epoch);
});

test("hidden, disabled, panel and file controls cannot be acted on from a joined device", () => {
  const d = devices();
  const hidden = el(d.host, "div", { hidden: "" }, el(d.host, "button", {}, "Borrar"));
  const disabled = el(d.host, "input", { type: "search" });
  disabled.disabled = true;
  const file = el(d.host, "input", { type: "file" });
  const panel = el(d.host, "dialog", { "data-share-skip": "" });
  panel.open = true;
  panel.setAttribute("open", "");
  let closed = 0;
  panel.close = () => { closed += 1; };
  const clicks = [];
  hidden.childNodes[0].addEventListener("click", () => clicks.push(1));
  for (const node of [hidden, disabled, file]) d.page.appendChild(node);
  d.host.body.appendChild(panel);
  sync(d);
  const button = hidden.childNodes[0];
  assert.equal(actionable(button), false);
  assert.equal(d.source.applyInput({ k: "click", node: d.source.idOf(button) }, win), false);
  assert.equal(d.source.applyInput({ k: "val", node: d.source.idOf(disabled), v: "x" }, win), false);
  assert.equal(d.source.applyInput({ k: "val", node: d.source.idOf(file), v: "C:/x" }, win), false);
  assert.equal(d.source.applyInput({ k: "click" }, win), false, "no node: nothing");
  d.source.applyInput({ k: "key", key: "Escape" }, win);
  assert.deepEqual([clicks, closed, disabled.value], [[], 0, ""], "the share panel stays open");
});

test("an <audio> muted because its sound went elsewhere is unmuted when the sound comes back", () => {
  class Media {
    constructor() { this.dataset = {}; this.muted = false; this.src = "/api/sessions/s1/audio"; this.currentTime = 0; }
    addEventListener() {}
  }
  Media.prototype.play = function play() { return Promise.resolve(); };
  const original = Media.prototype.play;
  let remote = true;
  const sent = [];
  const bridge = installSoundBridge({
    win: { HTMLMediaElement: Media, location: { href: "https://x.test/", origin: "https://x.test" } },
    remote: () => remote, send: (body) => sent.push(body.k),
  });
  const audio = new Media();
  audio.play();
  assert.equal(audio.muted, true);
  remote = false;
  audio.play();
  assert.equal(audio.muted, false, "back on this device: heard again");
  remote = true;
  audio.play();
  bridge.uninstall();
  assert.equal(audio.muted, false, "sharing ended: never left muted");
  assert.equal(Media.prototype.play, original);
  assert.deepEqual(sent, ["media", "media"]);
});

test("typing in two fields within the hold keeps both: each field has its own hold", () => {
  const d = devices();
  const first = el(d.host, "input", { type: "search" });
  const second = el(d.host, "input", { type: "text" });
  d.page.appendChild(first);
  d.page.appendChild(second);
  sync(d);
  const typedAt = new Map();
  let now = 0;
  d.view.hold = (id) => now - (typedAt.get(id) ?? -1e9) < 1200;
  const [a, b] = [first, second].map((f) => d.view.nodeById(d.source.idOf(f)));
  a.value = "Ana";
  typedAt.set(d.source.idOf(first), now);
  now = 400;
  b.value = "Bar";
  typedAt.set(d.source.idOf(second), now);
  first.value = "An"; // an older echo of the first field
  second.value = "Ba";
  sync(d);
  assert.deepEqual([a.value, b.value], ["Ana", "Bar"]);
});
