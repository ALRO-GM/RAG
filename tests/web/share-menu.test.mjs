// «Compartir pantalla» (#278): the menu button decides by the layout of the
// device that pressed it. A tablet joined in its own layout drives a
// desktop that shares: its press must open the drawer (nav-open, which the
// tablet's mirror shows at tablet width), never fold the desktop's sidebar
// (nav-collapsed) in front of the audience. Closing works the same way.
import test from "node:test";
import assert from "node:assert/strict";

class ClassList {
  constructor() { this.set = new Set(); }
  contains(c) { return this.set.has(c); }
  toggle(c, on) { const want = on ?? !this.set.has(c); if (want) this.set.add(c); else this.set.delete(c); return want; }
  add(c) { this.set.add(c); }
  remove(c) { this.set.delete(c); }
}

class Fake {
  constructor(id) { this.id = id; this.listeners = {}; this.attrs = {}; this.classList = new ClassList(); }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  focus() {}
  fire(type, extra = {}) { for (const fn of this.listeners[type] ?? []) fn({ target: this, preventDefault() {}, ...extra }); }
}

const elements = Object.fromEntries(["nav-toggle", "nav-more", "scrim", "sidebar", "theme-toggle", "palette-open", "palette-key"].map((id) => [id, new Fake(id)]));
const root = new Fake("html");
root.dataset = {};
const docListeners = {};
globalThis.document = {
  documentElement: root,
  getElementById: (id) => elements[id] ?? null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: (type, fn) => { (docListeners[type] ??= []).push(fn); },
  createElement: (tag) => new Fake(tag),
  createElementNS: (_ns, tag) => new Fake(tag),
  createTextNode: (text) => ({ textContent: text }),
};
globalThis.window = { addEventListener() {} };
globalThis.location = { pathname: "/", hash: "#/" };
let desktop = true; // the sharing device is a desktop
globalThis.matchMedia = () => ({ matches: desktop, addEventListener() {} });

const { actingAs, installShell, isWide } = await import("../../transcriber/adapters/inbound/web/js/shell.js");
installShell();
const press = () => elements["nav-toggle"].fire("click");

test("the layout that counts is the acting device's while its action is applied", () => {
  assert.equal(isWide(), true);
  assert.equal(actingAs(false, () => isWide()), false);
  assert.equal(actingAs(false, () => actingAs(true, () => isWide())), true, "nested, then restored");
  assert.equal(isWide(), true, "restored after");
  assert.equal(actingAs(null, () => isWide()), true, "no layout said: this page's");
  assert.throws(() => actingAs(false, () => { throw new Error("x"); }));
  assert.equal(isWide(), true, "restored after an error");
});

test("a tablet's menu press opens the drawer on a desktop that shares, never folds its sidebar", () => {
  actingAs(false, press);
  assert.ok(root.classList.contains("nav-open"), "the drawer is open: the tablet's mirror shows it");
  assert.ok(!root.classList.contains("nav-collapsed"), "the projector's sidebar did not fold");
  assert.equal(elements["nav-toggle"].getAttribute("aria-expanded"), "true");
  actingAs(false, press);
  assert.ok(!root.classList.contains("nav-open"), "a second press closes it");
});

test("the drawer closes from the tablet by the scrim, a link or Escape", () => {
  actingAs(false, press);
  actingAs(false, () => elements.scrim.fire("click"));
  assert.ok(!root.classList.contains("nav-open"), "scrim");
  actingAs(false, press);
  actingAs(false, () => elements.sidebar.fire("click", { target: { closest: () => ({}) } }));
  assert.ok(!root.classList.contains("nav-open"), "a link");
  actingAs(false, press);
  for (const fn of docListeners.keydown ?? []) fn({ key: "Escape", preventDefault() {} });
  assert.ok(!root.classList.contains("nav-open"), "Escape");
});

test("a press on the desktop itself, or from a wide joined device, folds the sidebar as before", () => {
  press();
  assert.ok(root.classList.contains("nav-collapsed"));
  actingAs(true, press);
  assert.ok(!root.classList.contains("nav-collapsed"));
  assert.ok(!root.classList.contains("nav-open"));
});
