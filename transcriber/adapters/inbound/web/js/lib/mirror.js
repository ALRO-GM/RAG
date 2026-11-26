// «Compartir pantalla» (#278): the page of the sharing device mirrored as
// DOM, never as pixels. No browser globals here: the document (and, for
// events, its window) is injected, so node --test runs it on a stand-in DOM.
//
// MirrorSource (the sharing device) gives every node of <body> it shows a
// number, serialises the page once (snapshot) and then sends only what
// changed (delta): per changed element its whole attribute set, per changed
// text its text, per element whose children changed the list of its
// children (a known child by number, a new one serialised whole). A delta
// says how things ARE, not what happened, so applying it twice or after a
// lost intermediate state gives the same page: a slow viewer can skip
// states, and anything a viewer cannot place makes it ask for the whole
// page again. Form values, checked boxes and open dialogs are properties,
// not attributes: they travel as `p` (props) and are compared every tick.
//
// MirrorView (a joined device) builds the same tree in another document (an
// <iframe> without scripts) and keeps the same numbers, so an action on a
// joined device names the node by number and the sharing device applies it
// to its own node (applyInput).
//
// Never mirrored: <script>, <noscript>, <template>, <iframe>, <object>,
// <embed>, anything marked data-share-skip (the «Compartir pantalla» panel
// itself), on* attributes, javascript: addresses, the src of <audio> and
// <video> (the sound plays on one device only, see share-sound.js). Values
// of password fields and of anything marked data-share-sensitive are masked;
// `scrub` takes out the access code wherever it would appear.

export const MASK = "••••••";
export const SVG_NS = "http://www.w3.org/2000/svg";
const SKIP_TAGS = new Set(["script", "noscript", "template", "iframe", "object", "embed", "link", "style", "meta", "base"]);
const MEDIA_TAGS = new Set(["audio", "video", "source", "track"]);
const URL_ATTRS = new Set(["href", "src", "action", "formaction", "xlink:href", "poster"]);
const DROP_ATTRS = new Set(["srcdoc", "autofocus", "nonce", "autoplay"]);
const FORM_TAGS = new Set(["input", "textarea", "select"]);
/** What a click on a joined device is sent to: the control, not the icon inside it. */
export const ACTIONABLE = "a[href],button,summary,label,input,select,textarea,option,dialog,[role=button],[role=option],[role=tab],[role=menuitem],[role=switch],[tabindex]";

const ELEMENT = 1;
const TEXT = 3;

function tagOf(node) {
  return String(node.localName ?? node.tagName ?? "").toLowerCase();
}

/** Whether `node` (and everything inside it) is left out of the mirror. */
export function skipped(node) {
  if (node.nodeType === TEXT) return false;
  if (node.nodeType !== ELEMENT) return true;
  return SKIP_TAGS.has(tagOf(node)) || node.hasAttribute?.("data-share-skip") === true;
}

/** Whether the value or text of `el` must reach viewers masked. */
export function sensitive(el) {
  if (!el || el.nodeType !== ELEMENT) return false;
  if (tagOf(el) === "input" && String(el.getAttribute?.("type") ?? el.type ?? "").toLowerCase() === "password") return true;
  return Boolean(el.closest?.("[data-share-sensitive]"));
}

function mirroredChildren(node) {
  return [...(node.childNodes ?? [])].filter((child) => !skipped(child));
}

/** The attributes of `el` that viewers get, cleaned. */
export function attributesOf(el, scrub = (s) => s) {
  const tag = tagOf(el);
  const out = {};
  for (const { name, value } of [...(el.attributes ?? [])]) {
    const key = name.toLowerCase();
    if (key.startsWith("on") || DROP_ATTRS.has(key)) continue;
    if (URL_ATTRS.has(key) && /^\s*(javascript|vbscript|data:text\/html)/i.test(value)) continue;
    if (MEDIA_TAGS.has(tag) && (key === "src" || key === "srcset")) continue;
    if (tag === "dialog" && key === "open") continue; // a prop: modal or not
    if (FORM_TAGS.has(tag) && key === "value" && sensitive(el)) {
      out[name] = MASK;
      continue;
    }
    out[name] = scrub(value);
  }
  return out;
}

/** The props of `el` (value, checked, an open dialog), or null when it has none. */
export function propsOf(el, scrub = (s) => s) {
  const tag = tagOf(el);
  if (tag === "dialog") return { o: el.open ? 1 : 0, m: el.open && el.matches?.(":modal") ? 1 : 0 };
  if (!FORM_TAGS.has(tag)) return null;
  const type = String(el.type ?? "").toLowerCase();
  if (type === "file") return null;
  if (type === "checkbox" || type === "radio") return { c: el.checked ? 1 : 0 };
  if (sensitive(el)) return { v: el.value ? MASK : "" };
  return { v: scrub(String(el.value ?? "")) };
}

/**
 * The sharing device's side. `send(body)` gets every message body; call
 * `record(mutations)` with MutationObserver records and `tick()` on a timer.
 */
export class MirrorSource {
  constructor({ doc, scrub = (s) => s }) {
    this.doc = doc;
    this.scrub = scrub;
    this.ids = new WeakMap();
    this.nodes = new Map();
    this.next = 1;
    this.dirtyAttrs = new Set();
    this.dirtyText = new Set();
    this.dirtyKids = new Set();
    this.removed = new Set();
    this.lastProps = new Map(); // id -> JSON of the props last sent
    this.visited = null;
  }

  idOf(node) {
    return this.ids.get(node) ?? null;
  }

  nodeById(id) {
    return this.nodes.get(id) ?? null;
  }

  /** How many nodes the viewers hold (they check theirs against it). */
  get count() {
    return this.nodes.size;
  }

  _assign(node) {
    let id = this.ids.get(node);
    if (id === undefined) {
      id = this.next++;
      this.ids.set(node, id);
    }
    this.nodes.set(id, node);
    return id;
  }

  _forget(node) {
    const id = this.ids.get(node);
    if (id !== undefined) {
      this.ids.delete(node);
      this.nodes.delete(id);
      this.lastProps.delete(id);
    }
    for (const child of node.childNodes ?? []) this._forget(child);
  }

  _text(node) {
    return sensitive(node.parentNode) ? MASK : this.scrub(String(node.data ?? node.textContent ?? ""));
  }

  /** Serialise `node` whole: [id, text] or [id, tag, attrs, children(, "s" for SVG)]. */
  serialize(node) {
    const id = this._assign(node);
    this.visited?.add(id);
    if (node.nodeType === TEXT) return [id, this._text(node)];
    const out = [id, tagOf(node), attributesOf(node, this.scrub), mirroredChildren(node).map((c) => this.serialize(c))];
    if (node.namespaceURI === SVG_NS) out.push("s");
    return out;
  }

  /** Every prop of the page (snapshot) or only the changed ones. */
  props(all = false) {
    const out = {};
    let any = false;
    for (const el of this.doc.body?.querySelectorAll?.("input,textarea,select,dialog") ?? []) {
      const id = this.ids.get(el);
      if (id === undefined) continue;
      const props = propsOf(el, this.scrub);
      if (!props) continue;
      const json = JSON.stringify(props);
      if (!all && this.lastProps.get(id) === json) continue;
      this.lastProps.set(id, json);
      out[id] = props;
      any = true;
    }
    return any || all ? out : null;
  }

  /** The whole page now: { tree, p }. Pending changes are folded in. */
  snapshot() {
    this.dirtyAttrs.clear();
    this.dirtyText.clear();
    this.dirtyKids.clear();
    this.removed.clear();
    this.lastProps.clear();
    this.visited = new Set();
    const tree = this.serialize(this.doc.body);
    // Exactly what was sent is held: viewers check their count against it.
    for (const [id, node] of [...this.nodes]) {
      if (!this.visited.has(id)) {
        this.nodes.delete(id);
        this.ids.delete(node);
      }
    }
    this.visited = null;
    return { tree, p: this.props(true), n: this.count };
  }

  /** Take MutationObserver records. */
  record(mutations) {
    for (const m of mutations) {
      if (m.type === "attributes") this.dirtyAttrs.add(m.target);
      else if (m.type === "characterData") this.dirtyText.add(m.target);
      else if (m.type === "childList") {
        this.dirtyKids.add(m.target);
        for (const node of m.removedNodes ?? []) this.removed.add(node);
      }
    }
  }

  /** What changed since the last call, as a delta body ({k: "d", ...}), or null. */
  delta() {
    const body = { k: "d" };
    const kids = [];
    for (const parent of this.dirtyKids) {
      const id = this.ids.get(parent);
      if (id === undefined || !parent.isConnected) continue;
      kids.push([id, mirroredChildren(parent).map((c) => (this.ids.has(c) ? this.ids.get(c) : this.serialize(c)))]);
    }
    const attrs = [];
    for (const el of this.dirtyAttrs) {
      const id = this.ids.get(el);
      if (id !== undefined && el.isConnected) attrs.push([id, attributesOf(el, this.scrub)]);
    }
    const texts = [];
    for (const node of this.dirtyText) {
      const id = this.ids.get(node);
      if (id !== undefined && node.isConnected) texts.push([id, this._text(node)]);
    }
    for (const node of this.removed) if (!node.isConnected) this._forget(node);
    this.dirtyKids.clear();
    this.dirtyAttrs.clear();
    this.dirtyText.clear();
    this.removed.clear();
    const props = this.props();
    if (kids.length) body.c = kids;
    if (attrs.length) body.a = attrs;
    if (texts.length) body.t = texts;
    if (props) body.p = props;
    if (!kids.length && !attrs.length && !texts.length && !props) return null;
    body.n = this.count;
    return body;
  }

  /**
   * Apply an action of a joined device ({k: "click" | "key" | "val" |
   * "focus" | "sc", node, ...}) to this page, as if it happened here.
   * Returns true when it was applied. `win` gives the event classes.
   */
  applyInput(body, win) {
    const el = body?.node ? this.nodeById(body.node) : null;
    if (body?.node && (!el || !el.isConnected)) return false;
    switch (body?.k) {
      case "click": {
        el.focus?.({ preventScroll: true });
        if (typeof el.click === "function") el.click();
        else el.dispatchEvent(new win.MouseEvent("click", { bubbles: true, cancelable: true }));
        return true;
      }
      case "val": {
        if (!FORM_TAGS.has(tagOf(el))) return false;
        if (body.c !== undefined) el.checked = Boolean(body.c);
        else el.value = String(body.v ?? "");
        el.dispatchEvent(new win.Event("input", { bubbles: true }));
        el.dispatchEvent(new win.Event("change", { bubbles: true }));
        return true;
      }
      case "focus":
        el?.focus?.({ preventScroll: true });
        return Boolean(el);
      case "key":
        return this._key(body, el, win);
      default:
        return false;
    }
  }

  _key(body, el, win) {
    const target = el ?? this.doc.activeElement ?? this.doc.body;
    const init = {
      key: String(body.key ?? ""), code: String(body.code ?? ""), bubbles: true, cancelable: true,
      ctrlKey: Boolean(body.ctrl), metaKey: Boolean(body.meta), shiftKey: Boolean(body.shift), altKey: Boolean(body.alt),
    };
    const go = target.dispatchEvent(new win.KeyboardEvent("keydown", init));
    target.dispatchEvent(new win.KeyboardEvent("keyup", init));
    if (!go) return true;
    if (init.key === "Escape") {
      const open = [...(this.doc.querySelectorAll?.("dialog[open]") ?? [])].pop();
      if (open && open.dispatchEvent(new win.Event("cancel", { cancelable: true }))) open.close?.();
    } else if (init.key === "Enter" && tagOf(target) === "input" && target.form) {
      // With a submit button the joined device already sent its click (implicit submission).
      if (!target.form.querySelector?.("button:not([type=button]):not([type=reset]),input[type=submit]")) target.form.requestSubmit?.();
    }
    return true;
  }
}

/** A joined device's side: builds and updates the mirror in `doc` (an iframe's document). */
export class MirrorView {
  constructor({ doc }) {
    this.doc = doc;
    this.nodes = new Map();
    this.ids = new WeakMap();
  }

  idOf(node) {
    return this.ids.get(node) ?? null;
  }

  nodeById(id) {
    return this.nodes.get(id) ?? null;
  }

  get count() {
    return this.nodes.size;
  }

  _keep(id, node) {
    this.nodes.set(id, node);
    this.ids.set(node, id);
    return node;
  }

  _forget(node) {
    const id = this.ids.get(node);
    if (id !== undefined && this.nodes.get(id) === node) this.nodes.delete(id);
    this.ids.delete(node);
    for (const child of node.childNodes ?? []) this._forget(child);
  }

  /** Build the node of a serialised one (or move the known node of a number). */
  build(entry) {
    if (typeof entry === "number") {
      const known = this.nodes.get(entry);
      if (!known) throw new RangeError(`unknown node ${entry}`);
      return known;
    }
    const [id, second, attrs, kids, ns] = entry;
    if (entry.length === 2) return this._keep(id, this.doc.createTextNode(String(second)));
    const el = ns === "s" ? this.doc.createElementNS(SVG_NS, second) : this.doc.createElement(second);
    this._keep(id, el);
    setAttributes(el, attrs);
    for (const kid of kids) el.appendChild(this.build(kid));
    return el;
  }

  /**
   * Apply a snapshot ({tree, p}) or a delta ({c, a, t, p}). Throws a
   * RangeError when a delta names a node this view does not hold: the
   * caller then asks for the whole page.
   */
  apply(body) {
    if (body.tree) {
      this.nodes.clear();
      this.ids = new WeakMap();
      const [id, , attrs, kids] = body.tree;
      const target = this.doc.body;
      this._keep(id, target);
      setAttributes(target, attrs);
      target.replaceChildren(...kids.map((kid) => this.build(kid)));
    }
    const gone = [];
    for (const [id, kids] of body.c ?? []) {
      const parent = this.nodes.get(id);
      if (!parent) throw new RangeError(`unknown node ${id}`);
      const wanted = kids.map((kid) => this.build(kid));
      const keep = new Set(wanted);
      for (const old of [...parent.childNodes]) if (!keep.has(old)) gone.push(old);
      setChildren(parent, wanted);
    }
    for (const [id, attrs] of body.a ?? []) {
      const el = this.nodes.get(id);
      if (!el) throw new RangeError(`unknown node ${id}`);
      setAttributes(el, attrs);
    }
    for (const [id, text] of body.t ?? []) {
      const node = this.nodes.get(id);
      if (!node) throw new RangeError(`unknown node ${id}`);
      node.data = text;
    }
    for (const node of gone) if (!node.isConnected) this._forget(node);
    for (const [id, props] of Object.entries(body.p ?? {})) {
      const el = this.nodes.get(Number(id));
      if (el) setProps(el, props);
    }
  }
}

/** Make `el`'s attributes exactly `attrs` (style through the CSSOM: CSP allows it). */
export function setAttributes(el, attrs) {
  for (const { name } of [...(el.attributes ?? [])]) {
    if (!(name in attrs)) {
      if (name === "style" && el.style) el.style.cssText = "";
      el.removeAttribute(name);
    }
  }
  for (const [name, value] of Object.entries(attrs)) {
    if (el.getAttribute(name) === value) continue;
    if (name === "style" && el.style) el.style.cssText = value;
    else {
      try {
        el.setAttribute(name, value);
      } catch {
        // a name the browser refuses: left out
      }
    }
  }
}

/** Make `parent`'s children exactly `wanted`, moving as little as possible. */
export function setChildren(parent, wanted) {
  wanted.forEach((node, i) => {
    const at = parent.childNodes[i] ?? null;
    if (at !== node) parent.insertBefore(node, at);
  });
  while (parent.childNodes.length > wanted.length) parent.removeChild(parent.childNodes[parent.childNodes.length - 1]);
}

/** Apply props to a mirrored element: value, checked, or a dialog open (modal or not). */
export function setProps(el, props) {
  if ("v" in props && el.value !== props.v) el.value = props.v;
  if ("c" in props) el.checked = Boolean(props.c);
  if ("o" in props) {
    try {
      if (props.o && !el.open) {
        if (props.m && el.showModal) el.showModal();
        else if (el.show) el.show();
        else el.setAttribute("open", "");
      } else if (!props.o && el.open) {
        if (el.close) el.close();
        else el.removeAttribute("open");
      }
    } catch {
      if (props.o) el.setAttribute("open", "");
    }
  }
}

/**
 * Where a pointer is, said so that another layout can place it: the
 * nearest mirrored element and the fraction of its box ({node, fx, fy}).
 */
export function pointerAt(target, clientX, clientY, idOf) {
  let el = target;
  while (el && idOf(el) === null) el = el.parentNode;
  if (!el || typeof el.getBoundingClientRect !== "function") return null;
  const box = el.getBoundingClientRect();
  const fx = box.width ? (clientX - box.left) / box.width : 0;
  const fy = box.height ? (clientY - box.top) / box.height : 0;
  return { node: idOf(el), fx: round(fx), fy: round(fy) };
}

/**
 * The scroll of a document said two ways: its offsets and fractions (the
 * same layout) and an anchor (the element at the top of the window and how
 * far into it the window starts) for another layout.
 */
export function scrollOf(doc, idOf) {
  const scroller = doc.scrollingElement ?? doc.documentElement;
  const win = doc.defaultView;
  const top = scroller.scrollTop;
  const range = Math.max(scroller.scrollHeight - scroller.clientHeight, 0);
  const out = { top: Math.round(top), left: Math.round(scroller.scrollLeft), ry: range ? round(top / range) : 0 };
  const probe = doc.elementFromPoint?.((win?.innerWidth ?? 0) / 2, Math.min(80, (win?.innerHeight ?? 0) / 4));
  let el = probe;
  while (el && idOf(el) === null) el = el.parentNode;
  if (el && el !== doc.body && typeof el.getBoundingClientRect === "function") {
    const box = el.getBoundingClientRect();
    out.a = idOf(el);
    out.af = box.height ? round(-box.top / box.height) : 0;
  }
  return out;
}

/** Scroll a mirror's document to `state`: by offset in the same layout, by anchor in another. */
export function scrollTo(doc, state, sameLayout, nodeById) {
  const scroller = doc.scrollingElement ?? doc.documentElement;
  if (!scroller || !state) return;
  let top = state.top;
  if (!sameLayout) {
    const anchor = state.a ? nodeById(state.a) : null;
    if (anchor?.getBoundingClientRect) {
      const box = anchor.getBoundingClientRect();
      top = scroller.scrollTop + box.top + state.af * box.height;
    } else {
      top = state.ry * Math.max(scroller.scrollHeight - scroller.clientHeight, 0);
    }
  }
  scroller.scrollTop = Math.max(0, Math.round(top));
  if (sameLayout) scroller.scrollLeft = state.left ?? 0;
}

/** One scrolled element: [id, top, left, ry] (ry: the fraction of its range). */
export function elementScrollOf(el, id) {
  const range = Math.max(el.scrollHeight - el.clientHeight, 0);
  return [id, Math.round(el.scrollTop), Math.round(el.scrollLeft), range ? round(el.scrollTop / range) : 0];
}

/** Scroll a mirrored element to [id, top, left, ry]: by offset in the same layout, by fraction in another. */
export function elementScrollTo(el, [, top, left, ry], sameLayout) {
  if (sameLayout) {
    el.scrollTop = top;
    el.scrollLeft = left;
  } else {
    el.scrollTop = Math.round(ry * Math.max(el.scrollHeight - el.clientHeight, 0));
  }
}

function round(x) {
  return Math.round(x * 1000) / 1000;
}
