// The «Análisis» page driven through a minimal stand-in DOM: what the
// person changes in the job form survives the 3 s refresh (#115, #121).
import test from "node:test";
import assert from "node:assert/strict";

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
    for (const key of ["id", "type", "value", "className", "textContent", "placeholder", "rows", "href", "htmlFor", "title"]) {
      this[key] = "";
    }
    this.checked = false;
    this.hidden = false;
    this.disabled = false;
    this.open = false;
  }

  setAttribute(key, value) {
    this.attributes[key] = value;
  }

  addEventListener(type, listener) {
    (this.listeners[type] ??= []).push(listener);
  }

  dispatch(type) {
    for (const listener of this.listeners[type] ?? []) listener({ target: this });
  }

  append(...nodes) {
    this.children.push(...nodes);
  }

  replaceChildren(...nodes) {
    this.children = nodes;
  }

  contains(node) {
    return node === this || this.children.some((child) => child.contains?.(node));
  }

  find(id) {
    if (this.id === id) return this;
    for (const child of this.children) {
      const found = child.find?.(id);
      if (found) return found;
    }
    return null;
  }
}

/** Open the page on a running analysis whose stored job allows Claude; return its parts. */
async function openPage() {
  const timers = [];
  const saved = {
    document: globalThis.document, Node: globalThis.Node,
    setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout,
  };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  globalThis.setTimeout = (callback) => timers.push(callback);
  globalThis.clearTimeout = () => {};
  const restore = () => Object.assign(globalThis, saved);
  try {
    const { analysisView } = await import("../../transcriber/adapters/inbound/web/js/views/analysis.js");
    const run = {
      session_id: "demo-es", status: "running", active: true, steps: [],
      job: { text: "Jefe de turno", use_claude: true },
    };
    const ctx = {
      api: { analysisStatus: async () => run, analysisJudge: async () => ({ mode: "local" }) },
      fail: () => {},
    };
    const root = new FakeElement("main");
    const close = await analysisView(ctx, root, ["demo-es"]);
    // The next 3 s refresh, run at once.
    const refresh = async () => timers.pop()();
    return { root, refresh, close, restore };
  } catch (error) {
    restore();
    throw error;
  }
}

test("unticking «Usar Claude» survives the next refresh", async () => {
  const page = await openPage();
  try {
    const box = page.root.find("an-claude");
    assert.equal(box.checked, true);
    box.checked = false;
    box.dispatch("change");
    await page.refresh();
    assert.equal(page.root.find("an-claude").checked, false);
    assert.equal(page.root.find("an-job").value, "Jefe de turno");
  } finally {
    page.close();
    page.restore();
  }
});

test("without a change event the refresh shows the stored leave again", async () => {
  const page = await openPage();
  try {
    const box = page.root.find("an-claude");
    box.checked = false;
    await page.refresh();
    assert.equal(box.checked, true);
  } finally {
    page.close();
    page.restore();
  }
});

test("typing a job survives the next refresh", async () => {
  const page = await openPage();
  try {
    const job = page.root.find("an-job");
    job.value = "Encargada de obrador";
    job.dispatch("input");
    await page.refresh();
    assert.equal(job.value, "Encargada de obrador");
  } finally {
    page.close();
    page.restore();
  }
});
