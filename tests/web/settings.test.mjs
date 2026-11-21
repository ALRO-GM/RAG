// «Ajustes» (#/settings): the pure helpers of the settings page, the
// dictionaries holding a label and a measured trade-off for every option
// of the contract, and the page drawn in a stand-in DOM. The language is
// set explicitly (Node 21+ has a global navigator).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  GROUPS, PRESETS, analysisPrefill, applyPreset, changedSteps, costOf, defaultChoices, keys, label, needsRestart,
  presetOf, saveBody, stepsByGroup,
} from "../../transcriber/adapters/inbound/web/js/lib/settings.js";
import { crumbsFor } from "../../transcriber/adapters/inbound/web/js/lib/shell.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";
import { translate } from "../../transcriber/adapters/inbound/web/js/lib/i18n.js";

const ROOT = new URL("../../", import.meta.url);
const schema = JSON.parse(readFileSync(new URL("contracts/settings.v1.schema.json", ROOT), "utf8"));
const example = JSON.parse(readFileSync(new URL("contracts/examples/settings.v1.json", ROOT), "utf8"));
const STEPS = Object.entries(schema.properties.choices.properties).map(([id, s]) => ({ id, options: s.enum }));

const tEs = (key, params) => translate({ es, en }, "es", key, params);
const tEn = (key, params) => translate({ es, en }, "en", key, params);

// A catalogue shaped like GET /api/settings answers it (from the contract).
const GROUP_OF = (id) => id.split(".")[0];
const catalogue = {
  steps: STEPS.map((s) => ({
    id: s.id, group: GROUP_OF(s.id), options: s.options, default: s.options[0],
    applies: s.id === "transcription.recogniser" ? "restart" : "next_job", consumer: "rag",
  })),
  presets: {
    quality: Object.fromEntries(STEPS.map((s) => [s.id, s.options[0]])),
    balanced: { ...Object.fromEntries(STEPS.map((s) => [s.id, s.options[0]])), "analysis.confirmation_model": "haiku" },
    cost: { ...Object.fromEntries(STEPS.map((s) => [s.id, s.options[0]])), "analysis.judge": "local" },
  },
};

test("every step and option of the contract has a label and a trade-off in both languages", () => {
  const missing = [];
  for (const step of STEPS) {
    for (const key of [keys.step(step.id), keys.group(GROUP_OF(step.id))]) {
      if (!es[key] || !en[key]) missing.push(key);
    }
    for (const option of step.options) {
      for (const key of [keys.option(step.id, option), keys.fact(step.id, option)]) {
        if (!es[key] || !en[key]) missing.push(key);
      }
    }
  }
  assert.deepEqual(missing, []);
  for (const name of PRESETS) {
    assert.ok(es[`set.preset.${name}`] && en[`set.preset_desc.${name}`], name);
  }
  assert.equal(tEs("set.preset.quality"), "Máxima calidad");
  assert.equal(tEs("set.preset.cost"), "Menor coste");
  assert.equal(tEn("set.preset.balanced"), "Balanced");
});

test("the contract's example is a preset the page recognises", () => {
  assert.equal(example.preset, "balanced");
  assert.equal(presetOf(example.choices, catalogue), "balanced");
});

test("steps are grouped in page order; an unknown group goes last", () => {
  const groups = stepsByGroup({ steps: [...catalogue.steps, { id: "voicebot.voice", group: "voicebot", options: ["a"] }, { id: "z.z", group: "zeta", options: ["a"] }] });
  assert.deepEqual(groups.map((g) => g.group), ["transcription", "live", "analysis", "knowledge", "voicebot", "zeta"]);
  assert.ok(GROUPS.includes("voicebot"), "the voice bot has a place waiting for it");
  assert.deepEqual(stepsByGroup(null), []);
});

test("presets are recognised, applied and anything else is custom", () => {
  const defaults = defaultChoices(catalogue);
  assert.equal(presetOf(defaults, catalogue), "quality");
  assert.equal(presetOf({}, catalogue), "quality");
  const cost = applyPreset("cost", catalogue, defaults);
  assert.equal(cost["analysis.judge"], "local");
  assert.equal(presetOf(cost, catalogue), "cost");
  const mine = { ...cost, "knowledge.ocr": "off" };
  assert.equal(presetOf(mine, catalogue), "custom");
  assert.deepEqual(applyPreset("custom", catalogue, mine), mine);
  assert.deepEqual(changedSteps(defaults, mine).sort(), ["analysis.judge", "knowledge.ocr"]);
});

test("a change to the recogniser warns that it needs a restart", () => {
  assert.equal(needsRestart(["transcription.recogniser"], catalogue), true);
  assert.equal(needsRestart(["analysis.judge", "live.asr"], catalogue), false);
});

test("the cost chip says local, quota, or that the image leaves the PC", () => {
  assert.equal(costOf("transcription.diarizer", "diarizen"), "local");
  assert.equal(costOf("analysis.judge", "claude_code"), "quota");
  assert.equal(costOf("analysis.judge", "local"), "local");
  assert.equal(costOf("knowledge.captions", "claude"), "cloud");
  assert.equal(costOf("knowledge.captions", "off"), null);
  assert.equal(costOf("live.deadline", "1800"), null);
  assert.match(tEs(`set.cost.${costOf("knowledge.captions", "claude")}`), /sale del PC/);
  assert.equal(tEs("set.cost.local"), "local, gratis");
});

test("the analysis form is prefilled only from saved settings", () => {
  assert.deepEqual(analysisPrefill(null), {});
  assert.deepEqual(analysisPrefill({ stored: false, choices: catalogue.presets.balanced }), {});
  assert.deepEqual(analysisPrefill({ stored: true, choices: { ...catalogue.presets.balanced, "analysis.synthesis_model": "weird" } }), {
    confirmation: "haiku", assessment: "opus",
  });
});

test("the save body carries the draft and the revision read", () => {
  assert.deepEqual(saveBody({ "live.asr": "cpu" }, 3), { choices: { "live.asr": "cpu" }, revision: 3 });
  assert.equal(saveBody({}, undefined).revision, 0);
  assert.equal(label(tEs, "set.nothing", "fallback"), "fallback");
  assert.equal(label(tEs, keys.step("live.asr"), "x"), "Reconocedor en directo");
});

test("the breadcrumbs of #/settings", () => {
  assert.deepEqual(crumbsFor("#/settings", tEs), [{ label: "Ajustes", href: null }]);
});

// ---- the page in a stand-in DOM ---------------------------------------------

class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.classList = { add: () => {}, toggle: () => {} };
    this.hidden = false;
    this.disabled = false;
  }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k] ?? null; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { for (const n of nodes.flat()) if (n !== null && n !== undefined && n !== false) this.children.push(typeof n === "object" ? n : new Text(n)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  remove() {}
  focus() {}
  querySelector() { return null; }
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.children = [new Text(v)]; }
  all() { return [this, ...this.children.flatMap((c) => (c instanceof Node ? c.all() : []))]; }
  fire(type) { for (const fn of this.listeners[type] ?? []) fn({ target: this, preventDefault() {} }); }
}
class Text extends Node {
  constructor(v) { super("#text"); this.value = String(v); }
  get textContent() { return this.value; }
  set textContent(v) { this.value = String(v); }
}

function installDom() {
  const body = new Node("body");
  globalThis.Node = Node;
  globalThis.document = {
    createElement: (tag) => new Node(tag),
    createElementNS: (_ns, tag) => new Node(tag),
    createTextNode: (v) => new Text(v),
    getElementById: () => null,
    body,
    documentElement: new Node("html"),
  };
  globalThis.localStorage = { getItem: () => "es", setItem: () => {}, removeItem: () => {} };
}

test("the page draws the presets, every step with its trade-off, and saves the draft", async () => {
  installDom();
  const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
  setLanguage("es");
  const { settingsView } = await import("../../transcriber/adapters/inbound/web/js/views/settings.js");
  const saved = { schema: "settings.v1", revision: 1, updated_at: "2026-11-21T10:00:00Z", stored: true, preset: "quality", choices: defaultChoices(catalogue), catalogue };
  const sent = [];
  const api = {
    pipelineSettings: async () => saved,
    savePipelineSettings: async (body) => { sent.push(body); return { ...saved, revision: 2, choices: body.choices, preset: presetOf(body.choices, catalogue) }; },
  };
  const root = new Node("main");
  await settingsView({ api, fail: (e) => { throw e; } }, root);
  const text = root.textContent;
  for (const word of ["Máxima calidad", "Equilibrado", "Menor coste", "Personalizado", "Separar voces (quién habla)", "WER 0,088", "AMI DER 0,089", "local, gratis", "Al reiniciar el servicio", "Desde el próximo trabajo", "La imagen sale del PC"]) {
    assert.ok(text.includes(word), word);
  }
  const radios = root.all().filter((n) => n.tagName === "INPUT" && n.attributes.type === "radio");
  const stepRadios = radios.filter((n) => n.attributes.name !== "set-preset");
  assert.equal(stepRadios.length, STEPS.reduce((sum, s) => sum + s.options.length, 0), "one radio per option");
  const cost = radios.find((n) => n.attributes.name === "set-preset" && n.attributes.value === "cost");
  cost.fire("change");
  const save = root.all().find((n) => n.tagName === "BUTTON" && n.textContent === "Guardar");
  save.fire("click");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(sent.length, 1);
  assert.equal(sent[0].revision, 1);
  assert.equal(sent[0].choices["analysis.judge"], "local");
});

test("without a RAG the page explains why", async () => {
  installDom();
  const { settingsView } = await import("../../transcriber/adapters/inbound/web/js/views/settings.js");
  const root = new Node("main");
  const failure = Object.assign(new Error("x"), { kind: "unavailable", status: 503, code: "analysis_unavailable" });
  await settingsView({ api: { pipelineSettings: async () => { throw failure; } }, fail: () => {} }, root);
  assert.match(root.textContent, /no están disponibles/);
});
