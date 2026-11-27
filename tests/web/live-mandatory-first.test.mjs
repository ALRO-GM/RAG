// #287: «Lo que hay que decir» is the first block of the live page's side
// (above «Temas del puesto»), so the visual, keyboard and screen-reader
// orders agree in every layout; texts pinned in both languages.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const VIEW = readFileSync(new URL("../../transcriber/adapters/inbound/web/js/views/live.js", import.meta.url), "utf8");
const CSS = readFileSync(new URL("../../transcriber/adapters/inbound/web/style.css", import.meta.url), "utf8");

test("the side lists the mandatory block first, before the topics", () => {
  const side = VIEW.match(/const side = h\("aside"[^;]*;/s)[0];
  const order = side.match(/\b(checklist|topics|nowCard|coreRow)\b/g);
  assert.deepEqual(order.slice(0, 2), ["checklist", "nowCard"]);
  assert.ok(order.indexOf("checklist") < order.indexOf("topics"));
});

test("the side comes before the conversation in the DOM (stacked layouts read it first)", () => {
  assert.match(VIEW, /h\("div", \{ class: "lv-body" \}, side, talk\)/);
});

test("no CSS rule reorders the containers or the block away from the DOM order", () => {
  for (const sel of ["lv-body", "lv-talk", "lv-side", "lv-tosay", "lv-fold", "lv-row"]) {
    const rules = [...CSS.matchAll(new RegExp(`[^{}]*\\.${sel}\\b[^{}]*\\{[^}]*\\}`, "g"))].map((m) => m[0]);
    for (const rule of rules) assert.doesNotMatch(rule, /[\s;{]order\s*:/, rule);
  }
});

test("the reminder line is outside the folding part", () => {
  assert.match(VIEW, /checklist\.append\(checklistFold, checklistRemind\)/);
});

test("the block collapses to one calm line, in both languages", () => {
  assert.match(VIEW, /live\.mandatory_done/);
  assert.equal(es["live.mandatory_title"], "Lo que hay que decir");
  assert.equal(es["live.mandatory_done"], "Todo lo obligatorio está dicho");
  assert.ok(en["live.mandatory_title"] && en["live.mandatory_done"]);
});
