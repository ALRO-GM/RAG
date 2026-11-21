// The language switch's pure logic, the number formats, and an audit that
// the modules drawing the pages hold no user-visible text of their own:
// every sentence a person reads comes from js/i18n/es.js and en.js.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import {
  chooseLanguage, formatDecimal, formatPercent, pickLanguage,
} from "../../transcriber/adapters/inbound/web/js/lib/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const WEB = new URL("../../transcriber/adapters/inbound/web/", import.meta.url).pathname;

test("a language chosen before wins over the browser's; an unknown one is ignored", () => {
  const avail = ["es", "en"];
  assert.equal(chooseLanguage("en", "es-ES", avail), "en");
  assert.equal(chooseLanguage("es", "en-GB", avail), "es");
  assert.equal(chooseLanguage(null, "en-GB", avail), "en");
  assert.equal(chooseLanguage("fr", "en-US", avail), "en");
  assert.equal(chooseLanguage(undefined, undefined, avail), "es");
  assert.equal(chooseLanguage("fr", "de", avail), pickLanguage("de", avail));
});

test("numbers use each language's decimal mark and percent sign", () => {
  assert.equal(formatDecimal("es", 3.456, 1), "3,5");
  assert.equal(formatDecimal("en", 3.456, 1), "3.5");
  assert.equal(formatDecimal("es", 0.5, 2), "0,50");
  assert.equal(formatDecimal("en", 2, 3), "2.000");
  assert.equal(formatPercent("es", 0.125), "12,5 %");
  assert.equal(formatPercent("en", 0.125), "12.5%");
});

test("the switch's own texts exist in both languages", () => {
  for (const key of ["lang.switch", "lang.busy", "research.title", "app.title"]) {
    assert.ok(es[key], key);
    assert.ok(en[key], key);
    assert.notEqual(es[key], en[key], key);
  }
});

// ---- audit of hard-coded text --------------------------------------------

/** Modules that draw what a person sees (the dictionaries themselves excluded). */
function pageModules() {
  const top = readdirSync(join(WEB, "js")).filter((f) => f.endsWith(".js") && f !== "i18n.js").map((f) => join("js", f));
  const views = readdirSync(join(WEB, "js", "views")).filter((f) => f.endsWith(".js")).map((f) => join("js", "views", f));
  return [...top, ...views].sort();
}

/** Every CSS class the stylesheet defines: a string made only of them is a class list. */
function cssClasses() {
  const css = readFileSync(join(WEB, "style.css"), "utf8");
  return new Set([...css.matchAll(/\.([A-Za-z_][\w-]*)/g)].map((m) => m[1]));
}

/** The string literals of `source` outside comments; ${...} parts of templates dropped. */
export function literals(source) {
  const out = [];
  const pattern = /\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`/g;
  for (const [token] of source.matchAll(pattern)) {
    if (token.startsWith("/")) continue;
    let text = token.slice(1, -1);
    if (token.startsWith("`")) text = text.replace(/\$\{[^}]*\}/g, " ");
    out.push(text);
  }
  return out;
}

// camelCase, PascalCase with two humps or more, Header-Case: names, not words.
const IDENTIFIER = /^(?:[a-z]+(?:[A-Z][a-z0-9]*)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[A-Z][a-z]*(?:-[A-Z][a-z]*)+)$/;
// Values of HTML attributes such as rel.
const HTML_KEYWORDS = new Set(["noopener", "noreferrer"]);

/** Whether `text` reads like words for a person rather than an identifier or class list. */
export function looksLikeText(text, classes) {
  if (!/\p{L}/u.test(text)) return false;
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 1 && IDENTIFIER.test(words[0])) return false;
  const classLike = (w) => classes.has(w) || HTML_KEYWORDS.has(w) || /^[a-z][\w-]*-$/.test(w)
    || [...classes].some((c) => c.startsWith(`${w}-`));
  if (words.every(classLike)) return false;
  return /\p{Lu}\p{Ll}/u.test(text) || /[^\x00-\x7f]/.test(text.replace(/[^\p{L}]/gu, "")) || /\p{L}{2,}\s+\p{L}{2,}/u.test(text);
}

// What may stay: names of languages written in themselves, keyboard keys,
// codes the server sends (compared, never shown), technical names.
const ALLOWED = new Map([
  ["js/views/record.js", ["Español", "English", "stack reauth"]],
  ["js/views/sessions.js", ["Español", "English"]],
  ["js/shell.js", ["Español", "English", "Enter", "Escape"]],
  ["js/views/voicebot.js", ["Enter"]],
  ["js/views/live.js", ["Enter", "Escape"]],
  ["js/views/lab-progress.js", ["no speech in the audio"]],
]);

test("no page module holds user-visible text outside the dictionaries", () => {
  const classes = cssClasses();
  const found = [];
  for (const file of pageModules()) {
    const allowed = new Set(ALLOWED.get(file) ?? []);
    const source = readFileSync(join(WEB, file), "utf8");
    for (const text of literals(source)) {
      if (looksLikeText(text, classes) && !allowed.has(text)) found.push(`${file}: ${JSON.stringify(text)}`);
    }
  }
  assert.deepEqual(found, []);
});

test("the audit tells text from class lists and identifiers", () => {
  const classes = new Set(["row", "wrap", "primary", "big"]);
  assert.equal(looksLikeText("row wrap", classes), false);
  assert.equal(looksLikeText("primary big", classes), false);
  assert.equal(looksLikeText("failed", classes), false);
  assert.equal(looksLikeText("NotAllowedError", classes), false);
  assert.equal(looksLikeText("Content-Type", classes), false);
  assert.equal(looksLikeText("Pruebas", classes), true);
  assert.equal(looksLikeText("application/json", classes), false);
  assert.equal(looksLikeText("Guardar", classes), true);
  assert.equal(looksLikeText("ya está", classes), true);
  assert.equal(looksLikeText("save the file", classes), true);
  assert.deepEqual(literals('h("p", {}, "Hola") // "no"\n/* "tampoco" */ `a ${x} b`'), ["p", "Hola", "a   b"]);
});
