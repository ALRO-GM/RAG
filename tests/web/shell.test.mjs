// The app shell's pure logic: theme cycle, current link, breadcrumbs and
// the quick switcher's ranking.
import test from "node:test";
import assert from "node:assert/strict";
import {
  crumbsFor, currentHref, fold, matchScore, moveSelection, nextTheme, paletteShortcut, rankCommands, sectionOf, storedTheme,
} from "../../transcriber/adapters/inbound/web/js/lib/shell.js";

const t = (key) => `<${key}>`;

test("the theme button cycles system -> light -> dark -> system; unknown values mean system", () => {
  assert.equal(nextTheme("system"), "light");
  assert.equal(nextTheme("light"), "dark");
  assert.equal(nextTheme("dark"), "system");
  assert.equal(nextTheme(null), "light");
  assert.equal(storedTheme("purple"), "system");
  assert.equal(storedTheme("dark"), "dark");
});

test("a route belongs to the section of its first part; sessions own transcripts and analyses", () => {
  assert.equal(sectionOf(""), "#/");
  assert.equal(sectionOf("#/s/abc"), "#/");
  assert.equal(sectionOf("#/analysis/abc"), "#/");
  assert.equal(sectionOf("#/lab/x/y"), "#/lab");
  assert.equal(sectionOf("#/knowledge/hr"), "#/knowledge");
});

test("the current link matches index.html routes and the other pages by file", () => {
  const hrefs = ["#/", "#/record", "#/knowledge", "#/lab", "research.html", "study.html"];
  assert.equal(currentHref(hrefs, "/", "#/knowledge/hr"), "#/knowledge");
  assert.equal(currentHref(hrefs, "/index.html", ""), "#/");
  assert.equal(currentHref(hrefs, "/research.html", ""), "research.html");
  const fromStudy = ["index.html#/", "index.html#/lab", "study.html"];
  assert.equal(currentHref(fromStudy, "/study.html", ""), "study.html");
  assert.equal(currentHref(fromStudy, "/other.html", ""), null);
});

test("breadcrumbs name the section, then the ids from the address; the last has no link", () => {
  assert.deepEqual(crumbsFor("", t), [{ label: "<nav.sessions>", href: null }]);
  assert.deepEqual(crumbsFor("#/s/s%201", t), [{ label: "<nav.sessions>", href: "#/" }, { label: "s 1", href: null }]);
  assert.deepEqual(crumbsFor("#/analysis/abc", t).map((c) => c.href), ["#/", "#/s/abc", null]);
  assert.deepEqual(crumbsFor("#/knowledge", t), [{ label: "<nav.knowledge>", href: null }]);
  assert.deepEqual(crumbsFor("#/knowledge/hr", t), [{ label: "<nav.knowledge>", href: "#/knowledge" }, { label: "hr", href: null }]);
  const lab = crumbsFor("#/lab/sess/0123456789abcdef/review", t);
  assert.deepEqual(lab.map((c) => c.label), ["<nav.lab>", "sess", "01234567", "review"]);
  assert.equal(lab.at(-1).href, null);
  assert.equal(crumbsFor("#/%E0%A4%A", t).length, 1, "a broken escape does not throw");
});

test("the switcher folds accents and ranks prefixes over substrings over scattered letters", () => {
  assert.equal(fold("Investigación"), "investigacion");
  assert.ok(matchScore("Investigación", "inv") > matchScore("Registrar voz", "voz"));
  assert.ok(matchScore("Registrar voz", "voz") > matchScore("Conocimiento", "cimi"));
  assert.ok(matchScore("Conocimiento", "cimi") > matchScore("Conocimiento", "cnmt"));
  assert.equal(matchScore("Sesiones", "xyz"), 0);
  const commands = [{ label: "Sesiones" }, { label: "Estudio" }, { label: "Usar tema oscuro", keywords: "theme" }];
  assert.deepEqual(rankCommands(commands, "s").map((c) => c.label), ["Sesiones", "Estudio", "Usar tema oscuro"], "a prefix first, ties in order");
  assert.deepEqual(rankCommands(commands, "estu").map((c) => c.label), ["Estudio"]);
  assert.deepEqual(rankCommands(commands, "theme").map((c) => c.label), ["Usar tema oscuro"]);
  assert.equal(rankCommands(commands, "").length, 3, "an empty query lists everything in order");
});

test("arrow keys wrap around the list; the shortcut label follows the platform", () => {
  assert.equal(moveSelection(0, -1, 3), 2);
  assert.equal(moveSelection(2, 1, 3), 0);
  assert.equal(moveSelection(0, 1, 0), -1);
  assert.equal(paletteShortcut("MacIntel"), "⌘ K");
  assert.equal(paletteShortcut("Linux x86_64"), "Ctrl K");
});
