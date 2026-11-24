// The home (#/) and the Administración overview (#/admin): the services
// and tools they list, the latest sessions, where a row leads and the
// one-line conclusion read from a stored synthesis.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  ADMIN_TOOLS, CONCLUSION_MAX, LATEST, SERVICES, conclusionOf, hasSynthesis, isAnalysed, latestSessions, mayHaveAnalysis, oneLine,
  serviceState, sessionHref,
} from "../../transcriber/adapters/inbound/web/js/lib/home.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const WEB = new URL("../../transcriber/adapters/inbound/web/", import.meta.url).pathname;

test("the home offers the three services, each with its words in both languages", () => {
  assert.deepEqual(SERVICES.map((s) => s.href), ["#/sessions", "#/live", "#/voicebot"]);
  for (const s of SERVICES) {
    for (const key of [`home.${s.id}.title`, `home.${s.id}.body`]) {
      assert.ok(es[key] && en[key], key);
      assert.notEqual(es[key], en[key], key);
    }
  }
  assert.ok(es["home.try"] && en["home.try"]);
});

test("every admin tool is named and described in both languages and reachable from the sidebar", () => {
  const index = readFileSync(`${WEB}index.html`, "utf8");
  const sidebar = index.slice(index.indexOf('<div class="nav-admin">'), index.indexOf("</nav>"));
  for (const tool of ADMIN_TOOLS) {
    for (const key of [`nav.${tool.id}`, `admin.${tool.id}`]) assert.ok(es[key] && en[key], key);
    assert.ok(sidebar.includes(`href="${tool.href}"`), `${tool.href} in the Administración group`);
  }
  assert.ok(sidebar.includes('href="#/admin"'), "the service status opens the area");
  const work = index.slice(index.indexOf('<nav data-i18n-label="nav.main"'), index.indexOf('<div class="nav-admin">'));
  assert.deepEqual([...work.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1]), ["#/", "#/sessions", "#/live", "#/voicebot"]);
});

test("the latest sessions are the newest few, whatever the input", () => {
  const sessions = ["2026-11-01", "2026-11-20", "2026-11-05", "2026-11-22", "2026-11-10"]
    .map((day, i) => ({ id: `s${i}`, started_at: `${day}T10:00:00Z` }));
  assert.deepEqual(latestSessions(sessions).map((s) => s.id), ["s3", "s1", "s4", "s2"]);
  assert.equal(latestSessions(sessions).length, LATEST);
  assert.deepEqual(latestSessions(sessions, 2).map((s) => s.id), ["s3", "s1"]);
  assert.deepEqual(latestSessions(null), []);
  assert.deepEqual(latestSessions(sessions, -1), []);
  assert.equal(sessions[0].id, "s0", "the input is not reordered");
});

test("only a transcribed session is asked for its analysis; a row leads to the analysis when it has results", () => {
  assert.ok(mayHaveAnalysis({ state: "done" }));
  assert.ok(!mayHaveAnalysis({ state: "running" }));
  assert.ok(!mayHaveAnalysis(null));
  assert.equal(sessionHref({ id: "a b" }, true), "#/analysis/a%20b");
  assert.equal(sessionHref({ id: "a b" }, false), "#/s/a%20b");
  const run = { steps: [{ name: "assessment", status: "done" }, { name: "synthesis", status: "done" }] };
  assert.ok(isAnalysed(run));
  assert.ok(hasSynthesis(run));
  assert.ok(!hasSynthesis({ steps: [{ name: "synthesis", status: "running" }] }));
  assert.ok(!isAnalysed({ steps: [{ name: "sentences", status: "done" }] }));
  assert.ok(!hasSynthesis(null));
});

test("a conclusion is the first sentence of the first person's fit, else their first strength", () => {
  const synthesis = {
    people: [
      { participant: "Ana", status: "skipped", fit: [{ text: "Not used." }] },
      { participant: "Luis", fit: [{ text: "  " }, { text: "Encaja con el puesto. Reforzar pruebas." }], strengths: [{ text: "Escucha." }] },
    ],
  };
  assert.deepEqual(conclusionOf(synthesis), { person: "Luis", text: "Encaja con el puesto." });
  assert.deepEqual(conclusionOf({ people: [{ participant: "Eva", fit: [], strengths: [{ text: "Explica con ejemplos" }] }] }),
    { person: "Eva", text: "Explica con ejemplos" });
  assert.equal(conclusionOf({ people: [{ participant: "Eva" }] }), null);
  assert.equal(conclusionOf(null), null);
});

test("one line is the first sentence, cut at a word with an ellipsis when too long", () => {
  assert.equal(oneLine("Uno. Dos."), "Uno.");
  assert.equal(oneLine("v1.2 sigue. Otra"), "v1.2 sigue.", "a dot inside a word does not end the sentence");
  assert.equal(oneLine("  a\n b  "), "a b");
  assert.equal(oneLine(null), "");
  const long = oneLine("palabra ".repeat(40));
  assert.ok(long.length <= CONCLUSION_MAX, long.length);
  assert.ok(long.endsWith("palabra…"), long);
  assert.equal(oneLine("x".repeat(200), 10), `${"x".repeat(9)}…`, "no space: a hard cut");
});

test("the service is up only when the health check answered ok", () => {
  assert.equal(serviceState({ status: "ok" }), "ok");
  assert.equal(serviceState(null), "down");
  assert.equal(serviceState({ status: "starting" }), "down");
});
