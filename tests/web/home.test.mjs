// The home (#/) and the Administración overview (#/admin): the services
// and tools they list, the latest sessions, where a row leads and the
// one-line conclusion read from a stored synthesis.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  ADMIN_TOOLS, CONCLUSION_MAX, LATEST, SERVICES, analysisFacts, conclusionOf, firstSentence, hasSynthesis, isAnalysed, latestSessions, mayHaveAnalysis, oneLine,
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

/** The hrefs of the worker part and of the Administración group of a page's sidebar. */
function navSplit(page) {
  const html = readFileSync(`${WEB}${page}`, "utf8");
  const nav = html.slice(html.indexOf('<nav data-i18n-label="nav.main"'), html.indexOf("</nav>"));
  const at = nav.indexOf('<div class="nav-admin"');
  const hrefs = (part) => [...part.matchAll(/<a href="([^"]+)"/g)].map((m) => m[1].replace(/^index\.html/, ""));
  return { work: hrefs(nav.slice(0, at)), admin: hrefs(nav.slice(at)), group: nav.slice(at, nav.indexOf(">", at) + 1) };
}

test("every admin tool is named and described in both languages and reachable from the sidebar", () => {
  const { work, admin, group } = navSplit("index.html");
  for (const tool of ADMIN_TOOLS) {
    for (const key of [`nav.${tool.id}`, `admin.${tool.id}`]) assert.ok(es[key] && en[key], key);
    assert.ok(admin.includes(tool.href), `${tool.href} in the Administración group`);
  }
  assert.equal(admin[0], "#/admin", "the service status opens the area");
  assert.deepEqual(work, ["#/", "#/sessions", "#/live", "#/voicebot"]);
  assert.match(group, /role="group"/);
  assert.match(group, /aria-labelledby="nav-admin-title"/);
});

test("research.html and study.html have the same worker / Administración split as index.html", () => {
  const index = navSplit("index.html");
  for (const page of ["research.html", "study.html"]) {
    const other = navSplit(page);
    assert.deepEqual(other.work, index.work, page);
    assert.deepEqual(other.admin, index.admin, page);
    assert.equal(other.group, index.group, page);
  }
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

test("a full stop after an abbreviation or a single letter does not end the sentence", () => {
  assert.equal(firstSentence("Habló con el Sr. García del plazo. Luego calló."), "Habló con el Sr. García del plazo.");
  assert.equal(firstSentence("Da ejemplos, p. ej. de su equipo. Otra."), "Da ejemplos, p. ej. de su equipo.");
  assert.equal(firstSentence("Met Dr. Smith, e.g. on Monday. Then left."), "Met Dr. Smith, e.g. on Monday.");
  assert.equal(firstSentence("Lo firmó J. Pérez ayer. Bien."), "Lo firmó J. Pérez ayer.");
  assert.equal(firstSentence("Dijo que no. Luego sí."), "Dijo que no.");
  assert.equal(firstSentence("¿Encaja? Sí."), "¿Encaja?");
  assert.equal(firstSentence("Sin punto final"), "Sin punto final");
  assert.equal(oneLine("Trató al Sr. Ruiz con calma. Resto."), "Trató al Sr. Ruiz con calma.");
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

/** A stand-in for the API: answers or failures per call. */
function fakeApi({ run, synthesis, runFails = false, synthesisFails = false }) {
  const calls = [];
  return {
    calls,
    async analysisStatus(id) { calls.push(["status", id]); if (runFails) throw new Error("down"); return run; },
    async analysisStep(id, step) { calls.push([step, id]); if (synthesisFails) throw new Error("down"); return synthesis; },
  };
}

const DONE = (names) => ({ steps: names.map((name) => ({ name, status: "done" })) });
const SYNTHESIS = { people: [{ participant: "Ana", fit: [{ text: "Encaja. Más." }] }] };

test("the home asks only a transcribed session for its analysis", async () => {
  const api = fakeApi({ run: DONE(["assessment"]) });
  assert.equal(await analysisFacts(api, { id: "x", state: "running" }), null);
  assert.deepEqual(api.calls, []);
});

test("no analysis, or the RAG away: the row shows the session alone", async () => {
  assert.equal(await analysisFacts(fakeApi({ runFails: true }), { id: "x", state: "done" }), null);
});

test("an analysis with results and a synthesis gives the conclusion", async () => {
  const api = fakeApi({ run: DONE(["assessment", "reports", "synthesis"]), synthesis: SYNTHESIS });
  assert.deepEqual(await analysisFacts(api, { id: "x", state: "done" }), { analysed: true, conclusion: { person: "Ana", text: "Encaja." } });
  assert.deepEqual(api.calls, [["status", "x"], ["synthesis", "x"]]);
});

test("results without a synthesis: analysed, no conclusion, no second call", async () => {
  const api = fakeApi({ run: DONE(["assessment"]) });
  assert.deepEqual(await analysisFacts(api, { id: "x", state: "done" }), { analysed: true, conclusion: null });
  assert.deepEqual(api.calls, [["status", "x"]]);
  assert.deepEqual(await analysisFacts(fakeApi({ run: DONE(["sentences"]) }), { id: "x", state: "done" }), { analysed: false, conclusion: null });
});

test("a synthesis that cannot be read keeps the row analysed and linked to the analysis", async () => {
  const api = fakeApi({ run: DONE(["assessment", "synthesis"]), synthesisFails: true });
  const facts = await analysisFacts(api, { id: "x", state: "done" });
  assert.deepEqual(facts, { analysed: true, conclusion: null });
  assert.equal(sessionHref({ id: "x" }, facts.analysed), "#/analysis/x");
});
