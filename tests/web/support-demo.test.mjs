// The call-centre example (#250) without a browser: which demo a route
// opens, where the live demo ends, the call report's data joined to the
// call, the voice-assistant example played with a fake clock and voice,
// and the words of every new page in both languages. The language is
// pinned where Spanish text is asserted (Node 22 has navigator.language).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  demoCardExtras, lineNumbers, pickDemo, resultLink, sourcesOf, voiceDemo,
} from "../../transcriber/adapters/inbound/web/js/lib/call-demo.js";
import {
  callLines, countParts, effectCounts, isDemoReport, itemText, lineOf, mmss, outcomeCounts, stepCounts,
} from "../../transcriber/adapters/inbound/web/js/lib/call-report.js";
import {
  TIMING, abortableSleep, browserNarrator, narrationLimitMs, play, readMs, stepsOf,
} from "../../transcriber/adapters/inbound/web/js/lib/voice-demo.js";
import { SERVICES } from "../../transcriber/adapters/inbound/web/js/lib/home.js";
import { crumbsFor } from "../../transcriber/adapters/inbound/web/js/lib/shell.js";
import { t as tr, setLanguage } from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { LiveState, citations, signalLines } from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const read = (f) => readFileSync(new URL(`../../${f}`, import.meta.url), "utf8");
const DEMO = JSON.parse(read("examples/demo/llamada-lumbre.demo.json"));
const WEB = "transcriber/adapters/inbound/web/js/";

// What GET /demos/{id}/report answers (LiveSessions.demo_part).
const DATA = {
  ...DEMO.report, demo: true, id: DEMO.id, title: DEMO.title, note: DEMO.note, en: DEMO.en, sources: DEMO.sources,
  mandatory_items: DEMO.mandatory.items.map(({ said_by, ...rest }) => rest),
  utterances: DEMO.transcript.utterances, speakers: DEMO.transcript.speakers,
};
const LISTED = [
  { id: "entrevista-lucia", demo: true, product: "live_assist.interview", result_session: "demo-x" },
  { id: DEMO.id, demo: true, product: "live_assist.support", report: true, voicebot: true },
];

test("the routes open the right demo and the end leads to the right place", () => {
  assert.equal(pickDemo(LISTED).id, "entrevista-lucia");
  assert.equal(pickDemo(LISTED, "support").id, DEMO.id);
  assert.equal(pickDemo([LISTED[0]], "support"), null);
  assert.equal(pickDemo([LISTED[1]]), null);
  assert.equal(pickDemo([]), null);
  assert.equal(pickDemo(undefined), null);
  assert.equal(voiceDemo(LISTED).id, DEMO.id);
  assert.equal(voiceDemo([LISTED[0]]), null);
  assert.deepEqual(resultLink(LISTED[1]), { kind: "report", href: `#/call-report/${DEMO.id}` });
  assert.deepEqual(resultLink(LISTED[0]), { kind: "analysis", href: "#/analysis/demo-x" });
  assert.equal(resultLink({ id: "a" }), null);
  assert.equal(resultLink(null), null);
});

test("lines are numbered from 1 and sources found by id", () => {
  const numbers = lineNumbers(DEMO.transcript.utterances);
  assert.equal(numbers.get("t0.s0"), 1);
  assert.equal(numbers.get("t18.s0"), 19);
  assert.deepEqual(sourcesOf(["s18", "zz"], DEMO.sources).map((s) => s.id), ["s18"]);
  assert.deepEqual(sourcesOf(null, null), []);
});

test("the report is joined to the call and every reference resolves", () => {
  assert.equal(isDemoReport(DATA), true);
  assert.equal(isDemoReport({ ...DATA, demo: false }), false);
  assert.equal(isDemoReport(null), false);
  const call = callLines(DATA);
  assert.equal(call.lines.length, 19);
  assert.equal(call.agent, "agente");
  assert.equal(lineOf(call, "t4.s0").n, 5);
  assert.equal(lineOf(call, "zz"), null);
  const ids = [
    ...DATA.lines_review.flatMap((r) => [r.utterance, r.reaction]),
    ...DATA.summary.strengths.flatMap((p) => p.about), ...DATA.summary.improve.flatMap((p) => p.about),
    ...DATA.script.steps.flatMap((s) => s.about), ...DATA.script.verdict.about,
    ...DATA.top_scorers.moments.map((m) => m.utterance), ...DATA.unanswered.map((u) => u.utterance),
    ...DATA.mandatory.map((m) => m.utterance).filter(Boolean),
  ];
  for (const id of ids) assert.ok(lineOf(call, id), id);
  assert.equal(mmss(94.04), "1:34");
  assert.equal(mmss(-3), "0:00");
  assert.equal(mmss("x"), "0:00");
});

test("the counts the summary shows add up to the parts", () => {
  assert.deepEqual(effectCounts(DATA), { helped: 5, neutral: 3, hurt: 1 });
  assert.equal(Object.values(stepCounts(DATA)).reduce((a, b) => a + b), DATA.script.steps.length);
  assert.deepEqual(outcomeCounts(DATA), { said: 2, said_late: 1, never: 1 });
  assert.equal(effectCounts(null).hurt, 0);
  assert.equal(itemText("aceptacion-expresa", DATA.mandatory_items).startsWith("Pedir la aceptación"), true);
  assert.equal(itemText("zz", null), "zz");
});

test("the report says what is illustrative and what is judged by the reaction", () => {
  assert.equal(DATA.top_scorers.illustrative, true);
  assert.match(DATA.top_scorers.note, /ilustrativo/);
  assert.match(DATA.top_scorers.en.note, /Illustrative/);
  const hurt = DATA.lines_review.find((r) => r.effect === "hurt");
  assert.ok(hurt.better);
  assert.equal(DATA.lines_review.every((r) => r.reaction !== r.utterance), true);
});

/** The orchestrator's feed of the support demo, as the page receives it. */
function feed() {
  const page = { product: "live_assist.support", signals: true, modes: ["pincelada"], script_phases: [], roles: { agente: "agent", cliente: "customer" } };
  const out = [{ type: "session", session: { id: "d", mode: "demo", page, demo: LISTED[1] } }];
  const first = DEMO.suggestions[0];
  const u = DEMO.transcript.utterances.find((x) => x.id === first.after);
  const source = DEMO.sources.find((s) => s.id === first.cites[0]);
  out.push({ type: "event", event: { type: "final", stream_time: u.end, utterance: { ...u, revision: 0 } } });
  out.push({ type: "suggestion", record: {
    call: 1, status: "ok", trigger: { utterance: u.id, stream_time: u.end }, stages: {},
    retrieved: [{ id: source.id, text: source.text, metadata: source.metadata }],
    suggestions: [{ about: [u.id], cites: first.cites, text: first.text, kind: first.kind,
      expressed: first.expressed.map((signal) => ({ signal, utterance: u.id, quote: u.text })) }],
  } });
  out.push({ type: "mandatory", checklist: { items: DEMO.mandatory.items.map((i) => ({ id: i.id, text: i.text, state: "pending" })), remind_seconds: 45 } });
  return out;
}

test("the live page state shows the quoted signal, the source and the pending items", () => {
  const state = new LiveState();
  for (const m of feed()) state.apply(m);
  const card = state.cards()[0];
  assert.deepEqual(card.signals.map((s) => s.signal), ["complaint", "churn_risk"]);
  assert.match(card.signals[0].quote, /Lumbre Plus/);
  assert.equal(card.cites[0].id, "s18");
  assert.match(card.cites[0].text, /cancelada antes de su renovación/);
  assert.equal(state.pendingItems().length, 4);
  assert.equal(state.discomfort().record, state.records[0]);
  assert.equal(citations({ cites: ["x"] }, { retrieved: [] })[0].text, null);
  assert.deepEqual(signalLines({ expressed: [] }, { signals: true, product: "live_assist.support" }), []);
});

test("the home links each service to its example", () => {
  for (const s of SERVICES) assert.ok(s.demo, s.id);
  assert.equal(SERVICES.find((s) => s.id === "analyse").demo, `#/call-report/${DEMO.id}`);
  assert.equal(SERVICES.find((s) => s.id === "live").demo, "#/live/demo");
  assert.deepEqual(SERVICES.find((s) => s.id === "live").more.map((m) => m.href), ["#/live/demo/support"]);
  assert.equal(SERVICES.find((s) => s.id === "voicebot").demo, "#/voicebot/demo");
  for (const key of ["home.analyse.demo", "home.live.demo", "home.live.demo_support", "home.voicebot.demo"]) {
    assert.ok(es[key] && en[key], key);
    assert.notEqual(es[key], en[key], key);
  }
});

test("the breadcrumbs of the new pages", () => {
  const t = (k) => k;
  assert.deepEqual(crumbsFor("#/call-report/llamada-lumbre", t), [{ label: "report.title", href: null }]);
  assert.deepEqual(crumbsFor("#/voicebot/demo", t), [{ label: "nav.voicebot", href: "#/voicebot" }, { label: "vdemo.crumb", href: null }]);
  assert.deepEqual(crumbsFor("#/voicebot", t), [{ label: "nav.voicebot", href: null }]);
});

const VOICE = { ...DEMO.voicebot, sources: DEMO.sources };

test("the voice example's steps follow the exchanges and carry their sources", () => {
  const steps = stepsOf(VOICE, DEMO.sources, "es");
  assert.equal(steps.length, VOICE.exchanges.length * 3);
  assert.deepEqual(steps.slice(0, 3).map((s) => s.kind), ["user", "thinking", "bot"]);
  assert.equal(steps[2].sources[0].id, "s18");
  assert.match(steps[2].sources[0].text, /renovación/);
  assert.match(stepsOf(VOICE, DEMO.sources, "en")[0].text, /^Hello/);
  assert.deepEqual(stepsOf(null, null), []);
  assert.equal(readMs("a b c", 100), TIMING.minRead);
  assert.equal(readMs("a ".repeat(20), 100), 2000);
});

test("the example plays in order with a fake clock, speaks when it can and stops on abort", async () => {
  const steps = stepsOf(VOICE, DEMO.sources, "es");
  const shown = [];
  const waited = [];
  const done = await play(steps, { show: (s) => shown.push(s.kind), sleep: async (ms) => { waited.push(ms); } });
  assert.equal(done, true);
  assert.equal(shown.length, steps.length);
  assert.ok(waited.length > steps.length);

  const spoken = [];
  const narrated = await play(steps.slice(0, 3), {
    show() {}, sleep: async () => {}, narrate: async (text) => { spoken.push(text); return true; },
  });
  assert.equal(narrated, true);
  assert.equal(spoken.length, 1);

  const controller = new AbortController();
  let count = 0;
  const stopped = await play(steps, {
    show() { count += 1; if (count === 2) controller.abort(); }, signal: controller.signal, sleep: abortableSleep(controller.signal, (fn) => { fn(); return 0; }),
  });
  assert.equal(stopped, false);
  assert.equal(count, 2);

  await assert.rejects(() => play(steps, { show() { throw new Error("boom"); }, sleep: async () => {} }), /boom/);
});

test("abortableSleep resolves, and rejects at once when aborted", async () => {
  const live = new AbortController();
  await abortableSleep(live.signal, (fn) => { fn(); return 1; }, () => {})(5);
  const gone = new AbortController();
  gone.abort();
  await assert.rejects(() => abortableSleep(gone.signal)(5), { name: "AbortError" });
  const later = new AbortController();
  let cleared = false;
  const pending = abortableSleep(later.signal, () => 7, () => { cleared = true; })(50);
  later.abort();
  await assert.rejects(() => pending, { name: "AbortError" });
  assert.equal(cleared, true);
});

test("the browser narrator uses a Spanish voice, or says it could not", async () => {
  assert.equal(browserNarrator(null, class {}), null);
  assert.equal(browserNarrator({}, null), null);
  const made = [];
  class Utter { constructor(text) { this.text = text; made.push(this); } }
  const synth = (voices) => ({
    getVoices: () => voices, cancelled: 0, speak(u) { u.onend(); }, cancel() { this.cancelled += 1; },
  });
  const spanish = browserNarrator(synth([{ lang: "en-GB" }, { lang: "es-ES", name: "x" }]), Utter, "es");
  assert.equal(await spanish.speak("hola"), true);
  assert.equal(made[0].lang, "es-ES");
  spanish.cancel();
  const none = browserNarrator(synth([{ lang: "fr-FR" }]), Utter, "es");
  assert.equal(await none.speak("hola"), false);
  const unloaded = browserNarrator(synth([]), Utter, "en");
  assert.equal(await unloaded.speak("hi"), true);
  assert.equal(made.at(-1).lang, "en-GB");
  const failing = browserNarrator({ getVoices: () => [], speak(u) { u.onerror(); } }, Utter, "es");
  assert.equal(await failing.speak("x"), false);
});

test("every word the new pages use exists in both languages", () => {
  const files = ["views/call-report.js", "views/voice-demo.js", "views/live.js", "views/home.js", "views/voicebot.js"];
  const used = new Set();
  for (const f of files) for (const m of read(WEB + f).matchAll(/\bt\(\s*["`]((?:report|vdemo|live\.support_demo|live\.report|live\.source_kb)[a-z_.${}A-Z]*)["`]/g)) used.add(m[1]);
  assert.ok(used.size > 20);
  for (const key of used) {
    if (key.includes("${")) continue;
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  for (const key of Object.keys(es).filter((k) => /^(report|vdemo)\./.test(k))) {
    assert.ok(en[key], `en ${key}`);
    assert.notEqual(es[key], en[key], key);
  }
  for (const kind of ["helped", "neutral", "hurt"]) assert.ok(es[`report.effect.${kind}`] && en[`report.effect.${kind}`]);
  for (const s of ["followed", "reordered", "skipped"]) assert.ok(es[`report.step.${s}`] && en[`report.step.${s}`]);
  for (const o of ["said", "said_late", "never"]) assert.ok(es[`report.outcome.${o}`] && en[`report.outcome.${o}`]);
  for (const g of ["question", "data"]) assert.ok(es[`report.gap.${g}`] && en[`report.gap.${g}`]);
});

test("a real session's card gets no source block, a scripted demo's does", () => {
  const state = new LiveState();
  for (const m of feed()) state.apply(m);
  const card = state.cards()[0];
  assert.ok(card.cites.length);
  assert.deepEqual(demoCardExtras(card, null), { signals: [], sources: [] });
  assert.deepEqual(demoCardExtras(card, undefined, state.records[0]), { signals: [], sources: [] });
  const demo = demoCardExtras(card, LISTED[1]);
  assert.equal(demo.sources[0].id, "s18");
  assert.equal(demo.signals.length, 2);
  // The banner already quotes a discomfort signal of the same record: not repeated.
  assert.deepEqual(demoCardExtras(card, LISTED[1], card.record).signals, []);
  assert.equal(demoCardExtras(null, LISTED[1]).sources.length, 0);
  // And the view only asks for it in a demo.
  assert.match(read(`${WEB}views/live.js`), /demo \? callSupport\(card\) : null/);
});

test("counts use the singular and the plural of each language", () => {
  const spanish = (key, p) => es[key].replace("{n}", p.n);
  assert.deepEqual(countParts(spanish, [["report.n_hurt", 1], ["report.n_hurt", 2]]), ["1 perjudicó", "2 perjudicaron"]);
  assert.deepEqual(countParts(spanish, [["report.n_late", 1], ["report.n_said", 1], ["report.n_said", 3]]), ["1 dicho tarde", "1 dicho a tiempo", "3 dichos a tiempo"]);
  assert.deepEqual(countParts(spanish, [["report.n_helped", 1], ["report.n_neutral", 3], ["report.n_skipped", 1]]), ["1 ayudó", "3 neutras", "1 omitido"]);
  for (const key of Object.keys(es).filter((k) => /^report\.n_.*_one$/.test(k))) {
    const other = key.replace(/_one$/, "_other");
    assert.ok(es[other] && en[key] && en[other], key);
  }
  const english = (key, p) => en[key].replace("{n}", p.n);
  assert.deepEqual(countParts(english, [["report.n_hurt", 1], ["report.n_late", 2]]), ["1 hurt", "2 said late"]);
  setLanguage("es");
  assert.match(tr("report.counts", { parts: "1 ayudó" }), /1 ayudó/);
});

test("mandatory item names and source titles have an English version", () => {
  assert.match(itemText("aviso-datos", DATA.mandatory_items, "en"), /^Legal data-protection notice/);
  assert.match(itemText("aviso-datos", DATA.mandatory_items, "es"), /^Aviso legal/);
  assert.equal(itemText("aviso-datos", [{ id: "aviso-datos", text: "x" }], "en"), "x");
  const [step] = stepsOf(VOICE, DEMO.sources, "en").filter((x) => x.kind === "bot");
  assert.equal(step.sources[0].title, "Cancelling before renewal and refunding the charge");
  assert.equal(stepsOf(VOICE, DEMO.sources, "es").find((x) => x.kind === "bot").sources[0].title, "Baja antes de la renovación y devolución del cobro");
  assert.ok(DEMO.sources.every((x) => x.title && x.en?.title));
  assert.match(es["report.top_title"], /\(ejemplo\)/);
  assert.match(en["report.top_title"], /\(example\)/);
  assert.equal(es["live.source_es"] !== "", true);
  assert.match(en["live.source_es"], /in Spanish/);
});

test("a voice that never reports its end cannot hang the example", async () => {
  assert.equal(narrationLimitMs("a b c"), TIMING.minRead * 3 + 3000);
  let fire = null;
  let cancelled = 0;
  const synth = { getVoices: () => [{ lang: "es-ES" }], speak() {}, cancel() { cancelled += 1; } };
  class Utter { constructor(text) { this.text = text; } }
  const narrator = browserNarrator(synth, Utter, "es", { schedule: (fn, ms) => { fire = { fn, ms }; return 1; }, cancel: () => {} });
  const pending = narrator.speak("hola que tal");
  assert.equal(fire.ms, narrationLimitMs("hola que tal"));
  fire.fn();
  assert.equal(await pending, false);
  assert.equal(cancelled, 1);
});

test("the voice example does not announce each step twice", () => {
  const view = read(`${WEB}views/voice-demo.js`);
  assert.doesNotMatch(view, /vd-talk[^\n]*aria-live/);
  assert.match(view, /role: "status"/);
  assert.doesNotMatch(view, /h\("code", \{\}, s\.id\)/);
});
