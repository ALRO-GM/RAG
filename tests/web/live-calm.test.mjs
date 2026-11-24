// The calm live page (LV-17, #232) without a browser: the one «Ahora»
// card, «Hecho» when the employee asks what was proposed, lines better
// avoided and «Reformulado», the topic chips, and the scripted demo's
// timeline fed message by message as the orchestrator sends it.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  INTERVIEW, LiveState, clockText, contentWords, echoes,
} from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { SERVICES } from "../../transcriber/adapters/inbound/web/js/lib/home.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const DEMO = JSON.parse(readFileSync(new URL("../../examples/demo/entrevista-lucia.demo.json", import.meta.url), "utf8"));
const PAGE = {
  product: INTERVIEW, signals: false, modes: ["pincelada", "guiado"],
  script_phases: ["opening", "experience", "technical", "behavioural", "closing"],
  roles: { entrevistador: "interviewer", candidata: "candidate" },
};

function event(type, u, extra = {}) {
  return { type: "event", event: { type, stream_time: u ? u.end : 0, utterance: u ? { ...u, revision: 0 } : undefined, ...extra } };
}

function record(call, after, suggestion) {
  return { type: "suggestion", record: {
    call, status: "ok", trigger: { utterance: after.id, stream_time: after.end }, stages: {}, backend: "demo-script",
    suggestions: [{ about: [after.id], cites: [], ...suggestion }],
  } };
}

/** The orchestrator's feed for the demo, as ScriptedAssist publishes it. */
function demoFeed() {
  const out = [{ type: "session", session: { id: "demo-1", mode: "demo", page: PAGE, demo: { id: DEMO.id, title: DEMO.title, candidate: DEMO.candidate, result_session: DEMO.result_session } } }];
  out.push({ type: "topics", topics: DEMO.topics.map((x) => ({ id: x.id, name: x.name, done: false })) });
  out.push({ type: "event", event: { type: "session_started", stream_time: 0, speakers: DEMO.transcript.speakers.map((s) => ({ ...s, role: PAGE.roles[s.id] })) } });
  const covered = new Set();
  let call = 0;
  const flags = [];
  for (const u of DEMO.transcript.utterances) {
    const words = u.text.split(" ");
    out.push(event("partial", { ...u, text: words.slice(0, Math.ceil(words.length / 2)).join(" ") }));
    out.push(event("final", u));
    for (const s of DEMO.suggestions.filter((x) => x.after === u.id)) {
      call += 1;
      out.push(record(call, u, { kind: s.kind, text: s.text, example: s.example, script_phase: s.script_phase }));
    }
    for (const a of DEMO.avoid.filter((x) => x.utterance === u.id)) {
      flags.push({ ...a, at: u.start, state: "flagged" });
      out.push({ type: "avoid", flags: flags.map((f) => ({ ...f })) });
    }
    if (u.text.startsWith("Perdona") && flags.length) {
      flags[0].state = "rephrased";
      flags[0].rephrased_by = u.id;
      out.push({ type: "avoid", flags: flags.map((f) => ({ ...f })) });
    }
    for (const x of DEMO.topics.filter((y) => y.covered_by === u.id)) {
      covered.add(x.id);
      out.push({ type: "topics", topics: DEMO.topics.map((y) => ({ id: y.id, name: y.name, done: covered.has(y.id) })) });
    }
  }
  out.push({ type: "status", session: { id: "demo-1", state: "ended", page: PAGE } });
  return out;
}

function nowLabel(state) {
  const now = state.now();
  if (now.kind === "idle") return "idle";
  if (now.kind === "avoid") return `avoid:${now.flag.category}${now.done ? "+done" : ""}`;
  return `${now.card.suggestion.text}${now.used ? "+done" : ""}`;
}

test("content words drop short words, accents and punctuation", () => {
  assert.deepEqual(contentWords("¿Cómo supiste que el problema…?"), ["como", "supiste", "problema"]);
  assert.equal(echoes("¿Cómo comprobasteis que nada se rompía al migrar?", "Y ¿cómo comprobasteis que nada se rompía al migrar el sistema?"), true);
  assert.equal(echoes("Pregunta cómo midió la mejora", "¿Cómo supiste que el problema estaba en la base de datos?"), false);
  assert.equal(echoes("", "algo"), false);
  assert.equal(clockText(0), "00:00");
  assert.equal(clockText(69.5), "01:09");
  assert.equal(clockText(-3), "00:00");
});

test("the demo shows one thing at a time, in the mock-up's order", () => {
  const state = new LiveState();
  const seen = [];
  for (const message of demoFeed()) {
    state.apply(message);
    const label = nowLabel(state);
    if (seen.at(-1) !== label) seen.push(label);
  }
  assert.deepEqual(seen, [
    "idle",
    "Pregunta cómo midió la mejora", "Pregunta cómo midió la mejora+done",
    "Aún no habéis hablado de pruebas",
    "avoid:age", "avoid:age+done",
    "Aún no habéis hablado de pruebas", "Aún no habéis hablado de pruebas+done",
    "Pide un ejemplo de trabajo en equipo", "Pide un ejemplo de trabajo en equipo+done",
    "Cierra con los siguientes pasos", "Cierra con los siguientes pasos+done",
  ]);
  assert.equal(state.state, "ended");
  assert.equal(state.demo.result_session, "demo-entrevista-lucia-marin");
  assert.deepEqual(state.topicChips().map((c) => c.done), [true, true, true, true, true]);
  assert.equal(state.flagsOf("t4.s0")[0].state, "rephrased");
  assert.deepEqual(state.flagsOf("t4.s1"), []);
  assert.equal(state.earlier().length, 3);
  assert.ok(state.streamTime > 60 && state.streamTime < 70);
});

test("the speaking side and the topics of a plain interview", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: PAGE } });
  assert.equal(state.speaking(), null);
  state.apply(event("partial", { id: "u1", speaker: "candidata", text: "Cinco", start: 1, end: 2 }));
  assert.equal(state.speaking(), "other");
  state.apply(event("final", { id: "u1", speaker: "candidata", text: "Cinco años.", start: 1, end: 2 }));
  assert.equal(state.speaking(), null);
  // Without a demo the chips are the script phases, done once a suggestion was about them.
  assert.deepEqual(state.topicChips().map((c) => c.done), [false, false, false, false, false]);
  state.apply(record(1, { id: "u1", end: 2 }, { kind: "question", text: "Pregunta por Python", script_phase: "technical" }));
  assert.deepEqual(state.topicChips().filter((c) => c.done).map((c) => c.phase), ["technical"]);
  assert.equal(nowLabel(state), "Pregunta por Python");
});

test("a guided sentence counts as done when the employee says it", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: PAGE } });
  const after = { id: "u1", speaker: "candidata", text: "Migré la API.", start: 1, end: 2 };
  state.apply(event("final", after));
  state.apply({ type: "suggestion", record: { call: 1, status: "ok", mode: "guiado", trigger: { stream_time: 2 }, suggestions: [{ kind: "question", text: "¿Cómo mediste la mejora de la API?" }] } });
  assert.equal(state.now().used, false);
  state.apply(event("final", { id: "u2", speaker: "entrevistador", text: "¿Y cómo mediste la mejora de la API?", start: 3, end: 4 }));
  assert.equal(state.now().used, true);
});

test("a rephrased flag gives way to a newer suggestion", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: PAGE, avoid: [{ utterance: "u1", category: "age", title: "Edad", why: "w", alternative: "a", state: "flagged" }] } });
  assert.equal(nowLabel(state), "avoid:age");
  state.apply(event("final", { id: "u2", speaker: "entrevistador", text: "¿Cuántos años con Python?", start: 3, end: 4 }));
  state.apply({ type: "avoid", flags: [{ utterance: "u1", category: "age", title: "Edad", why: "w", alternative: "a", state: "rephrased", rephrased_by: "u2" }] });
  assert.equal(nowLabel(state), "avoid:age+done");
  state.apply(record(1, { id: "u2", end: 4 }, { kind: "question", text: "Pregunta por pruebas" }));
  assert.equal(nowLabel(state), "Pregunta por pruebas");
  state.apply({ type: "avoid", flags: "junk" });
  state.apply({ type: "topics", topics: [{ id: "a", name: "A" }, null, { id: "b" }] });
  assert.deepEqual(state.topicChips(), [{ id: "a", name: "A", done: false }]);
});

test("the home offers the sample interview and every new text is in both languages", () => {
  assert.equal(SERVICES.find((s) => s.id === "live").demo, "#/live/demo");
  for (const key of ["live.now", "live.avoid", "live.avoid_done", "live.now_used", "live.topics", "live.history",
    "live.options", "live.end_result", "live.see_result", "live.scripted.title", "home.live.demo", "live.notice.avoid_unavailable"]) {
    assert.ok(es[key], key);
    assert.ok(en[key], key);
  }
  assert.equal(es["live.avoid"], "Mejor evitar");
  assert.equal(es["live.avoid_done"], "Reformulado");
  // Hints are about the question: no word rates the candidate.
  for (const text of [es["live.avoid_hint"], en["live.avoid_hint"]]) assert.match(text, /no|not/);
});
