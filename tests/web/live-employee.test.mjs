// LV-16 (#206) on the live page without a browser: the employee and the
// other party, names given during the session, which lines are shown, how
// many suggestions stay open, the assistance mode, the mandatory items, the
// customer's discomfort (support only) and the new API calls. The texts are
// checked with the language set explicitly, in Spanish and in English.
import test from "node:test";
import assert from "node:assert/strict";
import {
  DISCOMFORT, INTERVIEW, LiveState, MAX_SUGGESTIONS, MODES, PREFS_KEY, SUPPORT,
  readPrefs, reminderDue, splitCards, viewPrefs, wordCount, writePrefs,
} from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import { setLanguage, t } from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const SUPPORT_PAGE = { product: SUPPORT, signals: true, modes: MODES, script_phases: [], roles: { S1: "agent", S2: "customer", ana: "agent" } };
const INTERVIEW_PAGE = { product: INTERVIEW, signals: false, modes: MODES, script_phases: [], roles: { S1: "interviewer", S2: "candidate" } };

function event(type, extra = {}) {
  return { type: "event", event: { schema: "live.event.v1", session_id: "s", seq: 0, type, at: "2026-11-21T10:00:00Z", stream_time: 0, ...extra } };
}

function final(id, speaker, text, start) {
  return event("final", { utterance: { id, turn: 0, index: 0, speaker, start, end: start + 1, text, revision: 0 } });
}

function suggestion(text, expressed = []) {
  return { kind: "answer", text, cites: [], about: ["t1.s0"], expressed };
}

function record(status, suggestions) {
  return { type: "suggestion", record: { status, suggestions, retrieved: [], stages: {} } };
}

function memoryStorage() {
  const kept = new Map();
  return { getItem: (k) => kept.get(k) ?? null, setItem: (k, v) => kept.set(k, String(v)), kept };
}

const broken = {
  getItem() { throw new Error("blocked"); },
  setItem() { throw new Error("blocked"); },
};

// ---- view choices ---------------------------------------------------------

test("view choices: «Ajustes» gives the defaults, this browser's choice wins", () => {
  assert.deepEqual(viewPrefs(), { showEmployee: true, showOther: true, max: 3, mode: "pincelada" });
  const saved = { "live.show_employee": "off", "live.show_other": "on", "live.max_suggestions": "5", "live.assist_mode": "guiado" };
  assert.deepEqual(viewPrefs(saved), { showEmployee: false, showOther: true, max: 5, mode: "guiado" });
  assert.deepEqual(viewPrefs({ "live.max_suggestions": "9", "live.assist_mode": "loud" }), viewPrefs());
  const mine = JSON.stringify({ showEmployee: true, showOther: false, max: 1 });
  assert.deepEqual(viewPrefs(saved, mine), { showEmployee: true, showOther: false, max: 1, mode: "guiado" });
  assert.deepEqual(viewPrefs(saved, "{not json"), viewPrefs(saved));
  assert.deepEqual(viewPrefs(saved, JSON.stringify({ max: 4, showOther: "no" })), viewPrefs(saved));
  assert.deepEqual(MAX_SUGGESTIONS, [1, 2, 3, 5]);
});

test("view choices are kept in this browser, and blocked storage is not an error", () => {
  const storage = memoryStorage();
  assert.equal(readPrefs(storage), null);
  assert.equal(writePrefs({ showEmployee: false, showOther: true, max: 2, mode: "guiado" }, storage), true);
  assert.deepEqual(JSON.parse(storage.kept.get(PREFS_KEY)), { showEmployee: false, showOther: true, max: 2 });
  assert.equal(viewPrefs({}, readPrefs(storage)).max, 2);
  assert.equal(readPrefs(broken), null);
  assert.equal(writePrefs({ showEmployee: true, showOther: true, max: 3 }, broken), false);
  assert.equal(readPrefs(undefined), null);
});

test("the newest suggestions stay open, the older ones fold", () => {
  const cards = [1, 2, 3, 4, 5].map((n) => ({ n }));
  assert.deepEqual(splitCards(cards, 3), { shown: cards.slice(0, 3), older: cards.slice(3) });
  assert.deepEqual(splitCards(cards, 1).older.length, 4);
  assert.deepEqual(splitCards(cards, 0).shown.length, 3);
  assert.deepEqual(splitCards([], 3), { shown: [], older: [] });
});

test("reminders come back every remind_seconds while items are pending", () => {
  assert.equal(reminderDue(2, 45, 0, 44_999), false);
  assert.equal(reminderDue(2, 45, 0, 45_000), true);
  assert.equal(reminderDue(0, 45, 0, 99_000), false);
  assert.equal(reminderDue(1, undefined, 0, 99_000), false);
  assert.equal(wordCount(" Ofrecer  cambio de fecha "), 4);
  assert.equal(wordCount(null), 0);
});

// ---- the session state ------------------------------------------------------

test("the employee, names given live and the sides of the conversation", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: {
    page: SUPPORT_PAGE, employee: { id: "ana", name: "Ana", role: "agent", speaker: "ana" },
    names: { ana: "Ana" }, assist_mode: "pincelada",
  } });
  state.apply(event("session_started", { speakers: [{ id: "S1", role: "customer" }, { id: "S2", role: "customer" }, { id: "ana", role: "unknown", enrolled: true }] }));
  state.apply(final("t0.s0", "S1", "Hola, quiero contratar.", 0));
  state.apply(final("t1.s0", "ana", "Le informo de que sus datos…", 2));
  assert.equal(state.roleOf("ana"), "agent");
  assert.equal(state.sideOf("ana"), "employee");
  assert.equal(state.sideOf("S1"), "other");
  assert.equal(state.nameOf("ana"), "Ana");
  assert.equal(state.nameOf("S1"), null);
  state.apply({ type: "rename", speaker: "S1", name: "Carla" });
  assert.equal(state.nameOf("S1"), "Carla");
  const ids = (opts) => state.visibleLines(opts).map((u) => u.id);
  assert.deepEqual(ids(), ["t0.s0", "t1.s0"]);
  assert.deepEqual(ids({ showEmployee: false, showOther: true }), ["t0.s0"]);
  assert.deepEqual(ids({ showEmployee: true, showOther: false }), ["t1.s0"]);
  assert.deepEqual(ids({ showEmployee: false, showOther: false }), []);
  assert.equal(state.mode, "pincelada");
  state.apply({ type: "mode", mode: "guiado" });
  state.apply({ type: "mode", mode: "loud" });
  assert.equal(state.mode, "guiado");
});

test("the mandatory checklist, notices and the hand-off follow the feed", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: SUPPORT_PAGE } });
  assert.deepEqual(state.pendingItems(), []);
  const items = [
    { id: "aviso-datos", text: "Aviso de datos", state: "pending" },
    { id: "desistimiento", text: "14 días", state: "said", utterance: "t3.s0" },
  ];
  state.apply({ type: "mandatory", checklist: { remind_seconds: 45, items } });
  assert.deepEqual(state.pendingItems().map((i) => i.id), ["aviso-datos"]);
  state.apply({ type: "notice", code: "mandatory_unavailable", message: "down" });
  assert.deepEqual(state.notices, [{ code: "mandatory_unavailable", message: "down" }]);
  const handoff = { mandatory: { missing: [{ id: "aviso-datos", text: "Aviso de datos" }], said: ["desistimiento"], ticked: [] }, names: {} };
  state.apply({ type: "handoff", handoff });
  assert.equal(state.handoff, handoff);
  const again = new LiveState();
  again.apply({ type: "session", session: { page: SUPPORT_PAGE, mandatory: { items }, handoff } });
  assert.equal(again.pendingItems().length, 1);
  assert.equal(again.handoff, handoff);
});

test("discomfort is highlighted from the newest support suggestion only", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: SUPPORT_PAGE } });
  state.apply(final("t1.s0", "S2", "Estoy harto, es la tercera vez.", 1));
  assert.equal(state.discomfort(), null);
  const upset = [{ signal: "stated_frustration", utterance: "t1.s0", quote: "Estoy harto, es la tercera vez." },
    { signal: "confusion", utterance: "t1.s0", quote: "¿Qué?" }];
  state.apply(record("ok", [suggestion("Pida disculpas", upset)]));
  const found = state.discomfort();
  assert.deepEqual(found.signals, ["stated_frustration"]);
  assert.equal(found.utterance, "t1.s0");
  assert.equal(found.quote, "Estoy harto, es la tercera vez.");
  state.apply(record("timeout", []));
  assert.ok(state.discomfort(), "a failed call keeps the last answer");
  state.apply(record("ok", [suggestion("Confirme el pedido", [{ signal: "satisfaction", utterance: "t2.s0", quote: "Gracias" }])]));
  assert.equal(state.discomfort(), null);
  assert.deepEqual(DISCOMFORT, ["complaint", "stated_frustration", "churn_risk", "urgency"]);
});

test("an interview never highlights discomfort about the candidate", () => {
  const state = new LiveState();
  state.apply({ type: "session", session: { page: { ...INTERVIEW_PAGE, signals: true } } });
  state.apply(record("ok", [suggestion("¿Y luego?", [{ signal: "complaint", utterance: "t1.s0", quote: "Fue horrible" }])]));
  assert.equal(state.discomfort(), null);
  assert.deepEqual(state.cards()[0].signals, []);
  assert.equal(state.sideOf("S1"), "employee");
  assert.equal(state.sideOf("S2"), "other");
});

// ---- API ----------------------------------------------------------------------

test("mode, names and manual ticks go to /api/assist", async () => {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
  });
  await api.liveMode("demo-1", "guiado");
  await api.renameLiveSpeaker("demo-1", "S2", "Carla");
  await api.tickMandatory("demo-1", "aviso-datos");
  await api.startLive({ product: SUPPORT, audio: "mic", session_id: "l", employee: { id: "ana", name: "Ana" } });
  assert.deepEqual(calls.map((c) => [c.init.method, c.url, JSON.parse(c.init.body)]), [
    ["POST", "/api/assist/sessions/demo-1/mode", { mode: "guiado" }],
    ["POST", "/api/assist/sessions/demo-1/speakers/S2", { name: "Carla" }],
    ["POST", "/api/assist/sessions/demo-1/mandatory/aviso-datos", { confirm: true }],
    ["POST", "/api/assist/sessions", { product: SUPPORT, audio: "mic", session_id: "l", employee: { id: "ana", name: "Ana" } }],
  ]);
});

// ---- texts ------------------------------------------------------------------

const KEYS = [
  "live.employee", "live.employee_enrol", "live.show_employee", "live.show_other", "live.max_suggestions",
  "live.assist.pincelada", "live.assist.guiado", "live.say", "live.discomfort", "live.mandatory_tick",
  "live.mandatory_confirm_body", "live.stop_pending_body", "live.summary_missing", "live.older",
];

test("the new texts exist in both languages and read in the language set", () => {
  for (const key of KEYS) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  setLanguage("es");
  assert.equal(t("live.assist.guiado"), "Guiado");
  assert.equal(t("live.mandatory_pending", { n: 2 }), "Falta por decir (2)");
  setLanguage("en");
  assert.equal(t("live.assist.guiado"), "Guided");
  assert.equal(t("live.older", { n: 4 }), "4 earlier suggestions");
  setLanguage("es");
});

test("the live page's own settings show no cost chip", async () => {
  const { costOf } = await import("../../transcriber/adapters/inbound/web/js/lib/settings.js");
  for (const step of ["live.assist_mode", "live.max_suggestions", "live.show_employee", "live.show_other"]) {
    assert.equal(costOf(step, "on"), null, step);
  }
  assert.equal(costOf("live.asr", "gpu"), "local");
});
