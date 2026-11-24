// #249 on the live page without a browser: the employee's voice check of a
// one-microphone session («Preséntate…» → «Reconociendo tu voz…» → «Voz
// reconocida»), the lines the attribution is not sure of, the «Esta soy yo» /
// «Esta no soy yo» corrections and their API call. Texts are checked with the
// language set explicitly (CI's Node picks English by itself).
import test from "node:test";
import assert from "node:assert/strict";
import {
  INTERVIEW, LiveState, MODES, voiceProgress, voiceStatus,
} from "../../transcriber/adapters/inbound/web/js/lib/live.js";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import { setLanguage, t } from "../../transcriber/adapters/inbound/web/js/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const PAGE = { product: INTERVIEW, signals: false, modes: MODES, script_phases: [], roles: { ana: "interviewer", S2: "candidate" } };

function event(type, extra = {}) {
  return { type: "event", event: { schema: "live.event.v1", session_id: "s", seq: 0, type, at: "2026-11-24T10:00:00Z", stream_time: 0, ...extra } };
}

function utterance(id, speaker, extra = {}) {
  return { id, turn: 0, index: 0, speaker, start: 1, end: 2, text: "hola", revision: 0, ...extra };
}

function check(state, seconds, needed = 5) {
  return event("voice_check", { voice_check: { state, speaker: "ana", seconds, needed } });
}

function session() {
  const state = new LiveState();
  state.apply({ type: "session", session: { id: "s", state: "running", mode: "live", page: PAGE, employee: { id: "ana", name: "Ana", role: "interviewer", speaker: "ana" } } });
  return state;
}

test("the check reads listening, recognising, then recognised", () => {
  const state = session();
  assert.equal(state.voiceStatus(), null);
  state.apply(check("listening", 0));
  assert.equal(state.voiceStatus(), "listening");
  assert.equal(voiceProgress(state.voice), 0);
  state.apply(check("listening", 2.5));
  assert.equal(voiceProgress(state.voice), 0.5);
  state.apply(check("listening", 6));
  assert.equal(state.voiceStatus(), "recognising");
  assert.equal(voiceProgress(state.voice), 1);
  state.apply(check("recognised", 6));
  assert.equal(state.voiceStatus(), "recognised");
});

test("other states pass through; nothing without a check", () => {
  for (const s of ["short", "skipped", "failed"]) assert.equal(voiceStatus({ state: s, seconds: 1, needed: 5 }), s);
  assert.equal(voiceStatus(null), null);
  assert.equal(voiceStatus({ state: "listening", seconds: 3, needed: 0 }), "listening");
  assert.equal(voiceProgress({ needed: 0 }), 0);
  assert.equal(voiceProgress({ seconds: "x", needed: 5 }), 0);
});

test("lines carry the unsure mark and keep it through speaker updates", () => {
  const state = session();
  state.apply(check("recognised", 5));
  state.apply(event("final", { utterance: utterance("t1.s0", "S2", { speaker_unsure: true }) }));
  assert.equal(state.utterances.get("t1.s0").unsure, true);
  state.apply(event("speaker_update", { utterance: utterance("t1.s0", "ana", { revision: 1 }) }));
  const line = state.utterances.get("t1.s0");
  assert.equal(line.speaker, "ana");
  assert.equal(line.unsure, false);
});

test("each line offers the correction of its side, only with a check", () => {
  const state = session();
  state.apply(event("final", { utterance: utterance("t1.s0", "S2") }));
  state.apply(event("final", { utterance: utterance("t2.s0", "ana") }));
  const [theirs, mine] = [state.utterances.get("t1.s0"), state.utterances.get("t2.s0")];
  assert.equal(state.correctionFor(theirs), null); // no voice check in this session
  state.apply(check("skipped", 0));
  assert.equal(state.correctionFor(theirs), "mine");
  assert.equal(state.correctionFor(mine), "not_mine");
  assert.equal(state.correctionFor({ ...theirs, partial: true }), null);
  state.state = "ended";
  assert.equal(state.correctionFor(theirs), null);
});

test("skip and corrections go straight to the transcriber's live session", async () => {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
  });
  await api.liveVoice("live-1", { action: "skip" });
  await api.liveVoice("live-1", { action: "mine", utterance: "t3.s0" });
  assert.deepEqual(calls.map((c) => [c.init.method, c.url, JSON.parse(c.init.body)]), [
    ["POST", "/api/live/sessions/live-1/voice", { action: "skip" }],
    ["POST", "/api/live/sessions/live-1/voice", { action: "mine", utterance: "t3.s0" }],
  ]);
});

const KEYS = [
  "live.audio_hint.mic_check", "live.voice.listening", "live.voice.listening_head", "live.voice.listening_hint",
  "live.voice.recognising", "live.voice.progress", "live.voice.private", "live.voice.skip", "live.voice.chip_recognising",
  "live.voice.chip_recognised", "live.voice.chip_skipped", "live.voice.short", "live.voice.failed", "live.voice.unsure",
  "live.voice.mine", "live.voice.not_mine", "live.voice.done_mine", "live.voice.done_not_mine",
];

test("the voice check's texts exist in both languages and read in the language set", () => {
  for (const key of KEYS) {
    assert.ok(es[key], `es ${key}`);
    assert.ok(en[key], `en ${key}`);
  }
  setLanguage("es");
  assert.equal(t("live.voice.listening"), "Preséntate para que reconozca tu voz");
  assert.equal(t("live.voice.chip_recognised"), "Voz reconocida");
  assert.equal(t("live.voice.mine"), "Esta soy yo");
  assert.equal(t("live.voice.not_mine"), "Esta no soy yo");
  setLanguage("en");
  assert.equal(t("live.voice.mine"), "This is me");
  setLanguage("es");
});
