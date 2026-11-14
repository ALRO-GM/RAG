import test from "node:test";
import assert from "node:assert/strict";
import {
  suggestId, uniqueId, idProblem, formatTime, sessionIdFor, countdown,
} from "../../transcriber/adapters/inbound/web/js/lib/text.js";
import { pickLanguage, translate } from "../../transcriber/adapters/inbound/web/js/lib/i18n.js";
import {
  seekTime, parseLead, activeIndex, sortUtterances, speakerName, filterBySpeaker,
  isAnonymous, mergeTargets, previousHint, needsReview, filterUtterances, doubtNote,
} from "../../transcriber/adapters/inbound/web/js/lib/transcript.js";
import {
  classifyFailure, isRetryable, nextPollMs, codeFromFragment, fragmentWithoutCode,
  adoptCodeFrom, codeFromInput,
} from "../../transcriber/adapters/inbound/web/js/lib/net.js";
import {
  progressPercent, actionsFor, sortSessions, transcriptionBody, stageLabel,
} from "../../transcriber/adapters/inbound/web/js/lib/sessions.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

test("suggestId makes lower-case ASCII ids", () => {
  assert.equal(suggestId("Ángela Núñez"), "angela-nunez");
  assert.equal(suggestId("  Ana  "), "ana");
  assert.equal(suggestId("José-Luis O'Brien"), "jose-luis-o-brien");
  assert.equal(suggestId("李雷"), "");
  assert.equal(suggestId(""), "");
  assert.equal(suggestId(null), "");
  assert.equal(suggestId("a".repeat(100)).length, 64);
});

test("suggestId avoids the reserved anonymous labels", () => {
  assert.equal(suggestId("S1"), "s1-p");
  assert.equal(idProblem("S1"), "reserved");
  assert.equal(idProblem("s12"), "reserved");
  assert.equal(idProblem("s1-p"), null);
});

test("idProblem and uniqueId", () => {
  assert.equal(idProblem(""), "empty");
  assert.equal(idProblem("-a"), "format");
  assert.equal(idProblem("a b"), "format");
  assert.equal(idProblem("a".repeat(65)), "format");
  assert.equal(idProblem("ana.b_c-d"), null);
  assert.equal(uniqueId("ana", ["ana", "ana-2"]), "ana-3");
  assert.equal(uniqueId("luis", ["ana"]), "luis");
  assert.equal(uniqueId("", ["ana"]), "");
});

test("formatTime", () => {
  assert.equal(formatTime(0), "0:00");
  assert.equal(formatTime(5.9), "0:05");
  assert.equal(formatTime(65), "1:05");
  assert.equal(formatTime(3600), "1:00:00");
  assert.equal(formatTime(3725), "1:02:05");
  assert.equal(formatTime(-3), "0:00");
  assert.equal(formatTime(NaN), "0:00");
});

test("sessionIdFor gives an id the API accepts", () => {
  const id = sessionIdFor(new Date(2026, 10, 14, 9, 5, 7), "Equipo A: reparto");
  assert.equal(id, "2026-11-14-090507-equipo-a-reparto");
  assert.equal(idProblem(id), null);
  assert.equal(sessionIdFor(new Date(2026, 0, 2, 13, 0, 0), ""), "2026-01-02-130000");
});

test("countdown", () => {
  assert.deepEqual(countdown(0, 30), { left: 30, done: false });
  assert.deepEqual(countdown(12.2, 30), { left: 18, done: false });
  assert.deepEqual(countdown(30, 30), { left: 0, done: true });
  assert.deepEqual(countdown(45, 30), { left: 0, done: true });
});

test("i18n picks by navigator.language, falls back to Spanish then to the key", () => {
  const avail = ["es", "en"];
  assert.equal(pickLanguage("es-ES", avail), "es");
  assert.equal(pickLanguage("en-GB", avail), "en");
  assert.equal(pickLanguage("EN_us", avail), "en");
  assert.equal(pickLanguage("fr-FR", avail), "es");
  assert.equal(pickLanguage(undefined, avail), "es");
  const d = { es: { hi: "Hola {name}", only: "solo" }, en: { hi: "Hi {name}" } };
  assert.equal(translate(d, "en", "hi", { name: "Ana" }), "Hi Ana");
  assert.equal(translate(d, "en", "only"), "solo");
  assert.equal(translate(d, "en", "missing"), "missing");
  assert.equal(translate(d, "es", "hi"), "Hola {name}");
});

test("both dictionaries have exactly the same keys and placeholders", () => {
  assert.deepEqual(Object.keys(en).sort(), Object.keys(es).sort());
  const holes = (s) => (s.match(/\{\w+\}/g) ?? []).sort().join();
  for (const k of Object.keys(es)) assert.equal(holes(en[k]), holes(es[k]), k);
});

test("seekTime applies the lead and never goes below zero", () => {
  assert.equal(seekTime(44.12, 3), 41.12);
  assert.equal(seekTime(1.5, 3), 0);
  assert.equal(seekTime(10, 0), 10);
  assert.equal(seekTime(10, -2), 10);
  assert.equal(seekTime(10, NaN), 10);
});

test("parseLead", () => {
  assert.equal(parseLead("2,5"), 2.5);
  assert.equal(parseLead("0"), 0);
  assert.equal(parseLead(""), 3);
  assert.equal(parseLead("-1"), 3);
  assert.equal(parseLead("abc", 5), 5);
  assert.equal(parseLead("999"), 60);
});

test("activeIndex is a binary search on start times", () => {
  const starts = [1, 4, 4.5, 9, 20];
  assert.equal(activeIndex(starts, 0), -1);
  assert.equal(activeIndex(starts, 1), 0);
  assert.equal(activeIndex(starts, 3.99), 0);
  assert.equal(activeIndex(starts, 4), 1);
  assert.equal(activeIndex(starts, 8.9), 2);
  assert.equal(activeIndex(starts, 100), 4);
  assert.equal(activeIndex([], 5), -1);
  const ends = [3, 4.4, 8, 12, 25];
  assert.equal(activeIndex(starts, 10, ends), 3);
  assert.equal(activeIndex(starts, 14, ends), 3, "within the grace after the end");
  assert.equal(activeIndex(starts, 17, ends), -1, "long silence after the sentence");
  assert.equal(activeIndex(starts, 21, ends), 4);
  const many = Array.from({ length: 5000 }, (_, i) => i * 2);
  for (const t of [0, 1.9, 2, 4999, 9998, 20000]) {
    const slow = many.reduce((acc, s, i) => (s <= t ? i : acc), -1);
    assert.equal(activeIndex(many, t), slow);
  }
});

test("sortUtterances is stable and by start", () => {
  const u = [{ id: "a", start: 5 }, { id: "b", start: 1 }, { id: "c", start: 5 }];
  assert.deepEqual(sortUtterances(u).map((x) => x.id), ["b", "a", "c"]);
});

test("speaker helpers", () => {
  const speakers = [
    { id: "ana", name: "Ana", enrolled: true },
    { id: "S1", name: null, enrolled: false },
    { id: "S2", name: "Luis", enrolled: false },
  ];
  assert.equal(speakerName(speakers, "ana"), "Ana");
  assert.equal(speakerName(speakers, "S1"), "S1");
  assert.equal(speakerName(speakers, "zzz"), "zzz");
  assert.equal(isAnonymous(speakers[0]), false);
  assert.equal(isAnonymous(speakers[1]), true);
  assert.equal(isAnonymous(speakers[2]), true);
  const us = [{ speaker: "ana" }, { speaker: "S1" }, { speaker: "ana" }];
  assert.equal(filterBySpeaker(us, "ana").length, 2);
  assert.equal(filterBySpeaker(us, "").length, 3);
  assert.deepEqual(mergeTargets(speakers, [{ id: "ana", name: "Ana" }, { id: "eva", name: "Eva" }], "S1"),
    [{ id: "ana", name: "Ana" }, { id: "eva", name: "Eva" }]);
  assert.deepEqual(mergeTargets(speakers, [], "ana"), []);
  assert.equal(previousHint({ speaker: "ana", previous_speaker: "S1" }, speakers), "S1");
  assert.equal(previousHint({ speaker: "ana", previous_speaker: "S2" }, speakers), "Luis");
  assert.equal(previousHint({ speaker: "ana", previous_speaker: null }, speakers), null);
  assert.equal(previousHint({ speaker: "ana", previous_speaker: "ana" }, speakers), null);
});

test("doubtful attributions are marked and can be shown alone", () => {
  const speakers = [{ id: "ana", name: "Ana", enrolled: true }, { id: "S1", name: null, enrolled: false }];
  const sure = { speaker: "ana", speaker_certainty: 0.8, doubtful: false, overlap: false };
  const doubt = { speaker: "ana", speaker_certainty: 0.04, doubtful: true, overlap: false, alternative_speaker: "S1" };
  const mixed = { speaker: "S1", speaker_certainty: 0.5, doubtful: false, overlap: true, alternative_speaker: "ana" };
  const old = { speaker: "S1" }; // a transcript made before the fields existed
  const us = [sure, doubt, mixed, old];
  assert.deepEqual(us.map(needsReview), [false, true, true, false]);
  assert.deepEqual(filterUtterances(us, "", true), [doubt, mixed]);
  assert.deepEqual(filterUtterances(us, "ana", true), [doubt]);
  assert.deepEqual(filterUtterances(us, "ana"), [sure, doubt]);
  assert.equal(doubtNote(sure, speakers), null);
  assert.equal(doubtNote(old, speakers), null);
  assert.deepEqual(doubtNote(doubt, speakers),
    { doubtful: true, overlap: false, alternative: "S1", certainty: 0.04 });
  assert.deepEqual(doubtNote(mixed, speakers),
    { doubtful: false, overlap: true, alternative: "Ana", certainty: 0.5 });
  assert.equal(doubtNote({ ...doubt, speaker_certainty: null, alternative_speaker: null }, speakers).certainty, null);
});

test("classifyFailure", () => {
  assert.equal(classifyFailure(null).kind, "unavailable");
  for (const s of [502, 503, 504]) assert.equal(classifyFailure(s).kind, "unavailable");
  assert.equal(classifyFailure(401).kind, "unauthorized");
  assert.deepEqual(classifyFailure(429, "12"), { kind: "rate", waitSeconds: 12 });
  assert.equal(classifyFailure(429, null).waitSeconds, 30);
  assert.equal(classifyFailure(429, "soon").waitSeconds, 30);
  assert.equal(classifyFailure(500).kind, "server");
  assert.equal(classifyFailure(409).kind, "client");
  // 409 busy (enrolment) carries Retry-After: the page waits that long.
  assert.deepEqual(classifyFailure(409, "7"), { kind: "client", waitSeconds: 7 });
  assert.equal(classifyFailure(400, "x").waitSeconds, 0);
  assert.equal(classifyFailure(422).kind, "client");
  assert.equal(isRetryable("unavailable"), true);
  assert.equal(isRetryable("server"), true);
  assert.equal(isRetryable("rate"), true);
  assert.equal(isRetryable("client"), false);
  assert.equal(isRetryable("unauthorized"), false);
});

test("nextPollMs polls fast while work is going on and backs off when quiet", () => {
  assert.equal(nextPollMs([{ state: "running" }, { state: "queued" }], 0), 2000);
  assert.equal(nextPollMs([{ state: "queued" }], 5), 4000);
  assert.equal(nextPollMs([{ state: "done" }], 0), 10000);
  assert.equal(nextPollMs([], 1), 20000);
  assert.equal(nextPollMs([], 2), 40000);
  assert.equal(nextPollMs([], 99), 60000);
  assert.equal(nextPollMs([], -1), 10000);
});

test("the access code travels in the fragment only", () => {
  assert.equal(codeFromFragment("#k=abc123"), "abc123");
  assert.equal(codeFromFragment("#/x&k=a%20b"), "a b");
  assert.equal(codeFromFragment("#/sessions"), null);
  assert.equal(codeFromFragment(""), null);
  assert.equal(codeFromFragment("#k=%E0%A4%A"), null);
  assert.equal(fragmentWithoutCode("#k=abc"), "");
  assert.equal(fragmentWithoutCode("#/x&k=abc"), "#/x");
  assert.equal(fragmentWithoutCode("#/x"), "#/x");
});

test("the code is kept and removed from the address bar", () => {
  const replaced = [];
  const kept = [];
  const history = { replaceState: (...args) => replaced.push(args) };
  const loc = { pathname: "/", search: "?x=1", hash: "#/record&k=secret-code-0123456789" };
  assert.equal(adoptCodeFrom(loc, history, (c) => kept.push(c)), true);
  assert.deepEqual(kept, ["secret-code-0123456789"]);
  assert.deepEqual(replaced, [[null, "", "/?x=1#/record"]]);
  assert.ok(!replaced[0][2].includes("secret"));
  const none = { pathname: "/", search: "", hash: "#/" };
  assert.equal(adoptCodeFrom(none, history, (c) => kept.push(c)), false);
  assert.equal(replaced.length, 1, "nothing replaced without a code");
});

test("codeFromInput takes a pasted link or a bare code", () => {
  const code = "AbC_def-0123456789xyz";
  assert.equal(codeFromInput(`https://rag.peer4u.app/#k=${code}`), code);
  assert.equal(codeFromInput(`  ${code}\n`), code);
  assert.equal(codeFromInput(`https://rag.peer4u.app/#/record&k=${code}`), code);
  assert.equal(codeFromInput("short"), null);
  assert.equal(codeFromInput("https://rag.peer4u.app/#/"), null);
  assert.equal(codeFromInput("has spaces in it 0123456789"), null);
  assert.equal(codeFromInput(""), null);
  assert.equal(codeFromInput(null), null);
});

test("job stages are shown in the page's language", () => {
  for (const stage of ["decode", "recognise", "voices", "identify", "sentences", "save"]) {
    const t = (key) => translate({ es, en }, "es", key);
    assert.notEqual(stageLabel(t, stage), stage, stage);
    assert.notEqual(stageLabel((key) => translate({ es, en }, "en", key), stage), `stage.${stage}`);
  }
  const t = (key) => translate({ es, en }, "es", key);
  assert.equal(stageLabel(t, "recognise"), es["stage.recognise"]);
  assert.equal(stageLabel(t, "something-new"), "something-new");
  assert.equal(stageLabel(t, null), "");
});

test("sessions helpers", () => {
  assert.equal(progressPercent({ state: "running", fraction: 0.456 }), 46);
  assert.equal(progressPercent({ state: "running", fraction: 3 }), 100);
  assert.equal(progressPercent({ state: "queued", fraction: 0 }), null);
  assert.equal(progressPercent({ state: "running" }), null);
  assert.deepEqual(actionsFor({ state: "recording" }), ["resume", "finish", "delete"]);
  assert.deepEqual(actionsFor({ state: "uploaded" }), ["transcribe", "delete"]);
  assert.deepEqual(actionsFor({ state: "failed" }), ["transcribe", "delete"]);
  assert.deepEqual(actionsFor({ state: "done" }), ["open", "transcribe", "delete"]);
  assert.deepEqual(actionsFor({ state: "running" }), [], "no Borrar while the job holds it");
  assert.deepEqual(actionsFor({ state: "queued" }), ["delete"]);
  const s = [{ id: "a", started_at: "2026-01-01T10:00:00+01:00" }, { id: "b", started_at: null },
    { id: "c", started_at: "2026-02-01T10:00:00+01:00" }];
  assert.deepEqual(sortSessions(s).map((x) => x.id), ["c", "a", "b"]);
  assert.deepEqual(transcriptionBody("es", [], ""), { language: "es" });
  assert.deepEqual(transcriptionBody("en", ["ana", "luis"], "3"),
    { language: "en", expected_speakers: ["ana", "luis"], num_speakers: 3 });
  assert.deepEqual(transcriptionBody("es", [], "0"), { language: "es" });
});
