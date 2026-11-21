import test from "node:test";
import assert from "node:assert/strict";
import { Api, ApiFailure } from "../../transcriber/adapters/inbound/web/js/api.js";

/** An Api whose fetch records the calls and answers with `reply`. */
function recorder(reply = () => new Response(null, { status: 204 })) {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return reply(url, init);
  });
  return { api, calls };
}

const json = (status, body, headers = {}) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json", ...headers },
});

test("the consent notice is asked for by kind and language, with the code", async () => {
  const { api, calls } = recorder(() => json(200, { version: "consent-1" }));
  assert.deepEqual(await api.consent("recording", "es"), { version: "consent-1" });
  assert.equal(calls[0].url, "/api/consent?kind=recording&language=es");
  assert.equal(calls[0].init.headers["X-Access-Code"], "code-123");
});

test("a new session carries the consent version and the options chosen", async () => {
  const { api, calls } = recorder(() => json(201, {}));
  await api.createSession("s1", "es", "Reparto", {
    consent_version: "consent-1", num_speakers: 3, expected_speakers: undefined,
  });
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    id: "s1", language: "es", title: "Reparto", consent_version: "consent-1", num_speakers: 3,
  });
  await api.createSession("s2", "en");
  assert.deepEqual(JSON.parse(calls[1].init.body), { id: "s2", language: "en" });
});

test("an upload names its file and the notice shown, never the code, in the query", async () => {
  const { api, calls } = recorder(() => json(201, {}));
  await api.uploadRecording("s1", { name: "a b.m4a" }, "es", "", "consent-1");
  const url = new URL(calls[0].url, "https://x");
  assert.equal(url.pathname, "/api/sessions/s1/upload");
  assert.equal(url.searchParams.get("filename"), "a b.m4a");
  assert.equal(url.searchParams.get("consent_version"), "consent-1");
  assert.ok(!calls[0].url.includes("code-123"));
});

test("the conversation's type and goal go with a new session, an upload and a change", async () => {
  const { api, calls } = recorder(() => json(200, {}));
  await api.createSession("s1", "es", "", { type: "negociacion", goal: "Precio", consent_version: null });
  assert.deepEqual(JSON.parse(calls[0].init.body), { id: "s1", language: "es", type: "negociacion", goal: "Precio" });
  await api.uploadRecording("s2", { name: "a.m4a" }, "es", "", "consent-1", { type: "venta", goal: null });
  const url = new URL(calls[1].url, "https://x");
  assert.equal(url.searchParams.get("type"), "venta");
  assert.equal(url.searchParams.has("goal"), false, "an unset goal is left out");
  await api.uploadRecording("s3", { name: "a.m4a" }, "es", "", "consent-1");
  assert.equal(new URL(calls[2].url, "https://x").searchParams.has("type"), false);
  await api.describeSession("s 4", { type: null, goal: "Algo" });
  assert.equal(calls[3].url, "/api/sessions/s%204");
  assert.equal(calls[3].init.method, "PATCH");
  assert.equal(calls[3].init.headers["X-Access-Code"], "code-123");
  assert.deepEqual(JSON.parse(calls[3].init.body), { type: null, goal: "Algo" }, "null clears a field");
});

test("the playback cookie is a POST to /api/access/cookie with the code", async () => {
  const { api, calls } = recorder();
  assert.equal(await api.audioCookie(), null);
  assert.equal(calls[0].url, "/api/access/cookie");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["X-Access-Code"], "code-123");
});

test("409 busy and 400 unsupported_format come back with their codes", async () => {
  const { api } = recorder((url) => (url.includes("enrolment")
    ? json(409, { code: "busy", message: "busy" }, { "Retry-After": "7" })
    : json(400, { code: "unsupported_format", message: "Upload an audio" })));
  await assert.rejects(api.enrol("ana", { name: "Ana", language: "es", pcm: new Uint8Array(2) }), (error) => {
    assert.ok(error instanceof ApiFailure);
    assert.equal(error.code, "busy");
    assert.equal(error.waitSeconds, 7);
    return true;
  });
  await assert.rejects(api.uploadRecording("s1", { name: "a.txt" }, "es"), (error) => {
    assert.equal(error.code, "unsupported_format");
    assert.equal(error.kind, "client");
    return true;
  });
});

test("login sends the password without a code and returns the code", async () => {
  const { api, calls } = recorder(() => json(200, { code: "the-access-code" }));
  assert.deepEqual(await api.login("pw-1234"), { code: "the-access-code" });
  assert.equal(calls[0].url, "/api/login");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["X-Access-Code"], undefined);
  assert.deepEqual(JSON.parse(calls[0].init.body), { password: "pw-1234" });
  assert.ok(!calls[0].url.includes("pw-1234"));
});

test("a wrong password is unauthorized and a lock is a rate failure with the wait", async () => {
  const wrong = recorder(() => json(401, { code: "unauthorized", message: "Wrong password." }));
  await assert.rejects(wrong.api.login("x"), (e) => e instanceof ApiFailure && e.kind === "unauthorized");
  const locked = recorder(() => json(429, { code: "too_many_attempts", message: "x" }, { "Retry-After": "60" }));
  await assert.rejects(locked.api.login("x"), (e) => e.kind === "rate" && e.waitSeconds === 60);
});

test("cloning consent is sent only when ticked, and can be withdrawn", async () => {
  const { api, calls } = recorder(() => json(201, {}));
  await api.enrol("ana", { name: "Ana", language: "es", pcm: new Uint8Array(2) });
  assert.ok(!calls[0].url.includes("clone_consent"));
  await api.enrol("ana", { name: "Ana", language: "es", pcm: new Uint8Array(2), cloneConsent: "yes" });
  assert.ok(!calls[1].url.includes("clone_consent"), "only true counts");
  await api.enrol("ana", { name: "Ana", language: "es", pcm: new Uint8Array(2), cloneConsent: true });
  assert.match(calls[2].url, /clone_consent=true/);
  await api.withdrawVoiceClone("ana");
  assert.equal(calls[3].url, "/api/speakers/ana/voice-clone");
  assert.equal(calls[3].init.method, "DELETE");
});

test("the voice sample comes back as audio with what was used", async () => {
  const { api, calls } = recorder(() => new Response(new Uint8Array([82, 73, 70, 70]), {
    status: 200, headers: { "Content-Type": "audio/wav", "X-Voice-Clone": "1", "X-Voice-Length-Scale": "1.2", "X-Voice-Note": "" },
  }));
  const sample = await api.voiceSample({ "voicebot.voice": "clone:ana" });
  assert.equal(calls[0].url, "/api/assist/voicebot/sample");
  assert.deepEqual(JSON.parse(calls[0].init.body), { choices: { "voicebot.voice": "clone:ana" } });
  assert.equal(sample.clone, true);
  assert.equal(sample.lengthScale, 1.2);
  assert.equal(sample.note, "");
  assert.equal(sample.engine, "");
  assert.equal(sample.blob.size, 4);
  const natural = recorder(() => new Response(new Uint8Array([82]), {
    status: 200, headers: { "Content-Type": "audio/wav", "X-Voice-Engine": "pocket-cpu", "X-Voice-Note": "qwen3-cuda: no live GPU lease" },
  })).api;
  const spoke = await natural.voiceSample({ "voicebot.voice": "qwen3" });
  assert.equal(spoke.engine, "pocket-cpu");
  assert.equal(spoke.note, "qwen3-cuda: no live GPU lease");
  const failing = recorder(() => json(503, { code: "unavailable", message: "no voice bot here" })).api;
  await assert.rejects(failing.voiceSample({}), (error) => error.status === 503);
});
