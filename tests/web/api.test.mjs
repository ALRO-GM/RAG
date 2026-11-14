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
