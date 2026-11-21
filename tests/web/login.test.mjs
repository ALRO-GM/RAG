import test from "node:test";
import assert from "node:assert/strict";
import { loginMessage, waitText } from "../../transcriber/adapters/inbound/web/js/lib/login.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

test("each way a login ends has its own message", () => {
  assert.equal(loginMessage({ kind: "unauthorized" }).key, "login.wrong");
  assert.deepEqual(loginMessage({ kind: "rate", waitSeconds: 600 }), { key: "login.locked", values: { time: "10 min" } });
  assert.equal(loginMessage({ kind: "unavailable" }).key, "login.unavailable");
  assert.equal(loginMessage(null).key, "login.unavailable");
});

test("the wait is shown in seconds under a minute, else in minutes", () => {
  assert.equal(waitText(10), "10 s");
  assert.equal(waitText(59.2), "1 min");
  assert.equal(waitText(61), "2 min");
  assert.equal(waitText(0), "1 s");
  assert.equal(waitText(undefined), "1 s");
});

test("every login text exists in both languages", () => {
  for (const key of ["title", "body", "password", "submit", "wrong", "locked", "unavailable", "logout", "logout_busy"]) {
    assert.ok(es[`login.${key}`], `es ${key}`);
    assert.ok(en[`login.${key}`], `en ${key}`);
  }
  assert.match(es["login.locked"], /\{time\}/);
  assert.match(en["login.locked"], /\{time\}/);
});
