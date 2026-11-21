import test from "node:test";
import assert from "node:assert/strict";
import { WakeLock } from "../../transcriber/adapters/inbound/web/js/lib/wake-lock.js";

function fakeEnv({ refuse = false } = {}) {
  const listeners = {};
  const requests = [];
  const doc = {
    visibilityState: "visible",
    addEventListener: (name, fn) => { listeners[name] = fn; },
    removeEventListener: (name) => { delete listeners[name]; },
  };
  const nav = {
    wakeLock: {
      request: async (type) => {
        if (refuse) throw new Error("NotAllowedError");
        const sentinel = { type, released: false, release: async () => { sentinel.released = true; } };
        requests.push(sentinel);
        return sentinel;
      },
    },
  };
  return { doc, nav, listeners, requests };
}

test("acquire takes a screen lock and release frees it", async () => {
  const env = fakeEnv();
  const lock = new WakeLock({ navigator: env.nav, document: env.doc });
  assert.equal(await lock.acquire(), true);
  assert.equal(env.requests.length, 1);
  assert.equal(env.requests[0].type, "screen");
  assert.equal(lock.active, true);
  await lock.release();
  assert.equal(env.requests[0].released, true);
  assert.equal(lock.active, false);
  assert.equal(env.listeners.visibilitychange, undefined);
});

test("the lock is taken again when the page becomes visible", async () => {
  const env = fakeEnv();
  const lock = new WakeLock({ navigator: env.nav, document: env.doc });
  await lock.acquire();
  env.requests[0].released = true; // the browser drops it when hidden
  env.doc.visibilityState = "hidden";
  env.listeners.visibilitychange();
  await Promise.resolve();
  assert.equal(env.requests.length, 1);
  env.doc.visibilityState = "visible";
  env.listeners.visibilitychange();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(env.requests.length, 2);
  assert.equal(lock.active, true);
});

test("acquiring twice requests once", async () => {
  const env = fakeEnv();
  const lock = new WakeLock({ navigator: env.nav, document: env.doc });
  await lock.acquire();
  await lock.acquire();
  assert.equal(env.requests.length, 1);
});

test("no API or a refused request never throws", async () => {
  const lock = new WakeLock({ navigator: {}, document: fakeEnv().doc });
  assert.equal(lock.supported, false);
  assert.equal(await lock.acquire(), false);
  await lock.release();
  const env = fakeEnv({ refuse: true });
  const refused = new WakeLock({ navigator: env.nav, document: env.doc });
  assert.equal(await refused.acquire(), false);
  await refused.release();
});

test("a release while the request is pending frees the late lock", async () => {
  const env = fakeEnv();
  const lock = new WakeLock({ navigator: env.nav, document: env.doc });
  const pending = lock.acquire();
  await lock.release();
  assert.equal(await pending, false);
  assert.equal(env.requests[0].released, true);
  assert.equal(lock.active, false);
});
