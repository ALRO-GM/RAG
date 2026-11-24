// The connection check's requests (#247): a few small GETs to the public
// server (/healthz) and through it to the processing server (/api/health),
// the services' words (/api/connection) and one small upload
// (/api/connection/upload), each timed with the page's clock and given up
// after a timeout. Only our own origin is asked. The figures and the
// verdict are worked out in lib/connection.js.

import { CHECK } from "./lib/connection.js";

/**
 * One timed request: { ok, ms, status } (status null when it failed or
 * timed out). `ok` means an answer within the timeout with a status that
 * `accept` takes.
 */
export async function timed(fetchFn, now, url, { timeoutMs = CHECK.timeoutMs, accept = (s) => s === 200, init = {} } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const started = now();
  try {
    const response = await fetchFn(url, { cache: "no-store", credentials: "same-origin", ...init, signal: controller.signal });
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    const ms = now() - started;
    return { ok: accept(response.status), ms, status: response.status, body };
  } catch {
    return { ok: false, ms: now() - started, status: null, body: null };
  } finally {
    clearTimeout(timer);
  }
}

/** Sequential pings; stops early after two time-outs in a row (the check stays short). */
async function pings(fetchFn, now, url, count, accept) {
  const samples = [];
  let missed = 0;
  for (let i = 0; i < count && missed < 2; i += 1) {
    const answer = await timed(fetchFn, now, url, { accept });
    samples.push(answer);
    missed = answer.status === null ? missed + 1 : 0;
  }
  return samples;
}

const anyAnswer = (status) => status !== null && status > 0;
/** The processing server answers 200; 502-504 come from the public server when the link is down. */
const linkAnswer = (status) => status === 200;

/**
 * Run one check. `getCode` gives the access code (the services and the
 * upload are behind it); `upload` whether to time the upload. Returns the
 * `result` lib/connection.js assesses, with `at` (the page's clock).
 */
export async function runCheck({
  getCode, upload = true,
  fetchFn = (...args) => globalThis.fetch(...args),
  now = () => globalThis.performance.now(),
} = {}) {
  // The public server: any HTTP answer means it is reachable (a development
  // server without /healthz answers 404, which is still an answer).
  const web = await pings(fetchFn, now, "/healthz", CHECK.webPings, anyAnswer);
  const all = await pings(fetchFn, now, "/api/health", CHECK.linkPings, linkAnswer);
  const linkSamples = all.slice(1).map(({ ok, ms }) => ({ ok, ms }));
  const reachable = all.some((s) => s.ok);
  const result = {
    web: { reachable: web.some((s) => s.status !== null), samples: web.map(({ ok, ms }) => ({ ok, ms })) },
    link: { reachable, samples: reachable ? linkSamples : [] },
    services: null,
    upload: null,
  };
  const code = getCode?.();
  if (!reachable || !code) return result;
  const headers = { "X-Access-Code": code };
  const services = await timed(fetchFn, now, "/api/connection", { init: { headers } });
  if (services.ok && services.body && typeof services.body.services === "object") result.services = services.body.services;
  if (upload) {
    const sent = await timed(fetchFn, now, "/api/connection/upload", {
      timeoutMs: CHECK.timeoutMs * 2,
      init: { method: "POST", headers: { ...headers, "Content-Type": "application/octet-stream" }, body: new Uint8Array(CHECK.uploadBytes) },
    });
    if (sent.ok) result.upload = { bytes: CHECK.uploadBytes, ms: sent.ms };
  }
  return result;
}

/** One in-session ping of the processing server: { ok, ms }. */
export async function pingOnce({ timeoutMs, fetchFn = (...args) => globalThis.fetch(...args), now = () => globalThis.performance.now() } = {}) {
  const { ok, ms } = await timed(fetchFn, now, "/api/health", { timeoutMs, accept: linkAnswer });
  return { ok, ms };
}

/** The services' words now, or null when they cannot be asked. */
export async function servicesOnce({ getCode, fetchFn = (...args) => globalThis.fetch(...args), now = () => globalThis.performance.now() } = {}) {
  const code = getCode?.();
  if (!code) return null;
  const answer = await timed(fetchFn, now, "/api/connection", { init: { headers: { "X-Access-Code": code } } });
  return answer.ok && answer.body && typeof answer.body.services === "object" ? answer.body.services : null;
}
