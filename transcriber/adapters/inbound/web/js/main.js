// Entry point: the access code, the router, the screens for "no code",
// "workstation not available" and "too many attempts".

import { h } from "./dom.js";
import { t, translateStatic } from "./i18n.js";
import { Api } from "./api.js";
import { adoptCodeFrom } from "./lib/net.js";
import { isBusy, setLeaveGuard } from "./device.js";
import { sessionsView } from "./views/sessions.js";
import { recordView } from "./views/record.js";
import { enrolView } from "./views/enrol.js";
import { transcriptView } from "./views/transcript.js";

const CODE_KEY = "transcriber.code";
let memoryCode = null;

function readCode() {
  if (memoryCode) return memoryCode;
  try {
    return sessionStorage.getItem(CODE_KEY);
  } catch {
    return null;
  }
}

function keepCode(code) {
  memoryCode = code;
  try {
    sessionStorage.setItem(CODE_KEY, code);
  } catch {
    // kept in memory for this visit
  }
}

function forgetCode() {
  memoryCode = null;
  try {
    sessionStorage.removeItem(CODE_KEY);
  } catch {
    // nothing stored
  }
}

/** The code arrives as #k=<code>: keep it, then take it out of the address bar. */
function adoptCode() {
  return adoptCodeFrom(location, history, keepCode);
}

// Views that want to know when a new code arrives without a reload (a
// recording that must not be interrupted).
const codeListeners = new Set();

adoptCode();

const app = document.getElementById("app");
const api = new Api(readCode);
let cleanup = null;
let token = 0;

const ROUTES = [
  [/^#?\/?$/, sessionsView],
  [/^#\/record(?:\/([^/]+))?$/, recordView],
  [/^#\/enrol$/, enrolView],
  [/^#\/s\/([^/]+)$/, transcriptView],
];

const ctx = {
  api,
  navigate(hash) {
    if (location.hash === hash) route();
    else location.hash = hash;
  },
  fail(error, retry) {
    blocking(error, retry ?? route);
  },
  /** Use `code` from now on (pasted by the person). */
  setCode(code) {
    keepCode(code);
  },
  /** Call `listener` when a new code arrives by link; returns the unsubscribe. */
  onNewCode(listener) {
    codeListeners.add(listener);
    return () => codeListeners.delete(listener);
  },
};

function resolve(hash) {
  for (const [pattern, view] of ROUTES) {
    const m = pattern.exec(hash);
    if (m) return { view, params: m.slice(1).filter((x) => x !== undefined) };
  }
  return { view: sessionsView, params: [] };
}

function stopCurrent() {
  try {
    cleanup?.();
  } catch {
    // a view that cannot clean up must not block the next one
  }
  cleanup = null;
}

async function route() {
  stopCurrent();
  const mine = ++token;
  app.replaceChildren();
  if (!readCode()) return noCode();
  const { view, params } = resolve(location.hash);
  try {
    const done = await view(ctx, app, params);
    if (mine !== token) done?.();
    else cleanup = done ?? null;
  } catch (error) {
    if (mine === token) blocking(error, route);
  }
}

function noCode() {
  app.replaceChildren(h("section", { class: "stack gate" },
    h("h1", {}, t("gate.title")), h("p", {}, t("gate.body"))));
}

/** A call that the page cannot go on without failed: say why, offer to retry. */
function blocking(error, retry) {
  stopCurrent();
  token += 1;
  if (error?.kind === "unauthorized") {
    forgetCode();
    return noCode();
  }
  const retryButton = h("button", { type: "button", class: "primary big" }, t("action.retry"));
  const status = h("p", { role: "status", class: "hint" });
  let title = t("err.unavailable");
  let body = t("err.unavailable_body");
  if (error?.kind === "rate") {
    title = t("err.rate_title");
    body = t("err.rate", { seconds: error.waitSeconds });
    retryButton.disabled = true;
    setTimeout(() => { retryButton.disabled = false; }, error.waitSeconds * 1000);
  } else if (error?.kind !== "unavailable") {
    title = t("err.generic");
    body = error?.message ?? "";
  }
  retryButton.addEventListener("click", async () => {
    retryButton.disabled = true;
    status.textContent = t("err.checking");
    try {
      await api.health();
    } catch {
      status.textContent = t("err.still_down");
      retryButton.disabled = false;
      return;
    }
    retry();
  });
  app.replaceChildren(h("section", { class: "stack gate", role: "alert" }, h("h1", {}, title), h("p", {}, body), retryButton, status));
  retryButton.focus();
}

setLeaveGuard(() => false);
translateStatic(document);
let lastHash = location.hash;
let reverting = false;
window.addEventListener("hashchange", () => {
  if (adoptCode()) {
    if (isBusy()) {
      // A recording is going on: keep the page, use the new code.
      history.replaceState(null, "", location.pathname + location.search + lastHash);
      for (const listener of codeListeners) listener();
      return;
    }
    lastHash = location.hash;
    route();
    return;
  }
  if (reverting) {
    reverting = false;
    return;
  }
  if (isBusy() && !confirm(t("leave.confirm"))) {
    reverting = true;
    location.hash = lastHash;
    return;
  }
  lastHash = location.hash;
  route();
});
route();
