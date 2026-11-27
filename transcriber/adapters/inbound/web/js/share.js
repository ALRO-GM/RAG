// «Compartir pantalla» (#278): one device shares, the others join with a
// name and 4 digits, and every device shows the same application, not as
// video: the sharing device's page is mirrored as DOM (js/lib/mirror.js)
// through the relay (WS /api/share) and the actions of joined devices that
// may send are applied on the sharing device. Only the sharing device runs
// the application, so its live session, its microphone and its timers are
// the only ones: nothing can drift. docs/web/SCREEN_SHARE.md explains why.
//
// This module draws the button, the panel, the indicator on every device,
// the joined device's stage (an <iframe> without scripts that holds the
// mirror, in this device's layout or the sharing device's, scaled), the
// pointer marker, and wires the sound (share-sound.js). The panel, the
// indicator and the button carry data-share-skip: they are never mirrored.

import { h, fill } from "./dom.js";
import { t } from "./i18n.js";
import { icon } from "./ui.js";
import { WakeLock } from "./device.js";
import { WIDE_QUERY, actingAs } from "./shell.js";
import {
  ACTIONABLE, MirrorSource, MirrorView, elementScrollOf, elementScrollTo, pointerAt, scrollOf, scrollTo,
} from "./lib/mirror.js";
import {
  ShareLink, cleanLabel, deviceKind, endedKey, externalUrl, randomPin, refusalKey, socketUrl, statusKey, validName, validPin,
} from "./lib/share.js";
import { SoundPlayer, installSoundBridge } from "./share-sound.js";

const TICK_MS = 100;
const POINTER_MS = 50;
const POINTER_HIDE_MS = 2500;
const LOCAL_SCROLL_MS = 700;
const RESYNC_MS = 2000;
const KEEP_KEY = "ragputin.share";
/** A field typed in on a joined device keeps its own value this long after the last key (the echo is older). */
const HOLD_MS = 1200;
/** Longest value a joined device sends for a field (its messages are at most 16 KB). */
const MAX_VALUE = 4000;
/**
 * «El sonido sale por» (#278 stage 2): the code to play the sound on a
 * joined device exists (share-sound.js) but was never run in a browser
 * with real voices, so the choice is not offered: the sound comes out of
 * the sharing device.
 */
const SOUND_CHOICE = false;
const BLOCKED = ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "keydown", "keyup",
  "keypress", "beforeinput", "wheel", "touchstart", "touchmove", "touchend", "dragstart"];

function keep(value) {
  try {
    if (value) sessionStorage.setItem(KEEP_KEY, JSON.stringify(value));
    else sessionStorage.removeItem(KEEP_KEY);
  } catch {
    // a reload then asks for the password again
  }
}

function kept() {
  try {
    const value = JSON.parse(sessionStorage.getItem(KEEP_KEY) ?? "null");
    return value && typeof value.token === "string" && ["host", "viewer"].includes(value.role) ? value : null;
  } catch {
    return null;
  }
}

/** The address of the link at `node` when it leaves this site (#284), else null. */
function leavingUrl(node) {
  const link = node?.closest?.("a[href]");
  return link ? externalUrl(link.getAttribute("href"), location.href) : null;
}

/** The theme the sharing device shows, resolved (a joined device may follow another system theme). */
function rootState() {
  const root = document.documentElement;
  const dark = globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return {
    lang: root.lang, theme: root.dataset.theme ?? (dark ? "dark" : "light"),
    cls: [...root.classList].filter((c) => !c.startsWith("share-")),
  };
}

/**
 * Add «Compartir pantalla» to the page.
 *   api      the Api (its access code, the playback cookie)
 *   readCode () => the access code (sent as the socket's first message, never mirrored)
 *   isBusy   () => true while this device records
 *   pause    () => stop the application on this device (it joins)
 *   resume   () => show the application again (it left)
 */
export function installShare({ api, readCode, isBusy = () => false, pause = () => {}, resume = () => {} }) {
  const root = document.documentElement;
  const panel = h("dialog", { class: "share-panel", "data-share-skip": true, "aria-labelledby": "share-title" });
  const pill = h("div", { class: "share-pill", "data-share-skip": true, hidden: true });
  const marker = h("div", { class: "share-pointer", "data-share-skip": true, "aria-hidden": "true", hidden: true });
  document.body.append(panel, pill, marker);
  panel.addEventListener("click", (event) => { if (event.target === panel) panel.close(); });

  let s = null; // the running share: role, link, roster, …

  const scrub = (text) => {
    const code = readCode();
    return code && code.length >= 8 && text.includes(code) ? text.split(code).join("••••••") : text;
  };

  // ---- the panel ----------------------------------------------------------

  function openPanel(tab = "share") {
    renderPanel(tab);
    if (!panel.open) panel.showModal();
    panel.querySelector("input, button.primary")?.focus();
  }

  function renderPanel(tab = "share") {
    const close = h("button", { type: "button", class: "icon-button share-close", "aria-label": t("share.close"), onClick: () => panel.close() }, icon("close"));
    const head = h("div", { class: "share-head" }, h("h2", { id: "share-title" }, t("share.title")), close);
    if (!s) fill(panel, head, choosePanel(tab));
    else if (s.role === "host") fill(panel, head, hostPanel());
    else fill(panel, head, viewerPanel());
  }

  function choosePanel(tab) {
    const tabs = h("div", { class: "share-tabs", role: "tablist", "aria-label": t("share.title") },
      ["share", "join"].map((name) => h("button", {
        type: "button", role: "tab", id: `share-tab-${name}`, "aria-selected": String(tab === name), "aria-controls": "share-form",
        tabindex: tab === name ? "0" : "-1", onClick: () => openPanel(name),
        onKeydown: (event) => {
          if (event.key === "ArrowRight" || event.key === "ArrowLeft") openPanel(name === "share" ? "join" : "share");
        },
      }, t(`share.tab.${name}`))));
    return h("div", { class: "stack" }, h("p", { class: "hint" }, t("share.intro")), tabs,
      tab === "share" ? shareForm() : joinForm());
  }

  function field(id, label, input, hint = null) {
    return h("div", { class: "field" }, h("label", { for: id }, label), input, hint ? h("p", { class: "hint", id: `${id}-hint` }, hint) : null);
  }

  function defaultLabel() {
    return t(`share.kind.${deviceKind(navigator.userAgent, navigator.maxTouchPoints ?? 0)}`);
  }

  function shareForm() {
    const name = h("input", { id: "share-name", required: true, maxLength: 40, autocomplete: "off", value: t("share.default_name") });
    const pin = h("input", { id: "share-pin", required: true, inputMode: "numeric", pattern: "[0-9]{4}", maxLength: 4, autocomplete: "off", value: randomPin(), "aria-describedby": "share-pin-hint" });
    const label = h("input", { id: "share-label", maxLength: 40, autocomplete: "off", value: defaultLabel() });
    const error = h("p", { class: "hint share-error", role: "alert" });
    const again = h("button", { type: "button", class: "ghost small", onClick: () => { pin.value = randomPin(); pin.focus(); } }, t("share.pin_new"));
    const form = h("form", { class: "stack", id: "share-form", role: "tabpanel", "aria-labelledby": "share-tab-share", novalidate: true },
      h("p", { class: "hint" }, t("share.share_lead")),
      field("share-name", t("share.name"), name),
      field("share-pin", t("share.pin"), h("div", { class: "row" }, pin, again), t("share.pin_hint")),
      field("share-label", t("share.label"), label),
      error, h("button", { type: "submit", class: "primary" }, icon("share"), t("share.start")));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!validName(name.value)) return say(error, t("share.error.name"), name);
      if (!validPin(pin.value)) return say(error, t("share.error.pin"), pin);
      startHost({ name: name.value.trim(), pin: pin.value, label: cleanLabel(label.value) || defaultLabel() }, error);
      return null;
    });
    return form;
  }

  function joinForm() {
    const name = h("input", { id: "share-name", required: true, maxLength: 40, autocomplete: "off" });
    const pin = h("input", { id: "share-pin", required: true, inputMode: "numeric", pattern: "[0-9]{4}", maxLength: 4, autocomplete: "off" });
    const label = h("input", { id: "share-label", maxLength: 40, autocomplete: "off", value: defaultLabel() });
    const error = h("p", { class: "hint share-error", role: "alert" });
    const form = h("form", { class: "stack", id: "share-form", role: "tabpanel", "aria-labelledby": "share-tab-join", novalidate: true },
      h("p", { class: "hint" }, t("share.join_lead")),
      field("share-name", t("share.name"), name),
      field("share-pin", t("share.pin"), pin),
      field("share-label", t("share.label"), label),
      error, h("button", { type: "submit", class: "primary" }, t("share.join")));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (isBusy()) return say(error, t("share.error.busy_here"), null);
      if (!validName(name.value)) return say(error, t("share.error.name"), name);
      if (!validPin(pin.value)) return say(error, t("share.error.pin"), pin);
      startViewer({ name: name.value.trim(), pin: pin.value, label: cleanLabel(label.value) || defaultLabel() }, error);
      return null;
    });
    return form;
  }

  function say(where, text, focus) {
    where.textContent = text;
    focus?.focus();
    return null;
  }

  function deviceRow(device) {
    const mine = device.id === s.me;
    const state = device.online ? t("share.online") : t("share.away");
    const sends = h("input", {
      type: "checkbox", role: "switch", id: `share-sends-${device.id}`, checked: device.sends,
      disabled: s.role !== "host" && !(mine && device.sends),
      onChange: (event) => {
        if (s.role === "host") s.link.send({ type: "set", device: device.id, sends: event.target.checked });
        else s.link.send({ type: "watch" });
      },
    });
    return h("li", { class: "share-device" },
      h("div", {}, h("strong", {}, device.label),
        h("span", { class: "hint" }, " · ", device.host ? t("share.role.host") : t("share.role.viewer"), mine ? ` · ${t("share.this_device")}` : "", " · ", state)),
      h("label", { class: "share-switch", for: `share-sends-${device.id}` }, sends, h("span", {}, t("share.sends"))),
      s.role === "host" && !device.host
        ? h("button", { type: "button", class: "ghost small", onClick: () => s.link.send({ type: "remove", device: device.id }) }, t("share.remove"))
        : null);
  }

  function rosterList() {
    const devices = s.roster?.devices ?? [];
    return h("ul", { class: "share-devices", "aria-label": t("share.devices") }, devices.map(deviceRow));
  }

  function hostPanel() {
    const showPin = h("button", { type: "button", class: "ghost small", "aria-pressed": "false" }, t("share.pin_show"));
    const pinText = h("span", { class: "share-pin-value" }, "••••");
    showPin.addEventListener("click", () => {
      const shown = showPin.getAttribute("aria-pressed") !== "true";
      showPin.setAttribute("aria-pressed", String(shown));
      pinText.textContent = shown ? s.pin ?? "••••" : "••••";
    });
    const sound = h("select", { id: "share-sound" },
      (s.roster?.devices ?? []).map((d) => h("option", { value: d.id, selected: d.id === s.roster.sound }, d.label)));
    sound.addEventListener("change", () => s.link.send({ type: "sound", device: sound.value }));
    const stop = h("button", { type: "button", class: "danger solid", onClick: () => stopShare() }, t("share.stop"));
    return h("div", { class: "stack" },
      h("p", {}, t("share.hosting", { name: s.name })),
      s.pin ? h("p", { class: "row wrap" }, t("share.pin_is"), " ", pinText, showPin) : null,
      h("p", { class: "hint" }, t("share.join_how")),
      rosterList(),
      h("p", { class: "hint" }, t("share.order_rule")),
      SOUND_CHOICE ? field("share-sound", t("share.sound"), sound) : h("p", { class: "hint" }, t("share.sound_here_host")),
      stop);
  }

  function viewerPanel() {
    const views = h("fieldset", { class: "field share-views" }, h("legend", {}, t("share.view")),
      ["own", "host"].map((mode) => h("label", { class: "row" },
        h("input", { type: "radio", name: "share-view", value: mode, checked: s.view === mode, onChange: () => setView(mode) }),
        t(`share.view.${mode}`))));
    const leave = h("button", { type: "button", class: "danger solid", onClick: () => stopShare() }, t("share.leave"));
    return h("div", { class: "stack" },
      h("p", {}, t("share.joined", { name: s.roster?.name ?? s.name })),
      views, rosterList(),
      h("p", { class: "hint" }, t("share.watch_hint")),
      leave);
  }

  // ---- shared parts ---------------------------------------------------------

  function makeLink(hello, token, onMessage) {
    const link = new ShareLink({
      url: socketUrl(location), code: readCode(), hello,
      openSocket: (url) => new WebSocket(url), onMessage, onState: (state, detail) => linkState(state, detail),
    });
    link.token = token ?? null;
    return link;
  }

  function linkState(state, detail) {
    if (!s) return;
    if (state === "open") {
      keep({ role: s.role, token: s.link.token, name: s.name, view: s.view });
      s.me = detail.device;
      s.opened = true;
      s.limits = detail.limits ?? s.limits;
      if (s.role === "host") sendSnapshot(null);
      else if (!s.frame) buildStage();
      if (panel.open && s.formError) panel.close();
      s.formError = null;
    } else if (state === "ended") {
      const error = s.formError;
      const code = detail?.code;
      const reason = detail?.reason;
      const wasOpen = s.opened;
      stopShare(false);
      if (!wasOpen && error) {
        error.textContent = t(refusalKey(code), { seconds: Math.max(detail?.retryAfter ?? 0, 1) });
      } else {
        announce(t(endedKey(reason ?? (code === "ended" ? "expired" : null))));
      }
      return;
    }
    s.state = state;
    drawIndicator();
  }

  /** Keep the screen on while in a shared session (the presenter's device must not sleep). */
  function keepAwake(state) {
    const lock = new WakeLock();
    lock.request().catch?.(() => {});
    state.cleanup.push(() => { lock.release(); });
  }

  function announce(text) {
    const live = h("div", { class: "toast", role: "status" }, text);
    const host = document.getElementById("toasts") ?? document.body;
    // The toasts of the page are mirrored; this one says something about this device only.
    live.setAttribute("data-share-skip", "");
    host.append(live);
    setTimeout(() => live.remove(), 6000);
  }

  function onMessage(message) {
    if (!s) return;
    if (message.type === "roster") {
      s.roster = message;
      s.name = message.name;
      const me = message.devices.find((d) => d.id === s.me);
      s.sends = Boolean(me?.sends);
      s.hostOnline = Boolean(message.devices.find((d) => d.host)?.online);
      s.soundHere = message.sound === s.me;
      if (s.role === "host") {
        syncHostBlock();
        if (message.sound === s.me) s.bridge?.restore(); // the sound is back here
      }
      else syncViewerInert();
      if (!s.soundHere) s.player?.stop();
      drawIndicator();
      if (panel.open) renderPanel();
    } else if (message.type === "ended") {
      s.endReason = message.reason;
    } else if (s.role === "host") {
      hostMessage(message);
    } else if (message.type === "m") {
      viewerBody(message.body);
    }
  }

  function stopShare(tell = true) {
    if (!s) return;
    const was = s;
    s = null;
    keep(null);
    if (tell) was.link.stop();
    else was.link.stopped = true;
    for (const fn of was.cleanup) {
      try {
        fn();
      } catch {
        // keep cleaning
      }
    }
    pill.hidden = true;
    marker.hidden = true;
    root.classList.remove("share-hosting", "share-viewing", "share-watch");
    if (panel.open) renderPanel();
    if (was.role === "viewer") resume();
  }

  function drawIndicator() {
    if (!s) return;
    const status = statusKey({ state: s.state, host: s.role === "host", hostOnline: s.hostOnline !== false });
    const count = (s.roster?.devices ?? []).filter((d) => !d.host).length;
    const role = s.role === "host"
      ? t("share.pill.host", { name: s.name, n: count })
      : t("share.pill.viewer", { name: s.roster?.name ?? s.name });
    const mode = s.sends ? t("share.mode.sends") : t("share.mode.watch");
    const sentence = [role, mode, status ? t(status) : null].filter(Boolean).join(" · ");
    const target = s.role === "host" ? pill : s.bar;
    if (!target) return;
    const text = target.querySelector(".share-status");
    if (text && text.textContent !== sentence) text.textContent = sentence; // announced once per change
    target.classList.toggle("is-lost", Boolean(status));
    if (s.role === "viewer" && s.soundButton) s.soundButton.hidden = !s.soundHere || s.audio?.state === "running";
  }

  // ---- the sharing device ------------------------------------------------------

  function startHost({ name, pin, label }, error = null, token = null) {
    if (s) return;
    const source = new MirrorSource({ doc: document, scrub });
    s = { role: "host", name, pin, link: null, source, cleanup: [], formError: error, sends: true, roster: null, state: "connecting" };
    s.link = makeLink({ type: "host", name, pin, label }, token, onMessage);
    keepAwake(s);
    root.classList.add("share-hosting");
    const observer = new MutationObserver((records) => s?.source.record(records));
    observer.observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
    s.observer = observer;
    const rootObserver = new MutationObserver(() => { if (s) s.rootDirty = true; });
    rootObserver.observe(root, { attributes: true });
    const dark = globalThis.matchMedia?.("(prefers-color-scheme: dark)");
    const onScheme = () => { if (s) s.rootDirty = true; };
    dark?.addEventListener?.("change", onScheme);
    s.scrolled = new Map(); // id -> element scrolled at least once
    s.scrollDirty = new Set();
    const onScroll = (event) => {
      if (!s) return;
      const target = event.target;
      if (target === document || target === document.documentElement || target === document.body) s.scrollDirty.add("win");
      else if (target?.nodeType === 1 && !target.closest("[data-share-skip]")) {
        const id = source.idOf(target);
        if (id !== null) {
          s.scrolled.set(id, target);
          s.scrollDirty.add(id);
        }
      }
    };
    document.addEventListener("scroll", onScroll, true);
    const onFocus = () => { if (s) s.focusDirty = true; };
    document.addEventListener("focusin", onFocus, true);
    let lastPointer = 0;
    const onPointer = (event) => {
      if (!s?.sends || !event.isTrusted || event.target?.closest?.("[data-share-skip]")) return;
      const now = performance.now();
      if (event.type === "pointermove" && now - lastPointer < POINTER_MS) return;
      lastPointer = now;
      const at = pointerAt(event.target, event.clientX, event.clientY, (n) => source.idOf(n));
      if (at) sendBody({ k: "pt", who: s.me, down: event.type === "pointerdown" ? 1 : 0, ...at });
    };
    document.addEventListener("pointermove", onPointer, true);
    document.addEventListener("pointerdown", onPointer, true);
    // #284: a link to another site opens in a new tab while sharing, so this
    // page (and the shared session with it) stays.
    const onLinkClick = (event) => {
      if (!s || !event.isTrusted || event.defaultPrevented || event.button !== 0) return;
      const url = leavingUrl(event.target);
      if (!url) return;
      event.preventDefault();
      globalThis.open(url, "_blank", "noopener,noreferrer");
    };
    document.addEventListener("click", onLinkClick);
    const bridge = installSoundBridge({
      win: globalThis,
      remote: () => Boolean(s?.roster && s.roster.sound !== s.me),
      send: (body) => { if (s?.roster) s.link.send({ type: "m", to: s.roster.sound, body }); },
    });
    const timer = setInterval(hostTick, TICK_MS);
    s.bridge = bridge;
    s.cleanup.push(() => observer.disconnect(), () => rootObserver.disconnect(), () => dark?.removeEventListener?.("change", onScheme),
      () => document.removeEventListener("scroll", onScroll, true), () => document.removeEventListener("focusin", onFocus, true),
      () => document.removeEventListener("pointermove", onPointer, true), () => document.removeEventListener("pointerdown", onPointer, true),
      () => document.removeEventListener("click", onLinkClick), bridge.uninstall, () => clearInterval(timer), () => setHostBlock(false));
    fill(pill, h("button", { type: "button", class: "share-pill-button", onClick: () => openPanel() },
      icon("share"), h("span", { class: "share-status", role: "status" })));
    pill.hidden = false;
    drawIndicator();
    s.link.start();
  }

  function tooLarge(message, limit) {
    const text = JSON.stringify(message);
    if (text.length * 3 <= limit) return false;
    return new TextEncoder().encode(text).length > limit;
  }

  /**
   * Send a body of the sharing device. One larger than the relay takes is
   * never sent (the relay would close the link): a change is replaced by
   * the whole page, a whole page too large is skipped and tried again.
   */
  function sendBody(body, extra = {}) {
    if (!s) return false;
    const message = { type: "m", ...extra, body };
    if (tooLarge(message, (s.limits?.host_message_bytes ?? 4194304) - 1024)) {
      if (body.k !== "s") s.needFull = true;
      else s.retryFull = Date.now() + 5000;
      return false;
    }
    return s.link.send(message);
  }

  function viewport() {
    return { vw: innerWidth, vh: innerHeight };
  }

  function elementScrolls() {
    const out = [];
    for (const [id, el] of s.scrolled) {
      if (!el.isConnected) s.scrolled.delete(id);
      else out.push(elementScrollOf(el, id));
    }
    return out;
  }

  function sendSnapshot(to) {
    if (!s || s.link.state !== "open") return;
    s.snapshotting = true;
    hostTick(); // what changed so far goes first, to everyone up to date
    s.snapshotting = false;
    const snap = s.source.snapshot();
    const body = { k: "s", ...snap, root: rootState(), ...viewport(), sc: scrollOf(document, (n) => s.source.idOf(n)), es: elementScrolls(), f: s.source.idOf(document.activeElement) };
    s.lastViewport = `${body.vw}x${body.vh}`;
    sendBody(body, to ? { snap: true, to } : { snap: true });
  }

  function hostTick() {
    if (!s || s.role !== "host") return;
    s.source.record(s.observer.takeRecords()); // nothing pending is left out
    const delta = s.source.delta();
    if (s.link.state !== "open") return; // a snapshot follows the reconnection
    if (delta) sendBody(delta);
    if (!s.snapshotting && (s.needFull || (s.retryFull && Date.now() > s.retryFull)) && s.link.state === "open") {
      s.needFull = false;
      s.retryFull = 0;
      sendSnapshot(null);
      return;
    }
    if (s.rootDirty) {
      s.rootDirty = false;
      sendBody({ k: "r", ...rootState() });
    }
    const size = viewport();
    if (`${size.vw}x${size.vh}` !== s.lastViewport) {
      s.lastViewport = `${size.vw}x${size.vh}`;
      sendBody({ k: "v", ...size });
    }
    if (s.scrollDirty.size) {
      const body = { k: "sc" };
      if (s.scrollDirty.has("win")) body.w = scrollOf(document, (n) => s.source.idOf(n));
      const els = [...s.scrollDirty].filter((id) => id !== "win" && s.scrolled.get(id)?.isConnected).map((id) => elementScrollOf(s.scrolled.get(id), id));
      if (els.length) body.e = els;
      s.scrollDirty.clear();
      sendBody(body);
    }
    if (s.focusDirty) {
      s.focusDirty = false;
      sendBody({ k: "f", id: s.source.idOf(document.activeElement) });
    }
  }

  function hostMessage(message) {
    if (message.type === "need_snapshot") sendSnapshot(message.device);
    else if (message.type === "in") applyRemote(message.from, message.body ?? {});
  }

  function applyRemote(from, body) {
    // Numbers of another page load name other nodes: such an action is dropped.
    if (body.ep !== s.source.epoch) return;
    try {
      // Layout-dependent controls (the menu button) act as on the device that pressed them.
      actingAs(body.wide === undefined ? null : body.wide === 1, () => applyAction(from, body));
    } catch {
      // an action the page cannot take (a file field, a node gone): nothing happens
    }
  }

  function applyAction(from, body) {
    if (body.k === "pt") {
      placeHostMarker(body);
      sendBody({ ...body, who: from });
    } else if (body.k === "sc") {
      s.remoteScrollAt = performance.now();
      if (body.w) scrollTo(document, body.w, body.same === 1, (id) => s.source.nodeById(id));
      for (const entry of body.e ?? []) {
        const el = s.source.nodeById(entry[0]);
        if (el) elementScrollTo(el, entry, body.same === 1);
      }
    } else if (body.k === "click" && leavingUrl(s.source.nodeById(body.node))) {
      // #284: never leave for another site on a joined device's behalf (it opened the link itself).
    } else {
      s.source.applyInput(body, globalThis);
    }
  }

  function placeHostMarker(body) {
    const el = s.source.nodeById(body.node);
    if (!el?.getBoundingClientRect) return;
    const box = el.getBoundingClientRect();
    showMarker(marker, box.left + body.fx * box.width, box.top + body.fy * box.height, body.down);
  }

  // A sharing device set to only watch: its own touches and keys do nothing (a joined device acts).
  function blockLocal(event) {
    if (!event.isTrusted || event.target?.closest?.("[data-share-skip]")) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }

  function setHostBlock(on) {
    for (const type of BLOCKED) {
      if (on) window.addEventListener(type, blockLocal, { capture: true, passive: false });
      else window.removeEventListener(type, blockLocal, { capture: true });
    }
    root.classList.toggle("share-watch", on);
  }

  function syncHostBlock() {
    if (s.blocked === !s.sends) return;
    s.blocked = !s.sends;
    setHostBlock(s.blocked);
  }

  // ---- a joined device ----------------------------------------------------------

  function startViewer({ name, pin, label }, error = null, token = null, view = "own") {
    if (s) return;
    s = { role: "viewer", name, link: null, cleanup: [], formError: error, sends: false, roster: null, state: "connecting", view, hostOnline: true };
    s.link = makeLink({ type: "join", name, pin, label }, token, onMessage);
    keepAwake(s);
    s.link.start(); // the application stops here only once the join is accepted (buildStage)
  }

  function buildStage() {
    pause();
    root.classList.add("share-viewing");
    const status = h("span", { class: "share-status", role: "status" });
    const viewButton = h("button", { type: "button", class: "ghost small", onClick: () => setView(s.view === "own" ? "host" : "own") });
    const soundButton = h("button", { type: "button", class: "primary small", hidden: true, onClick: () => wakeSound() }, icon("play"), t("share.sound_here"));
    const bar = h("div", { class: "share-bar-top" },
      icon("share"), status, h("div", { class: "row wrap share-bar-actions" }, soundButton, viewButton,
        h("button", { type: "button", class: "ghost small", onClick: () => openPanel() }, t("share.panel")),
        h("button", { type: "button", class: "ghost small", onClick: () => stopShare() }, t("share.leave"))));
    const frame = h("iframe", { class: "share-frame", title: t("share.frame_title", { name: s.name }) });
    frame.setAttribute("sandbox", "allow-same-origin");
    const box = h("div", { class: "share-box" }, frame);
    const pointer = h("div", { class: "share-pointer", "aria-hidden": "true", hidden: true });
    const area = h("div", { class: "share-area" }, box, pointer);
    const stage = h("div", { class: "share-stage", "data-share-skip": true }, bar, area);
    document.body.append(stage);
    const doc = frame.contentDocument;
    const sheet = doc.createElement("link");
    sheet.rel = "stylesheet";
    sheet.href = new URL("style.css", location.href).href;
    doc.head.append(sheet);
    doc.documentElement.className = "mirror-doc";
    Object.assign(s, { bar, frame, box, area, pointer, doc, viewButton, soundButton, view: s.view ?? "own" });
    // A field being typed in here keeps what was typed: the sharing device's
    // echo of an older value never overwrites it (HOLD_MS after the last key).
    s.typing = new Map(); // node -> when it was last typed in here
    s.mirror = new MirrorView({ doc, hold: (id) => performance.now() - (s?.typing?.get(id) ?? -1e9) < HOLD_MS });
    s.player = new SoundPlayer({
      context: () => {
        s.audio ??= new (globalThis.AudioContext ?? globalThis.webkitAudioContext)();
        return s.audio;
      },
      prepareMedia: () => api.audioCookie().catch(() => null),
    });
    wireFrame(doc);
    const onResize = () => layout();
    window.addEventListener("resize", onResize);
    s.cleanup.push(() => stage.remove(), () => window.removeEventListener("resize", onResize), () => s.player?.stop(), () => s.audio?.close?.());
    setView(s.view);
    syncViewerInert();
    drawIndicator();
  }

  function wakeSound() {
    s.audio ??= new (globalThis.AudioContext ?? globalThis.webkitAudioContext)();
    s.audio.resume?.().then(() => drawIndicator(), () => {});
  }

  function setView(mode) {
    if (!s || s.role !== "viewer") return;
    s.view = mode;
    keep({ role: "viewer", token: s.link.token, name: s.name, view: mode });
    if (s.viewButton) s.viewButton.textContent = t(mode === "own" ? "share.view.to_host" : "share.view.to_own");
    layout();
    if (panel.open) renderPanel();
  }

  function layout() {
    if (!s?.frame) return;
    const { frame, box, area } = s;
    const width = area.clientWidth;
    const height = area.clientHeight;
    if (s.view === "host" && s.vw && s.vh) {
      const scale = Math.min(width / s.vw, height / s.vh, 4);
      s.scale = scale;
      frame.style.width = `${s.vw}px`;
      frame.style.height = `${s.vh}px`;
      frame.style.transform = `scale(${scale})`;
      box.style.width = `${Math.floor(s.vw * scale)}px`;
      box.style.height = `${Math.floor(s.vh * scale)}px`;
    } else {
      s.scale = 1;
      frame.style.width = "100%";
      frame.style.height = "100%";
      frame.style.transform = "";
      box.style.width = "100%";
      box.style.height = "100%";
    }
    if (s.lastScroll) applyScroll(s.lastScroll);
  }

  function syncViewerInert() {
    if (!s?.frame) return;
    const watching = !s.sends;
    s.frame.inert = watching;
    s.frame.tabIndex = watching ? -1 : 0;
    root.classList.toggle("share-watch", watching);
  }

  function resync() {
    const now = performance.now();
    if (s.resyncAt && now - s.resyncAt < RESYNC_MS) return;
    s.resyncAt = now;
    s.link.send({ type: "resync" });
  }

  function viewerBody(body) {
    if (!s?.mirror || !body) return;
    try {
      switch (body.k) {
        case "s":
          s.epoch = body.e ?? null;
          s.mirror.apply(body);
          setRoot(body);
          setViewport(body);
          applyScroll({ w: body.sc, e: body.es });
          setFocus(body.f);
          break;
        case "d":
          s.mirror.apply(body);
          if (Number.isInteger(body.n) && body.n !== s.mirror.count) resync();
          if (s.focusId) setFocus(s.focusId);
          break;
        case "r": setRoot(body); break;
        case "v": setViewport(body); break;
        case "sc": applyScroll(body); break;
        case "f": setFocus(body.id); break;
        case "pt": if (body.who !== s.me) placeViewerMarker(body); break;
        default: if (s.soundHere) s.player.handle(body);
      }
    } catch (error) {
      if (error instanceof RangeError) resync();
    }
  }

  function setRoot({ root: state, lang, theme, cls }) {
    const value = state ?? { lang, theme, cls };
    const html = s.doc.documentElement;
    html.lang = value.lang ?? "";
    html.dataset.theme = value.theme ?? "light";
    html.className = [...(value.cls ?? []), "mirror-doc"].join(" ");
  }

  function setViewport({ vw, vh }) {
    s.vw = vw;
    s.vh = vh;
    layout();
  }

  function applyScroll(body) {
    s.lastScroll = { w: body.w ?? s.lastScroll?.w, e: body.e ?? s.lastScroll?.e };
    if (performance.now() - (s.localScrollAt ?? -1e9) < LOCAL_SCROLL_MS) return;
    s.applyingScroll = performance.now();
    const same = s.view === "host";
    if (body.w) scrollTo(s.doc, body.w, same, (id) => s.mirror.nodeById(id));
    for (const entry of body.e ?? []) {
      const el = s.mirror.nodeById(entry[0]);
      if (el) elementScrollTo(el, entry, same);
    }
  }

  function setFocus(id) {
    s.doc.querySelector(".share-focus")?.classList.remove("share-focus");
    s.focusId = id ?? null;
    const el = id ? s.mirror.nodeById(id) : null;
    if (el && el !== s.doc.body) el.classList?.add("share-focus");
  }

  function placeViewerMarker(body) {
    const el = s.mirror.nodeById(body.node);
    if (!el?.getBoundingClientRect) return;
    const box = el.getBoundingClientRect();
    const frame = s.frame.getBoundingClientRect();
    const area = s.area.getBoundingClientRect();
    const scale = s.scale ?? 1;
    const x = frame.left - area.left + (box.left + body.fx * box.width) * scale;
    const y = frame.top - area.top + (box.top + body.fy * box.height) * scale;
    showMarker(s.pointer, x, y, body.down);
  }

  function send(body) {
    if (!s?.sends || !s.epoch) return;
    // The layout of this device's view of the page (its own, or the sharing device's).
    const view = s.doc?.defaultView;
    const wide = view?.matchMedia ? (view.matchMedia(WIDE_QUERY).matches ? 1 : 0) : undefined;
    const message = { type: "in", body: { ...body, ep: s.epoch, wide } };
    if (tooLarge(message, (s.limits?.viewer_message_bytes ?? 16384) - 256)) return;
    s.link.send(message);
  }

  function wireFrame(doc) {
    const capture = { capture: true, passive: false };
    const idOf = (node) => s?.mirror.idOf(node) ?? null;
    let lastPointer = 0;
    let lastScroll = 0;
    const on = (type, fn) => {
      doc.addEventListener(type, fn, capture);
      s.cleanup.push(() => doc.removeEventListener(type, fn, capture));
    };
    on("click", (event) => {
      event.preventDefault(); // nothing happens here: the sharing device does it, and its page comes back
      if (!s?.sends) return;
      const target = event.target?.closest?.(ACTIONABLE) ?? event.target;
      // #284: a link to another site opens here, in a new tab; the sharing device stays where it is.
      const away = leavingUrl(event.target);
      if (away) {
        globalThis.open(away, "_blank", "noopener,noreferrer");
        return;
      }
      const node = idOf(target);
      if (node !== null) send({ k: "click", node });
    });
    for (const type of ["submit", "cancel", "dragstart", "contextmenu"]) on(type, (event) => event.preventDefault());
    const value = (event) => {
      const el = event.target;
      const node = idOf(el);
      if (node === null || !s?.sends) return;
      const type = String(el.type ?? "").toLowerCase();
      if (type === "checkbox" || type === "radio") return; // the click carries it
      // Each field has its own hold: typing in two fields in a row loses nothing.
      s.typing.set(node, performance.now());
      s.typingTimers ??= new Map();
      clearTimeout(s.typingTimers.get(node));
      s.typingTimers.set(node, setTimeout(() => {
        if (!s?.mirror || performance.now() - (s.typing.get(node) ?? 0) < HOLD_MS) return;
        s.typing.delete(node);
        s.typingTimers.delete(node);
        s.mirror.release(node);
      }, HOLD_MS + 50));
      send({ k: "val", node, v: String(el.value ?? "").slice(0, MAX_VALUE) });
    };
    on("input", value);
    on("change", value);
    on("keydown", (event) => {
      if (!s?.sends) {
        event.preventDefault();
        return;
      }
      if (event.ctrlKey || event.metaKey || event.key === "Escape") event.preventDefault();
      send({
        k: "key", node: idOf(doc.activeElement), key: event.key, code: event.code,
        ctrl: event.ctrlKey ? 1 : 0, meta: event.metaKey ? 1 : 0, shift: event.shiftKey ? 1 : 0, alt: event.altKey ? 1 : 0,
      });
    });
    on("focusin", (event) => {
      const node = idOf(event.target);
      if (node !== null && s?.sends) send({ k: "focus", node });
    });
    on("scroll", (event) => {
      if (performance.now() - (s?.applyingScroll ?? -1e9) < 120) return; // our own scrolling
      s.localScrollAt = performance.now();
      if (!s?.sends || performance.now() - lastScroll < TICK_MS) return;
      lastScroll = performance.now();
      const target = event.target;
      const body = { k: "sc", same: s.view === "host" ? 1 : 0 };
      if (target === doc || target === doc.documentElement || target === doc.body) body.w = scrollOf(doc, idOf);
      else if (idOf(target) !== null) body.e = [elementScrollOf(target, idOf(target))];
      send(body);
    });
    const pointer = (event) => {
      if (!s?.sends) return;
      const now = performance.now();
      if (event.type === "pointermove" && now - lastPointer < POINTER_MS) return;
      lastPointer = now;
      const at = pointerAt(event.target, event.clientX, event.clientY, idOf);
      if (at) send({ k: "pt", down: event.type === "pointerdown" ? 1 : 0, ...at });
    };
    on("pointermove", pointer);
    on("pointerdown", pointer);
  }

  // ---- the pointer marker -------------------------------------------------------

  function showMarker(el, x, y, down) {
    el.hidden = false;
    el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    el.classList.toggle("is-down", Boolean(down));
    clearTimeout(el.hideTimer);
    el.hideTimer = setTimeout(() => { el.hidden = true; }, POINTER_HIDE_MS);
  }

  // ---- start ----------------------------------------------------------------------

  const button = document.getElementById("share-open");
  button?.addEventListener("click", () => openPanel());
  if (button) button.title = t("share.open");

  // A reload within the resume window comes back without the password.
  const again = kept();
  if (again && readCode()) {
    if (again.role === "host") startHost({ name: again.name, pin: null, label: null }, null, again.token);
    else startViewer({ name: again.name, pin: null, label: null }, null, again.token, again.view === "host" ? "host" : "own");
  }

  return {
    /** Whether this device is a joined one (its application is paused). */
    viewing: () => s?.role === "viewer" && Boolean(s.frame),
    /** Redraw the panel's and the indicator's texts (the language changed). */
    relabel: () => {
      if (button) button.title = t("share.open");
      drawIndicator();
      if (panel.open) renderPanel();
    },
  };
}
