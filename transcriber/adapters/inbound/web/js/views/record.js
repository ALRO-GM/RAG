// Recording page: consent, who is present, record, upload in numbered chunks.

import { h } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { Capture, CaptureError } from "../capture.js";
import { WakeLock, setLeaveGuard } from "../device.js";
import { ChunkQueue } from "../lib/chunks.js";
import { ChunkUploader } from "../lib/uploader.js";
import { formatTime, sessionIdFor } from "../lib/text.js";
import { transcriptionBody } from "../lib/sessions.js";

const OPTIONS_KEY = "transcriber.options.";

/** Options chosen at the start of a recording, kept to queue the transcription later. */
export function savedOptions(sessionId) {
  try {
    return JSON.parse(sessionStorage.getItem(OPTIONS_KEY + sessionId) ?? "null");
  } catch {
    return null;
  }
}

function saveOptions(sessionId, options) {
  try {
    sessionStorage.setItem(OPTIONS_KEY + sessionId, JSON.stringify(options));
  } catch {
    // private mode: the transcription is then queued without them
  }
}

export async function recordView(ctx, root, params) {
  const resumeId = params[0] ?? null;
  const [enrolled, existing] = await Promise.all([
    ctx.api.speakers(),
    resumeId ? ctx.api.getSession(resumeId) : Promise.resolve(null),
  ]);
  if (existing && existing.state !== "recording") {
    root.append(h("p", { class: "notice" }, t("record.not_recording")), h("a", { class: "button", href: "#/" }, t("nav.sessions")));
    return null;
  }

  let capture = null;
  let uploader = null;
  let phase = "form"; // form | starting | recording | finishing | done
  let sessionId = existing?.id ?? null;
  let options = null;
  let timer = null;
  const wake = new WakeLock((on) => {
    wakeNote.hidden = on;
  });

  // ---- form ---------------------------------------------------------
  const title = h("input", { id: "rec-title", type: "text", maxLength: 200, autocomplete: "off", value: existing?.title ?? "" });
  const language = h("select", { id: "rec-lang" },
    h("option", { value: "es" }, "Español"), h("option", { value: "en" }, "English"));
  language.value = existing?.language ?? lang;
  const people = enrolled.speakers.map((p) => ({
    id: p.id,
    box: h("input", { type: "checkbox", value: p.id, id: `who-${p.id}` }),
    name: p.name || p.id,
  }));
  const count = h("input", { id: "rec-count", type: "number", min: 1, max: 20, inputMode: "numeric" });
  const consent = h("input", { id: "rec-consent", type: "checkbox" });
  const start = h("button", { class: "primary big", type: "button", disabled: true, onClick: () => begin() }, t(existing ? "record.resume" : "record.start"));
  consent.addEventListener("change", () => { start.disabled = !consent.checked; });
  const formError = h("p", { class: "error", role: "alert", hidden: true });

  const form = h("section", { class: "stack" },
    h("h1", {}, t(existing ? "record.resume_title" : "record.title")),
    existing ? null : [
      h("div", { class: "field" }, h("label", { for: "rec-title" }, t("record.session_title")), title),
      h("div", { class: "field" }, h("label", { for: "rec-lang" }, t("record.language")), language),
      h("fieldset", { class: "field" },
        h("legend", {}, t("record.who")),
        people.length
          ? people.map((p) => h("label", { class: "check" }, p.box, h("span", {}, p.name)))
          : h("p", { class: "hint" }, t("record.nobody_enrolled")),
        h("a", { href: "#/enrol" }, t("record.enrol_link"))),
      h("div", { class: "field" }, h("label", { for: "rec-count" }, t("record.count")), count,
        h("p", { class: "hint" }, t("record.count_hint"))),
    ],
    h("section", { class: "consent", "aria-labelledby": "consent-h" },
      h("h2", { id: "consent-h" }, t("consent.title")),
      h("ul", {}, ["consent.what", "consent.where", "consent.delete"].map((k) => h("li", {}, t(k))))),
    h("label", { class: "check strong" }, consent, h("span", {}, t("consent.accept"))),
    formError,
    start);

  // ---- recording ----------------------------------------------------
  const clock = h("p", { class: "clock", role: "timer", "aria-live": "off" }, "0:00");
  const meter = h("meter", { min: 0, max: 1, value: 0, "aria-label": t("record.level") });
  const statusLine = h("p", { class: "status", role: "status" });
  const warning = h("p", { class: "warn", role: "alert", hidden: true });
  const wakeNote = h("p", { class: "hint", hidden: true }, t("record.wake_off"));
  const pause = h("button", { type: "button", class: "big", onClick: () => togglePause() }, t("record.pause"));
  const stop = h("button", { type: "button", class: "primary big", onClick: () => finish() }, t("record.stop"));
  const retry = h("button", { type: "button", class: "big", hidden: true, onClick: () => finalise() }, t("action.retry"));
  const live = h("section", { class: "stack", hidden: true },
    h("h1", {}, title.value || t("record.recording")),
    clock, meter, statusLine, warning, wakeNote,
    h("div", { class: "row" }, pause, stop, retry));

  root.append(form, live);

  setLeaveGuard(() => phase === "recording" || phase === "finishing" || Boolean(uploader && !uploader.state().drained));

  function showWarning(text) {
    warning.hidden = !text;
    warning.textContent = text ?? "";
  }

  function renderUpload(state) {
    const parts = [];
    if (state.pendingSeconds >= 1) parts.push(t("record.pending", { seconds: Math.round(state.pendingSeconds) }));
    else if (phase === "recording") parts.push(t("record.sent_all"));
    if (state.lastError) parts.push(t(state.lastError === "unavailable" ? "record.retrying" : "record.retrying_other"));
    statusLine.textContent = parts.join(" · ");
    if (state.fatal) showWarning(t(`record.fatal.${state.fatal}`));
    else if (state.large) showWarning(t("record.large_backlog", { seconds: Math.round(state.pendingSeconds) }));
    else if (phase === "recording") showWarning(null);
  }

  async function begin() {
    if (phase !== "form") return;
    phase = "starting";
    start.disabled = true;
    formError.hidden = true;
    const early = [];
    capture = new Capture({
      onPcm: (bytes) => (uploader ? uploader.push(bytes) : early.push(bytes)),
      onLevel: (v) => { meter.value = Math.max(v, meter.value * 0.85); },
      onProblem: (kind) => showWarning(t(kind === "ended" ? "record.mic_lost" : "record.mic_suspended")),
    });
    try {
      await capture.start();
    } catch (error) {
      phase = "form";
      start.disabled = !consent.checked;
      formError.hidden = false;
      formError.textContent = t(error instanceof CaptureError ? `mic.${error.reason}` : "mic.failed");
      return;
    }
    try {
      if (!existing) {
        sessionId = sessionIdFor(new Date(), title.value);
        await ctx.api.createSession(sessionId, language.value, title.value.trim());
        options = transcriptionBody(language.value, people.filter((p) => p.box.checked).map((p) => p.id), count.value);
        saveOptions(sessionId, options);
      }
    } catch (error) {
      await capture.stop();
      phase = "form";
      start.disabled = !consent.checked;
      formError.hidden = false;
      formError.textContent = describeError(error);
      return;
    }
    const queue = new ChunkQueue({ firstNumber: existing?.chunks ?? 0 });
    uploader = new ChunkUploader(ctx.api, sessionId, queue, { onState: renderUpload });
    for (const bytes of early) uploader.push(bytes);
    window.addEventListener("online", onOnline);
    wake.request();
    phase = "recording";
    form.hidden = true;
    live.hidden = false;
    live.querySelector("h1").textContent = title.value || sessionId;
    timer = setInterval(() => {
      clock.textContent = formatTime(capture.seconds);
    }, 500);
    stop.focus();
  }

  function onOnline() {
    uploader?.retryNow();
  }

  function togglePause() {
    capture.paused = !capture.paused;
    pause.textContent = t(capture.paused ? "record.resume_capture" : "record.pause");
    pause.setAttribute("aria-pressed", String(capture.paused));
    live.classList.toggle("paused", capture.paused);
  }

  async function finish() {
    if (phase !== "recording") return;
    phase = "finishing";
    pause.disabled = true;
    stop.disabled = true;
    clearInterval(timer);
    await capture.stop();
    clock.textContent = formatTime(capture.seconds);
    uploader.finish();
    await finalise();
  }

  async function finalise() {
    retry.hidden = true;
    statusLine.textContent = t("record.finishing");
    const fatal = await uploader.whenDone();
    if (fatal) {
      showWarning(t(`record.fatal.${fatal}`));
      return;
    }
    try {
      await ctx.api.closeSession(sessionId);
      const saved = options ?? savedOptions(sessionId);
      await ctx.api.queueTranscription(sessionId, saved ?? {});
    } catch (error) {
      if (error?.kind === "unauthorized") {
        ctx.fail(error, () => finalise());
        return;
      }
      showWarning(describeError(error));
      retry.hidden = false;
      return;
    }
    phase = "done";
    await wake.release();
    window.removeEventListener("online", onOnline);
    setLeaveGuard(() => false);
    ctx.navigate("#/");
  }

  return () => {
    clearInterval(timer);
    window.removeEventListener("online", onOnline);
    if (phase === "recording" || phase === "starting") capture?.stop();
    wake.release();
    setLeaveGuard(() => false);
  };
}
