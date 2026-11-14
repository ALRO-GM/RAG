// Recording page: consent, who is present, record, upload in numbered chunks.

import { h } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { Capture, CaptureError } from "../capture.js";
import { WakeLock, setLeaveGuard } from "../device.js";
import { ChunkQueue } from "../lib/chunks.js";
import { ChunkUploader } from "../lib/uploader.js";
import { formatTime, sessionIdFor } from "../lib/text.js";
import { codeFromInput } from "../lib/net.js";
import { transcriptionBody } from "../lib/sessions.js";
import { noticeSection } from "../consent.js";

export async function recordView(ctx, root, params) {
  const resumeId = params[0] ?? null;
  // The notice comes from the API: its text and version are the ones the
  // session records (consent_version).
  const [enrolled, existing, notice] = await Promise.all([
    ctx.api.speakers(),
    resumeId ? ctx.api.getSession(resumeId) : Promise.resolve(null),
    ctx.api.consent("recording", lang),
  ]);
  if (existing && existing.state !== "recording") {
    root.append(h("p", { class: "notice" }, t("record.not_recording")), h("a", { class: "button", href: "#/" }, t("nav.sessions")));
    return null;
  }

  let capture = null;
  let uploader = null;
  let phase = "form"; // form | starting | recording | finishing | done
  let sessionId = existing?.id ?? null;
  let timer = null;
  const wake = new WakeLock((on) => {
    wakeNote.hidden = on;
  });

  // ---- form ---------------------------------------------------------
  const title = h("input", { id: "rec-title", type: "text", maxLength: 200, autocomplete: "off", value: existing?.title ?? "" });
  const language = h("select", { id: "rec-lang" },
    h("option", { value: "es" }, "Español"), h("option", { value: "en" }, "English"),
    h("option", { value: "auto" }, t("record.language_auto")));
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
        h("p", { class: "hint" }, t("record.who_hint")),
        people.length
          ? people.map((p) => h("label", { class: "check" }, p.box, h("span", {}, p.name)))
          : h("p", { class: "hint" }, t("record.nobody_enrolled")),
        h("a", { href: "#/enrol" }, t("record.enrol_link"))),
      h("div", { class: "field" }, h("label", { for: "rec-count" }, t("record.count")), count,
        h("p", { class: "hint" }, t("record.count_hint"))),
    ],
    noticeSection(notice, consent, "rec"),
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
  // A new access code (the service restarted without --keep-code): the
  // uploads wait, capture goes on, the person pastes the new link here.
  const codeInput = h("input", { id: "rec-code", type: "text", autocomplete: "off", spellcheck: false });
  const codeStatus = h("p", { class: "hint", role: "status" });
  const codeNote = h("p", { class: "warn", role: "alert" });
  const codeButton = h("button", { type: "submit", class: "primary" }, t("record.reauth_use"));
  let afterCode = null; // what to do once a code works
  const codeForm = h("form", { class: "stack reauth", hidden: true, onSubmit: (event) => {
    event.preventDefault();
    tryCode(codeFromInput(codeInput.value));
  } },
  codeNote,
  h("div", { class: "field" }, h("label", { for: "rec-code" }, t("record.reauth_label")), codeInput),
  codeButton, codeStatus);
  const live = h("section", { class: "stack", hidden: true },
    h("h1", {}, title.value || t("record.recording")),
    clock, meter, statusLine, warning, codeForm, wakeNote,
    h("div", { class: "row" }, pause, stop, retry));

  root.append(form, live);

  setLeaveGuard(() => phase === "recording" || phase === "finishing" || Boolean(uploader && !uploader.state().drained));

  function showWarning(text) {
    warning.hidden = !text;
    warning.textContent = text ?? "";
  }

  function askForCode(state, then) {
    afterCode = then;
    const mb = (state.pendingBytes / 1e6).toFixed(1);
    codeNote.textContent = t("record.paused_auth", { seconds: Math.round(state.pendingSeconds), mb });
    if (codeForm.hidden) {
      codeForm.hidden = false;
      codeStatus.textContent = "";
      codeInput.focus();
    }
  }

  /** Use a pasted code, if it looks like one. */
  function tryCode(code) {
    if (!code) {
      codeStatus.textContent = t("record.reauth_invalid");
      return;
    }
    ctx.setCode(code);
    verify();
  }

  /** Check the code in use with a cheap authenticated call, then go on. */
  async function verify() {
    codeButton.disabled = true;
    codeStatus.textContent = t("record.reauth_checking");
    let view;
    try {
      view = await ctx.api.getSession(sessionId);
    } catch (error) {
      codeButton.disabled = false;
      codeStatus.textContent = error?.kind === "unauthorized" ? t("record.reauth_wrong") : describeError(error);
      return;
    }
    codeButton.disabled = false;
    codeForm.hidden = true;
    codeInput.value = "";
    const then = afterCode;
    afterCode = null;
    then?.(view);
  }

  // The new link opened in this tab: main.js kept the code without a reload.
  const stopListening = ctx.onNewCode?.(() => {
    if (!codeForm.hidden) verify();
  });

  function renderUpload(state) {
    if (state.paused === "unauthorized") askForCode(state, (view) => uploader.resume(view.chunks));
    if (state.full && phase === "recording") {
      showWarning(t("record.full", { seconds: Math.round(state.pendingSeconds) }));
      finish();
      return;
    }
    const parts = [];
    if (state.pendingSeconds >= 1) parts.push(t("record.pending", { seconds: Math.round(state.pendingSeconds) }));
    else if (phase === "recording") parts.push(t("record.sent_all"));
    if (state.lastError && !state.paused) parts.push(t(state.lastError === "unavailable" ? "record.retrying" : "record.retrying_other"));
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
        // The options are kept on the session, so closing it from any
        // device queues the transcription with them.
        const options = transcriptionBody(language.value, people.filter((p) => p.box.checked).map((p) => p.id), count.value);
        await ctx.api.createSession(sessionId, language.value, title.value.trim(), {
          consent_version: notice.version,
          num_speakers: options.num_speakers,
          expected_speakers: options.expected_speakers,
        });
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

  /** Queue the transcription; already queued (a lost answer) is fine. */
  async function queueOnce() {
    try {
      await ctx.api.queueTranscription(sessionId, {}); // the session's own options
    } catch (error) {
      if (error?.code !== "wrong_state") throw error;
      const view = await ctx.api.getSession(sessionId);
      if (!["queued", "running", "done"].includes(view.state)) throw error;
    }
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
      // Closing again is harmless: a retry after a lost answer goes on.
      await ctx.api.closeSession(sessionId);
      await queueOnce();
    } catch (error) {
      if (error?.kind === "unauthorized") {
        askForCode(uploader.state(), () => finalise());
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
    stopListening?.();
    clearInterval(timer);
    window.removeEventListener("online", onOnline);
    if (phase === "recording" || phase === "starting") capture?.stop();
    wake.release();
    setLeaveGuard(() => false);
  };
}
