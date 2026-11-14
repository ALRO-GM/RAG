// Sessions list: state and progress, actions, upload of a file.

import { h, keepFocus } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime, sessionIdFor } from "../lib/text.js";
import { nextPollMs } from "../lib/net.js";
import { actionsFor, progressPercent, sortSessions, stageLabel } from "../lib/sessions.js";
import { noticeSection } from "../consent.js";

export async function sessionsView(ctx, root) {
  let sessions = (await ctx.api.listSessions()).sessions;
  let unreadable = 0;
  let quiet = 0;
  let timer = null;
  let stopped = false;
  let signature = "";
  let offline = false;

  const message = h("p", { class: "notice", role: "status", hidden: true });
  const list = h("ul", { class: "cards" });

  function say(text, isError = false) {
    message.hidden = !text;
    message.textContent = text ?? "";
    message.classList.toggle("error", isError);
  }

  function describe(s) {
    const pct = progressPercent(s);
    return h("li", { class: `card state-${s.state}` },
      h("h2", {}, s.title || s.id),
      h("p", { class: "meta" },
        h("span", { class: "badge" }, t(`state.${s.state}`)),
        s.started_at ? ` ${new Date(s.started_at).toLocaleString(lang)}` : "",
        s.duration > 0 ? ` · ${formatTime(s.duration)}` : ""),
      s.state === "running" || s.state === "queued"
        ? h("div", { class: "progress" },
          h("progress", { max: 100, value: pct ?? undefined, "aria-label": t("sessions.progress") }),
          h("span", {}, [pct !== null ? `${pct}%` : t("state.queued"), s.stage ? ` · ${stageLabel(t, s.stage)}` : "", s.message ? ` · ${s.message}` : ""].join("")))
        : null,
      s.state === "failed" && s.failure ? h("p", { class: "error" }, s.failure) : null,
      s.message && s.state !== "running" && s.state !== "queued" ? h("p", { class: "hint" }, s.message) : null,
      h("div", { class: "row" }, actionsFor(s).map((a) => button(s, a))));
  }

  function button(s, action) {
    const handlers = {
      open: () => ctx.navigate(`#/s/${encodeURIComponent(s.id)}`),
      resume: () => ctx.navigate(`#/record/${encodeURIComponent(s.id)}`),
      finish: () => run(() => ctx.api.closeSession(s.id)),
      transcribe: () => {
        if (s.state === "done" && !confirm(t("sessions.requeue_confirm"))) return;
        run(() => ctx.api.queueTranscription(s.id, {})); // options kept on the session
      },
      delete: () => {
        if (confirm(t("sessions.delete_confirm", { name: s.title || s.id }))) run(() => ctx.api.deleteSession(s.id));
      },
    };
    return h("button", {
      type: "button", class: action === "open" ? "primary" : action === "delete" ? "danger" : "",
      "data-focus-key": `${s.id}:${action}`, onClick: handlers[action],
    }, t(`action.${action}`));
  }

  async function run(call) {
    say(null);
    try {
      await call();
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error, refresh);
      // The session moved on meanwhile (a job started, another device
      // acted): not a failure, the refreshed list shows where it is.
      if (error.code === "wrong_state") say(t("sessions.changed"));
      else say(describeError(error), true);
    }
    await refresh();
  }

  function render() {
    const sig = JSON.stringify(sessions.map((s) => [s.id, s.state, s.stage, s.fraction, s.message, s.title, s.duration, s.failure]));
    if (sig === signature) return;
    signature = sig;
    keepFocus(root, () => {
      list.replaceChildren(...(sessions.length
        ? sortSessions(sessions).map(describe)
        : [h("li", { class: "hint" }, t("sessions.empty"))]));
    });
    if (unreadable) say(t("sessions.unreadable", { n: unreadable }));
  }

  async function refresh() {
    clearTimeout(timer);
    try {
      const data = await ctx.api.listSessions();
      sessions = data.sessions;
      unreadable = data.unreadable ?? 0;
      if (offline) say(null);
      offline = false;
      quiet = sessions.some((s) => s.state === "queued" || s.state === "running") ? 0 : quiet + 1;
      render();
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error, refresh);
      say(describeError(error), true);
      offline = true;
      quiet = Math.max(quiet, 1);
    }
    schedule();
  }

  function schedule() {
    clearTimeout(timer);
    if (stopped) return;
    timer = setTimeout(() => {
      if (document.visibilityState === "visible") refresh();
      else schedule();
    }, nextPollMs(sessions, quiet));
  }

  const visible = () => { if (document.visibilityState === "visible" && !stopped) refresh(); };
  document.addEventListener("visibilitychange", visible);

  root.append(
    h("h1", {}, t("sessions.title")),
    h("div", { class: "row" },
      h("a", { class: "button primary big", href: "#/record" }, t("sessions.new")),
      h("a", { class: "button big", href: "#/enrol" }, t("sessions.enrol"))),
    message, list, uploadForm());
  render();
  schedule();

  function uploadForm() {
    const file = h("input", { id: "up-file", type: "file", accept: "audio/*,video/mp4,.m4a,.mp3,.wav,.ogg,.opus,.flac,.aac,.mp4,.webm" });
    const title = h("input", { id: "up-title", type: "text", maxLength: 200, autocomplete: "off" });
    const language = h("select", { id: "up-lang" }, h("option", { value: "es" }, "Español"), h("option", { value: "en" }, "English"));
    language.value = lang;
    const ok = h("input", { id: "up-ok", type: "checkbox" });
    const go = h("button", { type: "submit", class: "primary", disabled: true }, t("upload.send"));
    const status = h("p", { role: "status", class: "hint" });
    const notice = h("div", { class: "stack" }, h("p", { class: "hint" }, t("consent.loading")));
    let shown = null; // the recording notice shown, once loaded
    const sync = () => { go.disabled = !(file.files?.length && ok.checked && shown); };
    async function loadNotice() {
      if (shown) return;
      try {
        shown = await ctx.api.consent("recording", lang);
        notice.replaceChildren(...noticeSection(shown, ok, "up"),
          h("p", { class: "hint" }, t("upload.consent")));
      } catch (error) {
        notice.replaceChildren(h("p", { class: "error" }, describeError(error)));
        if (error.kind === "unauthorized") ctx.fail(error, refresh);
      }
      sync();
    }
    file.addEventListener("change", sync);
    ok.addEventListener("change", sync);
    const form = h("form", { class: "stack", onSubmit: async (event) => {
      event.preventDefault();
      const chosen = file.files[0];
      go.disabled = true;
      status.textContent = t("upload.sending");
      try {
        await ctx.api.uploadRecording(sessionIdFor(new Date(), title.value || chosen.name.replace(/\.[^.]+$/, "")), chosen, language.value, title.value.trim(), shown.version);
        status.textContent = t("upload.done");
        form.reset();
        await refresh();
      } catch (error) {
        status.textContent = describeError(error);
        if (error.kind === "unauthorized") ctx.fail(error, refresh);
      }
      sync();
    } },
    h("div", { class: "field" }, h("label", { for: "up-file" }, t("upload.file")), file),
    h("div", { class: "field" }, h("label", { for: "up-title" }, t("record.session_title")), title),
    h("div", { class: "field" }, h("label", { for: "up-lang" }, t("record.language")), language),
    notice,
    go, status);
    return h("details", { class: "upload", onToggle: (event) => { if (event.target.open) loadNotice(); } },
      h("summary", {}, t("upload.title")), form);
  }

  return () => {
    stopped = true;
    clearTimeout(timer);
    document.removeEventListener("visibilitychange", visible);
  };
}
