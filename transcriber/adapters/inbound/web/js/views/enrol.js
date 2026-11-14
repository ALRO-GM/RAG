// Enrolment: name, consent, read a text for 30 s, send the sample.

import { h } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { Capture, CaptureError } from "../capture.js";
import { WakeLock, setLeaveGuard } from "../device.js";
import { concatBytes } from "../lib/audio.js";
import { suggestId, uniqueId, idProblem, countdown } from "../lib/text.js";
import { noticeSection } from "../consent.js";

export const GOAL_SECONDS = 30;
const MAX_SECONDS = 90;

export async function enrolView(ctx, root) {
  // The consent notice comes from the API, in force and versioned.
  const [known, guide] = await Promise.all([ctx.api.speakers(), ctx.api.enrolmentText(lang)]);
  const taken = known.speakers.map((s) => s.id);
  let capture = null;
  let timer = null;
  let retryTimer = null;
  let phase = "form"; // form | reading | recording | sending
  let idEdited = false;
  const parts = [];
  const wake = new WakeLock();

  const name = h("input", { id: "en-name", type: "text", maxLength: 100, autocomplete: "name", required: true });
  const id = h("input", { id: "en-id", type: "text", maxLength: 64, autocapitalize: "none", spellcheck: false, autocomplete: "off" });
  const idHint = h("p", { class: "hint", role: "status" }, t("enrol.id_hint"));
  const consent = h("input", { id: "en-consent", type: "checkbox" });
  const next = h("button", { class: "primary big", type: "button", disabled: true, onClick: () => toReading() }, t("enrol.continue"));
  const formError = h("p", { class: "error", role: "alert", hidden: true });

  function validate() {
    const problem = idProblem(id.value);
    idHint.textContent = problem ? t(`enrol.id_${problem}`) : t("enrol.id_hint");
    idHint.classList.toggle("error", Boolean(problem) && id.value !== "");
    next.disabled = !(name.value.trim() && !problem && consent.checked);
  }
  name.addEventListener("input", () => {
    if (!idEdited) id.value = uniqueId(suggestId(name.value), taken);
    validate();
  });
  id.addEventListener("input", () => { idEdited = true; validate(); });
  consent.addEventListener("change", validate);

  const form = h("section", { class: "stack" },
    h("h1", {}, t("enrol.title")),
    h("p", {}, t("enrol.intro")),
    h("p", { class: "hint" }, t("enrol.advice")),
    h("p", { class: "hint" }, t("enrol.wait_note")),
    h("div", { class: "field" }, h("label", { for: "en-name" }, t("enrol.name")), name),
    h("div", { class: "field" }, h("label", { for: "en-id" }, t("enrol.id")), id, idHint),
    noticeSection(guide.consent, consent, "en"),
    formError, next);

  // ---- reading and recording ---------------------------------------
  const readingText = h("p", { class: "reading", lang });
  const clock = h("p", { class: "clock", role: "timer", "aria-live": "off" }, `${GOAL_SECONDS}`);
  const meter = h("meter", { min: 0, max: 1, value: 0, "aria-label": t("record.level") });
  const goNote = h("p", { class: "hint" }, t("enrol.read_hint"));
  const record = h("button", { class: "primary big", type: "button", onClick: () => startRecording() }, t("enrol.record"));
  const done = h("button", { class: "primary big", type: "button", hidden: true, disabled: true, onClick: () => send() }, t("enrol.finish"));
  const result = h("div", { class: "stack", role: "status", "aria-live": "polite" });
  const reading = h("section", { class: "stack", hidden: true },
    h("h1", {}, t("enrol.read_title")), readingText, goNote, clock, meter,
    h("div", { class: "row" }, record, done), result);

  root.append(form, reading, h("section", { class: "stack" }, h("h2", {}, t("enrol.enrolled")), enrolledList()));
  validate();
  setLeaveGuard(() => phase === "recording" || phase === "sending");

  function enrolledList() {
    const list = h("ul", { class: "plain" });
    const fill = (people) => {
      list.replaceChildren(...people.map((p) => h("li", { class: "line" },
        h("span", {}, `${p.name || p.id}`),
        h("button", { type: "button", onClick: () => forget(p) }, t("enrol.forget")))));
      if (!people.length) list.append(h("li", { class: "hint" }, t("enrol.none")));
    };
    fill(known.speakers);
    async function forget(p) {
      if (!confirm(t("enrol.forget_confirm", { name: p.name || p.id }))) return;
      try {
        await ctx.api.deleteSpeaker(p.id);
        known.speakers = known.speakers.filter((x) => x.id !== p.id);
        taken.splice(taken.indexOf(p.id), 1);
        fill(known.speakers);
      } catch (error) {
        alert(describeError(error));
      }
    }
    return list;
  }

  let textVersion = null;
  async function toReading() {
    next.disabled = true;
    formError.hidden = true;
    try {
      const text = await ctx.api.enrolmentText(lang, name.value.trim());
      textVersion = text.version;
      readingText.textContent = text.text;
    } catch (error) {
      formError.hidden = false;
      formError.textContent = describeError(error);
      validate();
      return;
    }
    form.hidden = true;
    reading.hidden = false;
    phase = "reading";
    record.focus();
  }

  async function startRecording() {
    result.replaceChildren();
    parts.length = 0;
    record.disabled = true;
    capture = new Capture({
      onPcm: (bytes) => parts.push(bytes),
      onLevel: (v) => { meter.value = Math.max(v, meter.value * 0.85); },
      onProblem: (kind) => result.replaceChildren(h("p", { class: "warn" }, t(kind === "ended" ? "record.mic_lost" : "record.mic_suspended"))),
    });
    try {
      await capture.start();
    } catch (error) {
      record.disabled = false;
      result.replaceChildren(h("p", { class: "error" }, t(error instanceof CaptureError ? `mic.${error.reason}` : "mic.failed")));
      return;
    }
    wake.request();
    phase = "recording";
    record.hidden = true;
    done.hidden = false;
    timer = setInterval(tick, 250);
    tick();
  }

  function tick() {
    const { left, done: reached } = countdown(capture.seconds, GOAL_SECONDS);
    clock.textContent = reached ? t("enrol.enough") : String(left);
    done.disabled = !reached;
    if (capture.seconds >= MAX_SECONDS) send();
  }

  async function send() {
    if (phase !== "recording") return;
    phase = "sending";
    clearInterval(timer);
    done.disabled = true;
    await capture.stop();
    await wake.release();
    const pcm = concatBytes(parts);
    parts.length = 0;
    await upload(pcm);
  }

  async function upload(pcm) {
    result.replaceChildren(h("p", {}, t("enrol.sending")), h("p", { class: "hint" }, t("enrol.wait_note")));
    try {
      await ctx.api.enrol(id.value, { name: name.value.trim(), language: lang, textVersion, pcm });
    } catch (error) {
      if (error.code === "busy") {
        // Other enrolments are waiting for the computer: try again by itself.
        const seconds = Math.max(error.waitSeconds || 10, 1);
        result.replaceChildren(h("p", { class: "warn" }, t("enrol.busy")),
          h("p", { class: "hint" }, t("enrol.busy_in", { seconds })));
        retryTimer = setTimeout(() => upload(pcm), seconds * 1000);
        return;
      }
      const again = h("button", { type: "button", class: "primary big", onClick: () => reset() }, t("enrol.retry"));
      const resend = error.kind === "unavailable" || error.kind === "server"
        ? h("button", { type: "button", class: "big", onClick: () => upload(pcm) }, t("action.retry")) : null;
      result.replaceChildren(h("p", { class: "error" }, describeError(error)), ...[resend, again].filter(Boolean));
      phase = "reading";
      return;
    }
    phase = "reading";
    result.replaceChildren(
      h("p", { class: "ok" }, t("enrol.ok", { name: name.value.trim() })),
      h("a", { class: "button primary", href: "#/" }, t("nav.sessions")));
    done.hidden = true;
  }

  function reset() {
    result.replaceChildren();
    record.hidden = false;
    record.disabled = false;
    done.hidden = true;
    done.disabled = true;
    clock.textContent = String(GOAL_SECONDS);
    record.focus();
  }

  return () => {
    clearInterval(timer);
    clearTimeout(retryTimer);
    if (phase === "recording") capture?.stop();
    wake.release();
    setLeaveGuard(() => false);
  };
}
