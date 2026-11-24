// «Ejemplo del asistente de voz» (#/voicebot/demo, #250): a scripted
// conversation between a customer and the voice assistant of the same
// fictitious company, with the assistant's sources next to each answer.
// The customer's lines are shown as text, the assistant's are written and
// (unless muted) spoken by the browser's own voice: nothing is recorded,
// no microphone, no server session, voice service, GPU or model is used,
// so it never takes anything from a real conversation. Pure parts:
// lib/voice-demo.js.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { localized } from "../lib/live.js";
import { abortableSleep, browserNarrator, play, stepsOf } from "../lib/voice-demo.js";
import { emptyState, icon, pageHead } from "../ui.js";

export async function voiceDemoView(ctx, root, demoId) {
  root.classList.add("wide");
  let data = null;
  try {
    data = await ctx.api.liveDemoVoice(demoId);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("vdemo.title"), { lead: t("vdemo.lead") }),
      emptyState({
        iconName: "voicebot", title: t("vdemo.unavailable"),
        body: error?.code === "not_found" ? t("vdemo.unavailable_body") : describeError(error),
      }));
    return null;
  }
  const exchanges = data.exchanges ?? [];
  const narrator = browserNarrator(globalThis.speechSynthesis, globalThis.SpeechSynthesisUtterance, lang);
  let controller = null;
  let closed = false;
  let sound = Boolean(narrator);

  const talk = h("ol", { class: "vd-talk", "aria-label": t("vdemo.talk"), "aria-live": "polite" });
  const status = h("p", { class: "hint", role: "status" });
  const start = h("button", { type: "button", class: "primary big", onClick: () => run() }, icon("play"), t("vdemo.start"));
  const stop = h("button", { type: "button", hidden: true, onClick: () => halt() }, icon("stop"), t("vdemo.stop"));
  const mute = h("label", { class: "check vd-sound" },
    h("input", { type: "checkbox", checked: sound, disabled: !narrator, onChange: (e) => { sound = e.target.checked; if (!sound) narrator?.cancel(); } }),
    narrator ? t("vdemo.sound") : t("vdemo.no_voice"));

  fill(root,
    pageHead(localized(data, "title", lang) || t("vdemo.title"), { lead: t("vdemo.lead") }),
    h("p", { class: "notice info cr-demo", role: "note" }, icon("info"), " ", t("vdemo.scripted")),
    h("p", { class: "hint" }, localized(data, "intro", lang)),
    h("div", { class: "row wrap vd-actions" }, start, stop, mute),
    status,
    h("section", { class: "card vd-card", "aria-labelledby": "vd-talk-title" }, h("h2", { id: "vd-talk-title", class: "lv-label" }, t("vdemo.talk")), talk),
    h("p", { class: "hint" }, t("vdemo.real"), " ", h("a", { href: "#/voicebot" }, t("vdemo.try_real"))));
  if (!exchanges.length) return null;

  function show(step) {
    if (closed) return;
    if (step.kind === "thinking") {
      talk.querySelector(".vd-thinking")?.remove();
      talk.append(h("li", { class: "vd-turn is-bot vd-thinking" }, h("span", { class: "vd-who" }, t("voicebot.bot")), h("span", { class: "vd-text" }, t("voicebot.thinking"))));
    } else if (step.kind === "user") {
      talk.append(h("li", { class: "vd-turn is-user" }, h("span", { class: "vd-who" }, t("vdemo.customer")), h("span", { class: "vd-text" }, step.text)));
    } else {
      talk.querySelector(".vd-thinking")?.remove();
      talk.append(h("li", { class: "vd-turn is-bot" }, h("span", { class: "vd-who" }, t("voicebot.bot")), h("span", { class: "vd-text" }, step.text),
        step.sources.length ? h("span", { class: "vd-sources" }, h("span", { class: "cr-label" }, t("vdemo.sources")),
          step.sources.map((s) => h("details", { open: true }, h("summary", {}, t("live.source"), " ", h("code", {}, s.id)), h("p", {}, s.text)))) : null));
    }
    talk.lastElementChild?.scrollIntoView?.({ block: "nearest" });
    status.textContent = step.kind === "bot" ? t("vdemo.speaking") : step.kind === "thinking" ? t("voicebot.thinking") : t("vdemo.listening");
  }

  async function run() {
    halt();
    fill(talk);
    controller = new AbortController();
    const mine = controller;
    start.hidden = true;
    stop.hidden = false;
    const done = await play(stepsOf(data, data.sources, lang), {
      show, sleep: abortableSleep(mine.signal), signal: mine.signal,
      narrate: narrator ? (text) => (sound ? narrator.speak(text) : Promise.resolve(false)) : null,
    });
    if (closed || controller !== mine) return;
    start.hidden = false;
    stop.hidden = true;
    start.lastChild.textContent = t("vdemo.again");
    status.textContent = done ? t("vdemo.done") : "";
  }

  function halt() {
    controller?.abort();
    narrator?.cancel();
  }

  return () => {
    closed = true;
    halt();
  };
}
