// «Ejemplo del asistente de voz» (#/voicebot/demo, #250): a scripted
// conversation between a customer and the voice assistant of the same
// fictitious company, with the assistant's sources next to each answer.
// Every line is written; with the clips rendered ahead of time (#254) both
// the customer and the assistant are heard, each with their own synthetic
// voice, and every line lasts as long as its clip with the sound on or off.
// Without clips the assistant is spoken by the browser's own voice and the
// page says the voices are not available. Nothing plays before «Empezar»;
// nothing is recorded, no microphone, server session, voice service, GPU or
// model is used. Pure parts: lib/voice-demo.js, lib/demo-voices.js.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { localized } from "../lib/live.js";
import { abortableSleep, browserNarrator, play, stepsOf } from "../lib/voice-demo.js";
import { ClipPlayer, clipsOf, readSound, voicesNoteKey, writeSound } from "../lib/demo-voices.js";
import { emptyState, icon, pageHead } from "../ui.js";

export async function voiceDemoView(ctx, root, demoId) {
  root.classList.add("wide");
  let data = null;
  // The rendered voices are optional: without them the example is text.
  const voicesLoading = Promise.resolve(ctx.api.liveDemoVoices?.(demoId)).catch(() => null);
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
  const voices = await voicesLoading;
  const clips = clipsOf(voices, "voicebot");
  const narrator = clips ? null : browserNarrator(globalThis.speechSynthesis, globalThis.SpeechSynthesisUtterance, lang);
  let controller = null;
  let closed = false;
  let sound = Boolean(clips || narrator) && readSound();
  const voiceNote = h("p", { class: "hint vd-voices", role: "status" });
  const player = clips ? new ClipPlayer({
    load: async (key) => URL.createObjectURL(await ctx.api.liveDemoClip(demoId, key)),
    makeAudio: (url) => new Audio(url),
    onFail: () => { voiceNote.textContent = `${t(voicesNoteKey(voices, "voicebot"))} ${t("dvoice.failed")}`; },
  }) : null;
  player?.setSound(sound);

  const talk = h("ol", { class: "vd-talk", "aria-label": t("vdemo.talk") });
  const status = h("p", { class: "hint", role: "status" });
  const start = h("button", { type: "button", class: "primary big", onClick: () => run() }, icon("play"), t("vdemo.start"));
  const stop = h("button", { type: "button", hidden: true, onClick: () => halt() }, icon("stop"), t("vdemo.stop"));
  const mute = h("label", { class: "check vd-sound" },
    h("input", {
      type: "checkbox", checked: sound, disabled: !clips && !narrator,
      onChange: (e) => {
        sound = e.target.checked;
        writeSound(sound);
        player?.setSound(sound);
        if (!sound) narrator?.cancel();
      },
    }),
    clips ? t("dvoice.sound") : narrator ? t("vdemo.sound") : t("vdemo.no_voice"));

  fill(root,
    pageHead(localized(data, "title", lang) || t("vdemo.title"), { lead: t("vdemo.lead") }),
    h("p", { class: "notice info cr-demo", role: "note" }, icon("info"), " ", t("vdemo.scripted")),
    h("p", { class: "hint" }, localized(data, "intro", lang)),
    h("div", { class: "row wrap vd-actions" }, start, stop, mute),
    voiceNote,
    status,
    h("section", { class: "card vd-card", "aria-labelledby": "vd-talk-title" }, h("h2", { id: "vd-talk-title", class: "lv-label" }, t("vdemo.talk")), talk),
    h("p", { class: "hint" }, t("vdemo.real"), " ", h("a", { href: "#/voicebot" }, t("vdemo.try_real"))));
  // Said once, for screen readers too: what the voices are, or that there are none.
  queueMicrotask(() => { voiceNote.textContent = t(voicesNoteKey(voices, "voicebot")); });
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
        step.sources.length ? h("span", { class: "vd-sources" }, h("span", { class: "cr-label" }, t("vdemo.sources"), lang === "en" ? ` ${t("live.source_es")}` : ""),
          step.sources.map((s) => h("details", { open: true }, h("summary", {}, t("live.source"), ": ", s.title), h("p", {}, s.text)))) : null));
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
    const steps = stepsOf(data, data.sources, lang);
    player?.prefetch(steps.filter((x) => clips.has(x.key)).map((x) => x.key));
    const done = await play(steps, {
      show, sleep: abortableSleep(mine.signal), signal: mine.signal,
      narrate: narrator ? (text) => (sound ? narrator.speak(text) : Promise.resolve(false)) : null,
      clips, voice: player ? (key, seconds) => player.play(key, seconds) : null,
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
    player?.stop();
  }

  return () => {
    closed = true;
    halt();
    player?.dispose((url) => URL.revokeObjectURL(url));
  };
}
