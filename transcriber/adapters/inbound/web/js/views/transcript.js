// Transcript player: one row per utterance, click to hear it.

import { h } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import {
  DEFAULT_LEAD, activeIndex, filterBySpeaker, isAnonymous, mergeTargets, parseLead,
  previousHint, seekTime, sortUtterances, speakerName,
} from "../lib/transcript.js";

const LEAD_KEY = "transcriber.lead";
const SPEEDS = [0.75, 1, 1.25, 1.5, 2];

function storedLead() {
  try {
    return parseLead(localStorage.getItem(LEAD_KEY) ?? "", DEFAULT_LEAD);
  } catch {
    return DEFAULT_LEAD;
  }
}

function storeLead(value) {
  try {
    localStorage.setItem(LEAD_KEY, String(value));
  } catch {
    // not persisted; still used for this visit
  }
}

export async function transcriptView(ctx, root, params) {
  const id = decodeURIComponent(params[0] ?? "");
  const session = await ctx.api.getSession(id);
  if (session.state !== "done") {
    root.append(h("h1", {}, session.title || id),
      h("p", { class: "notice" }, t("transcript.not_ready", { state: t(`state.${session.state}`) })),
      h("a", { class: "button", href: "#/" }, t("nav.sessions")));
    return null;
  }
  let transcript = await ctx.api.getTranscript(id);
  let enrolled = [];
  try {
    enrolled = (await ctx.api.speakers()).speakers;
  } catch {
    enrolled = [];
  }
  let audioReady = true;
  try {
    await ctx.api.audioCookie(); // <audio> cannot send headers: an HttpOnly cookie lets it in
  } catch (error) {
    if (error.kind === "unauthorized") throw error;
    audioReady = false;
  }

  let lead = storedLead();
  let utterances = [];
  let starts = [];
  let ends = [];
  let rows = [];
  let current = -1;
  let filter = "";
  let follow = true;

  const audio = h("audio", { preload: "metadata", hidden: true });
  if (audioReady) audio.src = `/api/sessions/${encodeURIComponent(id)}/audio`;
  const clock = h("span", { class: "time", role: "timer", "aria-live": "off" }, "0:00");
  const play = h("button", { type: "button", class: "primary", onClick: () => (audio.paused ? audio.play() : audio.pause()) }, t("player.play"));
  const back = h("button", { type: "button", onClick: () => skip(-5) }, t("player.back"));
  const fwd = h("button", { type: "button", onClick: () => skip(5) }, t("player.forward"));
  const speed = h("select", { id: "pl-speed", "aria-label": t("player.speed"), onChange: () => { audio.playbackRate = Number(speed.value); } },
    SPEEDS.map((s) => h("option", { value: s, selected: s === 1 }, `${s}×`)));
  const leadField = h("input", { id: "pl-lead", type: "number", min: 0, max: 60, step: 0.5, inputMode: "decimal", value: String(lead) });
  leadField.addEventListener("change", () => {
    lead = parseLead(leadField.value, DEFAULT_LEAD);
    leadField.value = String(lead);
    storeLead(lead);
  });
  const audioError = h("p", { class: "error", role: "alert", hidden: audioReady }, t("player.no_audio"));
  audio.addEventListener("error", () => { audioError.hidden = false; });
  audio.addEventListener("play", () => { play.textContent = t("player.pause"); });
  audio.addEventListener("pause", () => { play.textContent = t("player.play"); });
  audio.addEventListener("timeupdate", () => highlight(audio.currentTime));
  audio.addEventListener("seeked", () => highlight(audio.currentTime));

  const filterSelect = h("select", { id: "pl-filter", onChange: () => { filter = filterSelect.value; applyFilter(); } });
  const followBox = h("input", { id: "pl-follow", type: "checkbox", checked: true, onChange: () => { follow = followBox.checked; } });
  const message = h("p", { class: "notice", role: "status", hidden: true });
  const speakersPanel = h("div", { class: "stack" });
  const list = h("ol", { class: "utterances" });

  const bar = h("div", { class: "player" },
    h("div", { class: "row wrap" }, play, back, fwd, clock, speed),
    audioError);
  const options = h("div", { class: "row wrap options" },
    h("label", { for: "pl-lead" }, t("player.lead")), leadField,
    h("label", { for: "pl-filter" }, t("player.filter")), filterSelect,
    h("label", { class: "check" }, followBox, h("span", {}, t("player.follow"))));

  root.append(
    h("h1", {}, transcript.session.title || id),
    h("p", { class: "meta" }, `${formatTime(transcript.session.duration)} · `, h("a", { href: "#/" }, t("nav.sessions"))),
    bar, options, audio, message,
    h("details", { class: "speakers" }, h("summary", {}, t("speakers.title")), speakersPanel),
    list,
    h("button", { type: "button", onClick: download }, t("transcript.download")));

  function skip(delta) {
    audio.currentTime = Math.max(0, audio.currentTime + delta);
  }

  function say(text, isError = false) {
    message.hidden = !text;
    message.textContent = text ?? "";
    message.classList.toggle("error", isError);
  }

  function playFrom(index) {
    audio.currentTime = seekTime(utterances[index].start, lead);
    audio.play().catch(() => { audioError.hidden = false; });
  }

  // ---- rows --------------------------------------------------------
  function build() {
    utterances = sortUtterances(transcript.utterances);
    starts = utterances.map((u) => u.start);
    ends = utterances.map((u) => u.end);
    current = -1;
    const speakers = transcript.speakers;
    rows = utterances.map((u, index) => {
      const hint = previousHint(u, speakers);
      const button = h("button", {
        type: "button", class: "row-button", "data-index": index,
        onClick: () => playFrom(index),
      },
      h("span", { class: "time" }, formatTime(u.start)),
      h("span", { class: "who" }, speakerName(speakers, u.speaker)),
      h("span", { class: "text" }, u.text),
      hint ? h("span", { class: "after" }, t("transcript.after", { name: hint })) : null);
      return h("li", { class: "utterance", "data-speaker": u.speaker }, button, changeButton(u));
    });
    list.replaceChildren(...rows);
    applyFilter();
    highlight(audio.currentTime || 0, true);
  }

  function changeButton(u) {
    const slot = h("span", { class: "change" });
    const open = h("button", { type: "button", class: "small", onClick: () => {
      const choose = h("select", { "aria-label": t("transcript.reassign_label") },
        h("option", { value: "" }, t("transcript.reassign_pick")),
        transcript.speakers.filter((s) => s.id !== u.speaker).map((s) => h("option", { value: s.id }, speakerName(transcript.speakers, s.id))),
        mergeTargets(transcript.speakers, enrolled, u.speaker)
          .filter((p) => !transcript.speakers.some((s) => s.id === p.id))
          .map((p) => h("option", { value: p.id }, p.name)));
      choose.addEventListener("change", async () => {
        if (!choose.value) return;
        await edit(() => ctx.api.reassignUtterance(id, u.id, choose.value));
      });
      choose.addEventListener("blur", () => slot.replaceChildren(open));
      slot.replaceChildren(choose);
      choose.focus();
    } }, t("transcript.reassign"));
    slot.append(open);
    return slot;
  }

  function highlight(time, force = false) {
    const shown = formatTime(time);
    if (clock.textContent !== shown) clock.textContent = shown;
    const index = activeIndex(starts, time, ends);
    if (index === current && !force) return;
    if (current >= 0 && rows[current]) {
      rows[current].classList.remove("active");
      rows[current].firstElementChild.removeAttribute("aria-current");
    }
    current = index;
    if (index < 0) return;
    const row = rows[index];
    row.classList.add("active");
    row.firstElementChild.setAttribute("aria-current", "true");
    if (follow && !row.hidden) {
      const box = row.getBoundingClientRect();
      const top = bar.getBoundingClientRect().bottom;
      if (box.top < top || box.bottom > window.innerHeight) row.scrollIntoView({ block: "center" });
    }
  }

  function applyFilter() {
    const shown = new Set(filterBySpeaker(utterances, filter));
    utterances.forEach((u, i) => { rows[i].hidden = !shown.has(u); });
  }

  // ---- speakers ----------------------------------------------------
  function renderSpeakers() {
    const previous = filterSelect.value;
    filterSelect.replaceChildren(h("option", { value: "" }, t("player.everyone")),
      ...transcript.speakers.map((s) => h("option", { value: s.id }, speakerName(transcript.speakers, s.id))));
    filterSelect.value = transcript.speakers.some((s) => s.id === previous) ? previous : "";
    filter = filterSelect.value;
    speakersPanel.replaceChildren(...transcript.speakers.map((s) => speakerRow(s)));
  }

  function speakerRow(s) {
    const label = h("strong", {}, speakerName(transcript.speakers, s.id));
    if (!isAnonymous(s)) return h("div", { class: "line" }, label, h("span", { class: "hint" }, s.enrolled ? t("speakers.enrolled") : ""));
    const name = h("input", { type: "text", maxLength: 100, autocomplete: "off", value: s.name ?? "", "aria-label": t("speakers.name_for", { label: s.id }) });
    const save = h("button", { type: "button", onClick: () => {
      if (name.value.trim()) edit(() => ctx.api.renameSpeaker(id, s.id, { name: name.value.trim() }));
    } }, t("speakers.rename"));
    const targets = mergeTargets(transcript.speakers, enrolled, s.id);
    const merge = h("select", { "aria-label": t("speakers.merge_label", { label: s.id }) },
      h("option", { value: "" }, t("speakers.merge_pick")),
      targets.map((p) => h("option", { value: p.id }, p.name)));
    merge.addEventListener("change", () => {
      if (!merge.value) return;
      const target = targets.find((p) => p.id === merge.value);
      if (confirm(t("speakers.merge_confirm", { from: speakerName(transcript.speakers, s.id), to: target.name }))) {
        edit(() => ctx.api.renameSpeaker(id, s.id, { merge_into: merge.value }));
      } else {
        merge.value = "";
      }
    });
    return h("div", { class: "stack tight" }, label, h("div", { class: "row wrap" }, name, save), targets.length ? merge : null);
  }

  async function edit(call) {
    say(null);
    try {
      transcript = await call();
      renderSpeakers();
      build();
      say(t("transcript.saved"));
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error, () => {});
      say(describeError(error), true);
    }
  }

  function download() {
    const contract = { ...transcript, utterances: transcript.utterances.map(({ words, ...rest }) => rest) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(contract, null, 2)], { type: "application/json" }));
    const link = h("a", { href: url, download: `${id}.transcript.json` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  renderSpeakers();
  build();

  return () => {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  };
}
