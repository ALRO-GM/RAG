// «Asistente de voz» (#/voicebot): talk to the voice bot (docs/live/VOICEBOT.md).
// The microphone goes to the transcriber as a one-person live session of
// the "voicebot" product; the orchestrator answers out loud: its feed
// carries the user's sentences, the bot's sentences with their audio
// (base64 WAV, played back to back) and interruptions. Push-to-talk sends
// silence while the button is up (so the transcriber still closes the
// sentence); hands-free sends the microphone all the time. Barge-in: with
// a headset, speaking over the bot stops its voice at once; without one,
// the talk button does. The state lives in lib/voicebot.js; this module
// draws it and plays the audio.
//
// Topics (#208): the page offers every product that speaks — the support
// bot (whose knowledge domain can be changed) and the AI interviewer
// (voicebot.interview), which first needs a job offer and a reviewed
// interview plan (views/interview.js), then speaks first, shows the
// recruiter a live coverage panel and, at the end, a visual report.

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { setLeaveGuard } from "../device.js";
import { LiveAudio } from "../live-audio.js";
import { newSessionId } from "../lib/live.js";
import {
  EnergyVad, LISTENING_MS, MODES, SpeechQueue, VOICEBOT, VoiceBotState, base64ToBytes, playQuiet, turnLatency,
} from "../lib/voicebot.js";
import { emptyState, icon, pageHead, toast } from "../ui.js";
import { INTERVIEW, InterviewState, speakerProducts } from "../lib/interview.js";
import { coveragePanel, planSetup, reportCard } from "./interview.js";
import { connectionCard } from "./connection.js";
import { needsOf } from "../lib/connection.js";

const RETRY_MS = 2000;

function botError(error) {
  if (error?.code === "orchestrator_unavailable") return t("voicebot.unavailable_body");
  if (error?.code === "transcriber") return t("live.err_transcriber");
  return describeError(error);
}

function maybe(key, fallback, params) {
  const text = t(key, params);
  return text === key ? fallback : text;
}

export async function voicebotView(ctx, root) {
  root.classList.add("wide");
  let product = null;
  let topics = [];
  try {
    const { products } = await ctx.api.liveProducts();
    topics = speakerProducts(products);
    product = topics.find((p) => p.product === VOICEBOT) ?? topics[0] ?? null;
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    product = null;
  }
  if (!product) {
    root.append(pageHead(t("voicebot.title"), { lead: t("voicebot.lead") }),
      emptyState({ iconName: "voicebot", title: t("voicebot.unavailable"), body: t("voicebot.unavailable_body") }));
    return null;
  }

  let closed = false;
  let controller = null;
  let audio = null;
  let audioCtx = null;
  let running = false;
  setLeaveGuard(() => running);

  function cleanup() {
    closed = true;
    controller?.abort();
    audio?.stop();
    audioCtx?.close?.().catch(() => {});
    setLeaveGuard(() => false);
  }

  showForm();
  return cleanup;

  // ---- start ------------------------------------------------------------

  function showForm() {
    root.replaceChildren();
    const topic = h("select", { id: "vb-topic" }, topics.map((p) => h("option", { value: p.product, selected: p.product === product.product },
      maybe(`voicebot.topic.${p.product}`, p.product))));
    const topicHint = h("p", { class: "hint" });
    const domain = h("select", { id: "vb-domain" });
    const domainBox = h("div", { class: "field" }, h("label", { for: "vb-domain" }, t("voicebot.domain")), domain);
    const interviewBox = h("div", { class: "vb-interview" });
    let domainsLoaded = false;
    const mode = h("select", { id: "vb-mode" }, MODES.map((m) => h("option", { value: m }, t(`voicebot.mode.${m}`))));
    const modeHint = h("p", { class: "hint" });
    const headset = h("input", { type: "checkbox", id: "vb-headset", checked: true });
    const start = h("button", { type: "button", class: "primary big", onClick: () => begin() }, icon("voicebot"), t("voicebot.start"));
    const formError = h("p", { class: "error", role: "alert", hidden: true });
    // «Calidad de la conexión» (#247): the bot hears this browser's microphone.
    const connection = connectionCard({
      getCode: () => ctx.api.getCode?.() ?? null,
      options: () => ({ needs: needsOf("live"), upload: true, liveAudio: Boolean(product.live) }),
    });
    const sync = () => {
      modeHint.textContent = t(`voicebot.mode_hint.${mode.value}`);
      product = topics.find((p) => p.product === topic.value) ?? product;
      connection.redraw();
      const interview = Boolean(product.interview);
      topicHint.textContent = maybe(`voicebot.topic_hint.${product.product}`, "");
      domainBox.hidden = interview || !product.domain;
      start.hidden = interview;
      interviewBox.hidden = !interview;
      if (interview && !interviewBox.firstChild) {
        interviewBox.append(planSetup(ctx, { onReady: (plan, candidate) => begin({ plan: plan.id, candidate: candidate || undefined }) }));
      }
      if (!interview && product.domain && !domainsLoaded) loadDomains();
    };
    async function loadDomains() {
      domainsLoaded = true;
      try {
        const rows = await ctx.api.knowledgeDomains();
        domain.replaceChildren(...rows.map((d) => h("option", { value: d.id, selected: d.id === product.domain }, `${d.title} (${d.id})`)));
      } catch {
        domain.replaceChildren(h("option", { value: product.domain }, product.domain));
      }
    }
    mode.addEventListener("change", sync);
    topic.addEventListener("change", sync);
    if (!product.live) {
      start.disabled = true;
      formError.textContent = t("voicebot.no_mic");
      formError.hidden = false;
    }

    async function begin(extra = {}) {
      formError.hidden = true;
      start.disabled = true;
      start.textContent = t("voicebot.starting");
      const body = { product: product.product, audio: "mic", session_id: newSessionId(product.interview ? "entrevista" : "voz"), ...extra };
      if (!product.interview && domain.value && domain.value !== product.domain) body.domain = domain.value;
      // Created inside the click, so the browser lets it play the bot's voice.
      const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
      audioCtx = AudioContextClass ? new AudioContextClass() : null;
      audioCtx?.resume?.();
      try {
        const session = await ctx.api.startLive(body);
        showSession(session, { mode: mode.value, headset: headset.checked, interview: Boolean(product.interview) });
      } catch (error) {
        if (error?.kind === "unauthorized") return ctx.fail(error);
        formError.textContent = botError(error);
        formError.hidden = false;
        if (product.interview) toast(botError(error), { error: true });
        start.disabled = false;
        start.replaceChildren(icon("voicebot"), t("voicebot.start"));
      }
      return null;
    }

    root.append(
      pageHead(t("voicebot.title"), { lead: t("voicebot.lead") }),
      connection.node,
      h("section", { class: "card vb-start" },
        h("p", { class: "notice info", role: "note" }, t("voicebot.notice")),
        topics.length > 1 ? h("div", { class: "field" }, h("label", { for: "vb-topic" }, t("voicebot.topic")), topic) : null,
        topicHint, domainBox,
        h("div", { class: "field" }, h("label", { for: "vb-mode" }, t("voicebot.mode")), mode), modeHint,
        h("label", { class: "check", for: "vb-headset" }, headset, " ", t("voicebot.headset")),
        h("p", { class: "hint" }, t("voicebot.headset_hint")),
        start),
      formError, interviewBox);
    sync();
    connection.check();
  }

  // ---- a running conversation ---------------------------------------------

  function showSession(session, { mode, headset, interview = false }) {
    root.replaceChildren();
    const state = new VoiceBotState();
    state.apply({ type: "session", session });
    const ivState = interview ? new InterviewState() : null;
    const coverage = h("div", { class: "iv-live" });
    const report = h("div", { class: "iv-result" });
    let stopping = false;
    running = true;
    let speaking = false;
    let pressed = false;
    let currentTurn = 0;
    let listeningUntil = 0;
    const queue = audioCtx ? new SpeechQueue(audioCtx, { onPlaying: (on) => { speaking = on; schedule(); if (!on) closeWhenDone(); } }) : null;
    const vad = new EnergyVad();

    const stop = h("button", { type: "button", class: "danger", onClick: () => end() }, icon("stop"), t("voicebot.stop"));
    const again = h("button", { type: "button", class: "primary", hidden: true, onClick: () => showForm() }, t("voicebot.again"));
    const status = h("div", { class: "row wrap live-status", role: "status" });
    const alerts = h("div", { class: "stack tight", "aria-live": "polite" });
    const lines = h("ol", { class: "vb-lines", "aria-label": t("voicebot.conversation") });
    const meter = h("span", { class: "vb-meter-fill" });
    const talk = h("button", { type: "button", class: "vb-talk", "aria-pressed": "false" }, icon("record"), h("span", { class: "label" }));
    const latencyLine = h("p", { class: "meta vb-latency" });
    root.append(
      pageHead(t("voicebot.title"), { actions: [stop, again] }),
      h("p", { class: "notice info", role: "note" }, t("voicebot.notice")),
      status, alerts,
      h("div", { class: interview ? "vb-layout iv-layout" : "vb-layout" },
        h("section", { class: "vb-pane" }, h("h2", {}, t("voicebot.conversation")), lines, latencyLine),
        interview ? coverage : null,
        h("section", { class: "vb-control card" },
          talk,
          h("span", { class: "vb-meter", "aria-hidden": "true" }, meter),
          h("p", { class: "hint" }, t(`voicebot.mode_hint.${mode}`)))),
      report);

    let drawing = false;
    function schedule() {
      if (drawing) return;
      drawing = true;
      (globalThis.requestAnimationFrame ?? setTimeout)(() => {
        drawing = false;
        draw();
      });
    }

    function thinking() {
      return state.lines().filter((l) => l.kind === "bot").at(-1)?.state === "thinking";
    }

    function phase() {
      if (speaking) return "speaking";
      // The bot's voice is still loading: its first line waits for it.
      if (state.preparing) return "preparing";
      if (thinking()) return "thinking";
      return "listening";
    }

    function draw() {
      const now = phase();
      fill(status,
        h("span", { class: `badge is-${state.state === "running" ? "processing" : state.state === "failed" ? "failed" : "indexed"}` },
          maybe(`voicebot.state.${state.state}`, state.state)),
        h("span", { class: "tag accent" }, t(`voicebot.mode.${mode}`)),
        state.state === "running" ? h("span", { class: `tag vb-phase is-${now}` }, t(now === "thinking" ? "voicebot.thinking" : `voicebot.${now}`)) : null,
        state.state === "running" && performance.now() < listeningUntil
          ? h("span", { class: "tag accent vb-backchannel", role: "status" }, t("voicebot.active_listening")) : null);
      const notes = [];
      if (state.halted) notes.push(h("p", { class: "notice error", role: "alert" }, t(`voicebot.halted.${state.halted}`)));
      if (state.state === "failed") notes.push(h("p", { class: "notice error", role: "alert" }, t("voicebot.failed", { error: state.error ?? "" })));
      fill(alerts, notes);
      const atEnd = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 40;
      const shown = state.lines();
      fill(lines, shown.length ? shown.map((l) => h("li", {
        class: ["vb-line", `is-${l.kind}`, l.partial ? "partial" : "", l.state ? `state-${l.state}` : ""].filter(Boolean).join(" "),
      },
      h("span", { class: "who" }, l.kind === "user" ? t("voicebot.you") : t("voicebot.bot"),
        l.state === "interrupted" ? h("span", { class: "tag warn" }, t("voicebot.interrupted")) : null),
      h("span", { class: "text" }, l.text || (l.state === "thinking" ? t("voicebot.thinking") : "…")))) : h("li", { class: "hint" }, t(interview ? "voicebot.waiting_interview" : "voicebot.waiting")));
      if (ivState) {
        fill(coverage, coveragePanel(ivState));
        if (ivState.result && !report.firstChild) fill(report, reportCard(ivState.result));
      }
      if (atEnd) lines.scrollTop = lines.scrollHeight;
      const record = state.records.at(-1);
      const ms = turnLatency(record);
      latencyLine.hidden = ms.first === null;
      if (ms.first !== null) {
        latencyLine.textContent = t("voicebot.latency", {
          total: ms.total ?? "–", lag: ms.lag === null ? "–" : Math.round(ms.lag),
          knowledge: ms.knowledge === null ? "–" : Math.round(ms.knowledge), first: Math.round(ms.first),
        });
      }
      const label = talk.querySelector(".label");
      talk.disabled = state.state !== "running";
      talk.setAttribute("aria-pressed", String(pressed));
      talk.classList.toggle("is-on", pressed || (mode === "hands_free" && state.state === "running"));
      if (mode === "push") label.textContent = pressed ? t("voicebot.talking") : t("voicebot.talk");
      else label.textContent = speaking ? t("voicebot.interrupt") : t("voicebot.listening");
      stop.hidden = state.state !== "running";
      again.hidden = state.state === "running";
    }

    function bargeIn(always = false) {
      // An interview also reports the candidate's voice when the bot is
      // silent: it restarts the wait for the end of the answer (#213).
      const busy = speaking || thinking();
      if (!busy && !always) return;
      if (busy) queue?.stop(currentTurn);
      ctx.api.bargeIn(session.id).catch(() => {});
    }

    function backchannel(item) {
      listeningUntil = performance.now() + LISTENING_MS;
      if (item.audio && audioCtx && !speaking) playQuiet(audioCtx, base64ToBytes(item.audio), item.volume);
      setTimeout(schedule, LISTENING_MS + 50);
    }

    function closeWhenDone() {
      // The interviewer said goodbye: stop the session once its voice is out.
      if (!ivState?.closed || speaking || stopping || state.state !== "running") return;
      stopping = true;
      setTimeout(() => {
        // The goodbye may still be decoding: wait for it to play out.
        if (speaking) stopping = false;
        else if (state.state === "running") end();
      }, 1500);
    }

    function onMessage(message) {
      state.apply(message.data, message.id);
      ivState?.apply(message.data);
      const item = message.data;
      if (item.type === "interview" && item.state === "closed") closeWhenDone();
      if (item.type === "bot") currentTurn = Math.max(currentTurn, item.turn);
      if (item.type === "speech" && item.audio && queue) queue.enqueue(item.turn, base64ToBytes(item.audio));
      if (item.type === "interrupt") queue?.stop(item.turn);
      if (item.type === "backchannel") backchannel(item);
      schedule();
    }

    function onLevel(level) {
      meter.style.transform = `scaleX(${Math.min(1, level * 3).toFixed(3)})`;
      const listening = mode === "hands_free" || pressed;
      if (!listening) return;
      if (vad.push(level, performance.now()) !== "start") return;
      if (speaking) {
        if (headset) bargeIn();
      } else if (interview) {
        bargeIn(true);
      }
    }

    function press(on) {
      if (state.state !== "running") return;
      if (mode === "hands_free") {
        if (on) bargeIn();
        return;
      }
      if (on === pressed) return;
      pressed = on;
      if (audio) audio.muted = !on;
      if (on) bargeIn();
      vad.reset();
      schedule();
    }

    talk.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      talk.setPointerCapture?.(event.pointerId);
      press(true);
    });
    talk.addEventListener("pointerup", () => press(false));
    talk.addEventListener("pointercancel", () => press(false));
    talk.addEventListener("keydown", (event) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) {
        event.preventDefault();
        press(true);
      }
    });
    talk.addEventListener("keyup", (event) => {
      if (event.key === " " || event.key === "Enter") press(false);
    });

    async function follow() {
      controller = new AbortController();
      while (!closed) {
        try {
          const how = await ctx.api.followLive(session.id, {
            after: state.lastId, signal: controller.signal,
            onItem: (item) => {
              let data = null;
              try {
                data = JSON.parse(item.data);
              } catch {
                return;
              }
              onMessage({ id: item.id, data });
            },
          });
          if (how === "end") break;
        } catch (error) {
          if (error?.kind === "unauthorized") return ctx.fail(error);
        }
        if (closed) break;
        await new Promise((resolve) => setTimeout(resolve, RETRY_MS));
      }
      running = false;
      if (state.state === "running") state.state = "ended";
      await audio?.stop();
      audio = null;
      schedule();
      return null;
    }

    async function end() {
      stop.disabled = true;
      try {
        queue?.stop();
        await audio?.stop();
        await ctx.api.stopLive(session.id);
      } catch (error) {
        toast(botError(error), { error: true });
      }
      stop.disabled = false;
    }

    audio = new LiveAudio({
      id: session.id, code: ctx.api.getCode(), echoCancellation: true, onLevel,
      onProblem: (kind) => toast(maybe(`live.mic.${kind}`, t("live.mic.socket")), { error: true }),
    });
    audio.muted = mode === "push";
    audio.start().catch(() => {
      toast(t("live.mic.failed"), { error: true });
      ctx.api.stopLive(session.id).catch(() => {});
    });
    draw();
    follow();
  }
}
