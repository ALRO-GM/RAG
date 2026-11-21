// «En directo» (#/live): live assist during a support call or an interview.
// Start a live session (stereo channels from the workstation, or this
// browser's microphone) or a demo that replays a recorded conversation;
// then follow the orchestrator's stream: the transcript with speakers and
// partials, suggestion cards with their sources (knowledge chunks) and the
// sentences they are about, the script-phase coverage of interviews and the
// notice banner. Customer signals appear only in support calls; an
// interview never shows signals or ratings about the candidate. The state
// lives in lib/live.js (LiveState); this module only draws it.

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { setLeaveGuard } from "../device.js";
import { LiveAudio } from "../live-audio.js";
import { AUDIO, LiveState, SPEEDS, latency, newSessionId, productKind } from "../lib/live.js";
import { emptyState, icon, pageHead, toast } from "../ui.js";

const RETRY_MS = 2000;
const TYPE_OF = { support: "soporte", interview: "entrevista" };

function liveError(error) {
  if (error?.code === "orchestrator_unavailable") return t("live.unavailable_body");
  if (error?.code === "transcriber") return t("live.err_transcriber");
  return describeError(error);
}

/** `t(key)`, or `fallback` when the dictionaries have no such key (codes from the server). */
function maybe(key, fallback, params) {
  const text = t(key, params);
  return text === key ? fallback : text;
}

function roleText(role) {
  return maybe(`live.role.${role}`, role);
}

/** The demo transcript that fits a product (its conversation type), else the first. */
function defaultTranscript(transcripts, kind) {
  return (transcripts.find((x) => x.type === TYPE_OF[kind]) ?? transcripts[0])?.id ?? "";
}

export async function liveView(ctx, root) {
  root.classList.add("wide");
  let products = [];
  let transcripts = [];
  try {
    [{ products }, { transcripts }] = await Promise.all([ctx.api.liveProducts(), ctx.api.liveTranscripts()]);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("live.title"), { lead: t("live.lead") }),
      emptyState({ iconName: "live", title: t("live.unavailable"), body: liveError(error) }));
    return null;
  }

  let closed = false;
  let controller = null;
  let audio = null;
  let running = false;
  setLeaveGuard(() => running);

  function cleanup() {
    closed = true;
    controller?.abort();
    audio?.stop();
    setLeaveGuard(() => false);
  }

  showForm();
  return cleanup;

  // ---- start ----------------------------------------------------------

  function showForm() {
    root.replaceChildren();
    const productSelect = h("select", { id: "live-product" },
      products.map((p) => h("option", { value: p.product }, t(`live.product.${productKind(p.product)}`))));
    const audioSelect = h("select", { id: "live-audio" }, AUDIO.map((a) => h("option", { value: a }, t(`live.audio.${a}`))));
    const transcriptSelect = h("select", { id: "live-transcript" },
      transcripts.map((x) => h("option", { value: x.id }, `${x.title} · ${t("live.utterances", { n: x.utterances })}`)));
    const speedSelect = h("select", { id: "live-speed" }, SPEEDS.map((s) => h("option", { value: String(s) }, `${s}×`)));
    const banner = h("p", { class: "notice info", role: "note" });
    const liveHint = h("p", { class: "hint" });
    const start = h("button", { type: "button", class: "primary big", onClick: () => begin("live") }, icon("live"), t("live.start"));
    const demo = h("button", { type: "button", class: "big", onClick: () => begin("replay") }, icon("play"), t("live.demo"));
    const formError = h("p", { class: "error", role: "alert", hidden: true });

    function chosen() {
      return products.find((p) => p.product === productSelect.value) ?? products[0];
    }

    function sync() {
      const product = chosen();
      const kind = productKind(product?.product);
      banner.textContent = maybe(`live.banner.${product?.banner}`, t("live.banner.default"));
      start.disabled = !product?.live;
      audioSelect.disabled = !product?.live;
      liveHint.textContent = product?.live ? t(`live.audio_hint.${audioSelect.value}`, { who: roleText(product.suggest_about?.[0] ?? "unknown") }) : t("live.live_off");
      transcriptSelect.value = defaultTranscript(transcripts, kind);
      demo.disabled = !transcripts.length;
    }

    async function begin(mode) {
      formError.hidden = true;
      start.disabled = true;
      demo.disabled = true;
      const product = chosen().product;
      try {
        let session;
        if (mode === "live") {
          session = await ctx.api.startLive({ product, audio: audioSelect.value, session_id: newSessionId("live") });
        } else {
          session = await ctx.api.startReplay(newSessionId("demo"), { product, transcript: transcriptSelect.value, speed: Number(speedSelect.value) });
        }
        showSession(session, mode === "live" ? audioSelect.value : null);
      } catch (error) {
        if (error?.kind === "unauthorized") return ctx.fail(error);
        formError.textContent = liveError(error);
        formError.hidden = false;
        sync();
      }
      return null;
    }

    productSelect.addEventListener("change", sync);
    audioSelect.addEventListener("change", sync);
    const field = (input, key) => h("div", { class: "field" }, h("label", { for: input.id }, t(key)), input);
    root.append(
      pageHead(t("live.title"), { lead: t("live.lead") }),
      h("div", { class: "live-start" },
        h("section", { class: "card" },
          h("h2", {}, t("live.new")),
          field(productSelect, "live.product"), banner,
          field(audioSelect, "live.audio"), liveHint, start),
        h("section", { class: "card" },
          h("h2", {}, t("live.demo_title")), h("p", { class: "hint" }, t("live.demo_body")),
          field(transcriptSelect, "live.transcript"), field(speedSelect, "live.speed"), demo)),
      formError);
    sync();
  }

  // ---- a running session ----------------------------------------------

  function showSession(session, audioKind) {
    root.replaceChildren();
    const state = new LiveState();
    state.apply({ type: "session", session });
    running = true;
    const stop = h("button", { type: "button", class: "danger", onClick: () => end() }, icon("stop"), t("live.stop"));
    const again = h("button", { type: "button", class: "primary", hidden: true, onClick: () => showForm() }, t("live.again"));
    const status = h("div", { class: "row wrap live-status", role: "status" });
    const alerts = h("div", { class: "stack tight", "aria-live": "polite" });
    const phases = h("ol", { class: "live-phases", "aria-label": t("live.phases") });
    const lines = h("ol", { class: "live-lines", "aria-label": t("live.transcript_title") });
    const cards = h("div", { class: "live-cards" });
    const banner = h("p", { class: "notice info", role: "note" }, maybe(`live.banner.${state.page?.banner}`, t("live.banner.default")));
    root.append(
      pageHead(t("live.title"), { actions: [stop, again] }),
      banner, status, alerts, phases,
      h("div", { class: "live-layout" },
        h("section", { class: "live-pane" }, h("h2", {}, t("live.transcript_title")), lines),
        h("section", { class: "live-pane" }, h("h2", {}, t("live.suggestions_title")), cards)));
    let drawing = false;
    let reconnecting = false;

    function schedule() {
      if (drawing) return;
      drawing = true;
      (globalThis.requestAnimationFrame ?? setTimeout)(() => {
        drawing = false;
        draw();
      });
    }

    function draw() {
      const page = state.page ?? {};
      const kind = productKind(page.product);
      const counts = state.counts();
      fill(status,
        h("span", { class: `badge is-${state.state === "running" ? "processing" : state.state === "failed" ? "failed" : "indexed"}` }, t(`live.state.${state.state}`)),
        h("span", { class: "tag accent" }, t(`live.product.${kind}`)),
        h("span", { class: "tag" }, session.mode === "replay" ? t("live.mode.replay") : t(`live.audio.${audioKind ?? "stereo"}`)),
        h("span", { class: "meta" }, t("live.calls", { ok: counts.ok ?? 0, all: state.records.length })));
      const notes = [];
      if (state.halted) notes.push(h("p", { class: "notice error", role: "alert" }, t(`live.halted.${state.halted}`)));
      if (state.state === "failed") notes.push(h("p", { class: "notice error", role: "alert" }, t("live.failed", { error: state.error ?? "" })));
      if (reconnecting && state.state === "running") notes.push(h("p", { class: "notice" }, t("live.reconnecting")));
      for (const w of state.warnings.slice(-3)) notes.push(h("p", { class: "notice" }, maybe(`live.warning.${w.code}`, w.message)));
      fill(alerts, notes);
      const coverage = state.coverage();
      phases.hidden = !coverage.length;
      fill(phases, coverage.map((c) => h("li", { class: `live-phase is-${c.state}` },
        maybe(`live.phase.${c.phase}`, c.phase), c.count ? h("span", { class: "meta" }, ` ${c.count}`) : null)));
      drawLines();
      drawCards();
      stop.hidden = state.state !== "running";
      again.hidden = state.state === "running";
    }

    function who(speaker) {
      const role = state.roleOf(speaker);
      const name = state.speakers.get(speaker)?.name;
      return name ? `${name} · ${roleText(role)}` : roleText(role);
    }

    function drawLines() {
      const atEnd = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 40;
      const shown = state.lines();
      fill(lines, shown.length ? shown.map((u) => h("li", {
        id: `u-${u.id}`, class: `live-line role-${state.roleOf(u.speaker)}${u.partial ? " partial" : ""}`,
      }, h("span", { class: "who" }, who(u.speaker)), h("span", { class: "text" }, u.text))) : h("li", { class: "hint" }, t("live.waiting")));
      if (atEnd) lines.scrollTop = lines.scrollHeight;
    }

    function point(id) {
      const target = document.getElementById(`u-${id}`);
      if (!target) return;
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      target.classList.add("active");
      setTimeout(() => target.classList.remove("active"), 2000);
    }

    function drawCards() {
      const shown = state.cards().slice(0, 30);
      if (!shown.length) {
        fill(cards, h("p", { class: "hint" }, state.state === "running" ? t("live.no_suggestions") : t("live.none")));
        return;
      }
      const kind = productKind(state.page?.product);
      fill(cards, shown.map((c, i) => {
        const ms = latency(c.record.stages);
        return h("article", { class: `card live-card${i === 0 ? " newest" : ""}` },
          h("div", { class: "row wrap" },
            h("span", { class: "tag accent" }, maybe(`live.kind.${c.suggestion.kind}`, c.suggestion.kind)),
            kind === "interview" && c.suggestion.script_phase ? h("span", { class: "tag" }, maybe(`live.phase.${c.suggestion.script_phase}`, c.suggestion.script_phase)) : null),
          h("p", { class: "live-text" }, c.suggestion.text),
          c.signals.length ? h("ul", { class: "live-signals" }, c.signals.map((s) => h("li", {},
            h("span", { class: "tag warn" }, t(`live.signal.${s.signal}`)), " ",
            h("span", {}, t("live.customer_says")), " ", h("q", {}, s.quote)))) : null,
          c.about.length ? h("p", { class: "hint live-about" }, t("live.about"), " ",
            c.about.map((u) => h("button", { type: "button", class: "ghost small", onClick: () => point(u.id) }, u.text ? `«${u.text.slice(0, 60)}${u.text.length > 60 ? "…" : ""}»` : u.id))) : null,
          c.cites.length ? h("div", { class: "live-cites" }, c.cites.map((ci) => h("details", {},
            h("summary", {}, t("live.source"), " ", h("code", {}, ci.id)),
            h("p", {}, ci.text ?? t("live.source_missing"))))) : null,
          h("p", { class: "meta live-latency" }, t("live.latency", {
            knowledge: ms.knowledge ?? "–", reasoner: ms.reasoner ?? "–", total: ms.total ?? "–",
          })));
      }));
    }

    async function follow() {
      controller = new AbortController();
      while (!closed) {
        try {
          const how = await ctx.api.followLive(session.id, {
            after: state.lastId, signal: controller.signal,
            onItem: (item) => {
              reconnecting = false;
              try {
                state.apply(JSON.parse(item.data), item.id);
              } catch {
                return;
              }
              schedule();
            },
          });
          if (how === "end") break;
        } catch (error) {
          if (error?.kind === "unauthorized") return ctx.fail(error);
        }
        if (closed) break;
        reconnecting = true;
        schedule();
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
        await audio?.stop();
        await ctx.api.stopLive(session.id);
      } catch (error) {
        toast(liveError(error), { error: true });
      }
      stop.disabled = false;
    }

    if (audioKind === "mic") {
      audio = new LiveAudio({
        id: session.id, code: ctx.api.getCode(),
        onProblem: (kind) => toast(maybe(`live.mic.${kind}`, t("live.mic.socket")), { error: true }),
      });
      audio.start().catch(() => {
        toast(t("live.mic.failed"), { error: true });
        ctx.api.stopLive(session.id).catch(() => {});
      });
    }
    draw();
    follow();
  }
}
