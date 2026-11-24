// «En directo» (#/live): live assist during a support call or an interview.
// Start a live session (stereo channels from the workstation, or this
// browser's microphone), replay a recorded conversation, or watch the
// scripted demo interview (#/live/demo); then follow the orchestrator's
// stream. The state lives in lib/live.js (LiveState); this module only
// draws it.
//
// LV-17 (#232), the calm page (docs/web/mockups/Live.dc.html): the
// conversation on one side, muted for the employee («Tú»), typing in as it
// is heard; on the other one «Ahora» card with one thing at a time (a line
// better avoided, else the newest suggestion not used yet, with «Hecho» or
// «Reformulado» once done); the topics as a quiet row of chips; what must
// be said as another row; earlier suggestions with their sources, and the
// options of LV-16 (which lines are shown, how many earlier suggestions
// stay open, the assistance mode), under two disclosures. Nothing is drawn
// twice: lines, chips and the card are kept by key, so only what is new
// fades in. Customer signals appear only in support calls; an interview
// never shows signals or ratings about the candidate.

import { append, h, fill, svg } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { setLeaveGuard } from "../device.js";
import { LiveAudio } from "../live-audio.js";
import {
  AUDIO, LiveState, MAX_SUGGESTIONS, MODES, SPEEDS, clockText, followsNewest, holdLines, latency, localized, newSessionId, productKind,
  readPrefs, reminderDue, splitCards, viewPrefs, voiceProgress, writePrefs,
} from "../lib/live.js";
import { emptyState, icon, openDialog, pageHead, toast } from "../ui.js";
import { coreCount, coreRows } from "../lib/candidate.js";
import { demoCardExtras, pickDemo, resultLink } from "../lib/call-demo.js";
import { ConnectionWatch, WATCH, needsOf } from "../lib/connection.js";
import { pingOnce, servicesOnce, watchLoop } from "../connection.js";
import { confirmStep } from "./candidate.js";
import { confirmStart, connectionCard, connectionChip, connectionNote, lastLinkLevel } from "./connection.js";

const RETRY_MS = 2000;
const TICK_MS = 1000;
const REMIND_MS = 3000;
const RESTART_TRIES = 12;
const RESTART_WAIT_MS = 250;
const TYPE_OF = { support: "soporte", interview: "entrevista" };
const BACK_SHOWN_MS = 30000; // «La conexión ha vuelto…» stays this long

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

/** How long the voice chip says a correction was taken. */
const CORRECTED_MS = 4000;

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function dot(kind, extra = "") {
  return h("span", { class: `state-dot ${kind} ${extra}`.trim(), "aria-hidden": "true" });
}

export async function liveView(ctx, root, [sub, which] = []) {
  root.classList.add("wide");
  let products = [];
  let transcripts = [];
  let demos = [];
  try {
    const [listed, recorded] = await Promise.all([ctx.api.liveProducts(), ctx.api.liveTranscripts()]);
    products = listed.products.filter((p) => productKind(p.product)); // the voice bot has its own page
    transcripts = recorded.transcripts ?? [];
    demos = recorded.demos ?? [];
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("live.title"), { lead: t("live.lead") }),
      emptyState({ iconName: "live", title: t("live.unavailable"), body: liveError(error) }));
    return null;
  }
  // The enrolled people (to choose the employee) and the saved «Ajustes»: both optional.
  const [people, saved] = await Promise.all([
    Promise.resolve(ctx.api.speakers?.()).then((r) => r?.speakers ?? []).catch(() => []),
    Promise.resolve(ctx.api.pipelineSettings?.()).catch(() => null),
  ]);
  const prefs = viewPrefs(saved?.choices ?? {}, readPrefs());

  let closed = false;
  let controller = null;
  let audio = null;
  let running = false;
  let guarded = false; // a demo can be left at any time; a real session asks first
  let ticker = null;
  let playing = null; // the id of a demo session this page started
  setLeaveGuard(() => running && guarded);

  function cleanup() {
    closed = true;
    // A demo left behind would hold the one session slot until it ends.
    if (playing && running) ctx.api.stopLive(playing).catch(() => {});
    controller?.abort();
    audio?.stop();
    clearInterval(ticker);
    setLeaveGuard(() => false);
  }

  const wanted = pickDemo(demos, which);
  if (sub === "demo" && wanted) startDemo(wanted);
  else showForm(sub === "demo" ? t("live.scripted.none") : null);
  return cleanup;

  // ---- start ----------------------------------------------------------

  async function startDemo(demo) {
    for (let i = 0; i < RESTART_TRIES; i += 1) {
      try {
        // A few random letters: two demos started within one second get different ids.
        const id = `${newSessionId("demo")}-${Math.random().toString(36).slice(2, 6) || "x"}`;
        const session = await ctx.api.startReplay(id, { demo: demo.id, speed: 1 });
        if (!closed) showSession(session, null);
        return null;
      } catch (error) {
        if (error?.kind === "unauthorized") return ctx.fail(error);
        if (!["busy", "wrong_state"].includes(error?.code) || i === RESTART_TRIES - 1) {
          if (!closed) showForm(liveError(error));
          return null;
        }
        await wait(RESTART_WAIT_MS); // the session before is still ending
      }
    }
    return null;
  }

  function showForm(message = null) {
    clearInterval(ticker);
    root.replaceChildren();
    const productSelect = h("select", { id: "live-product" },
      products.map((p) => h("option", { value: p.product }, t(`live.product.${productKind(p.product)}`))));
    const audioSelect = h("select", { id: "live-audio" }, AUDIO.map((a) => h("option", { value: a }, t(`live.audio.${a}`))));
    const employeeSelect = h("select", { id: "live-employee" },
      h("option", { value: "" }, t("live.employee_none")),
      people.map((p) => h("option", { value: p.id }, p.name || p.id)));
    const employeeHint = h("p", { class: "hint" },
      people.length ? t("live.employee_hint") : t("live.employee_missing"), " ",
      h("a", { href: "#/enrol" }, t("live.employee_enrol")));
    const transcriptSelect = h("select", { id: "live-transcript" },
      transcripts.map((x) => h("option", { value: x.id }, `${x.title} · ${t("live.utterances", { n: x.utterances })}`)));
    const speedSelect = h("select", { id: "live-speed" }, SPEEDS.map((s) => h("option", { value: String(s) }, `${s}×`)));
    const banner = h("p", { class: "notice info", role: "note" });
    const liveHint = h("p", { class: "hint" });
    const start = h("button", { type: "button", class: "primary big", onClick: () => begin("live") }, icon("live"), t("live.start"));
    const replay = h("button", { type: "button", onClick: () => begin("replay") }, icon("play"), t("live.demo"));
    const formError = h("p", { class: "error", role: "alert", hidden: !message }, message ?? "");
    // «Calidad de la conexión» (#247): checked when the form opens; a poor
    // result asks before a live session starts, it never blocks it.
    const connection = connectionCard({
      getCode: () => ctx.api.getCode?.() ?? null,
      options: () => ({ needs: needsOf("live"), upload: audioSelect.value === "mic", liveAudio: Boolean(chosen()?.live) }),
    });
    // An interview may name the role (an interview plan) and the candidate
    // (#235): their answers to the core questions are kept across rounds.
    const planSelect = h("select", { id: "live-plan" }, h("option", { value: "" }, t("live.plan_none")));
    const candidateSelect = h("select", { id: "live-candidate" }, h("option", { value: "" }, t("live.candidate_new")));
    const candidateName = h("input", { id: "live-candidate-name", type: "text", maxlength: "80", autocomplete: "off", placeholder: t("live.candidate_name_hint") });
    const candidateHint = h("p", { class: "hint" }, t("live.candidate_hint"));
    const nameField = h("div", { class: "field" }, h("label", { for: "live-candidate-name" }, t("live.candidate_name")), candidateName);
    const roleBox = h("div", { class: "stack tight lv-role-box", hidden: true },
      h("div", { class: "field" }, h("label", { for: "live-plan" }, t("live.plan")), planSelect),
      h("div", { class: "field" }, h("label", { for: "live-candidate" }, t("live.candidate")), candidateSelect),
      nameField, candidateHint);
    let plansLoaded = false;

    async function loadPlans() {
      if (plansLoaded) return;
      plansLoaded = true;
      try {
        const found = await ctx.api.interviewPlans();
        for (const plan of found?.plans ?? []) planSelect.append(h("option", { value: plan.id }, plan.title || plan.id));
      } catch (error) {
        if (error?.kind === "unauthorized") ctx.fail(error);
      }
      syncRole();
    }

    async function loadCandidates() {
      fill(candidateSelect, h("option", { value: "" }, t("live.candidate_new")));
      if (!planSelect.value) return syncRole();
      try {
        const found = await ctx.api.planCandidates(planSelect.value);
        for (const c of found?.candidates ?? []) {
          candidateSelect.append(h("option", { value: c.id }, t("live.candidate_option", { name: c.name, done: c.answered, all: (c.questions ?? []).length })));
        }
      } catch (error) {
        if (error?.kind === "unauthorized") ctx.fail(error);
      }
      return syncRole();
    }

    function syncRole() {
      const on = Boolean(planSelect.value);
      candidateSelect.disabled = !on;
      candidateName.disabled = !on || Boolean(candidateSelect.value);
      nameField.hidden = Boolean(candidateSelect.value);
    }

    function roleChoice() {
      if (roleBox.hidden || !planSelect.value) return {};
      if (candidateSelect.value) return { plan: planSelect.value, candidate_id: candidateSelect.value };
      const name = candidateName.value.trim();
      return name ? { plan: planSelect.value, candidate: name } : { plan: planSelect.value };
    }

    function chosen() {
      return products.find((p) => p.product === productSelect.value) ?? products[0];
    }

    function employee() {
      const person = people.find((p) => p.id === employeeSelect.value);
      return person ? { id: person.id, name: person.name || person.id } : undefined;
    }

    function sync() {
      const product = chosen();
      const kind = productKind(product?.product);
      banner.textContent = maybe(`live.banner.${product?.banner}`, t("live.banner.default"));
      start.disabled = !product?.live;
      audioSelect.disabled = !product?.live;
      // One microphone, two people (#249): the session starts with a voice check.
      const micCheck = audioSelect.value === "mic" && product?.two_party === true && saved?.choices?.["live.diarization"] !== "off";
      liveHint.textContent = !product?.live ? t("live.live_off")
        : micCheck ? t("live.audio_hint.mic_check")
          : t(`live.audio_hint.${audioSelect.value}`, { who: roleText(product.suggest_about?.[0] ?? "unknown") });
      transcriptSelect.value = defaultTranscript(transcripts, kind);
      replay.disabled = !transcripts.length;
      roleBox.hidden = kind !== "interview" || !product?.live;
      if (!roleBox.hidden) loadPlans();
    }

    async function begin(mode) {
      if (mode === "live") {
        const go = await confirmStart({
          card: connection,
          onWait: (on) => {
            start.disabled = on;
            if (on) start.replaceChildren(icon("live"), t("conn.checking"));
            else start.replaceChildren(icon("live"), t("live.start"));
          },
          ask: (verdict) => openDialog({
            title: t("conn.start_anyway_title"),
            content: [h("p", {}, t("conn.start_anyway_body")), h("p", { class: "hint" }, t(`conn.advice.${verdict.advice}`))],
            confirm: t("conn.start_anyway"),
          }),
        });
        if (!go || closed) {
          start.disabled = !chosen()?.live;
          return null;
        }
      }
      formError.hidden = true;
      start.disabled = true;
      replay.disabled = true;
      const product = chosen().product;
      try {
        let session;
        if (mode === "live") {
          session = await ctx.api.startLive({
            product, audio: audioSelect.value, session_id: newSessionId("live"), employee: employee(), ...roleChoice(),
          });
        } else {
          session = await ctx.api.startReplay(newSessionId("demo"), {
            product, transcript: transcriptSelect.value, speed: Number(speedSelect.value), employee: employee(),
          });
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
    for (const select of [productSelect, audioSelect]) select.addEventListener("change", () => connection.redraw());
    employeeSelect.addEventListener("change", sync);
    planSelect.addEventListener("change", loadCandidates);
    candidateSelect.addEventListener("change", syncRole);
    const field = (input, key) => h("div", { class: "field" }, h("label", { for: input.id }, t(key)), input);
    const scripted = pickDemo(demos);
    const supportDemo = pickDemo(demos, "support");
    // append() of the h helper skips null (the DOM's own would write "null").
    append(root, [
      pageHead(t("live.title"), { lead: t("live.lead") }),
      formError,
      scripted ? h("section", { class: "card lv-demo-card" },
        h("div", { class: "lv-demo-text" },
          h("h2", {}, t("live.scripted.title")),
          h("p", { class: "hint" }, t("live.scripted.body"))),
        h("button", { type: "button", class: "primary big", onClick: () => startDemo(scripted) },
          icon("play"), t("live.scripted.start"))) : null,
      supportDemo ? h("section", { class: "card lv-demo-card" },
        h("div", { class: "lv-demo-text" },
          h("h2", {}, t("live.support_demo.title")),
          h("p", { class: "hint" }, t("live.support_demo.body"))),
        h("button", { type: "button", class: "primary big", onClick: () => startDemo(supportDemo) },
          icon("play"), t("live.support_demo.start"))) : null,
      h("div", { class: "live-start" },
        connection.node,
        h("section", { class: "card lv-new" },
          h("h2", {}, t("live.new")),
          field(productSelect, "live.product"), banner,
          field(employeeSelect, "live.employee"), employeeHint, roleBox,
          field(audioSelect, "live.audio"), liveHint, start),
        h("details", { class: "card lv-recorded" },
          h("summary", {}, t("live.recorded_title")),
          h("p", { class: "hint" }, t("live.demo_body")),
          field(transcriptSelect, "live.transcript"), field(speedSelect, "live.speed"), replay)),
    ]);
    sync();
    connection.check();
  }

  // ---- a running session ----------------------------------------------

  function showSession(session, audioKind) {
    clearInterval(ticker);
    controller?.abort();
    root.replaceChildren();
    const state = new LiveState();
    state.apply({ type: "session", session });
    running = true;
    guarded = session.mode !== "demo";
    playing = session.mode === "demo" ? session.id : null;
    const demo = state.demo;
    const result = resultLink(demo);
    const resultHref = result?.href ?? null;

    // ---- header: clock, what this is, who, and the actions
    const clock = h("span", { class: "lv-clock" });
    const liveDot = dot("avoid", "lv-pulse");
    const stop = h("button", { type: "button", onClick: () => end() }, icon("stop"), t("live.stop"));
    const again = h("button", { type: "button", class: "lv-icon-button", "aria-label": t("live.replay_again"), title: t("live.replay_again"), onClick: () => restart() },
      icon("play"), h("span", { class: "lv-button-text" }, t("live.replay_again")));
    const toResult = resultHref ? h("a", {
      class: "button primary", href: resultHref,
      onClick: (e) => {
        if (state.state !== "running") return;
        e.preventDefault();
        ctx.api.stopLive(session.id).catch(() => {}).finally(() => { location.hash = resultHref; });
      },
    }, t("live.end_result")) : null;
    const fresh = h("button", { type: "button", class: "primary", onClick: () => showForm() }, t("live.again"));
    const otherName = h("span", { class: "lv-other" });
    // Talk-time balance (#235): one quiet bar, no alert, drawn once a sentence is final.
    const balance = h("span", { class: "lv-balance", hidden: true });
    // The connection (#247): a quiet chip that changes only after a sustained problem.
    const connChip = connectionChip();
    // The employee's voice check (#249): «Reconociendo tu voz…» → «Voz reconocida», quietly.
    const voiceChip = h("span", { class: "lv-voice", hidden: true, role: "status" });
    let corrected = null; // { fix, until }: the last correction, said once in the chip
    const header = h("header", { class: "lv-head" },
      h("div", { class: "lv-meta" },
        h("span", { class: "lv-live" }, liveDot, clock),
        h("strong", { class: "lv-title" }, demo ? localized(demo, "title", lang) : t(`live.product.${productKind(state.page?.product)}`)),
        otherName,
        demo ? h("span", { class: "type-chip" }, t("live.scripted.badge")) : null,
        balance, voiceChip, connChip.node),
      h("div", { class: "lv-actions" }, demo ? again : null, demo ? toResult : stop, demo ? null : fresh));

    // ---- the conversation
    const lines = h("ol", { class: "lv-lines", "aria-label": t("live.talk"), tabindex: "0" });
    // Stick to the newest line unless the user scrolled back to read.
    let followLines = true;
    lines.addEventListener("scroll", () => { followLines = followsNewest(lines); }, { passive: true });
    const speaking = h("p", { class: "lv-speaking", "aria-live": "polite" });
    const talk = h("section", { class: "lv-talk" }, h("h2", { class: "lv-label" }, t("live.talk")), lines, speaking);

    // ---- the side: Ahora, topics, what to say, earlier, options
    const discomfort = h("div", { "aria-live": "polite" });
    const nowCard = h("div", { class: "lv-now", "aria-live": "polite" });
    const topics = h("section", { class: "lv-row", hidden: true });
    const coreRow = h("section", { class: "lv-row lv-core", hidden: true });
    const confirmBox = h("div", {}); // «Confirmar datos» once the session ends
    const checklist = h("section", { class: "lv-row lv-tosay", hidden: true });
    const historyBody = h("div", { class: "lv-history" });
    const history = h("details", { class: "lv-more" }, h("summary", {}, t("live.history")), historyBody);
    const status = h("div", { class: "row wrap live-status", role: "status" });
    const alerts = h("div", { class: "stack tight lv-alerts", "aria-live": "polite" });
    const banner = h("p", { class: "hint" }, maybe(`live.banner.${state.page?.banner}`, t("live.banner.default")));

    const showEmployee = h("input", { type: "checkbox", id: "live-show-employee", checked: prefs.showEmployee });
    const showOther = h("input", { type: "checkbox", id: "live-show-other", checked: prefs.showOther });
    const maxSelect = h("select", { id: "live-max" },
      MAX_SUGGESTIONS.map((n) => h("option", { value: String(n), selected: n === prefs.max }, String(n))));
    for (const input of [showEmployee, showOther, maxSelect]) {
      input.addEventListener("change", () => {
        prefs.showEmployee = showEmployee.checked;
        prefs.showOther = showOther.checked;
        prefs.max = Number(maxSelect.value);
        writePrefs(prefs);
        historyKey = null;
        schedule();
      });
    }
    const modeButtons = MODES.map((m) => h("button", {
      type: "button", class: "small", "data-mode": m, "aria-pressed": "false",
      onClick: () => switchMode(m),
    }, t(`live.assist.${m}`)));
    const modeGroup = h("div", { class: "live-modes", role: "group", "aria-label": t("live.assist_label") },
      h("span", { class: "meta" }, t("live.assist_label")), modeButtons);
    const options = h("details", { class: "lv-more" },
      h("summary", {}, t("live.options")),
      h("div", { class: "lv-options" },
        demo ? h("p", { class: "hint" }, localized(demo, "note", lang) || t("live.scripted.note")) : null,
        banner, status,
        h("label", { class: "check" }, showEmployee, t("live.show_employee")),
        h("label", { class: "check" }, showOther, t("live.show_other")),
        h("label", { class: "row tight live-max", for: "live-max" }, t("live.max_suggestions"), maxSelect),
        modeGroup));
    const summary = h("div", {});
    const side = h("aside", { class: "lv-side" },
      h("h2", { class: "lv-label" }, t("live.now")), discomfort, nowCard, confirmBox, topics, coreRow, checklist, history, options, summary);

    // Errors and one-time notices stay in sight, under the header; only the choices are in «Opciones».
    // Lost / back (#247): one persistent live region, changed only when that state changes.
    const note = connectionNote();
    root.append(header, note.node, alerts, h("div", { class: "lv-body" }, talk, side));

    let drawing = false;
    let reconnecting = false;
    const kind = session.mode === "demo" ? "demo" : session.mode === "live" ? "live" : "replay";
    const watch = new ConnectionWatch({ initial: lastLinkLevel(), needs: needsOf(kind) });
    let lastReminder = Date.now();
    let remind = false;
    let editing = null; // the speaker whose name is being typed
    let nowKey = null;
    let historyKey = null;
    const lineNodes = new Map(); // utterance id -> { li, label, labelKey, text, flags }
    const chipNodes = new Map(); // topic id -> { chip, done }

    function schedule() {
      if (drawing) return;
      drawing = true;
      (globalThis.requestAnimationFrame ?? setTimeout)(() => {
        drawing = false;
        draw();
      });
    }

    async function switchMode(mode) {
      if (state.mode === mode || state.state !== "running") return;
      const before = state.mode;
      state.mode = mode;
      schedule();
      try {
        await ctx.api.liveMode(session.id, mode);
      } catch (error) {
        state.mode = before;
        toast(liveError(error), { error: true });
        schedule();
      }
    }

    function otherSpeaker() {
      return [...state.speakers.keys()].find((s) => state.sideOf(s) === "other") ?? null;
    }

    function draw() {
      const page = state.page ?? {};
      const live = state.state === "running";
      clock.textContent = t(live ? "live.clock" : "live.clock_ended", { time: clockText(state.streamTime) });
      liveDot.hidden = !live;
      connChip.node.hidden = !live;
      connChip.set(watch.state);
      const other = otherSpeaker();
      otherName.textContent = demo?.candidate ?? (other ? state.nameOf(other) ?? "" : "");
      stop.hidden = !live;
      fresh.hidden = live;
      if (toResult) toResult.textContent = result.kind === "report" ? t(live ? "live.report_end" : "live.report_see") : live ? t("live.end_result") : t("live.see_result", { name: demo.candidate });
      const counts = state.counts();
      fill(status,
        h("span", { class: `badge is-${live ? "processing" : state.state === "failed" ? "failed" : "indexed"}` }, t(`live.state.${state.state}`)),
        h("span", { class: "tag" }, session.mode === "live" ? t(`live.audio.${audioKind ?? "stereo"}`) : t("live.mode.replay")),
        state.employee ? h("span", { class: "tag" }, icon("enrol"), " ", state.nameOf(state.employee.speaker) ?? state.employee.name) : null,
        h("span", { class: "meta" }, t("live.calls", { ok: counts.ok ?? 0, all: state.records.length })));
      modeGroup.hidden = !(page.modes ?? []).length;
      for (const button of modeButtons) {
        const on = button.dataset.mode === (state.mode ?? MODES[0]);
        button.setAttribute("aria-pressed", String(on));
        button.classList.toggle("primary", on);
        button.disabled = !live;
      }
      const notes = [];
      if (state.halted) notes.push(h("p", { class: "notice error", role: "alert" }, t(`live.halted.${state.halted}`)));
      if (state.state === "failed") notes.push(h("p", { class: "notice error", role: "alert" }, t("live.failed", { error: state.error ?? "" })));
      if (!live) note.set(null);
      // The persistent «connection lost» line says it already; no second one.
      if (reconnecting && live && note.kind !== "lost") notes.push(h("p", { class: "notice" }, t("live.reconnecting")));
      for (const w of state.warnings.slice(-3)) notes.push(h("p", { class: "notice" }, maybe(`live.warning.${w.code}`, w.message)));
      for (const n of state.notices.slice(-2)) notes.push(h("p", { class: "notice" }, maybe(`live.notice.${n.code}`, n.message)));
      const voice = state.voiceStatus();
      if (["short", "failed"].includes(voice) && live) notes.push(h("p", { class: "notice" }, t(`live.voice.${voice}`)));
      fill(alerts, notes);
      drawVoice(voice, live);
      drawDiscomfort();
      drawLines();
      drawNow();
      drawTopics();
      drawCore();
      drawBalance(other);
      drawChecklist();
      drawHistory();
      drawSummary();
    }

    // ---- conversation lines, kept by id so only new ones fade in

    function nameTag(speaker) {
      const employee = state.sideOf(speaker) === "employee";
      const name = state.nameOf(speaker);
      if (editing === speaker) {
        const input = h("input", { type: "text", class: "live-name-input", value: name ?? "", maxlength: "60", "aria-label": t("live.rename_label") });
        let done = false;
        const save = async () => {
          if (done) return;
          done = true;
          const value = input.value.trim();
          editing = null;
          if (value && value !== name) {
            state.names.set(speaker, value);
            try {
              await ctx.api.renameLiveSpeaker(session.id, speaker, value);
            } catch (error) {
              toast(liveError(error), { error: true });
            }
          }
          schedule();
        };
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") {
            done = true;
            editing = null;
            schedule();
          }
        });
        input.addEventListener("blur", save);
        setTimeout(() => input.focus(), 0);
        return input;
      }
      const label = employee ? t("live.you") : name || roleText(state.roleOf(speaker));
      if (state.state !== "running" || demo) return h("span", { class: "lv-name" }, label);
      return h("button", {
        type: "button", class: "lv-name live-who", title: t("live.rename_label"),
        onClick: () => {
          editing = speaker;
          schedule();
        },
      }, label);
    }

    // «Esta soy yo» / «Esta no soy yo» (#249): corrects that voice from now on and its earlier lines.
    // Named with its line (aria-describedby: the line's text); after a
    // correction the header's quiet voice chip (role status) says so once.
    function correctButton(u, fix) {
      const button = h("button", {
        type: "button", class: "lv-fix", "data-fix": fix, "aria-describedby": `lv-text-${u.id}`,
        onClick: async () => {
          button.disabled = true;
          try {
            await ctx.api.liveVoice(session.id, { action: fix, utterance: u.id });
            corrected = { fix, until: Date.now() + CORRECTED_MS };
            schedule();
            setTimeout(schedule, CORRECTED_MS + 50);
          } catch (error) {
            toast(liveError(error), { error: true });
          }
          button.disabled = false;
        },
      }, t(`live.voice.${fix}`));
      return button;
    }

    function lineNode(u) {
      let node = lineNodes.get(u.id);
      if (!node) {
        const label = h("span", { class: "lv-line-head" });
        const text = h("span", { class: "lv-text", id: `lv-text-${u.id}` });
        const li = h("li", { class: "lv-line lv-rise", id: `u-${u.id}` }, label, text);
        node = { li, label, labelKey: null, text, flags: "" };
        lineNodes.set(u.id, node);
      }
      const side = state.sideOf(u.speaker);
      node.li.classList.toggle("is-employee", side === "employee");
      node.li.classList.toggle("is-other", side === "other");
      node.li.classList.toggle("partial", Boolean(u.partial));
      node.li.classList.toggle("discomfort", u.id === state.discomfort()?.utterance);
      if (node.text.textContent !== u.text) node.text.textContent = u.text;
      node.li.classList.toggle("is-unsure", Boolean(u.unsure) && !u.partial);
      const flags = state.flagsOf(u.id);
      const fix = demo ? null : state.correctionFor(u);
      const key = `${u.speaker}|${state.nameOf(u.speaker)}|${editing === u.speaker}|${state.state}|${flags.map((f) => f.category).join(",")}|${fix}|${Boolean(u.unsure)}`;
      if (key !== node.labelKey) {
        node.labelKey = key;
        const isNew = flags.length && !node.flags;
        node.flags = flags.map((f) => f.category).join(",");
        fill(node.label, nameTag(u.speaker),
          u.unsure && !u.partial ? h("span", { class: "lv-unsure" }, t("live.voice.unsure")) : null,
          flags.length ? h("span", { class: `lv-flag${isNew ? " lv-rise" : ""}`, title: flags.map((f) => localized(f, "title", lang)).join(" · ") },
            dot("avoid"), t("live.avoid")) : null,
          fix ? correctButton(u, fix) : null);
      }
      return node.li;
    }

    function drawLines() {
      if (editing && holdLines(document.activeElement)) return; // keep the name being typed
      const all = state.lines();
      const shown = state.visibleLines(prefs);
      talk.hidden = !prefs.showEmployee && !prefs.showOther;
      if (!shown.length) {
        lineNodes.clear();
        fill(lines, h("li", { class: "hint lv-empty" }, all.length ? t("live.lines_hidden") : t("live.waiting")));
      } else {
        lines.querySelector(".lv-empty")?.remove();
        const wanted = shown.map(lineNode);
        const keep = new Set(wanted);
        for (const child of [...lines.children]) if (!keep.has(child)) child.remove();
        for (const [id, node] of lineNodes) if (!keep.has(node.li)) lineNodes.delete(id);
        wanted.forEach((li, i) => {
          if (lines.children[i] !== li) lines.insertBefore(li, lines.children[i] ?? null);
        });
      }
      if (followLines) lines.scrollTop = lines.scrollHeight;
      const who = state.state === "running" ? state.speaking() : null;
      const other = otherSpeaker();
      const said = !who ? "" : who === "employee" ? t("live.speaking_you")
        : other && state.nameOf(other) ? t("live.speaking_other", { name: state.nameOf(other).split(" ")[0] }) : t("live.speaking_someone");
      if (said !== speaking.textContent) fill(speaking, said ? [dot("suggest", "lv-pulse"), said] : []);
    }

    // ---- the «Ahora» card: one thing at a time

    function drawVoice(voice, live) {
      const fixed = live && corrected && Date.now() < corrected.until ? corrected.fix : null;
      const shown = live && (fixed || ["recognising", "recognised", "skipped"].includes(voice));
      voiceChip.hidden = !shown;
      if (!shown) return;
      const text = fixed ? t(`live.voice.done_${fixed}`) : t(`live.voice.chip_${voice}`);
      if (voiceChip.textContent === text) return;
      fill(voiceChip, fixed || voice === "recognised" ? icon("check") : voice === "recognising" ? dot("suggest", "lv-pulse") : null, text);
      voiceChip.classList.toggle("is-done", Boolean(fixed) || voice === "recognised");
    }

    // While the voice check listens, it is the one thing «Ahora» shows.
    // Built once per state; the progress moves in place, so a screen reader
    // does not read the card again and «Saltar» keeps its focus.
    let voiceBar = null;
    function voiceCard(voice) {
      const bar = h("span", { class: "lv-voice-bar" });
      const skip = h("button", {
        type: "button", class: "small",
        onClick: async () => {
          skip.disabled = true;
          try {
            await ctx.api.liveVoice(session.id, { action: "skip" });
          } catch (error) {
            skip.disabled = false;
            toast(liveError(error), { error: true });
          }
        },
      }, t("live.voice.skip"));
      return h("div", { class: "lv-card lv-rise lv-voice-card" },
        h("div", { class: "lv-card-head is-suggest" }, dot("suggest", "lv-pulse"),
          voice === "recognising" ? t("live.voice.recognising") : t("live.voice.listening_head")),
        h("p", { class: "lv-big" }, t("live.voice.listening")),
        h("p", { class: "lv-example" }, t("live.voice.listening_hint")),
        (voiceBar = { bar, track: h("span", { class: "lv-voice-track", role: "progressbar", "aria-label": t("live.voice.progress"),
          "aria-valuemin": "0", "aria-valuemax": "100" }, bar) }).track,
        h("div", { class: "row wrap" }, voice === "listening" ? skip : null,
          h("span", { class: "hint" }, t("live.voice.private"))));
    }

    function drawNow() {
      const ended = state.state !== "running" && demo;
      const voice = state.state === "running" ? state.voiceStatus() : null;
      if (voice === "listening" || voice === "recognising") {
        const key = `voice|${voice}`;
        if (key !== nowKey) {
          nowKey = key;
          fill(nowCard, voiceCard(voice));
          nowCard.className = "lv-now";
        }
        const done = String(Math.round(voiceProgress(state.voice) * 100));
        if (voiceBar && voiceBar.track.getAttribute("aria-valuenow") !== done) {
          voiceBar.track.setAttribute("aria-valuenow", done);
          voiceBar.bar.style.width = `${done}%`;
        }
        return;
      }
      voiceBar = null;
      const shown = state.now();
      const key = ended ? "ended"
        : shown.kind === "avoid" ? `avoid|${shown.flag.utterance}|${shown.flag.category}`
          : shown.kind === "suggestion" ? `card|${shown.card.record.call}|${shown.card.suggestion.text}` : "idle";
      if (key !== nowKey) {
        nowKey = key;
        fill(nowCard, ended ? endCard() : shown.kind === "avoid" ? avoidCard(shown.flag)
          : shown.kind === "suggestion" ? suggestionCard(shown.card) : h("p", { class: "lv-idle lv-rise" }, t("live.now_idle")));
        nowCard.className = `lv-now${shown.kind === "avoid" && !ended ? " is-avoid" : ""}`;
      }
      const badge = nowCard.querySelector(".lv-badge");
      if (badge) badge.hidden = !(shown.kind === "avoid" ? shown.done : shown.used);
    }

    function suggestionCard(card) {
      const guided = card.record?.mode === "guiado" && !card.suggestion.example;
      return h("div", { class: "lv-card lv-rise" },
        h("div", { class: "lv-card-head is-suggest" }, dot("suggest"), t("live.now_suggestion"),
          h("span", { class: "lv-badge done lv-rise", hidden: true }, icon("check"), t("live.now_used"))),
        guided ? h("span", { class: "live-say-label" }, t("live.say")) : null,
        h("p", { class: "lv-big" }, localized(card.suggestion, "text", lang)),
        card.suggestion.example ? h("p", { class: "lv-example" }, t("live.now_example", { text: localized(card.suggestion, "example", lang) })) : null,
        demo ? callSupport(card) : null);
    }

    // A support call's «Ahora» card (#250): what the customer expressed, quoted, and the
    // source in the knowledge base the suggestion comes from.
    function callSupport(card) {
      const extras = demoCardExtras(card, demo, state.discomfort()?.record);
      const signalsRow = extras.signals.length ? h("ul", { class: "lv-signals" }, extras.signals.map((s) => h("li", {},
        h("span", { class: "tag warn" }, t(`live.signal.${s.signal}`)), " ", h("span", {}, t("live.customer_says")), " ", h("q", {}, s.quote)))) : null;
      const sources = extras.sources.length ? h("div", { class: "lv-sources" }, extras.sources.map((c) => h("p", { class: "lv-source" },
        h("span", { class: "lv-source-label" }, t("live.source_kb")), " ", c.text ?? t("live.source_missing"),
        lang === "en" ? h("span", { class: "meta" }, " ", t("live.source_es")) : null))) : null;
      return signalsRow || sources ? h("div", { class: "lv-support" }, signalsRow, sources) : null;
    }

    function avoidCard(flag) {
      return h("div", { class: "lv-card lv-rise" },
        h("div", { class: "lv-card-head is-avoid" }, dot("avoid"), `${t("live.avoid")} · ${localized(flag, "title", lang)}`,
          h("span", { class: "lv-badge done lv-rise", hidden: true }, icon("check"), t("live.avoid_done"))),
        h("p", { class: "lv-big" }, localized(flag, "why", lang)),
        h("div", { class: "lv-alt" }, h("span", { class: "lv-alt-label" }, t("live.avoid_alternative")), h("span", {}, localized(flag, "alternative", lang))));
    }

    function endCard() {
      return h("div", { class: "lv-card lv-rise" },
        h("div", { class: "lv-card-head is-done" }, dot("done"), t(result?.kind === "report" ? "live.report_ended_title" : "live.ended_title")),
        h("p", { class: "lv-big" }, resultHref ? t(result.kind === "report" ? "live.report_ended" : "live.ended_body") : t("live.ended_plain")),
        h("div", { class: "row wrap" },
          resultHref ? h("a", { class: "button primary", href: resultHref }, result.kind === "report" ? t("live.report_see") : t("live.see_result", { name: demo.candidate }), icon("arrow")) : null,
          h("button", { type: "button", onClick: () => restart() }, icon("play"), t("live.replay_again"))));
    }

    // ---- quiet rows of chips: the topics and what must be said

    function chip(done, label, extra = "") {
      return h("span", { class: `lv-chip${done ? " done" : ""} ${extra}`.trim() },
        done ? icon("check") : h("span", { class: "lv-ring", "aria-hidden": "true" }), label);
    }

    function drawTopics() {
      const chips = state.topicChips();
      topics.hidden = !chips.length;
      if (!chips.length) return;
      if (!topics.firstChild) {
        topics.append(h("div", { class: "lv-row-head" }, h("h3", {}), h("span", { class: "meta lv-count" })),
          h("div", { class: "lv-chips" }));
      }
      // A later round of a candidate: the chips start from what earlier rounds left.
      const rounds = chips.some((c) => ["weak", "revisit", "covered"].includes(c.earlier));
      topics.querySelector("h3").textContent = rounds ? t("live.topics_next") : state.topics ? t("live.topics") : t("live.topics_phases");
      topics.querySelector(".lv-count").textContent = t("live.topics_count", { done: chips.filter((c) => c.done).length, all: chips.length });
      const row = topics.querySelector(".lv-chips");
      const wanted = chips.map((c) => {
        const name = c.name ? localized(c, "name", lang) : maybe(`live.phase.${c.phase}`, c.phase);
        const key = `${c.id}|${name}`;
        const kept = chipNodes.get(key);
        if (kept && kept.done === c.done) return kept.chip;
        const earlier = c.earlier && rounds ? ` is-${c.earlier}` : "";
        const made = chip(c.done, name, `${kept ? "lv-rise" : ""}${earlier}`.trim());
        if (earlier) made.title = t(`live.earlier.${c.earlier}`);
        if (kept) kept.chip.replaceWith(made);
        chipNodes.set(key, { chip: made, done: c.done });
        return made;
      });
      const keep = new Set(wanted);
      for (const child of [...row.children]) if (!keep.has(child)) child.remove();
      for (const [id, kept] of chipNodes) if (!keep.has(kept.chip)) chipNodes.delete(id);
      wanted.forEach((node, i) => {
        if (row.children[i] !== node) row.insertBefore(node, row.children[i] ?? null);
      });
    }

    let balanceKey = null;
    function drawBalance(other) {
      const share = state.talkBalance();
      balance.hidden = !share;
      if (!share) return;
      const who = (demo?.candidate ?? (other ? state.nameOf(other) : null)) || t(`live.role.${state.roleOf(other ?? "") || "unknown"}`);
      const you = Math.round(share.employee);
      const key = `${you}|${who}`;
      if (key === balanceKey) return;
      balanceKey = key;
      const label = t("live.talk_label", { you, who, other: 100 - you });
      fill(balance,
        svg("svg", { class: "lv-balance-svg", viewBox: "0 0 100 8", preserveAspectRatio: "none", role: "img", "aria-label": label, focusable: "false" },
          svg("rect", { class: "lv-balance-you", x: 0, y: 0, width: Math.max(you, 0.5), height: 8, rx: 2 }),
          svg("rect", { class: "lv-balance-other", x: you, y: 0, width: Math.max(100 - you, 0.5), height: 8, rx: 2 })),
        h("span", { "aria-hidden": "true" }, t("live.talk_you", { value: you }), " · ", t("live.talk_other", { who, value: 100 - you })));
      balance.title = t("live.talk_balance");
    }

    // The core questions (#235): asked once per candidate; an answered one
    // shows its answer and is never suggested again.
    let coreKey = null;
    function drawCore() {
      const profile = state.core;
      coreRow.hidden = !profile;
      if (!profile) return;
      const rows = coreRows(t, profile, lang);
      const key = JSON.stringify([state.state, rows.map((r) => [r.id, r.text, r.proposal])]);
      if (key === coreKey) return;
      coreKey = key;
      const count = coreCount(profile);
      fill(coreRow,
        h("div", { class: "lv-row-head" }, h("h3", {}, t("live.core")),
          h("span", { class: "meta lv-count" }, t("live.core_count", { done: count.done, all: count.all }))),
        h("div", { class: "lv-chips" }, rows.map((r) => {
          const label = r.answered && r.text ? `${r.label}: ${r.text.length > 32 ? `${r.text.slice(0, 31)}…` : r.text}`
            : r.proposed ? t("live.core_proposed", { what: r.label }) : r.label;
          const made = chip(r.answered, label, r.answered ? "" : r.proposed ? "is-open is-proposed" : "is-open");
          made.title = r.answered ? `${r.label}: ${r.text}` : r.proposed ? t("live.core_proposed_title", { value: r.proposal }) : r.question;
          return made;
        })),
        h("p", { class: "hint" }, t("live.core_hint", { name: profile.name })));
      drawConfirm();
    }

    // «Confirmar datos» (#235): once the session ends, what was read from
    // the conversation is confirmed, edited or discarded here; nothing
    // counts as answered before.
    function drawConfirm() {
      const profile = state.core;
      const step = profile && state.state !== "running"
        ? confirmStep(profile, { api: ctx.api, onChange: (changed) => { state.core = changed; coreKey = null; drawCore(); } })
        : null;
      fill(confirmBox, step ? h("section", { class: "lv-row lv-confirm" }, step) : null);
    }

    function drawChecklist() {
      const items = state.checklist?.items ?? [];
      checklist.hidden = !items.length;
      if (!items.length) return;
      const pending = state.pendingItems();
      checklist.classList.toggle("remind", remind && pending.length > 0);
      fill(checklist,
        h("div", { class: "lv-row-head" }, h("h3", {}, t("live.to_say")),
          h("span", { class: "meta" }, pending.length ? t("live.to_say_count", { n: pending.length }) : t("live.to_say_done"))),
        remind && pending.length ? h("p", { class: "hint", role: "status" }, t("live.to_say_remind", { items: pending.map((i) => i.text).join(" · ") })) : null,
        h("div", { class: "lv-chips" }, items.map((item) => (item.state === "pending" && state.state === "running"
          ? h("button", { type: "button", class: "lv-chip is-button", title: t("live.mandatory_tick"), onClick: () => tick(item) },
            h("span", { class: "lv-ring", "aria-hidden": "true" }), item.text)
          : chip(item.state !== "pending", item.text, item.state === "ticked" ? "ticked" : "")))));
    }

    async function tick(item) {
      const yes = await openDialog({
        title: t("live.mandatory_confirm_title"),
        content: [h("p", {}, t("live.mandatory_confirm_body")), h("blockquote", {}, item.text)],
        confirm: t("live.mandatory_confirm"),
      });
      if (!yes) return;
      try {
        const view = await ctx.api.tickMandatory(session.id, item.id);
        if (view?.items) state.checklist = view;
        schedule();
      } catch (error) {
        toast(liveError(error), { error: true });
      }
    }

    // ---- earlier suggestions, their sources and the lines better avoided

    function point(id) {
      const target = document.getElementById(`u-${id}`);
      if (!target) return;
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      target.classList.add("active");
      setTimeout(() => target.classList.remove("active"), 2000);
    }

    function earlierCard(c) {
      const ms = latency(c.record.stages);
      const kind = productKind(state.page?.product);
      return h("article", { class: "lv-earlier" },
        h("div", { class: "row wrap" },
          h("span", { class: "tag accent" }, maybe(`live.kind.${c.suggestion.kind}`, c.suggestion.kind)),
          kind === "interview" && c.suggestion.script_phase ? h("span", { class: "tag" }, maybe(`live.phase.${c.suggestion.script_phase}`, c.suggestion.script_phase)) : null,
          state.used(c) ? h("span", { class: "lv-badge done" }, icon("check"), t("live.now_used")) : null),
        h("p", { class: "live-text" }, localized(c.suggestion, "text", lang)),
        c.suggestion.example ? h("p", { class: "hint" }, t("live.now_example", { text: localized(c.suggestion, "example", lang) })) : null,
        c.signals.length ? h("ul", { class: "live-signals" }, c.signals.map((s) => h("li", {},
          h("span", { class: "tag warn" }, t(`live.signal.${s.signal}`)), " ",
          h("span", {}, t("live.customer_says")), " ", h("q", {}, s.quote)))) : null,
        c.about.length ? h("p", { class: "hint live-about" }, t("live.about"), " ",
          c.about.map((u) => h("button", { type: "button", class: "ghost small", onClick: () => point(u.id) }, u.text ? `«${u.text.slice(0, 60)}${u.text.length > 60 ? "…" : ""}»` : u.id))) : null,
        c.cites.length ? h("div", { class: "live-cites" }, c.cites.map((ci) => h("details", {},
          h("summary", {}, t("live.source"), " ", h("code", {}, ci.id)),
          h("p", {}, ci.text ?? t("live.source_missing"))))) : null,
        ms.total !== null ? h("p", { class: "meta live-latency" }, t("live.latency", {
          knowledge: ms.knowledge ?? "–", reasoner: ms.reasoner ?? "–", total: ms.total ?? "–",
        })) : null);
    }

    function drawHistory() {
      const earlier = state.earlier().slice(0, 30);
      const key = [prefs.max, ...earlier.map((c) => `${c.record.call}:${state.used(c)}`),
        ...state.avoid.map((f) => `${f.utterance}:${f.state}`)].join("|");
      if (key === historyKey) return;
      historyKey = key;
      const { shown, older } = splitCards(earlier, prefs.max);
      const avoided = state.avoid;
      fill(historyBody,
        earlier.length ? null : h("p", { class: "hint" }, t("live.history_empty")),
        shown.map(earlierCard),
        older.length ? h("details", { class: "live-older" },
          h("summary", {}, t("live.older", { n: older.length })), older.map(earlierCard)) : null,
        avoided.length ? h("section", { class: "lv-avoided" },
          h("h3", {}, t("live.history_avoided")),
          h("p", { class: "hint" }, t("live.avoid_hint")),
          avoided.map((f) => h("article", { class: "lv-earlier" },
            h("div", { class: "row wrap" }, h("span", { class: "lv-flag" }, dot("avoid"), `${t("live.avoid")} · ${localized(f, "title", lang)}`),
              f.state === "rephrased" ? h("span", { class: "lv-badge done" }, icon("check"), t("live.avoid_done")) : null),
            state.utterances.get(f.utterance) ? h("button", { type: "button", class: "ghost small", onClick: () => point(f.utterance) },
              `«${state.utterances.get(f.utterance).text}»`) : null,
            h("p", {}, localized(f, "why", lang)), h("p", { class: "hint" }, localized(f, "alternative", lang))))) : null);
    }

    function drawDiscomfort() {
      const found = state.discomfort();
      if (!found) {
        fill(discomfort);
        return;
      }
      fill(discomfort, h("p", { class: "lv-note", role: "status" },
        dot("avoid"), h("strong", {}, t("live.discomfort")), " ",
        found.signals.map((s) => h("span", { class: "tag warn" }, t(`live.signal.${s}`))), " ",
        found.quote ? h("q", {}, found.quote) : null));
    }

    function drawSummary() {
      const handoff = state.handoff;
      const mandatory = handoff?.mandatory;
      if (!handoff || state.state === "running") {
        fill(summary);
        return;
      }
      const missing = mandatory?.missing ?? [];
      const names = Object.entries(handoff.names ?? {});
      fill(summary, h("section", { class: "card live-summary" },
        h("h2", {}, t("live.summary_title")),
        mandatory ? (missing.length
          ? h("div", { class: "notice", role: "status" },
            h("strong", {}, t("live.summary_missing", { n: missing.length })),
            h("ul", {}, missing.map((m) => h("li", {}, m.text))))
          : h("p", { class: "notice info" }, t("live.summary_all"))) : null,
        mandatory?.ticked?.length ? h("p", { class: "hint" }, t("live.summary_ticked", { n: mandatory.ticked.length })) : null,
        names.length ? h("p", { class: "hint" }, t("live.summary_names"), " ",
          names.map(([s, n]) => `${n} (${roleText(state.roleOf(s))})`).join(", ")) : null));
    }

    // ---- the stream and the actions

    async function follow() {
      const mine = new AbortController();
      controller = mine;
      while (!closed && !mine.signal.aborted) {
        try {
          const how = await ctx.api.followLive(session.id, {
            after: state.lastId, signal: mine.signal,
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
        if (closed || mine.signal.aborted) return null;
        reconnecting = true;
        schedule();
        await wait(RETRY_MS);
      }
      if (mine.signal.aborted) return null;
      running = false;
      if (state.state === "running") state.state = "ended";
      await audio?.stop();
      audio = null;
      schedule();
      return null;
    }

    // One small ping every WATCH.everyMs (sooner after a failure), the
    // services every WATCH.servicesEvery pings; stops with the session.
    function watchConnection() {
      const alive = () => !closed && controller === mine && state.state === "running";
      return watchLoop({
        watch, alive, wait,
        ping: () => pingOnce({ timeoutMs: WATCH.timeoutMs }),
        services: () => servicesOnce({ getCode: () => ctx.api.getCode?.() ?? null }),
        onChange: (seen) => {
          if (seen.lost) note.set("lost");
          if (seen.back) {
            const at = Date.now();
            note.set("back", at);
            setTimeout(() => { if (note.key === `back:${at}`) note.set(null); }, BACK_SHOWN_MS);
          }
          schedule();
        },
      });
    }

    async function end() {
      const pending = state.pendingItems();
      if (pending.length) {
        const go = await openDialog({
          title: t("live.stop_pending_title"),
          content: [h("p", {}, t("live.stop_pending_body")), h("ul", { class: "live-pending-list" }, pending.map((i) => h("li", {}, i.text)))],
          confirm: t("live.stop_anyway"),
        });
        if (!go) return;
      }
      stop.disabled = true;
      try {
        await audio?.stop();
        await ctx.api.stopLive(session.id);
      } catch (error) {
        toast(liveError(error), { error: true });
      }
      stop.disabled = false;
    }

    async function restart() {
      if (!demo) return;
      again.disabled = true;
      controller?.abort();
      if (state.state === "running") await ctx.api.stopLive(session.id).catch(() => {});
      running = false;
      await startDemo(demo);
    }

    // The clock moves every second; unsaid items come back every remind_seconds, quietly.
    ticker = setInterval(() => {
      if (state.state === "running") schedule();
      const pending = state.pendingItems();
      if (state.state !== "running" || !reminderDue(pending.length, state.checklist?.remind_seconds, lastReminder, Date.now())) return;
      lastReminder = Date.now();
      remind = true;
      schedule();
      setTimeout(() => {
        remind = false;
        schedule();
      }, REMIND_MS);
    }, TICK_MS);

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
    const mine = controller;
    watchConnection();
  }
}
