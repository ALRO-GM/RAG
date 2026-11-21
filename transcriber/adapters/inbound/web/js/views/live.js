// «En directo» (#/live): live assist during a support call or an interview.
// Start a live session (stereo channels from the workstation, or this
// browser's microphone) or a demo that replays a recorded conversation;
// then follow the orchestrator's stream: the transcript with speakers and
// partials, suggestion cards with their sources (knowledge chunks) and the
// sentences they are about, the script-phase coverage of interviews and the
// notice banner. Customer signals appear only in support calls; an
// interview never shows signals or ratings about the candidate. The state
// lives in lib/live.js (LiveState); this module only draws it.
//
// LV-16 (#206): the enrolled employee chosen at the start, names edited
// inline, which side's lines are shown, the newest suggestions open (the
// rest collapsed), the assistance mode (pincelada / guiado) switched live,
// the mandatory items in red until they are said, and the customer's
// discomfort highlighted (support only).

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { setLeaveGuard } from "../device.js";
import { LiveAudio } from "../live-audio.js";
import {
  AUDIO, LiveState, MAX_SUGGESTIONS, MODES, SPEEDS, latency, newSessionId, productKind,
  readPrefs, reminderDue, splitCards, viewPrefs, writePrefs,
} from "../lib/live.js";
import { emptyState, icon, openDialog, pageHead, toast } from "../ui.js";

const RETRY_MS = 2000;
const TICK_MS = 1000;
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
    products = products.filter((p) => productKind(p.product)); // the voice bot has its own page
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
  let ticker = null;
  setLeaveGuard(() => running);

  function cleanup() {
    closed = true;
    controller?.abort();
    audio?.stop();
    clearInterval(ticker);
    setLeaveGuard(() => false);
  }

  showForm();
  return cleanup;

  // ---- start ----------------------------------------------------------

  function showForm() {
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
    const demo = h("button", { type: "button", class: "big", onClick: () => begin("replay") }, icon("play"), t("live.demo"));
    const formError = h("p", { class: "error", role: "alert", hidden: true });

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
      const micEmployee = audioSelect.value === "mic" && employeeSelect.value;
      liveHint.textContent = !product?.live ? t("live.live_off")
        : micEmployee ? t("live.audio_hint.mic_employee")
          : t(`live.audio_hint.${audioSelect.value}`, { who: roleText(product.suggest_about?.[0] ?? "unknown") });
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
          session = await ctx.api.startLive({ product, audio: audioSelect.value, session_id: newSessionId("live"), employee: employee() });
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
    employeeSelect.addEventListener("change", sync);
    const field = (input, key) => h("div", { class: "field" }, h("label", { for: input.id }, t(key)), input);
    root.append(
      pageHead(t("live.title"), { lead: t("live.lead") }),
      h("div", { class: "live-start" },
        h("section", { class: "card" },
          h("h2", {}, t("live.new")),
          field(productSelect, "live.product"), banner,
          field(employeeSelect, "live.employee"), employeeHint,
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
    const discomfort = h("div", { "aria-live": "polite" });
    const phases = h("ol", { class: "live-phases", "aria-label": t("live.phases") });
    const checklist = h("section", { class: "live-mandatory", "aria-label": t("live.mandatory_title"), hidden: true });
    const lines = h("ol", { class: "live-lines", "aria-label": t("live.transcript_title") });
    const cards = h("div", { class: "live-cards" });
    const summary = h("div", {});
    const banner = h("p", { class: "notice info", role: "note" }, maybe(`live.banner.${state.page?.banner}`, t("live.banner.default")));

    // View choices: remembered in this browser, defaults from «Ajustes».
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
        schedule();
      });
    }
    const modeButtons = MODES.map((m) => h("button", {
      type: "button", class: "small", "data-mode": m, "aria-pressed": "false",
      onClick: () => switchMode(m),
    }, t(`live.assist.${m}`)));
    const modeGroup = h("div", { class: "live-modes", role: "group", "aria-label": t("live.assist_label") },
      h("span", { class: "meta" }, t("live.assist_label")), modeButtons);
    const controls = h("div", { class: "row wrap live-controls" },
      h("label", { class: "check" }, showEmployee, t("live.show_employee")),
      h("label", { class: "check" }, showOther, t("live.show_other")),
      h("label", { class: "row tight live-max", for: "live-max" }, t("live.max_suggestions"), maxSelect),
      modeGroup);

    root.append(
      pageHead(t("live.title"), { actions: [stop, again] }),
      banner, status, controls, alerts, discomfort, checklist, phases,
      h("div", { class: "live-layout" },
        h("section", { class: "live-pane" }, h("h2", {}, t("live.transcript_title")), lines),
        h("section", { class: "live-pane" }, h("h2", {}, t("live.suggestions_title")), cards)),
      summary);
    let drawing = false;
    let reconnecting = false;
    let lastReminder = Date.now();
    let remind = false;
    let editing = null; // the speaker whose name is being typed

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

    function draw() {
      const page = state.page ?? {};
      const kind = productKind(page.product);
      const counts = state.counts();
      fill(status,
        h("span", { class: `badge is-${state.state === "running" ? "processing" : state.state === "failed" ? "failed" : "indexed"}` }, t(`live.state.${state.state}`)),
        h("span", { class: "tag accent" }, t(`live.product.${kind}`)),
        h("span", { class: "tag" }, session.mode === "replay" ? t("live.mode.replay") : t(`live.audio.${audioKind ?? "stereo"}`)),
        state.employee ? h("span", { class: "tag" }, icon("enrol"), " ", state.nameOf(state.employee.speaker) ?? state.employee.name) : null,
        h("span", { class: "meta" }, t("live.calls", { ok: counts.ok ?? 0, all: state.records.length })));
      modeGroup.hidden = !(page.modes ?? []).length;
      for (const button of modeButtons) {
        const on = button.dataset.mode === (state.mode ?? MODES[0]);
        button.setAttribute("aria-pressed", String(on));
        button.classList.toggle("primary", on);
        button.disabled = state.state !== "running";
      }
      const notes = [];
      if (state.halted) notes.push(h("p", { class: "notice error", role: "alert" }, t(`live.halted.${state.halted}`)));
      if (state.state === "failed") notes.push(h("p", { class: "notice error", role: "alert" }, t("live.failed", { error: state.error ?? "" })));
      if (reconnecting && state.state === "running") notes.push(h("p", { class: "notice" }, t("live.reconnecting")));
      for (const w of state.warnings.slice(-3)) notes.push(h("p", { class: "notice" }, maybe(`live.warning.${w.code}`, w.message)));
      for (const n of state.notices.slice(-2)) notes.push(h("p", { class: "notice" }, maybe(`live.notice.${n.code}`, n.message)));
      fill(alerts, notes);
      drawDiscomfort();
      drawChecklist();
      const coverage = state.coverage();
      phases.hidden = !coverage.length;
      fill(phases, coverage.map((c) => h("li", { class: `live-phase is-${c.state}` },
        maybe(`live.phase.${c.phase}`, c.phase), c.count ? h("span", { class: "meta" }, ` ${c.count}`) : null)));
      drawLines();
      drawCards();
      drawSummary();
      stop.hidden = state.state !== "running";
      again.hidden = state.state === "running";
    }

    function drawDiscomfort() {
      const found = state.discomfort();
      if (!found) {
        fill(discomfort);
        return;
      }
      fill(discomfort, h("p", { class: "notice live-discomfort", role: "alert" },
        h("strong", {}, t("live.discomfort")), " ",
        found.signals.map((s) => h("span", { class: "tag warn" }, t(`live.signal.${s}`))), " ",
        found.quote ? h("q", {}, found.quote) : null));
    }

    function drawChecklist() {
      const items = state.checklist?.items ?? [];
      checklist.hidden = !items.length;
      if (!items.length) return;
      const pending = state.pendingItems();
      checklist.classList.toggle("is-done", !pending.length);
      checklist.classList.toggle("remind", remind && pending.length > 0);
      fill(checklist,
        h("h2", {}, pending.length ? t("live.mandatory_pending", { n: pending.length }) : t("live.mandatory_done")),
        h("ul", {}, items.map((item) => h("li", { class: `live-item is-${item.state}` },
          h("span", { class: "live-item-mark", "aria-hidden": "true" }, item.state === "pending" ? "!" : "✓"),
          h("span", { class: "live-item-text" }, item.text),
          item.state === "said" ? h("span", { class: "meta" }, t("live.mandatory_said")) : null,
          item.state === "ticked" ? h("span", { class: "meta" }, t("live.mandatory_ticked")) : null,
          item.state === "pending" && state.state === "running"
            ? h("button", { type: "button", class: "ghost small", onClick: () => tick(item) }, t("live.mandatory_tick"))
            : null))));
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

    function nameTag(speaker) {
      const role = roleText(state.roleOf(speaker));
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
        return h("span", { class: "who" }, input, ` · ${role}`);
      }
      const label = name ? `${name} · ${role}` : role;
      if (state.state !== "running") return h("span", { class: "who" }, label);
      return h("button", {
        type: "button", class: "who live-who", title: t("live.rename_label"),
        onClick: () => {
          editing = speaker;
          drawLines(true);
        },
      }, label, " ", icon("edit"));
    }

    function drawLines(force = false) {
      if (editing && !force) return; // keep the name being typed
      const atEnd = lines.scrollHeight - lines.scrollTop - lines.clientHeight < 40;
      const all = state.lines();
      const shown = state.visibleLines(prefs);
      const upset = state.discomfort()?.utterance;
      let body;
      if (!all.length) body = h("li", { class: "hint" }, t("live.waiting"));
      else if (!shown.length) body = h("li", { class: "hint" }, t("live.lines_hidden"));
      else {
        body = shown.map((u) => h("li", {
          id: `u-${u.id}`,
          class: `live-line role-${state.roleOf(u.speaker)} side-${state.sideOf(u.speaker)}${u.partial ? " partial" : ""}${u.id === upset ? " discomfort" : ""}`,
        }, nameTag(u.speaker), h("span", { class: "text" }, u.text)));
      }
      fill(lines, body);
      lines.closest(".live-pane").hidden = !prefs.showEmployee && !prefs.showOther;
      if (atEnd) lines.scrollTop = lines.scrollHeight;
    }

    function point(id) {
      const target = document.getElementById(`u-${id}`);
      if (!target) return;
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      target.classList.add("active");
      setTimeout(() => target.classList.remove("active"), 2000);
    }

    function card(c, i, upset) {
      const ms = latency(c.record.stages);
      const kind = productKind(state.page?.product);
      const guided = (state.page?.modes ?? []).length > 0 && (state.mode ?? MODES[0]) === "guiado";
      const worried = upset && c.record === upset.record;
      return h("article", { class: `card live-card${i === 0 ? " newest" : ""}${worried ? " discomfort" : ""}` },
        h("div", { class: "row wrap" },
          h("span", { class: "tag accent" }, maybe(`live.kind.${c.suggestion.kind}`, c.suggestion.kind)),
          kind === "interview" && c.suggestion.script_phase ? h("span", { class: "tag" }, maybe(`live.phase.${c.suggestion.script_phase}`, c.suggestion.script_phase)) : null),
        guided
          ? h("div", { class: "live-say" }, h("span", { class: "live-say-label" }, t("live.say")), h("p", { class: "live-say-text" }, c.suggestion.text))
          : h("p", { class: "live-text" }, c.suggestion.text),
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
    }

    function drawCards() {
      const all = state.cards().slice(0, 30);
      if (!all.length) {
        fill(cards, h("p", { class: "hint" }, state.state === "running" ? t("live.no_suggestions") : t("live.none")));
        return;
      }
      const upset = state.discomfort();
      const { shown, older } = splitCards(all, prefs.max);
      fill(cards, shown.map((c, i) => card(c, i, upset)),
        older.length ? h("details", { class: "live-older" },
          h("summary", {}, t("live.older", { n: older.length })),
          older.map((c, i) => card(c, i + shown.length, upset))) : null);
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
          ? h("div", { class: "notice error", role: "alert" },
            h("strong", {}, t("live.summary_missing", { n: missing.length })),
            h("ul", {}, missing.map((m) => h("li", {}, m.text))))
          : h("p", { class: "notice info" }, t("live.summary_all"))) : null,
        mandatory?.ticked?.length ? h("p", { class: "hint" }, t("live.summary_ticked", { n: mandatory.ticked.length })) : null,
        names.length ? h("p", { class: "hint" }, t("live.summary_names"), " ",
          names.map(([s, n]) => `${n} (${roleText(state.roleOf(s))})`).join(", ")) : null));
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
      clearInterval(ticker);
      if (state.state === "running") state.state = "ended";
      await audio?.stop();
      audio = null;
      schedule();
      return null;
    }

    async function end() {
      const pending = state.pendingItems();
      if (pending.length) {
        const go = await openDialog({
          title: t("live.stop_pending_title"),
          content: [h("p", {}, t("live.stop_pending_body")), h("ul", { class: "live-pending-list" }, pending.map((i) => h("li", {}, i.text)))],
          confirm: t("live.stop_anyway"), danger: true,
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

    // Insist on the unsaid items: every remind_seconds they flash and a toast names them.
    ticker = setInterval(() => {
      const pending = state.pendingItems();
      if (state.state !== "running" || !reminderDue(pending.length, state.checklist?.remind_seconds, lastReminder, Date.now())) return;
      lastReminder = Date.now();
      remind = true;
      toast(t("live.mandatory_remind", { items: pending.map((i) => i.text).join(" · ") }), { error: true, ms: 6000 });
      schedule();
      setTimeout(() => {
        remind = false;
        schedule();
      }, 3000);
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
  }
}
