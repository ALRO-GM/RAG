// "Pruebas": the model lab. #/lab picks a recording, #/lab/<session> chooses
// models, runs them and follows the runs, #/lab/<session>/<run>[/<section>]
// shows a run's comparison, timeline, fragments to label and scores.

import { h } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import { sortSessions } from "../lib/sessions.js";
import { STAGES, defaultSelection, groupCatalogue, planCounts, runRequest } from "../lib/lab.js";
import { runView } from "./lab-run.js";

const POLL_MS = 2000;
const ACTIVE = new Set(["queued", "running"]);

export async function labView(ctx, root, params) {
  const [session, run, section] = params.map((p) => decodeURIComponent(p));
  if (session && run) return runView(ctx, root, session, run, section || "overview");
  if (session) return chooseView(ctx, root, session);
  return recordingsView(ctx, root);
}

/** A link back to the start of the tab and to the recording. */
export function crumbs(session, run = null) {
  return h("p", { class: "meta crumbs" },
    h("a", { href: "#/lab" }, t("lab.title")),
    session ? [" / ", h("a", { href: `#/lab/${encodeURIComponent(session)}` }, session)] : null,
    run ? [" / ", h("span", {}, t("lab.run_short", { id: run.slice(0, 8) }))] : null);
}

// ---- recordings ----------------------------------------------------------

async function recordingsView(ctx, root) {
  const sessions = sortSessions((await ctx.api.listSessions()).sessions).filter((s) => s.state !== "recording");
  root.append(h("h1", {}, t("lab.title")), h("p", { class: "hint" }, t("lab.intro")));
  if (!sessions.length) {
    root.append(h("p", { class: "notice" }, t("lab.no_recordings")));
    return null;
  }
  root.append(h("h2", {}, t("lab.pick_recording")),
    h("ul", { class: "cards" }, sessions.map((s) => h("li", { class: "card" },
      h("h2", {}, h("a", { href: `#/lab/${encodeURIComponent(s.id)}` }, s.title || s.id)),
      h("p", { class: "meta" },
        s.started_at ? new Date(s.started_at).toLocaleString(lang) : s.id,
        s.duration > 0 ? ` · ${formatTime(s.duration)}` : "")))));
  return null;
}

// ---- choose models, run, follow ------------------------------------------

function modelBadges(model) {
  const badges = [];
  if (model.licence_commercial === false) badges.push(h("span", { class: "badge warn" }, t("lab.non_commercial")));
  if (model.gated) badges.push(h("span", { class: "badge" }, model.state === "needs_token" ? t("lab.needs_token") : t("lab.gated")));
  if (model.state === "not_installed") badges.push(h("span", { class: "badge muted" }, t("lab.not_installed")));
  if (model.state === "ready" && model.weights_cached === false) badges.push(h("span", { class: "badge muted" }, t("lab.downloads")));
  return badges;
}

function modelRow(model, selected, onChange) {
  const id = `lab-m-${model.id}`;
  const box = h("input", { type: "checkbox", id, checked: selected.has(model.id), disabled: model.state !== "ready" });
  box.addEventListener("change", () => {
    if (box.checked) selected.add(model.id);
    else selected.delete(model.id);
    onChange();
  });
  const facts = [model.licence ? t("lab.licence", { licence: model.licence }) : null,
    Number.isFinite(model.vram_gb) ? t("lab.vram", { gb: model.vram_gb }) : null].filter(Boolean).join(" · ");
  return h("li", { class: `model state-${model.state}` },
    h("label", { class: "check", for: id }, box, h("span", {}, model.display_name || model.id)),
    h("div", { class: "row wrap tight" }, modelBadges(model)),
    facts ? h("p", { class: "hint" }, facts) : null,
    model.state !== "ready" && model.reason ? h("p", { class: "hint" }, model.reason) : null,
    model.token_url && model.state === "needs_token"
      ? h("p", { class: "hint" }, h("a", { href: model.token_url, rel: "noopener noreferrer", target: "_blank" }, t("lab.accept_conditions")))
      : null);
}

export function stepLabel(step) {
  return t("lab.step", { stage: t(`lab.stage.${step.stage}`), model: step.model, denoiser: step.denoiser });
}

async function chooseView(ctx, root, session) {
  let catalogue;
  try {
    catalogue = await ctx.api.labCatalogue();
  } catch (error) {
    if (error.kind === "unauthorized" || error.kind === "unavailable" || error.kind === "rate") throw error;
    root.append(crumbs(session), h("p", { class: "error", role: "alert" }, describeError(error)));
    return null;
  }
  const models = catalogue.models;
  const selected = defaultSelection(models);
  const groups = groupCatalogue(models);
  let timer = null;
  let stopped = false;
  let runs = [];

  const counts = h("p", { class: "meta", role: "status" });
  const start = h("button", { type: "button", class: "primary big" }, t("lab.start"));
  const message = h("p", { class: "notice", role: "status", hidden: true });
  const runsList = h("ul", { class: "cards" });
  const live = h("div", { class: "stack" });

  function say(text, isError = false) {
    message.hidden = !text;
    message.textContent = text ?? "";
    message.classList.toggle("error", isError);
  }

  function update() {
    const plan = planCounts(runRequest(selected, models));
    counts.textContent = plan.valid ? t("lab.plan", plan) : t("lab.plan_empty");
    start.disabled = !plan.valid || runs.some((r) => ACTIVE.has(r.state));
  }

  start.addEventListener("click", async () => {
    start.disabled = true;
    say(null);
    try {
      await ctx.api.labStart(session, runRequest(selected, models));
      await refresh();
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error);
      say(describeError(error), true);
      update();
    }
  });

  const stageLists = STAGES.map((stage) => h("fieldset", { class: "field stage" },
    h("legend", {}, t(`lab.stage.${stage}`)),
    h("ul", { class: "plain models" }, groups[stage].map((m) => modelRow(m, selected, update)))));

  root.append(crumbs(session), h("h1", {}, t("lab.choose_title")),
    catalogue.token_present ? null : h("p", { class: "hint" }, t("lab.no_token")),
    h("div", { class: "stages" }, stageLists), counts, start, message, live,
    h("h2", {}, t("lab.runs")), runsList);

  function runCard(run) {
    const href = `#/lab/${encodeURIComponent(session)}/${encodeURIComponent(run.id)}`;
    const ended = !ACTIVE.has(run.state);
    return h("li", { class: `card state-${run.state}` },
      h("h2", {}, ended && run.state !== "failed" ? h("a", { href }, t("lab.run_title", { id: run.id.slice(0, 8) })) : t("lab.run_title", { id: run.id.slice(0, 8) })),
      h("p", { class: "meta" },
        h("span", { class: "badge" }, t(`lab.state.${run.state}`)),
        ` ${new Date(run.created_at).toLocaleString(lang)} · `,
        t("lab.run_counts", { pipelines: run.pipeline_count, done: run.done_steps, total: run.total_steps })),
      run.message ? h("p", { class: "hint" }, run.message) : null,
      h("p", { class: "hint" }, [run.request.denoisers.join(", "), run.request.diarizers.join(", "), run.request.recognisers.join(", ")].filter(Boolean).join(" · ")),
      h("div", { class: "row wrap" },
        ended && run.state !== "failed" ? h("a", { class: "button", href }, t("lab.see_results")) : null,
        !ended ? h("button", { type: "button", onClick: () => act(() => ctx.api.labCancel(session, run.id)) }, t("lab.cancel")) : null,
        ended && run.state !== "done" ? h("button", { type: "button", onClick: () => act(() => ctx.api.labResume(session, run.id)) }, t("lab.resume")) : null));
  }

  async function act(call) {
    say(null);
    try {
      await call();
      await refresh();
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error);
      say(describeError(error), true);
    }
  }

  async function showLive(run) {
    let status;
    try {
      status = await ctx.api.labStatus(session, run.id);
    } catch (error) {
      live.replaceChildren(h("p", { class: "error" }, describeError(error)));
      return;
    }
    const job = status.job;
    const fraction = job && Number.isFinite(job.fraction) ? job.fraction : null;
    live.replaceChildren(h("section", { class: "card stack tight", "aria-live": "polite" },
      h("h2", {}, t("lab.progress_title")),
      h("div", { class: "progress" },
        h("progress", { max: status.total_steps || 1, value: status.done_steps, "aria-label": t("lab.progress_title") }),
        h("span", {}, t("lab.progress", { done: status.done_steps, total: status.total_steps }))),
      job?.stage ? h("p", { class: "hint" }, t("lab.now", { stage: job.stage, pct: fraction === null ? "" : ` ${Math.round(fraction * 100)}%` })) : null,
      h("ol", { class: "steps" }, status.steps.map((step) => h("li", { class: `step state-${step.state}` },
        h("span", { class: "badge" }, t(`lab.step_state.${step.state}`)), " ", stepLabel(step),
        step.cached ? h("span", { class: "hint" }, ` · ${t("lab.cached")}`) : null,
        Number.isFinite(step.seconds) ? h("span", { class: "hint" }, ` · ${step.seconds.toFixed(1)} s`) : null,
        step.error ? h("p", { class: "error" }, `${step.error.code}: ${step.error.message}`) : null)))));
  }

  async function refresh() {
    if (stopped) return;
    clearTimeout(timer);
    try {
      runs = (await ctx.api.labRuns(session)).runs;
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error);
      say(describeError(error), true);
      timer = setTimeout(refresh, POLL_MS * 3);
      return;
    }
    runsList.replaceChildren(...(runs.length ? runs.map(runCard) : [h("li", { class: "hint" }, t("lab.no_runs"))]));
    const active = runs.find((r) => ACTIVE.has(r.state));
    if (active) {
      await showLive(active);
      timer = setTimeout(refresh, POLL_MS);
    } else {
      live.replaceChildren();
    }
    update();
  }

  update();
  await refresh();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
