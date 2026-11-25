// #265: the audio of an interview that never reached the server (a long
// outage, a closed or crashed tab, «Terminar» without connection) waits in
// this browser (lib/live-store.js). This card offers to send it («Hay {min}
// min de la entrevista del {fecha} sin enviar», «Enviar» / «Descartar») and
// then follows the recovery on the server: «Pendiente de analizar» →
// «Analizando (n %)» → «Completado (n min recuperados)» or «No se pudo
// recuperar». Each line sits in a role="status" region: text, not colour.
// Used on the «En directo» start form (the next visit) and at the end of a
// session that kept audio here.

import { h } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { openDialog, toast } from "../ui.js";
import {
  IdbBackend, discardRecording, pendingRecordings, recoveryLine, recoveryMoving, uploadRecording,
} from "../lib/live-store.js";

const POLL_MS = 3000;
const RECENT_MS = 2 * 3600 * 1000; // a finished recovery is shown this long

function dateText(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "";
  return new Intl.DateTimeFormat(lang === "en" ? "en-GB" : "es-ES", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })
    .format(when);
}

/** The status line of one recovery (text for a role="status" region). */
export function recoveryText(view) {
  const line = recoveryLine(view);
  return line ? t(line.key, line.params) : "";
}

/**
 * The card, or null when nothing waits here and nothing moves there.
 *   only   a session id: just that session (the end of a session)
 *   deps   { backend, confirm, wait, now } (tests)
 */
export async function recoveryCard(ctx, { only = null, deps = {} } = {}) {
  const backend = deps.backend ?? new IdbBackend();
  const confirm = deps.confirm ?? ((options) => openDialog(options));
  const wait = deps.wait ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
  const now = deps.now ?? Date.now();
  let local = [];
  try {
    local = await pendingRecordings(backend, now);
  } catch {
    local = [];
  }
  let remote = [];
  try {
    remote = (await ctx.api.liveRecoveries())?.recoveries ?? [];
  } catch {
    remote = [];
  }
  const here = new Set(local.map((r) => r.id));
  const shown = remote.filter((r) => !here.has(r.session_id)
    && (recoveryMoving(r) || now - Date.parse(r.updated_at ?? "") < RECENT_MS));
  if (only) {
    local = local.filter((r) => r.id === only);
    shown.splice(0, shown.length, ...shown.filter((r) => r.session_id === only));
  }
  if (!local.length && !shown.length) return null;
  const card = h("section", { class: "card lv-recovery" }, h("h2", {}, t("live.recovery.title")));
  for (const kept of local) card.append(localRow(ctx, kept, { backend, confirm, wait }));
  for (const job of shown) card.append(remoteRow(ctx, job, { wait }));
  return card;
}

function follow(ctx, id, status, wait) {
  (async () => {
    for (;;) {
      let view = null;
      try {
        view = await ctx.api.recoveryStatus(id);
      } catch {
        return;
      }
      const text = recoveryText(view);
      if (status.textContent !== text) status.textContent = text;
      if (!recoveryMoving(view) || !status.isConnected) return;
      await wait(POLL_MS);
    }
  })();
}

function remoteRow(ctx, job, { wait }) {
  const status = h("p", { class: "hint", role: "status" }, recoveryText(job));
  const row = h("div", { class: "lv-recovery-row" },
    h("p", {}, t("live.recovery.of", { date: dateText(job.updated_at) })), status);
  if (recoveryMoving(job)) follow(ctx, job.session_id, status, wait);
  return row;
}

function localRow(ctx, kept, { backend, confirm, wait }) {
  const status = h("p", { class: "hint", role: "status" });
  const send = h("button", { type: "button", class: "primary" }, t("live.recovery.send"));
  const discard = h("button", { type: "button" }, t("live.recovery.discard"));
  const buttons = h("div", { class: "row tight" }, send, discard);
  const row = h("div", { class: "lv-recovery-row" },
    h("p", {}, t("live.recovery.banner", { min: kept.minutes, date: dateText(kept.started_at) })), buttons, status);

  send.addEventListener("click", async () => {
    send.disabled = true;
    discard.disabled = true;
    status.textContent = t("live.recovery.sending", { n: 0 });
    try {
      const view = await uploadRecording({
        backend, api: ctx.api, id: kept.id, wait,
        onProgress: (fraction) => { status.textContent = t("live.recovery.sending", { n: Math.round(fraction * 100) }); },
      });
      buttons.remove();
      status.textContent = recoveryText(view);
      follow(ctx, kept.id, status, wait);
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail?.(error);
      status.textContent = error?.status === 413 ? t("live.recovery.no_room") : describeError(error);
      send.disabled = false;
      discard.disabled = false;
    }
    return null;
  });

  discard.addEventListener("click", async () => {
    const go = await confirm({
      title: t("live.recovery.discard_title"),
      content: [h("p", {}, t("live.recovery.discard_body"))],
      confirm: t("live.recovery.discard"), danger: true,
    });
    if (!go) return;
    try {
      await discardRecording(backend, kept.id);
    } catch (error) {
      toast(describeError(error), { error: true });
      return;
    }
    try {
      await ctx.api.discardRecovery(kept.id);
    } catch (error) {
      if (error?.status !== 404) toast(t("live.recovery.server_later"), { error: true });
    }
    row.remove();
  });
  return row;
}

/**
 * The candidate's file (#265): one line per round whose lost audio is
 * being recovered or was, e.g. «Entrevista del 26 de noviembre:
 * Analizando (40 %)». Filled into `box` once the server answers; nothing
 * when no round has a recovery.
 */
export async function roundRecoveries(api, rounds, box, { wait } = {}) {
  const ids = new Set((rounds ?? []).map((r) => r.session_id).filter(Boolean));
  if (!api?.liveRecoveries || !ids.size) return;
  let jobs = [];
  try {
    jobs = ((await api.liveRecoveries())?.recoveries ?? []).filter((r) => ids.has(r.session_id));
  } catch {
    return;
  }
  if (!jobs.length) return;
  const sleep = wait ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
  box.replaceChildren(h("h3", {}, t("live.recovery.rounds")),
    ...jobs.map((job) => remoteRow({ api }, job, { wait: sleep })));
}
