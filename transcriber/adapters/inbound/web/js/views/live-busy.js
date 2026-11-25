// #263: the one live-session slot is taken. A tab that was closed during a
// session leaves it running for its grace (up to 10 min) and a new tab has
// nothing to reattach to: say which session runs and offer «Volver a la
// sesión» (reattach here, or on the page that owns it) and «Terminarla»
// (stop it; then the new one can start). Used by «En directo» and the
// voice assistant's start forms. When the session's microphone is still
// connected from another browser (someone may be in the middle of an
// interview), «Volver a la sesión» asks first: this page's microphone
// would take its place.

import { h } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { openDialog, toast } from "../ui.js";
import { productKind } from "../lib/live.js";
import { rememberSession, runningSession, sessionPage } from "../lib/live-link.js";

/** The name of a running session for people: its kind of conversation. */
function sessionName(session) {
  const kind = productKind(session?.product);
  return sessionPage(session) === "voicebot" ? t("voicebot.title") : t(`live.product.${kind}`);
}

/**
 * Whether going back to `session` here would take a microphone that is
 * still connected over (unknown counts as connected: ask).
 */
async function takesOver(ctx, session) {
  if (session.audio !== "mic") return false;
  try {
    const info = await ctx.api.liveSource(session.id);
    return info?.source === "connected";
  } catch {
    return true;
  }
}

/**
 * The card for a taken slot, or null when nothing runs (or the list could
 * not be read). `here` is this page ("live" | "voicebot"): a session of
 * this page is reattached with `onBack(session)`, one of the other page is
 * remembered and that page opened. `onEnded()` after «Terminarla».
 */
export async function busyNotice(ctx, {
  here, onBack, onEnded, go = (hash) => { location.hash = hash; }, confirm = (options) => openDialog(options),
}) {
  let sessions = [];
  try {
    sessions = (await ctx.api.liveSessions()).sessions ?? [];
  } catch {
    return null;
  }
  const running = runningSession(sessions);
  if (!running) return null;
  const page = sessionPage(running);
  const status = h("p", { class: "hint", role: "status" });
  const back = h("button", {
    type: "button", class: "primary",
    onClick: async () => {
      if (await takesOver(ctx, running)) {
        const yes = await confirm({
          title: t("live.busy.take_title"), content: [h("p", {}, t("live.busy.take_body"))], confirm: t("live.busy.take_confirm"),
        });
        if (!yes) return;
      }
      if (page === here) {
        card.remove();
        onBack(running);
        return;
      }
      rememberSession(running, running.audio, page);
      go(page === "voicebot" ? "#/voicebot" : "#/live");
    },
  }, t("live.busy.back"));
  const end = h("button", {
    type: "button",
    onClick: async () => {
      const yes = await confirm({
        title: t("live.busy.end_title"), content: [h("p", {}, t("live.busy.end_body"))], confirm: t("live.busy.end"),
      });
      if (!yes) return;
      back.disabled = true;
      end.disabled = true;
      try {
        await ctx.api.stopLive(running.id);
        status.textContent = t("live.busy.ended");
        back.hidden = true;
        end.hidden = true;
        onEnded();
      } catch (error) {
        toast(describeError(error), { error: true });
        back.disabled = false;
        end.disabled = false;
      }
    },
  }, t("live.busy.end"));
  const card = h("section", { class: "card", "aria-label": t("live.busy.title") },
    h("h2", {}, t("live.busy.title")),
    h("p", {}, t("live.busy.body", { name: sessionName(running) })),
    h("div", { class: "row wrap" }, back, end),
    status);
  return card;
}
