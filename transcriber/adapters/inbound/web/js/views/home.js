// The home (#/): three large service cards (analyse a conversation, live
// assist, voice bot), each with an icon, one sentence and «Probar»; below,
// the latest sessions, each with its type and, when its analysis wrote a
// synthesis, the one-line conclusion of it and, when the synthesis has an
// overall reading (#230), its percentage with one decimal. Pure helpers in lib/home.js.

import { h, fill } from "../dom.js";
import { t, lang, decimal } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import { typeLabel } from "../lib/describe.js";
import { SERVICES, analysisFacts, latestSessions, sessionHref } from "../lib/home.js";
import { emptyState, icon } from "../ui.js";

function serviceCard(service) {
  const id = `svc-${service.id}`;
  return h("li", { class: `service service-${service.id}` },
    h("span", { class: "service-icon" }, icon(service.icon)),
    h("h2", { id }, t(`home.${service.id}.title`)),
    h("p", {}, t(`home.${service.id}.body`)),
    h("div", { class: "service-links" },
      h("a", { class: "button primary service-go", href: service.href, "aria-describedby": id },
        t("home.try"), icon("arrow")),
      service.demo ? h("a", { class: "service-demo", href: service.demo }, icon("play"), t(`home.${service.id}.demo`)) : null));
}

// The overall percentage: seen as «67,5 %», read out as «Valoración general 67,5 %».
function scoreTag(value) {
  const shown = t("sf.percent", { value: decimal(value, 1) });
  return h("span", { class: "recent-score" },
    h("span", { "aria-hidden": "true" }, shown),
    h("span", { class: "sr-only" }, t("sf.home_score", { value: shown })));
}

function sessionRow(session, found) {
  const kind = typeLabel(t, session.type);
  const when = session.started_at ? new Date(session.started_at).toLocaleDateString(lang, { day: "numeric", month: "short" }) : "";
  const facts = [when, session.duration > 0 ? formatTime(session.duration) : ""].filter(Boolean).join(" · ");
  const line = found?.conclusion
    ? [found.conclusion.person ? h("strong", {}, `${found.conclusion.person}: `) : null, found.conclusion.text]
    : [t(`state.${session.state}`), facts ? ` · ${facts}` : ""];
  return h("li", {},
    h("a", { class: "recent-row", href: sessionHref(session, found?.analysed) },
      h("span", { class: "recent-text" },
        h("span", { class: "recent-head" },
          h("span", { class: "recent-title" }, session.title || session.id),
          kind ? h("span", { class: "type-chip" }, kind) : null),
        h("span", { class: "recent-line" }, line)),
      typeof found?.conclusion?.percent === "number" ? scoreTag(found.conclusion.percent) : null,
      found?.analysed ? h("span", { class: "recent-tag" }, h("span", { class: "state-dot done", "aria-hidden": "true" }), t("home.analysed")) : null));
}

export async function homeView(ctx, root) {
  root.classList.add("wide");
  let stopped = false;
  const list = h("ul", { class: "recent-list" });
  const recent = h("section", { class: "recent", "aria-labelledby": "recent-title" },
    h("div", { class: "section-head" },
      h("h2", { id: "recent-title" }, t("home.recent")),
      h("a", { href: "#/sessions", class: "see-all" }, t("home.see_all"))),
    list);

  root.append(
    h("div", { class: "home-head" },
      h("h1", {}, t("home.title")),
      h("p", { class: "lead" }, t("home.lead"))),
    h("ul", { class: "services" }, SERVICES.map(serviceCard)),
    recent);

  let sessions = [];
  try {
    sessions = latestSessions((await ctx.api.listSessions()).sessions);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    fill(list, h("li", {}, h("p", { class: "error" }, describeError(error))));
    return () => { stopped = true; };
  }
  if (!sessions.length) {
    fill(list, emptyState({ tag: "li", iconName: "inbox", title: t("sessions.empty"), body: t("sessions.empty_body") }));
    return () => { stopped = true; };
  }
  const found = new Map();
  const draw = () => fill(list, sessions.map((s) => sessionRow(s, found.get(s.id))));
  draw();
  Promise.all(sessions.map(async (s) => found.set(s.id, await analysisFacts(ctx.api, s)))).then(() => {
    if (!stopped) draw();
  });
  return () => { stopped = true; };
}
