// «Administración» (#/admin): the state of the service and the tools that
// set the system up (Ajustes, Registrar voz, Conocimiento, Pruebas,
// Estudio, Investigación), kept apart from the three services a worker
// uses. No password of its own for now. Pure helpers in lib/home.js.

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { ADMIN_TOOLS, serviceState } from "../lib/home.js";
import { icon, pageHead } from "../ui.js";

function toolCard(tool) {
  return h("li", {},
    h("a", { class: "tool", href: tool.href },
      h("span", { class: "tool-icon" }, icon(tool.icon)),
      h("span", { class: "tool-text" },
        h("span", { class: "tool-title" }, t(`nav.${tool.id}`)),
        h("span", { class: "tool-body" }, t(`admin.${tool.id}`)))));
}

export async function adminView(ctx, root) {
  root.classList.add("wide");
  const state = h("p", { class: "status-line", role: "status" }, t("admin.checking"));
  const check = h("button", { type: "button", onClick: () => probe() }, t("admin.check"));
  let stopped = false;

  async function probe() {
    check.disabled = true;
    let answer = null;
    try {
      answer = await ctx.api.health();
    } catch {
      answer = null;
    }
    if (stopped) return;
    const ok = serviceState(answer) === "ok";
    fill(state, h("span", { class: `state-dot ${ok ? "done" : "avoid"}`, "aria-hidden": "true" }), t(ok ? "admin.up" : "admin.down"));
    check.disabled = false;
  }

  root.append(
    pageHead(t("nav.admin"), { lead: t("admin.lead") }),
    h("section", { class: "surface status-card", "aria-labelledby": "status-title" },
      h("div", { class: "section-head" }, h("h2", { id: "status-title" }, t("nav.status")), check),
      state),
    h("ul", { class: "tools" }, ADMIN_TOOLS.map(toolCard)));
  probe();
  return () => { stopped = true; };
}
