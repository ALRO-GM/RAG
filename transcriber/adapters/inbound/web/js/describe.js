// The conversation type and goal fields, shared by the recording page,
// the upload form and the session page.

import { h } from "./dom.js";
import { t } from "./i18n.js";
import { describeError } from "./errors.js";
import { GOAL_MAX_LENGTH, SESSION_TYPES, describeBody, describeChanged } from "./lib/describe.js";

/**
 * The two fields, filled from `session` (unset by default). `prefix`
 * keeps their ids unique on the page. `body()` reads them as the API
 * wants them.
 */
export function describeFields(prefix, session = null) {
  const type = h("select", { id: `${prefix}-type` },
    h("option", { value: "" }, t("describe.unset")),
    SESSION_TYPES.map((value) => h("option", { value }, t(`type.${value}`))));
  type.value = session?.type ?? "";
  const goal = h("input", {
    id: `${prefix}-goal`, type: "text", maxLength: GOAL_MAX_LENGTH, autocomplete: "off",
    value: session?.goal ?? "",
  });
  const element = [
    h("div", { class: "field" }, h("label", { for: `${prefix}-type` }, t("describe.type")), type,
      h("p", { class: "hint" }, t("describe.type_hint"))),
    h("div", { class: "field" }, h("label", { for: `${prefix}-goal` }, t("describe.goal")), goal),
  ];
  return { element, type, goal, body: () => describeBody(type.value, goal.value) };
}

/**
 * An editor that saves the fields with PATCH /api/sessions/{id}.
 * `onSaved(view)` gets the session the API answered; an unauthorized
 * answer goes to `ctx.fail`.
 */
export function describeEditor(ctx, session, prefix, onSaved = () => {}) {
  const fields = describeFields(prefix, session);
  let current = session;
  const status = h("p", { class: "hint", role: "status" });
  const save = h("button", { type: "submit", class: "primary" }, t("describe.save"));
  const form = h("form", { class: "stack", onSubmit: async (event) => {
    event.preventDefault();
    const body = fields.body();
    if (!describeChanged(current, body)) {
      status.textContent = t("describe.saved");
      return;
    }
    save.disabled = true;
    status.textContent = t("describe.saving");
    try {
      current = await ctx.api.describeSession(session.id, body);
      status.textContent = t("describe.saved");
      onSaved(current);
    } catch (error) {
      status.textContent = describeError(error);
      if (error?.kind === "unauthorized") ctx.fail(error, () => {});
    }
    save.disabled = false;
  } }, fields.element, save, status);
  return form;
}
