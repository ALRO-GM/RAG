// Shared components of the design system (docs/web/DESIGN_SYSTEM.md):
// icons from icons.svg, page headers, empty states, toasts and dialogs.
// Every text comes from the caller (already translated).

import { h, svg } from "./dom.js";
import { t } from "./i18n.js";

/** A line icon from icons.svg (decorative: hidden from screen readers). */
export function icon(name, extra = "") {
  return svg("svg", { class: `icon ${extra}`.trim(), "aria-hidden": "true", focusable: "false" },
    svg("use", { href: `icons.svg#i-${name}` }));
}

/** The title of a page with an optional lead sentence and actions on the right. */
export function pageHead(title, { lead = null, actions = [] } = {}) {
  return h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, title), lead ? h("p", { class: "lead" }, lead) : null),
    actions.length ? h("div", { class: "row wrap" }, actions) : null);
}

/** A friendly "nothing here yet" box: icon, title, a sentence and an optional action. */
export function emptyState({ iconName = "inbox", title, body = null, action = null, tag = "div" }) {
  return h(tag, { class: "empty" }, icon(iconName), h("strong", {}, title), body ? h("p", {}, body) : null, action);
}

function toastHost() {
  let host = document.getElementById("toasts");
  if (!host) {
    host = h("div", { id: "toasts", class: "toasts", role: "status", "aria-live": "polite" });
    document.body.append(host);
  }
  return host;
}

/** A short message in the corner that goes away by itself. */
export function toast(message, { error = false, ms = 4000 } = {}) {
  const item = h("div", { class: error ? "toast error" : "toast" }, message);
  toastHost().append(item);
  setTimeout(() => item.remove(), ms);
  return item;
}

/**
 * A modal dialog with `content` (nodes) and two buttons; resolves true
 * when the person confirms, false when they cancel or press Escape.
 * `onConfirm` may return false to keep the dialog open (a form not valid).
 */
export function openDialog({ title, content = [], confirm, danger = false, onConfirm = () => true }) {
  return new Promise((resolve) => {
    const ok = h("button", { type: "submit", class: danger ? "danger solid" : "primary" }, confirm);
    const cancel = h("button", { type: "button", class: "ghost" }, t("action.cancel"));
    const heading = h("h2", { id: "dialog-title" }, title);
    const form = h("form", { class: "dialog-body", method: "dialog" }, heading, content, h("div", { class: "dialog-actions" }, cancel, ok));
    const dialog = h("dialog", { "aria-labelledby": "dialog-title" }, form);
    let answer = false;
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (onConfirm() === false) return;
      answer = true;
      dialog.close();
    });
    cancel.addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      dialog.remove();
      resolve(answer);
    });
    document.body.append(dialog);
    dialog.showModal();
    (form.querySelector("input, select, textarea") ?? ok).focus();
  });
}
