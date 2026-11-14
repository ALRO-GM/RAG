// A consent notice as the API gives it: title, points, version, and the
// sentence the person ticks. The text is never hard-coded in the pages.

import { h } from "./dom.js";
import { t } from "./i18n.js";

/** The section showing `notice`; `checkbox` is the box the person ticks. */
export function noticeSection(notice, checkbox, idPrefix) {
  const heading = `${idPrefix}-consent-h`;
  return [
    h("section", { class: "consent", "aria-labelledby": heading },
      h("h2", { id: heading }, notice.title),
      h("ul", {}, notice.points.map((point) => h("li", {}, point))),
      h("p", { class: "hint" }, t("consent.version", { version: notice.version }))),
    h("label", { class: "check strong" }, checkbox, h("span", {}, notice.accept)),
  ];
}
