// The ES | EN switch in the header of every page. Choosing a language
// remembers it and calls `onChange` so the page can redraw itself. While
// something would be lost by redrawing (a recording, chunks still to send)
// the switch is disabled and a click does nothing.

import { lang, setLanguage, t, translateStatic } from "./i18n.js";
import { isBusy } from "./device.js";

/** Wire the [data-lang] buttons inside `group`; `onChange` runs after a change. */
export function installLanguageSwitch(group, onChange = () => {}) {
  if (!group) return;
  const buttons = [...group.querySelectorAll("button[data-lang]")];
  const sync = () => {
    const busy = isBusy();
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(button.dataset.lang === lang));
      button.disabled = busy;
    }
    if (busy) group.title = t("lang.busy");
    else group.removeAttribute("title");
  };
  for (const button of buttons) {
    button.addEventListener("click", () => {
      if (!isBusy() && button.dataset.lang !== lang && setLanguage(button.dataset.lang)) {
        translateStatic(document);
        onChange();
      }
      sync();
    });
  }
  sync();
  // A recording starts and ends inside a view: follow it.
  setInterval(sync, 1000);
}
