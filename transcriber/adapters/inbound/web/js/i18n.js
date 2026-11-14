// The page's language: Spanish, English by navigator.language.

import { es } from "./i18n/es.js";
import { en } from "./i18n/en.js";
import { pickLanguage, translate } from "./lib/i18n.js";

const dictionaries = { es, en };

export const lang = pickLanguage(globalThis.navigator?.language, Object.keys(dictionaries));

/** Translate a key (with {placeholders}). */
export function t(key, params) {
  return translate(dictionaries, lang, key, params);
}

/** Translate the static parts of index.html marked with data-i18n. */
export function translateStatic(root) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-label]")) el.setAttribute("aria-label", t(el.dataset.i18nLabel));
  document.documentElement.lang = lang;
  document.title = t("app.title");
}
