// The page's language: Spanish or English, chosen with the ES | EN switch
// (remembered in this browser) or, the first time, by navigator.language.

import { es } from "./i18n/es.js";
import { en } from "./i18n/en.js";
import { chooseLanguage, formatDecimal, formatPercent, translate } from "./lib/i18n.js";

const dictionaries = { es, en };
const STORAGE_KEY = "transcriber.lang";

/** The languages there is a dictionary for. */
export const LANGUAGES = Object.keys(dictionaries);

function storedLanguage() {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) ?? null;
  } catch {
    return null;
  }
}

/** The current language (a live binding: it changes with `setLanguage`). */
export let lang = chooseLanguage(storedLanguage(), globalThis.navigator?.language, LANGUAGES);

/** Use `next` from now on and remember it in this browser; false if unknown. */
export function setLanguage(next) {
  if (!LANGUAGES.includes(next)) return false;
  lang = next;
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, next);
  } catch {
    // kept for this visit only
  }
  return true;
}

/** Translate a key (with {placeholders}). */
export function t(key, params) {
  return translate(dictionaries, lang, key, params);
}

/** A number with `digits` decimals in the current language. */
export function decimal(value, digits = 1) {
  return formatDecimal(lang, value, digits);
}

/** A ratio as a percentage in the current language. */
export function percent(ratio, digits = 1) {
  return formatPercent(lang, ratio, digits);
}

/** Translate the static parts of a page marked with data-i18n (its <title> too). */
export function translateStatic(root) {
  // A link with an icon keeps it: only its .label child takes the text.
  for (const el of root.querySelectorAll("[data-i18n]")) (el.querySelector(":scope > .label") ?? el).textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-label]")) el.setAttribute("aria-label", t(el.dataset.i18nLabel));
  document.documentElement.lang = lang;
}
