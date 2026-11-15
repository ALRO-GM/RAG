// Pure dictionary lookup keyed by the chosen language, and number formatting.

export const DEFAULT_LANGUAGE = "es";

/** "es-ES" -> "es"; a language we have no dictionary for -> the default. */
export function pickLanguage(navigatorLanguage, available) {
  const base = String(navigatorLanguage ?? "").toLowerCase().split(/[-_]/)[0];
  return available.includes(base) ? base : DEFAULT_LANGUAGE;
}

/** The language chosen before in this browser if we still have it, else the browser's. */
export function chooseLanguage(stored, navigatorLanguage, available) {
  return available.includes(stored) ? stored : pickLanguage(navigatorLanguage, available);
}

/** Look `key` up in `dictionaries[lang]`, then in Spanish, then the key itself. */
export function translate(dictionaries, lang, key, params = {}) {
  const raw = dictionaries[lang]?.[key] ?? dictionaries[DEFAULT_LANGUAGE]?.[key] ?? key;
  return raw.replace(/\{(\w+)\}/g, (whole, name) =>
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : whole);
}

/** `value` with exactly `digits` decimals in the language's notation ("3,5" / "3.5"). */
export function formatDecimal(lang, value, digits = 1) {
  return new Intl.NumberFormat(lang, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

/** A ratio as a percentage in the language's notation ("12,5 %" / "12.5%"). */
export function formatPercent(lang, ratio, digits = 1) {
  return new Intl.NumberFormat(lang, { style: "percent", minimumFractionDigits: digits, maximumFractionDigits: digits }).format(ratio);
}
