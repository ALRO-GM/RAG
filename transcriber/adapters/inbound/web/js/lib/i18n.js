// Pure dictionary lookup keyed by the browser's language.

export const DEFAULT_LANGUAGE = "es";

/** "es-ES" -> "es"; a language we have no dictionary for -> the default. */
export function pickLanguage(navigatorLanguage, available) {
  const base = String(navigatorLanguage ?? "").toLowerCase().split(/[-_]/)[0];
  return available.includes(base) ? base : DEFAULT_LANGUAGE;
}

/** Look `key` up in `dictionaries[lang]`, then in Spanish, then the key itself. */
export function translate(dictionaries, lang, key, params = {}) {
  const raw = dictionaries[lang]?.[key] ?? dictionaries[DEFAULT_LANGUAGE]?.[key] ?? key;
  return raw.replace(/\{(\w+)\}/g, (whole, name) =>
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : whole);
}
