// Pure logic of the app shell: the theme cycle, which navigation link is
// the current page, which routes form the Administración area, the
// breadcrumbs of a route and the ranking of the quick switcher
// (Ctrl/Cmd + K). No DOM here, so node --test can run it.

/** The themes in the order the top-bar button cycles through them. */
export const THEMES = ["system", "light", "dark"];

/** A stored theme we know, else "system" (follow the device). */
export function storedTheme(value) {
  return THEMES.includes(value) ? value : "system";
}

/** The theme after `current` when the button is pressed. */
export function nextTheme(current) {
  const at = THEMES.indexOf(storedTheme(current));
  return THEMES[(at + 1) % THEMES.length];
}

/** Routes that belong to the sessions list: a session, its analysis, the help, a recording. */
const SESSION_PARTS = new Set(["sessions", "s", "analysis", "help", "record"]);

/** Routes of the Administración area (besides its own overview, #/admin). */
export const ADMIN_PARTS = new Set(["admin", "settings", "enrol", "knowledge", "mandatory", "lab"]);

/**
 * The section of an app hash: "" -> "#/" (the home), "#/s/abc" ->
 * "#/sessions", "#/lab/x" -> "#/lab".
 */
export function sectionOf(hash) {
  const first = routePath(hash).split("/")[0];
  if (!first) return "#/";
  if (SESSION_PARTS.has(first)) return "#/sessions";
  return `#/${first}`;
}

/** Whether `hash` is a page of the Administración area. */
export function isAdminRoute(hash) {
  return ADMIN_PARTS.has(routePath(hash).split("/")[0]);
}

/** The path of an app hash without "#/" and without its query ("#/offers/x?sort=s1" -> "offers/x"). */
function routePath(hash) {
  return String(hash ?? "").replace(/^#?\/?/, "").split("?")[0];
}

/**
 * Which of `hrefs` (as written in the navigation) is the page at
 * `pathname` + `hash`; null when none. "index.html#/lab" and "#/lab" are
 * the same page; "research.html" matches by file name.
 */
export function currentHref(hrefs, pathname, hash) {
  const file = String(pathname ?? "").split("/").pop() || "index.html";
  const section = sectionOf(hash);
  for (const href of hrefs) {
    const [page, fragment] = String(href).split("#");
    const target = page || "index.html";
    if (fragment === undefined) {
      if (target === file) return href;
      continue;
    }
    if (target === file && sectionOf(`#${fragment}`) === section) return href;
  }
  return null;
}

/**
 * The breadcrumbs of a route of index.html: [{ label, href }] where the
 * last has no href (the current page). `t` translates dictionary keys;
 * ids from the address are shown as they are.
 */
export function crumbsFor(hash, t) {
  const trail = sectionCrumbs(hash, t);
  const head = routePath(hash).split("/")[0];
  // Every page of the Administración area but its overview starts from it.
  return isAdminRoute(hash) && head !== "admin" ? [{ label: t("nav.admin"), href: "#/admin" }, ...trail] : trail;
}

/** The breadcrumbs of a route within its own section (crumbsFor adds the area). */
function sectionCrumbs(hash, t) {
  const parts = routePath(hash).split("/").filter(Boolean).map(safeDecode);
  const sessions = { label: t("nav.sessions"), href: "#/sessions" };
  const here = (label) => ({ label, href: null });
  const [head, a, b, c] = parts;
  switch (head) {
    case undefined: return [here(t("nav.home"))];
    case "sessions": return [here(t("nav.sessions"))];
    case "record": return [sessions, here(t("nav.record"))];
    case "admin": return [here(t("nav.admin"))];
    case "enrol": return [here(t("nav.enrol"))];
    case "live": return [here(t("nav.live"))];
    case "voicebot": return a === "demo" ? [{ label: t("nav.voicebot"), href: "#/voicebot" }, here(t("vdemo.crumb"))] : [here(t("nav.voicebot"))];
    case "call-report": return [here(t("report.title"))];
    case "offers": {
      const offers = { label: t("nav.offers"), href: "#/offers" };
      if (!a) return [here(t("nav.offers"))];
      if (b === "person") return [offers, { label: a, href: `#/offers/${encodeURIComponent(a)}` }, here(c ?? "")];
      return [offers, here(a)];
    }
    case "settings": return [here(t("nav.settings"))];
    case "s": return [sessions, here(a ?? "")];
    case "analysis": return [sessions, { label: a ?? "", href: `#/s/${encodeURIComponent(a ?? "")}` }, here(t("nav.analysis"))];
    case "help": return [sessions, here(t("help.title"))];
    case "knowledge": return a ? [{ label: t("nav.knowledge"), href: "#/knowledge" }, here(a)] : [here(t("nav.knowledge"))];
    case "mandatory": return a ? [{ label: t("nav.mandatory"), href: "#/mandatory" }, here(a)] : [here(t("nav.mandatory"))];
    case "lab": {
      const lab = { label: t("nav.lab"), href: "#/lab" };
      if (!a) return [here(t("nav.lab"))];
      if (!b) return [lab, here(a)];
      const run = { label: b.slice(0, 8), href: `#/lab/${encodeURIComponent(a)}/${encodeURIComponent(b)}` };
      return c ? [lab, { label: a, href: `#/lab/${encodeURIComponent(a)}` }, run, here(c)] : [lab, { label: a, href: `#/lab/${encodeURIComponent(a)}` }, here(b.slice(0, 8))];
    }
    default: return [here(t("nav.home"))];
  }
}

function safeDecode(part) {
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
}

/** Lower case without accents, for matching "investigacion" to "Investigación". */
export function fold(text) {
  return String(text ?? "").normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

/**
 * How well `query` matches `label`: 0 means no match. A prefix beats a
 * word start, which beats a substring, which beats the letters in order.
 */
export function matchScore(label, query) {
  const l = fold(label);
  const q = fold(query).trim();
  if (!q) return 1;
  if (l.startsWith(q)) return 100 - Math.min(l.length - q.length, 50) / 100;
  if (l.split(/[\s/·:-]+/).some((word) => word.startsWith(q))) return 80;
  if (l.includes(q)) return 60;
  let at = 0;
  for (const ch of q) {
    at = l.indexOf(ch, at);
    if (at < 0) return 0;
    at += 1;
  }
  return 20;
}

/** The commands matching `query`, best first (ties keep their order). */
export function rankCommands(commands, query, limit = 12) {
  return commands
    .map((command, index) => ({ command, index, score: matchScore(`${command.label} ${command.keywords ?? ""}`, query) }))
    .filter((x) => x.score > 0)
    .sort((x, y) => y.score - x.score || x.index - y.index)
    .slice(0, limit)
    .map((x) => x.command);
}

/** The index after `index` moving `step` (+1 / -1) in a list of `length`, wrapping. */
export function moveSelection(index, step, length) {
  if (length <= 0) return -1;
  return (((index + step) % length) + length) % length;
}

/** "⌘ K" on Apple devices, "Ctrl K" elsewhere. */
export function paletteShortcut(platform) {
  return /mac|iphone|ipad|ipod/i.test(String(platform ?? "")) ? "⌘ K" : "Ctrl K";
}

/**
 * A breadcrumb setter for one page: it writes only while that page is still
 * the current one (`isCurrent`), so a late answer from the page the person
 * left never overwrites the next page's breadcrumbs.
 */
export function crumbsWhileCurrent(isCurrent, set) {
  return (items) => {
    if (isCurrent()) set(items);
  };
}
