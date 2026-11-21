// Pure logic of the app shell: the theme cycle, which navigation link is
// the current page, the breadcrumbs of a route and the ranking of the
// quick switcher (Ctrl/Cmd + K). No DOM here, so node --test can run it.

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

/** The section of an app hash: "#/s/abc" -> "#/", "#/lab/x" -> "#/lab". */
export function sectionOf(hash) {
  const first = String(hash ?? "").replace(/^#?\/?/, "").split("/")[0];
  if (!first || first === "s" || first === "analysis" || first === "help") return "#/";
  return `#/${first}`;
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
  const parts = String(hash ?? "").replace(/^#?\/?/, "").split("/").filter(Boolean).map(safeDecode);
  const sessions = { label: t("nav.sessions"), href: "#/" };
  const here = (label) => ({ label, href: null });
  const [head, a, b, c] = parts;
  switch (head) {
    case undefined: return [here(t("nav.sessions"))];
    case "record": return [sessions, here(t("nav.record"))];
    case "enrol": return [here(t("nav.enrol"))];
    case "live": return [here(t("nav.live"))];
    case "s": return [sessions, here(a ?? "")];
    case "analysis": return [sessions, { label: a ?? "", href: `#/s/${encodeURIComponent(a ?? "")}` }, here(t("nav.analysis"))];
    case "help": return [sessions, here(t("help.title"))];
    case "knowledge": return a ? [{ label: t("nav.knowledge"), href: "#/knowledge" }, here(a)] : [here(t("nav.knowledge"))];
    case "lab": {
      const lab = { label: t("nav.lab"), href: "#/lab" };
      if (!a) return [here(t("nav.lab"))];
      if (!b) return [lab, here(a)];
      const run = { label: b.slice(0, 8), href: `#/lab/${encodeURIComponent(a)}/${encodeURIComponent(b)}` };
      return c ? [lab, { label: a, href: `#/lab/${encodeURIComponent(a)}` }, run, here(c)] : [lab, { label: a, href: `#/lab/${encodeURIComponent(a)}` }, here(b.slice(0, 8))];
    }
    default: return [here(t("nav.sessions"))];
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
