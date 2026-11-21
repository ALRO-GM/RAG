// The app shell of every page: the sidebar (a drawer on phones, collapsible
// to icons on desktop), the top bar with the breadcrumbs, the theme button
// (system / light / dark, remembered in this browser) and the quick
// switcher opened with Ctrl/Cmd + K. js/boot.js applies the remembered
// theme before the first paint.

import { h, fill } from "./dom.js";
import { lang, setLanguage, t, translateStatic } from "./i18n.js";
import { isBusy } from "./device.js";
import { currentHref, moveSelection, nextTheme, paletteShortcut, rankCommands, storedTheme } from "./lib/shell.js";

const THEME_KEY = "transcriber.theme";
const NAV_KEY = "transcriber.nav";
const root = document.documentElement;
const wide = () => globalThis.matchMedia?.("(min-width: 64rem)").matches ?? true;

function remember(key, value) {
  try {
    if (value === null) globalThis.localStorage?.removeItem(key);
    else globalThis.localStorage?.setItem(key, value);
  } catch {
    // kept for this visit only
  }
}

function recall(key) {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

/** The theme in force: "system", "light" or "dark". */
export function currentTheme() {
  return storedTheme(root.dataset.theme ?? recall(THEME_KEY));
}

/** Use `theme` from now on and remember it. */
export function setTheme(theme) {
  const chosen = storedTheme(theme);
  if (chosen === "system") delete root.dataset.theme;
  else root.dataset.theme = chosen;
  remember(THEME_KEY, chosen === "system" ? null : chosen);
  syncThemeButton();
}

function syncThemeButton() {
  const button = document.getElementById("theme-toggle");
  if (!button) return;
  const name = t(`theme.${currentTheme()}`);
  button.setAttribute("aria-label", t("theme.label", { name }));
  button.title = t("theme.label", { name });
}

function setDrawer(open) {
  root.classList.toggle("nav-open", open);
  const toggle = document.getElementById("nav-toggle");
  if (!wide()) toggle?.setAttribute("aria-expanded", String(open));
  if (open) document.querySelector("#sidebar a")?.focus();
}

/** Mark the navigation link of the page shown with aria-current. */
export function markCurrent() {
  const links = [...document.querySelectorAll("#sidebar a, .bottom-nav a")];
  const current = currentHref(links.map((a) => a.getAttribute("href")), location.pathname, location.hash);
  for (const link of links) {
    if (link.getAttribute("href") === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

/** Show `items` ([{ label, href }], the last is the current page) in the top bar. */
export function setCrumbs(items) {
  const list = document.getElementById("crumbs");
  if (!list) return;
  fill(list, items.map((item, i) => h("li", {},
    item.href && i < items.length - 1
      ? h("a", { href: item.href }, item.label)
      : h("span", { "aria-current": "page" }, item.label))));
}

let skeletonTimer = null;

/** Show the loading skeleton if the page is still loading after a moment. */
export function showSkeleton() {
  clearTimeout(skeletonTimer);
  skeletonTimer = setTimeout(() => { document.getElementById("skeleton")?.removeAttribute("hidden"); }, 180);
}

/** Hide the loading skeleton. */
export function hideSkeleton() {
  clearTimeout(skeletonTimer);
  document.getElementById("skeleton")?.setAttribute("hidden", "");
}

// ---- quick switcher -------------------------------------------------------

function commands(onLanguage) {
  const seen = new Set();
  const pages = [];
  for (const link of document.querySelectorAll("#sidebar a")) {
    const href = link.getAttribute("href");
    if (seen.has(href)) continue;
    seen.add(href);
    pages.push({ label: link.textContent.trim(), kind: t("palette.page"), run: () => { location.href = link.href; } });
  }
  const themes = ["system", "light", "dark"].map((theme) => ({
    label: t("palette.theme", { name: t(`theme.${theme}`) }), kind: t("palette.action"), keywords: t("palette.theme_keywords"), run: () => setTheme(theme),
  }));
  const languages = isBusy() ? [] : [
    { code: "es", label: "Español", kind: t("palette.language"), keywords: t("palette.language_keywords"), run: () => onLanguage("es") },
    { code: "en", label: "English", kind: t("palette.language"), keywords: t("palette.language_keywords"), run: () => onLanguage("en") },
  ].filter((c) => c.code !== lang);
  return [...pages, ...themes, ...languages];
}

function installPalette(onLanguage) {
  const dialog = document.getElementById("palette");
  if (!dialog || typeof dialog.showModal !== "function") return () => {};
  const input = h("input", { type: "search", role: "combobox", "aria-expanded": "true", "aria-controls": "palette-list", "aria-autocomplete": "list", autocomplete: "off", spellcheck: false });
  const list = h("ul", { id: "palette-list", role: "listbox" });
  const foot = h("div", { class: "palette-foot", "aria-hidden": "true" });
  fill(dialog, input, list, foot);
  let shown = [];
  let selected = 0;

  function draw() {
    shown = rankCommands(commands(onLanguage), input.value);
    selected = Math.min(selected, Math.max(shown.length - 1, 0));
    fill(list, shown.length
      ? shown.map((c, i) => h("li", {
        id: `palette-${i}`, role: "option", "aria-selected": String(i === selected),
        onClick: () => choose(i), onMousemove: () => { if (selected !== i) { selected = i; mark(); } },
      }, c.label, h("span", { class: "kind" }, c.kind)))
      : h("li", { class: "none", role: "option", "aria-disabled": "true" }, t("palette.none")));
    mark();
  }

  function mark() {
    [...list.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === selected)));
    if (shown.length) {
      input.setAttribute("aria-activedescendant", `palette-${selected}`);
      list.children[selected]?.scrollIntoView?.({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function choose(i) {
    const command = shown[i];
    if (!command) return;
    dialog.close();
    command.run();
  }

  input.addEventListener("input", () => { selected = 0; draw(); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      selected = moveSelection(selected, event.key === "ArrowDown" ? 1 : -1, shown.length);
      mark();
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(selected);
    }
  });
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

  return function open() {
    if (dialog.open) return;
    input.value = "";
    input.setAttribute("aria-label", t("palette.label"));
    input.placeholder = t("palette.placeholder");
    dialog.setAttribute("aria-label", t("palette.label"));
    fill(foot, h("span", {}, t("palette.hint_move")), h("span", {}, t("palette.hint_open")), h("span", {}, t("palette.hint_close")));
    selected = 0;
    draw();
    dialog.showModal();
    input.focus();
  };
}

/**
 * Wire the shell of the page. `onLanguage(code)` changes the language the
 * way the page's ES | EN switch does (the quick switcher offers it too).
 */
export function installShell({ onLanguage = () => {} } = {}) {
  const changeLanguage = (code) => {
    if (isBusy() || code === lang || !setLanguage(code)) return;
    translateStatic(document);
    syncThemeButton();
    onLanguage(code);
  };
  const toggle = document.getElementById("nav-toggle");
  toggle?.addEventListener("click", () => {
    if (wide()) {
      const collapsed = !root.classList.contains("nav-collapsed");
      root.classList.toggle("nav-collapsed", collapsed);
      remember(NAV_KEY, collapsed ? "collapsed" : null);
      toggle.setAttribute("aria-expanded", String(!collapsed));
    } else {
      setDrawer(!root.classList.contains("nav-open"));
    }
  });
  if (toggle && wide()) toggle.setAttribute("aria-expanded", String(!root.classList.contains("nav-collapsed")));
  document.getElementById("nav-more")?.addEventListener("click", () => setDrawer(true));
  document.getElementById("scrim")?.addEventListener("click", () => setDrawer(false));
  document.getElementById("sidebar")?.addEventListener("click", (event) => {
    if (event.target.closest?.("a")) setDrawer(false);
  });
  globalThis.matchMedia?.("(min-width: 64rem)").addEventListener?.("change", () => setDrawer(false));

  document.getElementById("theme-toggle")?.addEventListener("click", () => setTheme(nextTheme(currentTheme())));
  syncThemeButton();

  const openPalette = installPalette(changeLanguage);
  const key = document.getElementById("palette-key");
  if (key) key.textContent = paletteShortcut(globalThis.navigator?.platform);
  const search = document.getElementById("palette-open");
  search?.addEventListener("click", openPalette);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openPalette();
    } else if (event.key === "Escape" && root.classList.contains("nav-open")) {
      setDrawer(false);
      toggle?.focus();
    }
  });
  // The skip link moves focus without touching the address (index.html routes on the hash).
  for (const skip of document.querySelectorAll(".skip-link")) {
    skip.addEventListener("click", (event) => {
      event.preventDefault();
      document.getElementById(skip.getAttribute("href").slice(1))?.focus();
    });
  }
  window.addEventListener("hashchange", markCurrent);
  markCurrent();
  return { changeLanguage };
}
