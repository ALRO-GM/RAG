// Loaded as a classic script in <head>, before the first paint: applies
// the theme and the collapsed sidebar remembered in this browser, so a
// dark page never flashes light. The rest of the shell is js/shell.js.
try {
  const theme = globalThis.localStorage?.getItem("transcriber.theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  if (globalThis.localStorage?.getItem("transcriber.nav") === "collapsed") document.documentElement.classList.add("nav-collapsed");
} catch {
  // storage blocked: follow the device's theme
}
