// Entry point of competition.html (#291): loads competition/data.json
// (kept by hand; how to update it: docs/web/COMPETITION_PAGE.md) and draws
// the page; redraws on a language change (nothing depends on the width:
// the bars stretch with their box).

import { translateStatic } from "./i18n.js";
import { installLanguageSwitch } from "./lang-switch.js";
import { installShell } from "./shell.js";
import { installPageShare } from "./page-share.js";
import { isCompetitionData } from "./lib/competition.js";
import { renderMessage, renderCompetition } from "./views/competition.js";

const main = document.getElementById("competition");
let data = null;

function draw() {
  translateStatic(document);
  if (data) renderCompetition(main, data);
}

async function start() {
  renderMessage(main, "comp.loading");
  try {
    const response = await fetch("competition/data.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(String(response.status));
    const loaded = await response.json();
    if (!isCompetitionData(loaded)) throw new Error("schema");
    data = loaded;
  } catch {
    renderMessage(main, "comp.error", true);
    return;
  }
  draw();
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}

translateStatic(document);
let share = null;
const relabel = () => { draw(); share?.relabel(); };
installLanguageSwitch(document.querySelector(".top .lang"), relabel);
installShell({ onLanguage: relabel });
share = installPageShare(); // #284: a shared session goes on here
start();
