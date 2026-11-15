// Entry point of study.html: loads study/data.json (built by
// `python -m transcriber.adapters.inbound.study_data` from the committed
// results) and draws the study; redraws on a language change and when
// the width changes.

import { translateStatic } from "./i18n.js";
import { installLanguageSwitch } from "./lang-switch.js";
import { isStudyData } from "./lib/study.js";
import { renderMessage, renderStudy } from "./views/study.js";

const main = document.getElementById("study");
let data = null;
let redraw = () => {};

function draw() {
  translateStatic(document);
  if (data) redraw = renderStudy(main, data);
}

async function start() {
  renderMessage(main, "study.loading");
  try {
    const response = await fetch("study/data.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(String(response.status));
    const loaded = await response.json();
    if (!isStudyData(loaded)) throw new Error("schema");
    data = loaded;
  } catch {
    renderMessage(main, "study.error", true);
    return;
  }
  draw();
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  let lastWidth = document.documentElement.clientWidth;
  let timer = null;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const width = document.documentElement.clientWidth;
      if (width !== lastWidth) { lastWidth = width; redraw(); }
    }, 150);
  });
}

translateStatic(document);
installLanguageSwitch(document.querySelector(".top .lang"), draw);
start();
