// "Analysis" section of the sessions page: which judge the RAG uses for
// the soft-skills analysis, and whether Claude's quota is available.
// Loaded when opened, so the list never waits on the RAG.

import { h } from "./dom.js";
import { t, lang } from "./i18n.js";
import { describeError } from "./errors.js";
import { JUDGE_MODES, quotaLine, selectedMode } from "./lib/analysis.js";

export function judgeSection(ctx) {
  const select = h("select", { id: "judge-mode", disabled: true },
    ...JUDGE_MODES.map((mode) => h("option", { value: mode }, t(`analysis.mode_${mode}`))));
  const quota = h("p", { class: "hint", role: "status" });
  const status = h("p", { class: "hint", role: "status" });
  let loaded = false;

  function show(answer) {
    select.value = selectedMode(answer);
    select.disabled = false;
    quota.textContent = quotaLine(t, answer, lang);
    status.textContent = "";
  }

  function fail(error) {
    select.disabled = true;
    status.textContent = error.code === "analysis_unavailable" ? t("analysis.unavailable") : describeError(error);
    if (error.kind === "unauthorized") ctx.fail(error);
  }

  async function load() {
    if (loaded) return;
    status.textContent = t("analysis.loading");
    try {
      show(await ctx.api.analysisJudge());
      loaded = true;
    } catch (error) {
      fail(error);
    }
  }

  select.addEventListener("change", async () => {
    select.disabled = true;
    status.textContent = t("analysis.saving");
    try {
      show(await ctx.api.setAnalysisJudge(select.value));
      status.textContent = t("analysis.saved");
    } catch (error) {
      fail(error);
      loaded = false;
    }
  });

  return h("details", { class: "upload", onToggle: (event) => { if (event.target.open) load(); } },
    h("summary", {}, t("analysis.title")),
    h("div", { class: "stack" },
      h("p", { class: "hint" }, t("analysis.help")),
      h("div", { class: "field" }, h("label", { for: "judge-mode" }, t("analysis.judge")), select),
      quota, status));
}
