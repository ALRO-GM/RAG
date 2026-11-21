// «Ajustes» (#/settings): the mode (highest quality, balanced, lowest cost
// or custom) and the model of every pipeline step that has measured
// alternatives, each option with what was measured with it. The RAG keeps
// the settings (settings.v1, GET/PUT /api/settings through this service);
// every job reads them when it starts. Pure helpers in lib/settings.js.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  PRESETS, applyPreset, changedSteps, costOf, keys, label, needsRestart, presetOf, saveBody, stepsByGroup,
} from "../lib/settings.js";
import { emptyState, icon, pageHead, toast } from "../ui.js";

function dateText(iso) {
  if (!iso) return "";
  const when = new Date(iso);
  return Number.isNaN(when.getTime()) ? "" : when.toLocaleString(lang, { dateStyle: "medium", timeStyle: "short" });
}

export async function settingsView(ctx, root) {
  root.classList.add("wide");
  const body = h("div", { class: "stack settings" });
  const save = h("button", { type: "button", class: "primary", onClick: () => store() }, t("set.save"));
  const discard = h("button", { type: "button", class: "ghost", onClick: () => { draft = { ...settings.choices }; render(); } }, t("set.discard"));
  const status = h("p", { class: "hint", role: "status" });
  root.append(pageHead(t("set.title"), { lead: t("set.lead") }), body);

  let settings = null;
  let draft = {};

  async function load() {
    try {
      settings = await ctx.api.pipelineSettings();
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      fill(body, emptyState({ iconName: "settings", title: t("set.unavailable_title"), body: `${t("set.unavailable_body")} ${describeError(error)}` }));
      return;
    }
    draft = { ...settings.choices };
    render();
  }

  async function store() {
    const changed = changedSteps(settings.choices, draft);
    if (!changed.length) {
      toast(t("set.unchanged"));
      return;
    }
    save.disabled = true;
    try {
      settings = await ctx.api.savePipelineSettings(saveBody(draft, settings.revision));
      draft = { ...settings.choices };
      toast(needsRestart(changed, settings.catalogue) ? t("set.saved_restart") : t("set.saved"));
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      if (error?.status === 409) {
        toast(t("set.conflict"), { error: true });
        await load();
        return;
      }
      toast(describeError(error), { error: true });
    } finally {
      save.disabled = false;
    }
    render();
  }

  function presetCards() {
    const current = presetOf(draft, settings.catalogue);
    return h("fieldset", { class: "presets" },
      h("legend", {}, t("set.preset_title")),
      h("div", { class: "preset-grid" }, PRESETS.map((name) => {
        const input = h("input", {
          type: "radio", name: "set-preset", id: `set-preset-${name}`, value: name, checked: current === name,
          onChange: () => {
            draft = applyPreset(name, settings.catalogue, draft);
            render();
            if (name === "custom") document.getElementById("set-groups")?.querySelector("input")?.focus();
          },
        });
        return h("label", { class: current === name ? "preset-card is-chosen" : "preset-card", for: input.id },
          h("span", { class: "row tight" }, input, h("strong", {}, t(`set.preset.${name}`))),
          h("span", { class: "hint" }, t(`set.preset_desc.${name}`)));
      })));
  }

  function optionRow(step, option) {
    const id = `set-${step.id}-${option}`.replace(/[^\w-]/g, "-");
    const input = h("input", {
      type: "radio", name: `set-${step.id}`, id, value: option, checked: draft[step.id] === option,
      onChange: () => { draft = { ...draft, [step.id]: option }; render(); },
    });
    const cost = costOf(step.id, option);
    return h("label", { class: draft[step.id] === option ? "option is-chosen" : "option", for: id },
      input,
      h("span", { class: "stack tight" },
        h("span", { class: "row wrap tight" },
          h("strong", {}, label(t, keys.option(step.id, option), option)),
          option === step.default ? h("span", { class: "tag" }, t("set.default_tag")) : null,
          cost ? h("span", { class: cost === "local" ? "tag" : "tag accent" }, t(`set.cost.${cost}`)) : null),
        h("span", { class: "hint" }, label(t, keys.fact(step.id, option), ""))));
  }

  function stepBlock(step) {
    const changed = settings.choices[step.id] !== draft[step.id];
    return h("fieldset", { class: changed ? "field step is-changed" : "field step" },
      h("legend", {}, label(t, keys.step(step.id), step.id), " ",
        h("span", { class: step.applies === "restart" ? "badge warn" : "badge muted" }, t(`set.applies.${step.applies}`))),
      h("div", { class: "options" }, step.options.map((option) => optionRow(step, option))),
      step.id === "analysis.act_tagger" ? h("p", { class: "hint" }, t("set.rg_note")) : null);
  }

  function render() {
    const changed = changedSteps(settings.choices, draft);
    const meta = settings.stored
      ? t("set.revision", { n: settings.revision, when: dateText(settings.updated_at) })
      : t("set.not_saved");
    fill(status, changed.length ? t("set.pending", { n: changed.length }) : meta);
    discard.disabled = !changed.length;
    const groups = stepsByGroup(settings.catalogue);
    const hasVoicebot = groups.some((g) => g.group === "voicebot");
    fill(body,
      presetCards(),
      h("div", { class: "row wrap save-bar" }, save, discard, status),
      h("div", { id: "set-groups", class: "stack" }, groups.map(({ group, steps }) => h("section", { class: "card" },
        h("h2", {}, label(t, keys.group(group), group)),
        steps.map(stepBlock)))),
      hasVoicebot ? null : h("p", { class: "hint" }, icon("info"), " ", t("set.voicebot_note")),
      h("p", { class: "hint" }, t("set.source")));
  }

  await load();
}
