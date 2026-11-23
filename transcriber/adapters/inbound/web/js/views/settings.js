// «Ajustes» (#/settings): the mode (highest quality, balanced, lowest cost
// or custom) and the model of every pipeline step that has measured
// alternatives, each option with what was measured with it. The RAG keeps
// the settings (settings.v1, GET/PUT /api/settings through this service);
// every job reads them when it starts. The voice bot's voice has sliders
// (Piper's controls), the people who agreed to their voice being cloned,
// rate matching and an «Escuchar ejemplo» button that plays the draft.
// Pure helpers in lib/settings.js.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  PRESETS, applyPreset, changedSteps, clampRange, costOf, isRange, keys, label, needsRestart, presetOf, rateOptions, saveBody,
  stepsByGroup, voiceOptions, voicebotChoices, controlsReach, PIPER_CONTROLS,
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
  let speakers = [];
  const sampleStatus = h("p", { class: "hint", role: "status" });
  let sampleUrl = null;

  async function load() {
    try {
      settings = await ctx.api.pipelineSettings();
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      fill(body, emptyState({ iconName: "settings", title: t("set.unavailable_title"), body: `${t("set.unavailable_body")} ${describeError(error)}` }));
      return;
    }
    try {
      speakers = (await ctx.api.speakers?.())?.speakers ?? [];
    } catch {
      speakers = [];
    }
    draft = { ...settings.choices };
    render();
  }

  async function playSample(button) {
    button.disabled = true;
    fill(sampleStatus, t("set.sample_wait"));
    try {
      const sample = await ctx.api.voiceSample(voicebotChoices(draft));
      if (sampleUrl) URL.revokeObjectURL(sampleUrl);
      sampleUrl = URL.createObjectURL(sample.blob);
      await new Audio(sampleUrl).play();
      fill(sampleStatus, sampleText(sample));
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      fill(sampleStatus, `${t("set.sample_unavailable")} ${describeError(error)}`);
    } finally {
      button.disabled = false;
    }
  }

  function sampleText(sample) {
    if (!sample.engine) {
      return sample.clone ? t("set.sample_clone") : sample.note ? t("set.sample_piper_because", { note: sample.note }) : t("set.sample_piper");
    }
    const engine = label(t, `set.engine.${sample.engine}`, sample.engine);
    return sample.note ? t("set.sample_engine_because", { engine, note: sample.note }) : t("set.sample_engine", { engine });
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

  function personName(id) {
    const person = speakers.find((p) => p.id === id);
    return person?.name || id;
  }

  function optionText(step, option) {
    if (step.id === "voicebot.voice" && option.startsWith("clone:")) {
      return [t("set.clone_option", { name: personName(option.slice(6)) }), t("set.fact.voicebot.voice.clone")];
    }
    if (step.id === "voicebot.match_rate" && option !== "off") {
      const person = speakers.find((p) => p.id === option);
      const rate = Number(person?.words_per_second);
      return [t("set.rate_option", { name: personName(option) }),
        rate > 0 ? t("set.rate_fact", { rate: rate.toLocaleString(lang, { maximumFractionDigits: 2 }) }) : t("set.rate_missing")];
    }
    return [label(t, keys.option(step.id, option), option), label(t, keys.fact(step.id, option), "")];
  }

  function optionsOf(step) {
    if (step.id === "voicebot.voice") return voiceOptions(step, speakers, draft[step.id]);
    if (step.id === "voicebot.match_rate") return rateOptions(step, speakers, draft[step.id]);
    return step.options;
  }

  function rangeBlock(step) {
    const changed = settings.choices[step.id] !== draft[step.id];
    const value = draft[step.id] ?? step.default;
    const shown = h("output", { for: `set-${step.id}`.replace(/[^\w-]/g, "-") }, value);
    const slider = h("input", {
      type: "range", id: `set-${step.id}`.replace(/[^\w-]/g, "-"), min: step.range.low, max: step.range.high,
      step: step.range.step, value,
      onInput: (event) => { shown.textContent = clampRange(step, event.target.value); },
      onChange: (event) => { draft = { ...draft, [step.id]: clampRange(step, event.target.value) }; render(); },
    });
    const reach = PIPER_CONTROLS.includes(step.id) ? controlsReach(draft) : "piper";
    if (reach === "none") slider.disabled = true;
    const reset = h("button", { type: "button", class: "ghost", disabled: value === step.default || reach === "none",
      onClick: () => { draft = { ...draft, [step.id]: step.default }; render(); } }, t("set.reset"));
    return h("fieldset", { class: changed ? "field step is-changed" : "field step" },
      h("legend", {}, label(t, keys.step(step.id), step.id), " ",
        h("span", { class: "badge muted" }, t(`set.applies.${step.applies}`))),
      h("div", { class: "row wrap tight" }, slider, shown, reset),
      h("p", { class: "hint" }, label(t, `set.range.${step.id}`, ""), " ",
        t("set.range_default", { value: step.default })),
      reach === "none" ? h("p", { class: "hint" }, t("set.controls_none")) : null,
      reach === "openvoice" ? h("p", { class: "hint" }, t("set.controls_openvoice")) : null,
      reach === "maybe" ? h("p", { class: "hint" }, t("set.controls_maybe")) : null);
  }

  function optionRow(step, option) {
    const id = `set-${step.id}-${option}`.replace(/[^\w-]/g, "-");
    const input = h("input", {
      type: "radio", name: `set-${step.id}`, id, value: option, checked: draft[step.id] === option,
      onChange: () => { draft = { ...draft, [step.id]: option }; render(); },
    });
    const cost = costOf(step.id, option);
    const [name, fact] = optionText(step, option);
    return h("label", { class: draft[step.id] === option ? "option is-chosen" : "option", for: id },
      input,
      h("span", { class: "stack tight" },
        h("span", { class: "row wrap tight" },
          h("strong", {}, name),
          option === step.default ? h("span", { class: "tag" }, t("set.default_tag")) : null,
          cost ? h("span", { class: cost === "local" ? "tag" : "tag accent" }, t(`set.cost.${cost}`)) : null),
        h("span", { class: "hint" }, fact)));
  }

  function stepBlock(step) {
    if (isRange(step)) return rangeBlock(step);
    const changed = settings.choices[step.id] !== draft[step.id];
    return h("fieldset", { class: changed ? "field step is-changed" : "field step" },
      h("legend", {}, label(t, keys.step(step.id), step.id), " ",
        h("span", { class: step.applies === "restart" ? "badge warn" : "badge muted" }, t(`set.applies.${step.applies}`))),
      h("div", { class: "options" }, optionsOf(step).map((option) => optionRow(step, option))),
      step.id === "analysis.act_tagger" ? h("p", { class: "hint" }, t("set.rg_note")) : null,
      step.id === "voicebot.voice" ? h("p", { class: "hint" }, t("set.clone_note")) : null);
  }

  function sampleBar() {
    const button = h("button", { type: "button", onClick: () => playSample(button) }, icon("play"), " ", t("set.sample"));
    return h("div", { class: "row wrap tight" }, button, sampleStatus);
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
        steps.map(stepBlock),
        group === "voicebot" ? sampleBar() : null))),
      hasVoicebot ? null : h("p", { class: "hint" }, icon("info"), " ", t("set.voicebot_note")),
      h("p", { class: "hint" }, t("set.source")));
  }

  await load();
}
