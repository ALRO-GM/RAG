// «Ajustes» (#/settings): pure helpers of the pipeline settings page
// (settings.v1, docs/SETTINGS.md). The steps, their options and the
// presets come from the server (GET /api/settings, `catalogue`); this
// module only orders, compares and labels them. No DOM.

/** Order of the groups on the page; a group the server adds later (the
 * voice bot's, for instance) is drawn after these with its own id. */
export const GROUPS = ["transcription", "live", "analysis", "knowledge", "voicebot"];

/** The preset cards, in order; "custom" is any other combination. */
export const PRESETS = ["quality", "balanced", "cost", "custom"];

/**
 * What an option costs to run, for the chip next to it:
 * "local" (this PC, free), "quota" (Claude subscription calls),
 * "cloud" (Claude, and the content leaves the PC), "external" (another
 * service). Options not listed show no chip.
 */
export const COST = {
  "live.suggester_model": { haiku: "quota", sonnet: "quota", opus: "quota" },
  "live.suggester_effort": { low: "quota", medium: "quota", high: "quota" },
  "analysis.judge": { claude_code: "quota", local: "local", both: "quota", opencode: "external" },
  "analysis.confirmation_model": { haiku: "quota", sonnet: "quota", opus: "quota" },
  "analysis.assessment_model": { haiku: "quota", sonnet: "quota", opus: "quota" },
  "analysis.synthesis_model": { haiku: "quota", sonnet: "quota", opus: "quota" },
  "knowledge.captions": { florence: "local", claude: "cloud" },
};

/** Steps that change no cost: how long to wait, and what the live page shows. */
export const NO_COST = ["live.deadline", "live.assist_mode", "live.max_suggestions", "live.show_employee", "live.show_other"];

/** The cost chip of `option` of `step`: local unless said otherwise. */
export function costOf(step, option) {
  const listed = COST[step];
  if (listed) return listed[option] ?? null;
  return NO_COST.includes(step) ? null : "local";
}

/** The steps of `catalogue` grouped: [{ group, steps }] in page order. */
export function stepsByGroup(catalogue) {
  const steps = Array.isArray(catalogue?.steps) ? catalogue.steps : [];
  const groups = [...new Set(steps.map((s) => s.group))];
  groups.sort((a, b) => rank(a) - rank(b));
  return groups.map((group) => ({ group, steps: steps.filter((s) => s.group === group) }));
}

function rank(group) {
  const at = GROUPS.indexOf(group);
  return at < 0 ? GROUPS.length : at;
}

/** Every step at its default (the first option). */
export function defaultChoices(catalogue) {
  return Object.fromEntries((catalogue?.steps ?? []).map((s) => [s.id, s.default ?? s.options?.[0]]));
}

/** The preset whose choices are exactly `choices`, else "custom". */
export function presetOf(choices, catalogue) {
  const presets = catalogue?.presets ?? {};
  const full = { ...defaultChoices(catalogue), ...choices };
  for (const name of PRESETS) {
    const wanted = presets[name];
    if (wanted && Object.keys(full).every((step) => wanted[step] === full[step])) return name;
  }
  return "custom";
}

/** The choices of preset `name` ("custom" keeps `current`). */
export function applyPreset(name, catalogue, current) {
  const wanted = catalogue?.presets?.[name];
  return wanted ? { ...defaultChoices(catalogue), ...wanted } : { ...current };
}

/** The steps whose option differs between `saved` and `draft`. */
export function changedSteps(saved, draft) {
  return Object.keys(draft).filter((step) => saved?.[step] !== draft[step]);
}

/** Whether any of `steps` only applies after a restart. */
export function needsRestart(steps, catalogue) {
  const restart = new Set((catalogue?.steps ?? []).filter((s) => s.applies === "restart").map((s) => s.id));
  return steps.some((step) => restart.has(step));
}

/** The i18n key of a step's label, option label and measured trade-off. */
export const keys = {
  group: (group) => `set.group.${group}`,
  step: (step) => `set.step.${step}`,
  option: (step, option) => `set.opt.${step}.${option}`,
  fact: (step, option) => `set.fact.${step}.${option}`,
};

/** `t(key)`, or `fallback` when the dictionary has no such key. */
export function label(t, key, fallback) {
  const text = t(key);
  return text === key ? fallback : text;
}

const JUDGE_STEPS = ["confirmation", "assessment", "synthesis"];

/**
 * The Claude model of each judged step to prefill the analysis start
 * form with: the saved settings' choices, or {} when nothing is saved.
 */
export function analysisPrefill(settings) {
  if (!settings?.stored || !settings.choices) return {};
  const out = {};
  for (const step of JUDGE_STEPS) {
    const model = settings.choices[`analysis.${step}_model`];
    if (["haiku", "sonnet", "opus"].includes(model)) out[step] = model;
  }
  return out;
}

/** The body of PUT /api/settings: the draft, based on the revision read. */
export function saveBody(draft, revision) {
  return { choices: { ...draft }, revision: Number.isInteger(revision) ? revision : 0 };
}
