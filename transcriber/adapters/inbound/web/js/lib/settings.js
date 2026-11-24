// «Ajustes» (#/settings): pure helpers of the pipeline settings page
// (settings.v1, docs/SETTINGS.md). The steps, their options and the
// presets come from the server (GET /api/settings, `catalogue`); this
// module only orders, compares and labels them. No DOM.

/** Order of the groups on the page; a group the server adds later (the
 * voice bot's, for instance) is drawn after these with its own id. */
export const GROUPS = ["transcription", "live", "analysis", "knowledge", "voicebot", "interview"];

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
  "interview.slot_filler": { haiku: "quota", heuristic: "local" },
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

// ---- the voice bot's voice (#207) ---------------------------------------

/** Whether `step` takes a number within its `range` (a slider). */
export function isRange(step) {
  return Boolean(step?.range && Number.isFinite(step.range.low) && Number.isFinite(step.range.high));
}

/** `value` as the server keeps it: at most three decimals, no trailing zeros ("1.10" -> "1.1"). */
export function numberText(value) {
  const n = Math.round(Number(value) * 1000) / 1000;
  return Number.isFinite(n) ? String(n) : "";
}

/** `value` within the step's range, as text, or the step's default. */
export function clampRange(step, value) {
  const n = Number(value);
  if (!isRange(step) || !Number.isFinite(n)) return step?.default ?? "";
  return numberText(Math.min(step.range.high, Math.max(step.range.low, n)));
}

/** The options of the voice step: Piper's, then "clone:<id>" of each person who agreed to cloning (and the saved one). */
export function voiceOptions(step, speakers, current) {
  const options = [...(step?.options ?? [])];
  for (const person of speakers ?? []) {
    if (person?.voice_clone) options.push(`clone:${person.id}`);
  }
  if (typeof current === "string" && current.startsWith("clone:") && !options.includes(current)) options.push(current);
  return options;
}

/** The options of the rate-matching step: "off", then each person whose speaking rate was measured (and the saved one). */
export function rateOptions(step, speakers, current) {
  const options = [...(step?.options ?? ["off"])];
  for (const person of speakers ?? []) {
    if (Number(person?.words_per_second) > 0) options.push(person.id);
  }
  if (typeof current === "string" && current && !options.includes(current)) options.push(current);
  return options;
}

/** Only the voice bot's choices of `draft` (what the sample button sends). */
export function voicebotChoices(draft) {
  return Object.fromEntries(Object.entries(draft ?? {}).filter(([step]) => step.startsWith("voicebot.")));
}

/** The voice bot's Piper controls (speed, intonation, cadence, pause). */
export const PIPER_CONTROLS = ["voicebot.length_scale", "voicebot.noise_scale", "voicebot.noise_w", "voicebot.sentence_silence"];

/**
 * Whether the voice bot's Piper controls reach the voice `draft` picks:
 * Piper's own voices, and a cloned voice made by OpenVoice (it speaks over
 * Piper and recolours it) yes; Pocket TTS, Qwen3-TTS and a Qwen3-TTS clone
 * have no such controls. Returns "piper", "openvoice" or "none".
 */
export function controlsReach(draft) {
  const voice = draft?.["voicebot.voice"] ?? "pocket:lola";
  if (voice.startsWith("clone:")) return (draft?.["voicebot.clone_engine"] ?? "qwen3") === "openvoice" ? "openvoice" : "maybe";
  return voice === "davefx" || voice === "sharvard" ? "piper" : "none";
}

/**
 * Play a sample's WAV bytes on `ctx` (an AudioContext, or a stand-in with
 * the same members); resolves to its seconds. Web Audio, not an <audio>
 * element: the public site's Content-Security-Policy (`media-src 'self'`)
 * refuses blob: and data: media, and a context made inside the click keeps
 * its permission to sound while a cloned voice takes seconds (#253).
 * Rejects with NotAllowedError when the browser keeps the context silent.
 */
export async function playWav(ctx, data) {
  const buffer = await ctx.decodeAudioData(data.slice(0));
  if (ctx.state === "suspended") {
    throw Object.assign(new Error("the browser keeps the audio suspended"), { name: "NotAllowedError" });
  }
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(ctx.currentTime);
  return buffer.duration;
}

/**
 * Why «Escuchar ejemplo» failed, in words for the page (#253): an i18n key
 * and a short detail from the server, or null when the generic error text
 * fits. `error` is an ApiFailure (the request) or a DOMException (playing).
 */
export function sampleFailure(error) {
  if (error?.name === "NotAllowedError") return { key: "set.sample_err.blocked", detail: "" };
  if (error?.name === "NotSupportedError" || error?.name === "EncodingError") return { key: "set.sample_err.format", detail: "" };
  if (error?.name === "NoAudio") return { key: "set.sample_err.no_audio", detail: "" };
  if (error?.code === "orchestrator_unavailable") {
    const slow = /timeout/i.test(String(error.message ?? ""));
    return { key: slow ? "set.sample_err.slow" : "set.sample_err.service", detail: "" };
  }
  if (error?.code === "unavailable") return { key: "set.sample_err.no_voice", detail: String(error.message ?? "").slice(0, 120) };
  return null;
}
