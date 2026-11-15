// Speakers' names found in what people said (TR-20, #102; grouped per
// voice, #116): pure helpers for the transcript page. No DOM here, so
// node --test can check them.

import { speakerName } from "./transcript.js";

/** Suggestion reasons the page has a sentence for. */
export const NAME_REASONS = [
  "addressed", "unsure", "several_names", "same_name", "name_taken", "conflict", "rules_only", "undone",
  "few_votes", "close",
];

/**
 * Rows to draw from `GET /api/sessions/{id}/names`: automatic names (with
 * the voice's label, to undo) and suggestions (to accept), each with the
 * speaker's current display name. Suggestions are ordered by voice, then
 * by self-introductions before names said by others, then by time.
 */
export function nameRows(names, speakers) {
  if (!names) return { auto: [], suggestions: [] };
  const known = new Set(speakers.map((s) => s.id));
  const auto = (names.auto ?? [])
    .filter((a) => known.has(a.speaker))
    .map((a) => ({ ...a, label: a.speaker, shown: speakerName(speakers, a.speaker) }));
  const order = new Map(speakers.map((s, i) => [s.id, i]));
  const suggestions = (names.suggestions ?? [])
    .filter((s) => known.has(s.speaker))
    .map((s) => ({ ...s, label: s.speaker, shown: speakerName(speakers, s.speaker) }))
    .sort((a, b) => (order.get(a.speaker) - order.get(b.speaker))
      || ((a.kind === "self" ? 0 : 1) - (b.kind === "self" ? 0 : 1))
      || (a.start - b.start));
  return { auto, suggestions, groups: groupByVoice(suggestions) };
}

/**
 * The suggestions of each voice together, in the order of `suggestions`
 * (voices in transcript order): `{ label, shown, options }`, the options
 * strongest first (more votes, then more pieces, then self-introductions,
 * then earliest), so the page shows "S2: probably Ruiz — 6 clues" with
 * the other options after it.
 */
export function groupByVoice(suggestions) {
  const groups = new Map();
  for (const s of suggestions) {
    if (!groups.has(s.label)) groups.set(s.label, { label: s.label, shown: s.shown, options: [] });
    groups.get(s.label).options.push(s);
  }
  for (const group of groups.values()) {
    group.options.sort((a, b) => ((b.votes ?? 0) - (a.votes ?? 0))
      || ((b.count ?? 1) - (a.count ?? 1))
      || ((a.kind === "self" ? 0 : 1) - (b.kind === "self" ? 0 : 1))
      || (a.start - b.start));
  }
  return [...groups.values()];
}

/** Whether a finished search changed something worth reloading the transcript for. */
export function searchFinished(before, after) {
  return Boolean(before?.running) && !after?.running;
}

/** The i18n key of a suggestion's reason (an unknown one gets the generic key). */
export function reasonKey(reason) {
  return NAME_REASONS.includes(reason) ? `names.reason.${reason}` : "names.reason.other";
}

/** How long to wait before trying a publish again while names are searched (ms). */
export const BUSY_RETRY_MS = 3000;

/**
 * Publish a session once its job has ended. While the speakers' names are
 * still searched (`busy()` says so, or the publish answers 409 `busy`) the
 * transcript is about to change, so this calls `onWait()`, waits `delay`
 * ms and tries again. Resolves true once published, false if `stopped()`
 * (the page was left); any other error is thrown. `busy` failing counts
 * as not busy: the publish itself decides.
 */
export async function publishWhenFree(publish, {
  busy = async () => false, onWait = () => {}, stopped = () => false,
  delay = BUSY_RETRY_MS, wait = (ms) => new Promise((done) => setTimeout(done, ms)),
} = {}) {
  for (;;) {
    if (stopped()) return false;
    let waiting = await busy().catch(() => false);
    if (!waiting) {
      try {
        await publish();
        return true;
      } catch (error) {
        if (error?.code !== "busy") throw error;
        waiting = true;
      }
    }
    onWait();
    await wait(delay);
  }
}
