// Pure logic of the transcript player.

const ANONYMOUS = /^S[0-9]+$/;
export const DEFAULT_LEAD = 3;

/** Utterances ordered by start (stable), as the player needs them. */
export function sortUtterances(utterances) {
  return utterances
    .map((u, i) => [u, i])
    .sort((a, b) => a[0].start - b[0].start || a[1] - b[1])
    .map((pair) => pair[0]);
}

/** Time to set the audio to for an utterance: `lead` seconds earlier, not before 0. */
export function seekTime(start, lead) {
  const l = Number.isFinite(lead) && lead > 0 ? lead : 0;
  return Math.max(0, start - l);
}

/** Read the lead field: a non-negative number, else the default. */
export function parseLead(text, fallback = DEFAULT_LEAD) {
  const n = Number.parseFloat(String(text).replace(",", "."));
  return Number.isFinite(n) && n >= 0 ? Math.min(n, 60) : fallback;
}

/**
 * Index of the utterance heard at `time`: the last one that started at or
 * before it (binary search over the sorted `starts`), or -1 before the
 * first. `ends` (optional) lets a long silence after an utterance clear it.
 */
export function activeIndex(starts, time, ends = null, grace = 2) {
  let lo = 0;
  let hi = starts.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] <= time) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (found >= 0 && ends && time > ends[found] + grace) return -1;
  return found;
}

/** Display name of a speaker id: its name, else its label. */
export function speakerName(speakers, id) {
  const s = speakers.find((x) => x.id === id);
  return s?.name || id;
}

/** Utterances of one speaker; a falsy `speakerId` means everybody. */
export function filterBySpeaker(utterances, speakerId) {
  return speakerId ? utterances.filter((u) => u.speaker === speakerId) : utterances;
}

/**
 * True when the attribution of an utterance should be checked: the voice
 * evidence called it doubtful, or heard two voices in it. Older
 * transcripts lack both fields: never doubtful.
 */
export function needsReview(utterance) {
  return utterance.doubtful === true || utterance.overlap === true;
}

/** Utterances to show: everybody's or one speaker's, optionally only doubtful ones. */
export function filterUtterances(utterances, speakerId, onlyDoubtful = false) {
  const bySpeaker = filterBySpeaker(utterances, speakerId);
  return onlyDoubtful ? bySpeaker.filter(needsReview) : bySpeaker;
}

/**
 * What to tell about an utterance's attribution, or null when nothing:
 * { doubtful, overlap, alternative (a display name or null), certainty }.
 */
export function doubtNote(utterance, speakers) {
  if (!needsReview(utterance)) return null;
  const alt = utterance.alternative_speaker;
  return {
    doubtful: utterance.doubtful === true,
    overlap: utterance.overlap === true,
    alternative: alt ? speakerName(speakers, alt) : null,
    certainty: Number.isFinite(utterance.speaker_certainty) ? utterance.speaker_certainty : null,
  };
}

/** True for a voice the person can still name or merge (S1, S2... not enrolled). */
export function isAnonymous(speaker) {
  return !speaker.enrolled && ANONYMOUS.test(speaker.id);
}

/** Enrolled people a voice can be merged into (transcript speakers + known). */
export function mergeTargets(speakers, enrolledList, exceptId) {
  const seen = new Map();
  for (const s of speakers) {
    if (s.enrolled && s.id !== exceptId) seen.set(s.id, s.name || s.id);
  }
  for (const s of enrolledList) {
    if (s.id !== exceptId && !seen.has(s.id)) seen.set(s.id, s.name || s.id);
  }
  return [...seen].map(([id, name]) => ({ id, name }));
}

/** Name for the "after X" hint, or null in the first turn / same speaker. */
export function previousHint(utterance, speakers) {
  if (!utterance.previous_speaker || utterance.previous_speaker === utterance.speaker) return null;
  return speakerName(speakers, utterance.previous_speaker);
}
