// Pure logic of the study page (study.html): checks the data file, names
// the combinations, lays out the bar lists and says what is still pending.
// No DOM and no text for people here (that comes from the dictionaries),
// so node --test can check it.

/** The data file's schema this page understands. */
export const SCHEMA = "study.v1";

/** Whether `data` looks like a study.v1 file the page can draw. */
export function isStudyData(data) {
  return Boolean(data) && data.schema === SCHEMA
    && Array.isArray(data.separation?.corpora) && data.separation.corpora.length > 0
    && data.separation.corpora.every((c) => Array.isArray(c.rows) && c.rows.length > 0)
    && Array.isArray(data.labelling?.judges)
    && typeof data.sources === "object" && data.sources !== null;
}

/** A linear map from [d0, d1] to [r0, r1]. */
export function linear([d0, d1], [r0, r1]) {
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

/** Evenly spaced ticks from lo to hi inclusive. */
export function ticks(lo, hi, step) {
  const out = [];
  const n = Math.round((hi - lo) / step);
  for (let i = 0; i <= n; i += 1) out.push(Number((lo + i * step).toFixed(10)));
  return out;
}

/** The denoiser's name, or `noneLabel` for the original audio. */
export function denoiserName(row, noneLabel) {
  return row.denoiser === "none" ? noneLabel : row.denoiser_name;
}

/** The diarizer's name, or `builtinLabel` for the transcriber's own one. */
export function diarizerName(row, builtinLabel) {
  return row.diarizer === "builtin" && builtinLabel ? builtinLabel : row.diarizer_name;
}

/** "DiariZen · sin limpiar (NC)": diarizer, denoiser, licence flag. */
export function comboLabel(row, noneLabel, builtinLabel) {
  const base = `${diarizerName(row, builtinLabel)} · ${denoiserName(row, noneLabel)}`;
  return row.non_commercial ? `${base} (NC)` : base;
}

/** The row of a corpus with this id, or undefined. */
export function rowById(corpus, id) {
  return corpus.rows.find((r) => r.id === id);
}

/** Series class of a bar: the best, the system in use today, or the rest. */
export function barKind(row) {
  if (row.best) return "best";
  if (row.builtin) return "builtin";
  return "other";
}

/** A fraction as percentage points (0.0402 -> 4.02). */
export function points(fraction) {
  return fraction * 100;
}

/**
 * Vertical layout of a bar list with the label above each bar (so long
 * names fit a phone). Returns each row's label and bar position and the
 * total height.
 */
export function layoutBars(count, { top = 26, labelH = 16, barH = 14, gap = 10 } = {}) {
  const rows = [];
  let y = top;
  for (let i = 0; i < count; i += 1) {
    rows.push({ labelY: y + 12, barY: y + labelH });
    y += labelH + barH + gap;
  }
  return { rows, height: y };
}

/** "pending" until the block has its measured results, then "done". */
export function status(block) {
  return block?.status === "done" ? "done" : "pending";
}

/** The best judge measured on the full sample (small samples never win). */
export function bestJudge(judges) {
  return judges.filter((j) => !j.small_sample)
    .reduce((best, j) => (best === null || j.accuracy > best.accuracy ? j : best), null);
}

/**
 * The best judge on the full sample against the next best and the first
 * one measured: { best, next, first } (next/first null when absent or the
 * same as the best).
 */
export function judgeHeadline(judges) {
  const best = bestJudge(judges);
  if (best === null) return { best: null, next: null, first: null };
  const full = judges.filter((j) => !j.small_sample && j !== best);
  const next = full.reduce((top, j) => (top === null || j.accuracy > top.accuracy ? j : top), null);
  const first = judges.find((j) => !j.small_sample) ?? null;
  return { best, next, first: first === best ? null : first };
}

/** "LAB.md §13.8.1" for a source with a numbered section, else "README.md, Contents". */
export function sourceLabel(source) {
  const file = String(source.path).split("/").pop();
  const number = String(source.section).match(/^(\d+(?:\.\d+)*)\.?\s/);
  return number ? `${file} §${number[1]}` : `${file}, ${String(source.section).split(" (")[0]}`;
}

/** Sum of the expected, derived and matched events of rows (for a check). */
export function eventTotals(rows) {
  return rows.reduce((acc, r) => ({
    expected: acc.expected + r.expected, derived: acc.derived + r.derived, matched: acc.matched + r.matched,
  }), { expected: 0, derived: 0, matched: 0 });
}
