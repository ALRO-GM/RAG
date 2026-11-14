// Pure logic of the research page's charts: scales, grouping, layout,
// the Spearman-Brown projection, detail texts and the reference filter.
// No DOM here, so node --test can check it.

/** Statistic kinds a panel may hold; a figure of another kind is an error. */
export const PANEL_KINDS = Object.freeze({
  kappa: ["κ de Cohen", "κ de Fleiss"],
  icc: ["ICC", "ICC1", "ICC2", "ICC(1,1)"],
  g: ["Eρ²", "G(q,1)"],
  rho_rr: ["ρrr"],
  corr: ["r", "r̄", "ρ de Spearman", "G(q,1)", "ICC(1,1)", "r (índice C1 convergente)"],
  auroc: ["AUROC (micro-media)"],
  rho_op: ["ρ"],
  r: ["r", "r̄"],
});

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

/** ".80", ".286", "−.26", "1": the APA style the sources use. */
export function fmt(value) {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  if (Math.abs(value) >= 1) return String(value);
  let text = value.toFixed(3).replace(/0+$/, "");
  if (text.split(".")[1].length < 2) text = value.toFixed(2);
  return text.replace(/^(-?)0\./, "$1.").replace(/^-/, "−");
}

/** The panels of one chart, in file order. */
export function panelsOf(data, chart) {
  return data.panels.filter((p) => p.chart === chart);
}

/** The figures of one panel, in file order. */
export function figuresOf(data, panelId) {
  return data.figures.filter((f) => f.panel === panelId);
}

/** Figures whose statistic does not belong to their panel's kind. */
export function mixedStatistics(data) {
  const kinds = new Map(data.panels.map((p) => [p.id, p.kind]));
  return data.figures.filter((f) => !(PANEL_KINDS[kinds.get(f.panel)] ?? []).includes(f.statistic));
}

/** Group items by key(item), keeping first-seen order of the groups. */
export function groupBy(items, key) {
  const groups = new Map();
  for (const item of items) {
    const k = key(item);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(item);
  }
  return [...groups].map(([name, members]) => ({ name, members }));
}

/** CSS class of a mark from its source's verification level. */
export function verifiedClass(level) {
  return level === "full-text" ? "solid" : level === "abstract-only" ? "weak abstract" : "weak secondary";
}

/** Reliability of one situation implied by `value` over `k` situations. */
export function singleFromAggregate(value, k) {
  return value / (k - (k - 1) * value);
}

/** Spearman-Brown: reliability of the mean of n parallel situations. */
export function spearmanBrown(single, n) {
  return (n * single) / (1 + (n - 1) * single);
}

/** Smallest n whose projected reliability reaches target. */
export function situationsFor(single, target, max = 50) {
  for (let n = 1; n <= max; n += 1) if (spearmanBrown(single, n) >= target) return n;
  return null;
}

/**
 * Vertical layout of a horizontal bar list with grouped rows.
 * rows: [{ group?: string, figure, note?: false | string }]; returns [{ kind: "group"|"bar", y, ... }]
 * and the total height. Labels sit above their bar so they fit on a phone.
 */
export function layoutRows(rows, { top = 28, groupGap = 30, labelH = 18, barH = 14, noteH = 16, gap = 10 } = {}) {
  const out = [];
  let y = top;
  let last = null;
  for (const row of rows) {
    if (row.group && row.group !== last) {
      out.push({ kind: "group", y: y + 14, text: row.group });
      y += groupGap;
      last = row.group;
    }
    const note = row.note === false ? 0 : noteH;
    out.push({ kind: "bar", labelY: y + 13, barY: y + labelH, noteY: note ? y + labelH + barH + 12 : null, figure: row.figure, series: row.series, note: row.note });
    y += labelH + barH + note + gap;
  }
  return { items: out, height: y + 4 };
}

/** One line under a bar: statistic, who rated, sample, source. */
export function noteLine(f) {
  return [f.statistic, f.raters, f.sample, `${f.cite}`].filter(Boolean).join(" · ");
}

/** The detail text shown for a focused or hovered bar. */
export function figureDetail(f, verifiedLabels) {
  const parts = [
    `${f.label ?? f.method}: ${fmt(f.value)} (${f.statistic})`,
    f.raters ? `Quién puntúa: ${f.raters}` : null,
    f.sample ? `Muestra: ${f.sample}` : null,
    f.sd !== undefined ? `DT de ρ: ${fmt(f.sd)}` : null,
    `Fuente: ${f.cite}, ${f.location}`,
    `Leído: ${verifiedLabels[f.verified] ?? f.verified}`,
  ];
  return parts.filter(Boolean).join(". ") + ".";
}

/** The detail text for one scored entry of the map. */
export function entryDetail(e, labels) {
  return `${e.name}. A ${e.A}, B ${e.B}, C ${e.C}. Confianza ${labels.confidence[e.confidence]}. ` +
    `Veredicto: ${labels.verdict[e.verdict]}.${e.popular_but_weak ? " Popular pero débil." : ""}`;
}

/** Fold accents and case for searching. */
export function fold(text) {
  return String(text ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/** Ids of the sources that match the search text, family and level. */
export function filterSources(sources, { query = "", family = "", verified = "" } = {}) {
  const words = fold(query).split(/\s+/).filter(Boolean);
  return sources.filter((s) => {
    if (family && !s.families.includes(family)) return false;
    if (verified && s.verified !== verified) return false;
    const hay = fold([s.cite, s.title, s.venue, s.authors, s.doi].join(" "));
    return words.every((w) => hay.includes(w));
  }).map((s) => s.id);
}

/** Families present in the sources, in the order of the labels. */
export function familiesOf(sources, familyLabels) {
  const present = new Set(sources.flatMap((s) => s.families));
  return Object.keys(familyLabels).filter((f) => present.has(f));
}

/** Entries of one family, highest C first. */
export function entriesOf(entries, family) {
  return entries.filter((e) => e.family === family).sort((a, b) => b.C - a.C || a.name.localeCompare(b.name));
}

/** Shorten a label to max characters on a word boundary. */
export function clip(text, max) {
  const s = String(text ?? "");
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1);
  const space = cut.lastIndexOf(" ");
  return `${space > max / 2 ? cut.slice(0, space) : cut}…`;
}
