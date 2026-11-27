// Pure logic of the «Info» page (info.html; «Competencia» in #291, renamed
// in #295): checks the data file, picks a text in the page's language,
// groups the competitors, computes the «con RAG / sin RAG» estimate, the
// cost of the same service per scenario and the task-by-task comparison
// from competition/data.json. No DOM and no text for people here, so
// node --test can check it.

/** The data file's schema this page understands. */
export const SCHEMA = "competition.v1";

/** What each competitor is for, in the order the page shows them. */
export const KINDS = ["callcentre", "interviews", "voicebot"];

/** Where a model or a product runs. */
export const WHERE = ["local", "external", "both"];

/** How a competitor's price was seen: vendor page, marketplace listing, or not public. */
export const BASES = ["official", "marketplace", "none"];

/** A {es, en} text in `lang`, falling back to Spanish. */
export function textOf(value, lang) {
  if (typeof value === "string") return value;
  return value?.[lang] ?? value?.es ?? "";
}

/** Whether `value` is a text in both languages. */
export function isBilingual(value) {
  return Boolean(value) && typeof value.es === "string" && value.es.trim() !== ""
    && typeof value.en === "string" && value.en.trim() !== "";
}

/** Whether `data` looks like a competition.v1 file the page can draw. */
export function isCompetitionData(data) {
  return Boolean(data) && data.schema === SCHEMA
    && typeof data.sources === "object" && data.sources !== null
    && Array.isArray(data.steps) && data.steps.length > 0
    && Array.isArray(data.competitors) && data.competitors.length > 0
    && Array.isArray(data.models) && data.models.length > 0
    && Boolean(data.rag?.price) && Array.isArray(data.rag?.libraries)
    && Array.isArray(data.tasks) && data.tasks.length > 0
    && Array.isArray(data.cost?.scenarios) && data.cost.scenarios.length > 0;
}

/** The competitors of one kind, in file order. */
export function competitorsOf(competitors, kind) {
  return competitors.filter((c) => c.kind === kind);
}

/** Dollars of one call to the model with `input` and `output` tokens, at list price. */
export function callCost(price, input, output) {
  const long = input > price.long_prompt_tokens;
  const inRate = long ? price.long_input_per_mtok : price.input_per_mtok;
  const outRate = long ? price.long_output_per_mtok : price.output_per_mtok;
  return (input * inRate + output * outRate) / 1e6;
}

/** One row of the estimate: tokens, cost of `rag.answers` answers, ratio against `against`. */
function estimateRow(rag, id, tokens, against, label = null, source = null) {
  return {
    id, label, source, tokens,
    cost: callCost(rag.price, tokens, rag.output_tokens) * rag.answers,
    ratio: tokens / against,
    long: tokens > rag.price.long_prompt_tokens,
  };
}

/** The part of the measured prompt that is not retrieved pieces. */
export function promptBase(rag) {
  return rag.prompt_tokens - rag.knowledge_tokens_in_prompt;
}

/**
 * The estimate on the sample libraries: the measured prompt with
 * retrieval (3 short pieces) and, for each library that is not a
 * maximum, the same request with the whole library in place of the
 * pieces. Each row: { id, label, tokens, cost, ratio, long }, ratio
 * against the measured prompt.
 */
export function ragRows(rag) {
  return [
    estimateRow(rag, "with", rag.prompt_tokens, rag.prompt_tokens),
    ...rag.libraries.filter((lib) => !lib.maximum)
      .map((lib) => estimateRow(rag, lib.id, promptBase(rag) + lib.tokens, rag.prompt_tokens, lib.label, lib.source)),
  ];
}

/**
 * The estimate on a library given at its maximum size (every piece full):
 * with retrieval, the products' `top_k` full pieces; without, the whole
 * library. Both sides use the same piece size; ratio against the first.
 */
export function ragMaxRows(rag, lib) {
  const withTokens = promptBase(rag) + (rag.top_k * rag.piece_chars) / rag.chars_per_token;
  return [
    estimateRow(rag, `${lib.id}_with`, withTokens, withTokens, lib.label, lib.source),
    estimateRow(rag, lib.id, promptBase(rag) + lib.tokens, withTokens, lib.label, lib.source),
  ];
}

/** The libraries given at their maximum size. */
export function maximumLibraries(rag) {
  return rag.libraries.filter((lib) => lib.maximum);
}

/** Decimals to show a dollar amount with (0.22 -> 2, 12.3 -> 1, 130 -> 0). */
export function digitsFor(value) {
  if (value >= 100) return 0;
  return value >= 10 ? 1 : 2;
}

/** Every source id the data file cites, with where it is cited. */
export function citedSources(data) {
  const cited = [];
  const add = (id, where) => cited.push({ id, where });
  for (const c of data.competitors) add(c.source, c.name);
  if (data.ours?.source) add(data.ours.source, "ours");
  add(data.rag.price.source, "rag.price");
  add(data.rag.prompt_source, "rag.prompt_tokens");
  add(data.rag.knowledge_source, "rag.knowledge_tokens_in_prompt");
  add(data.rag.timing_source, "rag.timing");
  add(data.rag.output_source, "rag.output_tokens");
  add(data.rag.top_k_source, "rag.top_k");
  for (const lib of data.rag.libraries) add(lib.source, `rag.libraries.${lib.id}`);
  if (data.rag.acts) add(data.rag.acts.source, "rag.acts");
  for (const item of data.content_today ?? []) add(item.source, "content_today");
  return cited;
}

// ---- «Info» (#295): tasks, the cost of the same service, one company ----

/** How a company answers one task: yes, in part, no, or not published / not checked. */
export const LEVELS = ["yes", "partial", "no", "unknown"];

/** Order of a level when two companies are compared (unknown is not comparable). */
const RANK = { yes: 2, partial: 1, no: 0 };

/** The rows of every chart, in the same order: us first, then the competitors in file order. */
export function companies(data) {
  return [{ id: "ours", ours: true, name: data.ours.name }, ...data.competitors.map((c) => ({ id: c.id, ours: false, name: c.name, kind: c.kind }))];
}

/** The cell of company `id` in `task`; a company the file does not fill is «no publicado». */
export function cellOf(task, id) {
  if (id === "ours") return task.ours;
  return task.cells?.[id] ?? { level: "unknown" };
}

/** How many competitors have the task (yes or in part), out of how many. */
export function haveIt(task, competitors) {
  const n = competitors.filter((c) => ["yes", "partial"].includes(cellOf(task, c.id).level)).length;
  return { n, of: competitors.length };
}

/** People, hours, calls and minutes of a month in one scenario. */
export function scenarioOf(cost, scenario) {
  const hours = scenario.agents * cost.hours_per_agent;
  return { id: scenario.id, agents: scenario.agents, hours, minutes: hours * 60, calls: (hours * 60) / cost.call_minutes };
}

/** Dollars a month of one price plan (`parts`) in scenario `sc`. */
export function planCost(parts, sc, minSeats = 0) {
  let total = 0;
  if (parts.seat) total += parts.seat * Math.max(sc.agents, minSeats);
  if (parts.seat_year) total += (parts.seat_year * Math.max(sc.agents, minSeats)) / 12;
  if (parts.minute) total += parts.minute * sc.minutes;
  if (parts.hour) total += parts.hour * sc.hours;
  if (parts.call) total += parts.call * sc.calls;
  if (parts.year) total += (parts.year + Math.max(0, sc.calls * 12 - (parts.year_calls ?? 0)) * (parts.extra_call ?? 0)) / 12;
  return total;
}

/** A competitor's published price applied to scenario `sc`: { low, high }, or null when it has none. */
export function competitorCost(cost, id, sc) {
  const spec = cost.competitors[id];
  if (!spec) return null;
  const a = planCost(spec.low, sc, spec.min_seats);
  const b = planCost(spec.high, sc, spec.min_seats);
  return { low: Math.min(a, b), high: Math.max(a, b), more: Boolean(spec.more), unclear: Boolean(spec.unclear) };
}

/** Our running cost per hour of conversation: live suggestions plus the analysis after the call, at `price`. */
export function oursPerHour(cost, rag, price = rag.price) {
  const o = cost.ours;
  const answersPerHour = (o.live_calls / o.live_seconds) * 3600;
  const live = answersPerHour * callCost(price, rag.prompt_tokens, rag.output_tokens);
  const scale = 3600 / o.analysis_seconds;
  const analysis = (o.analysis_input_tokens * scale * price.input_per_mtok + o.analysis_output_tokens * scale * price.output_per_mtok) / 1e6;
  return { answersPerHour, live, analysis, total: live + analysis };
}

/** Our estimated cost a month in scenario `sc`: Haiku with the measured tokens (low), plus the token margin (high), and the same with Opus for the analysis. */
export function oursCost(cost, rag, sc) {
  const haiku = oursPerHour(cost, rag);
  const opusPrice = { ...rag.price, ...cost.ours.opus };
  const opus = haiku.live + oursPerHour(cost, rag, opusPrice).analysis;
  return { low: haiku.total * sc.hours, high: haiku.total * sc.hours * cost.ours.token_margin, opus: opus * sc.hours, perHour: haiku };
}

/**
 * Us against one competitor, task by task: where we do more (their
 * answer is known and lower), where they publish nothing, where they do
 * more, and what both do. Each list holds task ids in file order.
 */
export function versus(data, id) {
  const out = { weMore: [], unknown: [], theyMore: [], both: [] };
  for (const task of data.tasks) {
    const ours = cellOf(task, "ours").level;
    const theirs = cellOf(task, id).level;
    if (theirs === "unknown") {
      if (RANK[ours] > 0) out.unknown.push(task.id);
    } else if (RANK[theirs] > RANK[ours]) out.theyMore.push(task.id);
    else if (RANK[ours] > RANK[theirs]) out.weMore.push(task.id);
    else if (RANK[ours] > 0) out.both.push(task.id);
  }
  return out;
}

/** Every source the #295 parts of the file cite, with where they are cited. */
export function citedInfoSources(data) {
  const cited = [];
  for (const task of data.tasks ?? []) {
    cited.push({ id: task.ours.source, where: `tasks.${task.id}.ours` });
    for (const [id, cell] of Object.entries(task.cells ?? {})) if (cell.source) cited.push({ id: cell.source, where: `tasks.${task.id}.${id}` });
  }
  for (const c of data.competitors) for (const offer of c.offers ?? []) cited.push({ id: offer.source, where: `${c.name}.offers` });
  const o = data.cost?.ours;
  if (o) for (const key of ["live_source", "analysis_source", "analysis_seconds_source", "token_margin_source"]) cited.push({ id: o[key], where: `cost.ours.${key}` });
  if (o?.opus) cited.push({ id: o.opus.source, where: "cost.ours.opus" });
  for (const [id, spec] of Object.entries(data.cost?.competitors ?? {})) for (const s of spec.also ?? []) cited.push({ id: s, where: `cost.${id}` });
  return cited;
}
