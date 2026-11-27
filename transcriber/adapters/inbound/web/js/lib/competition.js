// Pure logic of the «Competencia» page (competition.html, #291): checks
// the data file, picks a text in the page's language, groups the
// competitors and computes the «con RAG / sin RAG» estimate from the
// measured inputs of competition/data.json. No DOM and no text for people
// here, so node --test can check it.

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
    && Boolean(data.rag?.price) && Array.isArray(data.rag?.libraries);
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
