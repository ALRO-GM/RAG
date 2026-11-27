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

/**
 * The estimate's rows: with retrieval (what is sent today) and, for each
 * library, without it (the same request with the whole library in place
 * of the retrieved pieces). Each row: { id, tokens, cost, ratio, long }
 * where cost is for `rag.answers` answers and ratio is tokens against the
 * row with retrieval.
 */
export function ragRows(rag) {
  const base = rag.prompt_tokens - rag.knowledge_tokens_in_prompt;
  const row = (id, tokens, label = null, source = null) => ({
    id, label, source, tokens,
    cost: callCost(rag.price, tokens, rag.output_tokens) * rag.answers,
    ratio: tokens / rag.prompt_tokens,
    long: tokens > rag.price.long_prompt_tokens,
  });
  return [
    row("with", rag.prompt_tokens),
    ...rag.libraries.map((lib) => row(lib.id, base + lib.tokens, lib.label, lib.source)),
  ];
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
  for (const lib of data.rag.libraries) add(lib.source, `rag.libraries.${lib.id}`);
  if (data.rag.acts) add(data.rag.acts.source, "rag.acts");
  for (const item of data.content_today ?? []) add(item.source, "content_today");
  return cited;
}
