// The "Análisis" page of a session: pure helpers (no DOM), tested in node.
// The RAG runs the steps (docs/skills/DESIGN.md §12); the page shows their
// records and results as the transcriber forwards them.

/** The steps in the order they run (the RAG's names). */
export const STEPS = [
  "sentences", "interaction", "candidates", "acts", "events",
  "confirmation", "assessment", "reports",
];

/** Step statuses the page knows; anything else is shown as "pending". */
export const STATUSES = ["pending", "running", "done", "failed", "unavailable"];

/** Counts with a translated label; others are shown with their own key. */
export const COUNT_KEYS = [
  "sentences", "speakers", "turns", "words", "with_candidates", "candidates",
  "sent", "not_sent", "confirmed", "rejected", "unparsed", "participants",
  "judgements", "rated", "evidence", "reports", "doubtful", "by_encoder",
  "by_rules",
];

/** The 13 dialogue-act categories (their names are in the dictionaries). */
export const ACTS = [
  "pregunta", "respuesta", "informacion", "opinion", "propuesta", "peticion",
  "aceptacion", "rechazo", "compromiso", "escucha", "gestion", "relacion", "otro",
];

/** The things people must still do, by the RAG's ids. */
export const OPEN_ISSUES = ["consent", "human_review", "situations"];

/** Whether the page should keep asking the RAG (the run is still going). */
export function isActive(run) {
  return Boolean(run?.active) || run?.status === "running";
}

/** The status to show for a step record. */
export function stepStatus(step) {
  return STATUSES.includes(step?.status) ? step.status : "pending";
}

/** The steps of a run in order, an empty record for any the RAG left out. */
export function orderedSteps(run) {
  const byName = new Map((run?.steps ?? []).map((s) => [s.name, s]));
  return STEPS.map((name) => byName.get(name) ?? { name, status: "pending", counts: {} });
}

/** What the start button should do: start, resume after a failure, or nothing. */
export function startMode(run) {
  if (!run) return "start";
  if (isActive(run)) return "running";
  if (run.status === "failed" || run.status === "interrupted") return "resume";
  return "again";
}

/** "1:05" from seconds (minutes:seconds), "" when unknown. */
export function clock(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Sentences grouped by speaker, in order of first appearance. */
export function byPerson(sentences) {
  const groups = new Map();
  for (const s of sentences ?? []) {
    if (!groups.has(s.speaker)) groups.set(s.speaker, []);
    groups.get(s.speaker).push(s);
  }
  return [...groups].map(([speaker, items]) => ({ speaker, items }));
}

/**
 * The judge's verdict on a tagged sentence: "confirmed" (with the
 * indicator), "rejected", "unparsed", "not_sent" (it had candidates the
 * judge did not read) or "none" (no candidates at all).
 */
export function verdict(item) {
  const decision = item?.decision;
  if (decision?.status === "confirmed") return { kind: "confirmed", indicator: decision.indicator_id, polarity: decision.polarity };
  if (decision?.status === "rejected") return { kind: "rejected" };
  if (decision?.status === "unparsed") return { kind: "unparsed" };
  return { kind: (item?.tags ?? []).length ? "not_sent" : "none" };
}

/** A candidate tag as a short chip: "+LISTEN.1 0.62". */
export function tagChip(tag, format = (x) => x.toFixed(2)) {
  const sign = tag.polarity === "negative" ? "−" : "+";
  const score = typeof tag.similarity === "number" ? ` ${format(tag.similarity)}` : "";
  return `${sign}${tag.indicator_id}${score}`;
}

/** The sentences worth listing for a tags step: only those with candidates. */
export function tagged(result) {
  return (result?.sentences ?? []).filter((s) => (s.tags ?? []).length);
}

/** Rows of the assessment: one per person, competency → rating or null. */
export function ratingRows(assessment) {
  const cards = assessment?.scorecards ?? [];
  const competencies = [];
  for (const card of cards) {
    for (const a of card.assessments ?? []) if (!competencies.includes(a.competency_id)) competencies.push(a.competency_id);
  }
  return {
    competencies,
    people: cards.map((card) => ({
      name: card.participant,
      ratings: competencies.map((id) => (card.assessments ?? []).find((a) => a.competency_id === id)?.rating ?? null),
    })),
  };
}

/** The sentences of an acts.v1 document with their weighted labels (best first). */
export function actRows(acts) {
  return (acts?.utterances ?? []).map((u) => ({
    id: u.id, speaker: u.speaker, text: u.text, start: u.start, by: u.by ?? "",
    labels: (u.labels ?? (u.act ? [{ label: u.act, weight: 100 }] : [])).map((l) => ({
      label: l.label, weight: l.weight, doubtful: Boolean(l.doubtful),
    })),
  }));
}

/** How many sentences have a doubtful first label. */
export function doubtfulCount(acts) {
  return actRows(acts).filter((r) => r.labels[0]?.doubtful).length;
}

/** The events of step e, in spoken order. */
export function eventRows(events) {
  return [...(events?.events ?? [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}

/** The file name a downloaded report is saved under. */
export function reportFileName(session, person) {
  const safe = String(person?.name ?? "").normalize("NFKD").replace(/\p{M}+/gu, "").replace(/[^\w-]+/g, "-").replace(/^-+|-+$/g, "");
  return `${session}-${person?.position ?? 0}${safe ? `-${safe}` : ""}.html`;
}
