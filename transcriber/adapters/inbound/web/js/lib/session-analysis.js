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
  "by_rules", "judge_calls", "provisional", "observations",
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
  if (["failed", "interrupted", "partial"].includes(run.status)) return "resume";
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

/** Why a provisional estimate is not firm (the RAG's codes, #103). */
export const REASONS = [
  "few_quotes", "only_confirmations", "single_episode", "weak_match", "no_judge_rating", "short_participation",
];

/** Why a person has few firm levels (the RAG's codes, #103). */
export const WHY = [
  "short_recording", "little_participation", "below_floor", "few_quotes_proposed", "quotes_filtered", "not_shown",
];

/** Where an observation comes from (the RAG's codes, #103). */
export const SOURCES = ["below_minimum", "weak_match", "confirmation"];

/**
 * One cell of the assessment: a firm level, a provisional estimate
 * (labelled, with its confidence) or nothing. Older analyses have no
 * estimates and give firm levels or nothing.
 */
export function cellOf(item) {
  if (typeof item?.rating === "number") return { kind: "firm", rating: item.rating };
  if (item?.status === "provisional" && typeof item.estimate?.rating === "number") {
    return { kind: "provisional", rating: item.estimate.rating, confidence: item.estimate.confidence ?? "low" };
  }
  return { kind: "none", rating: null };
}

/** The text of a cell: "4", "~3" (provisional) or "–". */
export function cellText(cell) {
  if (cell.kind === "firm") return String(cell.rating);
  if (cell.kind === "provisional") return `~${cell.rating}`;
  return "–";
}

/** Rows of the assessment: one per person, competency → firm rating or null, and every cell. */
export function ratingRows(assessment) {
  const cards = assessment?.scorecards ?? [];
  const competencies = [];
  for (const card of cards) {
    for (const a of card.assessments ?? []) if (!competencies.includes(a.competency_id)) competencies.push(a.competency_id);
  }
  return {
    competencies,
    people: cards.map((card) => {
      const items = competencies.map((id) => (card.assessments ?? []).find((a) => a.competency_id === id));
      return { name: card.participant, ratings: items.map((a) => a?.rating ?? null), cells: items.map(cellOf) };
    }),
  };
}

/**
 * What a reviewer should check of one competency, in speaking order:
 * nothing for a firm level; otherwise its quotes below the minimum and
 * its observations (as the RAG's `CompetencyAssessment.observed`).
 */
export function observed(item) {
  if (!item || item.status === "rated" || typeof item.rating === "number") return [];
  const quotes = (item.evidence ?? []).map((e) => ({ ...e, source: "below_minimum" }));
  return [...quotes, ...(item.observations ?? [])].sort((a, b) => (a.turn ?? 0) - (b.turn ?? 0));
}

/** Per person: how many firm levels, provisional estimates and observations, and what to review. */
export function reviewRows(assessment) {
  return (assessment?.scorecards ?? []).map((card) => {
    const items = card.assessments ?? [];
    const review = items
      .map((a) => ({
        competency: a.competency_id,
        estimate: a.status === "provisional" ? a.estimate ?? null : null,
        observations: observed(a),
      }))
      .filter((r) => r.estimate || r.observations.length);
    return {
      name: card.participant,
      words: card.interaction?.words ?? 0,
      firm: items.filter((a) => typeof a.rating === "number").length,
      provisional: review.filter((r) => r.estimate).length,
      observations: review.reduce((n, r) => n + r.observations.length, 0),
      review,
    };
  });
}

/** Why an estimate is not firm, in words. */
export function reasonText(t, code, estimate) {
  if (!REASONS.includes(code)) return code;
  return t(`sa.reason_${code}`, { quotes: estimate?.quotes ?? 0, needed: estimate?.needed ?? 0, words: estimate?.words ?? 0 });
}

/** Why a person has few firm levels, in words. */
export function whyText(t, code, person, needed) {
  if (!WHY.includes(code)) return code;
  return t(`sa.why_${code}`, { name: person?.name ?? "", words: person?.words ?? 0, needed: needed ?? 0 });
}

/** The page of the transcript at a moment (whole seconds), or the transcript itself. */
export function transcriptLink(session, seconds) {
  const base = `#/s/${encodeURIComponent(session)}`;
  return typeof seconds === "number" && Number.isFinite(seconds) && seconds >= 0 ? `${base}/t/${Math.floor(seconds)}` : base;
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

/**
 * The report as a file to save: never `text/html`, so the browser can
 * only download it and never render it in the pages' origin.
 */
export function reportBlob(page) {
  return new Blob([page], { type: "application/octet-stream" });
}

/** How long after a start a missing record still means "starting" (ms). */
export const STARTING_MS = 60000;

/** Whether a start made at `since` (ms, 0: none) is recent enough to wait for. */
export function stillStarting(since, now) {
  return since > 0 && now - since < STARTING_MS;
}

/** The run shown between a start and the worker's first record. */
export function startingRun(id) {
  return { session_id: id, status: "running", active: true, steps: [], open_issues: [] };
}

/** What to tell when a start is refused, by the transcriber's code. */
export function startRefusal(t, error, describe) {
  if (error?.code === "not_found") return t("sa.not_published");
  if (error?.code === "limit_reached") return t("sa.limit");
  if (error?.code === "wrong_state") return t("sa.busy");
  return describe(error);
}
