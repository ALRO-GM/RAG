// The candidate's answers to the core questions (#235): pure helpers for
// the live page's «Preguntas comunes» row and the result page's «Datos de
// la candidata» block. A profile is the orchestrator's view of
// `interview.candidate.v1`: { id, plan_id, name, questions: [{ id, order,
// question: { es, en }, fields, answer }], answered, rounds }. No DOM here.

/** The core questions the dictionaries have a short label for (cand.q.<id>). */
export const CORE_IDS = ["work_mode_location", "work_authorization", "start_date_notice", "motivation", "salary_expectation"];

/** A whole number with the thousands grouped as the language writes it. */
export function grouped(value, language = "es") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  return new Intl.NumberFormat(language === "en" ? "en-GB" : "es-ES", { maximumFractionDigits: 0, useGrouping: true }).format(value);
}

/**
 * The answer of one question in a few words, or "" when there is none.
 * Yes/no answers are words of the dictionary; the salary a figure or a
 * range in euros per year; the start date the candidate's own phrase and
 * the notice in days; the motivation its kept words.
 */
export function answerText(t, row, language = "es") {
  const answer = row?.answer;
  if (!answer) return "";
  const fields = answer.fields ?? {};
  const yesNo = Object.values(fields).find((v) => typeof v === "boolean");
  if (typeof yesNo === "boolean") return t(yesNo ? "cand.yes" : "cand.no");
  if (row.id === "salary_expectation") {
    const low = fields["salary.expectation_min"];
    const high = fields["salary.expectation_max"];
    if (typeof low === "number" && typeof high === "number") {
      return low === high
        ? t("cand.salary_one", { value: grouped(low, language) })
        : t("cand.salary_range", { low: grouped(low, language), high: grouped(high, language) });
    }
  }
  if (row.id === "start_date_notice") {
    const start = typeof fields["availability.earliest_start"] === "string" ? fields["availability.earliest_start"] : "";
    const days = fields["availability.notice_days"];
    const notice = typeof days === "number" ? (days === 0 ? t("cand.no_notice") : t("cand.notice_days", { days })) : "";
    const text = [start, notice].filter(Boolean).join(" · ");
    if (text) return text;
  }
  return String(answer.summary ?? "").trim();
}

/** The question's short label (the dictionary's, else its own words). */
export function questionLabel(t, row, language = "es") {
  if (CORE_IDS.includes(row?.id)) return t(`cand.q.${row.id}`);
  return row?.question?.[language] ?? row?.question?.es ?? row?.id ?? "";
}

/**
 * The rows of a profile, in the questions' order: label, whether it is
 * answered, the answer in words, who gave it (captured or corrected),
 * the candidate's words kept (never for a yes/no) and the session.
 */
export function coreRows(t, profile, language = "es") {
  return [...(profile?.questions ?? [])]
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((row) => ({
      id: row.id,
      label: questionLabel(t, row, language),
      question: row.question?.[language] ?? row.question?.es ?? "",
      answered: Boolean(row.answer),
      text: answerText(t, row, language),
      source: row.answer?.source === "corrected" ? "corrected" : row.answer ? "captured" : "",
      quote: String(row.answer?.quote ?? ""),
      session: row.answer?.session_id ?? null,
    }));
}

/** How many core questions are answered, and how many there are. */
export function coreCount(profile) {
  const questions = profile?.questions ?? [];
  return { done: questions.filter((q) => q.answer).length, all: questions.length };
}
