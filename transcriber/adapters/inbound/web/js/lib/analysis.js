// The analysis judge control: pure helpers (no DOM), tested in node.

export const JUDGE_MODES = ["claude_code", "local", "both"];

/**
 * When the quota comes back, for people: the exact moment in local time
 * when the service gave one (`quota_resets_at`), else its own words.
 */
export function resetText(status, locale) {
  const at = status?.quota_resets_at ? new Date(status.quota_resets_at) : null;
  if (at && !Number.isNaN(at.getTime())) {
    return at.toLocaleString(locale, { dateStyle: "short", timeStyle: "short" });
  }
  return status?.quota_resets || "";
}

/** The quota line under the control: "available", "exhausted" or "unknown". */
export function quotaLine(t, status, locale) {
  const quota = JUDGE_QUOTA.includes(status?.claude_quota) ? status.claude_quota : "unknown";
  const when = quota === "exhausted" ? resetText(status, locale) : "";
  const resets = when ? ` ${t("analysis.resets", { when })}` : "";
  return `${t(`analysis.quota_${quota}`)}${resets}`;
}

const JUDGE_QUOTA = ["available", "exhausted", "unknown"];

/** The mode to show selected: the RAG's, or the default when it sent none. */
export function selectedMode(status) {
  return JUDGE_MODES.includes(status?.mode) ? status.mode : "claude_code";
}
