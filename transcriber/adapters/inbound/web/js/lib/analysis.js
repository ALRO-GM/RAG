// The analysis judge control: pure helpers (no DOM), tested in node.

export const JUDGE_MODES = ["claude_code", "local", "both"];

/** The quota line under the control: "available", "exhausted" or "unknown". */
export function quotaLine(t, status) {
  const quota = JUDGE_QUOTA.includes(status?.claude_quota) ? status.claude_quota : "unknown";
  const resets = quota === "exhausted" && status.quota_resets
    ? ` ${t("analysis.resets", { when: status.quota_resets })}` : "";
  return `${t(`analysis.quota_${quota}`)}${resets}`;
}

const JUDGE_QUOTA = ["available", "exhausted", "unknown"];

/** The mode to show selected: the RAG's, or the default when it sent none. */
export function selectedMode(status) {
  return JUDGE_MODES.includes(status?.mode) ? status.mode : "claude_code";
}
