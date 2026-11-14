// Plain-language text for API failures.

import { t } from "./i18n.js";

const BY_CODE = {
  consent_required: "err.consent",
  voice_model_mismatch: "err.voice_model",
  wrong_state: "err.wrong_state",
  not_found: "err.not_found",
  payload_too_large: "err.too_large",
  model_unavailable: "err.model",
  invalid_input: "err.invalid",
  invalid_request: "err.invalid",
};

const REASONS = {
  too_short: "reason.too_short",
  mostly_silent: "reason.mostly_silent",
  several_voices: "reason.several_voices",
};

/** What to tell the person about a failed call. */
export function describeError(error) {
  if (error?.code === "sample_rejected") {
    return t(REASONS[error.reason] ?? "reason.other");
  }
  switch (error?.kind) {
    case "unavailable": return t("err.unavailable");
    case "unauthorized": return t("err.unauthorized");
    case "rate": return t("err.rate", { seconds: error.waitSeconds });
    default: break;
  }
  const key = BY_CODE[error?.code];
  const detail = error?.message && error?.kind === "client" ? ` (${error.message})` : "";
  return `${t(key ?? "err.generic")}${key === "err.invalid" ? detail : ""}`;
}
