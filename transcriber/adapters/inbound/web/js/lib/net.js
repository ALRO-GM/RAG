// Pure interpretation of API answers and the polling rhythm.

export const UNAVAILABLE_STATUS = [502, 503, 504];

/** Seconds of a Retry-After header (delay form), or 0. */
export function retryAfterSeconds(retryAfter) {
  const n = Number.parseInt(retryAfter ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

/**
 * Sort a failed call into what the page should do:
 * "unavailable" (workstation off, network down), "unauthorized",
 * "rate" (wait `waitSeconds`), "client" (4xx: the request is wrong;
 * `waitSeconds` from Retry-After when the API sent one, e.g. 409 busy),
 * "server" (other 5xx). A null status means the network failed.
 */
export function classifyFailure(status, retryAfter = null) {
  if (status === null || status === undefined || UNAVAILABLE_STATUS.includes(status)) {
    return { kind: "unavailable", waitSeconds: 0 };
  }
  if (status === 401) return { kind: "unauthorized", waitSeconds: 0 };
  if (status === 429) return { kind: "rate", waitSeconds: retryAfterSeconds(retryAfter) || 30 };
  if (status >= 500) return { kind: "server", waitSeconds: 0 };
  return { kind: "client", waitSeconds: retryAfterSeconds(retryAfter) };
}

/** Whether uploading the same chunk again can help. */
export function isRetryable(kind) {
  return kind === "unavailable" || kind === "server" || kind === "rate";
}

/** Milliseconds until the next poll of the sessions list. */
export function nextPollMs(sessions, quietPolls) {
  if (sessions.some((s) => s.state === "running")) return 2000;
  if (sessions.some((s) => s.state === "queued")) return 4000;
  const steps = [10000, 20000, 40000, 60000];
  return steps[Math.min(Math.max(quietPolls, 0), steps.length - 1)];
}

/** Read the access code from a location fragment ("#k=abc"), or null. */
export function codeFromFragment(hash) {
  const m = /^#?(?:.*&)?k=([^&]+)/.exec(hash ?? "");
  if (!m) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return null;
  }
}

/** Fragment left after removing the code ("#k=abc" -> ""). */
export function fragmentWithoutCode(hash) {
  const rest = String(hash ?? "").replace(/^#/, "").split("&").filter((p) => p && !p.startsWith("k="));
  return rest.length ? `#${rest.join("&")}` : "";
}
