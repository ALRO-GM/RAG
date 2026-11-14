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

/**
 * The code arrives as #k=<code>: keep it with `keep`, then take it out of
 * the address bar (`history.replaceState`, so no new history entry and no
 * reload). `loc` and `hist` are `location` and `history` (fakes in tests).
 * Returns whether there was a code.
 */
export function adoptCodeFrom(loc, hist, keep) {
  const fromLink = codeFromFragment(loc.hash);
  if (!fromLink) return false;
  keep(fromLink);
  hist.replaceState(null, "", loc.pathname + loc.search + fragmentWithoutCode(loc.hash));
  return true;
}

/** Shortest access code the API accepts. */
export const MIN_CODE_LENGTH = 16;

/**
 * The code in what a person pasted: a whole join link
 * ("https://host/#k=abc") or the bare code. Null when it is neither.
 */
export function codeFromInput(text) {
  const raw = String(text ?? "").trim();
  const hash = raw.includes("#") ? raw.slice(raw.indexOf("#")) : null;
  const code = hash !== null ? codeFromFragment(hash) : raw;
  if (!code || code.length < MIN_CODE_LENGTH || !/^[A-Za-z0-9_-]+$/.test(code)) return null;
  return code;
}
