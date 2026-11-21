// What the login screen says for each way a login can end. Pure.

/**
 * The i18n key (and its values) of the message for a failed login.
 * `error` is an ApiFailure: 401 is a wrong password, 429 a lock (with the
 * seconds to wait), anything else a computer that cannot be reached.
 */
export function loginMessage(error) {
  if (error?.kind === "unauthorized") return { key: "login.wrong", values: {} };
  if (error?.kind === "rate") {
    return { key: "login.locked", values: { time: waitText(error.waitSeconds) } };
  }
  return { key: "login.unavailable", values: {} };
}

/** "45 s" under a minute, otherwise whole minutes ("15 min"). */
export function waitText(seconds) {
  const s = Math.max(Math.ceil(Number(seconds) || 0), 1);
  return s < 60 ? `${s} s` : `${Math.ceil(s / 60)} min`;
}
