// Pure text helpers: ids from names, time formatting.

const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const RESERVED = /^s[0-9]+$/i; // S1, S2...: anonymous voices (the API refuses S<n>)

/** Lower-case ASCII id from a person's name ("Ángela Núñez" -> "angela-nunez"). */
export function suggestId(name) {
  let id = String(name ?? "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64)
    .replace(/-+$/g, "");
  if (RESERVED.test(id)) id = `${id}-p`;
  return id;
}

/** Make `id` unique among `taken` by adding -2, -3... */
export function uniqueId(id, taken) {
  if (!id || !taken.includes(id)) return id;
  for (let n = 2; ; n++) {
    const candidate = `${id.slice(0, 60)}-${n}`;
    if (!taken.includes(candidate)) return candidate;
  }
}

/** Null when `id` is acceptable, else "empty" | "format" | "reserved". */
export function idProblem(id) {
  if (!id) return "empty";
  if (!ID_PATTERN.test(id)) return "format";
  if (RESERVED.test(id)) return "reserved";
  return null;
}

/** "m:ss" or "h:mm:ss" from seconds. */
export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, "0");
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
  return `${m}:${ss}`;
}

/** A session id from a date and a title: "2026-11-14-183005-equipo-a". */
export function sessionIdFor(date, title) {
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
  const slug = suggestId(title).slice(0, 30).replace(/-+$/g, "");
  return slug ? `${stamp}-${slug}` : stamp;
}

/** Seconds left of a countdown (never negative), and whether it is over. */
export function countdown(elapsed, goal) {
  const left = Math.max(0, Math.ceil(goal - elapsed));
  return { left, done: elapsed >= goal };
}
