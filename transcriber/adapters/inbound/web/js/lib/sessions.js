// Pure helpers of the sessions list.

/** Percent 0-100 of a running job, or null when there is no progress. */
export function progressPercent(session) {
  if (session.state !== "running" || typeof session.fraction !== "number") return null;
  return Math.round(Math.min(1, Math.max(0, session.fraction)) * 100);
}

/** Actions that make sense for a session in its state. */
export function actionsFor(session) {
  switch (session.state) {
    case "recording": return ["resume", "finish", "delete"];
    case "uploaded": return ["transcribe", "delete"];
    case "failed": return ["transcribe", "delete"];
    case "done": return ["open", "transcribe", "delete"];
    case "running": return []; // the job holds it: nothing to do but wait
    default: return ["delete"];
  }
}

/** Newest first, by start time when known, else by id. */
export function sortSessions(sessions) {
  return [...sessions].sort((x, y) => {
    const a = x.started_at ?? "";
    const b = y.started_at ?? "";
    if (a === b) return x.id < y.id ? 1 : -1;
    return a < b ? 1 : -1;
  });
}

/** Body of POST .../transcription from the new-session form. */
export function transcriptionBody(language, expected, numSpeakers) {
  const body = { language };
  if (expected.length) body.expected_speakers = expected;
  const n = Number.parseInt(numSpeakers, 10);
  if (Number.isFinite(n) && n >= 1) body.num_speakers = n;
  return body;
}

/** The job stage in the page's language (`t` translates), or as it came. */
export function stageLabel(t, stage) {
  if (!stage) return "";
  const key = `stage.${stage}`;
  const text = t(key);
  return text === key ? stage : text;
}
