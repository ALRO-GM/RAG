// Pure helpers of a session's conversation type and goal (#65).
// The type is declared by whoever records, never inferred
// (docs/skills/CONVERSATION.md §4.3); both fields are optional context.

/** The types of transcript.v1 `session.type`, in the order the pages list them. */
export const SESSION_TYPES = [
  "reunion_decision", "lluvia_ideas", "resolucion_problemas", "negociacion",
  "entrevista", "feedback", "conflicto", "coaching", "venta",
  "conversacion_libre", "otro",
];

/** Longest goal the API accepts, in characters. */
export const GOAL_MAX_LENGTH = 500;

/**
 * The body of PATCH /api/sessions/{id} (and the type and goal of a new
 * session) from the form's values: no choice or an unknown value is
 * null, the goal is trimmed and a blank one is null.
 */
export function describeBody(type, goal) {
  return {
    type: SESSION_TYPES.includes(type) ? type : null,
    goal: String(goal ?? "").trim() || null,
  };
}

/** Whether `body` would change what `session` holds. */
export function describeChanged(session, body) {
  return (session?.type ?? null) !== body.type || (session?.goal ?? null) !== body.goal;
}

/** The type in the page's language (`t` translates); "" when unset. */
export function typeLabel(t, type) {
  if (!type) return "";
  const key = `type.${type}`;
  const text = t(key);
  return text === key ? type : text;
}

/** "Negociación · Precio del contrato", the type or the goal alone, or "". */
export function describeLine(t, session) {
  return [typeLabel(t, session?.type), session?.goal ?? ""].filter(Boolean).join(" · ");
}

/**
 * Run `task` calls one after the other: a call made while another is
 * in flight waits for it, so two quick edits cannot send two requests
 * whose answers arrive in the wrong order. `task` reads the current
 * values when it runs, so the last edit is the one that is saved.
 */
export function serialised(task) {
  let tail = Promise.resolve();
  return () => {
    const run = tail.then(task);
    tail = run.catch(() => {});
    return run;
  };
}
