// Pure logic of the call-centre example (#250): which scripted demo a
// route opens, where the end of the live demo leads, and the numbers of
// the call's lines that the report and the live page both quote. No DOM
// here, so node --test can run it. Everything in these demos is
// fictitious and the pages say so.

export const SUPPORT_PRODUCT = "live_assist.support";
export const INTERVIEW_PRODUCT = "live_assist.interview";

/** The scripted demo `#/live/demo` (interview) or `#/live/demo/support` opens, or null. */
export function pickDemo(demos, which = null) {
  const list = (demos ?? []).filter((d) => d?.demo === true);
  if (which === "support") return list.find((d) => d.product === SUPPORT_PRODUCT) ?? null;
  return list.find((d) => d.product === INTERVIEW_PRODUCT) ?? list.find((d) => d.product !== SUPPORT_PRODUCT) ?? null;
}

/** The demo with a voice-assistant example, or null. */
export function voiceDemo(demos) {
  return (demos ?? []).find((d) => d?.demo === true && d.voicebot === true) ?? null;
}

/** Where the end of a live demo leads: the call report, the candidate's result, or nothing. */
export function resultLink(demo) {
  if (demo?.report === true) return { kind: "report", href: `#/call-report/${encodeURIComponent(demo.id)}` };
  if (demo?.result_session) return { kind: "analysis", href: `#/analysis/${encodeURIComponent(demo.result_session)}` };
  return null;
}

/** Number each sentence of the call from 1, in order: { id -> n }. */
export function lineNumbers(utterances) {
  return new Map((utterances ?? []).map((u, i) => [u.id, i + 1]));
}

/** The chunks a demo's suggestion cites, with their text: [{ id, text }]. */
export function sourcesOf(ids, sources) {
  const byId = new Map((sources ?? []).map((s) => [s.id, s]));
  return (ids ?? []).filter((id) => byId.has(id)).map((id) => ({ id, text: byId.get(id).text }));
}
