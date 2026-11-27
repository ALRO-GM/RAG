// Pure logic of «Ofertas» (#271, docs/live/VOICEBOT_INTERVIEWER.md §6.4): the
// list of job offers (interview plans) and, per offer, the table of the people
// interviewed from it. The server sends the rows already in the default order
// (best figure first, the rules in orchestrator/domain/offer_board.py); here
// the person re-sorts by a column, searches by name and filters by a required
// point's state or the kind of interview, and the choice is kept in the
// address (#/offers/<plan>?sort=…). Nothing here computes a figure or hides a
// person by default. No DOM and no text (every text is in the dictionaries),
// so node --test runs it.

import { fold } from "./shell.js";

export const KINDS = ["all", "ai", "live"];
export const DIRS = ["desc", "asc"];
/** Groups of a point's state the filter offers (none: every state). */
export const POINT_FILTERS = {
  any: null,
  solid_basic: ["solid", "correct_basic"],
  covered: ["solid", "correct_basic", "to_verify", "unclear", "answered"],
  doubtful: ["to_verify", "unclear"],
  wrong_off: ["wrong", "off_topic"],
  not_covered: ["not_treated", "does_not_know", "skipped", "not_asked"],
};
/** The accuracy counts the table shows, in this order. */
export const ACCURACY_KEYS = ["solid", "correct_basic", "wrong", "off_topic", "not_treated"];
/** Columns that are not points; points sort by their own id. */
export const FIXED_SORTS = ["fit", "name", "date", "to_check"];
const DEFAULTS = { sort: "fit", dir: "desc", q: "", kind: "all", point: "", state: "any", cols: [] };

/** The table's state from the address's query (`sort=s1&dir=asc&q=ana…`); unknown values fall back to the defaults. */
export function parseQuery(query) {
  const params = new URLSearchParams(String(query ?? "").replace(/^\?/, ""));
  const out = { ...DEFAULTS, cols: [] };
  const sort = params.get("sort");
  if (sort && (FIXED_SORTS.includes(sort) || /^s\d{1,3}$/.test(sort))) out.sort = sort;
  if (DIRS.includes(params.get("dir"))) out.dir = params.get("dir");
  else if (out.sort === "name") out.dir = "asc";
  out.q = (params.get("q") ?? "").slice(0, 80);
  if (KINDS.includes(params.get("kind"))) out.kind = params.get("kind");
  const point = params.get("point") ?? "";
  if (/^s\d{1,3}$/.test(point)) out.point = point;
  const state = params.get("state");
  if (state && Object.hasOwn(POINT_FILTERS, state)) out.state = state;
  out.cols = (params.get("cols") ?? "").split(",").filter((c) => /^s\d{1,3}$/.test(c)).slice(0, 12);
  return out;
}

/** The query of a table state, defaults left out ("" when all are defaults). */
export function queryString(state) {
  const params = new URLSearchParams();
  const s = { ...DEFAULTS, ...state };
  if (s.sort !== DEFAULTS.sort) params.set("sort", s.sort);
  if (s.dir !== defaultDir(s.sort)) params.set("dir", s.dir);
  if (s.q) params.set("q", s.q);
  if (s.kind !== "all") params.set("kind", s.kind);
  if (s.point && s.state !== "any") {
    params.set("point", s.point);
    params.set("state", s.state);
  }
  if (s.cols?.length) params.set("cols", s.cols.join(","));
  return params.toString();
}

/** The direction a column starts in: names A→Z, everything else highest first. */
export function defaultDir(sort) {
  return sort === "name" ? "asc" : "desc";
}

/** The state after a click on a column's header: the same column flips, another starts in its own direction. */
export function nextSort(state, key) {
  if (state.sort === key) return { ...state, dir: state.dir === "desc" ? "asc" : "desc" };
  return { ...state, sort: key, dir: defaultDir(key) };
}

/** Where an offer opens (with an optional table state). */
export function offerHref(planId, state = null) {
  const query = state ? queryString(state) : "";
  return `#/offers/${encodeURIComponent(planId)}${query ? `?${query}` : ""}`;
}

/** Where a person's file opens from an offer. */
export function personHref(planId, candidateId) {
  return `#/offers/${encodeURIComponent(planId)}/person/${encodeURIComponent(candidateId)}`;
}

/**
 * The offers of the list: open ones (and closed ones when `closed`), the
 * open ones first, then the latest change first.
 */
export function offerList(plans, { closed = false } = {}) {
  const shown = (plans ?? []).filter((p) => closed || p.state !== "closed");
  const when = (p) => String(p.last_interview ?? p.updated_at ?? p.created_at ?? "");
  return [...shown].sort((a, b) => (a.state === "closed") - (b.state === "closed") || when(b).localeCompare(when(a)));
}

/** How many offers the closed ones are (for the switch's label). */
export function closedCount(plans) {
  return (plans ?? []).filter((p) => p.state === "closed").length;
}

/** The point columns of the table: every required point, then the desirable ones chosen. */
export function pointColumns(board, cols = []) {
  const slots = board?.plan?.slots ?? [];
  const required = new Set(board?.required ?? slots.filter((s) => s.level === "minimo").map((s) => s.id));
  const chosen = new Set(cols);
  return slots.filter((s) => required.has(s.id) || chosen.has(s.id)).map((s) => ({ ...s, required: required.has(s.id) }));
}

/** The desirable (not required) points, for the column chooser. */
export function optionalPoints(board) {
  const required = new Set(board?.required ?? []);
  return (board?.plan?.slots ?? []).filter((s) => !required.has(s.id));
}

/** One row per person, flattened from the board for drawing, sorting and filtering. */
export function boardRows(board) {
  return (board?.people ?? []).map((person, i) => {
    const latest = person.latest ?? {};
    return {
      person: person.person, name: person.name ?? latest.candidate ?? null,
      candidateId: person.candidate_id ?? null, owned: Boolean(latest.owned),
      position: person.position ?? i + 1,
      session: latest.session_id, kind: latest.kind === "live" ? "live" : "ai", finished: latest.finished_at ?? null,
      figure: typeof latest.figure === "number" ? latest.figure : null, basis: latest.figure_basis ?? null,
      planChanged: Boolean(latest.plan_changed), lessReliable: Boolean(latest.less_reliable),
      points: new Map((latest.points ?? []).map((p) => [p.id, p])),
      missing: latest.missing_required ?? [], unasked: latest.unasked_required ?? [], notAsked: latest.not_asked ?? [],
      accuracy: latest.accuracy ?? {}, toCheck: latest.to_check ?? 0,
      others: person.others ?? [],
      mandatoryList: latest.mandatory_list ?? null, // #294: the list of «Lo que hay que decir» of a live round
    };
  });
}

/** Whether a row passes the search, the kind and the point filter. */
export function rowMatches(row, state) {
  if (state.kind && state.kind !== "all" && row.kind !== state.kind) return false;
  const q = fold(state.q ?? "").trim();
  if (q && !fold(row.name ?? row.session ?? "").includes(q)) return false;
  const group = POINT_FILTERS[state.state];
  if (state.point && group) {
    const found = row.points.get(state.point);
    if (!group.includes(found?.state ?? "not_asked")) return false;
  }
  return true;
}

function compareValues(a, b, dir) {
  // A missing value goes last whatever the direction.
  if (a === null || a === undefined) return b === null || b === undefined ? 0 : 1;
  if (b === null || b === undefined) return -1;
  const order = typeof a === "string" ? a.localeCompare(b) : a - b;
  return dir === "asc" ? order : -order;
}

function sortValue(row, key) {
  if (key === "name") return row.name ? fold(row.name) : null;
  if (key === "date") return row.finished ? String(row.finished) : null;
  if (key === "to_check") return row.toCheck;
  const point = row.points.get(key);
  return typeof point?.value === "number" ? point.value : null;
}

/**
 * The rows shown: filtered, then sorted by the chosen column. "fit" is the
 * server's order (descending) or its reverse; any other column breaks its
 * ties by the server's order. Nobody is hidden unless a filter says so.
 */
export function visibleRows(rows, state) {
  const shown = rows.filter((row) => rowMatches(row, state));
  if (state.sort === "fit") {
    // Rows without a figure stay last in both directions.
    const none = (r) => (r.figure === null ? 1 : 0);
    return shown.sort((a, b) => none(a) - none(b) || (state.dir === "asc" ? b.position - a.position : a.position - b.position));
  }
  return shown.sort((a, b) => compareValues(sortValue(a, state.sort), sortValue(b, state.sort), state.dir) || a.position - b.position);
}

/** `aria-sort` of a column's header: the sorted column says its direction, the rest "none". */
export function ariaSort(state, key) {
  if (state.sort !== key) return "none";
  return state.dir === "asc" ? "ascending" : "descending";
}

/** Whether any filter is on (so the page can say how many are hidden and offer to clear them). */
export function filtering(state) {
  return Boolean((state.q ?? "").trim()) || (state.kind ?? "all") !== "all" || (Boolean(state.point) && state.state !== "any");
}

/** The accuracy counts of a row in the table's order, zeros left out. */
export function accuracyParts(accuracy) {
  return ACCURACY_KEYS.map((key) => ({ key, n: accuracy?.[key] ?? 0 })).filter((x) => x.n > 0);
}
