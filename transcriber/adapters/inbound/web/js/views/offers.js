// «Ofertas» (#271, docs/live/VOICEBOT_INTERVIEWER.md §6.4): the job offers
// (interview plans) and, per offer, the people interviewed from it.
//
// - #/offers: the open offers (closed ones on request), «Nueva oferta».
// - #/offers/<plan>?sort=…: the offer's header (state, close / reopen,
//   «Editar las preguntas» with the plan editor of views/interview.js) and
//   one row per person, in the server's draft order (best figure first);
//   sortable by any column, searchable by name, filterable by a required
//   point's state and by the kind of interview; the choice stays in the
//   address. A «Cómo se ordena» note and the draft notice sit where the
//   order is shown.
// - #/offers/<plan>/person/<id>: the person's file (views/candidate.js).
//
// For the interviewer only: never shown to candidates. The order is a draft
// made by a model from short interviews, for a person to review; nothing
// here is a decision, and nobody is hidden unless a filter the person set
// says so. Pure parts in lib/offers.js.

import { h, fill, keepFocus } from "../dom.js";
import { t, decimal } from "../i18n.js";
import { describeError } from "../errors.js";
import { emptyState, icon, openDialog, pageHead, toast } from "../ui.js";
import {
  KINDS, POINT_FILTERS, accuracyParts, ariaSort, boardRows, closedCount, filtering, nextSort, offerHref, offerList,
  optionalPoints, parseQuery, personHref, pointColumns, visibleRows,
} from "../lib/offers.js";
import { clonePlan, planErrors, planForSave, reportHref } from "../lib/interview.js";
import { judgeTag, planEditorBody, shortDate } from "./interview.js";
import { candidateBlock } from "./candidate.js";

/** `key` with `{n}`, or its `_one` variant when `n` is 1 (the dictionaries' plural rule). */
function plural(key, n, params = {}) {
  return t(n === 1 ? `${key}_one` : key, { n, ...params });
}

/** «N puntos: R imprescindibles, D deseables», each count in its own number. */
function pointsLine(total, required, desirable) {
  return t("of.row_points", {
    total: plural("of.n_points", total), required: plural("of.n_required", required), desirable: plural("of.n_desirable", desirable),
  });
}

/** Wide screens show the filters open; on a phone they start folded. */
function wideScreen() {
  return globalThis.matchMedia?.("(min-width: 48rem)")?.matches ?? true;
}

/** Keep the table's state in the address without drawing the page again. */
function keepInAddress(hash) {
  try {
    globalThis.history?.replaceState?.(null, "", `${globalThis.location?.pathname ?? ""}${globalThis.location?.search ?? ""}${hash}`);
  } catch {
    // the address keeps its old state; the page still works
  }
}

// ---- #/offers -------------------------------------------------------------

/** `#/offers` (`?closed=1` also shows the closed positions). */
export async function offersView(ctx, root, [query] = []) {
  root.classList.add("wide");
  const { plans } = await ctx.api.interviewPlans();
  const closed = new URLSearchParams(String(query ?? "")).get("closed") === "1";
  root.append(offersPage(plans, { closed, onClosed: (on) => keepInAddress(on ? "#/offers?closed=1" : "#/offers") }));
  return null;
}

/** The list of offers (exported for the tests). */
export function offersPage(plans, { closed = false, onClosed = () => {} } = {}) {
  const list = h("div", { class: "of-list" });
  const shut = closedCount(plans);
  const toggle = h("input", { type: "checkbox", id: "of-closed", checked: closed });
  const draw = () => fill(list, offerCards(plans, toggle.checked));
  toggle.addEventListener("change", () => {
    onClosed(toggle.checked);
    draw();
  });
  draw();
  return h("div", { class: "stack" },
    pageHead(t("of.title"), {
      lead: t("of.lead"),
      actions: [h("a", { class: "button primary", href: "#/voicebot/interview" }, icon("plus"), t("of.new"))],
    }),
    shut ? h("label", { class: "check", for: "of-closed" }, toggle, " ", t("of.show_closed", { n: shut })) : null,
    list);
}

function offerCards(plans, closed) {
  if (!plans?.length) {
    return emptyState({
      iconName: "file", title: t("of.empty_title"), body: t("of.empty_body"),
      action: h("a", { class: "button primary", href: "#/voicebot/interview" }, t("of.new")),
    });
  }
  const shown = offerList(plans, { closed });
  if (!shown.length) return h("p", { class: "hint" }, t("of.none_open"));
  return h("ul", { class: "of-cards" }, shown.map((p) => h("li", { class: `card of-card is-${p.state === "closed" ? "closed" : "open"}` },
    h("div", { class: "row wrap of-card-head" },
      h("h2", {}, h("a", { href: offerHref(p.id) }, p.title)),
      stateTag(p.state)),
    h("p", { class: "meta" }, pointsLine(p.slots ?? 0, p.minimums ?? 0, p.desirables ?? Math.max(0, (p.slots ?? 0) - (p.minimums ?? 0)))),
    h("p", { class: "meta" }, p.interviews
      ? [t("of.row_interviews", {
        people: plural("of.n_people", p.people ?? p.interviews), interviews: plural("of.n_interviews", p.interviews),
        ai: p.interviews_ai ?? p.interviews, live: p.interviews_live ?? 0,
      }),
        p.last_interview ? [" · ", t("of.row_last", { date: shortDate(p.last_interview) })] : null]
      : t("of.row_none")),
    p.created_at ? h("p", { class: "meta" }, t("of.row_created", { date: shortDate(p.created_at) })) : null)));
}

function stateTag(state) {
  const closed = state === "closed";
  return h("span", { class: `badge of-state is-${closed ? "closed" : "open"}` }, t(closed ? "of.state.closed" : "of.state.open"));
}

// ---- #/offers/<plan> --------------------------------------------------------

/** `#/offers/<plan>?…`: one offer and the people interviewed from it. */
export async function offerView(ctx, root, [planId, query] = []) {
  root.classList.add("wide", "full");
  const id = decodeURIComponent(planId ?? "");
  let board;
  try {
    board = await ctx.api.planBoard(id);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("of.title")), emptyState({
      iconName: "file", title: t("of.missing_title"),
      body: error?.code === "not_found" ? t("of.not_found") : describeError(error),
      action: h("a", { class: "button", href: "#/offers" }, t("of.back")),
    }));
    return null;
  }
  ctx.setCrumbs?.([{ label: t("nav.offers"), href: "#/offers" }, { label: board.plan.title, href: null }]);
  const page = h("div", { class: "stack" });
  root.append(page);
  const reload = async () => {
    try {
      fill(page, offerPage(ctx, await ctx.api.planBoard(id), parseQuery(currentQuery()), reload));
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  };
  fill(page, offerPage(ctx, board, parseQuery(query), reload));
  return null;
}

function currentQuery() {
  const hash = String(globalThis.location?.hash ?? "");
  const at = hash.indexOf("?");
  return at < 0 ? "" : hash.slice(at + 1);
}

/** The whole offer page for a board (exported for the tests). `reload` reads the board again. */
export function offerPage(ctx, board, state, reload = () => {}) {
  const plan = board.plan;
  const closed = plan.state === "closed";
  const editor = h("div", { class: "of-editor" });
  const edit = h("button", { type: "button", "aria-expanded": "false", "aria-controls": "of-editor-card" }, icon("edit"), t("of.edit"));
  edit.addEventListener("click", () => {
    if (editor.firstChild) {
      fill(editor);
      edit.setAttribute("aria-expanded", "false");
      return;
    }
    edit.setAttribute("aria-expanded", "true");
    fill(editor, planEditorCard(ctx, plan, {
      onSaved: () => reload(),
      onCancel: () => { fill(editor); edit.setAttribute("aria-expanded", "false"); },
    }));
  });
  const position = h("button", { type: "button" }, t(closed ? "iv.reopen_position" : "iv.close_position"));
  position.addEventListener("click", async () => {
    if (!closed) {
      const yes = await openDialog({
        title: t("iv.close_title", { title: plan.title }), content: [h("p", {}, t("iv.close_body"))], confirm: t("iv.close_position"),
      });
      if (!yes) return;
    }
    try {
      await (closed ? ctx.api.reopenPlan(plan.id) : ctx.api.closePlan(plan.id));
      toast(t(closed ? "iv.reopened_ok" : "iv.closed_ok"));
      reload();
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  });
  const required = (board.required ?? []).length;
  const actions = [edit, position];
  if (!closed) actions.push(h("a", { class: "button", href: `#/voicebot/interview/${encodeURIComponent(plan.id)}` }, icon("voicebot"), t("of.start_ai")));
  return [
    pageHead(plan.title, { lead: plan.summary || null, actions }),
    h("p", { class: "row wrap of-head-meta" }, stateTag(plan.state), " ",
      h("span", { class: "meta" }, [
        t("of.minutes", { minutes: plan.minutes }),
        pointsLine(plan.slots.length, required, plan.slots.length - required),
        t("of.revision", { n: plan.revision ?? 0 }),
      ].join(" · "))),
    closed ? h("p", { class: "notice", role: "status" }, t("iv.closed_notice")) : null,
    editor,
    peopleSection(ctx, board, state, {
      onState: (next) => keepInAddress(offerHref(plan.id, next)),
      onDelete: async (row) => {
        const yes = await openDialog({
          title: t("iv.delete_report_title"), content: [h("p", {}, t("iv.delete_report_body"))], confirm: t("iv.delete_report"), danger: true,
        });
        if (!yes) return;
        try {
          await ctx.api.deletePlanReport(plan.id, row.session);
          toast(t("iv.deleted_report"));
          reload();
        } catch (error) {
          toast(describeError(error), { error: true });
        }
      },
    }),
  ];
}

/**
 * The plan editor on the offer page: the same editor as the voice-bot page's
 * review; saving makes a new revision (409 when someone saved first).
 */
export function planEditorCard(ctx, saved, { onSaved = () => {}, onCancel = () => {} } = {}) {
  const box = h("section", { class: "card of-edit", id: "of-editor-card", "aria-labelledby": "of-edit-title" });
  const plan = clonePlan(saved);
  function draw() {
    const errors = planErrors(plan);
    const dirty = JSON.stringify(planForSave(plan)) !== JSON.stringify(planForSave(saved));
    const save = h("button", { type: "button", class: "primary", disabled: !dirty || errors.length > 0 }, t("of.save"));
    save.addEventListener("click", async () => {
      save.disabled = true;
      try {
        const stored = await ctx.api.savePlan(saved.id, planForSave(plan));
        toast(t("of.saved", { n: stored?.revision ?? "" }));
        onSaved(stored);
      } catch (error) {
        toast(error?.code === "conflict" ? t("iv.err.conflict") : describeError(error), { error: true });
        save.disabled = false;
      }
    });
    keepFocus(box, () => fill(box,
      h("h2", { id: "of-edit-title" }, t("of.edit_title")),
      h("p", { class: "hint" }, t("of.edit_hint")),
      planEditorBody(plan, draw),
      h("div", { class: "row wrap" }, save, h("button", { type: "button", class: "ghost", onClick: onCancel }, t("action.cancel")),
        dirty ? h("span", { class: "hint" }, t("of.unsaved")) : null)));
  }
  draw();
  return box;
}

// ---- the people of an offer ---------------------------------------------------

/** The draft notice and «Cómo se ordena», next to the order. */
export function howOrdered() {
  return [
    h("p", { class: "draft-notice", role: "note" }, h("strong", {}, t("of.draft_title")), " ", t("of.draft_note")),
    h("details", { class: "of-how" },
      h("summary", {}, t("of.how_title")),
      h("ul", {}, ["figure", "ai", "live", "required", "ties", "edits", "rules", "scales"].map((k) => h("li", {}, t(`of.how.${k}`))))),
  ];
}

/**
 * Search, filters, column chooser and the table; redraws itself on each
 * change and reports the new state with `onState` (kept in the address).
 */
export function peopleSection(ctx, board, initial, { onState = () => {}, onDelete = null } = {}) {
  let state = { ...initial };
  const rows = boardRows(board);
  const plan = board.plan;
  const required = pointColumns(board, []);
  const tableBox = h("div", { class: "of-table-box" });
  const status = h("p", { class: "meta of-status", role: "status", "aria-live": "polite" });

  const search = h("input", { id: "of-search", type: "search", value: state.q, maxLength: 80, autocomplete: "off" });
  // Typing redraws the table at once; the status line is said after a pause (or on Enter / leaving the field).
  let pause = null;
  search.addEventListener("input", () => {
    change({ q: search.value }, { announce: false });
    clearTimeout(pause);
    pause = setTimeout(announce, 400);
  });
  search.addEventListener("change", () => {
    clearTimeout(pause);
    announce();
  });
  const kind = h("select", { id: "of-kind" }, KINDS.map((k) => h("option", { value: k, selected: k === state.kind }, t(`of.kind.${k}`))));
  kind.addEventListener("change", () => change({ kind: kind.value }));
  const point = h("select", { id: "of-point" },
    h("option", { value: "", selected: !state.point }, t("of.point_any")),
    required.map((s) => h("option", { value: s.id, selected: s.id === state.point }, s.label)));
  const pointState = h("select", { id: "of-point-state", disabled: !state.point },
    Object.keys(POINT_FILTERS).map((k) => h("option", { value: k, selected: k === state.state }, t(`of.filter.${k}`))));
  point.addEventListener("change", () => {
    pointState.disabled = !point.value;
    change({ point: point.value, state: point.value ? (pointState.value === "any" ? "covered" : pointState.value) : "any" });
    pointState.value = state.state;
  });
  pointState.addEventListener("change", () => change({ state: pointState.value }));
  const optional = optionalPoints(board);
  const chooser = optional.length ? h("details", { class: "of-columns" },
    h("summary", {}, t("of.columns")),
    h("fieldset", {}, h("legend", { class: "meta" }, t("of.columns_hint")),
      optional.map((s) => {
        const box = h("input", { type: "checkbox", id: `of-col-${s.id}`, checked: state.cols.includes(s.id) });
        box.addEventListener("change", () => change({
          cols: box.checked ? [...state.cols, s.id] : state.cols.filter((c) => c !== s.id),
        }));
        return h("label", { class: "check", for: `of-col-${s.id}` }, box, " ", s.label);
      }))) : null;
  const clear = h("button", { type: "button", class: "ghost small", hidden: !filtering(state) }, t("of.clear"));
  clear.addEventListener("click", () => {
    search.value = "";
    kind.value = "all";
    point.value = "";
    pointState.value = "any";
    pointState.disabled = true;
    change({ q: "", kind: "all", point: "", state: "any" });
  });

  const filterBox = h("details", { class: "of-filter-box", open: wideScreen() });
  const filterSummary = h("summary", {});
  const active = () => [(state.q ?? "").trim(), state.kind !== "all", state.point && state.state !== "any"].filter(Boolean).length;
  const syncSummary = () => { filterSummary.textContent = active() ? plural("of.filters_active", active()) : t("of.filters_title"); };

  function change(fields, { announce: say = true } = {}) {
    state = { ...state, ...fields };
    clear.hidden = !filtering(state);
    syncSummary();
    onState(state);
    draw();
    if (say) announce();
  }

  function announce() {
    const shown = visibleRows(rows, state);
    const sortName = columnName(state.sort, pointColumns(board, state.cols));
    const text = rows.length ? [
      t("of.count", { shown: shown.length, total: rows.length }),
      t(state.dir === "asc" ? "of.sorted_asc" : "of.sorted_desc", { column: sortName }),
    ].join(" ") : "";
    // The same words again (a pause, then leaving the field) are not said twice.
    if (status.textContent !== text) status.textContent = text;
  }

  function draw() {
    const shown = visibleRows(rows, state);
    const columns = pointColumns(board, state.cols);
    keepFocus(tableBox, () => fill(tableBox, rows.length
      ? peopleTable(plan, shown, columns, state, {
        onSort: (key) => change(nextSort(state, key)),
        onDelete,
      })
      : emptyState({ iconName: "inbox", title: t("of.no_people_title"), body: t("of.no_people") })));
  }

  draw();
  announce();
  syncSummary();
  fill(filterBox, filterSummary,
    h("div", { class: "row wrap of-filters", role: "search", "aria-label": t("of.filters") },
      h("div", { class: "field" }, h("label", { for: "of-search" }, t("of.search")), search),
      h("div", { class: "field" }, h("label", { for: "of-kind" }, t("of.kind")), kind),
      h("div", { class: "field" }, h("label", { for: "of-point" }, t("of.point")), point),
      h("div", { class: "field" }, h("label", { for: "of-point-state" }, t("of.point_state")), pointState),
      clear),
    chooser);
  return h("section", { class: "stack of-people", "aria-labelledby": "of-people-title" },
    h("h2", { id: "of-people-title" }, t("of.people_title")),
    howOrdered(),
    rows.length ? filterBox : null,
    status,
    tableBox,
    board.truncated ? h("p", { class: "notice", role: "status" }, t("of.truncated", { n: board.interviews })) : null,
    rows.some((r) => r.owned) ? h("p", { class: "hint" }, t("iv.retention")) : null,
    rows.some((r) => !r.owned) ? h("p", { class: "hint" }, t("iv.retention_no_file")) : null,
    h("p", { class: "hint" }, t("of.legal")));
}

function columnName(key, columns) {
  if (["fit", "name", "date", "to_check"].includes(key)) return t(`of.col.${key}`);
  return columns.find((c) => c.id === key)?.label ?? key;
}

function sortHeader(key, label, state, onSort, extra = null) {
  const button = h("button", { type: "button", class: "sort", "data-focus-key": `of-sort-${key}` }, label);
  button.addEventListener("click", () => onSort(key));
  return h("th", { scope: "col", "aria-sort": ariaSort(state, key), "data-key": key }, button, extra);
}

/** The table of people (exported for the tests). */
export function peopleTable(plan, rows, columns, state, { onSort = () => {}, onDelete = null } = {}) {
  const head = h("tr", {},
    sortHeader("name", t("of.col.name"), state, onSort),
    sortHeader("date", t("of.col.date"), state, onSort),
    sortHeader("fit", t("of.col.fit"), state, onSort),
    columns.map((c) => sortHeader(c.id, c.label, state, onSort,
      [" ", h("span", { class: "tag" }, t(c.required ? "of.required" : "of.desirable"))])),
    h("th", { scope: "col" }, t("of.col.accuracy")),
    sortHeader("to_check", t("of.col.to_check"), state, onSort),
    h("th", { scope: "col" }, t("of.col.open")));
  const body = rows.length
    ? rows.map((r) => personRow(plan, r, columns, onDelete))
    : h("tr", {}, h("td", { colspan: String(6 + columns.length) }, t("of.no_match")));
  return h("div", { class: "table-wrap of-table-wrap" },
    h("table", { class: "data of-table" },
      h("caption", {}, t("of.caption", { title: plan.title })),
      h("thead", {}, head),
      h("tbody", {}, body)));
}

function figureCell(r) {
  return h("td", { class: "of-figure" },
    h("span", { class: "of-figure-value" }, r.figure === null ? "–" : decimal(r.figure, 1)),
    h("span", { class: "of-small" }, t(`of.basis.${r.basis ?? "labels"}`)),
    r.missing.length ? h("span", { class: "tag of-flag is-missing" }, icon("alert"), t("of.missing", { n: r.missing.length })) : null,
    r.unasked.length ? h("span", { class: "tag of-flag is-missing" }, icon("alert"), t("of.unasked", { n: r.unasked.length })) : null,
    r.missing.length || r.unasked.length ? null : h("span", { class: "of-small" }, t("of.covers_all")),
    r.planChanged ? h("span", { class: "tag of-flag" }, t("of.plan_changed")) : null,
    r.lessReliable ? h("span", { class: "tag of-flag" }, t("of.less_reliable")) : null);
}

function personRow(plan, r, columns, onDelete) {
  const others = r.others.length ? h("details", { class: "of-others" },
    h("summary", {}, plural("of.others", r.others.length)),
    h("ul", {}, r.others.map((o) => h("li", {},
      h("a", { href: reportHref(plan.id, o.session_id) }, t(`iv.kind.${o.kind === "live" ? "live" : "ai"}`)),
      " · ", shortDate(o.finished_at),
      " · ", t("of.other_figure", { figure: typeof o.figure === "number" ? decimal(o.figure, 1) : "–" }))))) : null;
  const links = [h("a", { href: reportHref(plan.id, r.session) }, t("iv.open_report"))];
  if (r.owned && r.candidateId) links.push(h("a", { href: personHref(plan.id, r.candidateId) }, t("of.open_file")));
  if (!r.owned) {
    links.push(h("span", { class: "tag" }, t("iv.no_file")));
    if (onDelete) links.push(h("button", { type: "button", class: "small danger", onClick: () => onDelete(r) }, t("iv.delete_report")));
  }
  // The person cell also opens the report and the file, so they stay in reach when the table scrolls.
  const quick = h("span", { class: "of-quick" },
    h("a", { href: reportHref(plan.id, r.session) }, t("of.report_short")),
    r.owned && r.candidateId ? [" · ", h("a", { href: personHref(plan.id, r.candidateId) }, t("of.file_short"))] : null);
  return h("tr", { class: r.missing.length || r.unasked.length ? "of-row is-missing" : "of-row" },
    h("th", { scope: "row" }, h("span", { class: "of-name" }, r.name || t("of.no_name")), quick, others),
    h("td", {}, h("span", { class: "tag" }, t(`iv.kind.${r.kind}`)), h("span", { class: "of-small" }, shortDate(r.finished))),
    figureCell(r),
    columns.map((c) => pointCell(r.points.get(c.id), c, plan, r)),
    h("td", { class: "of-accuracy" }, accuracyParts(r.accuracy).map((p) => h("span", { class: "of-small" },
      t("iv.judge_count", { n: p.n, label: t(`iv.judge.${p.key}`) })))),
    h("td", {}, String(r.toCheck)),
    h("td", { class: "of-links" }, links));
}

/** One point of one person: its state in words, the score, the concepts and the evidence. */
function pointCell(point, column, plan, row) {
  const state = point?.state ?? "not_asked";
  const word = ["solid", "correct_basic", "wrong", "off_topic", "unclear", "to_verify", "not_treated", "does_not_know"].includes(state)
    ? judgeTag(state) : h("span", { class: `tag of-point-state is-${state}` }, t(`of.point_state.${state}`));
  const evidence = point && (point.quote || point.reason) ? h("details", { class: "of-evidence" },
    h("summary", {}, t("of.evidence")),
    point.quote ? h("p", {}, h("q", {}, point.quote)) : null,
    point.reason ? h("p", { class: "meta" }, point.reason) : null,
    h("a", { href: reportHref(plan.id, row.session) }, t("iv.open_report"))) : null;
  return h("td", { class: `of-point${column.required && !["solid", "correct_basic", "to_verify", "unclear", "answered"].includes(state) ? " is-gap" : ""}` },
    word,
    typeof point?.score === "number" ? h("span", { class: "of-small" }, t("of.score", { score: point.score })) : null,
    point?.concepts ? h("span", { class: "of-small" }, plural("of.concepts", point.concepts, { treated: point.treated ?? 0, total: point.concepts })) : null,
    askedNote(point, state),
    evidence);
}

/**
 * #286: whether the interviewer asked (a live report with coverage): a point
 * not treated says «se preguntó, sin respuesta» or «no se llegó a preguntar»
 * (the interview's gap, not the person's); otherwise how many concepts were
 * asked. Nothing for reports without it. The figure does not change.
 */
function askedNote(point, state) {
  if (typeof point?.asked !== "number") return null;
  // «Se preguntó» only with a mark of a question; a topic merely mentioned was not asked.
  if (state === "not_treated") return h("span", { class: "of-small" }, t(point.asked > 0 ? "of.asked_no_answer" : "of.never_asked"));
  return point.concepts ? h("span", { class: "of-small" }, t("of.asked", { n: point.asked })) : null;
}

// ---- #/offers/<plan>/person/<id> ------------------------------------------------

/** `#/offers/<plan>/person/<id>`: the person's file, from the offer page. */
export async function personFileView(ctx, root, [planId, candidateId] = []) {
  root.classList.add("wide");
  const plan = decodeURIComponent(planId ?? "");
  const id = decodeURIComponent(candidateId ?? "");
  let profile;
  try {
    profile = await ctx.api.planCandidate(plan, id);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("of.person_title")), emptyState({
      iconName: "file", title: t("of.person_title"),
      body: error?.code === "not_found" ? t("of.person_missing") : describeError(error),
      action: h("a", { class: "button", href: offerHref(plan) }, t("of.back_offer")),
    }));
    return null;
  }
  let title = plan;
  try {
    title = (await ctx.api.interviewPlan(plan))?.title ?? plan;
  } catch {
    // the breadcrumb keeps the plan's id
  }
  ctx.setCrumbs?.([{ label: t("nav.offers"), href: "#/offers" }, { label: title, href: offerHref(plan) },
    { label: profile.name ?? id, href: null }]);
  root.append(pageHead(profile.name ?? t("of.person_title"), { lead: t("of.person_lead") }),
    candidateBlock(profile, { api: ctx.api, onDeleted: () => ctx.navigate(offerHref(plan)) }),
    h("p", {}, h("a", { href: offerHref(plan) }, t("of.back_offer"))));
  return null;
}
