// «Lo que hay que decir» lists (#294, docs/live/MANDATORY_ITEMS.md): the pure
// part of the owner's editor, the live start form's choice and the names a
// session shows of the list it used. No DOM here, so node --test can run it.
//
// A list has a name and items; an item has the text shown on the live page,
// its moment (opening, during, closing) and one or more example sentences of
// how it is said (must_say). A list adds to the general list of its kind
// (default) or replaces it. The server gives item ids; the editor keeps the
// ids it was given so a saved item keeps its id.

/** The moments an item belongs to, in the order of a conversation. */
export const MOMENTS = ["opening", "during", "closing"];

/** The kinds of list: a call-centre campaign or offer, a job offer. */
export const KINDS = ["support", "interview"];

/** Bounds when the server does not send its own (`limits` of GET /mandatory/lists). */
export const DEFAULT_LIMITS = { items: 20, examples: 6, text: 300, name: 80 };

/** Where this browser keeps the last list chosen for a live call. */
export const LAST_LIST_KEY = "ragputin.live.mandatoryList";

/** The id of the general (read-only) list of a kind. */
export function generalId(kind) {
  return `general.${kind}`;
}

/** Whether `id` names a general list. */
export function isGeneral(id) {
  return KINDS.some((k) => generalId(k) === id);
}

/** A new, empty item (one empty example to fill in). */
export function newItem() {
  return { id: null, text: "", phase: "during", must_say: [""] };
}

/** An editable copy of a stored list (or a new one of `kind`). */
export function draftOf(list, kind = "support") {
  if (!list) return { id: null, version: 0, name: "", kind, mode: "add", items: [newItem()] };
  return {
    id: list.id ?? null, version: list.version ?? 0, name: list.name ?? "", kind: list.kind ?? kind,
    mode: list.mode === "replace" ? "replace" : "add",
    items: (list.items ?? []).map((i) => ({
      id: i.id ?? null, text: i.text ?? "", phase: MOMENTS.includes(i.phase) ? i.phase : null,
      must_say: (i.must_say ?? []).length ? [...i.must_say] : [""],
    })),
  };
}

const line = (value) => String(value ?? "").split(/\s+/).filter(Boolean).join(" ");

/** The body the server takes: texts on one line, empty examples left out. */
export function listForSave(draft) {
  const body = {
    name: line(draft.name), kind: draft.kind, mode: draft.mode === "replace" ? "replace" : "add",
    items: draft.items.map((i) => ({
      ...(i.id ? { id: i.id } : {}), text: line(i.text), phase: i.phase || null,
      must_say: i.must_say.map(line).filter(Boolean),
    })),
  };
  if (draft.id) body.version = draft.version;
  return body;
}

/**
 * What stops a save, as `{ key, n?, max? }` (dictionary keys under `mi.err.`).
 * `n` is the item's number (1-based) when the problem is in an item.
 */
export function listErrors(draft, limits = DEFAULT_LIMITS) {
  const found = [];
  const body = listForSave(draft);
  if (!body.name) found.push({ key: "name" });
  else if (body.name.length > limits.name) found.push({ key: "name_long", max: limits.name });
  if (!body.items.length) found.push({ key: "no_items" });
  if (body.items.length > limits.items) found.push({ key: "too_many", max: limits.items });
  body.items.forEach((item, i) => {
    const n = i + 1;
    if (!item.text) found.push({ key: "item_text", n });
    else if (item.text.length > limits.text) found.push({ key: "item_long", n, max: limits.text });
    if (!item.must_say.length) found.push({ key: "no_example", n });
    if (item.must_say.length > limits.examples) found.push({ key: "examples_many", n, max: limits.examples });
    if (item.must_say.some((e) => e.length > limits.text)) found.push({ key: "example_long", n, max: limits.text });
    const seen = new Set(item.must_say.map((e) => e.toLowerCase()));
    if (seen.size < item.must_say.length) found.push({ key: "example_twice", n });
  });
  return found;
}

/** Whether the draft differs from the list it was made from. */
export function isDirty(draft, saved) {
  return JSON.stringify(listForSave(draft)) !== JSON.stringify(listForSave(draftOf(saved, draft.kind)));
}

/** `items` with the item at `from` moved to `to` (clamped); a new array. */
export function moveItem(items, from, to) {
  const out = [...items];
  if (from < 0 || from >= out.length) return out;
  const target = Math.max(0, Math.min(out.length - 1, to));
  const [item] = out.splice(from, 1);
  out.splice(target, 0, item);
  return out;
}

/** The general list and the owner's lists of `kind`, from GET /mandatory/lists. */
export function listsOf(overview, kind) {
  const lists = overview?.lists ?? [];
  return {
    general: lists.find((l) => l.id === generalId(kind)) ?? null,
    own: lists.filter((l) => !l.general && l.kind === kind),
  };
}

/** The ids of the job offers that use `id` (from the overview's `offers`). */
export function offersUsing(overview, id) {
  return Object.entries(overview?.offers ?? {}).filter(([, used]) => used === id).map(([plan]) => plan);
}

/** The list remembered for `kind` in this browser, if it is still offered. */
export function rememberedList(kind, choices, storage = globalThis.localStorage) {
  let kept = null;
  try {
    kept = JSON.parse(storage?.getItem(LAST_LIST_KEY) ?? "null");
  } catch {
    kept = null;
  }
  const id = kept && typeof kept === "object" ? kept[kind] : null;
  return choices.some((c) => c.id === id) ? id : generalId(kind);
}

/** Keep the last list chosen for `kind` in this browser; false when storage refuses. */
export function rememberList(kind, id, storage = globalThis.localStorage) {
  try {
    let kept = {};
    try {
      kept = JSON.parse(storage.getItem(LAST_LIST_KEY) ?? "{}") ?? {};
    } catch {
      kept = {};
    }
    storage.setItem(LAST_LIST_KEY, JSON.stringify({ ...(typeof kept === "object" ? kept : {}), [kind]: id }));
    return true;
  } catch {
    return false;
  }
}

/**
 * The words naming a list a session used (`{ id, name, version, general }`):
 * `t` is the dictionary; null without a list.
 */
export function listUsedText(ref, t) {
  if (!ref || !ref.id) return null;
  if (ref.general || isGeneral(ref.id)) return t("mi.used_general", { version: ref.version ?? 0 });
  return t("mi.used_own", { name: ref.name ?? ref.id, version: ref.version ?? 0 });
}

/** The warnings of a review, one line each: `{ key, params }` (keys under `mi.warn.`). */
export function warningLines(warnings, itemText = (id) => id) {
  const out = [];
  for (const c of warnings?.close ?? []) {
    out.push({ key: "close", params: { item: itemText(c.item), other: itemText(c.other), example: c.example } });
  }
  for (const s of warnings?.short ?? []) {
    out.push({ key: "short", params: { item: itemText(s.item), example: s.example, n: s.words } });
  }
  return out;
}
