// «Lo que hay que decir» (#294, docs/live/DEFINE_MANDATORY.md): the owner's
// lists of what must be said, per call-centre campaign or commercial offer
// (support calls) and per job offer (interviews).
//
// - #/mandatory: the lists of each kind: the general list (read-only, from
//   the product's knowledge domain; it can be copied) and the owner's lists
//   (edit, copy, delete with an in-page confirmation), «Nueva lista».
// - #/mandatory/<id>: one list's editor (`new-support` / `new-interview` for
//   a new one): name, adds to / replaces the general list, items with their
//   moment and example sentences, reorder, save (a new version), and the
//   warnings of the review (examples close to another item's, very short
//   examples): heuristics, never a block.
//
// Synthetic wording; the owner is responsible for the legal texts. Pure
// parts in lib/mandatory.js.

import { h, fill } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { emptyState, icon, openDialog, pageHead, toast } from "../ui.js";
import {
  DEFAULT_LIMITS, KINDS, MOMENTS, draftOf, generalId, isDirty, isGeneral, listErrors, listForSave, listsOf, moveItem,
  newItem, offersUsing, warningLines,
} from "../lib/mandatory.js";

/** What a refused save says, in plain words (the server's codes). */
function saveError(error) {
  if (error?.code === "conflict") return t("mi.err.conflict");
  if (error?.code === "list_too_big") return t("mi.err.too_big");
  if (error?.code === "unavailable") return t("mi.err.unavailable");
  return describeError(error);
}

/** `#/mandatory/<id>` of a list. */
export function listHref(id) {
  return `#/mandatory/${encodeURIComponent(id)}`;
}

/** `#/mandatory` or `#/mandatory/<id>`. */
export async function mandatoryView(ctx, root, [listId] = []) {
  root.classList.add("wide");
  const id = listId ? decodeURIComponent(listId) : null;
  const overview = await ctx.api.mandatoryLists();
  if (!id) {
    root.append(listsPage(ctx, overview, { reload: () => ctx.navigate("#/mandatory") }));
    return null;
  }
  ctx.setCrumbs?.(crumbs(id.startsWith("new-") ? t("mi.new") : id));
  const fresh = id.match(/^new-(support|interview)$/);
  if (fresh) {
    root.append(pageHead(t("mi.new_title", { kind: t(`mi.kind.${fresh[1]}`) })),
      listEditor(ctx, null, { kind: fresh[1], limits: overview.limits, onSaved: (saved) => {
        // A new list opens as its own page; the toast is the one announcement.
        toast(t("mi.saved", { name: saved.name, n: saved.version }));
        ctx.navigate(listHref(saved.id));
      } }));
    return null;
  }
  let list;
  try {
    list = await ctx.api.mandatoryList(id);
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("nav.mandatory")), emptyState({
      iconName: "file", title: t("mi.missing"), body: error?.code === "not_found" ? t("mi.not_found") : describeError(error),
      action: h("a", { class: "button", href: "#/mandatory" }, t("mi.back")),
    }));
    return null;
  }
  ctx.setCrumbs?.(crumbs(listName(list)));
  if (list.general) {
    root.append(pageHead(listName(list), { lead: t("mi.general_lead") }), generalCard(ctx, list));
    return null;
  }
  root.append(pageHead(listName(list), { lead: t(`mi.kind.${list.kind}`) }),
    listEditor(ctx, list, { limits: overview.limits }));
  return null;
}

/** Administración › Lo que hay que decir › `last`. */
function crumbs(last) {
  return [{ label: t("nav.admin"), href: "#/admin" }, { label: t("nav.mandatory"), href: "#/mandatory" }, { label: last, href: null }];
}

/** The name shown for a list (the general one has its own words). */
export function listName(list) {
  return list?.general || isGeneral(list?.id) ? t("mi.general_name") : list?.name ?? list?.id ?? "";
}

// ---- #/mandatory ------------------------------------------------------------

/** The page with every list (exported for the tests). */
export function listsPage(ctx, overview, { reload = () => {} } = {}) {
  return h("div", { class: "stack" },
    pageHead(t("mi.title"), { lead: t("mi.lead") }),
    h("p", { class: "notice info", role: "note" }, t("mi.how")),
    KINDS.map((kind) => kindSection(ctx, overview, kind, reload)),
    h("p", { class: "hint" }, t("mi.legal")));
}

function kindSection(ctx, overview, kind, reload) {
  const { general, own } = listsOf(overview, kind);
  const titleId = `mi-${kind}-title`;
  return h("section", { class: "stack tight mi-kind", "aria-labelledby": titleId },
    h("div", { class: "row wrap mi-kind-head" },
      h("h2", { id: titleId }, t(`mi.kind.${kind}`)),
      h("a", { class: "button primary", href: listHref(`new-${kind}`) }, icon("plus"), t("mi.new"))),
    h("p", { class: "hint" }, t(`mi.kind_hint.${kind}`)),
    h("ul", { class: "mi-cards" },
      general ? h("li", { class: "card mi-card is-general" },
        h("div", { class: "row wrap" }, h("h3", {}, h("a", { href: listHref(general.id) }, t("mi.general_name"))),
          h("span", { class: "tag" }, t("mi.read_only"))),
        h("p", { class: "meta" }, general.unavailable ? t("mi.general_unavailable") : t("mi.items_n", { n: general.items ?? 0 })),
        h("div", { class: "row wrap" }, copyButton(ctx, general, kind))) : null,
      own.map((list) => ownCard(ctx, overview, list, kind, reload))),
    own.length ? null : h("p", { class: "hint" }, t("mi.none_yet")));
}

function ownCard(ctx, overview, list, kind, reload) {
  const used = offersUsing(overview, list.id).length;
  const remove = h("button", { type: "button", class: "small danger" }, icon("trash"), t("mi.delete"));
  remove.addEventListener("click", async () => {
    const yes = await openDialog({
      title: t("mi.delete_title", { name: list.name }),
      content: [h("p", {}, t("mi.delete_body")), used ? h("p", { class: "hint" }, t("mi.delete_offers", { n: used })) : null],
      confirm: t("mi.delete"), danger: true,
    });
    if (!yes) return;
    try {
      await ctx.api.deleteMandatoryList(list.id);
      toast(t("mi.deleted", { name: list.name }));
      reload();
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  });
  return h("li", { class: "card mi-card" },
    h("div", { class: "row wrap" }, h("h3", {}, h("a", { href: listHref(list.id) }, list.name)),
      h("span", { class: "tag" }, t(`mi.mode.${list.mode === "replace" ? "replace" : "add"}`))),
    h("p", { class: "meta" }, [t("mi.items_n", { n: list.items ?? 0 }), t("mi.version", { n: list.version ?? 1 }),
      kind === "interview" ? t("mi.offers_n", { n: used }) : null].filter(Boolean).join(" · ")),
    h("div", { class: "row wrap" },
      h("a", { class: "button small", href: listHref(list.id) }, icon("edit"), t("mi.edit")),
      copyButton(ctx, list, kind), remove));
}

/** «Duplicar»: asks for the new name in the page, then opens the copy. */
function copyButton(ctx, list, kind) {
  const button = h("button", { type: "button", class: "small" }, t("mi.copy"));
  button.addEventListener("click", () => copyList(ctx, list, kind));
  return button;
}

/** Copy `list` under a name asked for in a dialog; resolves to the copy (or null). */
export async function copyList(ctx, list, kind, { name = null, open = true } = {}) {
  const input = h("input", { id: "mi-copy-name", type: "text", maxLength: DEFAULT_LIMITS.name, autocomplete: "off",
    value: name ?? t("mi.copy_name", { name: listName(list) }) });
  const problem = h("p", { class: "error", role: "alert", hidden: true }, t("mi.err.name"));
  const yes = await openDialog({
    title: t("mi.copy_title", { name: listName(list) }),
    content: [h("div", { class: "field" }, h("label", { for: "mi-copy-name" }, t("mi.name")), input),
      h("p", { class: "hint" }, t(list.general || isGeneral(list.id) ? "mi.copy_general_hint" : "mi.copy_hint")), problem],
    confirm: t("mi.copy"),
    onConfirm: () => {
      problem.hidden = Boolean(input.value.trim()); // hidden when a name is given
      return problem.hidden;
    },
  });
  if (!yes) return null;
  try {
    const made = await ctx.api.createMandatoryList({ from: list.id ?? generalId(kind), name: input.value.trim() });
    toast(t("mi.copied", { name: made.name }));
    if (open) ctx.navigate(listHref(made.id));
    return made;
  } catch (error) {
    toast(saveError(error), { error: true }); // the general list down, too big: their own words
    return null;
  }
}

/** A general list, read-only, with «Duplicar». */
function generalCard(ctx, list) {
  return h("section", { class: "card mi-edit", "aria-labelledby": "mi-general-title" },
    h("h2", { id: "mi-general-title" }, t(`mi.kind.${list.kind}`)),
    h("p", { class: "hint" }, list.domain ? t("mi.general_source", { domain: list.domain, version: list.version ?? 0 }) : null),
    h("ol", { class: "mi-items" }, (list.items ?? []).map((item) => h("li", { class: "mi-item" },
      h("strong", {}, item.text), " ", item.phase ? h("span", { class: "tag" }, t(`mi.moment.${item.phase}`)) : null,
      h("ul", { class: "mi-examples-ro" }, (item.must_say ?? []).map((e) => h("li", {}, h("q", {}, e))))))),
    h("div", { class: "row wrap" }, copyButton(ctx, list, list.kind),
      h("a", { class: "button ghost", href: "#/mandatory" }, t("mi.back"))));
}

// ---- the editor ---------------------------------------------------------------

/**
 * The editor of one list (`saved` null: a new list of `kind`). Typing changes
 * the draft without drawing the page again (the focus stays); adding, moving
 * and removing draw the items again and put the focus on the moved control.
 * `status` says once what happened (saved, checked, an error).
 */
export function listEditor(ctx, saved, { kind = saved?.kind ?? "support", limits = DEFAULT_LIMITS, onSaved = () => {} } = {}) {
  const bounds = { ...DEFAULT_LIMITS, ...(limits ?? {}) };
  let base = saved;
  let draft = draftOf(saved, kind);
  const box = h("section", { class: "card mi-edit", "aria-labelledby": "mi-edit-title" });
  const status = h("p", { class: "meta mi-status", role: "status", "aria-live": "polite" });
  const errorsBox = h("div", { class: "mi-errors" });
  const warnBox = h("div", { class: "mi-warnings" });
  const itemsBox = h("ol", { class: "mi-items" });
  const save = h("button", { type: "button", class: "primary" }, t("mi.save"));
  const review = h("button", { type: "button" }, t("mi.review"));
  const unsaved = h("span", { class: "hint", hidden: true }, t("mi.unsaved"));
  const name = h("input", { id: "mi-name", type: "text", maxLength: bounds.name, autocomplete: "off", value: draft.name });
  name.addEventListener("input", () => { draft.name = name.value; sync(); });
  const modes = h("fieldset", { class: "field mi-mode" }, h("legend", {}, t("mi.mode_title")),
    ["add", "replace"].map((mode) => {
      const radio = h("input", { type: "radio", name: "mi-mode", id: `mi-mode-${mode}`, value: mode, checked: draft.mode === mode });
      radio.addEventListener("change", () => { if (radio.checked) { draft.mode = mode; sync(); } });
      return h("label", { class: "check", for: `mi-mode-${mode}` }, radio, " ",
        h("span", {}, h("strong", {}, t(`mi.mode.${mode}`)), " ", h("span", { class: "hint" }, t(`mi.mode_hint.${mode}`))));
    }));

  /** Say `text` once (the same words again are not repeated). */
  function say(text) {
    if (status.textContent !== text) status.textContent = text;
  }

  function sync() {
    const errors = listErrors(draft, bounds);
    const dirty = !base || isDirty(draft, base);
    // aria-disabled, not disabled: «Guardar» keeps the focus after a save.
    save.setAttribute("aria-disabled", String(!dirty || errors.length > 0));
    unsaved.hidden = !dirty || !base;
    fill(errorsBox, errors.length && dirty ? h("ul", { class: "hint mi-error-list" },
      errors.map((e) => h("li", {}, t(`mi.err.${e.key}`, { n: e.n, max: e.max })))) : null);
  }

  function focusLater(key) {
    const target = box.querySelector?.(`[data-focus-key="${key}"]`);
    target?.focus?.();
  }

  function drawItems(focusKey = null) {
    fill(itemsBox, draft.items.map((item, i) => itemEditor(item, i)));
    sync();
    if (focusKey) focusLater(focusKey);
  }

  function itemEditor(item, i) {
    const n = i + 1;
    const label = item.text.trim() || t("mi.item_n", { n });
    const text = h("input", { id: `mi-item-${i}-text`, type: "text", maxLength: bounds.text, autocomplete: "off", value: item.text,
      "data-focus-key": `item-${i}-text` });
    text.addEventListener("input", () => { item.text = text.value; sync(); });
    const phase = h("select", { id: `mi-item-${i}-phase` },
      h("option", { value: "", selected: !item.phase }, t("mi.moment.none")),
      MOMENTS.map((m) => h("option", { value: m, selected: item.phase === m }, t(`mi.moment.${m}`))));
    phase.addEventListener("change", () => { item.phase = phase.value || null; sync(); });
    const examples = h("ul", { class: "mi-examples" }, item.must_say.map((example, j) => {
      const input = h("input", { id: `mi-item-${i}-ex-${j}`, type: "text", maxLength: bounds.text, autocomplete: "off", value: example,
        "aria-label": t("mi.example_n", { n: j + 1, item: label }), "data-focus-key": `item-${i}-ex-${j}` });
      input.addEventListener("input", () => { item.must_say[j] = input.value; sync(); });
      const drop = h("button", { type: "button", class: "small ghost", disabled: item.must_say.length < 2,
        "aria-label": t("mi.example_remove", { n: j + 1, item: label }) }, icon("close"));
      drop.addEventListener("click", () => {
        item.must_say.splice(j, 1);
        drawItems(`item-${i}-ex-${Math.max(0, j - 1)}`);
      });
      return h("li", { class: "row mi-example" }, input, drop);
    }));
    const more = h("button", { type: "button", class: "small", disabled: item.must_say.length >= bounds.examples,
      "data-focus-key": `item-${i}-more` }, icon("plus"), t("mi.example_add"));
    more.addEventListener("click", () => {
      item.must_say.push("");
      drawItems(`item-${i}-ex-${item.must_say.length - 1}`);
    });
    const move = (to, key) => {
      draft.items = moveItem(draft.items, i, to);
      drawItems(`item-${Math.max(0, Math.min(draft.items.length - 1, to))}-${key}`);
      say(t("mi.moved", { item: label, n: Math.max(1, Math.min(draft.items.length, to + 1)) }));
    };
    const up = h("button", { type: "button", class: "small", disabled: i === 0, "aria-label": t("mi.move_up", { item: label }),
      "data-focus-key": `item-${i}-up` }, "↑");
    up.addEventListener("click", () => move(i - 1, "up"));
    const down = h("button", { type: "button", class: "small", disabled: i === draft.items.length - 1,
      "aria-label": t("mi.move_down", { item: label }), "data-focus-key": `item-${i}-down` }, "↓");
    down.addEventListener("click", () => move(i + 1, "down"));
    const remove = h("button", { type: "button", class: "small danger", "aria-label": t("mi.item_remove", { item: label }) },
      icon("trash"));
    remove.addEventListener("click", () => {
      draft.items.splice(i, 1);
      drawItems(draft.items.length ? `item-${Math.max(0, i - 1)}-text` : "add-item");
      say(t("mi.item_removed", { item: label }));
    });
    return h("li", { class: "mi-item" },
      h("fieldset", { class: "stack tight" },
        h("legend", {}, t("mi.item_n", { n })),
        h("div", { class: "field" }, h("label", { for: `mi-item-${i}-text` }, t("mi.item_text")), text),
        h("div", { class: "field" }, h("label", { for: `mi-item-${i}-phase` }, t("mi.moment_label")), phase),
        h("div", { class: "field" }, h("span", { class: "label" }, t("mi.examples")),
          h("p", { class: "hint" }, t("mi.examples_hint")), examples, more),
        h("div", { class: "row wrap mi-item-actions" }, up, down, remove)));
  }

  const add = h("button", { type: "button", "data-focus-key": "add-item" }, icon("plus"), t("mi.item_add"));
  add.addEventListener("click", () => {
    if (draft.items.length >= bounds.items) return say(t("mi.err.too_many", { max: bounds.items }));
    draft.items.push(newItem());
    return drawItems(`item-${draft.items.length - 1}-text`);
  });

  function showWarnings(warnings) {
    const lines = warningLines(warnings, (id) => draft.items.find((i) => i.id === id)?.text ?? id);
    fill(warnBox, h("div", { class: lines.length ? "notice" : "notice info" },
      h("strong", {}, lines.length ? t("mi.warn_title", { n: lines.length }) : t("mi.warn_none")),
      lines.length ? h("ul", {}, lines.map((l) => h("li", {}, t(`mi.warn.${l.key}`, l.params)))) : null,
      warnings?.checked === false ? h("p", { class: "hint" }, t("mi.warn_unchecked")) : null,
      h("p", { class: "hint" }, t("mi.warn_heuristic"))));
  }

  review.addEventListener("click", async () => {
    review.disabled = true;
    try {
      const found = await ctx.api.reviewMandatoryList(listForSave(draft));
      showWarnings(found);
      say(t("mi.reviewed"));
    } catch (error) {
      say(describeError(error));
    } finally {
      review.disabled = false;
    }
  });

  let saving = false;
  save.addEventListener("click", async () => {
    if (saving || save.getAttribute("aria-disabled") === "true") return;
    saving = true;
    save.setAttribute("aria-disabled", "true");
    try {
      const body = listForSave(draft);
      const fresh = !base;
      const stored = base ? await ctx.api.saveMandatoryList(base.id, body) : await ctx.api.createMandatoryList(body);
      base = stored;
      // The items stay as they are (no redraw, the focus stays): only the
      // server's ids and the new version come into the draft.
      draft.id = stored.id;
      draft.version = stored.version;
      (stored.items ?? []).forEach((item, i) => { if (draft.items[i]) draft.items[i].id = item.id; });
      sync();
      showWarnings(stored.warnings);
      if (!fresh) say(t("mi.saved", { name: stored.name, n: stored.version })); // one announcement
      onSaved(stored);
    } catch (error) {
      say(saveError(error));
      sync();
    } finally {
      saving = false;
    }
  });

  fill(box,
    h("h2", { id: "mi-edit-title" }, t(saved ? "mi.edit_title" : "mi.new")),
    h("p", { class: "hint" }, t("mi.edit_hint")),
    h("div", { class: "field" }, h("label", { for: "mi-name" }, t("mi.name")), name),
    modes,
    itemsBox,
    h("div", { class: "row wrap" }, add),
    errorsBox,
    warnBox,
    h("div", { class: "row wrap" }, save, review, unsaved, h("a", { class: "button ghost", href: "#/mandatory" }, t("mi.back"))),
    status,
    h("p", { class: "hint" }, t("mi.legal")));
  drawItems();
  if (saved?.warnings) showWarnings(saved.warnings);
  else if (saved && ctx.api?.reviewMandatoryList) {
    // A saved list shows its warnings when it opens (nothing is announced).
    ctx.api.reviewMandatoryList(listForSave(draft)).then(showWarnings).catch(() => {});
  }
  return box;
}

// ---- on the offer page ------------------------------------------------------------

/**
 * «Lo que hay que decir» of a job offer (the offer page): which list its
 * interviews use, chosen from the interview lists; «Crear una para esta
 * oferta» copies the general list under the offer's title and opens it.
 */
export function offerListCard(ctx, plan) {
  const box = h("section", { class: "card mi-offer", "aria-labelledby": "mi-offer-title" });
  if (!ctx?.api?.mandatoryLists) return null;
  const status = h("p", { class: "meta", role: "status", "aria-live": "polite" });
  async function draw() {
    let overview;
    let used;
    try {
      [overview, used] = await Promise.all([ctx.api.mandatoryLists(), ctx.api.offerMandatory(plan.id)]);
    } catch (error) {
      if (error?.kind === "unauthorized") return;
      fill(box, h("h2", { id: "mi-offer-title" }, t("mi.offer_title")), h("p", { class: "hint" }, describeError(error)));
      return;
    }
    const { own } = listsOf(overview, "interview");
    const current = used?.list ?? generalId("interview");
    const select = h("select", { id: "mi-offer-list" },
      h("option", { value: generalId("interview"), selected: current === generalId("interview") }, t("mi.general_name")),
      own.map((l) => h("option", { value: l.id, selected: l.id === current }, l.name)));
    const edit = h("a", { class: "button small", href: listHref(current), hidden: isGeneral(current) }, icon("edit"), t("mi.edit"));
    let shown = current;
    // The same select stays (the keyboard keeps its place): only the link and the status change.
    select.addEventListener("change", async () => {
      const chosen = select.value;
      try {
        const answer = await ctx.api.setOfferMandatory(plan.id, isGeneral(chosen) ? null : chosen);
        shown = chosen;
        edit.href = listHref(chosen);
        edit.hidden = isGeneral(chosen);
        status.textContent = t("mi.offer_set", { name: answer?.name ?? t("mi.general_name") });
      } catch (error) {
        select.value = shown;
        status.textContent = describeError(error);
      }
    });
    const create = h("button", { type: "button", class: "small" }, icon("plus"), t("mi.offer_create"));
    create.addEventListener("click", async () => {
      const made = await copyList(ctx, { id: generalId("interview"), general: true, kind: "interview" }, "interview",
        { name: plan.title, open: false });
      if (!made) return;
      try {
        await ctx.api.setOfferMandatory(plan.id, made.id);
        ctx.navigate?.(listHref(made.id));
      } catch (error) {
        // The copy exists (it is on the lists page): say so, never a silent orphan.
        status.textContent = t("mi.offer_assign_failed", { name: made.name, error: describeError(error) });
      }
    });
    fill(box,
      h("h2", { id: "mi-offer-title" }, t("mi.offer_title")),
      h("p", { class: "hint" }, t("mi.offer_hint")),
      h("div", { class: "row wrap mi-offer-row" },
        h("div", { class: "field" }, h("label", { for: "mi-offer-list" }, t("mi.offer_label")), select),
        edit, create),
      status);
  }
  fill(box, h("h2", { id: "mi-offer-title" }, t("mi.offer_title")), h("p", { class: "hint" }, t("mi.loading")));
  draw();
  return box;
}
