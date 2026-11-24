// «Datos de la candidata» (#235): the candidate's answers to the core
// questions on a session's result page. Compact first (one line per
// question, the answer, «Por confirmar» or «Pendiente»); then «Confirmar
// datos» when the conversation left proposals (confirm, edit or discard
// each, or confirm all at once: nothing counts until the interviewer
// does); under «Más información» how each answer was obtained, a field to
// correct it, and the deletion of the whole profile. The live page shows
// the same «Confirmar datos» step when a session ends. The pure parts are
// in lib/candidate.js.

import { h, fill } from "../dom.js";
import { t, lang } from "../i18n.js";
import { describeError } from "../errors.js";
import { coreCount, coreRows } from "../lib/candidate.js";
import { openDialog, toast } from "../ui.js";
import { moreInfo } from "./summary-first.js";

/**
 * «Confirmar datos»: the proposals read from the conversation, each with
 * «Confirmar», «Editar» (type the answer) and «Descartar», and one
 * «Confirmar todo». `onChange` gets the profile the orchestrator keeps
 * after each action. Null when there is nothing to confirm.
 */
export function confirmStep(profile, { api, onChange }) {
  const rows = coreRows(t, profile, lang).filter((r) => r.proposed);
  if (!rows.length || !api) return null;
  const box = h("section", { class: "sf-confirm stack tight", "aria-label": t("cand.confirm_title") });
  async function act(job) {
    try {
      const changed = await job();
      if (changed?.questions) onChange(changed);
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  }
  const ids = [profile.plan_id, profile.id];
  const line = (row) => {
    const input = h("input", { type: "text", maxlength: "280", value: row.proposal, hidden: true, "aria-label": t("cand.edit_label", { what: row.label }) });
    const save = h("button", { type: "button", class: "small", hidden: true }, t("cand.save"));
    const edit = h("button", { type: "button", class: "small ghost" }, t("cand.edit"));
    edit.addEventListener("click", () => {
      input.hidden = false;
      save.hidden = false;
      edit.hidden = true;
      input.focus?.();
    });
    save.addEventListener("click", () => {
      const text = input.value.trim();
      if (text) act(() => api.correctAnswer(...ids, row.id, text));
    });
    return h("li", { class: "sf-confirm-row" },
      h("span", { class: "sf-confirm-what" }, h("strong", {}, row.label), " ", row.proposal),
      h("span", { class: "row wrap tight" },
        h("button", { type: "button", class: "small", onClick: () => act(() => api.confirmAnswers(...ids, [row.id])) }, t("cand.confirm")),
        edit,
        h("button", { type: "button", class: "small ghost", onClick: () => act(() => api.discardProposal(...ids, row.id)) }, t("cand.discard")),
        input, save));
  };
  box.append(
    h("div", { class: "sf-cand-head" },
      h("h3", {}, t("cand.confirm_title")),
      rows.length > 1
        ? h("button", { type: "button", class: "small primary", onClick: () => act(() => api.confirmAnswers(...ids)) }, t("cand.confirm_all", { count: rows.length }))
        : null),
    h("p", { class: "hint" }, t("cand.confirm_hint")),
    h("ul", { class: "sf-confirm-list" }, rows.map(line)));
  return box;
}

/**
 * The block of one candidate. `api` corrects, clears and deletes (the
 * block draws itself again with the answer the orchestrator keeps);
 * without it the block is read-only. `onDeleted` runs once the profile
 * is gone.
 */
export function candidateBlock(profile, { api = null, onDeleted = null } = {}) {
  const box = h("section", { class: "card sf-candidate", "aria-labelledby": "sf-candidate-title" });
  let current = profile;

  async function act(job) {
    try {
      const changed = await job();
      if (changed?.questions) current = changed;
      draw();
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  }

  function editRow(row) {
    const input = h("input", { type: "text", maxlength: "280", value: row.text, "aria-label": t("cand.edit_label", { what: row.label }) });
    const save = h("button", { type: "button", class: "small" }, t("cand.save"));
    save.addEventListener("click", () => {
      const text = input.value.trim();
      if (!text) return;
      act(() => api.correctAnswer(current.plan_id, current.id, row.id, text));
    });
    const clear = row.answered
      ? h("button", { type: "button", class: "small ghost", onClick: () => act(() => api.clearAnswer(current.plan_id, current.id, row.id)) }, t("cand.clear"))
      : null;
    return h("div", { class: "row wrap tight sf-cand-edit" }, input, save, clear);
  }

  function detail(row) {
    return h("li", { class: "stack tight" },
      h("p", {}, h("strong", {}, row.label), " · ", row.question),
      h("p", {}, row.answered ? row.text || t("cand.answered") : t("cand.pending")),
      row.answered ? h("p", { class: "hint" }, t(row.source === "corrected" ? "cand.source_corrected" : "cand.source_confirmed")) : null,
      api ? editRow(row) : null);
  }

  async function remove() {
    const yes = await openDialog({
      title: t("cand.delete_title", { name: current.name }),
      content: [h("p", {}, t("cand.delete_body"))],
      confirm: t("cand.delete"),
      danger: true,
    });
    if (!yes) return;
    try {
      await api.deleteCandidate(current.plan_id, current.id);
      box.remove();
      toast(t("cand.deleted", { name: current.name }));
      onDeleted?.();
    } catch (error) {
      toast(describeError(error), { error: true });
    }
  }

  function draw() {
    const wasOpen = Boolean(box.querySelector?.("details.sf-more")?.open);
    const rows = coreRows(t, current, lang);
    const count = coreCount(current);
    const rounds = (current.rounds ?? []).length;
    fill(box,
      h("div", { class: "sf-cand-head" },
        h("h2", { id: "sf-candidate-title" }, t("cand.title")),
        h("span", { class: "meta" }, [current.name, t("cand.count", { done: count.done, all: count.all }),
          rounds > 1 ? t("cand.rounds", { count: rounds }) : ""].filter(Boolean).join(" · "))),
      h("dl", { class: "sf-cand-list" }, rows.map((row) => [
        h("dt", {}, row.label),
        h("dd", { class: row.answered ? "" : "is-pending" }, row.answered ? row.text || t("cand.answered")
          : row.proposed ? t("cand.proposed", { value: row.proposal }) : t("cand.pending")),
      ])),
      confirmStep(current, { api, onChange: (changed) => { current = changed; draw(); } }),
      moreInfo(t("cand.more_hint"), () => [
        h("ul", { class: "stack sf-cand-details" }, rows.map(detail)),
        h("p", { class: "hint" }, t("cand.privacy")),
        api ? h("button", { type: "button", class: "small danger", onClick: remove }, t("cand.delete")) : null,
      ], { eager: wasOpen }));
    if (wasOpen) box.querySelector("details.sf-more").open = true;
  }

  draw();
  return box;
}
