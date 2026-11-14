// Review of a lab run: the fragments to label (who really spoke, the
// right text) and the pipelines' scores against those labels.

import { h } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import { DEFAULT_LEAD, parseLead } from "../lib/transcript.js";
import {
  LabelQueue, bestBy, clearSpan, coveringLabel, exportPayload, groupAnswers, keyAction,
  labelProgress, labelsBody, makeLabel, placeLabel, playStart, textVariants,
} from "../lib/lab.js";
import { labPlayer } from "./lab-timeline.js";
import { num, pct, sortableTable } from "./lab-run.js";

const LEAD_KEY = "transcriber.lead"; // the same setting as the transcript player
const PAGE = 25;

function storedLead() {
  try {
    return parseLead(localStorage.getItem(LEAD_KEY) ?? "", DEFAULT_LEAD);
  } catch {
    return DEFAULT_LEAD;
  }
}

function storage() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function who(speakers) {
  return speakers?.length ? speakers.join(" + ") : t("lab.nobody");
}

function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = h("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---- fragments ------------------------------------------------------------

export async function fragmentsSection(body, env) {
  const { ctx, session, run, results } = env;
  let stored;
  let spans;
  try {
    [stored, spans] = await Promise.all([ctx.api.labLabels(session), ctx.api.labSpans(session, run).then((s) => s.spans)]);
  } catch (error) {
    if (error.kind === "unauthorized") throw error;
    body.replaceChildren(h("p", { class: "error", role: "alert" }, describeError(error)));
    return null;
  }
  const people = stored.speakers;
  const player = await labPlayer(ctx, env);
  let lead = storedLead();
  let sort = "disagreement";
  let unlabelled = false;
  let offset = 0;
  let total = 0;
  let fragments = [];
  let rows = [];
  let current = 0;

  const saveState = h("span", { class: "save-state", role: "status", "aria-live": "polite" });
  const progress = h("p", { class: "meta" });
  const queue = new LabelQueue({
    key: `transcriber.lab.labels.${session}`,
    storage: storage(),
    save: async (labels) => (await ctx.api.labPutLabels(session, labelsBody(labels))).labels,
    onChange: (state) => {
      saveState.textContent = t(`lab.save.${state}`) + (state === "failed" && queue.error ? ` (${describeError(queue.error)})` : "");
      saveState.className = `save-state ${state}`;
      showProgress();
    },
  });
  const restored = queue.load(stored.labels);

  function showProgress() {
    const p = labelProgress(queue.labels, spans, results.duration);
    progress.textContent = t("lab.label_progress", {
      seconds: formatTime(p.seconds), duration: formatTime(p.duration), fragments: p.fragments, total: p.total,
    });
  }

  const list = h("ol", { class: "fragments", "aria-label": t("lab.fragments") });
  const pager = h("div", { class: "row wrap" });
  const leadField = h("input", { id: "lab-lead", type: "number", min: 0, max: 60, step: 0.5, inputMode: "decimal", value: String(lead) });
  leadField.addEventListener("change", () => {
    lead = parseLead(leadField.value, DEFAULT_LEAD);
    leadField.value = String(lead);
    try {
      localStorage.setItem(LEAD_KEY, String(lead));
    } catch {
      // used for this visit only
    }
  });
  const sortSelect = h("select", { id: "lab-sort" }, ["disagreement", "time"].map((s) => h("option", { value: s }, t(`lab.sort.${s}`))));
  sortSelect.addEventListener("change", () => { sort = sortSelect.value; offset = 0; load(); });
  const onlyBox = h("input", { id: "lab-unlabelled", type: "checkbox" });
  onlyBox.addEventListener("change", () => { unlabelled = onlyBox.checked; offset = 0; load(); });

  const keys = [people.slice(0, 9).map((p, i) => `${i + 1} ${p}`).join(" · "), "B", "N", "U", "X", t("lab.key_space"), "J/K"];
  body.replaceChildren(
    h("p", { class: "hint" }, t("lab.review_about")),
    player.element,
    h("div", { class: "row wrap" },
      h("label", { for: "lab-sort" }, t("lab.sort")), sortSelect,
      h("label", { class: "check" }, onlyBox, h("span", {}, t("lab.only_unlabelled"))),
      h("label", { for: "lab-lead" }, t("player.lead")), leadField),
    h("div", { class: "line" }, progress, saveState),
    restored ? h("p", { class: "notice", role: "status" }, t("lab.restored")) : null,
    h("details", { class: "keys" }, h("summary", {}, t("lab.keys_title")), h("p", { class: "hint" }, t("lab.keys", { keys: keys.join(" · ") }))),
    list, pager);
  showProgress();
  queue.onChange(queue.state);

  function labelFor(fragment) {
    return coveringLabel(queue.labels, fragment.start, fragment.end);
  }

  function setLabel(index, kind, speakers = [], text = undefined) {
    const fragment = fragments[index];
    const previous = labelFor(fragment);
    const keep = text === undefined ? (previous && previous.start === fragment.start && previous.end === fragment.end ? previous.text : null) : text;
    if (kind === null) {
      queue.set(clearSpan(queue.labels, fragment.start, fragment.end));
    } else {
      const label = makeLabel(fragment, kind, speakers, keep);
      if (!label) return false;
      queue.set(placeLabel(queue.labels, label));
    }
    rows[index].replaceWith(rows[index] = row(fragment, index));
    return true;
  }

  function both(index) {
    if (people.length <= 2) return setLabel(index, "both", people);
    const choice = rows[index].querySelector(".both-choice");
    if (choice) choice.hidden = false;
    choice?.querySelector("input")?.focus();
    return false;
  }

  function focusRow(index) {
    if (!rows.length) return;
    current = Math.max(0, Math.min(rows.length - 1, index));
    rows[current].focus();
  }

  function playFragment(index) {
    const f = fragments[index];
    player.playFrom(playStart(f.start, lead), f.end);
  }

  function row(fragment, index) {
    const label = labelFor(fragment);
    const groups = groupAnswers(fragment.pipelines);
    const variants = textVariants(fragment.pipelines);
    const span = `${formatTime(fragment.start)}–${formatTime(fragment.end)} (${(fragment.end - fragment.start).toFixed(1)} s)`;
    const button = (text, kind, speakers, keyHint) => h("button", {
      type: "button", class: label && label.kind === kind && (kind !== "person" || label.speakers[0] === speakers[0]) ? "small chosen" : "small",
      "aria-pressed": label && label.kind === kind && (kind !== "person" || label.speakers[0] === speakers[0]) ? "true" : "false",
      onClick: () => { if (kind === "both" && !speakers) both(index); else if (setLabel(index, kind, speakers ?? [])) focusRow(index); },
    }, text, keyHint ? h("kbd", {}, keyHint) : null);
    const bothBoxes = people.length > 2 ? h("fieldset", { class: "both-choice", hidden: true }, h("legend", {}, t("lab.both_who")),
      people.map((p) => h("label", { class: "check" }, h("input", { type: "checkbox", value: p }), h("span", {}, p))),
      h("button", { type: "button", class: "small", onClick: (event) => {
        const chosen = [...event.target.closest("fieldset").querySelectorAll("input:checked")].map((i) => i.value);
        if (setLabel(index, "both", chosen)) focusRow(index);
      } }, t("lab.both_save"))) : null;
    const text = h("input", { type: "text", maxLength: 2000, autocomplete: "off", value: label?.text ?? "", "aria-label": t("lab.corrected_text"), placeholder: t("lab.corrected_text") });
    text.addEventListener("change", () => {
      const now = labelFor(fragment);
      if (!now) return;
      setLabel(index, now.kind, now.speakers, text.value);
    });
    return h("li", { class: label ? `fragment labelled kind-${label.kind}` : "fragment", tabindex: 0, "data-index": index,
      onFocus: () => { current = index; } },
    h("div", { class: "line" },
      h("span", { class: "time" }, span),
      h("span", { class: "hint" }, t("lab.disagreement", { value: pct(fragment.disagreement) })),
      h("button", { type: "button", class: "small", onClick: () => playFragment(index) }, t("lab.play_fragment"))),
    h("p", {}, h("strong", {}, t("lab.consensus_says")), " ", who(fragment.modal)),
    h("ul", { class: "plain answers" }, groups.map((g) => h("li", {}, h("details", {},
      h("summary", {}, t("lab.answer_group", { who: who(g.speakers), count: g.count })),
      h("p", { class: "hint ids" }, g.pipelines.join(", ")))))),
    variants.length ? h("details", { class: "variants" }, h("summary", {}, t("lab.text_variants", { count: variants.length })),
      h("ul", { class: "plain" }, variants.map((v) => h("li", {}, h("q", {}, v.text || t("lab.no_text")), " ",
        h("span", { class: "hint" }, t("lab.variant_sources", { count: v.pipelines, recognisers: v.recognisers.join(", ") })))))) : null,
    h("div", { class: "row wrap labelling", role: "group", "aria-label": t("lab.who_spoke") },
      people.map((p, i) => button(p, "person", [p], i < 9 ? String(i + 1) : null)),
      button(t("lab.both"), "both", null, "B"), button(t("lab.nobody"), "nobody", [], "N"),
      button(t("lab.unsure"), "unsure", [], "U"),
      label ? h("button", { type: "button", class: "small", onClick: () => { setLabel(index, null); focusRow(index); } }, t("lab.clear"), h("kbd", {}, "X")) : null),
    bothBoxes,
    label ? h("p", { class: "hint" }, t("lab.labelled_as", { what: t(`lab.kind.${label.kind}`), who: label.speakers.join(" + ") })) : null,
    label ? text : null);
  }

  list.addEventListener("keydown", (event) => {
    const action = keyAction(event, people.length);
    if (!action || !rows.length) return;
    event.preventDefault();
    const index = current;
    switch (action.action) {
      case "person": if (setLabel(index, "person", [people[action.index]])) focusRow(index + 1); break;
      case "both": if (both(index)) focusRow(index + 1); break;
      case "nobody": if (setLabel(index, "nobody")) focusRow(index + 1); break;
      case "unsure": if (setLabel(index, "unsure")) focusRow(index + 1); break;
      case "clear": setLabel(index, null); focusRow(index); break;
      case "play": if (player.audio.paused) playFragment(index); else player.audio.pause(); break;
      case "next": focusRow(index + 1); break;
      case "previous": focusRow(index - 1); break;
      default: break;
    }
  });

  async function load() {
    list.replaceChildren(h("li", { class: "hint" }, t("lab.loading")));
    let page;
    try {
      page = await ctx.api.labFragments(session, run, { offset, limit: PAGE, sort, unlabelled });
    } catch (error) {
      if (error.kind === "unauthorized") return ctx.fail(error);
      list.replaceChildren(h("li", { class: "error", role: "alert" }, describeError(error)));
      return;
    }
    fragments = page.fragments;
    total = page.total;
    rows = fragments.map((f, i) => row(f, i));
    list.replaceChildren(...(rows.length ? rows : [h("li", { class: "hint" }, t(unlabelled ? "lab.all_labelled" : "lab.no_fragments"))]));
    current = 0;
    pager.replaceChildren(
      h("button", { type: "button", disabled: offset === 0, onClick: () => { offset = Math.max(0, offset - PAGE); load(); } }, t("lab.previous_page")),
      h("span", { class: "meta" }, t("lab.page", { from: total ? offset + 1 : 0, to: Math.min(total, offset + PAGE), total })),
      h("button", { type: "button", disabled: offset + PAGE >= total, onClick: () => { offset += PAGE; load(); } }, t("lab.next_page")));
  }

  const leave = () => { queue.flush(); };
  window.addEventListener("pagehide", leave);
  await load();
  return () => {
    window.removeEventListener("pagehide", leave);
    queue.flush();
    player.stop();
  };
}

// ---- scores ----------------------------------------------------------------

const SCORE_COLUMNS = [
  ["pipeline", "lab.col.pipeline", (v) => v],
  ["rank", "lab.col.rank", (v) => v ?? "—"],
  ["error_rate", "lab.col.error_rate", pct],
  ["missed", "lab.col.missed", (v) => num(v)],
  ["false_alarm", "lab.col.false_alarm", (v) => num(v)],
  ["confusion", "lab.col.confusion", (v) => num(v)],
  ["words_right_share", "lab.col.words_right", pct],
  ["overlap_detected", "lab.col.overlap_detected", pct],
  ["wer", "lab.col.wer", pct],
  ["labelled_seconds", "lab.col.labelled", (v) => num(v)],
];

function bestTable(entries, groupLabel, choiceLabel, format) {
  if (!entries.length) return null;
  return h("table", { class: "data compact" },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, groupLabel), h("th", { scope: "col" }, choiceLabel), h("th", { scope: "col" }, t("lab.value")))),
    h("tbody", {}, entries.map((e) => h("tr", {}, h("th", { scope: "row" }, e.group), h("td", {}, e.choice), h("td", {}, format(e.value))))));
}

export async function scoresSection(body, env) {
  const { ctx, session, run } = env;
  let scores;
  let labels;
  try {
    [scores, labels] = await Promise.all([ctx.api.labScores(session, run), ctx.api.labLabels(session)]);
  } catch (error) {
    if (error.kind === "unauthorized") throw error;
    body.replaceChildren(h("p", { class: "error", role: "alert" }, describeError(error)));
    return null;
  }
  const exportButton = h("button", { type: "button", onClick: () => download(`${session}.${run.slice(0, 8)}.review.json`, exportPayload(session, run, labels.labels, scores)) }, t("lab.export"));
  if (!scores.labels) {
    body.replaceChildren(h("p", { class: "notice" }, t("lab.no_labels")),
      h("a", { class: "button", href: `#/lab/${encodeURIComponent(session)}/${encodeURIComponent(run)}/fragments` }, t("lab.go_label")));
    return null;
  }
  const rows = scores.scores;
  body.replaceChildren(
    h("p", { class: "notice", role: "note" }, t("lab.scores_meta", {
      labels: scores.labels, seconds: num(scores.labelled_seconds), unsure: num(scores.unsure_seconds), texts: scores.text_labels,
    })),
    h("p", { class: "hint" }, t("lab.scores_about")),
    sortableTable(rows, SCORE_COLUMNS, { initial: "rank", descending: false, rowClass: (r) => (r.licence_commercial === false ? "flag-nc" : null) }),
    h("h2", {}, t("lab.by_stage")),
    h("div", { class: "table-wrap stack" },
      h("h3", {}, t("lab.best_denoiser_per_diarizer")), bestTable(bestBy(rows, 1, 0, "error_rate"), t("lab.diarizer"), t("lab.denoiser"), pct),
      h("h3", {}, t("lab.best_diarizer_per_denoiser")), bestTable(bestBy(rows, 0, 1, "error_rate"), t("lab.denoiser"), t("lab.diarizer"), pct),
      h("h3", {}, t("lab.best_denoiser_per_recogniser")), bestTable(bestBy(rows, 2, 0, "wer"), t("lab.recogniser"), t("lab.denoiser"), pct) ?? h("p", { class: "hint" }, t("lab.no_text_labels"))),
    exportButton);
  return null;
}
