// «Conocimiento» (#/knowledge, #/knowledge/<domain>): knowledge domains and
// the company documents embedded in them. Drag files onto the zone (or
// pick them), watch each one go from queued to indexed, edit a document's
// metadata, delete it, and try a search. The state lives in
// lib/knowledge.js (KnowledgeModel); this module only draws it.

import { h, fill, keepFocus } from "../dom.js";
import { t, lang, decimal } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  ACCEPT, CATEGORIES, MODES, countByStatus, extensionOf, formatBytes, isDomainId, isMedia, KnowledgeModel, parseTags, slugify,
  timeRange,
} from "../lib/knowledge.js";
import { emptyState, icon, openDialog, pageHead, toast } from "../ui.js";

/** Plain-language text for a failed knowledge call (the server's message for a refused request). */
function knowledgeError(error) {
  if (error?.code === "unsupported_format") return t("kn.err_format");
  if (error?.code === "not_found") return t("kn.err_not_found");
  const base = describeError(error);
  return error?.kind === "client" && error.message && !base.includes(error.message) ? `${base} (${error.message})` : base;
}

function isPendingStatus(status) {
  return status === "queued" || status === "processing";
}

function dateText(iso) {
  if (!iso) return "";
  const when = new Date(iso);
  return Number.isNaN(when.getTime()) ? "" : when.toLocaleString(lang, { dateStyle: "medium", timeStyle: "short" });
}

function statusBadge(status, media = false) {
  const key = media && status === "processing" ? "kn.status.transcribing" : `kn.status.${status}`;
  return h("span", { class: `badge is-${status}` }, t(key));
}

/** The metadata fields (category, intent, phase, tags) prefilled from `metadata`. */
function metadataFields(prefix, metadata = {}) {
  const category = h("select", { id: `${prefix}-category` },
    h("option", { value: "" }, t("kn.meta.none")),
    CATEGORIES.map((c) => h("option", { value: c }, t(`kn.cat.${c}`))));
  category.value = metadata.category ?? "";
  const intent = h("input", { id: `${prefix}-intent`, type: "text", maxLength: 100, autocomplete: "off", value: metadata.intent ?? "" });
  const phase = h("input", { id: `${prefix}-phase`, type: "text", maxLength: 100, autocomplete: "off", value: metadata.script_phase ?? "" });
  const tags = h("input", { id: `${prefix}-tags`, type: "text", maxLength: 400, autocomplete: "off", value: (metadata.tags ?? []).join(", "), placeholder: t("kn.meta.tags_placeholder") });
  const field = (input, key, hint = null) => h("div", { class: "field" },
    h("label", { for: input.id }, t(key)), input, hint ? h("p", { class: "hint" }, t(hint)) : null);
  return {
    element: h("div", { class: "form-grid" },
      field(category, "kn.meta.category"), field(intent, "kn.meta.intent"),
      field(phase, "kn.meta.phase"), h("div", { class: "field" },
        h("label", { for: tags.id }, t("kn.meta.tags")), tags)),
    value: () => ({ category: category.value, intent: intent.value, script_phase: phase.value, tags: parseTags(tags.value) }),
  };
}

function metadataChips(metadata = {}) {
  const chips = [];
  if (metadata.category) chips.push(h("span", { class: "tag accent" }, t(`kn.cat.${metadata.category}`)));
  if (metadata.intent) chips.push(h("span", { class: "tag" }, `${t("kn.meta.intent")}: ${metadata.intent}`));
  if (metadata.script_phase) chips.push(h("span", { class: "tag" }, `${t("kn.meta.phase")}: ${metadata.script_phase}`));
  for (const tag of metadata.tags ?? []) chips.push(h("span", { class: "tag" }, `#${tag}`));
  return chips.length ? h("div", { class: "tags" }, chips) : null;
}

export async function knowledgeView(ctx, root, [wanted = null] = []) {
  root.classList.add("wide");
  const newDomain = h("button", { type: "button", class: "primary", onClick: () => createDomain() }, icon("plus"), t("kn.new_domain"));
  const body = h("div", { class: "stack" });
  root.append(pageHead(t("kn.title"), { lead: t("kn.lead"), actions: [newDomain] }), body);

  const model = new KnowledgeModel(ctx.api, { onChange: () => render() });
  // Parts kept across redraws so typing, the file picker and focus survive the 2 s polling.
  let shownDomain = undefined;
  let parts = null;

  async function act(call, done = null) {
    try {
      await call();
      if (done) toast(done);
    } catch (error) {
      if (error?.kind === "unauthorized") return ctx.fail(error);
      toast(knowledgeError(error), { error: true });
    }
  }

  async function createDomain() {
    const title = h("input", { id: "kn-new-title", type: "text", maxLength: 200, required: true, autocomplete: "off" });
    const id = h("input", { id: "kn-new-id", type: "text", maxLength: 64, required: true, autocomplete: "off", pattern: "[a-z0-9][a-z0-9._\\-]{0,63}" });
    const language = h("select", { id: "kn-new-lang" }, h("option", { value: "es" }, t("kn.lang.es")), h("option", { value: "en" }, t("kn.lang.en")));
    language.value = lang;
    const problem = h("p", { class: "error", role: "alert", hidden: true });
    let idEdited = false;
    title.addEventListener("input", () => { if (!idEdited) id.value = slugify(title.value); });
    id.addEventListener("input", () => { idEdited = true; });
    const ok = await openDialog({
      title: t("kn.new_domain"),
      confirm: t("kn.create"),
      content: [
        h("div", { class: "field" }, h("label", { for: title.id }, t("kn.domain_title")), title),
        h("div", { class: "field" }, h("label", { for: id.id }, t("kn.domain_id")), id, h("p", { class: "hint" }, t("kn.domain_id_hint"))),
        h("div", { class: "field" }, h("label", { for: language.id }, t("kn.domain_language")), language),
        problem,
      ],
      onConfirm: () => {
        const valid = title.value.trim() && isDomainId(id.value);
        problem.hidden = Boolean(valid);
        problem.textContent = valid ? "" : t("kn.domain_invalid");
        return Boolean(valid);
      },
    });
    if (!ok) return;
    await act(async () => {
      await model.createDomain({ id: id.value, title: title.value, language: language.value });
      ctx.navigate(`#/knowledge/${encodeURIComponent(id.value)}`);
    }, t("kn.domain_created"));
  }

  async function deleteDomain(domain) {
    const sure = await openDialog({
      title: t("kn.delete_domain_title", { name: domain.title || domain.id }),
      content: [h("p", {}, t("kn.delete_domain_body", { n: domain.documents ?? 0 }))],
      confirm: t("kn.delete_domain"),
      danger: true,
    });
    if (sure) {
      await act(async () => {
        await model.deleteDomain(domain.id);
        ctx.navigate(model.state.domain ? `#/knowledge/${encodeURIComponent(model.state.domain)}` : "#/knowledge");
      }, t("kn.domain_deleted"));
    }
  }

  async function editDocument(doc) {
    const fields = metadataFields("kn-edit", doc.metadata ?? {});
    const ok = await openDialog({ title: t("kn.edit_title", { name: doc.name }), content: [fields.element], confirm: t("kn.save") });
    if (ok) await act(() => model.updateDocument(doc.id, fields.value()), t("kn.saved"));
  }

  async function deleteDocument(doc) {
    const sure = await openDialog({
      title: t("kn.delete_doc_title"), content: [h("p", {}, t("kn.delete_doc_body", { name: doc.name }))],
      confirm: t("kn.delete_doc"), danger: true,
    });
    if (sure) await act(() => model.deleteDocument(doc.id), t("kn.doc_deleted"));
  }

  // ---- drawing ----------------------------------------------------------

  function domainList() {
    const { domains, domain } = model.state;
    return h("aside", { class: "kn-domains", "aria-label": t("kn.domains") },
      h("p", { class: "nav-group" }, t("kn.domains")),
      h("ul", {}, domains.map((d) => h("li", {},
        h("a", {
          class: "kn-domain", href: `#/knowledge/${encodeURIComponent(d.id)}`, "aria-current": d.id === domain ? "page" : null,
          "data-focus-key": `domain:${d.id}`,
        },
        h("span", { class: "name" }, d.title || d.id),
        h("span", { class: "meta" }, t("kn.domain_counts", { docs: d.documents ?? 0, chunks: d.chunks ?? 0 })))))));
  }

  function dropzone() {
    const input = h("input", { id: "kn-files", type: "file", multiple: true, accept: ACCEPT });
    const metadata = metadataFields("kn-up");
    const zone = h("label", { class: "dropzone", for: input.id },
      icon("upload"),
      h("strong", {}, t("kn.drop_title")),
      h("span", {}, t("kn.drop_body")),
      h("span", { class: "formats" }, t("kn.drop_formats")),
      h("span", { class: "formats" }, t("kn.drop_media")),
      input);
    const send = (files) => { if (files.length) model.addFiles(files, metadata.value()); };
    input.addEventListener("change", () => {
      send([...input.files]);
      input.value = "";
    });
    let depth = 0;
    zone.addEventListener("dragenter", (event) => { event.preventDefault(); depth += 1; zone.classList.add("over"); });
    zone.addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
    zone.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove("over"); });
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      depth = 0;
      zone.classList.remove("over");
      send([...(event.dataTransfer?.files ?? [])]);
    });
    return h("section", { class: "stack tight", "aria-labelledby": "kn-up-title" },
      h("div", { class: "section-title" }, h("h2", { id: "kn-up-title" }, t("kn.upload_title"))),
      zone,
      h("details", { class: "surface" }, h("summary", {}, t("kn.upload_metadata")),
        h("p", { class: "hint" }, t("kn.upload_metadata_hint")), metadata.element));
  }

  function uploadRow(item) {
    const states = { waiting: "waiting", uploading: "uploading", rejected: "failed", failed: "failed" };
    const why = item.state === "rejected" ? t(`kn.reject.${item.reason}`) : item.error ? knowledgeError(item.error) : null;
    return h("li", { class: "kn-file" },
      h("span", { class: "type", "aria-hidden": "true" }, extensionOf(item.name) || "?"),
      h("span", { class: "name", title: item.name }, item.name),
      h("span", { class: `badge is-${states[item.state]}` }, t(`kn.upload.${item.state}`)),
      h("span", { class: "meta" }, formatBytes(item.size, lang)),
      item.state === "uploading" ? h("progress", { "aria-label": t("kn.upload.uploading_named", { name: item.name }) }) : null,
      why ? h("p", { class: "error" }, why) : null,
      item.state === "rejected" || item.state === "failed"
        ? h("div", { class: "actions" },
          item.state === "failed" ? h("button", { type: "button", class: "small", "data-focus-key": `up:${item.key}:retry`, onClick: () => model.retryUpload(item.key) }, t("action.retry")) : null,
          h("button", { type: "button", class: "small ghost", "data-focus-key": `up:${item.key}:dismiss`, onClick: () => model.dismissUpload(item.key) }, t("kn.dismiss")))
        : null);
  }

  function documentRow(doc) {
    const media = isMedia(doc.name);
    const facts = [
      formatBytes(doc.size, lang),
      doc.pages ? t("kn.pages", { n: doc.pages }) : null,
      doc.status === "indexed" ? t("kn.chunks", { n: doc.chunks ?? 0 }) : null,
      dateText(doc.uploaded_at),
    ].filter(Boolean).join(" · ");
    return h("li", { class: "kn-file" },
      h("span", { class: "type", "aria-hidden": "true" }, doc.type || extensionOf(doc.name) || "?"),
      h("span", { class: "name", title: doc.name }, doc.name),
      statusBadge(doc.status, media),
      h("span", { class: "meta" }, facts),
      doc.status === "processing" ? h("progress", { "aria-label": t(media ? "kn.status.transcribing" : "kn.status.processing") }) : null,
      media && isPendingStatus(doc.status) ? h("p", { class: "hint" }, t("kn.media_wait")) : null,
      doc.status === "failed" ? h("p", { class: "error" }, doc.error || t("kn.failed_no_reason")) : null,
      metadataChips(doc.metadata),
      h("div", { class: "actions" },
        h("button", { type: "button", class: "small", "data-focus-key": `doc:${doc.id}:edit`, onClick: () => editDocument(doc), "aria-label": t("kn.edit_named", { name: doc.name }) }, icon("edit"), t("kn.edit")),
        h("button", { type: "button", class: "small danger", "data-focus-key": `doc:${doc.id}:delete`, onClick: () => deleteDocument(doc), "aria-label": t("kn.delete_named", { name: doc.name }) }, icon("trash"), t("action.delete"))));
  }

  function searchBox() {
    const text = h("input", { id: "kn-q", type: "search", maxLength: 4000, autocomplete: "off", placeholder: t("kn.search_placeholder") });
    const topK = h("select", { id: "kn-k" }, [3, 5, 10, 20].map((n) => h("option", { value: n }, String(n))));
    topK.value = "5";
    const mode = h("select", { id: "kn-mode" }, MODES.map((m) => h("option", { value: m }, t(`kn.mode.${m}`))));
    const go = h("button", { type: "submit", class: "primary" }, icon("search"), t("kn.search"));
    const results = h("div", { class: "stack tight", "aria-live": "polite" });
    const form = h("form", { onSubmit: (event) => {
      event.preventDefault();
      model.search({ text: text.value, topK: Number(topK.value), mode: mode.value });
    } },
    h("div", { class: "field" }, h("label", { for: text.id }, t("kn.search_label")), text),
    h("div", { class: "field" }, h("label", { for: topK.id }, t("kn.top_k")), topK),
    h("div", { class: "field" }, h("label", { for: mode.id }, t("kn.mode")), mode),
    go);
    return { element: h("section", { class: "kn-search stack tight", "aria-labelledby": "kn-search-title" },
      h("div", { class: "section-title" }, h("h2", { id: "kn-search-title" }, t("kn.search_title"))),
      h("p", { class: "hint" }, t("kn.search_hint")), form, results), results, go };
  }

  function drawResults() {
    const { running, result, error } = model.state.search;
    parts.go.disabled = running;
    if (running) return fill(parts.results, h("span", { class: "skeleton box" }), h("span", { class: "skeleton box" }));
    if (error) return fill(parts.results, h("p", { class: "notice error" }, knowledgeError(error)));
    if (!result) return fill(parts.results);
    const chunks = result.chunks ?? [];
    if (!chunks.length) return fill(parts.results, emptyState({ iconName: "search", title: t("kn.no_results"), body: t("kn.no_results_body") }));
    const total = result.timings?.total_ms;
    fill(parts.results,
      h("p", { class: "hint" }, t("kn.results", { n: chunks.length }), total !== undefined ? ` · ${t("kn.took", { ms: Math.round(total) })}` : ""),
      h("ol", { class: "kn-results" }, chunks.map((chunk, i) => h("li", { class: "kn-result" },
        h("div", { class: "head" },
          h("span", { class: "score" }, `#${i + 1} · ${decimal(chunk.score ?? 0, 3)}`),
          h("span", {}, chunk.metadata?.title || chunk.source?.file_path || chunk.id),
          timeRange(chunk.metadata) ? h("span", { class: "tag" }, t("kn.at_time", { time: timeRange(chunk.metadata) })) : null),
        h("p", {}, chunk.text),
        metadataChips(chunk.metadata)))));
  }

  function drawDocuments() {
    const { documents, documentsError, domain } = model.state;
    const uploads = model.state.uploads.filter((u) => u.domain === domain);
    const counts = countByStatus(documents);
    const pending = counts.queued + counts.processing;
    fill(parts.docHint, pending ? t("kn.pending", { n: pending }) : "");
    keepFocus(root, () => {
      fill(parts.docs,
        documentsError ? h("p", { class: "notice error" }, knowledgeError(documentsError)) : null,
        uploads.length || documents.length
          ? h("ul", { class: "kn-files", "aria-label": t("kn.documents") }, uploads.map(uploadRow), documents.map(documentRow))
          : emptyState({ iconName: "file", title: t("kn.no_docs"), body: t("kn.no_docs_body") }));
    });
  }

  function drawStats(domain) {
    fill(parts.stats,
      h("div", {}, h("dt", {}, t("kn.stat_docs")), h("dd", {}, String(domain.documents ?? model.state.documents.length))),
      h("div", {}, h("dt", {}, t("kn.stat_chunks")), h("dd", {}, String(domain.chunks ?? 0))),
      h("div", {}, h("dt", {}, t("kn.stat_language")), h("dd", {}, String(domain.language ?? "").toUpperCase())),
      h("div", {}, h("dt", {}, t("kn.stat_updated")), h("dd", { class: "hint" }, dateText(domain.updated_at) || "—")));
  }

  function buildMain(domain) {
    const search = searchBox();
    parts = {
      stats: h("dl", { class: "kn-stats" }),
      docs: h("div", {}),
      docHint: h("p", { class: "hint", role: "status" }),
      results: search.results,
      go: search.go,
    };
    return h("div", { class: "kn-main" },
      h("div", { class: "section-title" },
        h("div", {}, h("h2", {}, domain.title || domain.id), h("p", { class: "meta" }, domain.id, domain.index_version ? ` · ${t("kn.version", { v: String(domain.index_version).slice(0, 8) })}` : "")),
        h("button", { type: "button", class: "small danger", onClick: () => deleteDomain(domain) }, icon("trash"), t("kn.delete_domain"))),
      parts.stats,
      dropzone(),
      h("section", { class: "stack tight", "aria-labelledby": "kn-docs-title" },
        h("div", { class: "section-title" }, h("h2", { id: "kn-docs-title" }, t("kn.documents")), parts.docHint),
        parts.docs),
      search.element);
  }

  let layout = null;
  let side = null;

  function render() {
    const { status, error, domains, domain } = model.state;
    if (status === "loading") return;
    if (status === "missing") {
      shownDomain = undefined;
      newDomain.hidden = true;
      return fill(body, emptyState({
        iconName: "knowledge", title: t("kn.missing_title"), body: t("kn.missing_body"),
        action: h("button", { type: "button", onClick: () => model.load(wanted) }, t("action.retry")),
      }));
    }
    newDomain.hidden = false;
    if (status === "error") {
      shownDomain = undefined;
      if (error?.kind === "unauthorized") return ctx.fail(error, () => model.load(wanted));
      return fill(body, h("div", { class: "notice error", role: "alert" }, knowledgeError(error)),
        h("button", { type: "button", onClick: () => model.load(wanted) }, t("action.retry")));
    }
    if (!domains.length) {
      shownDomain = undefined;
      return fill(body, emptyState({
        iconName: "knowledge", title: t("kn.no_domains"), body: t("kn.no_domains_body"),
        action: h("button", { type: "button", class: "primary", onClick: () => createDomain() }, icon("plus"), t("kn.new_domain")),
      }));
    }
    const current = model.selected();
    if (shownDomain !== domain || !layout || !body.contains(layout)) {
      shownDomain = domain;
      side = h("div", {});
      layout = h("div", { class: "kn-layout" }, side, current ? buildMain(current) : h("div", {}));
      fill(body, layout);
    }
    keepFocus(root, () => fill(side, domainList()));
    if (current && parts) {
      drawStats(current);
      drawDocuments();
      drawResults();
    }
  }

  await model.load(wanted);
  return () => {
    model.stop();
    root.classList.remove("wide");
  };
}
