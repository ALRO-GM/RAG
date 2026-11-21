// The «Conocimiento» page's logic without the DOM: which files are
// accepted, the shape of what is sent (metadata, knowledge.query.v1) and
// KnowledgeModel, the state of the page (domains, documents, uploads,
// polling, test search) driven by an injected API and timer so node
// --test can run it against a fake backend.
//
// API (proxied by the transcriber under /api/knowledge/…):
//   GET/POST /knowledge/domains, DELETE /knowledge/domains/{id},
//   GET/POST /knowledge/domains/{id}/documents (multipart: file + metadata JSON),
//   PATCH/DELETE /knowledge/domains/{id}/documents/{doc},
//   POST /knowledge/query (knowledge.query.v1 -> knowledge.result.v1).

/** File extensions a company document may have. */
export const EXTENSIONS = ["pdf", "docx", "odt", "txt", "md", "markdown", "html", "htm", "csv", "xlsx", "png", "jpg", "jpeg", "tif", "tiff", "webp"];

/** The file picker's accept attribute (extensions and their media types). */
export const ACCEPT = [
  ...EXTENSIONS.map((e) => `.${e}`),
  "application/pdf", "text/plain", "text/markdown", "text/html", "text/csv",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.oasis.opendocument.text",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "image/png", "image/jpeg", "image/tiff", "image/webp",
].join(",");

/** Largest file sent (the server keeps its own, possibly smaller, limit). */
export const MAX_BYTES = 100 * 1024 * 1024;

/** The dialogue-act categories a chunk may be filed under (CONTRACTS.md §2). */
export const CATEGORIES = [
  "pregunta", "respuesta", "informacion", "opinion", "propuesta", "peticion", "aceptacion",
  "rechazo", "compromiso", "escucha", "gestion", "relacion", "otro",
];

/** Document statuses the server reports. */
export const STATUSES = ["queued", "processing", "indexed", "failed"];

/** How often the documents are asked for while one is queued or processing. */
export const POLL_MS = 2000;

/** Search modes of knowledge.query.v1. */
export const MODES = ["hybrid", "lexical", "semantic"];

const DOMAIN_ID = /^[a-z0-9][a-z0-9._-]{0,63}$/;

/** "Informe.Final.PDF" -> "pdf"; "" when there is none. */
export function extensionOf(name) {
  const m = /\.([A-Za-z0-9]+)$/.exec(String(name ?? ""));
  return m ? m[1].toLowerCase() : "";
}

/** Why `file` cannot be sent ("type", "empty", "size") or null when it can. */
export function checkFile(file, maxBytes = MAX_BYTES) {
  if (!EXTENSIONS.includes(extensionOf(file?.name))) return "type";
  if (!(file.size > 0)) return "empty";
  if (file.size > maxBytes) return "size";
  return null;
}

/** A byte count for people: "820 B", "1,4 MB" / "1.4 MB". */
export function formatBytes(bytes, lang = "es") {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || value >= 10 ? 0 : 1;
  return `${new Intl.NumberFormat(lang, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)} ${units[unit]}`;
}

/** Whether the server is still working on `doc`. */
export function isPending(doc) {
  return doc?.status === "queued" || doc?.status === "processing";
}

/** Whether the documents must be asked for again in POLL_MS. */
export function needsPolling(documents) {
  return (documents ?? []).some(isPending);
}

/** A domain id from a title: "Entrevistas Backend" -> "entrevistas-backend". */
export function slugify(title) {
  const slug = String(title ?? "").normalize("NFD").replace(/\p{M}/gu, "").toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-").replace(/^[^a-z0-9]+/, "").replace(/-+$/, "").slice(0, 64).replace(/-+$/, "");
  return slug;
}

/** Whether `id` is a valid domain id (the pattern of knowledge.query.v1). */
export function isDomainId(id) {
  return DOMAIN_ID.test(String(id ?? ""));
}

/** "a, b ,, a" -> ["a", "b"]: trimmed, no empties, no repeats, at most 30. */
export function parseTags(text) {
  const out = [];
  for (const raw of String(text ?? "").split(/[,;\n]/)) {
    const tag = raw.trim().slice(0, 64);
    if (tag && !out.includes(tag)) out.push(tag);
  }
  return out.slice(0, 30);
}

/** The metadata object sent for a document; blank fields become null. */
export function cleanMetadata({ category = "", intent = "", script_phase: phase = "", tags = [] } = {}) {
  const blank = (v) => (String(v ?? "").trim() ? String(v).trim() : null);
  return {
    category: CATEGORIES.includes(category) ? category : null,
    intent: blank(intent),
    script_phase: blank(phase),
    tags: Array.isArray(tags) ? parseTags(tags.join(",")) : parseTags(tags),
  };
}

/** A request id for knowledge.query.v1 (1-64 characters). */
export function requestId(now = Date.now(), random = Math.random()) {
  return `web-${now.toString(36)}-${Math.floor(random * 1e8).toString(36)}`.slice(0, 64);
}

/** The body of POST /knowledge/query (knowledge.query.v1). */
export function queryBody({ domain, text, topK = 5, mode = "hybrid", language = null, id = requestId() }) {
  const body = {
    schema: "knowledge.query.v1",
    request_id: id,
    domain,
    text: String(text ?? "").trim().slice(0, 4000),
    top_k: Number.isFinite(Number(topK)) ? Math.min(Math.max(Math.round(Number(topK)), 1), 50) : 5,
    mode: MODES.includes(mode) ? mode : "hybrid",
  };
  if (language) body.language = language;
  return body;
}

/** A list the server sent bare or wrapped ({ domains: [...] }, { documents: [...] }). */
export function listOf(answer, key) {
  if (Array.isArray(answer)) return answer;
  return Array.isArray(answer?.[key]) ? answer[key] : [];
}

/** Newest upload first, then by name. */
export function sortDocuments(documents) {
  return [...(documents ?? [])].sort((a, b) =>
    String(b.uploaded_at ?? "").localeCompare(String(a.uploaded_at ?? "")) || String(a.name).localeCompare(String(b.name)));
}

/** How many documents are in each status. */
export function countByStatus(documents) {
  const counts = Object.fromEntries(STATUSES.map((s) => [s, 0]));
  for (const doc of documents ?? []) if (doc.status in counts) counts[doc.status] += 1;
  return counts;
}

/** A failed call that means the knowledge service is not installed yet (404 on the list). */
export function serviceMissing(error) {
  return error?.status === 404;
}

/**
 * The page's state and actions. `api` has the knowledge methods of
 * js/api.js; `timer` is { set(fn, ms), clear(handle) }; `onChange` runs
 * after every change of `state`.
 */
export class KnowledgeModel {
  constructor(api, { timer = { set: (fn, ms) => setTimeout(fn, ms), clear: (h) => clearTimeout(h) }, onChange = () => {} } = {}) {
    this.api = api;
    this.timer = timer;
    this.onChange = onChange;
    this.handle = null;
    this.stopped = false;
    this.uploadKey = 0;
    this.uploading = false;
    this.state = {
      status: "loading", // loading | ready | missing | error
      error: null,
      domains: [],
      domain: null, // the selected domain's id
      documents: [],
      documentsError: null,
      uploads: [], // { key, name, size, state: waiting|uploading|rejected|failed, reason, error }
      search: { running: false, result: null, error: null },
    };
  }

  changed() {
    if (!this.stopped) this.onChange(this.state);
  }

  /** The selected domain's summary, or null. */
  selected() {
    return this.state.domains.find((d) => d.id === this.state.domain) ?? null;
  }

  /** Load the domains and select `wanted` (or the first). */
  async load(wanted = null) {
    try {
      this.state.domains = listOf(await this.api.knowledgeDomains(), "domains");
      this.state.status = "ready";
      this.state.error = null;
    } catch (error) {
      this.state.status = serviceMissing(error) ? "missing" : "error";
      this.state.error = error;
      this.changed();
      return;
    }
    const ids = this.state.domains.map((d) => d.id);
    await this.select(ids.includes(wanted) ? wanted : ids[0] ?? null);
  }

  /** Show domain `id`: its documents, and polling while some are pending. */
  async select(id) {
    this.timer.clear(this.handle);
    this.handle = null;
    this.state.domain = id;
    this.state.documents = [];
    this.state.documentsError = null;
    this.state.search = { running: false, result: null, error: null };
    this.changed();
    if (id) await this.refreshDocuments();
  }

  /** Ask for the selected domain's documents again; plan the next poll. */
  async refreshDocuments() {
    const id = this.state.domain;
    if (!id) return;
    try {
      const documents = listOf(await this.api.knowledgeDocuments(id), "documents");
      if (id !== this.state.domain) return;
      this.state.documents = sortDocuments(documents);
      this.state.documentsError = null;
    } catch (error) {
      if (id !== this.state.domain) return;
      this.state.documentsError = error;
    }
    this.changed();
    this.plan();
  }

  /** Poll every POLL_MS while a document is queued or processing. */
  plan() {
    if (this.stopped || this.handle !== null || !needsPolling(this.state.documents)) return;
    this.handle = this.timer.set(async () => {
      this.handle = null;
      await this.refreshDocuments();
      await this.refreshDomains();
    }, POLL_MS);
  }

  /** The domains' counts again (documents, chunks), keeping the selection. */
  async refreshDomains() {
    try {
      this.state.domains = listOf(await this.api.knowledgeDomains(), "domains");
      this.changed();
    } catch {
      // the counts stay as they were; the next poll tries again
    }
  }

  /** Create a domain and select it; throws what the API answered. */
  async createDomain({ id, title, language }) {
    await this.api.createKnowledgeDomain({ id, title: String(title).trim(), language });
    this.state.domains = listOf(await this.api.knowledgeDomains(), "domains");
    this.state.status = "ready";
    await this.select(id);
  }

  /** Delete the domain `id` with its documents; throws what the API answered. */
  async deleteDomain(id) {
    await this.api.deleteKnowledgeDomain(id);
    this.state.domains = this.state.domains.filter((d) => d.id !== id);
    if (this.state.domain === id) await this.select(this.state.domains[0]?.id ?? null);
    else this.changed();
  }

  /**
   * Queue `files` for the selected domain with `metadata` (same for all).
   * Files that cannot be sent are listed as rejected with their reason.
   * Returns once every queued file was tried (one at a time).
   */
  async addFiles(files, metadata = {}) {
    const domain = this.state.domain;
    if (!domain) return;
    for (const file of files) {
      const reason = checkFile(file);
      this.uploadKey += 1;
      this.state.uploads.push({
        key: this.uploadKey, file, domain, metadata: cleanMetadata(metadata), name: file.name, size: file.size,
        state: reason ? "rejected" : "waiting", reason, error: null,
      });
    }
    this.changed();
    await this.drain();
  }

  async drain() {
    if (this.uploading) return;
    this.uploading = true;
    try {
      for (let next = this.nextWaiting(); next; next = this.nextWaiting()) {
        next.state = "uploading";
        this.changed();
        try {
          const doc = await this.api.uploadKnowledgeDocument(next.domain, next.file, next.metadata);
          this.state.uploads = this.state.uploads.filter((u) => u !== next);
          if (next.domain === this.state.domain && doc && doc.id && !this.state.documents.some((d) => d.id === doc.id)) {
            this.state.documents = sortDocuments([...this.state.documents, doc]);
          }
        } catch (error) {
          next.state = "failed";
          next.error = error;
        }
        this.changed();
      }
    } finally {
      this.uploading = false;
    }
    await this.refreshDocuments();
  }

  nextWaiting() {
    return this.state.uploads.find((u) => u.state === "waiting") ?? null;
  }

  /** Try a failed upload again. */
  async retryUpload(key) {
    const item = this.state.uploads.find((u) => u.key === key && u.state === "failed");
    if (!item) return;
    item.state = "waiting";
    item.error = null;
    this.changed();
    await this.drain();
  }

  /** Forget a rejected or failed upload. */
  dismissUpload(key) {
    this.state.uploads = this.state.uploads.filter((u) => u.key !== key || u.state === "uploading");
    this.changed();
  }

  /** Change a document's metadata; throws what the API answered. */
  async updateDocument(docId, metadata) {
    const clean = cleanMetadata(metadata);
    const updated = await this.api.updateKnowledgeDocument(this.state.domain, docId, clean);
    this.state.documents = this.state.documents.map((d) => (d.id === docId
      ? (updated && updated.id ? updated : { ...d, metadata: clean })
      : d));
    this.changed();
    this.plan();
  }

  /** Delete a document; throws what the API answered. */
  async deleteDocument(docId) {
    await this.api.deleteKnowledgeDocument(this.state.domain, docId);
    this.state.documents = this.state.documents.filter((d) => d.id !== docId);
    this.changed();
    await this.refreshDomains();
  }

  /** Run a test search in the selected domain (knowledge.query.v1). */
  async search({ text, topK = 5, mode = "hybrid" }) {
    const domain = this.selected();
    if (!domain || !String(text ?? "").trim()) return;
    this.state.search = { running: true, result: null, error: null };
    this.changed();
    try {
      const result = await this.api.knowledgeQuery(queryBody({ domain: domain.id, text, topK, mode, language: domain.language ?? null }));
      this.state.search = { running: false, result, error: null };
    } catch (error) {
      this.state.search = { running: false, result: null, error };
    }
    this.changed();
  }

  /** Stop polling and stop reporting changes (the page was left). */
  stop() {
    this.stopped = true;
    this.timer.clear(this.handle);
    this.handle = null;
  }
}
