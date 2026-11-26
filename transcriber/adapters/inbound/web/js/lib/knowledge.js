// The «Conocimiento» page's logic without the DOM: which files are
// accepted, the shape of what is sent (metadata, knowledge.query.v1) and
// KnowledgeModel, the state of the page (domains, documents, uploads,
// polling, test search) driven by an injected API and timer so node
// --test can run it against a fake backend.
//
// API (proxied by the transcriber under /api/knowledge/…):
//   GET /knowledge/limits (the largest file of each kind, the piece size),
//   GET/POST /knowledge/domains, DELETE /knowledge/domains/{id},
//   GET/POST /knowledge/domains/{id}/documents (multipart: file + metadata JSON),
//   POST /knowledge/domains/{id}/uploads, GET/DELETE …/uploads/{up},
//   PUT …/uploads/{up}/pieces/{offset}, POST …/uploads/{up}/complete (#282),
//   PATCH/DELETE /knowledge/domains/{id}/documents/{doc},
//   POST /knowledge/query (knowledge.query.v1 -> knowledge.result.v1).
//
// A file up to one piece (limits.piece_bytes) goes in one request; a larger
// one in pieces (PieceUpload): progress, a calm retry after a cut that goes
// on from what the server already has, and cancel. The limits are the
// server's; the page keeps no copy of the numbers.

/** Audio and video formats: what is said in them is transcribed (KN-3a);
 * the transcriber's recording formats. */
export const MEDIA_EXTENSIONS = ["wav", "flac", "ogg", "opus", "mp3", "m4a", "mp4", "aac", "webm", "mov", "3gp", "amr", "wma"];

/** File extensions a company document may have. */
export const EXTENSIONS = [
  "pdf", "docx", "odt", "txt", "md", "markdown", "html", "htm", "csv", "xlsx", "png", "jpg", "jpeg", "tif", "tiff", "webp",
  ...MEDIA_EXTENSIONS,
];

/** The file picker's accept attribute (extensions and their media types). */
export const ACCEPT = [
  ...EXTENSIONS.map((e) => `.${e}`),
  "application/pdf", "text/plain", "text/markdown", "text/html", "text/csv",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.oasis.opendocument.text",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "image/png", "image/jpeg", "image/tiff", "image/webp", "audio/*", "video/*",
].join(",");

/** Media types that may hold video (the server gives them the video limit
 * when the content is a video container); the other media are audio. */
export const VIDEO_EXTENSIONS = ["mp4", "webm", "mov", "3gp"];

/** Waits before trying a piece again after a cut (then the last one, again). */
export const RETRY_MS = [1000, 2000, 5000, 10000, 20000, 30000];

/** Tries of one piece before the upload is shown as failed (≈ 15 min of cuts). */
export const MAX_TRIES = 40;

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

/** Whether a file or document name is audio or video (transcribed, not read). */
export function isMedia(name) {
  return MEDIA_EXTENSIONS.includes(extensionOf(name));
}

/** Which server limit a file name falls under: "pdf", "video", "audio" or "document". */
export function limitKind(name) {
  const ext = extensionOf(name);
  if (ext === "pdf") return "pdf";
  if (VIDEO_EXTENSIONS.includes(ext)) return "video";
  if (MEDIA_EXTENSIONS.includes(ext)) return "audio";
  return "document";
}

/** Why `file` cannot be sent ("type", "empty", "size") or null when it can.
 * `limits` are the server's ({ pdf, video, audio, document } bytes); without
 * them only the server checks the size. */
export function checkFile(file, limits = null) {
  if (!EXTENSIONS.includes(extensionOf(file?.name))) return "type";
  if (!(file.size > 0)) return "empty";
  const limit = Number(limits?.[limitKind(file.name)]);
  if (Number.isFinite(limit) && limit > 0 && file.size > limit) return "size";
  return null;
}

/** A limit for people: whole units when round ("500 MB", "1 GB"). */
export function limitText(bytes, lang = "es") {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${new Intl.NumberFormat(lang, { maximumFractionDigits: 1 }).format(value)} ${units[unit]}`;
}

/** Whole percent of `sent` out of `total` (0 for an empty total). */
export function percentOf(sent, total) {
  return total > 0 ? Math.min(100, Math.floor((sent / total) * 100)) : 0;
}

/** Seconds as the server writes them in a transcript: "03:12", "1:02:05". */
export function clockText(seconds) {
  const whole = Math.max(Math.floor(Number(seconds) || 0), 0);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  const two = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${two(m)}:${two(s)}` : `${two(m)}:${two(s)}`;
}

/** The part of a recording a retrieved chunk covers ("03:12–03:40"), or null
 * for a chunk of a document (knowledge.result.v1 metadata start_s/end_s). */
export function timeRange(metadata) {
  const start = metadata?.start_s;
  if (typeof start !== "number") return null;
  const end = metadata?.end_s;
  const from = clockText(start);
  const to = typeof end === "number" ? clockText(end) : from;
  return to === from ? from : `${from}–${to}`;
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

/** What a cancelled upload rejects with (`kind` "aborted", as the API's). */
export class UploadCancelled extends Error {
  constructor() {
    super("cancelled");
    this.kind = "aborted";
    this.code = "aborted";
    this.status = null;
  }
}

/** A failure worth trying again after a wait (a cut, a restart): not a refusal. */
function retryable(error) {
  if (error?.status === 507) return false; // no room: waiting will not help
  return error?.kind === "unavailable" || error?.kind === "server" || error?.kind === "rate";
}

/**
 * One file sent in pieces of `pieceBytes`, resumable. `onProgress` gets
 * { sent, total, retrying, finishing } after every change. A cut (network,
 * 5xx) waits RETRY_MS, asks the server what it holds and goes on from there;
 * a 409 (the server holds something else) realigns the same way; any other
 * refusal ends it with that error. `cancel()` stops it and deletes what the
 * server received. `uploadId` (a previous try's) resumes that upload.
 */
export class PieceUpload {
  constructor(api, { domain, file, metadata, pieceBytes, timer, onProgress = () => {}, uploadId = null }) {
    this.api = api;
    this.domain = domain;
    this.file = file;
    this.metadata = metadata;
    this.pieceBytes = Math.max(1, Number(pieceBytes) || 1);
    this.timer = timer;
    this.onProgress = onProgress;
    this.uploadId = uploadId;
    this.sent = 0;
    this.retrying = false;
    this.finishing = false;
    this.cancelled = false;
    this.controller = null;
    this.wake = null;
  }

  report() {
    this.onProgress({ sent: this.sent, total: this.file.size, retrying: this.retrying, finishing: this.finishing });
  }

  /** Wait `ms` on the injected timer; cancel() ends the wait at once. */
  pause(ms) {
    return new Promise((resolve) => {
      const handle = this.timer.set(() => { this.wake = null; resolve(); }, ms);
      this.wake = () => { this.timer.clear(handle); this.wake = null; resolve(); };
    });
  }

  checkCancelled() {
    if (this.cancelled) throw new UploadCancelled();
  }

  /**
   * Run `call` until it succeeds; a retryable failure waits, then `onCut`
   * (when given) realigns before trying again; with `onCut` a 409 (the
   * server holds something else) realigns at once, three times at most in
   * a row. Other failures are thrown.
   */
  async withRetries(call, onCut = null) {
    let conflicts = 0;
    for (let attempt = 0; ; attempt += 1) {
      this.checkCancelled();
      try {
        const result = await call();
        if (this.retrying) {
          this.retrying = false;
          this.report();
        }
        return result;
      } catch (error) {
        this.checkCancelled();
        const realign = onCut !== null && error?.status === 409 && conflicts < 3;
        if (!(retryable(error) || realign) || attempt + 1 >= MAX_TRIES) throw error;
        if (realign) {
          conflicts += 1;
        } else {
          conflicts = 0;
          this.retrying = true;
          this.report();
          await this.pause(RETRY_MS[Math.min(attempt, RETRY_MS.length - 1)]);
          this.checkCancelled();
        }
        if (onCut !== null) {
          try {
            await onCut();
          } catch (again) {
            this.checkCancelled();
            if (!retryable(again)) throw again;
          }
        }
      }
    }
  }

  /** Ask the server how much it holds and go on from there. */
  async realign() {
    const status = await this.api.knowledgeUploadStatus(this.domain, this.uploadId);
    this.sent = Math.min(Number(status?.received) || 0, this.file.size);
    this.report();
  }

  /** Send the piece that starts at what the server holds. */
  async sendPiece() {
    const start = this.sent;
    const piece = this.file.slice(start, Math.min(start + this.pieceBytes, this.file.size));
    this.controller = typeof AbortController === "function" ? new AbortController() : null;
    try {
      return await this.api.putKnowledgePiece(this.domain, this.uploadId, start, piece, { signal: this.controller?.signal });
    } finally {
      this.controller = null;
    }
  }

  /** Send the whole file; resolves with the document, rejects with the reason. */
  async run() {
    if (this.uploadId) {
      await this.withRetries(() => this.realign());
    } else {
      const upload = await this.withRetries(() => this.api.startKnowledgeUpload(this.domain, {
        name: this.file.name, size: this.file.size, metadata: this.metadata,
      }));
      this.uploadId = upload.id;
    }
    this.report();
    while (this.sent < this.file.size) {
      const view = await this.withRetries(() => this.sendPiece(), () => this.realign());
      if (view && Number.isFinite(Number(view.received))) this.sent = Math.min(Number(view.received), this.file.size);
      this.report();
    }
    this.finishing = true;
    this.report();
    return this.withRetries(() => this.api.completeKnowledgeUpload(this.domain, this.uploadId));
  }

  /** Stop and delete what the server holds of it. */
  async cancel() {
    this.cancelled = true;
    this.controller?.abort();
    this.wake?.();
    if (!this.uploadId) return;
    try {
      await this.api.cancelKnowledgeUpload(this.domain, this.uploadId);
    } catch {
      // the server sweeps what nobody finishes
    }
  }
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
      uploads: [], // { key, name, size, state: waiting|uploading|rejected|failed, reason, error, sent, retrying, finishing }
      limits: null, // the server's { pdf, video, audio, document, piece_bytes, max_media_seconds }, once read
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

  /** Read the server's limits (kept null when they cannot be read: the server still checks). */
  async loadLimits() {
    if (typeof this.api.knowledgeLimits !== "function") return;
    try {
      this.state.limits = await this.api.knowledgeLimits();
    } catch {
      this.state.limits = null;
    }
  }

  /** Load the domains and select `wanted` (or the first). */
  async load(wanted = null) {
    await this.loadLimits();
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
      const reason = checkFile(file, this.state.limits);
      const kind = limitKind(file.name);
      this.uploadKey += 1;
      this.state.uploads.push({
        key: this.uploadKey, file, domain, metadata: cleanMetadata(metadata), name: file.name, size: file.size,
        state: reason ? "rejected" : "waiting", reason, error: null, kind, limit: this.state.limits?.[kind] ?? null,
        sent: 0, retrying: false, finishing: false, uploadId: null, piece: null,
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
        next.retrying = false;
        this.changed();
        try {
          const doc = await this.send(next);
          this.state.uploads = this.state.uploads.filter((u) => u !== next);
          if (next.domain === this.state.domain && doc && doc.id && !this.state.documents.some((d) => d.id === doc.id)) {
            this.state.documents = sortDocuments([...this.state.documents, doc]);
          }
        } catch (error) {
          if (error?.kind === "aborted") {
            this.state.uploads = this.state.uploads.filter((u) => u !== next);
          } else {
            next.state = "failed";
            next.error = error;
          }
        } finally {
          next.piece = null;
          next.retrying = false;
          next.finishing = false;
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

  /**
   * Send one queued file: in one request up to one piece (or when the
   * server's limits are unknown), else in pieces with progress and resume.
   */
  async send(item) {
    const pieceBytes = Number(this.state.limits?.piece_bytes);
    if (!(pieceBytes > 0) || item.size <= pieceBytes || typeof this.api.startKnowledgeUpload !== "function") {
      return this.api.uploadKnowledgeDocument(item.domain, item.file, item.metadata);
    }
    item.piece = new PieceUpload(this.api, {
      domain: item.domain, file: item.file, metadata: item.metadata, pieceBytes, timer: this.timer, uploadId: item.uploadId,
      onProgress: ({ sent, retrying, finishing }) => {
        Object.assign(item, { sent, retrying, finishing, uploadId: item.piece?.uploadId ?? item.uploadId });
        this.changed();
      },
    });
    try {
      return await item.piece.run();
    } finally {
      item.uploadId = item.piece?.uploadId ?? item.uploadId;
    }
  }

  /** Cancel a waiting or uploading file (what the server received is deleted). */
  async cancelUpload(key) {
    const item = this.state.uploads.find((u) => u.key === key);
    if (!item) return;
    if (item.piece) {
      await item.piece.cancel();
      return; // drain() drops it once the cancelled upload rejects
    }
    if (item.state === "uploading") return; // one request: it is nearly done
    this.state.uploads = this.state.uploads.filter((u) => u !== item);
    if (item.uploadId) {
      try {
        await this.api.cancelKnowledgeUpload(item.domain, item.uploadId);
      } catch {
        // the server sweeps what nobody finishes
      }
    }
    this.changed();
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

  /** Forget a rejected or failed upload (what the server holds of it is deleted). */
  dismissUpload(key) {
    const item = this.state.uploads.find((u) => u.key === key && u.state !== "uploading");
    this.state.uploads = this.state.uploads.filter((u) => u.key !== key || u.state === "uploading");
    if (item?.uploadId && typeof this.api.cancelKnowledgeUpload === "function") {
      this.api.cancelKnowledgeUpload(item.domain, item.uploadId).catch(() => {});
    }
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
