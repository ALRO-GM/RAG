// Client of the transcriber API. Same origin, everything under /api, the
// access code in the X-Access-Code header on every call but /health.
// The code is never put in a URL.

import { classifyFailure } from "./lib/net.js";
import { startBody } from "./lib/session-analysis.js";
import { parseSse } from "./lib/live.js";

export class ApiFailure extends Error {
  constructor({ status, kind, code = null, message = "", waitSeconds = 0, reason = null }) {
    super(message || code || kind);
    this.name = "ApiFailure";
    this.status = status; // null when the network failed
    this.kind = kind; // see classifyFailure
    this.code = code; // the API's stable error code
    this.waitSeconds = waitSeconds;
    this.reason = reason; // sample_rejected: too_short | mostly_silent | several_voices
  }
}

export class Api {
  /**
   * @param {() => (string|null)} getCode  where the access code is read from
   * @param {typeof fetch} fetchFn         injectable for tests
   */
  constructor(getCode, fetchFn = (...args) => globalThis.fetch(...args)) {
    this.getCode = getCode;
    this.fetchFn = fetchFn;
  }

  async request(method, path, { query, json, body, contentType, auth = true, text = false, raw = false } = {}) {
    const headers = {};
    if (auth) {
      const code = this.getCode();
      if (!code) throw new ApiFailure({ status: 401, kind: "unauthorized", code: "unauthorized" });
      headers["X-Access-Code"] = code;
    }
    let payload = body;
    if (json !== undefined) {
      payload = JSON.stringify(json);
      headers["Content-Type"] = "application/json";
    } else if (contentType) {
      headers["Content-Type"] = contentType;
    }
    let url = `/api${path}`;
    if (query) {
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
      }
      const text = params.toString();
      if (text) url += `?${text}`;
    }
    let response;
    try {
      response = await this.fetchFn(url, {
        method, headers, body: payload, cache: "no-store", credentials: "same-origin",
      });
    } catch {
      throw new ApiFailure({ status: null, kind: "unavailable", code: "network" });
    }
    if (response.ok) {
      if (raw) return response;
      if (response.status === 204) return null;
      try {
        return text ? await response.text() : await response.json();
      } catch {
        throw new ApiFailure({ status: response.status, kind: "unavailable", code: "bad_body" });
      }
    }
    let info = {};
    try {
      info = await response.json();
    } catch {
      info = {};
    }
    const { kind, waitSeconds } = classifyFailure(response.status, response.headers.get("Retry-After"));
    throw new ApiFailure({
      status: response.status, kind, waitSeconds,
      code: typeof info.code === "string" ? info.code : null,
      message: typeof info.message === "string" ? info.message : "",
      reason: typeof info.reason === "string" ? info.reason : null,
    });
  }

  health() { return this.request("GET", "/health", { auth: false }); }

  /** Trade the password for the access code ({ code }); no code needed. */
  login(password) { return this.request("POST", "/login", { json: { password }, auth: false }); }

  /** Let <audio> play without headers: an HttpOnly cookie for the audio route. */
  audioCookie() { return this.request("POST", "/access/cookie"); }

  listSessions() { return this.request("GET", "/sessions"); }

  getSession(id) { return this.request("GET", `/sessions/${encodeURIComponent(id)}`); }

  /**
   * Open a session. `started` holds what was chosen at the start and is kept
   * on the session: consent_version, num_speakers, expected_speakers, and the
   * conversation's type and goal.
   */
  createSession(id, language, title, started = {}) {
    const json = { id, language };
    if (title) json.title = title;
    for (const [key, value] of Object.entries(started)) {
      if (value !== undefined && value !== null) json[key] = value;
    }
    return this.request("POST", "/sessions", { json });
  }

  putChunk(id, number, bytes) {
    return this.request("PUT", `/sessions/${encodeURIComponent(id)}/chunks/${number}`, {
      body: bytes, contentType: "application/octet-stream",
    });
  }

  closeSession(id) { return this.request("POST", `/sessions/${encodeURIComponent(id)}/close`); }

  /** `described` is the conversation's { type, goal } (both optional). */
  uploadRecording(id, file, language, title, consentVersion, described = {}) {
    return this.request("POST", `/sessions/${encodeURIComponent(id)}/upload`, {
      query: {
        filename: file.name, language, title, consent_version: consentVersion,
        type: described.type, goal: described.goal,
      },
      body: file, contentType: "application/octet-stream",
    });
  }

  /** Set or change the conversation's type and goal: a field left out is kept, null clears it. */
  describeSession(id, changes) {
    return this.request("PATCH", `/sessions/${encodeURIComponent(id)}`, { json: changes });
  }

  queueTranscription(id, body = {}) {
    return this.request("POST", `/sessions/${encodeURIComponent(id)}/transcription`, { json: body });
  }

  getTranscript(id) { return this.request("GET", `/sessions/${encodeURIComponent(id)}/transcript`); }

  renameSpeaker(id, label, change) {
    return this.request("PATCH", `/sessions/${encodeURIComponent(id)}/speakers/${encodeURIComponent(label)}`, { json: change });
  }

  /** Names found in what people said: given automatically and suggested. */
  getNames(id) { return this.request("GET", `/sessions/${encodeURIComponent(id)}/names`); }

  /** Search the finished session for names again (a background job). */
  searchNames(id) { return this.request("POST", `/sessions/${encodeURIComponent(id)}/names`); }

  /** Undo a name given automatically; returns the transcript. */
  undoName(id, label) {
    return this.request("DELETE", `/sessions/${encodeURIComponent(id)}/names/${encodeURIComponent(label)}`);
  }

  reassignUtterance(id, utterance, speaker) {
    return this.request("PATCH", `/sessions/${encodeURIComponent(id)}/utterances/${encodeURIComponent(utterance)}`, { json: { speaker } });
  }

  deleteSession(id) { return this.request("DELETE", `/sessions/${encodeURIComponent(id)}`); }

  speakers() { return this.request("GET", "/speakers"); }

  deleteSpeaker(id) { return this.request("DELETE", `/speakers/${encodeURIComponent(id)}`); }

  /** Withdraw the person's consent to the voice bot cloning their voice (the clip is deleted). */
  withdrawVoiceClone(id) { return this.request("DELETE", `/speakers/${encodeURIComponent(id)}/voice-clone`); }

  /** The consent notice in force: kind "recording" or "enrolment". */
  consent(kind, language) {
    return this.request("GET", "/consent", { query: { kind, language } });
  }

  enrolmentText(language, name) {
    return this.request("GET", "/enrolment/text", { query: { language, name } });
  }

  // ---- pipeline settings («Ajustes», docs/SETTINGS.md) ----

  /** The saved settings (settings.v1), whether saved, and the catalogue of options. */
  pipelineSettings() { return this.request("GET", "/settings"); }

  /** Save `body` ({ preset?, choices?, revision? }); resolves to the saved settings. */
  savePipelineSettings(body) { return this.request("PUT", "/settings", { json: body }); }

  /**
   * The voice bot's sample sentence said with the voice bot's `choices`
   * being edited (unsaved): { blob (audio/wav), clone, lengthScale, note }.
   */
  async voiceSample(choices) {
    const response = await this.request("POST", "/assist/voicebot/sample", { json: { choices }, raw: true });
    const headers = response.headers;
    return {
      blob: await response.blob(),
      clone: headers.get("X-Voice-Clone") === "1",
      lengthScale: Number(headers.get("X-Voice-Length-Scale")) || null,
      note: headers.get("X-Voice-Note") || "",
      engine: headers.get("X-Voice-Engine") || "",
    };
  }

  // ---- analysis (docs/skills/DESIGN.md §4) ----

  /** The RAG's analysis judge and whether Claude's quota is available. */
  analysisJudge() { return this.request("GET", "/analysis/judge"); }

  /** Choose the judge: claude_code, local, both or opencode. */
  setAnalysisJudge(mode) { return this.request("PUT", "/analysis/judge", { json: { mode } }); }

  /**
   * Start the RAG's whole analysis of a published session. `steps` runs
   * only those again; `job` (the job sought) and `claudeSynthesis` go to
   * the explanatory profiles and are sent only when `job` is a string.
   */
  startAnalysis(id, options = {}) {
    return this.request("POST", `/analysis/sessions/${encodeURIComponent(id)}`, { json: startBody(options) });
  }

  /** Stop the session's running analysis. */
  cancelAnalysis(id) { return this.request("DELETE", `/analysis/sessions/${encodeURIComponent(id)}/run`); }

  /** The steps of a session's analysis (status, time, judge, counts). */
  analysisStatus(id) { return this.request("GET", `/analysis/sessions/${encodeURIComponent(id)}`); }

  /** One step's stored result. */
  analysisStep(id, step) {
    return this.request("GET", `/analysis/sessions/${encodeURIComponent(id)}/steps/${encodeURIComponent(step)}`);
  }

  /** One person's report, as the HTML text of a page to save. */
  analysisReport(id, position) {
    return this.request("GET", `/analysis/sessions/${encodeURIComponent(id)}/reports/${Number(position)}`, { text: true });
  }

  /** Hand the transcript over to the RAG (needed before an analysis). */
  publish(id) { return this.request("POST", `/sessions/${encodeURIComponent(id)}/publish`); }

  // ---- knowledge domains (the RAG's /knowledge/..., proxied under /api/knowledge) ----

  /** The knowledge domains (id, title, language, documents, chunks, index_version, updated_at). */
  knowledgeDomains() { return this.request("GET", "/knowledge/domains"); }

  /** Create a domain: { id, title, language }. */
  createKnowledgeDomain(domain) { return this.request("POST", "/knowledge/domains", { json: domain }); }

  /** Delete a domain with its documents and index. */
  deleteKnowledgeDomain(id) { return this.request("DELETE", `/knowledge/domains/${encodeURIComponent(id)}`); }

  /** A domain's documents (name, type, size, pages, chunks, status, error, metadata, uploaded_at). */
  knowledgeDocuments(id) { return this.request("GET", `/knowledge/domains/${encodeURIComponent(id)}/documents`); }

  /**
   * Upload one document as multipart/form-data: `file` and, when given,
   * `metadata` as JSON. The browser writes the multipart boundary itself.
   */
  uploadKnowledgeDocument(id, file, metadata = null) {
    const form = new FormData();
    form.append("file", file, file.name);
    if (metadata) form.append("metadata", JSON.stringify(metadata));
    return this.request("POST", `/knowledge/domains/${encodeURIComponent(id)}/documents`, { body: form });
  }

  /** Change a document's metadata { category, intent, script_phase, tags }. */
  updateKnowledgeDocument(id, doc, metadata) {
    return this.request("PATCH", `/knowledge/domains/${encodeURIComponent(id)}/documents/${encodeURIComponent(doc)}`, { json: { metadata } });
  }

  /** Delete one document (its chunks leave the index). */
  deleteKnowledgeDocument(id, doc) {
    return this.request("DELETE", `/knowledge/domains/${encodeURIComponent(id)}/documents/${encodeURIComponent(doc)}`);
  }

  /** Search a domain: a knowledge.query.v1 body, a knowledge.result.v1 answer. */
  knowledgeQuery(body) { return this.request("POST", "/knowledge/query", { json: body }); }

  // ---- live assist (the orchestrator's routes, proxied under /api/assist) ----

  /** Live products: product, banner, signals, script_phases, roles, live (a transcriber is linked). */
  liveProducts() { return this.request("GET", "/assist/products"); }

  /** Recorded conversations a demo replay can use: id, title, utterances, seconds. */
  liveTranscripts() { return this.request("GET", "/assist/transcripts"); }

  /** The prepared report of a scripted support call (#250): demo id -> call report and the call. */
  liveDemoReport(id) { return this.request("GET", `/assist/demos/${encodeURIComponent(id)}/report`); }

  /** The scripted voice-assistant example of a demo (#250). */
  liveDemoVoice(id) { return this.request("GET", `/assist/demos/${encodeURIComponent(id)}/voicebot`); }

  /** Which examples of a demo can be heard and each clip's length, in seconds (#254). */
  liveDemoVoices(id) { return this.request("GET", `/assist/demos/${encodeURIComponent(id)}/voices`); }

  /** One rendered line of a demo as a Blob (audio/wav), fetched with the access code (#254). */
  async liveDemoClip(id, line) {
    const response = await this.request("GET", `/assist/demos/${encodeURIComponent(id)}/voices/${encodeURIComponent(line)}.wav`, { raw: true });
    return response.blob();
  }

  /** Start a live session: { product, audio: "stereo" | "mic", session_id, employee? }. */
  startLive(body) { return this.request("POST", "/assist/sessions", { json: body }); }

  /** Replay a transcript as a live session: { product, transcript, speed, employee? }. */
  startReplay(id, body) {
    return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/replay`, { json: body });
  }

  stopLive(id) { return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/stop`); }

  /** The user spoke over the voice bot: stop its turn ({ stopped }). */
  bargeIn(id) { return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/barge-in`); }

  liveSession(id) { return this.request("GET", `/assist/sessions/${encodeURIComponent(id)}`); }

  /** Switch a session's assistance mode: "pincelada" | "guiado" (LV-16). */
  liveMode(id, mode) {
    return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/mode`, { json: { mode } });
  }

  /** Name a speaker of a session (also in its transcript when live). */
  renameLiveSpeaker(id, speaker, name) {
    return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/speakers/${encodeURIComponent(speaker)}`, { json: { name } });
  }

  /**
   * The employee's voice check of a one-microphone session (#249):
   * { action: "skip" }, or { action: "mine" | "not_mine", utterance } to say a
   * line's voice is (not) theirs. Straight to the transcriber's live session.
   */
  liveVoice(id, body) {
    return this.request("POST", `/live/sessions/${encodeURIComponent(id)}/voice`, { json: body });
  }

  /** Tick a mandatory item by hand, after the person confirmed it was said. */
  tickMandatory(id, item) {
    return this.request("POST", `/assist/sessions/${encodeURIComponent(id)}/mandatory/${encodeURIComponent(item)}`, { json: { confirm: true } });
  }

  // ---- the AI interviewer's plans (#208) ----

  /** The interview plans: id, title, role_type, minutes, slots, minimums, interviews. */
  interviewPlans() { return this.request("GET", "/assist/interview/plans"); }

  /** Draft a plan from a job offer: { offer | document: { domain, id }, title?, role_type?, minutes? }. */
  createPlan(body) { return this.request("POST", "/assist/interview/plans", { json: body }); }

  interviewPlan(id) { return this.request("GET", `/assist/interview/plans/${encodeURIComponent(id)}`); }

  /** Save the recruiter's edit (with the revision it was based on; 409 when older). */
  savePlan(id, plan) { return this.request("PUT", `/assist/interview/plans/${encodeURIComponent(id)}`, { json: plan }); }

  /** The candidates interviewed from a plan, for the comparison. */
  planResults(id) { return this.request("GET", `/assist/interview/plans/${encodeURIComponent(id)}/results`); }

  // ---- candidates of a role and their core answers (#235) ----

  /** The candidates of a role (plan): { candidates: [profile view], questions, retention_days }. */
  planCandidates(planId) { return this.request("GET", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates`); }

  /** Add a candidate to a role: { name }. */
  createCandidate(planId, name) {
    return this.request("POST", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates`, { json: { name } });
  }

  /** Delete a candidate's profile (answers and rounds). */
  deleteCandidate(planId, id) {
    return this.request("DELETE", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates/${encodeURIComponent(id)}`);
  }

  /** Keep the interviewer's answer to a core question (typed words). */
  correctAnswer(planId, id, question, text) {
    return this.request("PUT", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates/${encodeURIComponent(id)}/answers/${encodeURIComponent(question)}`, { json: { text } });
  }

  /** Forget one answer: the question is open again. */
  clearAnswer(planId, id, question) {
    return this.request("DELETE", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates/${encodeURIComponent(id)}/answers/${encodeURIComponent(question)}`);
  }

  /** Confirm captured proposals (all of them without `questions`). */
  confirmAnswers(planId, id, questions) {
    return this.request("POST", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates/${encodeURIComponent(id)}/confirm`, { json: questions ? { questions } : {} });
  }

  /** Discard one captured proposal (the question stays open). */
  discardProposal(planId, id, question) {
    return this.request("DELETE", `/assist/interview/plans/${encodeURIComponent(planId)}/candidates/${encodeURIComponent(id)}/proposals/${encodeURIComponent(question)}`);
  }

  /** The candidate interviewed in a session (404 when none). */
  sessionCandidate(sessionId) {
    return this.request("GET", `/assist/interview/candidates/by-session/${encodeURIComponent(sessionId)}`);
  }

  /**
   * Follow a session's Server-Sent Events from number `after`: `onItem`
   * gets each { id, event, data }. Resolves "end" when the session is
   * over, "closed" when the connection ended first (call again with the
   * last id to resume). A fetch stream is used so the access code
   * travels in the header, as on every other call.
   */
  async followLive(id, { after = 0, signal, onItem }) {
    const code = this.getCode();
    if (!code) throw new ApiFailure({ status: 401, kind: "unauthorized", code: "unauthorized" });
    const url = `/api/assist/sessions/${encodeURIComponent(id)}/suggestions/stream?after=${Number(after) || 0}`;
    let response;
    try {
      response = await this.fetchFn(url, {
        method: "GET", headers: { "X-Access-Code": code, accept: "text/event-stream" },
        cache: "no-store", credentials: "same-origin", signal,
      });
    } catch {
      throw new ApiFailure({ status: null, kind: "unavailable", code: "network" });
    }
    if (!response.ok || !response.body) {
      let info = {};
      try {
        info = await response.json();
      } catch {
        info = {};
      }
      const { kind, waitSeconds } = classifyFailure(response.status, response.headers.get("Retry-After"));
      throw new ApiFailure({ status: response.status, kind, waitSeconds, code: typeof info.code === "string" ? info.code : null });
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) return "closed";
        buffer += decoder.decode(value, { stream: true });
        const { items, rest } = parseSse(buffer);
        buffer = rest;
        for (const item of items) {
          if (item.event === "end") return "end";
          onItem(item);
        }
      }
    } catch (error) {
      if (signal?.aborted) return "closed";
      throw new ApiFailure({ status: null, kind: "unavailable", code: "network", message: String(error?.message ?? "") });
    }
  }

  // ---- model lab (docs/transcriber/LAB.md §7) ----

  labCatalogue() { return this.request("GET", "/lab/catalogue"); }

  /** Every session's newest run: steps done, left and failed (§7). */
  labOverview() { return this.request("GET", "/lab/overview"); }

  /** Mean and spread of the reference scores over one corpus (§13.3). */
  labReferenceSummary(corpus) { return this.request("GET", "/lab/reference-summary", { query: { corpus } }); }

  /** An imported session's reference speaker turns (anonymous corpus ids). */
  labReferenceTurns(session) { return this.request("GET", `/lab/sessions/${encodeURIComponent(session)}/reference-turns`); }

  labRuns(session) { return this.request("GET", `/lab/sessions/${encodeURIComponent(session)}/runs`); }

  labStart(session, body) { return this.request("POST", `/lab/sessions/${encodeURIComponent(session)}/runs`, { json: body }); }

  labRunPath(session, run) { return `/lab/sessions/${encodeURIComponent(session)}/runs/${encodeURIComponent(run)}`; }

  labStatus(session, run) { return this.request("GET", this.labRunPath(session, run)); }

  labCancel(session, run) { return this.request("POST", `${this.labRunPath(session, run)}/cancel`); }

  labResume(session, run) { return this.request("POST", `${this.labRunPath(session, run)}/resume`); }

  labResults(session, run) { return this.request("GET", `${this.labRunPath(session, run)}/results`); }

  labTracks(session, run) { return this.request("GET", `${this.labRunPath(session, run)}/tracks`); }

  labSpans(session, run) { return this.request("GET", `${this.labRunPath(session, run)}/fragment-spans`); }

  labFragments(session, run, { offset = 0, limit = 50, sort = "disagreement", unlabelled = false } = {}) {
    return this.request("GET", `${this.labRunPath(session, run)}/fragments`, {
      query: { offset, limit, sort, unlabelled: unlabelled ? "true" : null },
    });
  }

  labScores(session, run) { return this.request("GET", `${this.labRunPath(session, run)}/scores`); }

  labLabels(session) { return this.request("GET", `/lab/sessions/${encodeURIComponent(session)}/labels`); }

  labPutLabels(session, body) { return this.request("PUT", `/lab/sessions/${encodeURIComponent(session)}/labels`, { json: body }); }

  /** The address an <audio> element plays (it carries the playback cookie, not the code). */
  labAudioUrl(session, run, denoiser) { return `/api${this.labRunPath(session, run)}/audio/${encodeURIComponent(denoiser)}`; }

  /** A short excerpt cut on the server (playback cookie), for the overview's samples. */
  labExcerptUrl(session, run, denoiser, start, seconds) {
    const query = new URLSearchParams({ start: String(start), seconds: String(seconds) });
    return `/api${this.labRunPath(session, run)}/excerpt/${encodeURIComponent(denoiser)}?${query}`;
  }

  /** `pcm` is 16 kHz mono 16-bit little-endian PCM (the API's default format). */
  /** `cloneConsent` (true only) is the separate consent to the voice bot cloning this voice. */
  enrol(id, { name, language, textVersion, pcm, cloneConsent = false }) {
    return this.request("POST", `/speakers/${encodeURIComponent(id)}/enrolment`, {
      query: { name, language, consent: "true", text_version: textVersion, clone_consent: cloneConsent === true ? "true" : undefined },
      body: pcm, contentType: "application/octet-stream",
    });
  }
}
