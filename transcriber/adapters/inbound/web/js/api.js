// Client of the transcriber API. Same origin, everything under /api, the
// access code in the X-Access-Code header on every call but /health.
// The code is never put in a URL.

import { classifyFailure } from "./lib/net.js";

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

  async request(method, path, { query, json, body, contentType, auth = true, text = false } = {}) {
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

  reassignUtterance(id, utterance, speaker) {
    return this.request("PATCH", `/sessions/${encodeURIComponent(id)}/utterances/${encodeURIComponent(utterance)}`, { json: { speaker } });
  }

  deleteSession(id) { return this.request("DELETE", `/sessions/${encodeURIComponent(id)}`); }

  speakers() { return this.request("GET", "/speakers"); }

  deleteSpeaker(id) { return this.request("DELETE", `/speakers/${encodeURIComponent(id)}`); }

  /** The consent notice in force: kind "recording" or "enrolment". */
  consent(kind, language) {
    return this.request("GET", "/consent", { query: { kind, language } });
  }

  enrolmentText(language, name) {
    return this.request("GET", "/enrolment/text", { query: { language, name } });
  }

  // ---- analysis (docs/skills/DESIGN.md §4) ----

  /** The RAG's analysis judge and whether Claude's quota is available. */
  analysisJudge() { return this.request("GET", "/analysis/judge"); }

  /** Choose the judge of the next analyses: claude_code, local or both. */
  setAnalysisJudge(mode) { return this.request("PUT", "/analysis/judge", { json: { mode } }); }

  /** Start the RAG's whole analysis of a published session. */
  startAnalysis(id, { restart = false, synthetic = false } = {}) {
    return this.request("POST", `/analysis/sessions/${encodeURIComponent(id)}`, { json: { restart, synthetic } });
  }

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
  enrol(id, { name, language, textVersion, pcm }) {
    return this.request("POST", `/speakers/${encodeURIComponent(id)}/enrolment`, {
      query: { name, language, consent: "true", text_version: textVersion },
      body: pcm, contentType: "application/octet-stream",
    });
  }
}
