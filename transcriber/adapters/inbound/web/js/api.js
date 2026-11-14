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

  async request(method, path, { query, json, body, contentType, auth = true } = {}) {
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
        return await response.json();
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
   * on the session: consent_version, num_speakers, expected_speakers.
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

  uploadRecording(id, file, language, title, consentVersion) {
    return this.request("POST", `/sessions/${encodeURIComponent(id)}/upload`, {
      query: { filename: file.name, language, title, consent_version: consentVersion },
      body: file, contentType: "application/octet-stream",
    });
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

  /** `pcm` is 16 kHz mono 16-bit little-endian PCM (the API's default format). */
  enrol(id, { name, language, textVersion, pcm }) {
    return this.request("POST", `/speakers/${encodeURIComponent(id)}/enrolment`, {
      query: { name, language, consent: "true", text_version: textVersion },
      body: pcm, contentType: "application/octet-stream",
    });
  }
}
