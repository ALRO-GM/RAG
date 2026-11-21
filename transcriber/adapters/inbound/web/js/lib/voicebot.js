// Pure logic of the voice-bot page (#/voicebot): the microphone's voice
// detector for barge-in, the queue that plays the bot's sentences back to
// back, and the conversation built from the orchestrator's feed (the
// user's live.event.v1 finals and partials, the bot's spoken sentences,
// interruptions and per-turn latency). No DOM here, so node --test runs it.
//
// Barge-in (docs/live/VOICEBOT.md): while the bot speaks, the user's voice
// on the microphone stops the playback at once (EnergyVad), and the page
// tells the orchestrator, which cancels the turn. This assumes a headset:
// with loudspeakers the microphone hears the bot itself, so the page only
// listens for barge-in when the person says they wear one.

export const VOICEBOT = "voicebot";
export const MODES = ["push", "hands_free"];

/**
 * A voice detector on the microphone's peak level, one value per audio block
 * (about 85 ms): speech starts after `startMs` above `threshold`, ends after
 * `quietMs` below it.
 */
export class EnergyVad {
  constructor({ threshold = 0.08, startMs = 150, quietMs = 600 } = {}) {
    this.threshold = threshold;
    this.startMs = startMs;
    this.quietMs = quietMs;
    this.reset();
  }

  reset() {
    this.speaking = false;
    this.loudSince = null;
    this.quietSince = null;
  }

  /** Feed one level at `nowMs`; return "start", "end" or null. */
  push(level, nowMs) {
    const loud = level >= this.threshold;
    if (!this.speaking) {
      if (!loud) {
        this.loudSince = null;
        return null;
      }
      this.loudSince ??= nowMs;
      if (nowMs - this.loudSince >= this.startMs) {
        this.speaking = true;
        this.quietSince = null;
        return "start";
      }
      return null;
    }
    if (loud) {
      this.quietSince = null;
      return null;
    }
    this.quietSince ??= nowMs;
    if (nowMs - this.quietSince >= this.quietMs) {
      this.reset();
      return "end";
    }
    return null;
  }
}

/** The bytes of a base64 string. */
export function base64ToBytes(text) {
  const raw = globalThis.atob(String(text ?? ""));
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

/**
 * Plays the bot's sentences one after the other on an AudioContext (or a
 * stand-in with the same few members). Chunks of a turn that was
 * interrupted are dropped, also when they arrive late.
 */
export class SpeechQueue {
  constructor(ctx, { onPlaying = () => {} } = {}) {
    this.ctx = ctx;
    this.onPlaying = onPlaying;
    this.endTime = 0;
    this.sources = new Set();
    this.stoppedTurn = 0;
    this.chain = Promise.resolve();
  }

  /** Whether audio is playing or waiting to play. */
  get playing() {
    return this.sources.size > 0 && this.ctx.currentTime < this.endTime;
  }

  /** Queue the WAV bytes of `turn`; resolves once scheduled (or dropped). */
  enqueue(turn, bytes) {
    this.chain = this.chain.then(() => this._schedule(turn, bytes)).catch(() => {});
    return this.chain;
  }

  async _schedule(turn, bytes) {
    if (turn <= this.stoppedTurn) return false;
    const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const buffer = await this.ctx.decodeAudioData(copy);
    if (turn <= this.stoppedTurn) return false;
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const at = Math.max(this.ctx.currentTime, this.endTime);
    source.start(at);
    this.endTime = at + buffer.duration;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      if (!this.playing) this.onPlaying(false);
    };
    this.onPlaying(true);
    return true;
  }

  /** Stop everything now and drop what is still to come of `turn` and older. */
  stop(turn = Infinity) {
    if (Number.isFinite(turn)) this.stoppedTurn = Math.max(this.stoppedTurn, turn);
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // already stopped
      }
    }
    this.sources.clear();
    this.endTime = 0;
    this.onPlaying(false);
  }
}

/** The conversation of a voice-bot session, from the orchestrator's feed. */
export class VoiceBotState {
  constructor() {
    this.page = null;
    this.state = "running";
    this.error = null;
    this.items = []; // { kind: "user" | "bot", key, text, partial?, state? }
    this.byKey = new Map();
    this.records = [];
    this.halted = null;
    this.lastId = 0;
  }

  _item(kind, key, fields) {
    let item = this.byKey.get(key);
    if (!item) {
      item = { kind, key, text: "" };
      this.byKey.set(key, item);
      this.items.push(item);
    }
    Object.assign(item, fields);
    return item;
  }

  /** Apply one feed message; `id` is its SSE number. */
  apply(message, id = null) {
    if (id !== null && Number(id) > this.lastId) this.lastId = Number(id);
    const type = message?.type;
    if (type === "session" || type === "status") {
      const session = message.session ?? {};
      this.page = session.page ?? this.page;
      this.state = session.state ?? this.state;
      this.error = session.error ?? null;
    } else if (type === "event") {
      const event = message.event ?? {};
      const u = event.utterance;
      if ((event.type === "partial" || event.type === "final") && u) {
        this._item("user", `u:${u.id}`, { text: u.text, partial: event.type === "partial" });
      } else if (event.type === "session_ended") {
        this.state = "ended";
      }
    } else if (type === "bot") {
      this._item("bot", `b:${message.turn}`, { turn: message.turn, state: "thinking" });
    } else if (type === "speech") {
      const item = this._item("bot", `b:${message.turn}`, { turn: message.turn });
      if (item.state !== "interrupted") item.state = "speaking";
      item.text = item.text ? `${item.text} ${message.text}` : message.text;
    } else if (type === "interrupt") {
      this._item("bot", `b:${message.turn}`, { turn: message.turn, state: "interrupted", by: message.by });
    } else if (type === "turn") {
      const record = message.record ?? {};
      this.records.push(record);
      const item = this._item("bot", `b:${record.turn}`, { turn: record.turn, record });
      if (item.state !== "interrupted") item.state = record.status === "ok" ? "done" : record.status;
      if (record.status === "quota" || record.status === "auth") this.halted = record.status;
    }
  }

  /** The conversation, oldest first; empty partials are left out. */
  lines() {
    return this.items.filter((item) => item.kind === "bot" || item.text);
  }
}

/** End of the user's turn → first audio of a turn record, and its parts (ms). */
export function turnLatency(record) {
  const stages = record?.stages ?? {};
  const lag = stages.event_lag_ms ?? null;
  const first = stages.first_audio_ms ?? null;
  return {
    lag, knowledge: stages.knowledge_ms ?? null, text: stages.first_text_ms ?? null,
    tts: stages.first_tts_ms ?? null, first,
    total: lag !== null && first !== null ? Math.round(lag + first) : null,
  };
}
