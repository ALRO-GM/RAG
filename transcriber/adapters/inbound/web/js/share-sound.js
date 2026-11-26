// «Compartir pantalla» (#278): the sound plays on one device only.
//
// The page that runs the application is the sharing device's, so every
// sound starts there: the scripted examples' voices and the voice bot
// (Web Audio: decodeAudioData + AudioBufferSourceNode.start) and the
// players of recordings (<audio>). While sharing, when the panel says the
// sound goes out through a joined device, those calls are caught here:
// locally the sound is played silent (so every "ended" still comes on time
// and the page goes on exactly as before), and the clip's bytes (once) and
// a "play" / "stop" go to that device only, which plays them
// (SoundPlayer). An <audio> sends its address (same-origin /api/ only);
// the joined device fetches it with its own access code.
//
// The microphone is never touched: it stays on the device that opened it.

const MAX_CLIP_BYTES = 3 * 1024 * 1024;
const MAX_CLIPS = 40;

function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return globalThis.btoa(text);
}

function fromBase64(text) {
  const raw = globalThis.atob(text);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes.buffer;
}

/**
 * Catch the sharing device's sounds. `remote()` tells whether they go to
 * another device now; `send(body)` sends a body to that device. Returns
 * { restore, uninstall }: restore() unmutes the <audio> elements muted
 * here (the sound came back), uninstall() restores and removes the catch.
 */
export function installSoundBridge({ win = globalThis, remote, send }) {
  const Base = win.BaseAudioContext ?? win.AudioContext;
  const Source = win.AudioBufferSourceNode;
  const Media = win.HTMLMediaElement;
  const undo = [];
  const clips = new WeakMap(); // AudioBuffer -> { id, bytes }
  const sentClips = new Set();
  const playing = new WeakMap(); // source -> play id
  const muted = new Set(); // <audio> elements muted here because their sound went elsewhere
  const followed = new WeakSet(); // <audio> elements whose pause/seek/end are told
  let next = 1;

  /** Unmute what was muted here (the sound is back on this device, or sharing ended). */
  function restore() {
    for (const el of muted) {
      el.muted = el.dataset.shareMuted === "was";
      delete el.dataset.shareMuted;
    }
    muted.clear();
  }

  if (Base?.prototype?.decodeAudioData) {
    const decode = Base.prototype.decodeAudioData;
    Base.prototype.decodeAudioData = function decodeAudioData(data, ...rest) {
      const copy = data && data.byteLength <= MAX_CLIP_BYTES ? data.slice(0) : null;
      const result = decode.call(this, data, ...rest);
      Promise.resolve(result).then((buffer) => {
        if (copy && buffer) clips.set(buffer, { id: next++, bytes: copy });
      }, () => {});
      return result;
    };
    undo.push(() => { Base.prototype.decodeAudioData = decode; });
  }

  if (Source?.prototype?.start) {
    const start = Source.prototype.start;
    const stop = Source.prototype.stop;
    Source.prototype.start = function startAudio(when = 0, offset = 0, ...rest) {
      const clip = this.buffer ? clips.get(this.buffer) : null;
      if (clip && remote()) {
        try {
          // Silent here: the same timing, the sound elsewhere.
          const silent = this.context.createGain();
          silent.gain.value = 0;
          this.disconnect();
          this.connect(silent);
          silent.connect(this.context.destination);
        } catch {
          // a node that cannot be rerouted plays here
        }
        if (!sentClips.has(clip.id)) {
          sentClips.add(clip.id);
          send({ k: "clip", id: clip.id, data: toBase64(clip.bytes) });
        }
        const play = next++;
        playing.set(this, play);
        const delay = Math.max(0, (when || 0) - (this.context.currentTime ?? 0));
        send({ k: "play", id: clip.id, play, delay: Math.round(delay * 1000) / 1000, offset: offset || 0 });
      }
      return start.call(this, when, offset, ...rest);
    };
    Source.prototype.stop = function stopAudio(...args) {
      const play = playing.get(this);
      if (play) send({ k: "stop", play });
      return stop.apply(this, args);
    };
    undo.push(() => { Source.prototype.start = start; Source.prototype.stop = stop; });
  }

  if (Media?.prototype?.play) {
    const play = Media.prototype.play;
    Media.prototype.play = function playMedia(...args) {
      const src = sameOriginApi(this.currentSrc || this.src, win.location);
      if (src && remote()) {
        if (!this.dataset.shareMuted) {
          this.dataset.shareMuted = this.muted ? "was" : "1";
          this.muted = true;
          muted.add(this);
        }
        if (!followed.has(this)) {
          followed.add(this);
          const tell = (op) => () => send({ k: "media", op, src, t: this.currentTime });
          const later = (op) => { const fn = tell(op); return () => { if (remote() && this.dataset.shareMuted) fn(); }; };
          this.addEventListener("pause", later("pause"));
          this.addEventListener("seeked", later("seek"));
          this.addEventListener("ended", later("pause"));
        }
        send({ k: "media", op: "play", src, t: this.currentTime });
      } else if (this.dataset.shareMuted) {
        this.muted = this.dataset.shareMuted === "was";
        delete this.dataset.shareMuted;
        muted.delete(this);
      }
      return play.apply(this, args);
    };
    undo.push(() => { Media.prototype.play = play; });
  }

  return {
    restore,
    uninstall: () => {
      restore();
      for (const fn of undo.reverse()) fn();
    },
  };
}

/** The path of `url` when it is an /api/ address of this origin, else null. */
export function sameOriginApi(url, loc) {
  try {
    const parsed = new URL(url, loc.href);
    return parsed.origin === loc.origin && parsed.pathname.startsWith("/api/") ? parsed.pathname + parsed.search : null;
  } catch {
    return null;
  }
}

/**
 * The device the sound goes out through: plays what the bridge sends.
 * `context()` gives (and makes) its AudioContext; `prepareMedia()` resolves
 * before an <audio> address is played (the playback cookie).
 */
export class SoundPlayer {
  constructor({ context, prepareMedia = async () => {}, makeAudio = (src) => new Audio(src), timers = globalThis }) {
    this.context = context;
    this.prepareMedia = prepareMedia;
    this.makeAudio = makeAudio;
    this.timers = timers;
    this.clips = new Map(); // id -> Promise<AudioBuffer>
    this.sources = new Map(); // play -> source
    this.media = null;
  }

  handle(body) {
    if (body.k === "clip") {
      const ctx = this.context();
      if (!ctx) return;
      const loading = ctx.decodeAudioData(fromBase64(body.data));
      loading.catch(() => {});
      this.clips.set(body.id, loading);
      while (this.clips.size > MAX_CLIPS) this.clips.delete(this.clips.keys().next().value);
    } else if (body.k === "play") {
      this._play(body);
    } else if (body.k === "stop") {
      const source = this.sources.get(body.play);
      this.sources.delete(body.play);
      try {
        source?.stop();
      } catch {
        // already over
      }
    } else if (body.k === "media") {
      this._media(body);
    }
  }

  async _play({ id, play, delay = 0, offset = 0 }) {
    const ctx = this.context();
    const loading = this.clips.get(id);
    if (!ctx || !loading) return;
    try {
      const buffer = await loading;
      if (ctx.state !== "running") await ctx.resume?.();
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => this.sources.delete(play);
      this.sources.set(play, source);
      source.start(ctx.currentTime + Math.max(0, delay), offset);
    } catch {
      // the clip cannot play here: the page goes on without it
    }
  }

  async _media({ op, src, t = 0 }) {
    if (op === "play") {
      if (!this.media || this.media.dataset?.src !== src) {
        this.media?.pause?.();
        await this.prepareMedia();
        this.media = this.makeAudio(src);
        if (this.media.dataset) this.media.dataset.src = src;
      }
      this.media.currentTime = t;
      this.media.play?.().catch?.(() => {});
    } else if (this.media) {
      this.media.currentTime = t;
      if (op === "pause") this.media.pause?.();
    }
  }

  /** Stop everything (the device is no longer the sound's). */
  stop() {
    for (const source of this.sources.values()) {
      try {
        source.stop();
      } catch {
        // over
      }
    }
    this.sources.clear();
    this.media?.pause?.();
    this.media = null;
  }
}
