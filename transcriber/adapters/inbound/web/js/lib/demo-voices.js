// Pure logic of the scripted examples' voices (#254): which examples can
// be heard (GET /api/assist/demos/{id}/voices), the remembered sound
// on/off, the note that says the voices are synthetic, and a clip player
// that never hangs an example: a clip that does not load or play within
// its length (plus a margin) is given up, the example goes on with the
// text and the page says once that the sound is not available. Nothing
// plays before the viewer starts an example.
//
// Clips play through Web Audio: the bytes are fetched with the access code
// and decoded by an AudioContext the page resumes inside the viewer's
// click. Never an <audio> on a blob: URL — the public site's
// Content-Security-Policy (`media-src 'self'`) refuses those. No DOM here:
// the context, the clip loader and the clock are injected, so node --test
// runs it with fakes.

/** Where the sound on/off is remembered (this browser only). */
export const SOUND_KEY = "ragputin.demo.sound";

/** Extra time a clip may take beyond its length before it is given up, in ms. */
export const CLIP_MARGIN_MS = 2500;

/** Whether the sound is on: what this browser remembered, on by default. */
export function readSound(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(SOUND_KEY) !== "off";
  } catch {
    return true;
  }
}

/** Remember the sound on/off (ignored where storage is blocked). */
export function writeSound(on, storage = globalThis.localStorage) {
  try {
    storage?.setItem(SOUND_KEY, on ? "on" : "off");
  } catch {
    // a private window: the choice lasts for this page only
  }
}

/** The clips of one example ("call" | "voicebot"): Map line key -> seconds, or null when it cannot be heard. */
export function clipsOf(view, example) {
  const found = view?.examples?.[example];
  if (!found?.available) return null;
  const clips = new Map();
  for (const [key, clip] of Object.entries(found.clips ?? {})) {
    const seconds = Number(clip?.seconds);
    if (Number.isFinite(seconds) && seconds > 0) clips.set(key, seconds);
  }
  return clips.size ? clips : null;
}

/**
 * The i18n key of the line under an example: what kind of voices it has
 * (cloned with permission, stock or both), or that they are not available.
 */
export function voicesNoteKey(view, example) {
  if (!clipsOf(view, example)) return "dvoice.none";
  const kinds = new Set(Object.values(view.voices ?? {}));
  if (kinds.has("clone") && kinds.has("stock")) return "dvoice.mixed";
  return kinds.has("clone") ? "dvoice.clone" : "dvoice.stock";
}

/** A new AudioContext (suspended until resumed by a click), or null without Web Audio. */
export function createAudioContext(win = globalThis) {
  const Context = win?.AudioContext ?? win?.webkitAudioContext;
  if (typeof Context !== "function") return null;
  try {
    return new Context();
  } catch {
    return null;
  }
}

/**
 * The output of a ClipPlayer over a Web Audio `context` (null: no sound):
 * `ready()` resumes it (true when running), `decode(bytes)` makes a buffer,
 * `start(buffer)` plays it and returns { ended: Promise, stop() }.
 */
export function webAudioOutput(context) {
  return {
    async ready() {
      if (!context) return false;
      if (context.state !== "running") {
        try {
          await context.resume();
        } catch {
          return false;
        }
      }
      return context.state === "running";
    },
    decode(bytes) {
      if (!context) return Promise.reject(new Error("no Web Audio"));
      return context.decodeAudioData(bytes);
    },
    start(buffer) {
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      const ended = new Promise((resolve) => { source.onended = () => resolve(); });
      source.start();
      return {
        ended,
        stop() {
          try {
            source.stop();
          } catch {
            // already stopped
          }
        },
      };
    },
  };
}

/**
 * Plays one clip at a time. `load(key)` resolves to the clip's bytes (an
 * ArrayBuffer fetched with the access code); `output` is a
 * webAudioOutput(); `onFail()` is called once, the first time a clip
 * cannot be played. `play(key, seconds)` resolves true when the clip
 * ended, false when it was off, stopped, failed or ran out of time: never
 * hangs.
 */
export class ClipPlayer {
  constructor({ load, output, onFail = () => {}, schedule = setTimeout, cancel = clearTimeout }) {
    this.load = load;
    this.output = output;
    this.onFail = onFail;
    // Called bare: the browser's setTimeout refuses another `this`.
    this.schedule = (fn, ms) => schedule(fn, ms);
    this.cancel = (id) => cancel(id);
    this.sound = true;
    this.failed = false;
    this.loads = new Map();
    this.current = null;
  }

  /** Start loading (and decoding) the clips of `keys`, one after the other. */
  async prefetch(keys) {
    for (const key of keys) {
      if (this.stopped) return;
      await this._load(key).catch(() => null);
    }
  }

  _load(key) {
    if (!this.loads.has(key)) {
      const loading = Promise.resolve().then(() => this.load(key)).then((bytes) => this.output.decode(bytes));
      loading.catch(() => {});
      this.loads.set(key, loading);
    }
    return this.loads.get(key);
  }

  /** Turn the sound on or off; off stops the clip playing now. */
  setSound(on) {
    this.sound = Boolean(on);
    if (!this.sound) this.stop();
  }

  _fail() {
    if (this.failed) return;
    this.failed = true;
    this.onFail();
  }

  /** Play the clip of `key`, `seconds` long; resolves when it is over. */
  play(key, seconds) {
    this.stop();
    if (!this.sound) return Promise.resolve(false);
    return new Promise((resolve) => {
      let playing = null;
      let timer = null;
      const finish = (ok, broken = false) => {
        if (this.current !== entry) return;
        this.current = null;
        this.cancel(timer);
        playing?.stop();
        if (broken) this._fail();
        resolve(ok);
      };
      const entry = { stop: () => finish(false) };
      this.current = entry;
      timer = this.schedule(() => finish(false, true), Math.max(0, seconds) * 1000 + CLIP_MARGIN_MS);
      Promise.all([this._load(key), this.output.ready()]).then(([buffer, ready]) => {
        if (this.current !== entry) return;
        // Autoplay rules: a context the viewer has not unlocked by a click stays suspended.
        if (!ready) return finish(false, true);
        playing = this.output.start(buffer);
        playing.ended.then(() => finish(true));
        return null;
      }).catch(() => finish(false, true));
    });
  }

  /** Stop the clip playing now, if any. */
  stop() {
    this.current?.stop();
  }

  /** Stop and forget everything. */
  dispose() {
    this.stopped = true;
    this.stop();
    this.loads.clear();
  }
}

/**
 * The live page's scripted call: play a line's clip when the line starts
 * (its first partial or, if none came, its final), each line once.
 * Returns the line id to play, or null.
 */
export function lineToPlay(message, clips, started) {
  if (message?.type !== "event" || !clips) return null;
  const event = message.event;
  if (!["partial", "final"].includes(event?.type)) return null;
  const id = event.utterance?.id;
  if (!id || !clips.has(id) || started.has(id)) return null;
  started.add(id);
  return id;
}
