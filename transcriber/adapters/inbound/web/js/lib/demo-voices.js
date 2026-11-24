// Pure logic of the scripted examples' voices (#254): which examples can
// be heard (GET /api/assist/demos/{id}/voices), the remembered sound
// on/off, the note that says the voices are synthetic, and a clip player
// that never hangs an example: a clip that does not load or play within
// its length (plus a margin) is given up, the example goes on with the
// text and the page says once that the sound is not available. Nothing
// plays before the viewer starts an example. No DOM here: the audio
// element, the clip loader and the clock are injected, so node --test
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

/**
 * Plays one clip at a time. `load(key)` resolves to an address the audio
 * element can play (an object URL of the clip fetched with the access
 * code); `makeAudio(url)` returns an element with play(), pause(),
 * addEventListener; `onFail()` is called once, the first time a clip
 * cannot be played. `play(key, seconds)` resolves true when the clip ended,
 * false when it was off, stopped, failed or ran out of time: never hangs.
 */
export class ClipPlayer {
  constructor({ load, makeAudio, onFail = () => {}, schedule = setTimeout, cancel = clearTimeout }) {
    this.load = load;
    this.makeAudio = makeAudio;
    this.onFail = onFail;
    this.schedule = schedule;
    this.cancel = cancel;
    this.sound = true;
    this.failed = false;
    this.loads = new Map();
    this.current = null;
  }

  /** Start loading the clips of `keys`, one after the other. */
  async prefetch(keys) {
    for (const key of keys) {
      if (this.stopped) return;
      await this._load(key).catch(() => null);
    }
  }

  _load(key) {
    if (!this.loads.has(key)) {
      const loading = Promise.resolve().then(() => this.load(key));
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
      let audio = null;
      let timer = null;
      const finish = (ok, broken = false) => {
        if (this.current !== entry) return;
        this.current = null;
        this.cancel(timer);
        try {
          audio?.pause?.();
        } catch {
          // already gone
        }
        if (broken) this._fail();
        resolve(ok);
      };
      const entry = { stop: () => finish(false) };
      this.current = entry;
      timer = this.schedule(() => finish(false, true), Math.max(0, seconds) * 1000 + CLIP_MARGIN_MS);
      this._load(key).then((url) => {
        if (this.current !== entry) return;
        audio = this.makeAudio(url);
        audio.addEventListener?.("ended", () => finish(true), { once: true });
        audio.addEventListener?.("error", () => finish(false, true), { once: true });
        // Autoplay rules: a browser that refuses to play (no click yet) rejects here.
        Promise.resolve(audio.play?.()).catch(() => finish(false, true));
      }, () => finish(false, true));
    });
  }

  /** Stop the clip playing now, if any. */
  stop() {
    this.current?.stop();
  }

  /** Stop and forget everything (`revoke(url)` frees each loaded address). */
  async dispose(revoke = () => {}) {
    this.stopped = true;
    this.stop();
    for (const loading of this.loads.values()) {
      const url = await loading.catch(() => null);
      if (url) revoke(url);
    }
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
