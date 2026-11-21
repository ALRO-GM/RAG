// Keep the screen awake while the microphone is in use (Screen Wake Lock API).
//
// A phone that dims and sleeps suspends the page: the live session or the
// voice-bot interview is cut and push-to-talk stops working. The browser
// drops the lock whenever the page is hidden, so it is taken again each
// time the page becomes visible while still held. Browsers without the API
// (or that refuse it: low battery, no HTTPS) simply keep their usual
// behaviour; nothing here throws.

export class WakeLock {
  /**
   * @param {object} [env] injectable for tests: { navigator, document }
   */
  constructor(env = {}) {
    this.nav = env.navigator ?? globalThis.navigator;
    this.doc = env.document ?? globalThis.document;
    this.wanted = false;
    this.sentinel = null;
    this._onVisible = () => {
      if (this.wanted && this.doc?.visibilityState === "visible") this._request();
    };
  }

  get supported() {
    return Boolean(this.nav?.wakeLock?.request);
  }

  get active() {
    return Boolean(this.sentinel && !this.sentinel.released);
  }

  /** Keep the screen on until release(); safe to call more than once. */
  async acquire() {
    if (this.wanted) return this.active;
    this.wanted = true;
    this.doc?.addEventListener?.("visibilitychange", this._onVisible);
    return this._request();
  }

  async release() {
    this.wanted = false;
    this.doc?.removeEventListener?.("visibilitychange", this._onVisible);
    const sentinel = this.sentinel;
    this.sentinel = null;
    try {
      await sentinel?.release?.();
    } catch {
      // already released by the browser
    }
  }

  async _request() {
    if (!this.supported || this.active) return this.active;
    try {
      const sentinel = await this.nav.wakeLock.request("screen");
      if (!this.wanted) {
        await sentinel.release?.();
        return false;
      }
      this.sentinel = sentinel;
      return true;
    } catch {
      return false;
    }
  }
}
