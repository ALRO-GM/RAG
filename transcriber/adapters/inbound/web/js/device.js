// Device features: screen wake lock and the "leaving the page" guard.

/** Keeps the screen on while wanted; re-acquires it when the page is shown again. */
export class WakeLock {
  constructor(onChange = () => {}) {
    this.sentinel = null;
    this.wanted = false;
    this.onChange = onChange;
    this.listener = () => {
      if (document.visibilityState === "visible" && this.wanted) this._acquire();
    };
  }

  get supported() {
    return "wakeLock" in navigator;
  }

  async request() {
    this.wanted = true;
    document.addEventListener("visibilitychange", this.listener);
    return this._acquire();
  }

  async _acquire() {
    if (!this.supported || this.sentinel) return Boolean(this.sentinel);
    try {
      this.sentinel = await navigator.wakeLock.request("screen");
      this.sentinel.addEventListener("release", () => {
        this.sentinel = null;
        this.onChange(false);
      });
      this.onChange(true);
      return true;
    } catch {
      this.onChange(false);
      return false;
    }
  }

  async release() {
    this.wanted = false;
    document.removeEventListener("visibilitychange", this.listener);
    const s = this.sentinel;
    this.sentinel = null;
    try {
      await s?.release();
    } catch {
      // already released
    }
  }
}

let busy = () => false;
let installed = false;

/** True while something would be lost by leaving (recording, pending chunks). */
export function isBusy() {
  return busy();
}

/** Warn before the page is closed while `isBusy()` is true. */
export function setLeaveGuard(isBusy) {
  busy = isBusy;
  if (installed) return;
  installed = true;
  window.addEventListener("beforeunload", (event) => {
    if (!busy()) return;
    event.preventDefault();
    event.returnValue = "";
  });
}
