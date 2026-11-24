// Pure logic of the voice-assistant example (#/voicebot/demo, #250): the
// scripted exchange between a customer and the voice assistant of the
// same fictitious company, played step by step: the customer speaks (text
// appears), the assistant «thinks», then answers with its sources. With
// rendered clips (#254) both sides are heard and every line lasts as long
// as its clip, with the sound on or off; without them the assistant is
// spoken by the browser's own voice when it has one. No DOM, microphone,
// server session or model here, so node --test runs it with a fake clock
// and narrator.

/** Pause before a customer line, a short «thinking» pause and the reading speed, in ms. */
export const TIMING = { beforeUser: 700, perWordUser: 260, thinking: 900, perWordBot: 330, after: 1200, minRead: 1200 };

/** How long a text stays on screen when nobody reads it aloud, in ms. */
export function readMs(text, perWord) {
  const words = String(text ?? "").trim().split(/\s+/).filter(Boolean).length;
  return Math.max(TIMING.minRead, words * perWord);
}

/** The steps of the example: [{ kind, index, text, sources? }] in the order they happen. */
export function stepsOf(demo, sources, language = "es") {
  const byId = new Map((sources ?? []).map((s) => [s.id, s]));
  const pick = (item, field) => (language === "en" && item?.en?.[field]) || item?.[field] || "";
  const steps = [];
  (demo?.exchanges ?? []).forEach((e, index) => {
    steps.push({ kind: "user", index, key: `vb${index}.user`, text: pick(e, "user") });
    steps.push({ kind: "thinking", index, text: "" });
    steps.push({
      kind: "bot", index, key: `vb${index}.bot`, text: pick(e, "bot"),
      sources: (e.cites ?? []).filter((id) => byId.has(id)).map((id) => ({
        id, text: byId.get(id).text, title: (language === "en" && byId.get(id).en?.title) || byId.get(id).title || id,
      })),
    });
  });
  return steps;
}

/**
 * Play the steps in order. `show(step)` draws one; `narrate(text)` speaks
 * the assistant's line and resolves when it ends (or at once when there
 * is no voice); `sleep(ms)` waits and rejects when `signal` aborts.
 * With `clips` (Map line key -> seconds) every line takes its clip's
 * length and `voice(key, seconds)` plays it (null: muted, same timing);
 * the browser's voice is then not used. Resolves true when it reached
 * the end, false when stopped.
 */
export async function play(steps, { show, narrate = null, sleep, signal = null, clips = null, voice = null }) {
  const heard = async (step) => {
    const seconds = clips?.get(step.key);
    if (!seconds) return false;
    // The clip and its length run together: muted or failed, the time is the same.
    await Promise.all([sleep(seconds * 1000), voice ? voice(step.key, seconds) : null]);
    return true;
  };
  try {
    for (const step of steps) {
      if (signal?.aborted) return false;
      if (step.kind === "user") await sleep(TIMING.beforeUser);
      show(step);
      if (step.kind === "user") {
        if (!(await heard(step))) await sleep(readMs(step.text, TIMING.perWordUser));
      } else if (step.kind === "thinking") await sleep(TIMING.thinking);
      else {
        if (!(await heard(step))) {
          const spoken = narrate ? await narrate(step.text) : false;
          if (!spoken) await sleep(readMs(step.text, TIMING.perWordBot));
        }
        await sleep(TIMING.after);
      }
    }
    return !signal?.aborted;
  } catch (error) {
    if (error?.name === "AbortError" || signal?.aborted) return false;
    throw error;
  }
}

/** Longest wait for a voice to report its end: three times the reading time plus a few seconds, in ms. */
export function narrationLimitMs(text) {
  return readMs(text, TIMING.perWordBot) * 3 + 3000;
}

/** A sleep that rejects with an AbortError when `signal` aborts. */
export function abortableSleep(signal, schedule = setTimeout, cancel = clearTimeout) {
  return (ms) => new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      return;
    }
    const id = schedule(() => {
      signal?.removeEventListener?.("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      cancel(id);
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    }
    signal?.addEventListener?.("abort", onAbort, { once: true });
  });
}

/**
 * A narrator on the browser's own speech synthesis (nothing leaves the
 * browser and no server voice, GPU or session is used), or null when the
 * browser has none. `speak(text)` resolves true when the voice ended.
 */
export function browserNarrator(synth, Utterance, language = "es", { schedule = setTimeout, cancel = clearTimeout } = {}) {
  if (!synth || typeof Utterance !== "function") return null;
  const tag = language === "en" ? "en" : "es";
  return {
    speak(text) {
      return new Promise((resolve) => {
        // A browser whose voice never reports the end must not hang the example.
        const limit = narrationLimitMs(text);
        let timer = null;
        const finish = (ok) => {
          cancel(timer);
          resolve(ok);
        };
        timer = schedule(() => {
          synth.cancel?.();
          resolve(false);
        }, limit);
        const u = new Utterance(text);
        const voices = synth.getVoices?.() ?? [];
        const voice = voices.find((v) => String(v.lang).toLowerCase().startsWith(tag === "es" ? "es-es" : "en")) ?? voices.find((v) => String(v.lang).toLowerCase().startsWith(tag));
        if (!voice && voices.length) {
          finish(false); // voices are loaded and none speaks this language: the text stays on screen
          return;
        }
        if (voice) u.voice = voice;
        u.lang = voice?.lang ?? (tag === "es" ? "es-ES" : "en-GB");
        u.onend = () => finish(true);
        u.onerror = () => finish(false);
        synth.speak(u);
      });
    },
    cancel() {
      synth.cancel?.();
    },
  };
}
