// Pure logic of the voice-assistant example (#/voicebot/demo, #250): the
// scripted exchange between a customer and the voice assistant of the
// same fictitious company, played step by step: the customer speaks (text
// appears), the assistant «thinks», then answers with its sources, spoken
// aloud when the browser has a voice. No DOM, microphone, server session
// or model here, so node --test runs it with a fake clock and narrator.

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
    steps.push({ kind: "user", index, text: pick(e, "user") });
    steps.push({ kind: "thinking", index, text: "" });
    steps.push({
      kind: "bot", index, text: pick(e, "bot"),
      sources: (e.cites ?? []).filter((id) => byId.has(id)).map((id) => ({ id, text: byId.get(id).text })),
    });
  });
  return steps;
}

/**
 * Play the steps in order. `show(step)` draws one; `narrate(text)` speaks
 * the assistant's line and resolves when it ends (or at once when there
 * is no voice); `sleep(ms)` waits and rejects when `signal` aborts.
 * Resolves true when it reached the end, false when stopped.
 */
export async function play(steps, { show, narrate = null, sleep, signal = null }) {
  try {
    for (const step of steps) {
      if (signal?.aborted) return false;
      if (step.kind === "user") await sleep(TIMING.beforeUser);
      show(step);
      if (step.kind === "user") await sleep(readMs(step.text, TIMING.perWordUser));
      else if (step.kind === "thinking") await sleep(TIMING.thinking);
      else {
        const spoken = narrate ? await narrate(step.text) : false;
        if (!spoken) await sleep(readMs(step.text, TIMING.perWordBot));
        await sleep(TIMING.after);
      }
    }
    return !signal?.aborted;
  } catch (error) {
    if (error?.name === "AbortError" || signal?.aborted) return false;
    throw error;
  }
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
export function browserNarrator(synth, Utterance, language = "es") {
  if (!synth || typeof Utterance !== "function") return null;
  const tag = language === "en" ? "en" : "es";
  return {
    speak(text) {
      return new Promise((resolve) => {
        const u = new Utterance(text);
        const voices = synth.getVoices?.() ?? [];
        const voice = voices.find((v) => String(v.lang).toLowerCase().startsWith(tag === "es" ? "es-es" : "en")) ?? voices.find((v) => String(v.lang).toLowerCase().startsWith(tag));
        if (!voice && voices.length) {
          resolve(false); // voices are loaded and none speaks this language: the text stays on screen
          return;
        }
        if (voice) u.voice = voice;
        u.lang = voice?.lang ?? (tag === "es" ? "es-ES" : "en-GB");
        u.onend = () => resolve(true);
        u.onerror = () => resolve(false);
        synth.speak(u);
      });
    },
    cancel() {
      synth.cancel?.();
    },
  };
}
