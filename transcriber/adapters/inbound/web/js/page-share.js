// «Compartir pantalla» on the pages that are documents of their own
// (#284): study.html, research.html and competition.html (#291). They
// are not part of the application (index.html), so going there unloads
// it; the shared session goes on because each of these pages installs
// the same component (js/share.js) and resumes it with the token this
// tab keeps, exactly as a reload does. The access code is the one the
// application kept in this tab (sessionStorage, CODE_KEY); without it the
// panel says to enter the application first (the relay refuses a socket
// without a valid code).

import { Api } from "./api.js";
import { installShare } from "./share.js";

/** Where the application keeps the access code for this tab (js/main.js). */
export const CODE_KEY = "transcriber.code";

/** The access code kept in this tab, or null. */
export function storedCode() {
  try {
    return sessionStorage.getItem(CODE_KEY);
  } catch {
    return null;
  }
}

/**
 * Add «Compartir pantalla» to a standalone page. Its own content needs no
 * pause: while this device shows a shared page, the shell is hidden
 * (`.share-viewing`) and nothing of the page acts on the API.
 */
export function installPageShare() {
  return installShare({ api: new Api(storedCode), readCode: storedCode });
}
