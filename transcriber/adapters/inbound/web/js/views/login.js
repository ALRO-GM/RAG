// The login screen: shown when this browser holds no access code. The
// password is traded for the code (POST /api/login); the code is kept the
// way the #k link keeps it. Nothing but the password field is asked.

import { h } from "../dom.js";
import { t } from "../i18n.js";
import { loginMessage } from "../lib/login.js";

/**
 * Render the login into `app`. `api.login(password)` gives `{ code }`;
 * `onCode(code)` keeps it and shows the page.
 */
export function loginView(api, app, onCode) {
  const input = h("input", {
    type: "password", id: "login-password", name: "password", required: true,
    autocomplete: "current-password", maxLength: 256,
  });
  const message = h("p", { class: "hint login-message", role: "alert" });
  const submit = h("button", { type: "submit", class: "primary big" }, t("login.submit"));
  let unlock = null;

  const form = h("form", { class: "stack", novalidate: true },
    h("div", { class: "field" }, h("label", { for: "login-password" }, t("login.password")), input),
    message, submit);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const password = input.value;
    if (!password) return;
    submit.disabled = true;
    message.textContent = "";
    try {
      const { code } = await api.login(password);
      input.value = "";
      onCode(code);
      return;
    } catch (error) {
      const { key, values } = loginMessage(error);
      message.textContent = t(key, values);
      input.value = "";
      if (error?.kind === "rate") {
        clearTimeout(unlock);
        unlock = setTimeout(() => { submit.disabled = false; }, Math.max(error.waitSeconds, 1) * 1000);
        return;
      }
    }
    submit.disabled = false;
    input.focus();
  });

  app.replaceChildren(h("section", { class: "stack gate login" },
    h("h1", {}, t("login.title")), h("p", {}, t("login.body")), form));
  input.focus();
  return () => clearTimeout(unlock);
}
