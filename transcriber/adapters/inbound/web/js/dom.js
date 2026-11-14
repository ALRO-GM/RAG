// Tiny DOM helpers. Text always goes in as text nodes (never innerHTML).

/** Create an element: h("button", { class: "x", onClick: fn }, "text", child...). */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") {
      el.className = value;
    } else if (key === "for") {
      el.htmlFor = value;
    } else if (key.length > 2 && key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key.includes("-") || key === "role") {
      el.setAttribute(key, value === true ? "" : String(value));
    } else if (key in el) {
      el[key] = value;
    } else {
      el.setAttribute(key, value === true ? "" : String(value));
    }
  }
  append(el, children);
  return el;
}

/** Append strings, nodes and (nested) arrays; null and false are skipped. */
export function append(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(parent, child);
    else parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Keep keyboard focus on the same logical control across a re-render. */
export function keepFocus(root, render) {
  const key = document.activeElement?.dataset?.focusKey;
  render();
  if (key) root.querySelector(`[data-focus-key="${CSS.escape(key)}"]`)?.focus();
}
