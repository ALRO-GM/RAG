# Design system of the web pages

The transcriber's pages (`transcriber/adapters/inbound/web/`, served at
rag.peer4u.app) share one look: a calm, neutral "work tool" style in the
line of today's SaaS dashboards, adapted to people who are not technical
(HR staff, interviewers) and who often use a phone. This page is the
brief; the tokens live at the top of `style.css`.

## Direction (research, November 2026)

What current product UIs converge on, and what we took:

| Trend | Source of the idea | Taken here |
|---|---|---|
| Collapsible left sidebar for apps with many sections | 2026 SaaS dashboard reviews (saasframe.io) | Sidebar with three groups; collapses to icons on desktop, becomes a drawer on phones |
| Command palette (Ctrl/Cmd + K) | Linear, saasui.design 2026 trends | Quick switcher: pages, theme, language |
| Neutral colour with a sparing accent, themes generated from a few inputs | Linear's redesign (LCH, base + accent + contrast) | 12-step neutral scale + one indigo accent, light and dark |
| A 12-step scale with fixed roles (1-2 canvas, 3-5 fills, 6-8 borders, 9-10 solid, 11-12 text) | Radix Colors docs | Same role mapping; our own values |
| Dense but readable, minimal chrome | Linear, Vercel, Stripe dashboards | Thin borders, soft shadows, 4 px spacing grid; larger type than those tools (15-16 px) because readers are not engineers |

Only the style is adopted: no logo, name, font file or asset of any brand.

## Constraints that shaped it

- **No build step, no framework.** Vanilla ES modules as before; a design
  system in CSS custom properties needs neither. One extra file, `js/boot.js`
  (classic script), applies the remembered theme before the first paint.
- **CSP** (`infra/Caddyfile`): `default-src 'self'; style-src 'self'; script-src 'self'`.
  No inline styles or scripts, no CDN, and `tests/integration/transcriber/test_web_pages.py`
  forbids `@font-face` and `url()` in CSS. So the font is the **system UI
  stack** (Inter first, when the device has it), and icons are an SVG sprite
  (`icons.svg`, `<use href="icons.svg#i-…">`, own simple line drawings).
- **Accessibility AA**: text contrast ≥ 4.5:1 in both themes. WCAG ratios
  of the token pairs (computed from the hex values): light text 15.45,
  muted on white 6.20 and on the canvas 5.84, links 6.00, white on the
  primary button 5.21, danger on its chip 4.85, ok on its chip 4.83, warning
  5.94; dark text 16.25, muted 9.06, links 8.53, primary button 7.60, chips
  7.75 / 7.86 / 10.26. Placeholders use the muted colour. 2 px accent focus ring (5.21 light, 7.98 dark), 44 px
  touch targets (40 px controls with a fine pointer), skip link, `aria-current`
  on the page's link, `prefers-reduced-motion` turns animation off.

## Tokens

| Group | Tokens |
|---|---|
| Neutral | `--gray-1` … `--gray-12` |
| Accent | `--accent-2` … `--accent-12` (indigo; `--accent-9` is the solid) |
| Status | `--red-*`, `--green-*`, `--amber-*` (3 = background, 9 = solid, 11 = text) |
| Roles (used by every rule) | `--bg`, `--surface`, `--surface-2`, `--sunken`, `--text`, `--muted`, `--line`, `--line-strong`, `--accent`, `--accent-ink`, `--active`, `--danger(-bg)`, `--ok(-bg)`, `--warn-bg`, `--warn-text`, `--focus` |
| Type | `--font`, `--mono`, `--fs-xs` 12 px … `--fs-2xl` 26 px |
| Space and shape | `--s-1` 4 px … `--s-10` 40 px; `--r-sm` 6, `--r-md` 8, `--r-lg` 12, `--r-xl` 16 px; `--shadow-sm/md/lg` |
| Motion | `--fast` 120 ms, `--normal` 200 ms, `--ease` |
| Charts and speakers | `--c0…6`, `--spk-*`, `--who-*`, `--cert-*` (kept from before, colour-blind-checked) |

Dark mode redefines the same names twice: under
`@media (prefers-color-scheme: dark)` for `:root:not([data-theme="light"])`,
and for `:root[data-theme="dark"]`. The theme button cycles
system → light → dark and remembers the choice in `localStorage`
(`transcriber.theme`, behind try/catch).

## Shell and components

- **Shell** (`js/shell.js`, pure parts in `js/lib/shell.js`): sidebar
  (`<header class="top sidebar">`, grouped: Conversaciones, Empresa,
  Laboratorio), top bar with menu button, breadcrumbs (from the route),
  "Ir a…" button and theme button; a bottom tab bar on phones
  (Sesiones, Grabar, Conocimiento, Más). A loading skeleton shows if a page
  takes more than 180 ms.
- **Components** (`style.css` §5, helpers in `js/ui.js`): buttons (primary,
  default, ghost, danger, small, big), inputs, cards, badges/status chips,
  tabs, tables, empty states, toasts, dialogs (`<dialog>`), skeletons, the
  drop zone, tags.
- Existing page classes (analysis results, lab, research and study charts)
  keep their names and pick up the new tokens.

## Pages

Every page uses the shell: Sesiones, Grabar, Registrar voz, Conocimiento
(new: `#/knowledge`), En directo (`#/live`: live assist with suggestion cards, sources, script phases and the notice banner; `docs/platform/ORCHESTRATOR.md`), Pruebas, the session and analysis pages, the help
page, and the static `research.html` and `study.html`. ES/EN is unchanged
(`js/i18n/es.js`, `en.js`).
