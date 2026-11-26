# Design system of the web pages

The transcriber's pages (`transcriber/adapters/inbound/web/`, served at
rag.peer4u.app) share one look: a calm, neutral "work tool" style in the
line of today's SaaS dashboards, adapted to people who are not technical
(HR staff, interviewers) and who often use a phone. This page is the
brief; the tokens live at the top of `style.css`.

## Calm palette (UI-1, issue #229)

The owner approved three mock-ups on 2026-11-24, kept in
[`docs/web/mockups/`](mockups/) (`Main.dc.html` the home, `Live.dc.html`
the live page, `Perfil.dc.html` a result; plain HTML with inline styles,
for reading only: they load a web font and a `support.js` the app never
uses). `Live.dc.html` sets the style for every page:

- a neutral light ground (`#f3f4f6`) with white cards and thin borders;
- one muted indigo accent (`#3f4fa8`) for links, the main button and the
  "suggestion" dot;
- colour only in small dots and labels, never in large blocks: terracotta
  means *better avoid*, green means *done*;
- minimum information first, the rest behind «Más información».

The home takes the layout of `Main.dc.html` (three service cards, then the
latest sessions) but in this calm palette, not its saturated colour blocks.
Fonts stay the system stack (the mock-ups' Inter and Space Grotesk are not
loaded: CSP).

## Direction (research, November 2026)

What current product UIs converge on, and what we took:

| Trend | Source of the idea | Taken here |
|---|---|---|
| Collapsible left sidebar for apps with many sections | 2026 SaaS dashboard reviews (saasframe.io) | Sidebar with three groups; collapses to icons on desktop, becomes a drawer on phones |
| Command palette (Ctrl/Cmd + K) | Linear, saasui.design 2026 trends | Quick switcher: pages, theme, language |
| Neutral colour with a sparing accent, themes generated from a few inputs | Linear's redesign (LCH, base + accent + contrast) | 12-step neutral scale + one muted indigo accent, light and dark |
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
  of the calm palette's token pairs, computed from the hex values with the
  WCAG 2 formula on 2026-11-24:

  | Pair | Light | Dark |
  |---|---|---|
  | `--text` on `--bg` / on `--surface` | 13.92 / 15.32 | 14.89 / 13.88 |
  | `--muted` on `--surface` / on `--bg` | 6.58 / 5.98 | 6.99 / 7.50 |
  | `--accent` (links, focus ring) on `--surface` / on `--bg` | 7.25 / 6.59 | 7.21 / 7.73 |
  | `--accent-text` on `--accent` (main button) | 7.25 | 7.24 |
  | `--accent-text` on `--accent-hover` | 9.44 | 9.05 |
  | `--accent-ink` on `--active` (current nav item) | 6.38 | 6.08 |
  | `--avoid` on `--avoid-bg` | 6.47 | 7.72 |
  | `--avoid-strong` on `--avoid-bg` | 10.34 | – |
  | `--done` on `--done-bg` | 6.00 | 8.67 |
  | `--warn-text` on `--warn-bg` | 6.42 | 9.01 |
  | `--surface` on `--danger` (solid danger button, error toast) | 7.65 | 8.61 |
  | `--faint` on `--surface` (dots and outlines only, 3:1 target) | 3.32 | 4.31 |

  Placeholders use the muted colour. 2 px accent focus ring, 44 px
  touch targets (40 px controls with a fine pointer), skip link, `aria-current`
  on the page's link, `prefers-reduced-motion` turns animation off.

## Tokens

| Group | Tokens |
|---|---|
| Neutral | `--gray-1` … `--gray-12` (`--gray-2` the ground, `--gray-5` the line, `--gray-11` muted text, `--gray-12` text) |
| Accent | `--accent-2` … `--accent-12` (muted indigo; `--accent-9` is the solid) |
| Status | `--red-*` (terracotta), `--green-*`, `--amber-*` (3 = background, 9 = solid/dot, 11 = text, `--red-12` strong text) |
| Roles (used by every rule) | `--bg`, `--surface`, `--surface-2`, `--sunken`, `--text`, `--muted`, `--faint`, `--line`, `--line-strong`, `--accent`, `--accent-hover`, `--accent-ink`, `--accent-text`, `--active`, `--danger(-bg)`, `--ok(-bg)`, `--warn-bg`, `--warn-text`, `--focus` |
| Meaning (live and summary pages) | `--suggest`, `--suggest-dot` (accent: a suggestion), `--avoid`, `--avoid-dot`, `--avoid-bg`, `--avoid-strong` (terracotta: better avoid), `--done`, `--done-bg` (green: done, covered) |
| Type | `--font`, `--mono`, `--fs-xs` 12 px … `--fs-2xl` 26 px, `--fs-display` (30–44 px, the home's question) |
| Space and shape | `--s-1` 4 px … `--s-10` 40 px; `--r-sm` 6, `--r-md` 10, `--r-lg` 14, `--r-xl` 20 px; `--shadow-sm/md/lg` (lighter than before) |
| Motion | `--fast` 120 ms, `--normal` 200 ms, `--ease` |
| Charts and speakers | `--c0…6`, `--spk-*`, `--who-*`, `--cert-*` (kept from before, colour-blind-checked) |

New rules use the role and meaning tokens rather than a scale step or a
hex value, so a later page (the summary-first analysis, #230; the live
page, #232) gets both themes for free. Some older rules (hover borders,
the skeleton, toasts, the drop zone, the interviewer's bars) still name a
scale step, which both themes define, and the print block
sets plain hex colours on purpose. Small components that carry the meaning
colours: `.state-dot` (`.done`, `.avoid`, `.suggest`), `.type-chip`.

Dark mode redefines the same names twice: under
`@media (prefers-color-scheme: dark)` for `:root:not([data-theme="light"])`,
and for `:root[data-theme="dark"]`. The theme button cycles
system → light → dark and remembers the choice in `localStorage`
(`transcriber.theme`, behind try/catch).

## Shell and components

- **Shell** (`js/shell.js`, pure parts in `js/lib/shell.js`): sidebar
  (`<header class="top sidebar">`) split in two. On top, what a worker
  uses: Inicio (`#/`), Sesiones (`#/sessions`), En directo, Asistente de
  voz, Ofertas (`#/offers`, #271). Below a rule, the **Administración** group (`.nav-admin`): Estado
  del servicio (`#/admin`), Ajustes, Registrar voz, Conocimiento, Pruebas,
  Estudio, Investigación (a `role="group"` labelled by its heading). No
  password of its own yet. `crumbsFor` builds the breadcrumbs of a route
  within its section and, when `isAdminRoute` says the route belongs to
  the area (`ADMIN_PARTS`: admin, settings, enrol, knowledge, lab), puts
  «Administración» (`#/admin`) in front, except on `#/admin` itself.
  Top bar with menu button, breadcrumbs (from the route), "Ir a…" button,
  «Compartir pantalla» (`#share-open`, #278, `docs/web/SCREEN_SHARE.md`;
  never mirrored: `data-share-skip`) and theme button; a bottom tab bar on phones (Inicio, Sesiones, En
  directo, Más). A loading skeleton shows if a page takes more than 180 ms.
- **Components** (`style.css` §5, helpers in `js/ui.js`): buttons (primary,
  default, ghost, danger, small, big), inputs, cards, badges/status chips,
  tabs, tables, empty states, toasts, dialogs (`<dialog>`), skeletons, the
  drop zone, tags.
- Existing page classes (analysis results, lab, research and study charts)
  keep their names and pick up the new tokens.

## Summary first (#230)

Result pages show the minimum a worker needs and put the rest in a
closed `<details class="sf-more">` «Más información» whose summary line
says what is inside. Nothing is deleted; the confidence, the caveats and
the «orientativo / provisional / evidencia insuficiente» notices live
there. Pure parts in `js/lib/summary-first.js`, drawing in
`js/views/summary-first.js`, styles in the «Summary first» section of
`style.css`, dictionary keys `sf.*` (grouped at the end of `es.js` and
`en.js`).

- **Análisis** (`#/analysis/<id>`, after `docs/web/mockups/Perfil.dc.html`
  in the calm palette): first the session in one paragraph (who, how
  long, decisions, tasks, open questions), the job, and one row per
  person in speaking order (percentage and verdict words; a click or
  Enter opens that person's card and moves the focus to it), with a
  one-line draft note. Then one card per person as a disclosure; only
  the first starts open (the data has no speaker roles, so the candidate
  of an interview cannot be picked, nor the interviewer left without a
  fit reading). A card's header is always visible: name, percentage,
  the verdict as descriptive words chosen by conversation type from
  `overall.verdict` (`sf.verdict.<job|coaching|sale|general>.<strong|adequate|weak>`,
  e.g. «Encaje alto con lo que pide el puesto»; never a decision such as
  «pasa a la siguiente fase») and the first sentence of `overall.line`.
  Open, it shows the score ring (SVG, the arc's `stroke-dasharray` from
  `ringDash`), a summary of at most two sentences, at most 3 strengths
  and 2 improvements (`top_strengths` / `top_improvements`), the bars of
  those (at most 5, in that order) and «Ver las N competencias» with
  every bar; the minimum-required mark only when a skill carries a
  `required` level, which the analysis does not write yet. Text is never
  cut inside a clause nor ended with «…»: a sentence too long for its
  place is cut at its last clause boundary (`clauseCut`), or left out
  (the competency name alone for a point). Each card's «Más
  información» holds how reliable it is (confidence, caveats in words),
  the whole reading (`overall.frame`, the full `overall.line`, every
  competency's verdict, score, full conclusion, basis, author and cited
  sentences), the reports' lists, the explained profile, every
  competency with its quotes and the participation figures; the
  session's «Más información» holds the facts, the draft and AI-Act
  notices, the share of words, the events and the transcript. The known
  open issues moved into «Detalles técnicos».
- **Older analyses** (synthesis v1, no `overall`): no ring nor verdict;
  the summary says once that the analysis predates the overall reading
  and offers «Rehacer solo la síntesis» when it can run now
  (`canRedoSynthesis`); strengths and improvements are the reports'
  names and the bars the stored firm or provisional levels.
- **Transcripción** (`#/s/<id>`): title, «Análisis», the player and the
  conversation; type and goal, names, voices, filters and the weak-voice
  notice under «Más información».
- **Informe del entrevistador** (voice bot, `reportCard`): suitability,
  summary, one bar per point of the plan and where the transcript went;
  the tiles, notes, confidence, quotes, gaps and facts under «Más
  información».
- The live page's hand-off is not covered here (the live page is being
  rebuilt separately, #232).

## Pages

Every page uses the shell.

- **Inicio** (`#/`, `js/views/home.js`, pure parts in `js/lib/home.js`):
  «¿Qué quieres hacer?», then three service cards (icon, one sentence,
  «Probar»): Analizar una conversación (→ `#/sessions`, where recording
  and upload live), En directo (`#/live`), Asistente de voz
  (`#/voicebot`). Below, the four latest sessions: title, type, and either
  the state or, when the analysis wrote a synthesis, its one-line
  conclusion (first sentence of the first person's job fit, else their
  first strength; a full stop after an abbreviation such as «Sr.» or
  «p. ej.» or a single letter does not end it) with a green «Analizada»
  dot, kept even when the synthesis itself cannot be read. When the
  synthesis has an overall reading (v2, `overall`; docs/skills/DESIGN.md
  §14.1) the line is that person's `overall.line` and the row shows its
  `percent` with one decimal («67,5 %»); the pages never compute their
  own score nor rank people.
- **Sesiones** (`#/sessions`): the former home. Each card shows the
  title, state, date, length, progress and the action that moves it
  forward (open, resume, finish, or transcribe when there is no
  transcript); the description, notes and the other actions (describe,
  transcribe again, delete) sit under «Más opciones» (`splitActions`).
- **Ofertas** (`#/offers`, `js/views/offers.js`, pure parts in
  `js/lib/offers.js`; #271): the open positions as cards (closed ones on
  request) and, per offer (`#/offers/<plan>`), a real `<table>` of the
  people interviewed: caption, column headers whose sort is a button
  with `aria-sort`, the person as a sticky row header, every state in
  words (a tinted cell or the terracotta mark of a missing required
  point always carries its word), search and filters as labelled native
  controls, a `role="status"` line saying how many are shown and the
  order, the state kept in the address. The table scrolls inside its
  `.table-wrap` on a phone. The order is the server's draft figure
  (`docs/live/VOICEBOT_INTERVIEWER.md` §6.4), shown with the draft notice
  and «Cómo se ordena»; the page computes no figure itself.
- **Administración** (`#/admin`, `js/views/admin.js`): the service status
  (the API's health check) and one card per set-up tool.
- The rest: Grabar, Registrar voz, Conocimiento (`#/knowledge`), Ajustes
  (`#/settings`: the quality/cost mode and the model of every step, each
  option with its measured trade-off; `docs/SETTINGS.md`), En directo
  (`#/live`, the calm page of `Live.dc.html` since #232: one «Ahora»
  card, topic chips, earlier suggestions and options under
  disclosures, `#/live/demo` the scripted interview;
  `docs/live/EMPLOYEE_UX.md`; its start form and «Asistente de voz»
  carry the «Calidad de la conexión» card and a session the `.lv-conn`
  chip, `docs/live/CONNECTION.md`), Pruebas, the
  session and analysis pages, the help page, and the static
  `research.html` and `study.html`. ES/EN in `js/i18n/es.js`, `en.js`.
