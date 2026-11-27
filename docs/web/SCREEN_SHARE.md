# «Compartir pantalla»: one application on several devices (#278)

For a presentation: a tablet in the presenter's hand runs the application
and a computer on the projector shows the same thing, without video. One
device **shares** (a name and a 4-digit password), the others **join**
with them. The owner's step-by-step guide is
[`PRESENTAR_CON_DOS_DISPOSITIVOS.md`](PRESENTAR_CON_DOS_DISPOSITIVOS.md).

Everything goes over the internet through the public site
(rag.peer4u.app → tunnel → the transcriber): no local network, no
discovery, no peer-to-peer, no local addresses. Each device only needs to
be inside the application already (its access code).

## The mechanism: one source of truth, mirrored as DOM

Two ways were possible:

1. **Replicated events**: each device runs its own copy of the
   application and the actions are replayed on every copy.
2. **Mirroring**: only the sharing device runs the application; its page
   (the DOM, form values, scroll, focus, viewport size, pointer) is sent
   to the others, and the actions of the joined devices that may act are
   sent back to the sharing device, where they are applied.

We use **mirroring**. Replicated events drift as soon as anything is not
a pure function of the actions: timers (the live page's clock, the
scripted demo's script), data loaded asynchronously (the same request
answered at different moments), a live session (only one page can own
the microphone and the session's audio socket: #263 hands the source to
the newest socket, so a second copy would *take the session over*),
random proposals (the 4-digit password itself), the browser's storage.
None of that can be proven not to drift. With mirroring there is one
application, so there is nothing to drift: what a joined device shows is
by construction a copy of one page, and every action has exactly one
effect, on that page. It also gives the local things for free: the
microphone, the live session, the timers and the sound all live on the
sharing device only, and a joined device never runs the live page, so it
can never trigger the #263 reattach or takeover.

No pixels are captured: what travels is the tree of elements and texts.

### How a page is mirrored (`js/lib/mirror.js`)

- **Numbers.** The sharing device gives every node of `<body>` it shows a
  number. A joined device builds the same tree in an `<iframe>` *without
  scripts* (`sandbox="allow-same-origin"`, filled by the parent through the
  DOM) with the same numbers and the same stylesheet.
- **Snapshot** `{k: "s", tree, p, root, vw, vh, sc, es, f, n}`: the whole
  `<body>` (`[id, text]` for a text, `[id, tag, attrs, children, "s"?]`
  for an element, `"s"` for SVG), the form values (`p`), the `<html>`
  language, theme (resolved: a joined device with another system theme
  still shows the sharing device's) and classes, the viewport, the
  scroll, the focus and the node count.
- **Delta** `{k: "d", c, a, t, p, n}`, at most every 100 ms and only when
  something changed (a `MutationObserver` collects what changed): for each
  element whose children changed, its whole child list (`c`: a known
  child by number, a new one serialised whole); for each element whose
  attributes changed, the attributes that changed and the names removed
  (`a`); for each changed text, its text (`t`); the form values and open
  dialogs that changed (`p`). A delta says how things **are**, not what
  happened, so a delta applied after a lost one is either still right or
  names a node the device does not have.
- **No drift, detectable.** Every delta carries `n`, the number of nodes
  the sharing device holds after it. A joined device whose count differs,
  or that meets a number it does not hold, sends `resync` and gets a new
  snapshot (at most every 2 s). A late joiner, a device that reconnects
  and a device that fell behind all start from a snapshot.
- **Other bodies**: `r` (the `<html>` changed: theme, language, classes),
  `v` (viewport size), `sc` (scroll: the window as offsets, fractions and
  an anchor — the element at the top and how far into it — and scrolled
  elements as `[id, top, left, fraction]`), `f` (focus), `pt` (pointer),
  and the sound bodies (`clip`, `play`, `stop`, `media`).
- **Properties** that are not attributes travel as `p`: `value`
  (`v`), `checked` (`c`), and dialogs (`o` open, `m` modal: a joined
  device opens its copy with `showModal()` so it looks the same).
- **Style** attributes are applied through the CSSOM (`el.style.cssText`):
  the site's CSP (`style-src 'self'`) refuses `style` set as an attribute.

### Never mirrored

| What | How |
|---|---|
| `<script>`, `<noscript>`, `<template>`, `<iframe>`, `<object>`, `<embed>`, `<link>`, `<style>`, `<meta>` | left out of the tree |
| The «Compartir pantalla» button, panel, indicator, pointer marker and the joined device's stage | `data-share-skip` |
| `on*` attributes, `srcdoc`, `autofocus`, `javascript:` / `vbscript:` / `data:text/html` addresses | dropped |
| `src` of `<audio>`, `<video>`, `<source>`, `<track>`; `autoplay` | dropped: sound plays on one device (below) |
| Password fields, anything inside `[data-share-sensitive]` | values and texts sent as `••••••` |
| The access code | taken out of every text and attribute (`scrub`) before anything leaves; it goes to the relay only as the socket's first message, as on the live sockets |
| Browser storage, permission prompts | not part of the DOM: never seen |
| The address bar's `#k=…` | the application removes it on arrival; the joined device's own address never changes |

### Actions of a joined device

Inside the `<iframe>` the parent listens (capture phase) and prevents the
default of every click, submit, dialog cancel and drag: *nothing happens
on the joined device itself*. If it may send, it sends
`{"type": "in", "body": …}`:

| Body | Sent on | Applied on the sharing device |
|---|---|---|
| `{k: "click", node}` | a click (on the nearest control: link, button, summary, label, field, option, `role=button/option/tab/menuitem/switch`, `tabindex`) | `focus()` then `click()` on the same node (links navigate, boxes toggle, buttons submit) |
| `{k: "val", node, v}` | `input` / `change` of a field | `value` set, `input` and `change` dispatched |
| `{k: "key", node, key, code, ctrl, meta, shift, alt}` | every key | `keydown` + `keyup` on the node; `Escape` not handled closes the top dialog; `Enter` in a form without a submit button submits it (with one, the joined device's browser already sent the click of implicit submission) |
| `{k: "focus", node}` | focus moved | `focus()` |
| `{k: "sc", same, w | e}` | the joined device scrolls | the sharing device scrolls (by offset in the same layout, by anchor or fraction in another) |
| `{k: "pt", node, fx, fy, down}` | pointer moves (at most 20 per second) and touches | drawn on the sharing device and passed to the other devices |

**Typing on a joined device.** The joined device's field shows each key
at once; the sharing device's echo of the value comes back 150–300 ms
later and is older than what has been typed since. So the value of the
field being typed in is *held* for 1.2 s after the last key (the echo's
other properties still apply) and put back afterwards if the sharing
device changed it (`MirrorView` `hold` / `release`). Without it, typing
«abcdef» at that latency ended as «abf» (`tests/web/share.test.mjs`).

**Numbers belong to one page load.** The snapshot carries an epoch
(`e`); every action carries it back (`ep`), and the sharing device drops
an action of another epoch (its page was reloaded and numbered anew).
An action on a hidden, inert or disabled control (by attribute: a control
only the joined device's own layout shows, such as the phone menu, still
works), on the
«Compartir pantalla» panel, on a file field or a password field does
nothing; a forwarded Escape never closes the panel.

**Controls that depend on the layout.** Every action carries the layout
of the joined device's view (`wide`: the app's 64 rem breakpoint, in its
own view or the sharing device's), and the sharing device applies it as
that layout (`actingAs` in `js/shell.js`). So the menu button pressed on a
tablet joined in its own layout opens the drawer (`nav-open`, which the
tablet's mirror shows at tablet width) and never folds the desktop
sidebar that shares (`nav-collapsed`); the scrim, a link and Escape close
it. The other viewport checks in the pages, all harmless: «Ofertas» opens
its filters at first on a screen from 48 rem (`views/offers.js`
`wideScreen`, the sharing device's width decides, they can be folded from
any device), the bottom bar and the phone menu are shown by CSS only, and
dialogs are sized by CSS in each device's own frame.

Each field being typed in has its own hold, so typing in two fields in a
row loses nothing. Each client may try 6 shares at once and then one
every 10 s (`too_many_attempts`), so the unique names cannot be probed
quickly.

**Who sends.** Each device in the panel has a switch «Envía acciones»
(`role="switch"`). Default: the sharing device sends, joined devices
only watch. Only the sharing device changes it, for any device including
itself; any device can set *itself* to only watch (`{"type": "watch"}`).
When several devices send, **actions are applied in the order they reach
the sharing device** (the relay keeps each device's order; the sharing
device's own touches are applied when they happen): the last one wins.

**Watch-only is enforced twice.** On the page: the `<iframe>` of a device
that only watches is `inert` (no focus, no clicks) and its keys are
swallowed. On the server: the relay drops every `in` message of a device
whose switch is off (`ShareRelay._from_viewer`), whatever the page sends
(tested: `tests/unit/transcriber/test_screen_share.py`,
`tests/integration/transcriber/test_share_api.py`). A sharing device set
to only watch ignores its own trusted touches, clicks, keys and wheel
(capture listeners on `window`); the synthetic events that apply joined
devices' actions still go through.

**The pointer.** Whoever sends has a discreet ring drawn on the other
devices (`.share-pointer`, 1.6 rem, the accent colour, filled while
touching, hidden after 2.5 s still), placed by element and fraction of
its box, so it lands on the same thing in another layout.

### The two views of a joined device

- **«Vista de este dispositivo»** (default): the `<iframe>` fills the
  screen under the device's bar, so the stylesheet's breakpoints apply to
  *this* screen: the projector shows the desktop layout with the sidebar
  while the tablet shows its own. Scroll goes by anchor.
- **«Vista del dispositivo que comparte»**: the `<iframe>` has the sharing
  device's viewport size and is scaled to fit (`transform: scale`),
  centred: the tablet's screen, as the tablet shows it. Scroll goes by
  offset.

One button in the bar switches («Ver como el que comparte» / «Ver a mi
tamaño»), also in the panel; it is remembered for the tab.

### Sound and microphone

The microphone and every audio capture stay on the device that opened
them (only the sharing device runs the application; a joined device's
own application is paused while it shows the mirror, and a device that
is recording cannot join). **The sound comes out of the sharing
device.** For sound on the projector, share from the computer on the
HDMI and join from the tablet with «Envía acciones» on.

Choosing another device for the sound («El sonido sale por») is written
but **not offered** (`SOUND_CHOICE = false` in `js/share.js`): that path
was never run in a browser with real voices. What it does when turned on
(`js/share-sound.js`):

- Web Audio (the scripted examples' voices, the voice bot): the bytes
  given to `decodeAudioData` are kept; when such a buffer starts, the
  sharing device plays it through a silent gain (same timing, every
  `ended` on time) and sends the clip once (`clip`, base64, at most 3 MB)
  and `play` (`delay`, `offset`) / `stop` to that device only (`to`).
- `<audio>` of an `/api/` address: muted locally; the address and time go
  as `media` and the chosen device plays it with its own access code (the
  playback cookie).
- A joined device needs one touch to be allowed to play sound (browser
  autoplay rules): its bar shows «Activar el sonido aquí» when needed.

Not exercised end to end in the two-browser check: the throwaway stack
has no rendered voices and no voice-bot TTS (see «What was measured»).

## Pages that are documents of their own (#284)

«Estudio» (`study.html`) and «Investigación» (`research.html`) are not
routes of the application (`index.html`): going there, or back, loads
another document, so the sharing device's page, its socket and its
`MirrorSource` are gone. Each of the two pages installs the same component
(`js/page-share.js` → `installShare` of `js/share.js`, the same button
`#share-open`, panel and indicator), so on a page change the sharing
device does exactly what a reload does: the new page finds the share
kept in this tab (`sessionStorage`, role and resume token) and the access
code the application kept in the same tab (`transcriber.code`), resumes
(`resume` with the token, never the name or password) and sends its
whole page with a new epoch. A joined device does nothing: it keeps
showing the last page (its indicator says «el dispositivo que comparte
se está reconectando» while the relay sees the sharing device away) and
replaces it with the snapshot of the new page (body, its classes, scroll,
theme, language). Actions numbered on the old page carry the old epoch
and are dropped.

| What | How on those pages |
|---|---|
| Start, manage, stop sharing; join | the same button, panel and indicator as in the application |
| A joined device that may send | goes there and back through the sidebar (and the bottom bar's «Más», which opens it), follows the page's table of contents (in-page anchors), scrolls, opens the data tables (`<details>`), types in the reference search, chooses in its filters |
| Charts | both pages draw their charts as inline SVG (`js/views/study.js`, `js/research/main.js`), which is mirrored as DOM like the rest; no `<canvas>` on these pages (nor in the application), so nothing needs a special path |
| Without the access code in this tab | (the page opened directly, not from the application) the panel shows the relay's refusal: «Vuelve a entrar en la aplicación y prueba otra vez.» |

Every other way out of the document:

| Way out | What happens |
|---|---|
| `index.html` ↔ `study.html` ↔ `research.html` by their links (also «Ir a…»; the browser's back and forward buttons load the pages the same way, but were not exercised in the check) | the session goes on (above) |
| A link to another site (the references of «Investigación», the sources of «Estudio») clicked on a **joined** device | opened on that device, in a new tab; never sent to the sharing device (and the sharing device drops such a click if one arrives) |
| The same link clicked on the **sharing** device while sharing | opened in a new tab there, so the shared page stays |
| Opening an application page in a new tab of the sharing device (middle click, «Abrir en una pestaña nueva») | the browser copies this tab's `sessionStorage`, so the new tab resumes the share and the relay closes the old tab's link (`4409`, replaced): **avoid it while sharing** |
| Closing the tab, or typing another address | the sharing device is away; the session ends after 10 min unless the same tab comes back |

The resume token stays where it was: this tab's `sessionStorage` on the
sharing device, never in the DOM (so never mirrored), sent only in the
socket's `resume` message, kept by the relay as a SHA-256 hash and valid
while the device is away at most 10 min. A device without it cannot take
the sharing device's place (`resume` with another token: refused,
`ended`); the relay still stores nothing.

## The relay: `WS /api/share`

`transcriber/adapters/inbound/share_ws.py` (adapter),
`transcriber/application/screen_share.py` (`ShareRelay`, in memory),
`transcriber/domain/screen_share.py` (rules and limits).

1. `{"type": "auth", "access_code"}` — as on the live sockets (the code is
   never in the address); wrong codes are throttled like every route.
2. Within 10 s one of `{"type": "host", "name", "pin", "label"}`,
   `{"type": "join", "name", "pin", "label"}` or
   `{"type": "resume", "token"}`.
3. `{"type": "welcome", "device", "host", "token"?, "limits"}`, then
   `roster` messages whenever something changes:
   `{"type": "roster", "name", "host", "sound", "devices": [{id, label, host, sends, online}]}`.

| From | Message | Effect |
|---|---|---|
| sharing device | `{"type": "m", "body", "to"?, "snap"?}` | to every joined device or to `to`; the server never reads `body` |
| sharing device | `{"type": "set", "device", "sends"}` · `{"type": "sound", "device"}` · `{"type": "remove", "device"}` | who sends, where the sound plays, take a device out |
| joined device | `{"type": "in", "body"}` | to the sharing device as `{"type": "in", "from", "body"}`, **only if that device sends** |
| joined device | `{"type": "resync"}` | the server asks the sharing device for a snapshot for it |
| any | `{"type": "watch"}` · `{"type": "leave"}` · `{"type": "ping"}` | only watch from now · leave (the sharing device leaving ends the session) · `pong` |
| server | `{"type": "need_snapshot", "device"}` | to the sharing device, once per device waiting |
| server | `{"type": "ended", "reason"}` | `stopped`, `removed`, `expired`, `host_left`, `too_many_attempts`, `left`, `gone` |

**Slow devices never slow the presenter.** Each device has an outbox of at
most 8 MB of changes. A device that cannot keep up loses the changes
waiting for it (never a control message) and is marked behind; the
server asks the sharing device once for a snapshot for it and drops its
changes until that snapshot arrives. The sharing device never waits for
anyone: its sends do not block, and its own outbox only holds actions.
A joined device that has not received its first snapshot is "behind" too:
a change can never reach a device before the page it applies to.

**Close codes.** 4400 invalid input, 4401 no valid access code, 4403 full
or too many sessions, 4404 no session with that name *and* password,
4408 no hello in time, 4409 replaced by a newer connection of the same
device, 4410 ended / removed, 4413 message too large, 4429 too many wrong
passwords. The page stops retrying on all of them.

## Security and privacy

| Measure | Value |
|---|---|
| Who can share or join | only a device inside the application: the socket needs the access code first, throttled like every route |
| Password | 4 digits; kept as PBKDF2-SHA256, 50 000 rounds (about 8 ms here), random 16-byte salt per session; never logged, never sent back |
| Names | compared trimmed, one space, without case; not enumerable: «no such name» and «wrong password» are the same answer (`no_match`, 4404) and cost the same time (a hash is computed for unknown names too); names are unique among open sessions (sharing with a name in use is refused: «Ya hay una sesión compartida con ese nombre. Elige otro.», 4403) |
| Wrong passwords per client | 5 within 15 min lock the client 1 min, doubled at every further lock up to 1 h |
| Wrong passwords per name | the same, counted for names that do not exist too; a right password clears only that client's own count, never the name's (so one cannot reset the lock of a name by joining a session of one's own) |
| Hashing | off the event loop (thread pool); the relay's state is only touched on the loop |
| Wrong passwords, all clients | 30 within 1 h lock every join for 15 min |
| Wrong passwords per session | 20 end the session (the sharing device is told to share again with a new password): whoever tries has at most a 20 in 10 000 (0.2 %) chance per session |
| Expiry | when the sharing device leaves; when it has been away 10 min; after 60 min without any message of the sharing device (its page pings every 15 s, so a static slide never ends the session while the device is connected); after 12 h in any case |
| A device away | keeps its place 10 min (`resume` with its token, no password): the tunnel's reset, a Wi-Fi drop, a device that slept or switched app; then it must join again |
| Screen lock | every device in a shared session holds a screen wake lock (`js/device.js` `WakeLock`, re-acquired when the page is shown again); the guide also says to turn auto-lock off |
| Resync | at most one pending per device (a `resync` while one is on its way is ignored); a message of the sharing device dropped for rate marks every device behind, so each gets the whole page again |
| Oversized messages | never sent: a change too large for the relay is replaced by the whole page; a page too large is retried 5 s later; a joined device's value is cut to 4 000 characters |
| Devices | at most 4 joined per session; at most 8 sessions on the server |
| Message size | 4 MB from the sharing device, 16 KB from a joined device (larger: closed, 4413) |
| Message rate | sharing device 60/s average, bursts of 120; joined device 40/s, bursts of 80; the excess is dropped |
| Resume token | 32 random bytes, kept by the server only as a SHA-256 hash, by the page in `sessionStorage` (this tab) so a reload comes back |
| What is stored | nothing: sessions are in memory; what passes through is neither read nor logged; the logs say counts and reasons only («Shared session opened (1 open)», «a device joined (2 devices)», «a wrong name or password», «ended (stopped)») |
| Removing | the sharing device can remove a device (it is told and disconnected); a removed device that knows the password can join again: to keep someone out, stop sharing and share again with a new password |

## Connection loss

The link is the reconnecting socket of #263, with a backoff capped at 1 s
(0.5, 1, 1, 1… s ±20 %: a presentation cannot wait 8 s, and a retry is
one small message), a ping every 15 s and a link silent for 40 s treated
as dead (a tunnel can drop without closing). A device that comes back
within 10 min sends its token (`resume`), never the password; the sharing
device then sends its whole page again, a joined device asks for it. In
the meantime each device keeps working on its own: the sharing device
runs the application as always (its changes are not queued: the snapshot
after the reconnection carries the latest state), a joined device keeps
showing the last state. Every device says it in its indicator, once, in
words: «sin conexión, se reconecta solo»; a joined device whose sharing
device is away says «el dispositivo que comparte se está reconectando».

## What was measured (2026-11-26)

A throwaway development stack (the fake runtime of `tools/demo_video`:
transcriber, orchestrator with fake knowledge and reasoner, no models,
CPU, its own data directory; an invented offer «Barista (oferta
inventada)» with three invented people) behind a stand-in for the public
path written for the check: the site's CSP and `X-Frame-Options: DENY`,
**75–150 ms each way (150–300 ms round trips) with jitter**, order kept,
byte counters, and a «tunnel reset» that closes every socket and refuses
new ones for N seconds. Two headless Chromes: the sharing device
820×1180 (a tablet), the joined one 1600×900 (the projector).
Screenshots of both at the same moments:
`~/.local/share/ragputin/demo-shots/share/` (`01`–`14`).

| Measure | Result |
|---|---|
| Join, navigate (drawer → Ofertas), open the offer, sort its table, type in its search field, the scripted interview (#/live/demo) for a minute | the joined device showed the same page, rows, sort (`aria-sort`), field value and demo state at every check (screenshots `01`–`08`); the demo's text length on both at 15/35/58 s: 858/858, 1203/1198, 1618/1615 characters (the joined device a tick behind) |
| Snapshot of the live page | about 58 KB |
| Scripted interview, one joined device | **169 KB per minute** sent by the sharing device and received by the joined one (233 messages) |
| Typical page (the offer's table) with the presenter's pointer moving 10 s of every 20, a scroll and a sort | **54 KB per minute** each way (527 messages, mostly the pointer) |
| Sharing device acts → joined device shows it | median 285 ms (265–337, n = 6) |
| Joined device acts → sharing device applies it → joined device shows it | median 542 ms (491–608, n = 6) |
| Tunnel reset of 3 / 4 / 5 s, a change made during it | on the joined device 4.0 / 4.9 / 5.9 s after the reset began, nobody typed anything (with an 8 s backoff cap it had been 4.1 / 8.2 / 8.8 s) |
| Reload of the joined device | the mirror back in 0.8 s, no password |
| A watch-only device clicks a link | nothing happened on the sharing device |
| The sharing device set to only watch | its own click did nothing; the joined device navigated it |
| Remove the joined device | it was told («El dispositivo que comparte te ha quitado de la sesión.») and its own application came back |

Page changes (#284, 2026-11-27, the same throwaway stack and stand-in
with 75–150 ms each way; the tablet shares, the desktop joins;
screenshots of both at the same moments in
`~/.local/share/ragputin/demo-shots/share-284/`, `01`–`13`):

| Measure | Result |
|---|---|
| The sharing device goes app → «Estudio» → «Investigación» → app; the joined device shows the new page complete (its data loaded, the charts drawn) | 1.29 / 1.35 / 0.88 s after the click; the old page stays shown meanwhile; its indicator never showed «se está reconectando» (polled every ~50 ms) |
| The same trip from the joined device with «Envía acciones» (sidebar links clicked inside the mirror) | 0.78 / 0.92 / 0.82 s |
| On «Estudio»: scroll (window, both ways, also the joined device's mouse wheel) and a data table opened | followed on the other device (scroll 3720 / 3497 px: anchor-based in the joined device's own layout) |
| On «Investigación»: the contents link «Gráficos» / «Estudios y enlaces», the reference search typed on the joined device | the sharing device went to the anchor; 33 / 33 rows shown on both; 40 SVG charts on both |
| A reference link (another site) clicked on the joined device | opened in a new tab of the joined device; the sharing device stayed on `research.html#estudios` with one tab |
| Join from `study.html` (the device's own «Estudio»), open the panel and stop sharing from `research.html` | joined, mirror shown; after stopping, the joined device got its own «Estudio» back |

Per extra joined device the sharing device sends the same again (the
relay fans out). Not measured: real rag.peer4u.app round trips (the relay
is not deployed until this branch is merged), sound routing with real
voices (no voices in the throwaway stack), a real tablet's touch events
(headless Chrome sent mouse events).
