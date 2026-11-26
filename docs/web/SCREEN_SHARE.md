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
is recording cannot join). The sound plays on **one** device, chosen by
the sharing device in the panel («El sonido sale por», default the
sharing device). When it is a joined device (`js/share-sound.js`):

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
| Names | compared trimmed, one space, without case; not enumerable: «no such name» and «wrong password» are the same answer (`no_match`, 4404) and cost the same time (a hash is computed for unknown names too); two sessions may have the same name (a join matches name *and* password) |
| Wrong passwords per client | 5 within 15 min lock the client 1 min, doubled at every further lock up to 1 h |
| Wrong passwords per name | the same, counted for names that do not exist too |
| Wrong passwords, all clients | 30 within 1 h lock every join for 15 min |
| Wrong passwords per session | 20 end the session (the sharing device is told to share again with a new password): whoever tries has at most a 20 in 10 000 (0.2 %) chance per session |
| Expiry | when the sharing device leaves; when it has been away 2 min; after 60 min without any change or action; after 12 h in any case |
| A device away | keeps its place 2 min (`resume` with its token, no password); then it must join again |
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
within 2 min sends its token (`resume`), never the password; the sharing
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

Per extra joined device the sharing device sends the same again (the
relay fans out). Not measured: real rag.peer4u.app round trips (the relay
is not deployed until this branch is merged), sound routing with real
voices (no voices in the throwaway stack), a real tablet's touch events
(headless Chrome sent mouse events).
