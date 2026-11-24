// The live conversation keeps its newest line in view (#233 review): it
// follows new lines while the box is at (or near) its bottom and stops
// following once the user scrolled back to read; coming back down follows
// again. The box itself has a fixed height tied to the viewport.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FOLLOW_SLACK_PX, followsNewest } from "../../transcriber/adapters/inbound/web/js/lib/live.js";

const box = (scrollHeight, scrollTop, clientHeight) => ({ scrollHeight, scrollTop, clientHeight });

test("a box at its bottom, or that does not scroll yet, follows new lines", () => {
  assert.equal(followsNewest(box(1200, 700, 500)), true);
  assert.equal(followsNewest(box(300, 0, 500)), true);
});

test("near the bottom still follows; scrolled back does not", () => {
  assert.equal(followsNewest(box(1200, 700 - FOLLOW_SLACK_PX, 500)), true);
  assert.equal(followsNewest(box(1200, 700 - FOLLOW_SLACK_PX - 1, 500)), false);
  assert.equal(followsNewest(box(1200, 0, 500)), false);
});

test("a missing box follows (nothing to keep the user's place in)", () => {
  assert.equal(followsNewest(null), true);
  assert.equal(followsNewest(box(1200, 0, 500), 2000), true);
});

test("the conversation box is tied to the viewport and follows the newest line", () => {
  const css = readFileSync(new URL("../../transcriber/adapters/inbound/web/style.css", import.meta.url), "utf8");
  const view = readFileSync(new URL("../../transcriber/adapters/inbound/web/js/views/live.js", import.meta.url), "utf8");
  assert.match(css, /\.lv-lines \{[^}]*height: clamp\(14rem, 55dvh, 32rem\)[^}]*scrollbar-width: none/);
  assert.match(css, /\.lv-lines \{[^}]*height: clamp\(18rem, calc\(100dvh - 20rem\), 40rem\)/);
  assert.match(view, /followLines = followsNewest\(lines\)/);
  assert.match(view, /if \(followLines\) lines\.scrollTop = lines\.scrollHeight/);
});
