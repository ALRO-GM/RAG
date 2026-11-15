// Names found in what people said (#102): the transcript page's pure helpers.

import test from "node:test";
import assert from "node:assert/strict";
import {
  NAME_REASONS, nameRows, reasonKey, searchFinished,
} from "../../transcriber/adapters/inbound/web/js/lib/names.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";

const speakers = [
  { id: "S1", name: null, enrolled: false },
  { id: "S2", name: "Ana", enrolled: false },
  { id: "S3", name: null, enrolled: false },
];

test("rows carry the label and the name shown, suggestions in order", () => {
  const names = {
    auto: [{ speaker: "S2", name: "Ana", quote: "soy Ana", start: 3 }, { speaker: "S9", name: "X", quote: "q", start: 0 }],
    suggestions: [
      { speaker: "S3", name: "Pedro", kind: "addressed", start: 9, reason: "addressed" },
      { speaker: "S1", name: "Marta", kind: "addressed", start: 12, reason: "addressed" },
      { speaker: "S1", name: "Luz", kind: "self", start: 20, reason: "unsure" },
      { speaker: "S1", name: "Eva", kind: "addressed", start: 2, reason: "addressed" },
      { speaker: "S7", name: "Nadie", kind: "self", start: 1, reason: "unsure" },
    ],
  };
  const rows = nameRows(names, speakers);
  assert.deepEqual(rows.auto.map((a) => [a.label, a.shown]), [["S2", "Ana"]]);
  assert.deepEqual(rows.suggestions.map((s) => s.name), ["Luz", "Eva", "Marta", "Pedro"]);
  assert.equal(rows.suggestions[0].shown, "S1");
});

test("no record and missing lists give no rows", () => {
  assert.deepEqual(nameRows(null, speakers), { auto: [], suggestions: [] });
  assert.deepEqual(nameRows({}, speakers), { auto: [], suggestions: [] });
});

test("a search finished when it was running and is not any more", () => {
  assert.equal(searchFinished({ running: true }, { running: false }), true);
  assert.equal(searchFinished({ running: false }, { running: false }), false);
  assert.equal(searchFinished(null, { running: false }), false);
  assert.equal(searchFinished({ running: true }, { running: true }), false);
});

test("every reason has a sentence in both languages", () => {
  for (const reason of [...NAME_REASONS, "something-new"]) {
    const key = reasonKey(reason);
    assert.ok(es[key], key);
    assert.ok(en[key], key);
  }
  assert.equal(reasonKey("bogus"), "names.reason.other");
  for (const key of ["names.title", "names.search", "names.undo", "names.accept", "stage.names"]) {
    assert.ok(es[key] && en[key] && es[key] !== en[key], key);
  }
});
