// Names found in what people said (#102): the transcript page's pure helpers.

import test from "node:test";
import assert from "node:assert/strict";
import {
  BUSY_RETRY_MS, NAME_REASONS, groupByVoice, nameRows, publishWhenFree, reasonKey, searchFinished,
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
  assert.deepEqual(nameRows({}, speakers), { auto: [], suggestions: [], groups: [] });
});

test("suggestions are grouped per voice, strongest option first (#116)", () => {
  const names = {
    suggestions: [
      { speaker: "S3", name: "Ruiz", kind: "addressed", start: 40, reason: "few_votes", count: 6, votes: 5.5,
        spellings: ["Ruis"] },
      { speaker: "S1", name: "Gil", kind: "addressed", start: 9, reason: "close", count: 2, votes: 3 },
      { speaker: "S1", name: "Vera", kind: "addressed", start: 3, reason: "few_votes", count: 4, votes: 4.5 },
      { speaker: "S3", name: "Gil", kind: "addressed", start: 2, reason: "few_votes", count: 1, votes: 1 },
    ],
  };
  const { groups } = nameRows(names, speakers);
  assert.deepEqual(groups.map((g) => [g.label, g.shown, g.options.map((o) => o.name)]), [
    ["S1", "S1", ["Vera", "Gil"]],
    ["S3", "S3", ["Ruiz", "Gil"]],
  ]);
  assert.deepEqual(groups[1].options[0].spellings, ["Ruis"]);
});

test("ties fall back to pieces, self-introductions and time", () => {
  const rows = [
    { label: "S1", shown: "S1", name: "B", kind: "addressed", start: 1 },
    { label: "S1", shown: "S1", name: "A", kind: "self", start: 5 },
    { label: "S1", shown: "S1", name: "C", kind: "addressed", start: 0, count: 3 },
  ];
  assert.deepEqual(groupByVoice(rows)[0].options.map((o) => o.name), ["C", "A", "B"]);
  assert.deepEqual(groupByVoice([]), []);
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
  assert.equal(es["names.busy"], "Buscando los nombres de quién habla…");
  for (const key of ["names.title", "names.search", "names.undo", "names.accept", "stage.names", "names.busy",
    "names.group", "names.other", "names.spellings", "names.reason.few_votes", "names.reason.close"]) {
    assert.ok(es[key] && en[key] && es[key] !== en[key], key);
  }
});

function busyError() {
  return Object.assign(new Error("busy"), { code: "busy" });
}

test("the publish waits while names are searched, then goes", async () => {
  const log = [];
  const answers = [busyError(), null];
  const running = [true, false, false];
  const sent = await publishWhenFree(async () => {
    log.push("publish");
    const answer = answers.shift();
    if (answer) throw answer;
  }, {
    busy: async () => running.shift(),
    onWait: () => log.push("wait"),
    wait: async (ms) => log.push(`sleep ${ms}`),
  });
  assert.equal(sent, true);
  assert.deepEqual(log, [
    "wait", `sleep ${BUSY_RETRY_MS}`, // GET /names: running
    "publish", "wait", `sleep ${BUSY_RETRY_MS}`, // 409 busy
    "publish",
  ]);
});

test("other errors are thrown; leaving the page stops the wait", async () => {
  const gone = Object.assign(new Error("gone"), { code: "not_found" });
  await assert.rejects(publishWhenFree(async () => { throw gone; }), (e) => e === gone);
  let left = false;
  const sent = await publishWhenFree(async () => { throw busyError(); }, {
    busy: async () => { throw new Error("names unavailable"); },
    stopped: () => left,
    wait: async () => { left = true; },
  });
  assert.equal(sent, false);
});
