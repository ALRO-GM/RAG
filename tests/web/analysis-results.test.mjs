// The analysis results for a non-technical reader: the pure helpers that
// rearrange the stored results (no score is computed), the Claude model
// choice, the dictionaries, and the overview drawn in a stand-in DOM.
import test from "node:test";
import assert from "node:assert/strict";
import {
  certaintyCounts, certaintyOf, doneResultSteps, duration, eventCounts, hasResults, peopleOf, plainSummary,
  progressOf, recordedAt, resultsKey, scalePoint, sentenceAt, sentenceAnchor, sessionFacts, sessionLength, skillsOf,
  opinionOf, timeline,
} from "../../transcriber/adapters/inbound/web/js/lib/analysis-results.js";
import {
  CLAUDE_MODEL, JUDGE_STEPS, MODEL_CHOICES, chosenModels, startBody, validModel,
} from "../../transcriber/adapters/inbound/web/js/lib/session-analysis.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";
import { en } from "../../transcriber/adapters/inbound/web/js/i18n/en.js";
import { translate } from "../../transcriber/adapters/inbound/web/js/lib/i18n.js";

const t = (key, params) => translate({ es, en }, "es", key, params);

const run = {
  session_id: "2026-11-15-143902-charla", status: "done", synthetic: false, active: false,
  started_at: "2026-11-16T10:04:18Z", finished_at: "2026-11-16T10:15:30Z",
  steps: [
    { name: "sentences", status: "done" }, { name: "interaction", status: "done" },
    { name: "events", status: "done" }, { name: "assessment", status: "done" },
    { name: "reports", status: "done" }, { name: "synthesis", status: "unavailable" },
  ],
};

const data = {
  sentences: {
    conversation_type: "reunion_decision", conversation_goal: "Decidir el verano",
    sentences: [
      { id: "t0.s0", speaker: "Ana", text: "Empezamos.", start: 0, end: 2 },
      { id: "t1.s0", speaker: "Luis", text: "¿Y el plazo?", start: 3, end: 5 },
      { id: "t2.s0", speaker: "Ana", text: "El jueves.", start: 61.2, end: 63.4 },
    ],
  },
  interaction: {
    turns: 3,
    per_speaker: [
      { speaker: "Ana", turns: 2, words: 4, word_share: 0.6, questions: 0, seconds: 4.2, overlapping_starts: 0 },
      { speaker: "Luis", turns: 1, words: 3, word_share: 0.4, questions: 1, seconds: 2, overlapping_starts: 1 },
    ],
  },
  events: {
    events: [
      { id: "e2", type: "decision", start: 61.2, order: 3, by: "Ana", text: "El jueves.", what: "Entregar el jueves", people: ["Ana"] },
      { id: "e1", type: "question_answered", start: 3, order: 2, by: "Luis", text: "¿Y el plazo?", people: ["Luis", "Ana"] },
      { id: "e3", type: "action_item", start: 61.2, order: 4, by: "Ana", text: "Lo hago yo", owner: "Ana", deadline: "jueves" },
      { id: "e4", type: "unknown_kind", start: 1 },
    ],
    summary: { per_person: [{ speaker: "Ana", proposals_made: 1, commitments: 1 }] },
  },
  assessment: {
    conversation_type: "reunion_decision",
    scorecards: [
      {
        participant: "Ana",
        assessments: [
          { competency_id: "COORD", status: "rated", rating: 4, rationale: "Fija plazos.",
            evidence: [{ quote: "El jueves.", start: 61.2, turn: 2, polarity: "positive", indicator_id: "COORD.P3" },
              { quote: "Empezamos.", start: 0, turn: 0, polarity: "positive", indicator_id: "COORD.P1" }] },
          { competency_id: "LISTEN", status: "provisional", rating: null, estimate: { rating: 3, confidence: "low" },
            evidence: [{ quote: "Empezamos.", start: 0, turn: 0, polarity: "negative", indicator_id: "LISTEN.N1" }] },
          { competency_id: "FEEDB", status: "insufficient_evidence", rating: null },
        ],
      },
      { participant: "Luis", assessments: [{ competency_id: "COORD", status: "insufficient_evidence", rating: null }] },
    ],
  },
  reports: {
    people: [
      { position: 1, name: "Ana", strengths: ["Coordinación"], to_improve: [], why: [] },
      { position: 2, name: "Luis", strengths: [], to_improve: [], why: ["little_participation"] },
    ],
  },
};

test("the session id gives the recording date; a simulation's does not", () => {
  assert.deepEqual(recordedAt("2026-11-15-143902-charla"), { date: "2026-11-15", time: "14:39" });
  assert.deepEqual(recordedAt("2026-11-18-092345"), { date: "2026-11-18", time: "09:23" });
  assert.equal(recordedAt("sim-reunion-verano"), null);
  assert.equal(recordedAt("2026-13-15-143902-x"), null);
});

test("length, durations and the plain one-line summary", () => {
  assert.equal(sessionLength(data.sentences), 63.4);
  assert.equal(sessionLength({ sentences: [] }), null);
  assert.equal(duration(t, 867.14), "14 min 27 s");
  assert.equal(duration(t, 45), "45 s");
  assert.equal(duration(t, null), "");
  const facts = sessionFacts(run.session_id, run, data);
  assert.equal(facts.people, 2);
  assert.equal(facts.type, "reunion_decision");
  assert.equal(facts.analysed, "2026-11-16");
  assert.equal(facts.status, "done");
  assert.equal(plainSummary(t, facts), "Participaron 2 personas durante 1 min 3 s; salieron 1 decisión, 1 tarea asignada.");
  const quiet = { ...facts, people: 1, seconds: null, events: eventCounts(null) };
  assert.equal(plainSummary(t, quiet), "Participó 1 persona.");
});

test("certainty: firm, provisional or insufficient, never a new score", () => {
  assert.equal(certaintyOf({ rating: 2 }), "firm");
  assert.equal(certaintyOf({ status: "provisional", estimate: { rating: 3 } }), "provisional");
  assert.equal(certaintyOf({ status: "provisional", estimate: {} }), "insufficient");
  assert.equal(certaintyOf({ status: "insufficient_evidence", rating: null }), "insufficient");
  assert.equal(scalePoint(3.6), 4);
  assert.equal(scalePoint(null), null);
  const skills = skillsOf(data.assessment.scorecards[0], { COORD: { name: "Coordinación", definition: "Organiza." } });
  assert.deepEqual(skills.map((s) => [s.id, s.name, s.certainty, s.rating]), [
    ["COORD", "Coordinación", "firm", 4], ["LISTEN", "LISTEN", "provisional", 3], ["FEEDB", "FEEDB", "insufficient", null],
  ]);
  assert.deepEqual(skills[0].quotes.map((q) => q.start), [0, 61.2], "quotes in spoken order");
  assert.equal(skills[1].quotes[0].positive, false);
  assert.deepEqual(certaintyCounts(skills), { firm: 1, provisional: 1, insufficient: 1 });
});

test("people come in speaking order with their stored facts, never sorted by a score", () => {
  const people = peopleOf(data);
  assert.deepEqual(people.map((p) => [p.name, p.colour, p.share, p.overlaps, p.quotes]), [
    ["Ana", 0, 0.6, 0, 3], ["Luis", 1, 0.4, 1, 0],
  ]);
  assert.deepEqual(people[0].strengths, ["Coordinación"]);
  assert.deepEqual(people[1].why, ["little_participation"]);
  assert.equal(people[0].events.commitments, 1);
  const reversed = peopleOf({ ...data, interaction: { per_speaker: [...data.interaction.per_speaker].reverse() } });
  assert.deepEqual(reversed.map((p) => p.name), ["Luis", "Ana"]);
});

test("the timeline keeps known events in time order; a moment finds its sentence", () => {
  const rows = timeline(data.events);
  assert.deepEqual(rows.map((e) => e.id), ["e1", "e2", "e3"]);
  assert.equal(rows[1].text, "Entregar el jueves");
  assert.equal(rows[2].deadline, "jueves");
  assert.deepEqual(timeline(data.events, ["decision"]).map((e) => e.id), ["e2"]);
  assert.equal(eventCounts(data.events).decision, 1);
  assert.equal(sentenceAt(data.sentences, 62).id, "t2.s0");
  assert.equal(sentenceAt(data.sentences, 2.9).id, "t1.s0");
  assert.equal(sentenceAnchor({ id: "t2.s0" }), "line-t2_s0");
});

test("which results to read, when to read them again, and the progress", () => {
  assert.deepEqual(doneResultSteps(run), ["sentences", "interaction", "events", "assessment", "reports"]);
  assert.equal(hasResults(run), true);
  assert.equal(hasResults({ steps: [{ name: "sentences", status: "done" }] }), false);
  assert.equal(hasResults(null), false);
  assert.notEqual(resultsKey(run), resultsKey({ ...run, finished_at: "later" }));
  assert.equal(resultsKey(null), "");
  const running = { active: true, steps: [{ name: "sentences", status: "done" }, { name: "interaction", status: "running" }] };
  assert.deepEqual(progressOf(running), { done: 1, total: 9, current: "interaction", active: true });
});

test("the Claude model choice: aliases, full ids, nothing that could be an option", () => {
  assert.deepEqual(JUDGE_STEPS, ["confirmation", "assessment", "synthesis"]);
  assert.deepEqual(MODEL_CHOICES, ["", "haiku", "sonnet", "opus", "custom"]);
  for (const ok of ["haiku", "opus[1m]", "claude-haiku-4-5", "claude-opus-4.1"]) assert.ok(validModel(ok), ok);
  for (const bad of ["", "-p", "--tools", "opus; id", "OPUS", "gpt-4o", "claude-", "$(id)", `claude-${"x".repeat(70)}`]) {
    assert.equal(validModel(bad), false, bad);
  }
  assert.ok(CLAUDE_MODEL.test("sonnet"));
  const chosen = chosenModels({
    confirmation: { choice: "haiku" }, assessment: { choice: "" },
    synthesis: { choice: "custom", custom: " claude-sonnet-4-5 " },
  });
  assert.deepEqual(chosen, { models: { confirmation: "haiku", synthesis: "claude-sonnet-4-5" }, invalid: [] });
  assert.deepEqual(chosenModels({ synthesis: { choice: "custom", custom: "rm -rf" } }).invalid, ["synthesis"]);
  assert.deepEqual(startBody({ claudeModels: { confirmation: "haiku", assessment: "", acts: "opus" } }), {
    restart: false, synthetic: false, claude_models: { confirmation: "haiku" },
  });
  assert.deepEqual(startBody({ claudeModels: {} }), { restart: false, synthetic: false });
});

test("every new text exists in both languages and never ranks people", () => {
  const keys = Object.keys(es).filter((k) => /^(res|gl|sf)\./.test(k) || /^sa\.(model|tradeoff|controls|tech)/.test(k));
  assert.ok(keys.length > 100);
  for (const key of keys) {
    assert.ok(en[key], `missing in en: ${key}`);
    for (const text of [es[key], en[key]]) {
      assert.doesNotMatch(text, /\b(mejor|peor|mejores|peores|best|worst|better|worse|ranking|top)\b/i, key);
    }
  }
  for (const key of Object.keys(en)) if (/^(res|gl|sf)\./.test(key)) assert.ok(es[key], `missing in es: ${key}`);
  for (const n of [1, 2, 3, 4, 5]) assert.ok(es[`res.scale_${n}`] && en[`res.scale_${n}`]);
  for (const key of ["res.draft_title", "res.draft_text", "res.fairness"]) assert.ok(es[key] && en[key], key);
});

test("the measured trade-off quotes the comparison's own figures", () => {
  for (const figure of ["77,5", "91,7", "72,1", "76,4", "0,046", "1,815"]) assert.ok(es["sa.tradeoff_haiku"].includes(figure), figure);
  for (const figure of ["77.5", "91.7", "72.1", "76.4", "0.046", "1.815"]) assert.ok(en["sa.tradeoff_haiku"].includes(figure), figure);
  assert.match(es["sa.tradeoff_source"], /MODEL_COMPARISON\.md/);
});

// ---- the overview in a stand-in DOM ----------------------------------------

class FakeNode {}
class FakeText extends FakeNode {
  constructor(text) { super(); this.textContent = text; }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    for (const key of ["id", "type", "value", "className", "textContent", "href", "htmlFor", "title"]) this[key] = "";
    this.open = false;
    this.hidden = false;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { (this.listeners[type] ??= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  all() { return [this, ...this.children.flatMap((c) => (c.all ? c.all() : []))]; }
  text() { return this.children.map((c) => (c.text ? c.text() : c.textContent)).join(""); }
}

async function withDom(fn) {
  const saved = { document: globalThis.document, Node: globalThis.Node };
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    createTextNode: (text) => new FakeText(text),
  };
  globalThis.Node = FakeNode;
  try {
    return await fn();
  } finally {
    Object.assign(globalThis, saved);
  }
}

test("the overview shows the summary, one card with its «Más información» per person, the caveats behind it", async () => {
  await withDom(async () => {
    const { resultsPanel, resultsState } = await import("../../transcriber/adapters/inbound/web/js/views/analysis-results.js");
    // Node 21+ has a global navigator ("en-US"): fix the language first.
    const { setLanguage } = await import("../../transcriber/adapters/inbound/web/js/i18n.js");
    setLanguage("es");
    const panel = resultsPanel({ id: run.session_id, run, data });
    const nodes = panel.all();
    const cards = nodes.filter((n) => n.tagName === "ARTICLE");
    assert.equal(cards.length, 2);
    const more = nodes.filter((n) => n.tagName === "DETAILS" && n.className === "sf-more");
    assert.equal(more.length, 2);
    assert.ok(more.every((d) => d.open === false));
    const text = panel.text();
    // The caveats are all there, inside the session's closed «Más información».
    const session = nodes.find((n) => n.id === "sf-session-more");
    assert.equal(session.open, false);
    for (const key of ["res.draft_title", "res.draft_text", "res.fairness"]) assert.ok(session.text().includes(es[key]), key);
    assert.ok(text.includes(es["sf.draft_short"]));
    // Every quote has a button that opens the transcript at its sentence.
    const transcript = nodes.find((n) => n.id === "an-transcript");
    assert.equal(transcript.open, false);
    const jump = nodes.find((n) => n.tagName === "BUTTON" && n.className === "moment");
    jump.listeners.click[0]();
    assert.equal(transcript.open, true);
    assert.equal(session.open, true);
    // Technical names never reach the overview.
    assert.doesNotMatch(text, /claude-code|tokens/);
    assert.ok(resultsState("loading").attributes["aria-busy"]);
    const error = resultsState("error", { reason: "x", onRetry: () => {} }).all();
    assert.ok(error.some((n) => n.tagName === "BUTTON"));
  });
});

test("the opinion of a person: overall, skills, top picks; old files have none", () => {
  assert.deepEqual(opinionOf(null).topStrengths, []);
  assert.equal(opinionOf({ status: "done" }).overall, null);
  const profile = {
    participant: "Ana", conversation_type: "coaching",
    overall: { score: 3.8, maximum: 5, verdict: "strong", line: "Bien.", frame: "q", confidence: "low", caveats: ["no_quotes"] },
    skills: [
      { competency_id: "a", name: "A", score: 4, verdict: "strong", conclusion: "ok", basis: "firm", confidence: "high", caveats: [], cites: [{ text: "hola", start: 3 }] },
      { competency_id: "b", name: "B", score: 2, verdict: "weak", conclusion: "poco", basis: "participation", caveats: ["no_quotes"] },
      { score: "x" },
    ],
    top_strengths: ["a", "zz"], top_improvements: ["b"],
  };
  const o = opinionOf(profile);
  assert.equal(o.overall.verdict, "strong");
  assert.equal(o.overall.percent, 70);
  assert.equal(opinionOf({ overall: { score: 3.5, maximum: 4, minimum: 0 } }).overall.percent, 87.5);
  assert.equal(opinionOf({ overall: { score: 4.4, maximum: 5, percent: 88.7 } }).overall.percent, 88.7);
  assert.deepEqual(o.overall.caveats, ["no_quotes"]);
  assert.deepEqual(o.topStrengths.map((k) => k.id), ["a"]);
  assert.deepEqual(o.topImprovements.map((k) => k.id), ["b"]);
  assert.equal(o.skills[0].cites[0].start, 3);
  assert.equal(o.conversationType, "coaching");
  assert.equal(o.skills[1].confidence, "low");
  const people = peopleOf({
    ...data, synthesis: { people: [profile] },
    assessment: { scorecards: [{ participant: "Ana", assessments: [{ competency_id: "a", status: "insufficient_evidence" }] }] },
  });
  const ana = people.find((p) => p.name === "Ana");
  assert.equal(ana.opinion.overall.score, 3.8);
  assert.equal(ana.skills[0].opinion.verdict, "strong");
  assert.equal(ana.skills[0].certainty, "insufficient");
});
