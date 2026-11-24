// The overview of a finished analysis, written for a person in HR,
// summary first (#230): the session at a glance, one summary per person,
// and every detail (levels, quotes, observations, caveats, events, the
// transcript) behind «Más información», without downloading anything. It
// only shows what the RAG stored (the percentages are the synthesis'
// own); it never ranks people. Technical data stays in the "Detalles
// técnicos" section of the page.

import { h } from "../dom.js";
import { t, lang, percent } from "../i18n.js";
import { competencies } from "../lib/help-analysis.js";
import { clock, transcriptLink } from "../lib/session-analysis.js";
import {
  TIMELINE_TYPES, duration, jobReading, peopleOf, plainSummary, rolesOf, sentenceAnchor, sentenceAt, sessionFacts, timeline,
} from "../lib/analysis-results.js";
import { oneLine } from "../lib/home.js";
import { anyLegacy, personSummary, verdictKey } from "../lib/summary-first.js";
import {
  legacyNote, moreHint, moreInfo, opinionBlock, percentText, personSummaryCard, reliabilityBlock,
} from "./summary-first.js";

/** A technical word with its plain explanation on hover or keyboard focus. */
export function term(text, key) {
  const tip = t(`gl.${key}`);
  return h("span", { class: "term", tabindex: "0", "data-tip": tip },
    text, h("span", { class: "sr-only" }, ` (${tip})`));
}

/** The loading, empty and error states, in words for anyone. */
export function resultsState(kind, { reason = "", onRetry = null } = {}) {
  if (kind === "loading") {
    return h("section", { class: "results-state", "aria-busy": "true" },
      h("p", { role: "status" }, t("res.loading")));
  }
  if (kind === "error") {
    return h("section", { class: "results-state" },
      h("p", { class: "error", role: "alert" }, t("res.error")),
      reason ? h("p", { class: "hint" }, t("res.error_reason", { reason })) : null,
      onRetry ? h("button", { type: "button", onClick: onRetry }, t("res.retry")) : null);
  }
  if (kind === "running") {
    return h("section", { class: "results-state" }, h("p", {}, t("res.running_empty")));
  }
  return h("section", { class: "results-state" },
    h("h2", {}, t("res.empty_title")), h("p", {}, t("res.empty")));
}

/**
 * The whole overview of a session's results, summary first (#230): the
 * session in one paragraph with each person's verdict, then one summary
 * per person (ring, verdict, strengths, improvements, bars), each with
 * its «Más información»; the session's details (facts, notices,
 * participation, events, transcript) in a last «Más información».
 * `onRedo` redoes only the synthesis (offered on results that predate
 * the opinion), or null when it cannot run now.
 */
export function resultsPanel({ id, run, data, onRedo = null, candidate = null }) {
  const names = Object.fromEntries(competencies(lang).map((c) => [c.id, c]));
  const facts = sessionFacts(id, run, data);
  const people = peopleOf(data, names);
  const summaries = people.map(personSummary);
  const kind = people.map((p) => p.opinion?.conversationType).find(Boolean) || facts.type;
  const withJob = Boolean(facts.job);
  // With roles (a live interview, #235) only the candidate is read against the job.
  const roles = rolesOf(data);
  const jobFor = (p) => withJob && jobReading(roles, p.name);
  const lines = new Map(); // sentence id -> its <li> in the transcript
  const transcript = transcriptBlock(id, run, data, people, lines);
  let sessionMore = null;

  // Open the transcript at the sentence of a moment and show it.
  function jump(seconds) {
    const sentence = sentenceAt(data?.sentences, seconds);
    const line = sentence ? lines.get(sentence.id) : null;
    if (!line) return;
    if (sessionMore) sessionMore.open = true;
    transcript.open = true;
    for (const other of lines.values()) other.classList?.remove("flash");
    line.classList?.add("flash");
    line.scrollIntoView?.({ block: "center", behavior: "smooth" });
    line.focus?.({ preventScroll: true });
  }

  const moment = (seconds) => {
    if (typeof seconds !== "number") return null;
    const label = clock(seconds) || "0:00";
    if (!data?.sentences) return h("span", { class: "time" }, label);
    return h("button", {
      type: "button", class: "moment", title: t("res.jump_title"),
      "aria-label": t("res.jump_label", { time: label }), onClick: () => jump(seconds),
    }, `▶ ${label}`);
  };

  // Only one card starts open: the candidate's when the roles are known,
  // else the first person's.
  const first = Math.max(0, people.findIndex((p) => roles[p.name] === "candidate"));
  const cards = people.map((p, i) => personSummaryCard(summaries[i], {
    conversationType: kind, withJob: jobFor(p), id: `person-${p.position}`, open: i === first,
    more: moreInfo(moreHint(summaries[i].more), () => [
      reliabilityBlock(summaries[i]), opinionBlock(p, moment), personDetailBody(p, moment, jobFor(p)),
    ]),
  }));
  // A row of the overview opens that person's card and moves the focus to it.
  const openPerson = (index) => {
    const card = cards[index];
    if (!card) return;
    card.open = true;
    card.scrollIntoView?.({ block: "start", behavior: "smooth" });
    card.querySelector?.("summary")?.focus?.({ preventScroll: true });
  };
  sessionMore = moreInfo(t("sf.session_more_hint"), () => [
    sessionDetails(facts),
    people.length ? participation(people) : null,
    eventsBlock(data, moment),
    transcript,
  ], { id: "sf-session-more", extraClass: "sf-session-more" });

  return h("section", { class: "results sf-results", "aria-labelledby": "res-title" },
    overview(facts, summaries, kind, run, openPerson, onRedo),
    candidate,
    people.length
      ? h("section", { class: "sf-people", "aria-label": t("res.people_title") }, cards)
      : h("p", { class: "hint" }, t("res.no_people")),
    sessionMore);
}

// ---- the session at a glance -------------------------------------------------

function overview(facts, summaries, kind, run, openPerson, onRedo) {
  const meta = [
    facts.recorded ? t("res.recorded_value", facts.recorded) : facts.analysed,
    duration(t, facts.seconds),
    facts.type ? t(`type.${facts.type}`) : "",
  ].filter(Boolean).join(" · ");
  const glance = summaries.some((s) => s.overall)
    ? h("ul", { class: "sf-glance" }, summaries.map((s, i) => h("li", {},
      h("button", { type: "button", class: "sf-glance-row", onClick: () => openPerson(i) },
        h("span", { class: `dot who-${s.colour}`, "aria-hidden": "true" }),
        h("span", { class: "sf-glance-name" }, s.name),
        s.overall && s.overall.percent !== null ? h("span", { class: "sf-glance-score" }, percentText(s.overall.percent)) : null,
        s.overall ? h("span", { class: "sf-glance-verdict" }, t(verdictKey(s.overall.verdict, kind))) : null))))
    : null;
  return h("header", { class: "session-head sf-head" },
    h("h2", { id: "res-title" }, t("res.title")),
    meta || facts.synthetic
      ? h("p", { class: "sf-meta" }, meta, facts.synthetic ? [" ", h("span", { class: "badge warn" }, t("res.synthetic_badge"))] : null)
      : null,
    h("p", { class: "lead" }, plainSummary(t, facts)),
    facts.job ? h("p", { class: "sf-job" }, h("strong", {}, t("sf.job")), " ", oneLine(facts.job)) : null,
    glance,
    anyLegacy(summaries) ? legacyNote(onRedo) : null,
    run?.status === "failed" ? h("p", { class: "notice" }, t("res.partial")) : null,
    h("p", { class: "sf-draft" }, t("sf.draft_short")));
}

// ---- header ----------------------------------------------------------------

function sessionDetails(facts) {
  const items = [];
  const add = (label, value) => {
    if (value === "" || value === null || value === undefined) return;
    items.push(h("div", {}, h("dt", {}, label), h("dd", {}, value)));
  };
  if (facts.recorded) add(t("res.recorded"), t("res.recorded_value", facts.recorded));
  add(t("res.analysed"), facts.analysed);
  add(t("res.length"), duration(t, facts.seconds));
  add(t("res.participants"), String(facts.people));
  add(t("res.type"), facts.type ? t(`type.${facts.type}`) : t("res.type_unknown"));
  add(t("res.status"), h("span", { class: `badge status-${facts.status}` }, t(`res.status_${facts.status}`)));
  return h("div", { class: "stack" },
    facts.goal ? h("p", { class: "hint" }, t("res.goal", { goal: facts.goal })) : null,
    facts.job ? h("p", { class: "hint" }, t("res.job", { job: facts.job })) : null,
    h("dl", { class: "facts" }, items),
    h("div", { class: "draft-notice", role: "note" },
      h("strong", {}, t("res.draft_title")), " ", t("res.draft_text"),
      facts.synthetic ? h("p", {}, h("span", { class: "badge warn" }, t("res.synthetic_badge")), " ", t("sa.synthetic_data")) : null,
      h("p", { class: "hint" }, t("res.fairness"))));
}

// A stacked bar of each person's share of the words, in speaking order:
// a neutral fact, with the same numbers in text next to it.
function participation(people) {
  const known = people.filter((p) => typeof p.share === "number");
  if (!known.length) return null;
  return h("section", { class: "block", "aria-labelledby": "res-share" },
    h("h2", { id: "res-share" }, term(t("res.share_title"), "share")),
    h("div", { class: "share-bar", "aria-hidden": "true" },
      known.map((p) => {
        // Set through the CSS object model: no inline style attribute.
        const seg = h("span", { class: `seg who-${p.colour}` });
        if (seg.style) seg.style.flexGrow = String(Math.max(p.share, 0.001));
        return seg;
      })),
    h("ul", { class: "share-legend" }, known.map((p) => h("li", {},
      h("span", { class: `dot who-${p.colour}`, "aria-hidden": "true" }), p.name, " ", h("strong", {}, percent(p.share, 0))))),
    h("p", { class: "hint" }, t("res.share_note")));
}

// ---- one card per person ---------------------------------------------------

function certaintyBadges(counts) {
  return h("ul", { class: "certainty-list" }, ["firm", "provisional", "insufficient"].map((kind) =>
    h("li", {}, h("span", { class: `badge cert-${kind}` }, String(counts[kind])), " ", term(t(`res.cert_${kind}_n`), kind))));
}

// What the reports list for a person and how sure each level is (inside «Más información»).
function reportFacts(p) {
  const names = (list) => (list.length ? list.join(", ") : t("res.none_yet"));
  return h("div", { class: "stack tight" },
    h("p", {}, h("strong", {}, term(t("res.strengths"), "strengths")), " ", names(p.strengths)),
    h("p", {}, h("strong", {}, term(t("res.to_develop"), "to_develop")), " ", names(p.toDevelop)),
    certaintyBadges(p.certainty),
    p.confidence
      ? h("p", { class: "hint" }, term(t("res.profile_confidence"), "confidence"), " ",
        h("span", { class: `badge conf-${p.confidence}` }, t(`sa.syn_conf_${p.confidence}`)))
      : null);
}

// ---- the detail of a person ------------------------------------------------

function quoteItem(q, moment) {
  return h("li", { class: q.positive ? "quote" : "quote negative" },
    moment(q.start), " ",
    h("q", {}, q.text),
    q.positive ? null : h("span", { class: "hint" }, ` ${t("res.quote_negative")}`));
}

function skillItem(skill, moment) {
  const level = skill.point === null
    ? t("res.level_none")
    : t(skill.certainty === "provisional" ? "res.level_about" : "res.level_of", { rating: skill.point });
  return h("details", { class: `skill cert-${skill.certainty}` },
    h("summary", {},
      h("span", { class: "skill-name" }, skill.name),
      h("span", { class: `badge cert-${skill.certainty}` }, t(`res.cert_${skill.certainty}`)),
      h("span", { class: "skill-level" }, level)),
    h("div", { class: "stack tight" },
      skill.point !== null ? h("p", {}, h("strong", {}, t("res.meaning")), " ", t(`res.scale_${skill.point}`)) : null,
      skill.certainty === "provisional"
        ? h("p", { class: "hint" }, t("res.provisional_note", { confidence: t(`sa.confidence_${skill.confidence}`) }))
        : null,
      skill.certainty === "insufficient" ? h("p", { class: "hint" }, t("res.insufficient_note")) : null,
      skill.definition ? h("p", { class: "hint" }, h("strong", {}, t("res.what_it_is")), " ", skill.definition) : null,
      skill.rationale ? h("p", {}, h("strong", {}, t("res.why_level")), " ", skill.rationale) : null,
      skill.quotes.length
        ? [h("h5", {}, term(t("res.quotes_title", { count: skill.quotes.length }), "evidence")),
          h("ul", { class: "quotes" }, skill.quotes.map((q) => quoteItem(q, moment)))]
        : null,
      skill.observations.length
        ? [h("h5", {}, term(t("res.observations_title", { count: skill.observations.length }), "observation")),
          h("ul", { class: "quotes" }, skill.observations.map((q) => quoteItem(q, moment)))]
        : null));
}

function profileBlock(p, moment, withJob) {
  const profile = p.profile;
  if (!profile) return h("p", { class: "hint" }, t("res.no_profile"));
  if (profile.status === "skipped") return h("p", { class: "hint" }, t("res.profile_skipped"));
  const point = (item) => h("li", {},
    h("p", {}, item.text),
    (item.cites ?? []).length
      ? h("ul", { class: "quotes" }, item.cites.map((c) => h("li", { class: "quote" }, moment(c.start), " ", h("q", {}, c.text))))
      : null);
  const list = (items) => (items?.length ? h("ul", { class: "points" }, items.map(point)) : h("p", { class: "hint" }, t("sa.syn_none")));
  return h("div", { class: "profile stack tight" },
    h("p", { class: "hint" }, t("sa.syn_note")),
    h("h4", {}, t("sa.syn_strengths")), list(profile.strengths),
    h("h4", {}, t(withJob ? "sa.syn_fit" : "sa.syn_tasks")), list(profile.fit),
    h("h4", {}, t("sa.syn_unobserved")),
    profile.unobserved?.length
      ? h("ul", {}, profile.unobserved.map((n) => h("li", {}, n)))
      : h("p", { class: "hint" }, t("sa.syn_unobserved_none")),
    profile.questions?.length
      ? [h("h4", {}, t("sa.syn_questions")), h("ul", {}, profile.questions.map((q) => h("li", {}, q)))]
      : null);
}

function participationTable(p) {
  const e = p.events ?? {};
  const rows = [
    ["res.share", p.share === null ? null : percent(p.share, 0)],
    ["res.turns", p.turns],
    ["res.words", p.words],
    ["res.speaking_time", duration(t, p.seconds)],
    ["res.questions", p.questions],
    ["res.overlaps", p.overlaps],
    ["res.proposals_made", e.proposals_made],
    ["res.proposals_accepted", e.proposals_accepted],
    ["res.commitments", e.commitments],
    ["res.decisions_involved", e.decisions_involved],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  return h("div", { class: "table-wrap" }, h("table", { class: "data facts-table" },
    h("tbody", {}, rows.map(([key, value]) => h("tr", {}, h("th", { scope: "row" }, t(key)), h("td", {}, String(value)))))));
}

// Everything else about a person, for «Más información»: the reports'
// lists and certainty, the explained profile, every competency with its
// quotes, and the participation figures.
function personDetailBody(p, moment, withJob) {
  return h("div", { class: "stack" },
    p.why.length ? h("p", { class: "notice" }, t("res.few_levels")) : null,
    reportFacts(p),
    h("details", { class: "sub" },
      h("summary", {}, h("h4", {}, t("res.profile_title"))), profileBlock(p, moment, withJob)),
    h("details", { class: "sub" },
      h("summary", {}, h("h4", {}, t("res.skills_title", { count: p.skills.length }))),
      h("p", { class: "hint" }, t("res.skills_hint")),
      p.skills.length ? h("div", { class: "skills" }, p.skills.map((s) => skillItem(s, moment))) : h("p", { class: "hint" }, t("res.no_skills"))),
    h("details", { class: "sub" },
      h("summary", {}, h("h4", {}, t("res.participation_title"))), participationTable(p)));
}

// ---- events and transcript -------------------------------------------------

function eventsBlock(data, moment) {
  if (!data?.events) return null;
  const all = timeline(data.events);
  const list = h("ol", { class: "timeline" });
  const select = h("select", { id: "res-event-type" },
    h("option", { value: "" }, t("res.events_all", { count: all.length })),
    TIMELINE_TYPES.map((type) => {
      const n = all.filter((e) => e.type === type).length;
      return n ? h("option", { value: type }, `${t(`sa.event_${type}`)} (${n})`) : null;
    }));
  const draw = () => {
    const shown = select.value ? all.filter((e) => e.type === select.value) : all;
    list.replaceChildren(...shown.map((e) => h("li", { class: `event ev-${e.type}` },
      moment(e.start), " ",
      h("span", { class: "badge" }, t(`sa.event_${e.type}`)), " ",
      e.by ? h("strong", {}, e.by) : null, e.by ? ": " : "",
      e.text,
      e.owner && e.owner !== e.by ? h("span", { class: "hint" }, ` ${t("res.event_owner", { name: e.owner })}`) : null,
      e.deadline ? h("span", { class: "hint" }, ` ${t("res.event_deadline", { when: e.deadline })}`) : null)));
  };
  select.addEventListener("change", draw);
  draw();
  return h("details", { class: "block events-block" },
    h("summary", {}, h("h2", {}, term(t("res.events_title"), "events"))),
    h("p", { class: "hint" }, t("res.events_hint")),
    h("div", { class: "field no-print" }, h("label", { for: "res-event-type" }, t("res.events_filter")), select),
    all.length ? list : h("p", { class: "hint" }, t("sa.no_events")));
}

function transcriptBlock(id, run, data, people, lines) {
  const sentences = data?.sentences?.sentences ?? [];
  const colours = new Map(people.map((p) => [p.name, p.colour]));
  const items = sentences.map((s) => {
    const li = h("li", { id: sentenceAnchor(s), tabindex: "-1" },
      h("span", { class: "time" }, clock(s.start)), " ",
      h("span", { class: `who who-${colours.get(s.speaker) ?? 0}` }, s.speaker), " ",
      h("span", {}, s.text));
    lines.set(s.id, li);
    return li;
  });
  return h("details", { class: "block transcript-block", id: "an-transcript" },
    h("summary", {}, h("h2", {}, t("res.transcript_title", { count: sentences.length }))),
    run?.synthetic ? null : h("p", { class: "hint no-print" }, h("a", { href: transcriptLink(id) }, t("res.transcript_page"))),
    sentences.length ? h("ol", { class: "transcript-lines" }, items) : h("p", { class: "hint" }, t("res.no_transcript")));
}
