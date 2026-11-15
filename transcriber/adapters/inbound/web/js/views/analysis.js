// "Análisis" of a session (#/analysis/<id>): start the RAG's whole analysis,
// watch the steps advance (status, time, judge, fallback) and open what
// each step produced. Person reports are downloaded as files: they are
// pages of their own, never shown inside this one.

import { h, fill } from "../dom.js";
import { t, decimal, percent } from "../i18n.js";
import { describeError } from "../errors.js";
import {
  COUNT_KEYS, OPEN_ISSUES, actRows, byPerson, clock, eventRows, isActive,
  orderedSteps, ratingRows, reportFileName, startMode, stepStatus, tagChip,
  tagged, verdict,
} from "../lib/session-analysis.js";

const POLL_MS = 3000;

export async function analysisView(ctx, root, [rawId]) {
  const id = decodeURIComponent(rawId ?? "");
  let run = null;
  let timer = null;
  let stopped = false;
  const opened = new Map(); // step name -> its result panel, kept open across refreshes

  const synthetic = h("input", { type: "checkbox", id: "an-synthetic" });
  const start = h("button", { type: "button", class: "primary big", onClick: () => begin(false) });
  const restart = h("button", { type: "button", onClick: () => begin(true) }, t("sa.restart"));
  const message = h("p", { role: "status", class: "hint" });
  const summary = h("p", { class: "meta" });
  const steps = h("ol", { class: "steps analysis-steps" });
  const issues = h("ul", {}, OPEN_ISSUES.map((key) => h("li", {}, t(`sa.issue_${key}`))));

  root.append(
    h("h1", {}, t("sa.title")),
    h("p", { class: "meta" }, id, " · ",
      h("a", { href: `#/s/${encodeURIComponent(id)}` }, t("sa.back_session")), " · ",
      h("a", { href: "#/" }, t("nav.sessions"))),
    h("p", { class: "hint" }, t("sa.intro")),
    h("div", { class: "notice" }, h("strong", {}, t("sa.issues_title")), issues),
    h("div", { class: "stack" },
      h("label", { class: "check" }, synthetic, h("span", {}, t("sa.synthetic"))),
      h("div", { class: "row" }, start, restart), message),
    summary, steps);

  function say(text, error = false) {
    message.textContent = text;
    message.className = error ? "error" : "hint";
  }

  function draw() {
    const mode = startMode(run);
    start.textContent = t(`sa.start_${mode}`);
    start.disabled = mode === "running";
    restart.hidden = !run || mode === "running";
    if (run?.synthetic) synthetic.checked = true;
    summary.textContent = run ? t(`sa.run_${isActive(run) ? "running" : run.status}`) : t("sa.never");
    fill(steps, orderedSteps(run).map(stepItem));
  }

  function stepItem(step) {
    const status = stepStatus(step);
    const seconds = typeof step.seconds === "number" ? t("sa.seconds", { value: decimal(step.seconds, 1) }) : "";
    const who = step.model ? [t("sa.who", { model: step.model })] : [];
    if (step.judge) who.push(t("sa.judge", { mode: t(`analysis.mode_${step.judge}`) }));
    const counts = Object.entries(step.counts ?? {}).map(([key, value]) =>
      `${COUNT_KEYS.includes(key) ? t(`sa.count_${key}`) : key.replace(/_/g, " ")} ${value}`).join(" · ");
    const lines = [
      h("div", { class: "row" },
        h("strong", {}, t(`sa.step_${step.name}`)),
        h("span", { class: "badge" }, t(`sa.status_${status}`)),
        seconds ? h("span", { class: "hint" }, seconds) : null),
      who.length ? h("p", { class: "hint" }, who.join(" · ")) : null,
      counts ? h("p", { class: "hint" }, counts) : null,
      step.fallback_reason ? h("p", { class: "notice" }, t("sa.fallback", { reason: step.fallback_reason })) : null,
      status === "unavailable" ? h("p", { class: "hint" }, t(`sa.unavailable_${step.name}`)) : null,
      status === "failed" ? h("p", { class: "error" }, t("sa.failed", { reason: step.error || "" })) : null,
    ];
    let panel = opened.get(step.name);
    if (status === "done") {
      const details = h("details", { class: "result" }, h("summary", {}, t("sa.open_result")));
      if (panel) details.open = true;
      details.addEventListener("toggle", () => {
        if (!details.open) {
          opened.delete(step.name);
          return;
        }
        panel = panel ?? h("div", { class: "stack" }, h("p", { class: "hint" }, t("sa.loading")));
        opened.set(step.name, panel);
        if (!details.contains(panel)) details.append(panel);
        load(step.name, panel);
      });
      if (panel) details.append(panel);
      lines.push(details);
    }
    return h("li", { class: `step state-${status}` }, lines);
  }

  async function load(name, panel) {
    try {
      const result = await ctx.api.analysisStep(id, name);
      fill(panel, render(name, result));
    } catch (error) {
      fill(panel, h("p", { class: "error" }, describeError(error)));
    }
  }

  function line(s, extra = []) {
    return h("li", {}, h("span", { class: "hint" }, `[${clock(s.start)}] `), h("strong", {}, s.speaker), ": ", s.text, extra);
  }

  function render(name, result) {
    if (name === "sentences") {
      return byPerson(result.sentences).map(({ speaker, items }) =>
        h("details", {}, h("summary", {}, t("sa.person_sentences", { name: speaker, count: items.length })),
          h("ul", { class: "plain" }, items.map((s) => h("li", {}, h("span", { class: "hint" }, `[${clock(s.start)}] `), s.text)))));
    }
    if (name === "interaction") {
      return h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, ["sa.col_person", "sa.col_turns", "sa.col_words", "sa.col_share", "sa.col_questions"].map((k) => h("th", {}, t(k))))),
        h("tbody", {}, (result.per_speaker ?? []).map((p) => h("tr", {},
          h("td", {}, p.speaker), h("td", {}, p.turns), h("td", {}, p.words),
          h("td", {}, percent(p.word_share, 0)), h("td", {}, p.questions))))));
    }
    if (name === "candidates" || name === "confirmation") {
      const items = tagged(result);
      return [
        h("p", { class: "hint" }, t(`sa.help_${name}`, { count: items.length })),
        h("ul", { class: "plain" }, items.map((s) => {
          const chips = (s.tags ?? []).map((tag) => h("span", { class: "chip" }, tagChip(tag, (x) => decimal(x, 2))));
          const extra = [" ", chips];
          if (name === "confirmation") {
            const v = verdict(s);
            extra.push(" → ", h("strong", {}, v.kind === "confirmed"
              ? `${v.polarity === "negative" ? "−" : "+"}${v.indicator}`
              : t(`sa.verdict_${v.kind}`)));
          }
          return line(s.sentence, extra);
        })),
      ];
    }
    if (name === "acts") {
      return [
        result.provisional ? h("p", { class: "notice" }, t("sa.acts_provisional")) : null,
        h("ul", { class: "plain" }, actRows(result).map((r) =>
          line(r, [" ", h("span", { class: "chip" }, r.label + (r.doubtful ? " ?" : ""))]))),
      ];
    }
    if (name === "events") {
      const rows = eventRows(result);
      return rows.length
        ? h("ul", { class: "plain" }, rows.map((e) => h("li", {},
          h("strong", {}, t(`sa.event_${e.type}`)), ` · ${e.by}: `, e.what || e.text,
          e.deadline ? h("span", { class: "hint" }, ` (${e.deadline})`) : null)))
        : h("p", { class: "hint" }, t("sa.no_events"));
    }
    if (name === "assessment") {
      const { competencies, people } = ratingRows(result);
      return [
        h("p", { class: "hint" }, t("sa.help_assessment")),
        h("div", { class: "table-wrap" }, h("table", { class: "data" },
          h("thead", {}, h("tr", {}, h("th", {}, t("sa.col_person")), competencies.map((c) => h("th", {}, c)))),
          h("tbody", {}, people.map((p) => h("tr", {}, h("td", {}, p.name),
            p.ratings.map((r) => h("td", {}, r === null ? "–" : String(r)))))))),
      ];
    }
    if (name === "reports") {
      return [
        result.synthetic ? h("p", { class: "notice" }, t("sa.synthetic_data")) : null,
        h("p", { class: "hint" }, t("sa.help_reports")),
        h("ul", { class: "plain" }, (result.people ?? []).map((person) => h("li", { class: "stack" },
          h("strong", {}, person.name),
          h("span", { class: "hint" }, t("sa.rated", { rated: person.rated, total: person.competencies })),
          person.strengths?.length ? h("span", {}, t("sa.strengths", { list: person.strengths.join(", ") })) : null,
          person.to_improve?.length ? h("span", {}, t("sa.to_improve", { list: person.to_improve.join(", ") })) : null,
          h("button", { type: "button", onClick: () => download(person) }, t("sa.download_report"))))),
      ];
    }
    return h("pre", {}, JSON.stringify(result, null, 1));
  }

  async function download(person) {
    say(t("sa.loading"));
    try {
      const page = await ctx.api.analysisReport(id, person.position);
      const url = URL.createObjectURL(new Blob([page], { type: "text/html" }));
      const link = h("a", { href: url, download: reportFileName(id, person) });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      say(t("sa.report_saved"));
    } catch (error) {
      say(describeError(error), true);
    }
  }

  async function begin(again) {
    if (again && !confirm(t("sa.restart_confirm"))) return;
    start.disabled = true;
    restart.disabled = true;
    say(t("sa.publishing"));
    try {
      await ctx.api.publish(id);
    } catch (error) {
      // A session only the RAG has (the simulated one) has nothing to publish.
      if (error?.code !== "not_found") {
        say(describeError(error), true);
        draw();
        restart.disabled = false;
        return;
      }
    }
    try {
      await ctx.api.startAnalysis(id, { restart: again, synthetic: synthetic.checked });
      say(t("sa.started"));
    } catch (error) {
      say(error?.code === "not_found" ? t("sa.not_published") : describeError(error), true);
    }
    restart.disabled = false;
    await refresh();
  }

  async function refresh() {
    if (stopped) return;
    clearTimeout(timer);
    try {
      run = await ctx.api.analysisStatus(id);
    } catch (error) {
      if (error?.code === "not_found") run = null;
      else if (error?.kind === "unauthorized") return ctx.fail(error);
      else say(error?.code === "analysis_unavailable" ? t("analysis.unavailable") : describeError(error), true);
    }
    draw();
    for (const [name, panel] of opened) if (isActive(run)) load(name, panel);
    if (isActive(run) && !stopped) timer = setTimeout(refresh, POLL_MS);
  }

  draw();
  await refresh();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
