// "Análisis" of a session (#/analysis/<id>): start the RAG's whole analysis,
// watch the steps advance (status, time, judge, fallback) and open what
// each step produced. Person reports are downloaded as files: they are
// pages of their own, never shown inside this one.

import { h, fill } from "../dom.js";
import { t, decimal, percent } from "../i18n.js";
import { describeError } from "../errors.js";
import { helpHref } from "../lib/help-analysis.js";
import {
  COUNT_KEYS, MAX_JOB, OPEN_ISSUES, SOURCES, actRows, asksForClaude, byPerson, canRedoSynthesis, cellText,
  clock, doubtfulCount, eventRows, isActive, jobForm, orderedSteps, ratingRows, reasonText,
  reportBlob, reportFileName, reviewRows, usageRows, startMode, startRefusal, startingRun, stepStatus, stillStarting,
  synthesisPeople, tagChip, tagged, transcriptLink, verdict, whyText,
} from "../lib/session-analysis.js";
import { publishWhenFree } from "../lib/names.js";

const POLL_MS = 3000;

export async function analysisView(ctx, root, [rawId]) {
  const id = decodeURIComponent(rawId ?? "");
  let run = null;
  let timer = null;
  let stopped = false;
  let starting = 0; // when this page last started the analysis (ms)
  const opened = new Map(); // step name -> its result panel, kept open across refreshes

  const synthetic = h("input", { type: "checkbox", id: "an-synthetic" });
  // The job sought (#113): kept with the analysis, read by the synthesis.
  let jobEdited = false;
  const job = h("textarea", {
    id: "an-job", rows: "4", maxlength: String(MAX_JOB), placeholder: t("sa.job_placeholder"),
    onInput: () => { jobEdited = true; },
  });
  const useClaude = h("input", {
    type: "checkbox", id: "an-claude",
    onChange: () => { jobEdited = true; },
  });
  const claudeLabel = h("label", { class: "check" }, useClaude, h("span", {}, t("sa.claude_synthesis")));
  claudeLabel.hidden = true;
  const redo = h("button", { type: "button", onClick: () => begin(false, ["synthesis"]) }, t("sa.redo_synthesis"));
  const start = h("button", { type: "button", class: "primary big", onClick: () => begin(false) });
  const restart = h("button", { type: "button", onClick: () => begin(true) }, t("sa.restart"));
  const cancel = h("button", { type: "button", class: "danger", onClick: () => stop() }, t("sa.cancel"));
  const message = h("p", { role: "status", class: "hint" });
  const summary = h("p", { class: "meta" });
  const steps = h("ol", { class: "steps analysis-steps" });
  const usageTotal = h("div", { class: "usage" });
  const issues = h("ul", {}, OPEN_ISSUES.map((key) => h("li", {}, t(`sa.issue_${key}`))));

  root.append(
    h("h1", {}, t("sa.title")),
    h("p", { class: "meta" }, id, " · ",
      h("a", { href: `#/s/${encodeURIComponent(id)}` }, t("sa.back_session")), " · ",
      h("a", { href: "#/" }, t("nav.sessions"))),
    h("p", { class: "hint" }, t("sa.intro"), " ", h("a", { href: helpHref() }, t("help.link"))),
    h("div", { class: "notice" }, h("strong", {}, t("sa.issues_title")), issues),
    h("div", { class: "stack" },
      h("label", { class: "check" }, synthetic, h("span", {}, t("sa.synthetic"))),
      h("label", { for: "an-job" }, h("strong", {}, t("sa.job_title"))),
      h("p", { class: "hint" }, t("sa.job_hint")),
      job, claudeLabel,
      h("div", { class: "stack" }, start, restart, redo, cancel), message),
    summary, usageTotal, steps);

  function say(text, error = false) {
    message.textContent = text;
    message.className = error ? "error" : "hint";
  }

  function draw() {
    const mode = startMode(run);
    start.textContent = t(`sa.start_${mode}`);
    start.disabled = mode === "running";
    restart.hidden = !run || mode === "running";
    cancel.hidden = mode !== "running";
    if (run?.synthetic) synthetic.checked = true;
    redo.hidden = !canRedoSynthesis(run);
    const form = jobForm(run, jobEdited);
    if (form) {
      job.value = form.job;
      useClaude.checked = form.useClaude;
    }
    summary.textContent = run ? t(`sa.run_${isActive(run) ? "running" : run.status}`) : t("sa.never");
    fill(steps, orderedSteps(run).map(stepItem));
    fill(usageTotal, usageLines(run?.usage, "sa.usage_total"));
  }

  function usageLines(usage, titleKey) {
    const rows = usageRows(usage);
    if (!rows) return [];
    const line = (u) => t("sa.usage_line", {
      calls: u.calls, input: u.input_tokens, output: u.output_tokens,
      write: u.cache_creation_input_tokens, read: u.cache_read_input_tokens,
    });
    return [
      h("p", { class: "hint" }, h("strong", {}, t(titleKey)), " ", line(rows.total),
        rows.total.cost_usd ? ` · ${t("sa.usage_cost", { usd: decimal(rows.total.cost_usd, 2) })}` : "",
        ` · ${t("sa.usage_cache_hits", { hits: rows.cacheHits })}`),
      ...rows.models.map((m) => h("p", { class: "hint" }, `${m.model}: ${line(m)}`)),
    ];
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
        h("a", { class: "help-link", href: helpHref(step.name), title: t("help.hint"), "aria-label": t("help.hint") }, "?"),
        h("span", { class: "badge" }, t(`sa.status_${status}`)),
        seconds ? h("span", { class: "hint" }, seconds) : null),
      who.length ? h("p", { class: "hint" }, who.join(" · ")) : null,
      counts ? h("p", { class: "hint" }, counts) : null,
      ...usageLines(step.usage, "sa.usage_step"),
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
      const info = result.tagger ?? {};
      return [
        h("p", { class: "hint" }, t("sa.acts_help", {
          name: info.name ?? "", version: info.version ?? "",
          doubtful: doubtfulCount(result), total: actRows(result).length,
        })),
        info.note ? h("p", { class: "notice" }, info.note) : null,
        h("ul", { class: "plain" }, actRows(result).map((r) =>
          line(r, [" ", r.labels.map((l) => h("span", { class: l.doubtful ? "chip doubtful" : "chip" },
            `${t(`act.${l.label}`)} ${l.weight}%${l.doubtful ? ` · ${t("sa.doubtful")}` : ""}`))]))),
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
            p.cells.map((c) => h("td", c.kind === "provisional"
              ? { class: "provisional", title: t("sa.estimate_title", { confidence: t(`sa.confidence_${c.confidence}`) }) }
              : {}, cellText(c)))))))),
        h("p", { class: "notice" }, t("sa.provisional_note")),
        reviewRows(result).map((person) => reviewBlock(person, result.min_incidents)),
      ];
    }
    if (name === "reports") {
      return [
        result.synthetic ? h("p", { class: "notice" }, t("sa.synthetic_data")) : null,
        h("p", { class: "hint" }, t("sa.help_reports")),
        h("ul", { class: "plain" }, (result.people ?? []).map((person) => h("li", { class: "stack" },
          h("strong", {}, person.name),
          h("span", { class: "hint" }, t("sa.rated", { rated: person.rated, total: person.competencies })),
          typeof person.provisional === "number"
            ? h("span", {}, t("sa.tiers", { firm: person.rated, provisional: person.provisional, observations: person.observations ?? 0 }))
            : null,
          person.why?.length
            ? h("div", { class: "notice" }, h("strong", {}, t("sa.why_title")),
              h("ul", {}, person.why.map((code) => h("li", {}, whyText(t, code, person, result.min_incidents)))))
            : null,
          person.strengths?.length ? h("span", {}, t("sa.strengths", { list: person.strengths.join(", ") })) : null,
          person.to_improve?.length ? h("span", {}, t("sa.to_improve", { list: person.to_improve.join(", ") })) : null,
          h("button", { type: "button", onClick: () => download(person) }, t("sa.download_report"))))),
      ];
    }
    if (name === "synthesis") {
      return [
        h("p", { class: "notice" }, t("sa.syn_note")),
        result.job ? h("p", { class: "hint" }, t("sa.syn_job", { job: result.job })) : null,
        synthesisPeople(result).map((person) => synthesisBlock(person, Boolean(result.job))),
      ];
    }
    return h("pre", {}, JSON.stringify(result, null, 1));
  }

  // One person's explanatory profile: every point with the quotes it
  // cites, each with its minute linking to the transcript.
  function synthesisBlock(person, withJob) {
    const head = h("summary", {}, h("strong", {}, person.name), " · ",
      t("sa.syn_confidence", { level: t(`sa.syn_conf_${person.confidence}`) }));
    if (person.status === "skipped") {
      const reason = person.reason === "no_evidence" ? t("sa.syn_no_evidence") : person.reason;
      return h("details", { class: "review" }, head, h("p", { class: "hint" }, t("sa.syn_skipped", { reason })));
    }
    const point = (p) => h("li", { class: "stack" },
      h("span", {}, p.text, " ", h("span", { class: "hint" }, `(${t(`sa.syn_basis_${p.basis}`)})`)),
      h("ul", { class: "plain" }, p.quotes.map((q) => h("li", {}, moment(q.start), " «", q.text, "» ",
        h("span", { class: "hint" }, `(${q.competency_id} · ${t(`sa.syn_basis_${q.kind}`)})`)))),
      p.events.length
        ? h("ul", { class: "plain" }, p.events.map((e) => h("li", { class: "hint" }, moment(e.start), " ", e.text)))
        : null);
    const points = (items) => (items.length
      ? h("ul", { class: "plain" }, items.map(point))
      : h("p", { class: "hint" }, t("sa.syn_none")));
    const b = person.basis;
    return h("details", { class: "review" }, head,
      h("h3", {}, t("sa.syn_strengths")), points(person.strengths),
      h("h3", {}, t("sa.syn_unobserved")),
      person.unobserved.length
        ? h("ul", {}, person.unobserved.map((n) => h("li", {}, n)))
        : h("p", { class: "hint" }, t("sa.syn_unobserved_none")),
      h("h3", {}, t(withJob ? "sa.syn_fit" : "sa.syn_tasks")), points(person.fit),
      person.questions.length
        ? [h("h3", {}, t("sa.syn_questions")), h("ul", {}, person.questions.map((q) => h("li", {}, q)))]
        : null,
      h("p", { class: "hint" }, t("sa.syn_conf_reason", {
        firm: b.firm ?? 0, firm_quotes: b.firm_quotes ?? 0, provisional: b.provisional ?? 0,
        quotes: b.quotes ?? 0, words: b.words ?? 0,
      })),
      person.dropped ? h("p", { class: "hint" }, t("sa.syn_dropped", { count: person.dropped })) : null,
      person.model ? h("p", { class: "hint" }, t("sa.syn_model", { model: person.model })) : null);
  }

  // One person's provisional estimates and observations, each sentence
  // with its minute linking to that moment of the transcript.
  function reviewBlock(person, needed) {
    const summaryLine = t("sa.person_tiers", {
      name: person.name, firm: person.firm, provisional: person.provisional, observations: person.observations,
    });
    if (!person.review.length) return h("p", { class: "hint" }, summaryLine);
    return h("details", { class: "review" }, h("summary", {}, summaryLine),
      h("ul", { class: "plain" }, person.review.map((item) => h("li", { class: "stack" },
        h("strong", {}, item.competency, " — ", item.estimate
          ? t("sa.estimate", { rating: item.estimate.rating, confidence: t(`sa.confidence_${item.estimate.confidence}`) })
          : t("sa.observations_only")),
        item.estimate?.reasons?.length
          ? h("span", { class: "hint" }, t("sa.not_firm", {
            reasons: item.estimate.reasons.map((code) => reasonText(t, code, { needed, ...item.estimate })).join("; "),
          }))
          : null,
        h("ul", { class: "plain" }, item.observations.map((o) => h("li", {},
          moment(o.start), " ",
          h("strong", {}, `${o.polarity === "negative" ? "−" : "+"}${o.indicator_id}`), " «", o.quote, "» ",
          h("span", { class: "hint" }, `(${SOURCES.includes(o.source) ? t(`sa.source_${o.source}`) : o.source})`))))))));
  }

  // The minute of a sentence; a link into the transcript unless the
  // session is simulated (it only exists in the analysis).
  function moment(seconds) {
    const text = `[${clock(seconds) || "–"}]`;
    if (run?.synthetic || typeof seconds !== "number") return h("span", { class: "hint" }, text);
    return h("a", { href: transcriptLink(id, seconds), title: t("sa.open_moment") }, text);
  }

  async function download(person) {
    say(t("sa.loading"));
    try {
      const page = await ctx.api.analysisReport(id, person.position);
      const url = URL.createObjectURL(reportBlob(page));
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

  async function stop() {
    if (!confirm(t("sa.cancel_confirm"))) return;
    cancel.disabled = true;
    try {
      await ctx.api.cancelAnalysis(id);
      say(t("sa.cancelled"));
    } catch (error) {
      say(error?.code === "wrong_state" ? t("sa.cancel_terminal") : describeError(error), true);
    }
    cancel.disabled = false;
    await refresh();
  }

  async function begin(again, steps = []) {
    if (again && !confirm(t("sa.restart_confirm"))) return;
    start.disabled = true;
    restart.disabled = true;
    redo.disabled = true;
    say(t("sa.publishing"));
    try {
      // Not while the speakers' names are searched: they would be missing.
      const sent = await publishWhenFree(() => ctx.api.publish(id), {
        busy: async () => Boolean((await ctx.api.getNames(id))?.running),
        onWait: () => say(t("names.busy")),
        stopped: () => stopped,
      });
      if (!sent) return;
    } catch (error) {
      // A session only the RAG has (the simulated one) has nothing to publish.
      if (error?.code !== "not_found") {
        say(describeError(error), true);
        draw();
        restart.disabled = false;
        redo.disabled = false;
        return;
      }
    }
    try {
      await ctx.api.startAnalysis(id, {
        restart: again, synthetic: synthetic.checked, steps,
        job: job.value, claudeSynthesis: !claudeLabel.hidden && useClaude.checked,
      });
      jobEdited = false;
      say(t("sa.started"));
      starting = Date.now();
    } catch (error) {
      say(startRefusal(t, error, describeError), true);
    }
    restart.disabled = false;
    redo.disabled = false;
    await refresh();
  }

  async function refresh() {
    if (stopped) return;
    clearTimeout(timer);
    try {
      run = await ctx.api.analysisStatus(id);
    } catch (error) {
      // Just after a start the worker may not have written its record yet.
      if (error?.code === "not_found") run = stillStarting(starting, Date.now()) ? startingRun(id) : null;
      else if (error?.kind === "unauthorized") return ctx.fail(error);
      else say(error?.code === "analysis_unavailable" ? t("analysis.unavailable") : describeError(error), true);
    }
    draw();
    for (const [name, panel] of opened) if (isActive(run)) load(name, panel);
    if (isActive(run) && !stopped) timer = setTimeout(refresh, POLL_MS);
  }

  // Ask before sending this session to Claude only when the judge is the local model.
  ctx.api.analysisJudge().then((judge) => {
    claudeLabel.hidden = !asksForClaude(judge);
  }).catch(() => {});
  draw();
  await refresh();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
