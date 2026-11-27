// One stored interview report (#262), opened from «Comparar candidatos» or
// from the candidate's file: an AI interview's report (scores and the
// judgement of the answers) or a live interview's (the judgement only; the
// live page never scores). For the interviewer: the candidate does not use
// this page. Kept in the person's file until a person deletes the file.

import { h } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { emptyState, pageHead } from "../ui.js";
import { accuracyBlock, reportCard, shortDate } from "./interview.js";
import { coverageBlock } from "./coverage.js";
import { listUsedText } from "../lib/mandatory.js";

/** `#/interview-report/<plan>/<session>`. */
export async function interviewReportView(ctx, root, [planId, sessionId] = []) {
  root.classList.add("wide");
  let sheet = null;
  try {
    sheet = await ctx.api.planReport(decodeURIComponent(planId ?? ""), decodeURIComponent(sessionId ?? ""));
  } catch (error) {
    if (error?.kind === "unauthorized") throw error;
    root.append(pageHead(t("rep.iv_title")), emptyState({
      iconName: "analysis", title: t("rep.iv_title"),
      body: error?.code === "not_found" ? t("rep.iv_missing") : describeError(error),
    }));
    return null;
  }
  root.append(reportPage(sheet));
  return null;
}

/** The page of one stored report (exported for the tests). */
export function reportPage(sheet) {
  const kind = sheet?.kind === "live" ? "live" : "ai";
  const lead = t("rep.iv_lead", { kind: t(`iv.kind.${kind}`), date: shortDate(sheet?.finished_at), title: sheet?.title ?? "" });
  return h("div", { class: "stack" },
    pageHead(sheet?.candidate || t("rep.iv_title"), { lead }),
    h("p", { class: "hint" }, t((typeof sheet?.owned === "boolean" ? sheet.owned : Boolean(sheet?.candidate_id))
      ? "iv.retention" : "iv.retention_no_file")),
    // #294: the list of «Lo que hay que decir» of a live round, and its version.
    sheet?.mandatory_list ? h("p", { class: "meta" }, t("live.summary_list", { list: listUsedText(sheet.mandatory_list, t) }),
      sheet.mandatory_list.instead_of ? [" ", t("mi.instead_of", { version: sheet.mandatory_list.instead_of.version ?? "?" })] : null)
      : null,
    kind === "ai" ? reportCard({ sheet, handoff: {} })
      : h("section", { class: "card iv-report" }, accuracyBlock(sheet) ?? h("p", { class: "hint" }, t("rep.iv_missing")),
        coverageBlock(sheet)));
}
