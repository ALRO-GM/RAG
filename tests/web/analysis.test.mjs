import test from "node:test";
import assert from "node:assert/strict";
import { Api } from "../../transcriber/adapters/inbound/web/js/api.js";
import { JUDGE_MODES, quotaLine, resetText, selectedMode } from "../../transcriber/adapters/inbound/web/js/lib/analysis.js";
import { translate } from "../../transcriber/adapters/inbound/web/js/lib/i18n.js";
import { es } from "../../transcriber/adapters/inbound/web/js/i18n/es.js";

const t = (key, values) => translate({ es }, "es", key, values);

test("the judge is read and chosen through the transcriber, with the code", async () => {
  const calls = [];
  const api = new Api(() => "code-123", async (url, init) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ mode: "local" }), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  assert.deepEqual(await api.analysisJudge(), { mode: "local" });
  await api.setAnalysisJudge("both");
  assert.equal(calls[0].url, "/api/analysis/judge");
  assert.equal(calls[0].init.method, "GET");
  assert.equal(calls[1].init.method, "PUT");
  assert.deepEqual(JSON.parse(calls[1].init.body), { mode: "both" });
  for (const call of calls) assert.equal(call.init.headers["X-Access-Code"], "code-123");
});

test("the quota line says available, used up (with when it comes back) or unknown", () => {
  assert.equal(quotaLine(t, { claude_quota: "available" }), es["analysis.quota_available"]);
  assert.equal(quotaLine(t, { claude_quota: "exhausted", quota_resets: "3pm" }),
    `${es["analysis.quota_exhausted"]} Vuelve: 3pm.`);
  assert.equal(quotaLine(t, { claude_quota: "exhausted" }), es["analysis.quota_exhausted"]);
  assert.equal(quotaLine(t, { claude_quota: "weird" }), es["analysis.quota_unknown"]);
  assert.equal(quotaLine(t, null), es["analysis.quota_unknown"]);
});

test("every mode has a Spanish label and an unknown mode shows the default", () => {
  for (const mode of JUDGE_MODES) assert.ok(es[`analysis.mode_${mode}`], mode);
  assert.equal(selectedMode({ mode: "both" }), "both");
  assert.equal(selectedMode({ mode: "api" }), "claude_code");
  assert.equal(selectedMode(undefined), "claude_code");
});

test("a reset given as a moment is shown in local time, in both languages", () => {
  const status = { claude_quota: "exhausted", quota_resets: "2025-11-15T14:00:00+00:00",
    quota_resets_at: "2025-11-15T14:00:00+00:00" };
  const es = resetText(status, "es-ES");
  const en = resetText(status, "en-GB");
  assert.equal(es, new Date(status.quota_resets_at).toLocaleString("es-ES", { dateStyle: "short", timeStyle: "short" }));
  assert.ok(!es.includes("T14:00") && !en.includes("T14:00"), "not the raw timestamp");
  assert.ok(quotaLine(t, status, "es-ES").endsWith(`Vuelve: ${es}.`));
  assert.equal(resetText({ quota_resets_at: "not a date", quota_resets: "3pm" }, "es-ES"), "3pm");
  assert.equal(resetText({}, "es-ES"), "");
});
