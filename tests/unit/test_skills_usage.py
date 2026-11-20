"""Token usage: value object, ledger, call budget and the analysis steps."""

import json
from typing import Any, Optional

from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.judges import CallBudget, CountingCache
from src.application.skills.session_analysis import JudgeSetup
from src.application.skills.tagging import TagConfirmer
from src.domain.skills.analysis_run import AnalysisRun, StepRecord
from src.domain.skills.usage import (
    StepUsage,
    Usage,
    UsageLedger,
    total_usage,
)
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryCache,
)
from tests.unit.test_skills_session_analysis import build, reply

A = Usage(calls=1, input_tokens=10, output_tokens=5,
          cache_creation_input_tokens=100, cache_read_input_tokens=7,
          cost_usd=0.5)


class MeteredModel(FakeChatModel):
    """A fake judge that reports tokens like the Claude Code adapter."""

    last_usage: Optional[Usage] = None

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer and keep the usage of this call."""
        text = super().complete(system, user, max_tokens)
        self.last_usage = A
        return text


def test_usage_adds_subtracts_and_totals_tokens() -> None:
    two = A.plus(A)
    assert two.calls == 2 and two.cache_creation_input_tokens == 200
    assert two.total_tokens == 244 and two.cost_usd == 1.0
    assert two.minus(A) == A and A.minus(A).is_empty()


def test_step_usage_sums_models_and_ignores_unknown_steps() -> None:
    first = StepUsage(models={"a": A}, cache_hits=1)
    second = StepUsage(models={"a": A, "b": A}, cache_hits=2)
    both = first.plus(second)
    assert both.models["a"].calls == 2 and both.cache_hits == 3
    assert both.total().calls == 3
    assert both.minus(first).models == second.models
    assert both.minus(both).is_empty()
    assert total_usage([None, None]) is None
    assert total_usage([None, first]) == first


def test_ledger_counts_calls_even_without_reported_tokens() -> None:
    ledger = UsageLedger()
    ledger.record("m", A)
    ledger.record("m", None)
    ledger.hit()
    seen = ledger.snapshot()
    assert seen.models["m"].calls == 2
    assert seen.models["m"].input_tokens == 10 and seen.cache_hits == 1


def test_budget_and_counting_cache_feed_the_ledger() -> None:
    ledger = UsageLedger()
    budget = CallBudget(MeteredModel(["x"], "m"), 5, ledger)
    budget.complete("s", "u", 5)
    plain = CallBudget(FakeChatModel(["x"], "local"), 5, ledger)
    plain.using(plain.judge).complete("s", "u", 5)
    cache = CountingCache(InMemoryCache(), ledger)
    assert cache.get("k") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    seen = ledger.snapshot()
    assert seen.models["m"] == A and seen.models["local"].calls == 1
    assert seen.models["local"].total_tokens == 0 and seen.cache_hits == 1
    assert cache.clear() == 1


def test_each_step_records_its_usage_and_the_run_sums_them() -> None:
    ledger = UsageLedger()
    model = MeteredModel(responder=reply)
    budget = CallBudget(model, 500, ledger)
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), FakeEmbedder(dim=64), min_words=4
    )

    def judges() -> JudgeSetup:
        """Judges that spend the budget and report to the ledger."""
        return JudgeSetup(
            "claude_code", TagConfirmer(budget),
            AssessmentService(retriever, budget, min_incidents=1),
            budget=budget, usage=ledger,
        )

    run = build(judges=judges).run("s1")
    confirmation = run.step("confirmation")
    assert confirmation.usage is not None
    calls = confirmation.usage.models["fake-judge"].calls
    assert calls == confirmation.counts["judge_calls"] > 0
    assert run.step("sentences").usage == StepUsage()
    assert run.usage is not None
    assert run.usage.total().calls == budget.calls
    assert run.usage.total().input_tokens == 10 * budget.calls


def test_runs_without_a_ledger_or_stored_before_read_as_unknown() -> None:
    run = build().run("s1")
    assert run.step("confirmation").usage is None
    assert run.step("assessment").usage is None
    assert run.usage is not None and run.usage.is_empty()
    old = json.loads(run.model_dump_json())
    for step in old["steps"]:
        step.pop("usage", None)
    old.pop("usage", None)
    again = AnalysisRun.model_validate(old)
    assert again.usage is None
    record: Any = StepRecord(name="acts", usage=StepUsage(cache_hits=2))
    assert StepRecord.model_validate_json(
        record.model_dump_json()
    ).usage == StepUsage(cache_hits=2)
