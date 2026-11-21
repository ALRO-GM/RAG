"""LiveAssist: one call in flight, cancel stale, failures as statuses."""

from typing import Any, Callable, Dict, List

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.jsonl_sink import MemorySink
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.live_assist import LiveAssist
from orchestrator.application.ports import (
    CancelToken,
    Document,
    KnowledgeUnavailable,
    ReasonerUnavailable,
)
from orchestrator.application.replay import default_roles, transcript_to_events
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import LiveEvent
from tests.unit.orchestrator.fixtures import product, transcript


class StepClock:
    """A clock that advances 10 ms per reading."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        self.t += 0.01
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class ManualExecutor:
    """Keeps the jobs until the test runs them."""

    def __init__(self) -> None:
        self.jobs: List[Callable[[], None]] = []

    def submit(self, job: Callable[[], None]) -> None:
        self.jobs.append(job)

    def wait_idle(self) -> None:
        while self.jobs:
            self.jobs.pop(0)()


class BrokenKnowledge:
    def query(self, query: Document) -> Document:
        raise KnowledgeUnavailable("down")


class BrokenReasoner:
    def reason(self, request: Document, cancel: CancelToken) -> Document:
        raise ReasonerUnavailable("down")


def events() -> List[LiveEvent]:
    t = transcript()
    return list(transcript_to_events(t, {"a": "agent", "c": "customer"}))


def assist(
    knowledge: Any = None, reasoner: Any = None, executor: Any = None,
    config: ProductConfig = product(),
) -> LiveAssist:
    return LiveAssist(
        config, knowledge or FakeKnowledge(), reasoner or FakeReasoner(),
        MemorySink(), StepClock(), executor or InlineExecutor(),
        {"a": "agent", "c": "customer"}, {"type": "soporte", "goal": None},
    )


def run(a: LiveAssist) -> MemorySink:
    for event in events():
        a.on_event(event)
    a.finish()
    assert isinstance(a.sink, MemorySink)
    return a.sink


def test_only_long_customer_sentences_ask() -> None:
    a = assist()
    sink = run(a)
    assert [r.trigger.utterance for r in sink.records] == ["t1.s0", "t4.s0"]
    assert a.skips == {"role": 2, "min_words": 1}
    record = sink.records[0]
    assert record.status == "ok" and record.suggestions
    assert record.stages.total_ms is not None
    assert record.stages.knowledge_ms is not None
    assert record.stages.reasoner_ms is not None


def test_request_follows_the_contract() -> None:
    reasoner = FakeReasoner()
    run(assist(reasoner=reasoner))
    request = reasoner.requests[0]
    assert request["schema"] == "reasoner.request.v1"
    assert request["template"] == "suggester.support@2"
    assert request["context"]["focus"] == "t1.s0"
    assert request["context"]["speaker_roles"] == {"a": "agent",
                                                   "c": "customer"}
    assert request["context"]["session"] == {"id": "unit-session",
                                             "type": "soporte"}
    assert [w["id"] for w in request["context"]["window"]] == [
        "t0.s0", "t1.s0"]
    assert request["constraints"] == {"max_suggestions": 1,
                                      "deadline_ms": 2800,
                                      "mode": "pincelada"}
    assert len(request["request_id"]) <= 64


def test_chunks_reach_the_reasoner() -> None:
    knowledge = FakeKnowledge(
        [{"id": "k1", "text": "Una factura duplicada se reembolsa."}]
    )
    reasoner = FakeReasoner()
    sink = run(assist(knowledge=knowledge, reasoner=reasoner))
    assert knowledge.queries[0]["domain"] == "support.demo"
    assert reasoner.requests[0]["retrieved"][0]["id"] == "k1"
    assert sink.records[0].chunks == 1
    assert sink.records[0].suggestions[0]["cites"] == ["k1"]


def test_knowledge_failure_goes_on_without_chunks() -> None:
    sink = run(assist(knowledge=BrokenKnowledge()))
    assert sink.records[0].knowledge_status == "unavailable"
    assert sink.records[0].status == "ok"


def test_reasoner_failure_is_a_status() -> None:
    sink = run(assist(reasoner=BrokenReasoner()))
    assert sink.records[0].status == "error"
    assert sink.records[0].error == {"kind": "transient",
                                     "message": "down"}


def test_service_status_is_kept() -> None:
    sink = run(assist(reasoner=FakeReasoner(status="quota")))
    assert {r.status for r in sink.records} == {"quota"}


def test_new_trigger_cancels_the_stale_call() -> None:
    executor = ManualExecutor()
    reasoner = FakeReasoner()
    a = assist(reasoner=reasoner, executor=executor)
    for event in events():
        a.on_event(event)
    assert len(executor.jobs) == 2
    a.finish()
    first, second = a.sink.records  # type: ignore[attr-defined]
    assert first.status == "cancelled" and first.suggestions == []
    assert second.status == "ok"
    assert len(reasoner.requests) == 1


def test_cancel_after_the_answer_drops_it() -> None:
    holder: Dict[str, CancelToken] = {}

    class Slow(FakeReasoner):
        def reason(self, request: Document, cancel: CancelToken) -> Document:
            out = super().reason(request, cancel)
            cancel.cancel()
            return out

    sink = run(assist(reasoner=Slow()))
    assert holder == {}
    assert {r.status for r in sink.records} == {"cancelled"}
    assert all(r.suggestions == [] for r in sink.records)


def test_without_cancel_stale_the_busy_trigger_is_skipped() -> None:
    config = product()
    assert config.trigger is not None
    config = config.model_copy(update={
        "trigger": config.trigger.model_copy(update={"cancel_stale": False})
    })
    executor = ManualExecutor()
    a = assist(executor=executor, config=config)
    for event in events():
        a.on_event(event)
    a.finish()
    assert len(a.sink.records) == 1  # type: ignore[attr-defined]
    assert a.skips["in_flight"] == 1


def test_a_live_product_is_required() -> None:
    import pytest
    with pytest.raises(ValueError):
        assist(config=product("post_call"))


def test_voicebot_fires_on_user_finals() -> None:
    config = product("voicebot")
    t = transcript()
    roles = {"a": "bot", "c": "user"}
    a = LiveAssist(config, FakeKnowledge(), FakeReasoner(), MemorySink(),
                   StepClock(), InlineExecutor(), roles)
    for event in transcript_to_events(t, roles):
        a.on_event(event)
    records = a.sink.records  # type: ignore[attr-defined]
    assert [r.trigger.utterance for r in records] == ["t1.s0", "t4.s0"]
    assert records[0].trigger.text.startswith("Tengo un problema")
    assert records[0].reply is not None


def test_default_roles_by_order() -> None:
    assert default_roles(product(), transcript()) == {
        "a": "agent", "c": "customer"}
    assert default_roles(product("voicebot"), transcript()) == {
        "a": "user", "c": "user"}
