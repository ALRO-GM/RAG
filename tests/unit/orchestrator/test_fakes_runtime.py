"""Fakes, executors and the JSONL sink."""

import json
import threading
from pathlib import Path
from typing import List

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.jsonl_sink import JsonlSink
from orchestrator.adapters.outbound.runtime import (
    InlineExecutor,
    SystemClock,
    ThreadExecutor,
)
from orchestrator.application.ports import CancelToken
from orchestrator.domain.records import SuggestionRecord, TriggerInfo


def test_fake_knowledge_matches_by_word_and_top_k() -> None:
    fake = FakeKnowledge(
        [{"id": "a", "text": "reembolso de factura duplicada"},
         {"id": "b", "text": "horario de oficina"}], delay=0.5,
        sleep=lambda s: slept.append(s),
    )
    slept: List[float] = []
    query = {"request_id": "r", "domain": "d", "text": "factura duplicada",
             "top_k": 1}
    assert [c["id"] for c in fake.query(query)["chunks"]] == ["a"]
    assert slept == [0.5]
    everything = FakeKnowledge([{"id": "b", "text": "x"}], everything=True)
    assert len(everything.query({**query, "text": "nada"})["chunks"]) == 1


def test_fake_reasoner_echoes_the_focus() -> None:
    fake = FakeReasoner(delay=0.2, sleep=lambda s: None)
    request = {"request_id": "r", "role": "suggester", "context": {
        "window": [{"id": "t1.s0", "speaker": "c", "text": "mi factura"}],
        "focus": "t1.s0"}}
    out = fake.reason(request, CancelToken())
    assert out["suggestions"][0]["about"] == ["t1.s0"]
    assert out["latency"]["total_ms"] == 200.0
    none = FakeReasoner(status="timeout").reason(request, CancelToken())
    assert none["suggestions"] == []


def test_executors() -> None:
    ran: List[str] = []
    InlineExecutor().submit(lambda: ran.append("inline"))
    InlineExecutor().wait_idle()
    threads = ThreadExecutor()
    gate = threading.Event()

    def slow() -> None:
        gate.wait(1)
        ran.append("thread")

    threads.submit(slow)
    gate.set()
    threads.submit(lambda: ran.append("second"))
    threads.wait_idle()
    assert sorted(ran) == ["inline", "second", "thread"]


def test_system_clock_moves() -> None:
    clock = SystemClock()
    before = clock.now()
    clock.sleep(0.001)
    assert clock.now() > before


def test_jsonl_sink_writes_a_line_per_record(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "deep" / "out.jsonl")
    record = SuggestionRecord(
        session_id="s", call=1, request_id="s-1", status="ok",
        trigger=TriggerInfo(utterance="t1.s0", speaker="c", role="customer",
                            stream_time=1.0, text="hola mundo"),
    )
    sink.write(record)
    sink.write(record)
    lines = (tmp_path / "deep" / "out.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["call"] for line in lines] == [1, 1]
    assert len(sink.records) == 2
