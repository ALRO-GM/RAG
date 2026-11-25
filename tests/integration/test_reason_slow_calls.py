"""Slow structured calls through ``POST /reason`` (#267).

The interview's planner and report take tens of seconds; they used to go
through the warm live backend, whose 10 s time-out cut them. Fake models
stand for the two backends: each has its own time-out and a simulated
time the answer takes (no real waiting), and says which limit it was
given.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.answering import AnswerService
from src.application.reasoning import Reasoner
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.domain.errors import ModelTimeoutError
from src.domain.reasoning.templates import SLOW_MAX_S, get_template
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.integration.test_transcript_ingest import AUTH, TOKEN

REPORT = json.dumps({"slots": [], "to_check": [], "summary": "ok"})
SUGGESTION = json.dumps({"suggestions": [{
    "kind": "follow_up", "text": "¿Y después?", "cites": [], "about": []}]})


class TimedModel:
    """A backend with its own time-out and a simulated answer time."""

    def __init__(self, cap_s: float, work_s: float, reply: str,
                 name: str) -> None:
        self.cap_s = cap_s
        self.work_s = work_s
        self.reply = reply
        self.name = name
        self.allowed: List[float] = []
        self.gate: Optional[threading.Event] = None
        self.started = threading.Event()

    @property
    def model_name(self) -> str:
        return f"fake:{self.name}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        return self.complete_within(system, user, max_tokens, None)

    def complete_within(self, system: str, user: str, max_tokens: int,
                        deadline_s: Optional[float]) -> str:
        allowed = self.cap_s if deadline_s is None else min(self.cap_s,
                                                            deadline_s)
        self.allowed.append(allowed)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.work_s > allowed:
            raise ModelTimeoutError(
                f"Claude Code did not answer within {allowed:g} s.")
        return self.reply


def client(tmp: Path, live: TimedModel, cold: TimedModel) -> TestClient:
    store = FileSystemTranscriptStore(str(tmp / "transcripts"))
    search = SearchService(FileSystemIndexRepository(str(tmp / "none")),
                           tokenize, FakeEmbedder())
    reasoner = Reasoner(lambda alias: cold, live_models=lambda alias: live)
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(IngestTranscript(store),
                         ShowIngestedTranscript(store), TOKEN,
                         reasoner=reasoner),
    ))


def request(template: str, role: str, deadline_ms: Optional[int],
            payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": "reasoner.request.v1", "request_id": "r-1", "role": role,
        "template": template, "language": "es",
        "context": {"window": [{"id": "t1.s0", "speaker": "S2",
                                "text": "Hola"}], "focus": "t1.s0"},
        "payload": payload or {"title": "Backend", "slots": []},
    }
    if deadline_ms is not None:
        body["constraints"] = {"deadline_ms": deadline_ms}
    return body


def test_a_30_s_report_succeeds_and_a_live_call_keeps_its_cap(
    tmp_path: Path,
) -> None:
    live = TimedModel(10.0, 12.0, SUGGESTION, "warm")
    cold = TimedModel(180.0, 30.0, REPORT, "cold")
    web = client(tmp_path, live, cold)
    report = web.post("/reason", json=request(
        "interview.report@2", "structured", 90000), headers=AUTH).json()
    assert report["status"] == "ok" and report["data"]["summary"] == "ok"
    assert len(cold.allowed) == 1 and 89.5 < cold.allowed[0] <= 90.0
    assert live.allowed == []
    late = web.post("/reason", json=request(
        "suggester.interview@2", "suggester", 1800), headers=AUTH).json()
    assert late["status"] == "timeout"
    assert live.allowed == [1.8]
    slow_live = web.post("/reason", json=request(
        "interviewer@2", "conversational", None,
        {"action": "ask", "topic": "x"}), headers=AUTH).json()
    assert slow_live["status"] == "timeout"
    assert "10 s" in slow_live["error"]["message"]  # the warm cap stays


def test_a_slow_call_gets_at_most_the_serving_limit(tmp_path: Path) -> None:
    cold = TimedModel(600.0, 1.0, REPORT, "cold")
    web = client(tmp_path, TimedModel(10.0, 1.0, SUGGESTION, "warm"), cold)
    for deadline, allowed in ((None, SLOW_MAX_S), (900000, SLOW_MAX_S),
                              (120000, 120.0)):
        web.post("/reason", json=request(
            "interview.planner@2", "structured", deadline,
            {"offer": "x" * 50, "minutes": 20}), headers=AUTH)
        assert allowed - 0.5 < cold.allowed[-1] <= allowed
    for name in ("interview.planner@1", "interview.planner@2",
                 "interview.report@1", "interview.report@2"):
        assert get_template(name, "structured").slow
    for name, role in (("interview.slots@2", "structured"),
                       ("interviewer@2", "conversational"),
                       ("suggester.interview@2", "suggester")):
        assert not get_template(name, role).slow  # type: ignore[arg-type]


def test_a_live_call_never_waits_behind_a_slow_one(tmp_path: Path) -> None:
    live = TimedModel(10.0, 0.5, SUGGESTION, "warm")
    cold = TimedModel(180.0, 30.0, REPORT, "cold")
    cold.gate = threading.Event()  # the report is "running"
    web = client(tmp_path, live, cold)
    done: List[Dict[str, Any]] = []
    slow = threading.Thread(target=lambda: done.append(web.post(
        "/reason", json=request("interview.report@2", "structured", 90000),
        headers=AUTH).json()))
    slow.start()
    assert cold.started.wait(5)
    quick = web.post("/reason", json=request(
        "suggester.interview@2", "suggester", 1800), headers=AUTH).json()
    assert quick["status"] == "ok" and not done  # answered meanwhile
    cold.gate.set()
    slow.join(5)
    assert done and done[0]["status"] == "ok"


def test_slow_calls_are_bounded_and_a_queued_one_waits_within_its_deadline(
    tmp_path: Path,
) -> None:
    """#267 review: at most ``slow_calls`` cold calls at once."""
    cold = TimedModel(180.0, 1.0, REPORT, "cold")
    cold.gate = threading.Event()
    live = TimedModel(10.0, 0.1, SUGGESTION, "warm")
    store = FileSystemTranscriptStore(str(tmp_path / "transcripts"))
    search = SearchService(FileSystemIndexRepository(str(tmp_path / "none")),
                           tokenize, FakeEmbedder())
    reasoner = Reasoner(lambda alias: cold, live_models=lambda alias: live,
                        slow_calls=1)
    web = TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(IngestTranscript(store),
                         ShowIngestedTranscript(store), TOKEN,
                         reasoner=reasoner),
    ))
    first: List[Dict[str, Any]] = []
    holder = threading.Thread(target=lambda: first.append(web.post(
        "/reason", json=request("interview.report@2", "structured", 90000),
        headers=AUTH).json()))
    holder.start()
    assert cold.started.wait(5)
    queued = web.post("/reason", json=request(
        "interview.report@2", "structured", 200), headers=AUTH).json()
    assert queued["status"] == "timeout"
    assert "No free slot" in queued["error"]["message"]
    live_call = web.post("/reason", json=request(
        "suggester.interview@2", "suggester", 1800), headers=AUTH).json()
    assert live_call["status"] == "ok"  # live calls are never queued here
    cold.gate.set()
    holder.join(5)
    assert first and first[0]["status"] == "ok"
    cold.gate = None
    again = web.post("/reason", json=request(
        "interview.report@2", "structured", 90000), headers=AUTH).json()
    assert again["status"] == "ok"  # the slot was given back
    assert cold.allowed[-1] <= 90.0
