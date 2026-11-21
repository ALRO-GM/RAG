"""LiveSessions: replays and live sessions feed the page the same way."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest

from orchestrator.adapters.outbound.fakes import FakeKnowledge, FakeReasoner
from orchestrator.adapters.outbound.jsonl_sink import MemorySink
from orchestrator.adapters.outbound.runtime import InlineExecutor
from orchestrator.application.ports import (
    CancelToken,
    Document,
    SuggestionSink,
    TranscriberUnavailable,
)
from orchestrator.application.replay import transcript_to_events
from orchestrator.application.sessions import (
    LiveSessions,
    SessionError,
    live_speakers,
    transcriber_body,
)
from orchestrator.domain.events import Transcript
from tests.unit.orchestrator.fixtures import product, transcript


class Clock:
    """Advances only when asked to wait."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class InlineRunner:
    """Runs a session to its end before returning, or keeps it."""

    def __init__(self, hold: bool = False) -> None:
        self.hold = hold
        self.jobs: List[Callable[[], None]] = []

    def start(self, job: Callable[[], None]) -> None:
        if self.hold:
            self.jobs.append(job)
        else:
            job()


class Transcripts:
    def __init__(self, items: Dict[str, Transcript]) -> None:
        self.items = items

    def list(self) -> List[Document]:
        return [{"id": k} for k in self.items]

    def load(self, transcript_id: str) -> Transcript:
        return self.items[transcript_id]


class SignalReasoner:
    """Answers with one suggestion that carries a complaint signal."""

    def __init__(self, status: str = "ok") -> None:
        self.status = status
        self.requests: List[Document] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        self.requests.append(request)
        focus = request["context"]["focus"]
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": "suggester",
            "status": self.status, "model": "fake", "backend": "local",
            "latency": {"total_ms": 1.0},
            "suggestions": [{"kind": "answer", "text": "Ofrece escalar",
                             "signals": ["complaint"], "about": [focus],
                             "cites": ["k1"], "script_phase": "technical"}],
        }


class FakeTranscriber:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.bodies: List[Document] = []
        self.stopped: List[str] = []

    def start(self, body: Document) -> Document:
        if self.fail:
            raise TranscriberUnavailable("409 busy")
        self.bodies.append(body)
        return {"id": body["session_id"], "state": "running"}

    def stop(self, session_id: str) -> None:
        self.stopped.append(session_id)
        if self.fail:
            raise TranscriberUnavailable("down")


class FakeEvents:
    def __init__(self, documents: List[Document],
                 fail: bool = False) -> None:
        self.documents = documents
        self.fail = fail
        self.followed: List[str] = []

    def follow(self, session_id: str, stop: CancelToken
               ) -> Iterator[Document]:
        self.followed.append(session_id)
        if self.fail:
            raise TranscriberUnavailable("closed 4404")
        yield from self.documents


def sessions(
    reasoner: Any = None,
    runner: Optional[InlineRunner] = None,
    transcriber: Optional[FakeTranscriber] = None,
    events: Optional[FakeEvents] = None,
    logs: Optional[List[MemorySink]] = None,
) -> LiveSessions:
    def log(session_id: str, name: str) -> SuggestionSink:
        sink = MemorySink()
        if logs is not None:
            logs.append(sink)
        return sink

    return LiveSessions(
        {"live_assist.support": product(),
         "live_assist.interview": product("live_assist.interview"),
         "post_call": product("post_call")},
        FakeKnowledge([{"id": "k1", "text": "factura cobro doble"}],
                      everything=True),
        reasoner or FakeReasoner(), Transcripts({"t": transcript()}),
        Clock(), runner or InlineRunner(), InlineExecutor,
        transcriber, events, log, new_id=lambda: "live-1",
    )


def messages(live: LiveSessions, session_id: str) -> List[Dict[str, Any]]:
    found, finished = live.feed(session_id).read(0, 0.01)
    assert finished
    return [m for _, m in found]


def test_products_say_what_the_page_shows() -> None:
    live = sessions()
    shown = {p["product"]: p for p in live.list_products()}
    assert set(shown) == {"live_assist.support", "live_assist.interview"}
    assert shown["live_assist.support"]["signals"] is True
    assert shown["live_assist.interview"]["signals"] is False
    assert shown["live_assist.interview"]["script_phases"][0] == "opening"
    assert shown["live_assist.support"]["banner"] == "support_ai_notice@1"
    assert shown["live_assist.support"]["live"] is False
    assert live.list_transcripts() == [{"id": "t"}]


def test_a_replay_feeds_events_suggestions_and_the_end() -> None:
    logs: List[MemorySink] = []
    live = sessions(reasoner=SignalReasoner(), logs=logs)
    view = live.start_replay("demo-1", "live_assist.support", "t", None)
    assert view["mode"] == "replay" and view["transcript"] == "t"
    found = messages(live, "demo-1")
    kinds = [m["type"] for m in found]
    assert kinds[0] == "session" and kinds[-1] == "status"
    assert found[0]["session"]["page"]["signals"] is True
    events = [m["event"] for m in found if m["type"] == "event"]
    assert all(e["session_id"] == "demo-1" for e in events)
    assert events[0]["type"] == "session_started"
    records = [m["record"] for m in found if m["type"] == "suggestion"]
    assert len(records) == 2 and len(logs[0].records) == 2
    first = records[0]
    assert first["stages"]["total_ms"] is not None
    assert first["retrieved"][0]["id"] == "k1"
    end = live.get("demo-1")
    assert end["state"] == "ended" and end["calls"] == {"ok": 2}
    assert end["events"] == len(events)
    assert live.list_sessions()[0]["id"] == "demo-1"


def test_one_session_at_a_time_and_bad_requests() -> None:
    runner = InlineRunner(hold=True)
    live = sessions(runner=runner)
    live.start_replay("a", "live_assist.support", "t", 1.0)
    cases: List[Tuple[Callable[[], Any], str]] = [
        (lambda: live.start_replay("b", "live_assist.support", "t"), "busy"),
        (lambda: live.start_replay("a", "live_assist.support", "t"),
         "wrong_state"),
        (lambda: live.start_replay("c", "post_call", "t"), "not_found"),
        (lambda: live.start_replay("c", "live_assist.support", "nope"),
         "not_found"),
        (lambda: live.start_replay("c", "live_assist.support", "t", 99.0),
         "invalid_request"),
        (lambda: live.start_replay("../x", "live_assist.support", "t"),
         "invalid_request"),
        (lambda: live.get("zzz"), "not_found"),
        (lambda: live.start_live("live_assist.support", "stereo"),
         "unavailable"),
    ]
    for call, code in cases:
        with pytest.raises(SessionError) as caught:
            call()
        assert caught.value.code == code
    stopped = live.stop("a")
    assert stopped["state"] == "running"
    runner.jobs.pop()()
    assert live.get("a")["state"] == "ended"
    assert live.stop("a")["state"] == "ended"


def test_a_live_session_starts_on_the_transcriber_and_follows_it() -> None:
    t = transcript()
    roles = {"a": "agent", "c": "customer"}
    rename = {"a": "S1", "c": "S2"}
    documents = [e.to_contract() for e in transcript_to_events(t, roles)]
    for doc in documents:
        doc["session_id"] = "live-1"
        if "utterance" in doc:
            doc["utterance"]["speaker"] = rename[doc["utterance"]["speaker"]]
    documents.insert(2, {"schema": "live.event.v1", "broken": True})
    transcriber = FakeTranscriber()
    events = FakeEvents(documents)
    live = sessions(reasoner=SignalReasoner(), transcriber=transcriber,
                    events=events)
    view = live.start_live("live_assist.support", "stereo")
    assert view["id"] == "live-1" and view["audio"] == "stereo"
    body = transcriber.bodies[0]
    assert body["channels"] == 2 and body["hang_ms"] == 500
    assert body["speakers"] == [
        {"channel": 0, "id": "S1", "role": "agent"},
        {"channel": 1, "id": "S2", "role": "customer"}]
    assert body["handoff"] == {"transcript_v1": True,
                               "run_post_call": False}
    assert events.followed == ["live-1"]
    end = live.get("live-1")
    assert end["state"] == "ended" and end["rejected_events"] == 1
    found = messages(live, "live-1")
    assert found[1]["type"] == "transcriber"
    assert end["calls"] == {"ok": 2}
    assert end["skipped"] == {"role": 2, "min_words": 1}


def test_live_failures_are_reported() -> None:
    live = sessions(transcriber=FakeTranscriber(fail=True),
                    events=FakeEvents([]))
    with pytest.raises(SessionError) as caught:
        live.start_live("live_assist.support", "mic", "x-1")
    assert caught.value.code == "transcriber"
    assert live.get("x-1")["state"] == "failed"
    with pytest.raises(SessionError):
        live.start_live("live_assist.support", "radio", "x-2")
    broken = sessions(transcriber=FakeTranscriber(),
                      events=FakeEvents([], fail=True))
    broken.start_live("live_assist.support", "mic", "x-3")
    view = broken.get("x-3")
    assert view["state"] == "failed" and view["error"] == "closed 4404"


def test_stopping_a_live_session_asks_the_transcriber() -> None:
    runner = InlineRunner(hold=True)
    transcriber = FakeTranscriber()
    live = sessions(runner=runner, transcriber=transcriber,
                    events=FakeEvents([]))
    live.start_live("live_assist.interview", "mic", "s-1")
    live.stop("s-1")
    assert transcriber.stopped == ["s-1"]
    transcriber.fail = True
    live2 = sessions(runner=InlineRunner(hold=True),
                     transcriber=transcriber, events=FakeEvents([]))
    transcriber.fail = False
    live2.start_live("live_assist.interview", "mic", "s-2")
    transcriber.fail = True
    assert live2.stop("s-2")["state"] == "running"


def test_speakers_for_each_audio_kind() -> None:
    support = product()
    assert live_speakers(support, "mic") == [
        {"channel": 0, "id": "S2", "role": "customer"}]
    interview = product("live_assist.interview")
    assert live_speakers(interview, "mic")[0]["role"] == "candidate"
    body = transcriber_body(interview, "s", live_speakers(interview, "mic"))
    assert body["channels"] == 1 and body["partial_every_ms"] == 800
    assert body["product"] == "live_assist.interview"


def test_quota_stops_further_calls() -> None:
    live = sessions(reasoner=SignalReasoner(status="quota"))
    live.start_replay("q", "live_assist.support", "t", None)
    view = live.get("q")
    assert view["calls"] == {"quota": 1}
    assert view["skipped"]["halted"] == 1
