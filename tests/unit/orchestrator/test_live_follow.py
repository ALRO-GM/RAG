"""Nothing lost between the transcriber and the orchestrator (#264).

A dropped event stream is reopened from the next event, without handing
any event on twice; a refusal that cannot change ends it at once; a
stream down for too long gives up. A stop waits until the follower has
read the session's last events (the last finals, ``session_ended``)
before it cancels, so they reach the trackers and the judgement.
"""

import threading
from typing import Any, Dict, Iterator, List, Optional

import pytest

from orchestrator.adapters.outbound.runtime import ThreadRunner
from orchestrator.application import sessions as sessions_module
from orchestrator.application.event_follow import ResumingEvents, permanent
from orchestrator.application.ports import (
    CancelToken,
    Document,
    TranscriberUnavailable,
)
from orchestrator.application.replay import transcript_to_events
from tests.unit.orchestrator.fixtures import transcript
from tests.unit.orchestrator.test_sessions import (
    FakeTranscriber,
    sessions,
)


def ev(seq: int) -> Document:
    return {"seq": seq}


class DroppingSource:
    """Serves ``events``; each tuple of ``plan`` is one opening of the
    stream: how many events it gives, then how it ends (``None``: a
    normal close, or the exception it raises)."""

    def __init__(self, events: List[Document],
                 plan: List[Any]) -> None:
        self.events = events
        self.plan = list(plan)
        self.opened: List[int] = []

    def follow(self, session_id: str, stop: CancelToken,
               from_seq: int = 0) -> Iterator[Document]:
        self.opened.append(from_seq)
        count, ending = self.plan.pop(0)
        # a resumed stream may repeat an event the reader already had
        start = max(from_seq - 1, 0)
        for document in self.events[start:start + count]:
            yield document
        if ending is not None:
            raise ending


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.waits: List[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.t += seconds


def test_a_dropped_stream_is_reopened_where_it_was_and_nothing_twice() -> None:
    events = [ev(n) for n in range(10)]
    drop = TranscriberUnavailable("closed 1006: reset")
    source = DroppingSource(events, [(4, drop), (3, drop), (10, None)])
    clock = Clock()
    follower = ResumingEvents(source, clock.sleep, clock.now)
    got = list(follower.follow("s", CancelToken()))
    assert got == events
    assert source.opened == [0, 4, 6]
    assert clock.waits == [0.5, 0.5], "an event in between resets the wait"


def test_a_stream_down_for_too_long_gives_up_with_growing_waits() -> None:
    drop = TranscriberUnavailable("closed 1011: down")
    source = DroppingSource([ev(0)], [(1, drop)] + [(0, drop)] * 20)
    clock = Clock()
    follower = ResumingEvents(source, clock.sleep, clock.now, give_up_s=20)
    got: List[Document] = []
    with pytest.raises(TranscriberUnavailable):
        for document in follower.follow("s", CancelToken()):
            got.append(document)
    assert got == [ev(0)]
    assert clock.waits[:5] == [0.5, 1.0, 2.0, 4.0, 5.0]
    assert sum(clock.waits) < 25


def test_a_refusal_that_cannot_change_ends_it_at_once() -> None:
    for reason in ("closed 4404: no live session", "not_found",
                   "closed 4401: unauthorized"):
        exc = TranscriberUnavailable(reason)
        assert permanent(exc)
        source = DroppingSource([], [(0, exc)])
        with pytest.raises(TranscriberUnavailable):
            list(ResumingEvents(source, Clock().sleep, Clock().now).follow(
                "s", CancelToken()))
        assert source.opened == [0]
    assert not permanent(TranscriberUnavailable("ConnectionResetError"))


def test_a_stop_ends_the_following_without_retrying() -> None:
    stop = CancelToken()

    class Source:
        opened = 0

        def follow(self, session_id: str, token: CancelToken,
                   from_seq: int = 0) -> Iterator[Document]:
            Source.opened += 1
            yield ev(0)
            token.cancel()
            raise TranscriberUnavailable("closed 1006")

    follower = ResumingEvents(Source(), Clock().sleep, Clock().now)
    with pytest.raises(TranscriberUnavailable):
        list(follower.follow("s", stop))
    assert Source.opened == 1


def documents() -> List[Document]:
    roles = {"a": "agent", "c": "customer"}
    rename = {"a": "S1", "c": "S2"}
    found = [e.to_contract()
             for e in transcript_to_events(transcript(), roles)]
    for doc in found:
        doc["session_id"] = "live-1"
        if "utterance" in doc:
            doc["utterance"]["speaker"] = rename[doc["utterance"]["speaker"]]
    return found


class LateTail:
    """A stream whose last events come only after the transcriber was
    asked to stop (they were still on their way), and that stops
    yielding once cancelled, as the real one does."""

    def __init__(self, released: threading.Event) -> None:
        self.released = released
        self.all = documents()

    def follow(self, session_id: str, stop: CancelToken,
               from_seq: int = 0) -> Iterator[Document]:
        yield self.all[0]
        self.released.wait(10)
        threading.Event().wait(0.1)  # on their way while stop goes on
        for document in self.all[1:]:
            if stop.cancelled:
                return
            yield document


class StoppingTranscriber(FakeTranscriber):
    def __init__(self, released: threading.Event) -> None:
        super().__init__()
        self.released = released

    def stop(self, session_id: str) -> None:
        super().stop(session_id)
        self.released.set()


def run_stop(drain: Optional[float],
             monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    if drain is not None:
        monkeypatch.setattr(sessions_module, "STOP_DRAIN_S", drain)
    released = threading.Event()
    live = sessions(runner=ThreadRunner(),  # type: ignore[arg-type]
                    transcriber=StoppingTranscriber(released),
                    events=LateTail(released))  # type: ignore[arg-type]
    live.start_live("live_assist.support", "stereo")
    live.stop("live-1")
    return live.get("live-1")


def test_a_stop_waits_for_the_last_events_before_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = run_stop(None, monkeypatch)
    assert view["state"] == "ended"
    assert view["events"] == len(documents()), "every event reached it"
    assert view["calls"] == {"ok": 2}, "the last answers were followed"


def test_without_the_wait_the_last_events_were_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = run_stop(0.0, monkeypatch)
    assert view["events"] < len(documents())
