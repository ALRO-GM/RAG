"""Nothing is lost at the end of a live session or on a crash (#264).

The worker dies mid-session: what was heard is still kept and handed
over (``partial``), with a warning to the readers. A stopped process
leaves a session with events and no transcript: at the next start it
is rebuilt from ``events.jsonl`` and ``session.json`` and handed over
(``recovered``). Publishing happens after the end. Fakes only.
"""

import threading
from typing import Any, Dict, List

from tests.unit.transcriber.live_fakes import (
    MemoryLiveLog,
    RecordingAnalysis,
    RecordingPublisher,
    bursts,
    chunks_of,
    pcm,
)
from tests.unit.transcriber.test_live_sessions import WAIT, Harness, speech
from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.application.live_handover import HandOver, facts_of
from transcriber.application.live_session import LiveState
from transcriber.domain.live_session import (
    ChannelSpeaker,
    Handoff,
    LiveSessionRequest,
)

ASKED = Handoff(transcript_v1=True, run_post_call=True)


def test_a_worker_that_dies_mid_session_still_hands_its_transcript_over(
) -> None:
    """Speech is heard, then the worker meets an error it cannot handle
    (here a broken piece of audio; on the GPU CUDA running out of
    memory): the session fails, but its transcript so far is kept and
    published (``partial``) and the readers are told."""
    publisher, analysis = RecordingPublisher(), RecordingAnalysis()
    harness = Harness(device="cpu", publisher=publisher, analysis=analysis)
    session = harness.live.start(LiveSessionRequest(
        session_id="oom", handoff=ASKED))
    for message in chunks_of(speech(), 3200):
        session.feed(message)
    for _ in range(500):
        if any(e["type"] == "final" for e in session.hub.history()):
            break
        threading.Event().wait(0.01)
    session._audio.put(([object()], 0.0))  # type: ignore[list-item]
    assert session.wait(WAIT)
    assert session.state == LiveState.FAILED
    assert session.error == "AttributeError" and session.end_reason == "failed"
    assert session.wait_handed_over(WAIT)
    summary = session.info()["transcript"]
    assert summary["partial"] is True and summary["utterances"] >= 1
    assert summary["published"] and analysis.started == ["oom"]
    assert publisher.documents == [harness.log.transcripts["oom"]]
    warnings = [e["warning"] for e in session.hub.history()
                if e["type"] == "warning"]
    assert warnings[-1]["code"] == "other"
    assert warnings[-1]["message"] == (
        "The live session stopped on an error (AttributeError); the"
        " transcript of what was heard until then is kept.")
    record = harness.log.sessions["oom"]
    assert record["state"] == "failed" and record["transcript"]["partial"]


def test_a_failure_before_anything_was_heard_hands_nothing_over() -> None:
    harness = Harness(device="cpu", publisher=RecordingPublisher())

    def broken(options: Any) -> Any:
        raise RuntimeError("no model")

    harness.live._p = harness.live._p.__class__(
        **{**harness.live._p.__dict__, "build": broken})
    session = harness.live.start(LiveSessionRequest(session_id="x"))
    assert session.wait(WAIT) and session.wait_handed_over(WAIT)
    assert session.state == LiveState.FAILED
    assert session.transcript is None and "x" not in harness.log.transcripts
    assert harness.log.sessions["x"]["state"] == "failed"


def test_the_record_follows_the_session_start_rename_and_end() -> None:
    harness = Harness(device="cpu")
    request = LiveSessionRequest(
        session_id="r", channels=2, speakers=[
            ChannelSpeaker(channel=0, id="S1", role="interviewer"),
            ChannelSpeaker(channel=1, id="S2", role="candidate"),
        ])
    session = harness.live.start(request)
    assert session.wait_ready(WAIT)
    for _ in range(200):
        if harness.log.sessions.get("r", {}).get("state") == "running":
            break
        threading.Event().wait(0.01)
    record = harness.log.sessions["r"]
    assert record["state"] == "running" and record["model"] == "small"
    assert record["compute"] == "int8"
    harness.live.rename("r", "S2", "Carla Ruiz")
    names = {s["id"]: s["name"] for s in harness.log.sessions["r"]["speakers"]}
    assert names["S2"] == "Carla Ruiz"
    harness.live.stop("r", WAIT)
    assert session.wait_handed_over(WAIT)
    final = harness.log.sessions["r"]
    assert final["state"] == "ended" and final["end_reason"] == "stop"
    assert "audio" not in final


def run_and_lose(log: MemoryLiveLog, session_id: str) -> List[Dict[str, Any]]:
    """A session that heard speech, as a process killed mid-session left
    it: its events and record, no transcript."""
    harness = Harness(device="cpu", log=log)
    session = harness.live.start(LiveSessionRequest(
        session_id=session_id, handoff=ASKED, speakers=[
            ChannelSpeaker(channel=0, id="S1", role="interviewer",
                           name="Ana")]))
    for message in chunks_of(speech(), 3200):
        session.feed(message)
    harness.live.stop(session_id, WAIT)
    assert session.wait_handed_over(WAIT)
    del log.transcripts[session_id]  # the process died before this
    record = log.sessions[session_id]
    record["state"] = "running"
    events = [e for e in log.events[session_id]
              if e["type"] != "session_ended"]
    log.events[session_id] = events
    return events


def test_a_restart_rebuilds_and_hands_over_what_a_dead_process_left() -> None:
    log = MemoryLiveLog()
    run_and_lose(log, "lost-1")
    publisher, analysis = RecordingPublisher(), RecordingAnalysis()
    again = Harness(device="cpu", log=log, publisher=publisher,
                    analysis=analysis)
    assert again.live.recover() == ["lost-1"]
    session = again.live.get("lost-1")
    assert session.info()["state"] == "recovered"
    assert session.info()["end_reason"] == "restart"
    for _ in range(500):
        if publisher.documents:
            break
        threading.Event().wait(0.01)
    document = log.transcripts["lost-1"]
    assert document["utterances"]
    assert {s["id"]: s.get("name") for s in document["speakers"]} \
        == {"S1": "Ana"}
    assert publisher.documents == [document]
    assert analysis.started == ["lost-1"]
    assert log.sessions["lost-1"]["state"] == "recovered"
    assert log.sessions["lost-1"]["transcript"]["recovered"] is True
    assert again.live.recover() == [], "once only: it has a transcript now"


def test_recovery_skips_running_sessions_and_reads_old_logs_without_record(
) -> None:
    log = MemoryLiveLog()
    run_and_lose(log, "old")
    del log.sessions["old"]  # kept before #264: no record
    harness = Harness(device="cpu", log=log, publisher=RecordingPublisher())
    running = harness.live.start(LiveSessionRequest(session_id="busy"))
    log.events["busy"] = [{"type": "session_started", "seq": 0}]
    assert harness.live.recover() == ["old"]
    assert log.transcripts["old"]["utterances"]
    assert "busy" not in log.transcripts
    facts = facts_of("old", log.events["old"], None)
    assert facts.request.handoff.transcript_v1 is False, \
        "without a record nobody asked to publish it: kept here only"
    harness.live.stop("busy", WAIT)
    del running


def test_a_record_that_cannot_be_read_is_left_alone() -> None:
    log = MemoryLiveLog()
    run_and_lose(log, "bad")
    log.sessions["bad"]["request"] = {"session_id": "../x"}
    handover = HandOver(log, None, None, SystemClock(), "9.9")
    assert handover.recover() == []
    assert "bad" not in log.transcripts


def test_publishing_comes_after_the_end_of_the_session() -> None:
    gate = threading.Event()

    class SlowPublisher(RecordingPublisher):
        def publish(self, document: Any) -> None:
            gate.wait(WAIT)
            super().publish(document)

    publisher = SlowPublisher()
    harness = Harness(device="cpu", publisher=publisher)
    session = harness.live.start(LiveSessionRequest(
        session_id="slow", handoff=Handoff(transcript_v1=True)))
    for message in chunks_of(pcm(bursts([(0.2, 1.6)], 2.6)), 3200):
        session.feed(message)
    harness.live.stop("slow", WAIT)
    assert session.state == LiveState.ENDED, "ended without the RAG"
    assert not session.handed_over.is_set()
    assert publisher.documents == []
    gate.set()
    assert session.wait_handed_over(WAIT)
    assert len(publisher.documents) == 1


def test_a_record_that_cannot_be_written_never_stops_a_session() -> None:
    harness = Harness(device="cpu", log=MemoryLiveLog(fail=True))
    session = harness.live.start(LiveSessionRequest(session_id="d"))
    harness.live.stop("d", WAIT)
    assert session.state == LiveState.ENDED


class MemoryMarker:
    """Keeps every announcement of the sessions that are on."""

    def __init__(self) -> None:
        self.marks: List[List[Dict[str, Any]]] = []

    def mark(self, sessions: Any) -> None:
        self.marks.append([dict(s) for s in sessions])


def test_the_sessions_that_are_on_are_announced_for_the_services_guard(
) -> None:
    from dataclasses import replace

    marker = MemoryMarker()
    harness = Harness(device="cpu")
    harness.live._p = replace(harness.live._p, marker=marker)
    session = harness.live.start(LiveSessionRequest(session_id="m"))
    assert marker.marks[0] == [{"id": "m", "state": "starting",
                                "source": "none"}]
    assert session.wait_ready(WAIT)
    assert not harness.live.source_connected()
    source = session.claim_source()
    assert marker.marks[-1][0]["source"] == "connected"
    assert harness.live.source_connected()
    session.release_source(source)
    assert marker.marks[-1] == [{"id": "m", "state": "running",
                                 "source": "lost"}], \
        "waiting for its source is still an interview on"
    assert not harness.live.source_connected()
    harness.live.stop("m", WAIT)
    assert marker.marks[-1] == []


def test_a_marker_that_cannot_be_written_never_stops_a_session() -> None:
    from dataclasses import replace

    class Broken:
        def mark(self, sessions: Any) -> None:
            raise OSError("read-only")

    harness = Harness(device="cpu")
    harness.live._p = replace(harness.live._p, marker=Broken())
    session = harness.live.start(LiveSessionRequest(session_id="b"))
    harness.live.stop("b", WAIT)
    assert session.state == LiveState.ENDED
