"""The live session's side of a recovery (#265), fake runtime.

The record keeps the start time and every silence put in for lost audio
as it happens (so a crash keeps them), the end count at the end, the
product's leave to buffer; a recovery whose transcription holds the GPU
is in the live marker (the services guard refuses a restart), and
``busy`` tells whether a session is still on.
"""

from dataclasses import replace

from tests.unit.transcriber.test_live_handover import MemoryMarker
from tests.unit.transcriber.test_live_sessions import WAIT, Harness
from transcriber.domain.live_session import LiveSessionRequest

S = 16000


def test_silence_is_written_to_the_record_as_it_happens() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(LiveSessionRequest(
        session_id="g", buffer_audio=True, buffer_max_hours=12))
    assert session.wait_ready(WAIT)
    record = harness.log.sessions["g"]
    assert record["started_at"] and record["silences"] == []
    assert record["request"]["buffer_audio"] is True
    assert record["request"]["buffer_max_hours"] == 12
    session.feed(b"\x00\x00" * S)
    session.silence(2 * S)
    record = harness.log.sessions["g"]
    assert record["silences"] == [[S, 3 * S]], "kept before any end"
    assert "received_samples" not in record, "the end count only at the end"
    assert harness.live.busy("g") and not harness.live.busy("other")
    harness.live.stop("g", WAIT)
    assert session.wait_handed_over(WAIT)
    record = harness.log.sessions["g"]
    assert record["received_samples"] == 3 * S
    assert record["silences"] == [[S, 3 * S]]
    assert not harness.live.busy("g")


def test_a_recovery_holding_the_gpu_is_in_the_marker() -> None:
    marker = MemoryMarker()
    harness = Harness(device="cpu")
    harness.live._p = replace(harness.live._p, marker=marker)
    harness.live.recovering("iv-1", True)
    assert marker.marks[-1] == [{"id": "iv-1", "state": "recovering",
                                 "source": "none"}]
    count = len(marker.marks)
    harness.live.recovering("iv-1", True)  # no change: not written again
    assert len(marker.marks) == count
    harness.live.recovering("iv-1", False)
    assert marker.marks[-1] == []
