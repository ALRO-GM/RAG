"""A live session survives a lost audio source (#263), with fakes.

The source goes away and comes back (``source_lost`` / ``source_back``
warnings, the longer grace instead of the idle stop), a newer socket
takes the source over without any audio taken twice, audio the client
lost becomes silence, a reattached page is put back on the clock, and
late audio gets no partials while its finals still come. No GPU, no
real model: the fake VAD and recogniser of ``live_fakes``.
"""

import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from tests.unit.transcriber.live_fakes import (
    LoudnessVad,
    ScriptedAsr,
    TimerClock,
    VirtualTimer,
    bursts,
    pcm,
)
from tests.unit.transcriber.test_live_sessions import (
    ENDPOINT,
    WAIT,
    Harness,
    growing,
)
from transcriber.application.live import LiveSettings, LiveTranscription
from transcriber.application.live_session import (
    LiveSession,
    LiveState,
    SourceNotice,
)
from transcriber.domain.errors import (
    InvalidInputError,
    SourceReplacedError,
    StateError,
)
from transcriber.domain.live import SAMPLE_RATE, LiveSpeaker
from transcriber.domain.live_session import (
    MAX_GAP_SECONDS,
    LiveServiceSettings,
    LiveSessionRequest,
    align_samples,
    gap_samples,
)


def warnings(session: LiveSession) -> List[Dict[str, Any]]:
    """Return the warnings the session emitted so far."""
    return [
        e["warning"] for e in session.hub.history() if e["type"] == "warning"
    ]


def wait_until(check: Any, timeout: float = WAIT) -> None:
    """Poll ``check`` until it is true (the worker runs on a thread)."""
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_gap_and_align_sizes_are_checked() -> None:
    assert gap_samples(0) == 0 and gap_samples(16000) == 16000
    for bad in (-1, 1.5, "10", True, None,
                int(MAX_GAP_SECONDS * SAMPLE_RATE) + 1):
        with pytest.raises(InvalidInputError):
            gap_samples(bad)
    assert align_samples(10.0, 16000 * 4, 16000 * 2) == 16000 * 4
    assert align_samples(1.0, 16000 * 4, 0) == 0
    assert align_samples(-3.0, 0, 0) == 0


def test_a_lost_source_waits_with_a_warning_and_comes_back() -> None:
    harness = Harness(LiveServiceSettings(
        idle_seconds=0.1, source_grace_seconds=WAIT,
    ), device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    session.feed(pcm(bursts([(0.1, 0.5)], 1.0)), source)
    session.release_source(source)
    wait_until(lambda: [w["code"] for w in warnings(session)]
               == ["source_lost"])
    time.sleep(0.4)  # longer than idle_seconds: the grace holds instead
    assert session.active
    again = session.claim_source()
    assert again is not None and again != source
    wait_until(lambda: len(warnings(session)) == 2)
    back = warnings(session)[1]
    assert back["code"] == "source_back"
    assert back["seconds"] >= 0.4
    session.feed(pcm(bursts([(0.0, 0.5)], 1.0)), again)
    assert session.received == 2 * SAMPLE_RATE
    harness.live.stop("a", WAIT)
    assert session.state == LiveState.ENDED
    assert session.end_reason == "stop"
    assert session.info()["received_samples"] == 2 * SAMPLE_RATE


def test_a_source_that_never_comes_back_ends_after_the_grace() -> None:
    harness = Harness(LiveServiceSettings(
        idle_seconds=WAIT, source_grace_seconds=0.2,
    ), device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    session.release_source(source)
    assert session.wait(WAIT)
    assert session.state == LiveState.ENDED
    assert session.end_reason == "source_lost"
    assert session.claim_source() is None


def test_without_a_lost_source_the_idle_stop_still_holds() -> None:
    harness = Harness(
        LiveServiceSettings(idle_seconds=0.05), device="cpu"
    )
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    assert session.wait(WAIT)
    assert session.end_reason == "idle"


def test_a_stop_is_never_a_lost_source() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    harness.live.stop("a", WAIT)
    session.release_source(source)
    assert [w["code"] for w in warnings(session)] == []


def test_a_replaced_socket_blocked_on_a_full_queue_is_let_go() -> None:
    gate = threading.Event()
    harness = Harness(
        LiveServiceSettings(queue_seconds=0.5, behind_seconds=10.0),
        device="cpu", gate=gate,
    )
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    old = session.claim_source()
    tone = pcm(bursts([(0.0, 0.25)], 0.25))
    outcome: List[str] = []

    def feed_all() -> None:
        try:
            for _ in range(40):
                session.feed(tone, old)
        except SourceReplacedError:
            outcome.append("replaced")

    feeder = threading.Thread(target=feed_all, daemon=True)
    feeder.start()
    time.sleep(0.3)
    taken = session.received
    new = session.claim_source()
    feeder.join(WAIT)
    assert outcome == ["replaced"]
    assert session.received == taken, "nothing of the old socket after it"
    gate.set()
    session.feed(tone, new)
    harness.live.stop("a", WAIT)


def lone_session(queue_seconds: float = 3600.0) -> Any:
    """A session with no worker, on a clock the test moves."""
    clock = {"t": 100.0}
    session = LiveSession(
        LiveSessionRequest(session_id="a"),
        LiveServiceSettings(queue_seconds=queue_seconds),
        [LiveSpeaker("S1")], lambda: clock["t"],
    )
    session.mark_ready()
    return session, clock


def test_a_gap_becomes_silence_and_keeps_the_count() -> None:
    session, _ = lone_session()
    source = session.claim_source()
    session.feed(pcm(bursts([(0.0, 0.5)], 0.5)), source)
    assert session.silence(int(2.5 * SAMPLE_RATE), source) == 40000
    assert session.received == 8000 + 40000
    assert session.gap_seconds == pytest.approx(2.5)
    queued = []
    while not session._audio.empty():
        queued.append(session._audio.get())
    lengths = [len(item[0][0]) for item in queued
               if isinstance(item, tuple)]
    assert lengths == [8000, 16000, 16000, 8000]
    assert all(not item[0][0].any() for item in queued[1:]
               if isinstance(item, tuple))
    with pytest.raises(InvalidInputError):
        session.silence(-5, source)


def test_a_reattached_page_is_put_back_on_the_clock() -> None:
    session, clock = lone_session()
    source = session.claim_source()
    assert session.align(0, source) == 0, "nothing before the first audio"
    session.feed(pcm(np.zeros(SAMPLE_RATE, dtype=np.float32)), source)
    clock["t"] += 9.0  # ten seconds after the first audio began
    filled = session.align(2 * SAMPLE_RATE, source)
    assert filled == 7 * SAMPLE_RATE
    assert session.received == 8 * SAMPLE_RATE
    assert session.align(0, source) == 2 * SAMPLE_RATE


def test_audio_after_the_end_is_refused() -> None:
    session, _ = lone_session()
    source = session.claim_source()
    session.stop()
    with pytest.raises(StateError):
        session.feed(b"\x00\x00", source)
    assert isinstance(session.next_audio(), type(None))


def test_the_source_notices_are_queued_in_order() -> None:
    session, clock = lone_session()
    source = session.claim_source()
    session.release_source(source)
    clock["t"] += 4.0
    session.claim_source()
    assert session.next_audio() == SourceNotice("lost")
    assert session.next_audio() == SourceNotice("back", 4.0)


def test_late_audio_gets_no_partials_but_its_finals_come() -> None:
    def run(catch_up: Any, late: float) -> List[str]:
        timer = VirtualTimer()
        live = LiveTranscription(
            LoudnessVad, ScriptedAsr(growing), TimerClock(timer), timer,
            LiveSettings(
                endpoint=ENDPOINT, realtime=True,
                catch_up_seconds=catch_up,
            ),
        )
        events: List[Dict[str, Any]] = []
        engine = live.engine("s", [LiveSpeaker("S1")], events.append)
        engine.start()
        engine.push([bursts([(0.2, 2.5)], 3.5)], arrived=timer.t - late)
        engine.finish()
        return [e["type"] for e in events]

    on_time = run(3.0, 0.0)
    assert "partial" in on_time and "final" in on_time
    late = run(3.0, 60.0)
    assert "partial" not in late and "final" in late
    assert "partial" in run(None, 60.0), "without the setting: unchanged"
