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
from orchestrator.domain.events import LiveEvent
from orchestrator.domain.triggers import is_late
from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.application.live import (
    LiveOptions,
    LiveSettings,
    LiveTranscription,
)
from transcriber.application.live_session import (
    LivePorts,
    LiveSession,
    LiveSessions,
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
    PaceWatch,
    align_samples,
    gap_fits,
    gap_samples,
    source_grace,
)


class ManualClock:
    """A monotonic clock the test moves; sleeps are not taken."""

    def __init__(self) -> None:
        self.t = 5000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        pass


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
    session.release_source(again)
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
    session.stop()
    assert session.wait(WAIT)
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
    session.release_source(new)
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


def engine_on(timer: VirtualTimer) -> Any:
    """A real-time engine on ``timer`` and the events it emits."""
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(growing), TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=True),
    )
    events: List[Dict[str, Any]] = []
    engine = live.engine("s", [LiveSpeaker("S1")], events.append)
    engine.start()
    return engine, events


def test_a_catching_up_engine_skips_partials_but_its_finals_come() -> None:
    for skip in (False, True):
        timer = VirtualTimer()
        engine, events = engine_on(timer)
        engine.skip_partials = skip
        assert engine.skip_partials is skip
        engine.push([bursts([(0.2, 2.5)], 3.5)], arrived=timer.t)
        engine.finish()
        kinds = [e["type"] for e in events]
        assert "final" in kinds
        assert ("partial" in kinds) is not skip


def test_lateness_is_a_backlog_not_a_lag_cpu_mode_keeps_its_partials(
) -> None:
    """#263 review (minor 6): a slow recogniser is late by seconds, but
    only a session catching up after its source came back skips
    partials; lateness alone never does."""
    timer = VirtualTimer()
    engine, events = engine_on(timer)
    engine.push([bursts([(0.2, 2.5)], 3.5)], arrived=timer.t - 8.0)
    engine.finish()
    assert "partial" in [e["type"] for e in events]


def run_paused(realign: bool) -> List[Dict[str, Any]]:
    """The reviewer's probe: 3 s on time, the microphone produces
    nothing for 30 s, then on-time speech in 100 ms pieces."""
    timer = VirtualTimer()
    engine, events = engine_on(timer)
    first = bursts([(0.2, 1.5)], 3.0)
    for at in range(0, len(first), 1600):
        timer.t += 0.1
        engine.push([first[at:at + 1600]], arrived=timer.t)
    timer.t += 30.0
    pace = PaceWatch()
    later = bursts([(1.5, 3.5)], 4.5)
    for at in range(0, len(later), 1600):
        timer.t += 0.1
        engine.push([later[at:at + 1600]], arrived=timer.t)
        if realign and pace.observe(1600, timer.t, True):
            engine.realign(timer.t)
    engine.finish()
    out: List[Dict[str, Any]] = events
    return out


def test_a_microphone_pause_does_not_make_later_speech_late_for_good(
) -> None:
    stale = [e for e in run_paused(False) if e["type"] == "final"]
    assert stale[-1]["timings"]["lag_ms"] > 25000, "the probe's failure"
    events = run_paused(True)
    finals = [e for e in events if e["type"] == "final"]
    assert finals[-1]["timings"]["lag_ms"] < 2000
    late = LiveEvent.model_validate(finals[-1])
    assert not is_late(late), "suggestions come again"
    later = [e for e in events if e["type"] == "partial"
             and e["stream_time"] > 4.5]
    assert later and all(e["timings"]["lag_ms"] < 2000 for e in later)


def test_realign_never_moves_back_nor_for_a_small_drift() -> None:
    timer = VirtualTimer()
    engine, _ = engine_on(timer)
    engine.push([np.zeros(16000, dtype=np.float32)], arrived=timer.t)
    assert not engine.realign(timer.t + 0.5), "within the slack"
    assert not engine.realign(timer.t - 10.0), "never backwards"
    assert engine.realign(timer.t + 5.0)


def test_the_pace_watch_wants_steady_pieces_with_nothing_queued() -> None:
    watch = PaceWatch(steady=3)
    t = 0.0
    assert not watch.observe(1600, t, True)
    for _ in range(2):
        t += 0.1
        assert not watch.observe(1600, t, True)
    t += 0.1
    assert watch.observe(1600, t, True)
    t += 0.001  # a burst of resent audio
    assert not watch.observe(1600, t, True)
    t += 0.1
    assert not watch.observe(1600, t, False), "something waits behind it"
    watch.reset()
    assert not watch.observe(1600, t + 0.1, True)


def test_a_session_re_anchors_once_its_audio_is_on_pace_again() -> None:
    """Through the worker: after a 30 s pause with no audio at all, the
    finals of on-time speech are on time and partials are made."""
    clock = ManualClock()

    def build(options: LiveOptions) -> LiveTranscription:
        return LiveTranscription(
            LoudnessVad, ScriptedAsr(growing), SystemClock(), clock,
            options.settings,
        )

    live = LiveSessions(
        LivePorts(build=build, clock=SystemClock(), timer=clock),
        LiveServiceSettings(idle_seconds=WAIT), "cpu",
    )
    session = live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    assert session.wait_ready(WAIT)

    def send(audio: Any) -> None:
        for at in range(0, len(audio), 1600):
            clock.t += 0.1
            session.feed(pcm(audio[at:at + 1600]), source)
            wait_until(lambda: session.idle()
                       and session._backlog.seconds < 1e-9)

    send(bursts([(0.2, 1.5)], 3.0))
    clock.t += 30.0
    send(bursts([(1.5, 3.5)], 4.5))
    session.release_source(source)
    live.stop("a", WAIT)
    events = session.hub.history()
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 2
    assert finals[-1]["timings"]["lag_ms"] < 2000
    assert any(e["type"] == "partial" and e["stream_time"] > 4.5
               for e in events)


def test_a_gap_cannot_take_the_stream_past_the_wall_clock() -> None:
    assert gap_fits(None, 0, 5 * SAMPLE_RATE)
    assert not gap_fits(None, 0, 6 * SAMPLE_RATE)
    assert gap_fits(60.0, 30 * SAMPLE_RATE, 35 * SAMPLE_RATE)
    assert not gap_fits(60.0, 30 * SAMPLE_RATE, 36 * SAMPLE_RATE)
    session, clock = lone_session()
    source = session.claim_source()
    session.feed(pcm(np.zeros(SAMPLE_RATE, dtype=np.float32)), source)
    clock["t"] += 9.0
    with pytest.raises(InvalidInputError):
        session.silence(3600 * SAMPLE_RATE, source)
    assert session.silence(9 * SAMPLE_RATE, source) == 9 * SAMPLE_RATE


def test_a_page_that_left_gets_a_shorter_grace_than_a_dropped_link() -> None:
    assert source_grace(1001, 600.0, 120.0) == 120.0
    for code in (None, 1000, 1005, 1006, 1011):
        assert source_grace(code, 600.0, 120.0) == 600.0
    harness = Harness(LiveServiceSettings(
        idle_seconds=WAIT, source_grace_seconds=WAIT,
        source_gone_seconds=0.2,
    ), device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    session.release_source(source, 1001)
    assert session.wait(WAIT)
    assert session.end_reason == "source_lost"


def test_a_stop_over_http_lets_the_source_finish_sending() -> None:
    harness = Harness(LiveServiceSettings(stop_drain_seconds=WAIT),
                      device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    source = session.claim_source()
    stopped = threading.Event()

    def stop() -> None:
        harness.live.stop("a", WAIT)
        stopped.set()

    threading.Thread(target=stop, daemon=True).start()
    time.sleep(0.2)
    session.feed(pcm(bursts([(0.0, 0.5)], 0.5)), source)  # still taken
    assert not stopped.is_set()
    session.stop()  # the page's own stop message
    assert stopped.wait(WAIT)
    assert session.received == 8000
    capped = Harness(LiveServiceSettings(stop_drain_seconds=0.1),
                     device="cpu")
    other = capped.live.start(LiveSessionRequest(session_id="b"))
    other.claim_source()
    capped.live.stop("b", WAIT)
    assert other.state == LiveState.ENDED, "the cap ends it anyway"
