"""Live sessions with fakes: the push engine, the socket-facing session,
backpressure, roles, the GPU lease and degraded mode, the live log and
the transcript handed over at the end (LV-2, LV-5, LV-7, LV-8, LV-11)."""

import threading
import time
from typing import Any, Dict, List, Optional

import jsonschema
import numpy as np
import pytest

from tests.unit.transcriber.live_fakes import (
    LIVE_SCHEMA,
    FakeFreeMemory,
    FakeLiveLease,
    LoudnessVad,
    MemoryLiveLog,
    RecordingAnalysis,
    RecordingPublisher,
    SchemaCheck,
    ScriptedAsr,
    TimerClock,
    VirtualTimer,
    bursts,
    chunks_of,
    pcm,
)
from transcriber.adapters.outbound.fake_models import (
    FakeStreamDiarizer,
    FakeVoiceEncoder,
)
from transcriber.adapters.outbound.system_clock import SystemClock
from transcriber.bootstrap import party_plan
from transcriber.application.live import (
    LiveOptions,
    LiveSettings,
    LiveTranscription,
    default_speakers,
)
from transcriber.application.live_diarization import (
    LiveNaming,
    MonoAttributor,
)
from transcriber.application.live_session import (
    EventHub,
    LivePorts,
    LiveSessions,
    LiveState,
)
from transcriber.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    PublishError,
    StateError,
)
from transcriber.domain.live import EndpointSettings, LiveSpeaker
from transcriber.domain.live_diarization import MonoDiarizationSettings
from transcriber.domain.voice_check import VoiceCheckSettings
from transcriber.domain.live_session import (
    ChannelSpeaker,
    Handoff,
    LiveServiceSettings,
    LiveSessionRequest,
    Parties,
)

ENDPOINT = EndpointSettings(hang_ms=500, min_speech_ms=0, pre_roll_ms=0)
WAIT = 10.0


def growing(seconds: float, final: bool) -> str:
    """One word per half second; a final ends with a period."""
    words = " ".join(["palabra"] * max(int(seconds * 2), 1))
    return words + "." if final else words


def engine_events(
    audio: List[Any], step: Optional[int], timer: VirtualTimer
) -> List[Dict[str, Any]]:
    """Run the engine over ``audio``, pushed ``step`` samples at a time."""
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(growing, timer, 0.1), TimerClock(timer),
        timer, LiveSettings(endpoint=ENDPOINT, realtime=False),
    )
    events: List[Dict[str, Any]] = []
    engine = live.engine("s", default_speakers(len(audio)), events.append)
    engine.start()
    size = step or len(audio[0])
    for first in range(0, len(audio[0]), size):
        engine.push([a[first:first + size] for a in audio])
    engine.finish()
    return events


def test_pushing_in_pieces_gives_the_same_events_as_at_once() -> None:
    audio = [bursts([(0.5, 2.0), (3.0, 4.5)], 6.0)]
    whole = engine_events(audio, None, VirtualTimer())
    pieces = engine_events(audio, 1600, VirtualTimer())
    assert [(e["type"], e.get("utterance", {}).get("text")) for e in whole] \
        == [(e["type"], e.get("utterance", {}).get("text")) for e in pieces]
    assert sum(e["type"] == "final" for e in whole) == 2


def test_the_engine_keeps_only_the_audio_it_may_need() -> None:
    timer = VirtualTimer()
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(growing, timer), TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=False),
    )
    engine = live.engine("s", [LiveSpeaker("S1")], lambda e: None)
    with pytest.raises(InvalidInputError):
        engine.push([np.zeros(512, dtype=np.float32)])
    engine.start()
    with pytest.raises(InvalidInputError):
        engine.push([np.zeros(512, dtype=np.float32)] * 2)
    silence = np.zeros(16000, dtype=np.float32)
    for _ in range(30):
        engine.push([silence])
    assert len(engine.streams[0].samples) <= 2 * 16000
    assert engine.stream_time == pytest.approx(30.0, abs=0.05)
    assert engine.summary.finals == 0


def test_audio_that_waited_counts_as_late() -> None:
    timer = VirtualTimer()
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(growing), TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, realtime=True),
    )
    events: List[Dict[str, Any]] = []
    engine = live.engine("s", [LiveSpeaker("S1")], events.append)
    engine.start()
    audio = bursts([(0.5, 1.5)], 3.0)
    engine.push([audio], arrived=timer.t - 5.0)
    final = next(e for e in events if e["type"] == "final")
    assert final["timings"]["lag_ms"] >= 5000


def test_an_engine_needs_one_or_two_speakers() -> None:
    timer = VirtualTimer()
    live = LiveTranscription(
        LoudnessVad, ScriptedAsr(growing), TimerClock(timer), timer,
    )
    with pytest.raises(InvalidInputError):
        live.engine("s", [], lambda e: None)


def test_the_event_hub_waits_and_closes() -> None:
    hub = EventHub()
    assert hub.read(0, 0.01) == ([], False)
    hub.publish({"seq": 0})
    assert hub.read(0, 0.01) == ([{"seq": 0}], False)
    hub.close()
    assert hub.closed and hub.read(1, 5.0) == ([], True)


class Harness:
    """Live sessions over fakes, with every option a test may set."""

    def __init__(
        self,
        settings: LiveServiceSettings = LiveServiceSettings(
            ack_grace=1.0, ack_timeout=3.0
        ),
        device: str = "cuda",
        lease: Optional[FakeLiveLease] = None,
        memory: Optional[FakeFreeMemory] = None,
        log: Optional[MemoryLiveLog] = None,
        publisher: Optional[RecordingPublisher] = None,
        analysis: Optional[RecordingAnalysis] = None,
        asr_cost: float = 0.0,
        gate: Optional[threading.Event] = None,
        diarizer: Optional[FakeStreamDiarizer] = None,
        cpu_only: Optional[Any] = None,
    ) -> None:
        self.built: List[LiveOptions] = []
        self.timer = PollTimer()
        self.gate = gate
        self.log = log if log is not None else MemoryLiveLog()

        def build(options: LiveOptions) -> LiveTranscription:
            self.built.append(options)
            asr = ScriptedAsr(growing)
            if gate is not None:
                original = asr.decode

                def slow(samples: Any, final: bool) -> Any:
                    gate.wait(WAIT)
                    return original(samples, final)

                asr.decode = slow  # type: ignore[method-assign]

            def attributor(given: Any) -> MonoAttributor:
                assert diarizer is not None
                parties = party_plan(options, given, VoiceCheckSettings())
                return MonoAttributor(
                    diarizer, RealTimer(), MonoDiarizationSettings(),
                    LiveNaming(FakeVoiceEncoder(), [])
                    if parties is not None else None,
                    given, parties=parties,
                )

            return LiveTranscription(
                LoudnessVad, asr, SystemClock(), RealTimer(),
                options.settings,
                attributor if diarizer is not None else None,
            )

        self.live = LiveSessions(
            LivePorts(
                build=build, clock=SystemClock(), timer=self.timer,
                lease=lease, memory=memory, log=self.log,
                check=SchemaCheck(), publisher=publisher, analysis=analysis,
                cpu_only=cpu_only,
            ),
            settings, device, "9.9",
        )


class PollTimer:
    """Real monotonic time; sleeps (the GPU polls) are only counted."""

    def __init__(self) -> None:
        self.slept = 0.0

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.slept += seconds


class RealTimer:
    """Monotonic time; never sleeps (sessions are not paced)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        raise AssertionError("a live session is never paced")


def speech(channels: int = 1) -> bytes:
    """Two seconds of tone then a second of silence, per channel."""
    tone = bursts([(0.2, 1.6)], 2.6)
    other = bursts([(1.8, 2.4)], 2.6) if channels == 2 else None
    return pcm(tone) if other is None else pcm(tone, other)


def run_session(
    harness: Harness, request: LiveSessionRequest, data: bytes
) -> List[Dict[str, Any]]:
    """Start, feed ``data`` in 100 ms messages, stop; return the events."""
    session = harness.live.start(request)
    for message in chunks_of(data, 3200 * request.channels):
        session.feed(message)
    harness.live.stop(request.session_id, WAIT)
    assert session.state == LiveState.ENDED, session.error
    events = session.hub.history()
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    return events


def test_a_session_turns_pcm_messages_into_contract_events() -> None:
    harness = Harness(device="cpu")
    events = run_session(
        harness, LiveSessionRequest(session_id="m-1"), speech()
    )
    kinds = [e["type"] for e in events]
    assert kinds[0] == "session_started" and kinds[-1] == "session_ended"
    assert "final" in kinds
    assert [e["seq"] for e in events] == list(range(len(events)))
    session = harness.live.get("m-1")
    info = session.info()
    assert info["state"] == "ended" and info["finals"] >= 1
    assert info["device"] == "cpu" and info["model"] == "small"
    assert harness.built[0].settings.realtime
    assert harness.log.events["m-1"] == events
    assert harness.log.closed == ["m-1"]


def test_stereo_channels_carry_their_speakers_and_roles() -> None:
    harness = Harness(device="cpu")
    request = LiveSessionRequest(
        session_id="st-1", channels=2, product="live_assist.interview",
        speakers=[
            ChannelSpeaker(channel=0, id="S1", role="interviewer"),
            ChannelSpeaker(channel=1, id="S2", role="candidate"),
        ],
    )
    events = run_session(harness, request, speech(2))
    started = events[0]
    assert started["source"]["kind"] == "stereo_channels"
    assert [(s["id"], s["channel"], s["role"]) for s in started["speakers"]] \
        == [("S1", 0, "interviewer"), ("S2", 1, "candidate")]
    finals = [e["utterance"] for e in events if e["type"] == "final"]
    assert {(u["speaker"], u["channel"]) for u in finals} == {
        ("S1", 0), ("S2", 1)
    }
    assert harness.live.get("st-1").info()["product"] == \
        "live_assist.interview"


def test_sessions_are_unique_limited_and_found_by_id() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    with pytest.raises(StateError):
        harness.live.start(LiveSessionRequest(session_id="a"))
    with pytest.raises(BusyError):
        harness.live.start(LiveSessionRequest(session_id="b"))
    with pytest.raises(NotFoundError):
        harness.live.get("nope")
    harness.live.stop("a", WAIT)
    with pytest.raises(StateError):
        session.feed(b"\x00\x00")
    assert not session.claim_source()
    harness.live.start(LiveSessionRequest(session_id="b"))
    assert [s.id for s in harness.live.list()] == ["a", "b"]
    harness.live.stop("b", WAIT)


def test_old_ended_sessions_are_forgotten() -> None:
    harness = Harness(LiveServiceSettings(kept_sessions=1), device="cpu")
    for name in ("a", "b", "c"):
        harness.live.start(LiveSessionRequest(session_id=name))
        harness.live.stop(name, WAIT)
    assert [s.id for s in harness.live.list()] == ["b", "c"]


def test_one_audio_source_at_a_time() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    assert session.claim_source() and not session.claim_source()
    session.release_source()
    assert session.claim_source()
    harness.live.stop("a", WAIT)


def test_a_full_queue_blocks_then_refuses_and_warns_it_is_behind() -> None:
    gate = threading.Event()
    harness = Harness(
        LiveServiceSettings(
            queue_seconds=1.0, behind_seconds=0.5, feed_timeout=0.2
        ),
        device="cpu", gate=gate,
    )
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    tone = pcm(bursts([(0.0, 0.25)], 0.25))
    with pytest.raises(BusyError):
        for _ in range(20):
            session.feed(tone)
    gate.set()
    harness.live.stop("a", WAIT)
    codes = [
        e["warning"]["code"] for e in session.hub.history()
        if e["type"] == "warning"
    ]
    assert codes == ["asr_behind"]


def test_a_session_without_audio_stops_by_itself() -> None:
    harness = Harness(LiveServiceSettings(idle_seconds=0.05), device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    assert session.wait(WAIT)
    assert session.state == LiveState.ENDED


def test_bad_pcm_is_refused_without_ending_the_session() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(
        LiveSessionRequest(session_id="a", channels=2)
    )
    with pytest.raises(InvalidInputError):
        session.feed(b"\x00\x00")
    assert session.active
    harness.live.stop("a", WAIT)


def test_the_gpu_is_taken_with_the_ack_and_enough_memory() -> None:
    lease = FakeLiveLease(ack=True)
    memory = FakeFreeMemory([1000, 1000, 5000])
    harness = Harness(device="cuda", lease=lease, memory=memory)
    events = run_session(
        harness, LiveSessionRequest(session_id="g"), speech()
    )
    assert lease.acquired == [2200] and lease.released >= 1
    assert harness.built[0].device == "cuda"
    assert harness.built[0].model == "large-v3-turbo"
    assert harness.timer.slept == pytest.approx(1.0)
    assert not [e for e in events if e["type"] == "warning"]


def test_the_cpu_chosen_on_ajustes_skips_the_gpu_quietly() -> None:
    lease = FakeLiveLease(ack=True)
    chosen = {"cpu": True}
    harness = Harness(
        device="cuda", lease=lease, memory=FakeFreeMemory([9000]),
        cpu_only=lambda: chosen["cpu"],
    )
    events = run_session(
        harness, LiveSessionRequest(session_id="c"), speech()
    )
    assert (harness.built[0].device, harness.built[0].model) == (
        "cpu", "small"
    )
    assert lease.acquired == [], "no lease is asked for"
    assert not [e for e in events if e["type"] == "warning"]
    chosen["cpu"] = False
    run_session(harness, LiveSessionRequest(session_id="c2"), speech())
    assert harness.built[1].device == "cuda", "read at every start"


def test_too_little_memory_degrades_to_the_cpu_with_a_warning() -> None:
    lease = FakeLiveLease(ack=False)
    harness = Harness(
        device="cuda", lease=lease, memory=FakeFreeMemory([800])
    )
    events = run_session(
        harness, LiveSessionRequest(session_id="d"), speech()
    )
    options = harness.built[0]
    assert (options.device, options.model, options.compute_type) == (
        "cpu", "small", "int8"
    )
    warnings = [e for e in events if e["type"] == "warning"]
    assert [w["warning"]["code"] for w in warnings] == ["gpu_degraded"]
    assert "800 MiB free" in warnings[0]["warning"]["message"]
    assert "no ack" in warnings[0]["warning"]["message"]
    assert events[1] is warnings[0] or events[1] == warnings[0]
    assert lease.released >= 1
    assert harness.timer.slept == pytest.approx(3.0)


def test_a_lease_held_by_another_degrades_at_once() -> None:
    harness = Harness(
        device="cuda", lease=FakeLiveLease(held_by_other=True),
        memory=FakeFreeMemory([9000]),
    )
    events = run_session(
        harness, LiveSessionRequest(session_id="o"), speech()
    )
    assert harness.built[0].device == "cpu"
    warning = next(e for e in events if e["type"] == "warning")
    assert "someone else" in warning["warning"]["message"]


def test_no_gpu_reading_degrades_and_no_memory_port_trusts_the_gpu() -> None:
    harness = Harness(device="cuda", memory=FakeFreeMemory([None]))
    events = run_session(
        harness, LiveSessionRequest(session_id="n"), speech()
    )
    assert "no GPU found" in next(
        e for e in events if e["type"] == "warning"
    )["warning"]["message"]
    plain = Harness(device="cuda")
    run_session(plain, LiveSessionRequest(session_id="p"), speech())
    assert plain.built[0].device == "cuda"


def test_events_that_break_the_contract_are_not_sent() -> None:
    harness = Harness(device="cpu")

    class Strict:
        def problem(self, event: Dict[str, Any]) -> Optional[str]:
            return "no" if event["type"] == "final" else None

    harness.live._p = harness.live._p.__class__(
        **{**harness.live._p.__dict__, "check": Strict()}
    )
    events = run_session(
        harness, LiveSessionRequest(session_id="c"), speech()
    )
    assert not [e for e in events if e["type"] == "final"]
    assert harness.live.get("c").info()["rejected_events"] >= 1


def test_a_failing_log_does_not_stop_the_session() -> None:
    harness = Harness(device="cpu", log=MemoryLiveLog(fail=True))
    events = run_session(
        harness, LiveSessionRequest(session_id="f"), speech()
    )
    assert events[-1]["type"] == "session_ended"


def test_the_transcript_is_kept_published_and_analysed() -> None:
    publisher, analysis = RecordingPublisher(), RecordingAnalysis()
    harness = Harness(device="cpu", publisher=publisher, analysis=analysis)
    request = LiveSessionRequest(
        session_id="h", handoff=Handoff(transcript_v1=True,
                                        run_post_call=True),
    )
    run_session(harness, request, speech())
    document = harness.log.transcripts["h"]
    assert document["schema"] == "transcript.v1"
    assert document["utterances"]
    assert document["pipeline"]["recogniser"] == "faster-whisper small int8"
    assert document["pipeline"]["transcriber"] == "9.9"
    assert publisher.documents == [document]
    assert analysis.started == ["h"]
    summary = harness.live.get("h").info()["transcript"]
    assert summary["published"] and summary["analysis"] == "started"


def test_the_transcript_stays_home_unless_asked() -> None:
    publisher = RecordingPublisher()
    harness = Harness(device="cpu", publisher=publisher)
    run_session(harness, LiveSessionRequest(session_id="h"), speech())
    assert publisher.documents == []
    assert harness.log.transcripts["h"]["utterances"]


@pytest.mark.parametrize("publisher,analysis,error", [
    (None, None, "no RAG to publish to"),
    (RecordingPublisher(PublishError("refused")), None, "refused"),
    (RecordingPublisher(), None, "no RAG analysis configured"),
])
def test_handoff_problems_are_recorded_not_raised(
    publisher: Optional[RecordingPublisher],
    analysis: Optional[RecordingAnalysis], error: str,
) -> None:
    harness = Harness(device="cpu", publisher=publisher, analysis=analysis)
    request = LiveSessionRequest(
        session_id="h", handoff=Handoff(transcript_v1=True,
                                        run_post_call=True),
    )
    run_session(harness, request, speech())
    assert harness.live.get("h").info()["transcript"]["error"] == error


def test_a_failing_recogniser_fails_the_session() -> None:
    harness = Harness(device="cpu")

    def broken(options: LiveOptions) -> LiveTranscription:
        raise RuntimeError("no model")

    harness.live._p = harness.live._p.__class__(
        **{**harness.live._p.__dict__, "build": broken}
    )
    session = harness.live.start(LiveSessionRequest(session_id="x"))
    assert session.wait(WAIT)
    assert session.state == LiveState.FAILED
    assert session.error == "RuntimeError"
    assert harness.log.closed == ["x"]
    with pytest.raises(StateError):
        session.feed(b"\x00\x00")


def test_a_session_is_ready_for_audio_once_the_recogniser_runs() -> None:
    harness = Harness(LiveServiceSettings(idle_seconds=0.05), device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="a"))
    assert session.wait_ready(WAIT)
    assert session.wait(WAIT)
    assert session.wait_ready(0)  # an ended session never holds a socket


# ------------------------------------------------------------ mono (LV-6)

def two_voices() -> bytes:
    """A low voice, a pause, a high voice, a pause."""
    rate = 16000
    out = np.zeros(int(4.2 * rate), dtype=np.float32)
    for start, end, hz in ((0.2, 1.4, 220.0), (2.2, 3.4, 880.0)):
        a, b = int(start * rate), int(end * rate)
        t = np.arange(b - a, dtype=np.float32) / rate
        out[a:b] = 0.5 * np.sin(2 * np.pi * hz * t)
    return pcm(out)


def test_a_diarized_mono_session_names_each_voice() -> None:
    lease = FakeLiveLease(ack=True)
    diarizer = FakeStreamDiarizer(lag_frames=3)
    publisher = RecordingPublisher()
    harness = Harness(
        device="cuda", lease=lease, memory=FakeFreeMemory([9000]),
        diarizer=diarizer, publisher=publisher,
    )
    request = LiveSessionRequest(
        session_id="mono", diarize=True,
        speakers=[ChannelSpeaker(channel=0, id="agente", role="agent"),
                  ChannelSpeaker(channel=0, id="cliente", role="customer")],
        handoff=Handoff(transcript_v1=True),
    )
    events = run_session(harness, request, two_voices())
    assert events[0]["source"]["kind"] == "mono_diarized"
    assert lease.acquired == [2200 + 1500]
    assert harness.built[0].diarize and harness.built[0].enrolled is None
    finals = [e["utterance"] for e in events if e["type"] == "final"]
    updates = [e["utterance"] for e in events
               if e["type"] == "speaker_update"]
    final_speakers = {u["id"]: u["speaker"] for u in finals}
    final_speakers.update({u["id"]: u["speaker"] for u in updates})
    assert sorted(final_speakers.values()) == ["agente", "cliente"]
    info = harness.live.get("mono").info()
    assert info["diarizer"] == "fake/stream-diarizer"
    assert [s["role"] for s in info["speakers"][:2]] == ["agent", "customer"]
    document = publisher.documents[0]
    assert document["pipeline"]["diarizer"] == "fake/stream-diarizer"
    assert sorted(s["id"] for s in document["speakers"]) == [
        "agente", "cliente",
    ]
    assert diarizer.started == 1 and diarizer.closed >= 1


def test_a_mono_session_without_a_diarizer_fails_cleanly() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(
        LiveSessionRequest(session_id="nod", diarize=True)
    )
    assert session.wait(WAIT)
    assert session.state == LiveState.FAILED
    assert "diarizer" in (session.error or "")


def test_diarize_requests_are_checked() -> None:
    with pytest.raises(ValueError):
        LiveSessionRequest(session_id="x", channels=2, diarize=True)
    with pytest.raises(ValueError):
        LiveSessionRequest(session_id="x", enrolled=["ana"])
    with pytest.raises(ValueError):
        LiveSessionRequest(
            session_id="x", diarize=True,
            speakers=[ChannelSpeaker(channel=1, id="a")],
        )
    with pytest.raises(ValueError):
        LiveSessionRequest(
            session_id="x", diarize=True,
            speakers=[ChannelSpeaker(channel=0, id=f"p{n}")
                      for n in range(5)],
        )
    with pytest.raises(ValueError):
        LiveSessionRequest(
            session_id="x", diarize=True,
            speakers=[ChannelSpeaker(channel=0, id="a")] * 2,
        )
    ok = LiveSessionRequest(
        session_id="x", diarize=True, enrolled=[],
        speakers=[ChannelSpeaker(channel=0, id="a", role="agent")],
    )
    options = LiveSessions.options(ok, "cuda", "m", "int8")
    assert options.diarize and options.enrolled == []


def test_a_two_party_session_takes_voice_requests_and_keeps_numbers() -> None:
    # #249: the voice check's events, the page's requests, and the
    # attribution's numbers kept next to the events (no text).
    diarizer = FakeStreamDiarizer(lag_frames=3)
    harness = Harness(device="cpu", diarizer=diarizer)
    request = LiveSessionRequest(
        session_id="two", diarize=True, enrolled=[],
        speakers=[ChannelSpeaker(channel=0, id="ana", role="agent"),
                  ChannelSpeaker(channel=0, id="S2", role="customer")],
        parties=Parties(employee="ana", other="S2"),
    )
    session = harness.live.start(request)
    assert session.wait_ready(WAIT)
    assert harness.live.voice("two", "skip") is session
    with pytest.raises(InvalidInputError):
        harness.live.voice("two", "dance")
    with pytest.raises(InvalidInputError):
        harness.live.voice("two", "mine", "not-an-id")
    harness.live.voice("two", "mine", "t0.s0")
    for message in chunks_of(two_voices(), 3200):
        session.feed(message)
    harness.live.stop("two", WAIT)
    assert session.state == LiveState.ENDED, session.error
    events = session.hub.history()
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    assert [s["id"] for s in events[0]["speakers"]] == ["ana", "S2"]
    states = [e["voice_check"]["state"] for e in events
              if e["type"] == "voice_check"]
    assert states[0] == "listening" and "skipped" in states
    records = harness.log.attribution["two"]
    assert {r["event"] for r in records} >= {"final", "check", "sentence"}
    assert not any("text" in r for r in records)
    with pytest.raises(StateError):
        harness.live.voice("two", "skip")


def test_voice_requests_need_a_two_party_session() -> None:
    harness = Harness(device="cpu")
    session = harness.live.start(LiveSessionRequest(session_id="solo"))
    assert session.wait_ready(WAIT)
    with pytest.raises(StateError):
        harness.live.voice("solo", "skip")
    harness.live.stop("solo", WAIT)


def test_parties_are_checked() -> None:
    speakers = [ChannelSpeaker(channel=0, id="a"),
                ChannelSpeaker(channel=0, id="b")]
    with pytest.raises(ValueError):
        LiveSessionRequest(session_id="x", speakers=speakers,
                           parties=Parties(employee="a", other="b"))
    with pytest.raises(ValueError):
        LiveSessionRequest(session_id="x", diarize=True, speakers=speakers,
                           parties=Parties(employee="a", other="c"))
    with pytest.raises(ValueError):
        LiveSessionRequest(session_id="x", diarize=True, speakers=speakers,
                           parties=Parties(employee="a", other="a"))
    ok = LiveSessionRequest(session_id="x", diarize=True, speakers=speakers,
                            parties=Parties(employee="a", other="b"))
    options = LiveSessions.options(ok, "cuda", "m", "int8")
    assert options.parties is not None and options.parties.voice_check
