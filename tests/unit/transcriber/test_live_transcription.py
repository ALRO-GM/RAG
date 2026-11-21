"""The live transcription use case with a fake VAD, recogniser and timer."""

from typing import Any, Dict, List

import jsonschema
import numpy as np
import pytest

from tests.unit.transcriber.live_fakes import (
    LIVE_SCHEMA,
    LoudnessVad,
    ScriptedAsr,
    TimerClock,
    VirtualTimer,
    bursts,
)
from transcriber.application.live import (
    LiveSettings,
    LiveSummary,
    LiveTranscription,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import FRAME_SECONDS, EndpointSettings

ENDPOINT = EndpointSettings(hang_ms=500, min_speech_ms=0, pre_roll_ms=0)


def growing(seconds: float, final: bool) -> str:
    """One word per half second; a final ends with a period."""
    words = " ".join(["palabra"] * max(int(seconds * 2), 1))
    return words + "." if final else words


def run(
    channels: List[Any], asr: ScriptedAsr, timer: VirtualTimer,
    realtime: bool = True, partial_every_ms: int = 800,
) -> "tuple[List[Dict[str, Any]], LiveSummary, List[LoudnessVad]]":
    vads: List[LoudnessVad] = []

    def vad() -> LoudnessVad:
        vads.append(LoudnessVad())
        return vads[-1]

    live = LiveTranscription(
        vad, asr, TimerClock(timer), timer,
        LiveSettings(endpoint=ENDPOINT, partial_every_ms=partial_every_ms,
                     realtime=realtime),
    )
    events: List[Dict[str, Any]] = []
    summary = live.run("sess-1", channels, events.append)
    return events, summary, vads


def test_mono_stream_gives_partials_then_finals_with_timings() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer, cost=0.2)
    audio = bursts([(1.0, 4.0)], 6.0)
    events, summary, vads = run([audio], asr, timer)
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "session_started" and kinds[-1] == "session_ended"
    assert "partial" in kinds and kinds.count("final") == 1
    assert events[0]["source"] == {
        "kind": "single_speaker", "sample_rate": 16000, "channels": 1,
    }
    assert events[0]["speakers"][0]["channel"] is None
    assert [e["seq"] for e in events] == list(range(len(events)))
    final = next(e for e in events if e["type"] == "final")
    u = final["utterance"]
    assert u["id"] == "t0.s0" and u["speaker"] == "S1"
    assert u["start"] == pytest.approx(1.0, abs=FRAME_SECONDS)
    assert u["end"] <= 4.0 + FRAME_SECONDS
    partial_ids = {e["utterance"]["id"] for e in events
                   if e["type"] == "partial"}
    assert partial_ids == {"t0.s0"}
    assert u["revision"] > 0
    t = final["timings"]
    assert 500 <= t["endpoint_ms"] <= 500 + 2 * FRAME_SECONDS * 1000
    assert t["asr_ms"] == pytest.approx(200)
    assert t["lag_ms"] == pytest.approx(t["endpoint_ms"] + 200, abs=1)
    assert summary.finals == 1 and summary.partials >= 1
    assert summary.lag_ms == [pytest.approx(t["lag_ms"], abs=0.1)]
    assert asr.warm == 1 and asr.calls[-1][1] is True
    assert all(not final for _, final in asr.calls[:-1])
    assert vads[0].frames == int(6.0 / FRAME_SECONDS)


def test_real_time_paces_the_reading() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer, cost=0.0)
    run([bursts([], 3.0)], asr, timer)
    assert timer.slept == pytest.approx(3.0, abs=FRAME_SECONDS)


def test_slow_decoding_shows_in_the_lag() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer, cost=1.5)
    events, summary, _ = run([bursts([(0.5, 3.0)], 6.0)], asr, timer)
    final = next(e for e in events if e["type"] == "final")
    # The partials made the stream run behind: the endpoint is late.
    assert final["timings"]["endpoint_ms"] > 600
    assert final["timings"]["lag_ms"] >= final["timings"]["endpoint_ms"]


def test_fast_mode_reports_stream_time_hang() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer, cost=0.25)
    events, summary, _ = run(
        [bursts([(0.5, 2.0)], 4.0)], asr, timer, realtime=False
    )
    assert timer.slept == 0
    t = next(e for e in events if e["type"] == "final")["timings"]
    assert t["endpoint_ms"] == pytest.approx(512)
    assert t["lag_ms"] == pytest.approx(762)
    assert "lag_ms" not in next(
        e for e in events if e["type"] == "partial"
    )["timings"]


def test_stereo_has_one_speaker_per_channel_and_turns() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer, cost=0.05)
    left = bursts([(0.5, 2.0), (5.0, 6.0)], 8.0)
    right = bursts([(2.5, 4.0)], 8.0)
    events, summary, vads = run([left, right], asr, timer)
    for event in events:
        jsonschema.validate(event, LIVE_SCHEMA)
    assert events[0]["source"]["kind"] == "stereo_channels"
    assert [(s["id"], s["channel"]) for s in events[0]["speakers"]] == [
        ("S1", 0), ("S2", 1),
    ]
    finals = [e["utterance"] for e in events if e["type"] == "final"]
    assert [(u["id"], u["speaker"], u["channel"]) for u in finals] == [
        ("t0.s0", "S1", 0), ("t1.s0", "S2", 1), ("t2.s0", "S1", 0),
    ]
    assert len(vads) == 2 and summary.finals == 3


def test_speech_until_the_end_is_flushed() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer)
    events, summary, _ = run([bursts([(1.0, 3.0)], 3.0)], asr, timer)
    assert summary.finals == 1
    assert events[-2]["type"] == "final"


def test_silence_or_empty_text_gives_no_final() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(lambda seconds, final: "", timer)
    events, summary, _ = run([bursts([(1.0, 2.0)], 4.0)], asr, timer)
    assert summary.finals == 0 and summary.lag_ms == []
    assert [e["type"] for e in events] == [
        "session_started", "session_ended",
    ]


def test_channel_count_is_checked() -> None:
    timer = VirtualTimer()
    asr = ScriptedAsr(growing, timer)
    silent = np.zeros(1600, dtype=np.float32)
    with pytest.raises(InvalidInputError):
        run([], asr, timer)
    with pytest.raises(InvalidInputError):
        run([silent] * 3, asr, timer)
