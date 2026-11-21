"""Endpointing (LV-3): speech start and end from per-frame probabilities."""

from typing import Dict, List, Sequence

import pytest
from pydantic import ValidationError

from transcriber.domain.live import (
    FRAME_SECONDS,
    EndpointSettings,
    Endpointer,
    SpeechEnd,
    SpeechEvent,
    SpeechStart,
)


def feed(endpointer: Endpointer, probabilities: Sequence[float]) -> List[
    SpeechEvent
]:
    events: List[SpeechEvent] = []
    for p in probabilities:
        events.extend(endpointer.push(p))
    return events


def frames(seconds: float) -> int:
    return round(seconds / FRAME_SECONDS)


def test_speech_starts_after_min_speech_with_pre_roll() -> None:
    e = Endpointer(EndpointSettings(min_speech_ms=96, pre_roll_ms=64))
    events = feed(e, [0.0] * 10 + [0.9] * 2)
    assert events == [] and not e.speaking
    events = feed(e, [0.9])
    start = 10 * FRAME_SECONDS - 0.064
    (started,) = events
    assert isinstance(started, SpeechStart)
    assert started.start == pytest.approx(start)
    assert e.speaking and e.segment_start == pytest.approx(start)


def test_pre_roll_never_goes_before_zero() -> None:
    e = Endpointer(EndpointSettings(min_speech_ms=0, pre_roll_ms=500))
    assert feed(e, [0.9]) == [SpeechStart(0.0)]


def test_a_short_blip_does_not_start_speech() -> None:
    e = Endpointer(EndpointSettings(min_speech_ms=96))
    assert feed(e, [0.9, 0.9, 0.1, 0.9, 0.9, 0.1] * 3) == []


def test_speech_ends_after_the_hang_at_the_last_voiced_frame() -> None:
    e = Endpointer(EndpointSettings(hang_ms=300, min_speech_ms=0,
                                    pre_roll_ms=0))
    events = feed(e, [0.0] * 5 + [0.9] * 20)
    (started,) = events
    assert isinstance(started, SpeechStart)
    assert started.start == pytest.approx(5 * FRAME_SECONDS)
    hang = frames(0.3) + 1  # 300 ms is 9.4 frames: 10 frames
    events = feed(e, [0.1] * (hang - 1))
    assert events == [] and e.speaking
    (ended,) = feed(e, [0.1])
    assert isinstance(ended, SpeechEnd)
    assert ended.start == pytest.approx(5 * FRAME_SECONDS)
    assert ended.end == pytest.approx(25 * FRAME_SECONDS)
    assert ended.endpoint_at == pytest.approx((25 + hang) * FRAME_SECONDS)
    assert not ended.forced and not e.speaking
    assert ended.endpoint_at - ended.end >= 0.3


def test_hysteresis_keeps_speaking_above_release() -> None:
    e = Endpointer(EndpointSettings(threshold=0.5, release=0.3,
                                    hang_ms=300, min_speech_ms=0))
    feed(e, [0.9])
    assert feed(e, [0.4] * 50) == [] and e.speaking


def test_a_pause_shorter_than_the_hang_keeps_one_segment() -> None:
    e = Endpointer(EndpointSettings(hang_ms=500, min_speech_ms=0))
    events = feed(e, [0.9] * 10 + [0.0] * 10 + [0.9] * 10)
    assert len(events) == 1 and e.speaking


def test_long_speech_is_cut_at_the_maximum_and_continues() -> None:
    e = Endpointer(EndpointSettings(max_segment_seconds=2.0,
                                    min_speech_ms=0, pre_roll_ms=0))
    events = feed(e, [0.9] * frames(5.0))
    ends = [x for x in events if isinstance(x, SpeechEnd)]
    starts = [x for x in events if isinstance(x, SpeechStart)]
    assert len(ends) == 2 and all(x.forced for x in ends)
    assert len(starts) == 3
    assert ends[0].end - ends[0].start == pytest.approx(2.0, abs=0.04)
    assert starts[1].start == ends[0].end


def test_flush_closes_the_open_segment() -> None:
    e = Endpointer(EndpointSettings(min_speech_ms=0, pre_roll_ms=0))
    assert e.flush() is None
    feed(e, [0.9] * 10 + [0.1] * 3)
    ended = e.flush()
    assert ended is not None and ended.start == 0.0
    assert ended.end == pytest.approx(10 * FRAME_SECONDS)
    assert ended.endpoint_at == pytest.approx(13 * FRAME_SECONDS)
    assert not e.speaking and e.flush() is None


def test_time_counts_frames() -> None:
    e = Endpointer()
    feed(e, [0.0] * 4)
    assert e.time == pytest.approx(4 * FRAME_SECONDS)


@pytest.mark.parametrize("bad", [
    {"hang_ms": 299}, {"hang_ms": 701}, {"threshold": 0.3, "release": 0.4},
    {"threshold": 1.0}, {"max_segment_seconds": 0.5},
])
def test_settings_are_checked(bad: Dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        EndpointSettings.model_validate(bad)


def test_hang_range_is_accepted() -> None:
    assert EndpointSettings(hang_ms=300).hang_ms == 300
    assert EndpointSettings(hang_ms=700).hang_ms == 700
