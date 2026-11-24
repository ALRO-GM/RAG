"""DM-1 (#233): captions and frame pacing of the demo video (pure logic)."""

from typing import Dict, List

import pytest

from tools.demo_video.timeline import (
    CAPTIONS,
    MAX_CAPTION_CHARS,
    Caption,
    Shown,
    fade_alpha,
    holds,
    jpeg_size,
    longest_hold,
    resample,
    resolve,
)

MARKS: Dict[str, float] = {
    "start": 0.0, "home": 2.4, "live": 11.8, "ended": 81.1,
    "result": 85.1, "closing": 110.9,
}
"""Marks as a recording keeps them (seconds from the video's start)."""


def test_the_scripted_captions_fit_a_recording() -> None:
    """The captions resolve on real marks, in order, within two minutes."""
    shown = resolve(CAPTIONS, MARKS, 113.8)
    assert len(shown) == len(CAPTIONS)
    assert [s.start for s in shown] == sorted(s.start for s in shown)
    assert all(len(s.text) <= MAX_CAPTION_CHARS for s in shown)
    texts = " ".join(s.text for s in shown)
    assert "la edad no puede influir en la selección" in texts
    assert "Sugerencias en tiempo real mientras entrevistas" in texts


def test_live_captions_stay_within_the_demo() -> None:
    """The live captions end before the scripted demo does (69.5 s)."""
    live = [c for c in CAPTIONS if c.mark == "live"]
    assert live and max(c.end for c in live) < 69.5


def test_resolve_moves_captions_to_their_mark() -> None:
    """A caption is shown at its mark plus its offsets."""
    shown = resolve([Caption("live", 1.0, 4.0, "Hola")], {"live": 10.0}, 20)
    assert shown == [Shown(11.0, 14.0, "Hola")]


@pytest.mark.parametrize("captions, message", [
    ([Caption("nope", 0, 3, "x")], "no mark"),
    ([Caption("live", 0, 3, "x" * (MAX_CAPTION_CHARS + 1))], "too long"),
    ([Caption("live", 0, 3, "two\nlines")], "too long"),
    ([Caption("live", 0, 1, "x")], "too short"),
    ([Caption("live", 0, 3, "a"), Caption("live", 2, 5, "b")], "overlap"),
    ([Caption("live", 0, 30, "x")], "after the end"),
])
def test_resolve_refuses_bad_captions(captions: List[Caption],
                                      message: str) -> None:
    """Missing marks, long or short lines, overlaps and late ends fail."""
    with pytest.raises(ValueError, match=message):
        resolve(captions, {"live": 0.0}, 20.0)


def test_resample_holds_the_last_frame_at_an_even_rate() -> None:
    """Each tick takes the last frame captured at or before it."""
    stamps = [100.0, 100.05, 100.30]
    assert resample(stamps, 100.0, 100.4, 10) == [0, 1, 1, 2]


def test_resample_runs_in_real_time() -> None:
    """The output lasts exactly the recorded time, however many frames."""
    stamps = [100.0 + i / 47 for i in range(47 * 3)]
    indices = resample(stamps, 100.0, 103.0, 30)
    assert len(indices) == 90
    assert indices == sorted(indices)


def test_resample_uses_the_first_frame_before_any_capture() -> None:
    """Ticks before the first frame show the first frame."""
    assert resample([5.0, 6.0], 4.0, 6.5, 2) == [0, 0, 0, 0, 1]


@pytest.mark.parametrize("stamps, start, end, fps", [
    ([], 0.0, 1.0, 30),
    ([1.0], 0.0, 1.0, 0),
    ([1.0], 2.0, 1.0, 30),
    ([2.0, 1.0], 0.0, 3.0, 30),
])
def test_resample_refuses_bad_input(stamps: List[float], start: float,
                                    end: float,
                                    fps: int) -> None:
    """No frames, a non-positive rate or times going back fail."""
    with pytest.raises(ValueError):
        resample(stamps, start, end, fps)


def test_holds_lists_frames_kept_on_screen() -> None:
    """A frame repeated for long enough is reported with its start."""
    indices = [0, 1, 1, 1, 1, 2, 3, 3]
    assert longest_hold(indices, 4) == 1.0
    assert holds(indices, 4, 0.5) == [(0.25, 1.0), (1.5, 0.5)]
    assert holds([], 4, 0.5) == []


def test_fade_alpha_is_an_ffmpeg_expression() -> None:
    """The expression fades in and out around the caption's times."""
    expression = fade_alpha(Shown(10.0, 14.0, "x"), 0.5)
    assert expression.startswith("if(lt(t,10.000),0,")
    assert "(t-10.000)/0.5" in expression
    assert "(14.000-t)/0.5" in expression


def test_jpeg_size_reads_the_frame_header() -> None:
    """Width and height come from the start-of-frame marker."""
    app0 = b"\xff\xe0\x00\x04\x00\x00"
    sof0 = b"\xff\xc0\x00\x11\x08" + (990).to_bytes(2, "big") + (
        1920).to_bytes(2, "big") + b"\x03"
    assert jpeg_size(b"\xff\xd8" + app0 + sof0) == (1920, 990)
    with pytest.raises(ValueError):
        jpeg_size(b"\xff\xd8not a jpeg")
