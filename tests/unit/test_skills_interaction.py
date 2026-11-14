"""Interaction pace: speaking rate and response latency from times."""

import pytest

from src.domain.skills.interaction import describe
from tests.fakes import make_transcript


def test_pace_from_timestamps() -> None:
    """Speaking rate and response latency need times on every turn."""
    transcript = make_transcript((
        ("Ana", "one two three four five six", 0.0, 3.0),
        ("Luis", "seven eight", 3.5, 4.5),          # responds after 0.5 s
        ("Ana", "nine ten eleven twelve", 4.2, 6.2),  # overlaps: 0 s
        ("Luis", "thirteen", 8.2, 8.7),             # responds after 2.0 s
    ))
    team = describe(transcript)
    ana, luis = team.of("Ana"), team.of("Luis")
    assert ana.words_per_minute == pytest.approx(60 * 10 / 5.0)
    assert luis.words_per_minute == pytest.approx(60 * 3 / 1.5)
    assert ana.median_response_seconds == 0.0
    assert luis.median_response_seconds == pytest.approx(1.25)


def test_pace_is_unknown_without_timestamps() -> None:
    """No times, no pace: nothing is invented."""
    team = describe(make_transcript((("Ana", "hello there"),
                                     ("Luis", "hi"))))
    assert team.of("Ana").words_per_minute is None
    assert team.of("Luis").median_response_seconds is None


def test_first_speaker_and_zero_length_turns_have_no_pace() -> None:
    """Nobody to respond to, and no time to divide by."""
    team = describe(make_transcript((("Ana", "hello", 1.0, 1.0),)))
    assert team.of("Ana").median_response_seconds is None
    assert team.of("Ana").words_per_minute is None
