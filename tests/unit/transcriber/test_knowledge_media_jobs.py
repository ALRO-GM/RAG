"""The knowledge library's recordings: limits and job ids (KN-3a)."""

import pytest

from transcriber.application.knowledge_media import check_job_id
from transcriber.application.transcribe import (
    TranscriptionOptions,
    check_duration,
    minutes_text,
)
from transcriber.domain.errors import InvalidInputError


@pytest.mark.parametrize("seconds, text", [
    (59, "1 min"), (60, "1 min"), (61, "2 min"), (7200, "2 h 00 min"),
    (7801, "2 h 11 min"),
])
def test_minutes_text(seconds: float, text: str) -> None:
    """Rounded up to the minute; hours from one hour."""
    assert minutes_text(seconds) == text


def test_check_duration() -> None:
    """Only a recording longer than a given limit is refused."""
    check_duration(7200.0, 7200.0)
    check_duration(99999.0, None)
    with pytest.raises(InvalidInputError,
                       match="lasts 2 h 01 min; the limit is 2 h 00 min"):
        check_duration(7201.0, 7200.0)


def test_options_and_ids() -> None:
    """A limit must be positive; job ids follow their pattern."""
    assert TranscriptionOptions().max_seconds is None
    with pytest.raises(ValueError):
        TranscriptionOptions(max_seconds=0)
    assert check_job_id("kn-" + "a" * 32) == "kn-" + "a" * 32
    for bad in ("kn-x", "session-1", "", "kn-" + "A" * 32):
        with pytest.raises(InvalidInputError):
            check_job_id(bad)
