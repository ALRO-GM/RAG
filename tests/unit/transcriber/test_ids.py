"""Every id is checked by the use cases before it reaches a repository."""

from typing import Callable, List

import pytest

from tests.unit.transcriber.fakes import (
    FakeSessionRepository,
    FakeVoiceprintRepository,
    RecordingPublisher,
    voice,
)
from tests.unit.transcriber.test_transcribe import World, world
from transcriber.application.enrol import ForgetSpeaker
from transcriber.application.publish import PublishTranscript
from transcriber.application.record import (
    AppendAudio,
    CloseSession,
    DeleteSession,
)
from transcriber.application.review import RenameSpeaker, ReassignUtterance
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.domain.errors import InvalidInputError

BAD = "../x"


class SpyingSessions(FakeSessionRepository):
    """A session store that records every id it is asked about."""

    def __init__(self) -> None:
        """Start with no id seen."""
        super().__init__()
        self.seen: List[str] = []

    def get(self, session_id: str):  # type: ignore[no-untyped-def]
        """Record the id, then look it up."""
        self.seen.append(session_id)
        return super().get(session_id)


class SpyingVoiceprints(FakeVoiceprintRepository):
    """A voiceprint store that records every id it is asked about."""

    def __init__(self) -> None:
        """Start with no id seen."""
        super().__init__()
        self.seen: List[str] = []

    def get(self, speaker_id: str):  # type: ignore[no-untyped-def]
        """Record the id, then look it up."""
        self.seen.append(speaker_id)
        return super().get(speaker_id)

    def consent(self, speaker_id: str):  # type: ignore[no-untyped-def]
        """Record the id, then look it up."""
        self.seen.append(speaker_id)
        return super().consent(speaker_id)

    def delete(self, speaker_id: str) -> bool:
        """Record the id, then delete."""
        self.seen.append(speaker_id)
        return super().delete(speaker_id)


def finished() -> World:
    """A transcribed session ``s1`` whose stores record the ids asked."""
    voices = SpyingVoiceprints()
    voices.enrol("luis", voice(1))
    w = world(sessions=SpyingSessions(), voiceprints=voices)
    w.run()
    return w


def session_calls(w: World) -> List[Callable[[str], object]]:
    """Every session use case, called with the session id given."""
    rename = RenameSpeaker(w.sessions, w.voiceprints, w.locks)
    reassign = ReassignUtterance(w.sessions, w.voiceprints, w.locks)
    return [
        lambda s: AppendAudio(w.sessions, w.locks)(s, 0, b"\x00\x00"),
        lambda s: CloseSession(w.sessions, w.locks)(s),
        lambda s: DeleteSession(w.sessions, w.locks)(s),
        lambda s: w.queue(s),
        lambda s: w.transcribe(s),
        lambda s: w.status(s),
        lambda s: PublishTranscript(w.sessions, RecordingPublisher())(s),
        lambda s: rename(s, "S1", name="Ana"),
        lambda s: reassign(s, "t0.s0", "luis"),
    ]


@pytest.mark.parametrize("index", range(9))
def test_a_bad_session_id_is_invalid_input(index: int) -> None:
    w = finished()
    assert isinstance(w.sessions, SpyingSessions)
    w.sessions.seen.clear()
    with pytest.raises(InvalidInputError, match="session id"):
        session_calls(w)[index](BAD)
    assert BAD not in w.sessions.seen


def test_forget_checks_the_speaker_id() -> None:
    voices = SpyingVoiceprints()
    with pytest.raises(InvalidInputError, match="speaker id"):
        ForgetSpeaker(voices)(BAD)
    assert voices.seen == []


def test_expected_speakers_are_checked() -> None:
    w = finished()
    assert isinstance(w.voiceprints, SpyingVoiceprints)
    w.voiceprints.seen.clear()
    with pytest.raises(InvalidInputError, match="speaker id"):
        w.queue("s1", TranscriptionOptions(expected_speakers=["luis", BAD]))
    assert BAD not in w.voiceprints.seen


@pytest.mark.parametrize("call", ["merge", "label", "reassign"])
def test_review_checks_the_speaker_ids(call: str) -> None:
    w = finished()
    assert isinstance(w.voiceprints, SpyingVoiceprints)
    w.voiceprints.seen.clear()
    with pytest.raises(InvalidInputError, match="speaker id"):
        if call == "merge":
            RenameSpeaker(w.sessions, w.voiceprints, w.locks)(
                "s1", "S1", merge_into=BAD
            )
        elif call == "label":
            RenameSpeaker(w.sessions, w.voiceprints, w.locks)(
                "s1", BAD, name="Ana"
            )
        else:
            ReassignUtterance(w.sessions, w.voiceprints, w.locks)(
                "s1", "t0.s0", BAD
            )
    assert BAD not in w.voiceprints.seen
