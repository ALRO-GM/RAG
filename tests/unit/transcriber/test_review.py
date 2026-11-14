"""Transcriber review and publishing: rename, reassign, publish."""

from typing import List, Optional, Tuple

import pytest

from tests.unit.transcriber.fakes import RecordingPublisher, voice
from tests.unit.transcriber.test_transcribe import World, turns, world
from transcriber.application.publish import PublishTranscript
from transcriber.application.review import RenameSpeaker, ReassignUtterance
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StateError,
)
from transcriber.domain.models import SessionState, Transcript
from transcriber.domain.parameters import TranscriptionParameters


def finished(enrol_luis: bool = False) -> World:
    """A world whose session ``s1`` is transcribed: S1, S2/luis, S1."""
    w = world()
    if enrol_luis:
        w.voiceprints.enrol("luis", voice(1), name="Luis")
    w.run()
    return w


def renamer(w: World) -> RenameSpeaker:
    """The rename use case of ``w``."""
    return RenameSpeaker(w.sessions, w.voiceprints, w.locks)


def reassigner(w: World) -> ReassignUtterance:
    """The reassign use case of ``w``."""
    return ReassignUtterance(w.sessions, w.voiceprints, w.locks)


def stored(w: World) -> Transcript:
    """The stored transcript of ``s1``."""
    transcript = w.sessions.transcript("s1")
    assert transcript is not None
    return transcript


def speakers(transcript: Transcript) -> List[Tuple[str, Optional[str], bool]]:
    """``(id, name, enrolled)`` of every listed speaker."""
    return [(s.id, s.name, s.enrolled) for s in transcript.speakers]


class TestRenameSpeaker:
    def test_a_name_for_an_anonymous_voice(self) -> None:
        w = finished()
        before = stored(w)
        out = renamer(w)("s1", "S2", name=" Luis ")
        assert speakers(out) == [("S1", None, False), ("S2", "Luis", False)]
        assert out.utterances == before.utterances
        assert stored(w) == out
        assert w.voiceprints.voiceprints == {}

    def test_merge_into_a_speaker_of_the_session(self) -> None:
        w = finished()
        out = renamer(w)("s1", "S2", merge_into="S1")
        assert turns(out) == [("S1", None)]
        assert speakers(out) == [("S1", None, False)]
        assert [u.id for u in out.utterances] == [
            f"t0.s{n}" for n in range(len(out.utterances))
        ]
        assert {w.speaker for w in w.sessions.words("s1")} == {"S1"}

    def test_merge_into_an_enrolled_person_not_recognised(self) -> None:
        w = finished()
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        prints = dict(w.voiceprints.voiceprints)
        out = renamer(w)("s1", "S2", merge_into="luis")
        assert turns(out) == [("S1", None), ("luis", "S1"), ("S1", "luis")]
        assert speakers(out) == [("S1", None, False), ("luis", "Luis", True)]
        moved = [u for u in out.utterances if u.speaker == "luis"]
        assert {u.speaker_confidence for u in moved} == {None}
        assert w.voiceprints.voiceprints == prints   # no voiceprint made
        assert "S2" not in {w.speaker for w in w.sessions.words("s1")}

    def test_invalid_renames(self) -> None:
        w = finished(enrol_luis=True)
        rename = renamer(w)
        with pytest.raises(InvalidInputError, match="either"):
            rename("s1", "S1")
        with pytest.raises(InvalidInputError, match="either"):
            rename("s1", "S1", name="Ana", merge_into="luis")
        with pytest.raises(InvalidInputError, match="enrolled person"):
            rename("s1", "luis", name="Lucho")
        with pytest.raises(NotFoundError, match="S7"):
            rename("s1", "S7", name="Ana")
        with pytest.raises(NotFoundError, match="nadie"):
            rename("s1", "S1", merge_into="nadie")
        with pytest.raises(InvalidInputError, match="itself"):
            rename("s1", "S1", merge_into="S1")
        with pytest.raises(InvalidInputError, match="empty"):
            rename("s1", "S1", name="  ")

    def test_only_finished_sessions(self) -> None:
        w = world()
        w.upload()
        with pytest.raises(StateError, match="done"):
            renamer(w)("s1", "S1", name="A")


class TestReassignUtterance:
    def test_moving_one_sentence_recomputes_the_turns(self) -> None:
        w = finished()
        before = stored(w)
        last_of_first = [u for u in before.utterances if u.turn == 0][-1]
        out = reassigner(w)(
            "s1", last_of_first.id, "S2"
        )
        assert turns(out) == [("S1", None), ("S2", "S1"), ("S1", "S2")]
        moved = [u for u in out.utterances if u.turn == 1][0]
        assert moved.text == last_of_first.text
        assert moved.speaker_confidence is None
        assert moved.previous_utterance == "t0.s{}".format(
            last_of_first.index - 1
        )
        moved_words = {(x.text, x.start) for x in last_of_first.words}
        for word in w.sessions.words("s1"):
            if (word.text, word.start) in moved_words:
                assert word.speaker == "S2"
        assert stored(w) == out

    def test_moving_a_whole_turn_merges_its_neighbours(self) -> None:
        w = finished()
        before = stored(w)
        middle = [u for u in before.utterances if u.turn == 1]
        reassign = reassigner(w)
        for utterance in middle:
            current = stored(w)
            target = next(u for u in current.utterances
                          if u.text == utterance.text)
            reassign("s1", target.id, "S1")
        out = stored(w)
        assert turns(out) == [("S1", None)]
        assert speakers(out) == [("S1", None, False)]

    def test_to_an_enrolled_person_and_to_the_same_speaker(self) -> None:
        w = finished()
        w.voiceprints.enrol("eva", voice(5), name="Eva")
        reassign = reassigner(w)
        same = reassign("s1", "t0.s0", "S1")
        assert same == stored(w)
        out = reassign("s1", "t0.s0", "eva")
        assert out.utterances[0].speaker == "eva"
        assert out.utterances[1].previous_speaker == "eva"
        assert ("eva", "Eva", True) in speakers(out)

    def test_invalid_reassignments(self) -> None:
        w = finished()
        reassign = reassigner(w)
        with pytest.raises(NotFoundError, match="t9.s9"):
            reassign("s1", "t9.s9", "S1")
        with pytest.raises(NotFoundError, match="nadie"):
            reassign("s1", "t0.s0", "nadie")
        with pytest.raises(NotFoundError):
            reassign("nope", "t0.s0", "S1")


class TestPublishTranscript:
    def test_publishes_the_contract_of_a_done_session(self) -> None:
        w = finished(enrol_luis=True)
        publisher = RecordingPublisher()
        document = PublishTranscript(w.sessions, publisher)("s1")
        assert publisher.documents == [document]
        assert document == stored(w).to_contract()
        assert document["schema"] == "transcript.v1"
        assert "words" not in document["utterances"][0]

    @pytest.mark.parametrize("state", [
        SessionState.UPLOADED, SessionState.FAILED,
    ])
    def test_only_done_sessions(self, state: SessionState) -> None:
        w = world()
        w.upload()
        w.sessions.save(w.sessions.sessions["s1"].model_copy(
            update={"state": state}
        ))
        publisher = RecordingPublisher()
        with pytest.raises(StateError, match="done"):
            PublishTranscript(w.sessions, publisher)("s1")
        assert publisher.documents == []

    def test_a_done_session_without_transcript(self) -> None:
        w = finished()
        w.sessions.transcripts.clear()
        with pytest.raises(NotFoundError, match="no transcript"):
            PublishTranscript(w.sessions, RecordingPublisher())("s1")


def doubtful() -> World:
    """A finished world where every sentence is doubtful (S1/S2)."""
    w = world(parameters=TranscriptionParameters(doubt_margin=2.0))
    w.run()
    return w


class TestCertaintyUnderReview:
    def test_a_sentence_moved_by_hand_loses_its_evidence(self) -> None:
        w = doubtful()
        first = stored(w).utterances[0]
        assert first.doubtful is True
        out = reassigner(w)("s1", first.id, "S2")
        moved = out.utterances[0]
        assert moved.speaker == "S2"
        assert (moved.speaker_certainty, moved.doubtful, moved.overlap,
                moved.alternative_speaker) == (None, None, None, None)
        assert out.utterances[1].doubtful is True  # the others keep it

    def test_a_merge_points_alternatives_at_the_target(self) -> None:
        w = doubtful()
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        out = renamer(w)("s1", "S2", merge_into="luis")
        kept = [u for u in out.utterances if u.speaker == "S1"]
        assert {u.alternative_speaker for u in kept} == {"luis"}

    def test_merging_into_the_alternative_clears_it(self) -> None:
        w = doubtful()
        out = renamer(w)("s1", "S2", merge_into="S1")
        assert {u.alternative_speaker for u in out.utterances} == {None}

    def test_the_target_of_a_merge_loses_its_certainty(self) -> None:
        """S1's margins were taken against S2, now the same person."""
        w = doubtful()
        before = stored(w)
        assert all(u.speaker_certainty is not None
                   for u in before.utterances)
        out = renamer(w)("s1", "S2", merge_into="S1")
        assert {u.speaker for u in out.utterances} == {"S1"}
        for u in out.utterances:
            assert (u.speaker_certainty, u.doubtful, u.overlap,
                    u.alternative_speaker) == (None, None, None, None)

    def test_a_speaker_left_without_speech_is_no_alternative(self) -> None:
        w = doubtful()
        while any(u.speaker == "S2" for u in stored(w).utterances):
            target = next(u for u in stored(w).utterances
                          if u.speaker == "S2")
            reassigner(w)("s1", target.id, "S1")
        out = stored(w)
        assert [s.id for s in out.speakers] == ["S1"]
        assert {u.alternative_speaker for u in out.utterances} == {None}
