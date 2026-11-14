"""Transcriber speakers: words to speakers, smoothing, turns, windows.

Copied from ``tests/unit/test_skills_speech.py``; only imports and the
frozen models of the transcriber change the setup.
"""

from typing import List, Optional, Sequence

import pytest

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Turn, Word
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import (
    UNKNOWN_SPEAKER,
    SpeakerSpan,
    analysis_windows,
    assign_speakers,
    rename_speakers,
    smooth_speakers,
    spans_from_windows,
    speaker_label,
    speakers_in_order,
    speech_regions,
    speech_seconds,
    turns_from_words,
)


def words(*items: tuple[str, float, float]) -> List[Word]:
    """Timed words from ``(text, start, end)`` tuples."""
    return [Word(text=t, start=s, end=e) for t, s, e in items]


def spoken_by(
    items: Sequence[Word], speakers: Sequence[Optional[str]]
) -> List[Word]:
    """Return ``items`` with the given speakers (the models are frozen)."""
    return [w.model_copy(update={"speaker": s})
            for w, s in zip(items, speakers)]


class TestAssignSpeakers:
    def test_largest_overlap_wins(self) -> None:
        spans = [SpeakerSpan(speaker="A", start=0, end=1.0),
                 SpeakerSpan(speaker="B", start=1.0, end=3.0)]
        out = assign_speakers(
            words(("uno", 0.1, 0.5), ("dos", 0.8, 1.6), ("tres", 2.0, 2.4)),
            spans,
        )
        assert [w.speaker for w in out] == ["A", "B", "B"]

    def test_word_in_a_gap_takes_the_nearest_span(self) -> None:
        spans = [SpeakerSpan(speaker="A", start=0, end=1.0),
                 SpeakerSpan(speaker="B", start=5.0, end=6.0)]
        out = assign_speakers(
            words(("x", 1.2, 1.4), ("y", 4.5, 4.8)), spans
        )
        assert [w.speaker for w in out] == ["A", "B"]

    def test_without_spans_one_unknown_speaker(self) -> None:
        out = assign_speakers(words(("x", 0, 1)), [])
        assert out[0].speaker == UNKNOWN_SPEAKER

    def test_input_is_not_modified(self) -> None:
        source = words(("x", 0, 1))
        assign_speakers(source, [SpeakerSpan(speaker="A", start=0, end=1)])
        assert source[0].speaker is None


class TestSmoothSpeakers:
    def test_one_word_flip_takes_its_neighbours_speaker(self) -> None:
        w = spoken_by(
            words(("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("d", 3, 4)),
            ["A", "B", "A", "A"],
        )
        assert [x.speaker for x in smooth_speakers(w)] == ["A"] * 4

    def test_longer_runs_and_edges_are_kept(self) -> None:
        w = spoken_by(
            words(("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("d", 3, 4),
                  ("e", 4, 5)),
            ["B", "A", "B", "B", "A"],
        )
        # "A" at 1 sits between B and B: one word, smoothed; the edges
        # (first B, last A) stay.
        assert [x.speaker for x in smooth_speakers(w)] == [
            "B", "B", "B", "B", "A",
        ]
        two = spoken_by(
            words(("a", 0, 1), ("b", 1, 2), ("c", 2, 3), ("d", 3, 4)),
            ["A", "B", "B", "A"],
        )
        assert [x.speaker for x in smooth_speakers(two)] == [
            "A", "B", "B", "A",
        ]
        assert [x.speaker for x in smooth_speakers(two, max_run=2)] == [
            "A", "A", "A", "A",
        ]

    def test_empty(self) -> None:
        assert smooth_speakers([]) == []


class TestTurnsFromWords:
    def test_split_on_speaker_change_and_long_silence(self) -> None:
        w = spoken_by(
            words((" Hola", 0.0, 0.4), (" a", 0.5, 0.6), ("todos.", 0.6, 1.0),
                  ("Sí.", 1.2, 1.5), ("Vale.", 4.0, 4.5), ("", 4.6, 4.7)),
            ["A", "A", "A", "B", "B", "B"],
        )
        turns = turns_from_words(w, max_gap=1.0)
        assert [(t.speaker, t.text, t.start, t.end) for t in turns] == [
            ("A", "Hola a todos.", 0.0, 1.0),
            ("B", "Sí.", 1.2, 1.5),
            ("B", "Vale.", 4.0, 4.5),
        ]
        assert len(turns[0].words) == 3

    def test_words_without_speaker_are_unknown(self) -> None:
        turns = turns_from_words(words(("x", 0, 1)))
        assert turns[0].speaker == UNKNOWN_SPEAKER

    def test_turns_feed_the_utterances_with_words_and_confidence(
        self,
    ) -> None:
        w = [Word(text="Hola", start=0, end=0.5, speaker="A",
                  probability=0.9),
             Word(text="tú.", start=0.6, end=1.0, speaker="A",
                  probability=0.7)]
        (utterance,) = build_utterances(turns_from_words(w))
        assert utterance.text == "Hola tú."
        assert (utterance.start, utterance.end) == (0, 1.0)
        assert [x.text for x in utterance.words] == ["Hola", "tú."]
        assert utterance.confidence == pytest.approx(0.8)


class TestRenameSpeakers:
    def turns(self) -> List[Turn]:
        """Two speakers; the first turn has a word."""
        return [
            Turn(speaker="SPEAKER_01", text="Hola.", start=0, end=1,
                 words=[Word(text="Hola.", start=0, end=1,
                             speaker="SPEAKER_01")]),
            Turn(speaker="SPEAKER_02", text="Buenas.", start=1, end=2),
        ]

    def test_names_replace_labels_in_turns_and_words(self) -> None:
        source = self.turns()
        named = rename_speakers(source, {"SPEAKER_01": "Ana"})
        assert speakers_in_order(named) == ["Ana", "SPEAKER_02"]
        assert named[0].words[0].speaker == "Ana"
        assert speakers_in_order(source) == ["SPEAKER_01", "SPEAKER_02"]

    def test_two_people_cannot_share_a_name(self) -> None:
        with pytest.raises(InvalidInputError, match="named Ana"):
            rename_speakers(
                self.turns(), {"SPEAKER_01": "Ana", "SPEAKER_02": "Ana"}
            )


class TestSpansAndWindows:
    def test_speaker_label(self) -> None:
        assert speaker_label(0) == "SPEAKER_01"
        assert speaker_label(11) == "SPEAKER_12"

    def test_adjacent_windows_merge_and_overlaps_are_halved(self) -> None:
        spans = spans_from_windows(
            [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (5.0, 6.0)],
            [0, 0, 1, 1],
        )
        assert [(s.speaker, s.start, s.end) for s in spans] == [
            ("SPEAKER_01", 0.0, 1.875),
            ("SPEAKER_02", 1.875, 6.0),
        ]

    def test_label_count_must_match(self) -> None:
        with pytest.raises(ValueError):
            spans_from_windows([(0.0, 1.0)], [])

    def test_windows_cover_regions(self) -> None:
        assert analysis_windows([(0.0, 1.0)]) == [(0.0, 1.0)]
        assert analysis_windows([(0.0, 0.2)]) == []
        assert analysis_windows([(0.0, 3.0)]) == [
            (0.0, 1.5), (0.75, 2.25), (1.5, 3.0),
        ]
        # 3.5 s: the last full window ends at 3.0, 0.5 s left -> anchored
        assert analysis_windows([(0.0, 3.5)])[-1] == (2.0, 3.5)
        # 3.2 s: only 0.2 s left, below the minimum -> no extra window
        assert analysis_windows([(0.0, 3.2)])[-1] == (1.5, 3.0)

    def test_window_parameters_are_validated(self) -> None:
        with pytest.raises(InvalidInputError):
            analysis_windows([(0.0, 1.0)], length=0)

    def test_speech_regions_join_close_words(self) -> None:
        w = words(("c", 5.0, 5.5), ("a", 0.0, 0.4), ("b", 0.6, 1.0))
        assert speech_regions(w, max_gap=0.5) == [(0.0, 1.0), (5.0, 5.5)]
        assert speech_regions([]) == []

    def test_speech_seconds_counts_overlaps_once(self) -> None:
        assert speech_seconds([]) == 0.0
        assert speech_seconds([(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)]) == 4.0
        assert speech_seconds([(0.0, 4.0), (1.0, 2.0)]) == 4.0
