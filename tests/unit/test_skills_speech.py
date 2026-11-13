"""Speech domain: speakers for words, turns, sentences, voice clusters."""

from typing import List

import numpy as np
import pytest

from src.domain.errors import InvalidInputError
from src.domain.skills.speech import (
    UNKNOWN_SPEAKER,
    SpeakerSpan,
    analysis_windows,
    assign_speakers,
    cluster_speakers,
    rename_speakers,
    segments_from_words,
    spans_from_windows,
    speaker_label,
    speech_regions,
    split_sentences,
    transcript_sentences,
    turn_sentences,
)
from src.domain.skills.transcript import Turn, Word, build_transcript


def words(*items: tuple[str, float, float]) -> List[Word]:
    """Timed words from ``(text, start, end)`` tuples."""
    return [Word(text=t, start=s, end=e) for t, s, e in items]


class TestSplitSentences:
    @pytest.mark.parametrize("text, expected", [
        ("Hola. ¿Qué tal? ¡Bien!", ["Hola.", "¿Qué tal?", "¡Bien!"]),
        ("One. Two? Three!", ["One.", "Two?", "Three!"]),
        ("Sin puntuación final", ["Sin puntuación final"]),
        ("", []),
        ("   ", []),
        ("Espera... Vale, sigo.", ["Espera...", "Vale, sigo."]),
        ("Dijo \"vale.\" Luego se fue.", ["Dijo \"vale.\"", "Luego se fue."]),
        ("Son 3.5 puntos. Bien.", ["Son 3.5 puntos.", "Bien."]),
        ("Uno.  Dos.\nTres.", ["Uno.", "Dos.", "Tres."]),
    ])
    def test_cases(self, text: str, expected: List[str]) -> None:
        assert split_sentences(text) == expected

    @pytest.mark.parametrize("text", [
        "El Sr. García llega tarde.",
        "Lo dijo la Dra. Ruiz ayer.",
        "See e.g. Table 2 for details.",
        "Lo firma J. Pérez hoy.",
        "Mira la pág. 4 del enunciado.",
    ])
    def test_abbreviations_do_not_end_a_sentence(self, text: str) -> None:
        assert split_sentences(text) == [text]

    def test_lowercase_after_period_is_not_a_boundary(self) -> None:
        """Recognisers put stray periods; a lower-case follow-up joins."""
        assert split_sentences("vale. pero no.") == ["vale. pero no."]


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


class TestSegmentsFromWords:
    def test_split_on_speaker_change_and_long_silence(self) -> None:
        w = words((" Hola", 0.0, 0.4), (" a", 0.5, 0.6), ("todos.", 0.6, 1.0),
                  ("Sí.", 1.2, 1.5), ("Vale.", 4.0, 4.5), ("", 4.6, 4.7))
        for word, speaker in zip(w, ["A", "A", "A", "B", "B", "B"]):
            word.speaker = speaker
        segments = segments_from_words(w, max_gap=1.0)
        assert [(s["speaker"], s["text"], s["start"], s["end"])
                for s in segments] == [
            ("A", "Hola a todos.", 0.0, 1.0),
            ("B", "Sí.", 1.2, 1.5),
            ("B", "Vale.", 4.0, 4.5),
        ]
        assert len(segments[0]["words"]) == 3

    def test_words_without_speaker_are_unknown(self) -> None:
        segments = segments_from_words(words(("x", 0, 1)))
        assert segments[0]["speaker"] == UNKNOWN_SPEAKER

    def test_segments_feed_the_transcript_with_words_and_confidence(
        self,
    ) -> None:
        w = [Word(text="Hola", start=0, end=0.5, speaker="A",
                  probability=0.9),
             Word(text="tú.", start=0.6, end=1.0, speaker="A",
                  probability=0.7)]
        transcript = build_transcript("s", segments_from_words(w))
        turn = transcript.turns[0]
        assert turn.text == "Hola tú."
        assert (turn.start, turn.end) == (0, 1.0)
        assert [x.text for x in turn.words] == ["Hola", "tú."]
        assert turn.confidence == pytest.approx(0.8)


class TestSentences:
    def test_times_from_words(self) -> None:
        w = words(("Hola", 0.0, 0.4), ("Ana.", 0.5, 0.9),
                  ("¿Empezamos?", 2.0, 2.8))
        turn = Turn(index=3, speaker="Luis", text="Hola Ana. ¿Empezamos?",
                    start=0.0, end=2.8, words=w)
        sentences = turn_sentences(turn)
        assert [(s.id, s.text, s.start, s.end) for s in sentences] == [
            ("t3.s0", "Hola Ana.", 0.0, 0.9),
            ("t3.s1", "¿Empezamos?", 2.0, 2.8),
        ]
        assert all(s.speaker == "Luis" and s.turn == 3 for s in sentences)
        assert sentences[0].n_words() == 2

    def test_times_interpolated_without_words(self) -> None:
        turn = Turn(index=0, speaker="A", text="Uno dos. Tres.",
                    start=10.0, end=24.0)
        first, second = turn_sentences(turn)
        assert first.start == 10.0
        boundary = pytest.approx(10 + 14 * 8 / 13, abs=1e-3)
        assert first.end == second.start == boundary
        assert second.end == pytest.approx(24.0)

    def test_mismatched_words_fall_back_to_interpolation(self) -> None:
        turn = Turn(index=0, speaker="A", text="Uno dos. Tres.",
                    start=0.0, end=4.0, words=words(("Uno", 0, 1)))
        assert turn_sentences(turn)[0].start == 0.0
        assert turn_sentences(turn)[1].end == pytest.approx(4.0)

    def test_no_times_without_timestamps(self) -> None:
        turn = Turn(index=0, speaker="A", text="Uno. Dos.")
        assert [(s.start, s.end) for s in turn_sentences(turn)] == [
            (None, None), (None, None),
        ]

    def test_whole_transcript(self) -> None:
        transcript = build_transcript("s", [
            {"speaker": "A", "text": "Uno. Dos.", "start": 0, "end": 2},
            {"speaker": "B", "text": "Tres."},
        ])
        ids = [s.id for s in transcript_sentences(transcript)]
        assert ids == ["t0.s0", "t0.s1", "t1.s0"]


class TestRenameSpeakers:
    def test_names_replace_labels_in_turns_and_words(self) -> None:
        transcript = build_transcript("s", [
            {"speaker": "SPEAKER_01", "text": "Hola.", "start": 0, "end": 1,
             "words": [{"text": "Hola.", "start": 0, "end": 1,
                        "speaker": "SPEAKER_01"}]},
            {"speaker": "SPEAKER_02", "text": "Buenas."},
        ])
        named = rename_speakers(transcript, {"SPEAKER_01": "Ana"})
        assert named.speakers() == ["Ana", "SPEAKER_02"]
        assert named.turns[0].words[0].speaker == "Ana"
        assert transcript.speakers() == ["SPEAKER_01", "SPEAKER_02"]

    def test_two_people_cannot_share_a_name(self) -> None:
        transcript = build_transcript("s", [
            {"speaker": "SPEAKER_01", "text": "Hola."},
            {"speaker": "SPEAKER_02", "text": "Buenas."},
        ])
        with pytest.raises(InvalidInputError, match="named Ana"):
            rename_speakers(
                transcript, {"SPEAKER_01": "Ana", "SPEAKER_02": "Ana"}
            )


def voices() -> np.ndarray:
    """Six embeddings: three voices, two windows each, first-seen order."""
    a, b, c = np.eye(3, dtype=np.float32)
    return np.stack([a, b, a * 0.9 + b * 0.1, c, b * 0.95 + c * 0.05,
                     c]).astype(np.float32)


class TestClusterSpeakers:
    def test_known_number_of_speakers(self) -> None:
        assert cluster_speakers(voices(), num_speakers=3) == [0, 1, 0, 2, 1, 2]

    def test_threshold_finds_the_number(self) -> None:
        assert cluster_speakers(voices(), threshold=0.5) == [0, 1, 0, 2, 1, 2]

    def test_one_speaker(self) -> None:
        assert cluster_speakers(voices(), num_speakers=1) == [0] * 6

    def test_low_threshold_merges_everything(self) -> None:
        assert set(cluster_speakers(voices(), threshold=-1.0)) == {0}

    def test_max_speakers_caps_an_unknown_count(self) -> None:
        distinct = np.eye(5, dtype=np.float32)
        labels = cluster_speakers(distinct, threshold=0.5, max_speakers=3)
        assert len(set(labels)) == 3

    def test_empty_and_invalid(self) -> None:
        assert cluster_speakers(np.zeros((0, 4), dtype=np.float32)) == []
        with pytest.raises(InvalidInputError):
            cluster_speakers(voices(), num_speakers=0)

    def test_zero_vectors_do_not_crash(self) -> None:
        vectors = np.zeros((2, 3), dtype=np.float32)
        assert cluster_speakers(vectors, num_speakers=2) == [0, 1]


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
