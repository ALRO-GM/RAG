"""Transcriber sentences: splitting, utterance ids, previous speaker."""

from typing import List, Optional

import pytest

from transcriber.domain.models import Turn, Word
from transcriber.domain.sentences import (
    build_utterances,
    merge_turns,
    split_sentences,
    utterance_id,
)


def words(*items: tuple[str, float, float]) -> List[Word]:
    """Timed words from ``(text, start, end)`` tuples."""
    return [Word(text=t, start=s, end=e) for t, s, e in items]


def turn(
    speaker: str, text: str, start: float = 0.0, end: float = 1.0,
    confidence: Optional[float] = None,
) -> Turn:
    """A turn without words."""
    return Turn(speaker=speaker, text=text, start=start, end=end,
                speaker_confidence=confidence)


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


class TestUtteranceTimes:
    def test_ids_and_times_from_words(self) -> None:
        w = [Word(text="Hola", start=0.0, end=0.4, probability=0.9),
             Word(text="Ana.", start=0.5, end=0.9, probability=0.7),
             Word(text="¿Empezamos?", start=2.0, end=2.8)]
        out = build_utterances([Turn(
            speaker="luis", text="Hola Ana. ¿Empezamos?", start=0.0,
            end=2.8, words=w, speaker_confidence=0.7,
        )])
        assert [(u.id, u.text, u.start, u.end) for u in out] == [
            ("t0.s0", "Hola Ana.", 0.0, 0.9),
            ("t0.s1", "¿Empezamos?", 2.0, 2.8),
        ]
        assert [len(u.words) for u in out] == [2, 1]
        assert out[0].confidence == pytest.approx(0.8)
        assert out[1].confidence is None
        assert all(u.speaker_confidence == 0.7 for u in out)

    def test_times_interpolated_without_words(self) -> None:
        first, second = build_utterances(
            [turn("a", "Uno dos. Tres.", 10.0, 24.0)]
        )
        assert first.start == 10.0
        boundary = pytest.approx(10 + 14 * 8 / 13, abs=1e-3)
        assert first.end == second.start == boundary
        assert second.end == pytest.approx(24.0)
        assert first.words == [] and first.confidence is None

    def test_mismatched_words_fall_back_to_interpolation(self) -> None:
        out = build_utterances([Turn(
            speaker="a", text="Uno dos. Tres.", start=0.0, end=4.0,
            words=words(("Uno", 0, 1)),
        )])
        assert out[0].start == 0.0
        assert out[1].end == pytest.approx(4.0)

    def test_utterance_id(self) -> None:
        assert utterance_id(12, 3) == "t12.s3"


class TestPreviousSpeaker:
    def test_first_turn_has_none(self) -> None:
        out = build_utterances([turn("marta", "Uno. Dos.")])
        assert all(
            u.previous_speaker is None and u.previous_utterance is None
            for u in out
        )

    def test_all_sentences_of_a_turn_share_it(self) -> None:
        out = build_utterances([
            turn("marta", "Uno. Dos.", 0, 2),
            turn("luis", "Tres. Cuatro. Cinco.", 2, 5),
            turn("S1", "Seis.", 5, 6),
        ])
        assert [(u.id, u.previous_speaker, u.previous_utterance)
                for u in out] == [
            ("t0.s0", None, None),
            ("t0.s1", None, None),
            ("t1.s0", "marta", "t0.s1"),
            ("t1.s1", "marta", "t0.s1"),
            ("t1.s2", "marta", "t0.s1"),
            ("t2.s0", "luis", "t1.s2"),
        ]

    def test_same_speaker_turns_are_merged_first(self) -> None:
        out = build_utterances([
            turn("marta", "Uno.", 0, 1),
            turn("luis", "Dos.", 1, 2),
            turn("luis", "Tres.", 2, 3),
            turn("marta", "Cuatro.", 3, 4),
        ])
        assert [(u.id, u.speaker, u.previous_speaker, u.previous_utterance)
                for u in out] == [
            ("t0.s0", "marta", None, None),
            ("t1.s0", "luis", "marta", "t0.s0"),
            ("t1.s1", "luis", "marta", "t0.s0"),
            ("t2.s0", "marta", "luis", "t1.s1"),
        ]

    def test_empty_turns_are_dropped_before_merging(self) -> None:
        out = build_utterances([
            turn("luis", "Uno."), turn("marta", "  "), turn("luis", "Dos."),
        ])
        assert [(u.id, u.previous_speaker) for u in out] == [
            ("t0.s0", None), ("t0.s1", None),
        ]

    def test_no_turns(self) -> None:
        assert build_utterances([]) == []


class TestMergeTurns:
    def test_merge_keeps_text_times_words_and_lower_confidence(self) -> None:
        a = Turn(speaker="a", text="Uno.", start=1.0, end=2.0,
                 words=words(("Uno.", 1.0, 2.0)), speaker_confidence=0.8)
        b = Turn(speaker="a", text=" Dos. ", start=2.5, end=3.0,
                 words=words(("Dos.", 2.5, 3.0)), speaker_confidence=0.6)
        (merged,) = merge_turns([a, b])
        assert merged.text == "Uno. Dos."
        assert (merged.start, merged.end) == (1.0, 3.0)
        assert [w.text for w in merged.words] == ["Uno.", "Dos."]
        assert merged.speaker_confidence == 0.6

    @pytest.mark.parametrize("first, second, expected", [
        (None, None, None), (0.5, None, 0.5), (None, 0.4, 0.4),
    ])
    def test_unknown_confidence(
        self, first: Optional[float], second: Optional[float],
        expected: Optional[float],
    ) -> None:
        (merged,) = merge_turns(
            [turn("a", "Uno.", confidence=first),
             turn("a", "Dos.", confidence=second)]
        )
        assert merged.speaker_confidence == expected
