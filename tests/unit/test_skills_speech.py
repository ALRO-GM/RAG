"""Sentences: turns split into sentences that keep speaker and time."""

from typing import List

import pytest

from src.domain.skills.speech import (
    split_sentences,
    transcript_sentences,
    turn_sentences,
)
from src.domain.skills.transcript import (
    Turn,
    TurnSentence,
    Word,
    build_transcript,
)


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


class TestGivenSentences:
    def test_a_turn_with_sentences_is_not_split_again(self) -> None:
        turn = Turn(
            index=4, speaker="Luis", text="Sr. Pérez dijo. Que no.",
            start=1.0, end=3.0, previous_speaker="Marta",
            sentences=[
                TurnSentence(id="t4.s0", text="Sr. Pérez dijo. Que no.",
                             start=1.0, end=3.0),
            ],
        )
        [sentence] = turn_sentences(turn)
        assert (sentence.id, sentence.turn, sentence.index) == (
            "t4.s0", 4, 0,
        )
        assert sentence.text == "Sr. Pérez dijo. Que no."
        assert (sentence.start, sentence.end) == (1.0, 3.0)
        assert sentence.previous_speaker == "Marta"

    def test_split_turns_have_no_previous_speaker(self) -> None:
        turn = Turn(index=0, speaker="A", text="Uno. Dos.")
        assert {s.previous_speaker for s in turn_sentences(turn)} == {None}
