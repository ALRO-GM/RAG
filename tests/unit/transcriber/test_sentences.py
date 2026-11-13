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

    def test_no_before_a_number_is_an_abbreviation(self) -> None:
        assert split_sentences("Mira el No. 5 de la lista.") == [
            "Mira el No. 5 de la lista."
        ]

    @pytest.mark.parametrize("text, expected", [
        ("Creo que no. Pero sigo.", ["Creo que no.", "Pero sigo."]),
        ("No. Eso no.", ["No.", "Eso no."]),
        ("I said no. Then I left.", ["I said no.", "Then I left."]),
    ])
    def test_no_as_a_word_ends_a_sentence(
        self, text: str, expected: List[str]
    ) -> None:
        assert split_sentences(text) == expected


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

    def test_empty_words_are_ignored_when_aligning(self) -> None:
        """Recognisers emit words with no text; they hold no token."""
        out = build_utterances([Turn(
            speaker="a", text="Hola. Vale.", start=0.0, end=3.7,
            words=words((" Hola.", 0.0, 0.4), (" Vale.", 3.0, 3.5),
                        ("", 3.6, 3.7)),
        )])
        assert [(u.start, u.end) for u in out] == [(0.0, 0.4), (3.0, 3.5)]
        assert [[w.text for w in u.words] for u in out] == [
            [" Hola."], [" Vale."],
        ]

    def test_a_word_holding_several_tokens(self) -> None:
        out = build_utterances([Turn(
            speaker="a", text="Otra vez de nuevo. Bien.", start=0.0,
            end=2.0, words=words(("Otra", 0.0, 0.2), ("vez", 0.3, 0.5),
                                 ("de nuevo.", 0.6, 1.0),
                                 ("Bien.", 1.5, 2.0)),
        )])
        assert [(u.start, u.end, len(u.words)) for u in out] == [
            (0.0, 1.0, 3), (1.5, 2.0, 1),
        ]

    def test_merged_piece_with_words_keeps_its_times(self) -> None:
        """Only the piece without words is interpolated."""
        out = build_utterances([
            Turn(speaker="a", text="Hola Ana.", start=0.0, end=1.0,
                 words=words(("Hola", 0.0, 0.4), ("Ana.", 0.5, 0.9))),
            Turn(speaker="a", text="Vale. Sigo.", start=5.0, end=7.0),
        ])
        assert [(u.text, u.start, u.end, len(u.words)) for u in out] == [
            ("Hola Ana.", 0.0, 0.9, 2),
            ("Vale.", 5.0, 6.0, 0),
            ("Sigo.", 6.0, 7.0, 0),
        ]

    def test_merged_piece_without_words_first(self) -> None:
        out = build_utterances([
            Turn(speaker="a", text="Vale.", start=0.0, end=1.0),
            Turn(speaker="a", text="Hola Ana.", start=2.0, end=3.0,
                 words=words(("Hola", 2.1, 2.4), ("Ana.", 2.5, 2.8))),
        ])
        assert [(u.start, u.end, len(u.words)) for u in out] == [
            (0.0, 1.0, 0), (2.1, 2.8, 2),
        ]

    def test_sentence_across_pieces_spans_both(self) -> None:
        out = build_utterances([
            Turn(speaker="a", text="Pero no", start=0.0, end=0.6,
                 words=words(("Pero", 0.0, 0.3), ("no", 0.4, 0.6))),
            Turn(speaker="a", text="estoy segura.", start=2.0, end=3.0),
        ])
        assert [(u.text, u.start, u.end, len(u.words)) for u in out] == [
            ("Pero no estoy segura.", 0.0, 3.0, 2),
        ]

    def test_turns_given_out_of_order_run_forward(self) -> None:
        out = build_utterances([
            turn("luis", "Tres.", 4.0, 5.0),
            turn("marta", "Uno.", 0.0, 1.0),
            turn("marta", "Dos.", 2.0, 3.0),
        ])
        assert [(u.id, u.text, u.previous_speaker) for u in out] == [
            ("t0.s0", "Uno.", None), ("t0.s1", "Dos.", None),
            ("t1.s0", "Tres.", "marta"),
        ]
        starts = [u.start for u in out]
        assert starts == sorted(starts)

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

    def test_different_speakers_are_not_merged(self) -> None:
        out = merge_turns([turn("b", " Dos. ", 1, 2), turn("a", "Uno.", 0, 1)])
        assert [(t.speaker, t.text) for t in out] == [
            ("a", "Uno."), ("b", "Dos."),
        ]

    def test_merge_puts_pieces_in_time_order(self) -> None:
        late = Turn(speaker="a", text="Dos.", start=2.5, end=3.0,
                    words=words(("Dos.", 2.5, 3.0)))
        early = Turn(speaker="a", text="Uno.", start=1.0, end=2.0,
                     words=words(("Uno.", 1.0, 2.0)))
        (merged,) = merge_turns([late, early])
        assert merged.text == "Uno. Dos."
        assert [w.text for w in merged.words] == ["Uno.", "Dos."]
        (first, second) = build_utterances([late, early])
        assert first.end <= second.start

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
