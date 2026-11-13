"""Transcription metrics: word error rate and speaker attribution."""

from typing import List

import pytest

from src.domain.skills.speech import SpeakerSpan
from src.domain.skills.speech_metrics import (
    normalise_words,
    reference_speaker,
    speaker_accuracy,
    speaker_mapping,
    word_error_rate,
)
from src.domain.skills.transcript import Word


def test_normalise_words_folds_case_accents_and_punctuation() -> None:
    assert normalise_words("¡Jehová, á los HIJOS!") == [
        "jehova", "a", "los", "hijos",
    ]
    assert normalise_words("") == []


@pytest.mark.parametrize("reference, hypothesis, expected", [
    ("uno dos tres", "uno dos tres", 0.0),
    ("uno dos tres", "Uno, dos; ¡tres!", 0.0),
    ("uno dos tres cuatro", "uno tres cuatro", 0.25),          # deletion
    ("uno dos tres cuatro", "uno dos y tres cuatro", 0.25),    # insertion
    ("uno dos tres cuatro", "uno dos seis cuatro", 0.25),      # substitution
    ("uno dos", "", 1.0),
    ("", "", 0.0),
    ("", "algo", 1.0),
    ("a b", "c d e f", 2.0),
])
def test_word_error_rate(
    reference: str, hypothesis: str, expected: float
) -> None:
    assert word_error_rate(reference, hypothesis) == pytest.approx(expected)


SPANS = [
    SpeakerSpan(speaker="Ana", start=0.0, end=4.0),
    SpeakerSpan(speaker="Luis", start=5.0, end=9.0),
]


def said(*items: tuple[str, float]) -> List[Word]:
    """Words labelled by the diarizer, one per ``(label, middle)``."""
    return [Word(text="x", start=t - 0.1, end=t + 0.1, speaker=label)
            for label, t in items]


def test_reference_speaker_inside_and_in_a_gap() -> None:
    assert reference_speaker(1.0, SPANS) == "Ana"
    assert reference_speaker(4.2, SPANS) == "Ana"
    assert reference_speaker(4.9, SPANS) == "Luis"


def test_labels_are_mapped_to_people_by_agreement() -> None:
    words = said(("S2", 1), ("S2", 2), ("S2", 3), ("S1", 6), ("S1", 7))
    assert speaker_mapping(words, SPANS) == {"S1": "Luis", "S2": "Ana"}
    accuracy, mapping = speaker_accuracy(words, SPANS)
    assert accuracy == 1.0
    assert mapping["S2"] == "Ana"


def test_wrong_words_lower_the_accuracy() -> None:
    words = said(("S1", 1), ("S1", 2), ("S1", 3), ("S1", 6), ("S2", 7))
    accuracy, _ = speaker_accuracy(words, SPANS)
    assert accuracy == pytest.approx(4 / 5)


def test_extra_voices_count_as_errors() -> None:
    """Three labels for two people: the unmapped label is all wrong."""
    words = said(("S1", 1), ("S1", 2), ("S3", 3), ("S2", 6), ("S2", 7))
    accuracy, mapping = speaker_accuracy(words, SPANS)
    assert accuracy == pytest.approx(4 / 5)
    assert "S3" not in mapping


def test_one_label_for_two_people() -> None:
    words = said(("S1", 1), ("S1", 2), ("S1", 6))
    accuracy, mapping = speaker_accuracy(words, SPANS)
    assert accuracy == pytest.approx(2 / 3)
    assert mapping == {"S1": "Ana"}


def test_nothing_to_score() -> None:
    assert speaker_accuracy([], SPANS) == (0.0, {})
    assert speaker_accuracy(said(("S1", 1)), []) == (0.0, {})
    assert speaker_mapping([], SPANS) == {}
