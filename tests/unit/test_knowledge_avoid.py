"""LV-17 (#232): the check of interview questions better avoided."""

from typing import List

import pytest

from src.application.knowledge.avoid import AvoidQuestions
from src.domain.errors import InvalidInputError
from src.domain.knowledge.avoid import (
    MAX_QUESTION,
    AvoidHint,
    in_language,
    language_of,
)

AGE = AvoidHint(category="age", title="Edad", why="No influye.",
                alternative="Pregunta por la experiencia.")


class FakeRules:
    """Flags any sentence that says "edad"; keeps what it was asked."""

    def __init__(self) -> None:
        self.asked: List[tuple[str, str]] = []

    def hints(self, question: str, language: str) -> List[AvoidHint]:
        self.asked.append((question, language))
        return [AGE] if "edad" in question else []


def test_a_sentence_gets_its_hints_in_the_language_asked() -> None:
    rules = FakeRules()
    check = AvoidQuestions(rules)
    found = check.check("  ¿Qué edad tienes?  ", "en")
    assert found == {"language": "en", "hints": [AGE.model_dump()]}
    assert rules.asked == [("¿Qué edad tienes?", "en")]
    assert check.check("¿Cuántos años llevas con Python?", "fr") == {
        "language": "es", "hints": []}


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_QUESTION + 1)])
def test_empty_or_long_sentences_are_refused(text: str) -> None:
    with pytest.raises(InvalidInputError):
        AvoidQuestions(FakeRules()).check(text)


def test_texts_fall_back_to_spanish() -> None:
    assert language_of("en") == "en" and language_of("de") == "es"
    assert in_language({"es": "Edad", "en": "Age"}, "en") == "Age"
    assert in_language({"es": "Edad"}, "en") == "Edad"
    assert in_language({}, "en") == ""
