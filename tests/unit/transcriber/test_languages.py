"""A session's language: codes, ``auto`` and the language of a sentence."""

import pytest

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.languages import (
    base_code,
    check_language,
    recognition_language,
    tag_languages,
)
from transcriber.domain.models import Utterance


def sentence(start: float, end: float, n: int = 0) -> Utterance:
    """A one-sentence turn ``n``."""
    return Utterance(
        id=f"t{n}.s0", turn=n, index=0, speaker="ana", start=start,
        end=end, text="Algo.", previous_speaker=None,
        previous_utterance=None,
    )


def test_codes() -> None:
    assert base_code(" es-ES ") == "es"
    assert base_code("pt_BR") == "pt"
    assert recognition_language("auto") is None
    assert recognition_language("Auto") is None
    assert recognition_language("en-GB") == "en"


def test_check() -> None:
    known = frozenset({"es", "en"})
    assert check_language("es-ES", known) == "es-ES"
    assert check_language("auto", known) == "auto"
    assert check_language("xx", None) == "xx"
    with pytest.raises(InvalidInputError, match="'xx'"):
        check_language("xx", known)


def test_the_language_shared_most_wins() -> None:
    [one, two, three] = tag_languages(
        [sentence(0, 4), sentence(5, 9, 1), sentence(20, 21, 2)],
        [(0, 3, "es"), (3, 10, "en")], "es",
    )
    assert (one.language, one.language_differs) == ("es", False)
    assert (two.language, two.language_differs) == ("en", True)
    assert (three.language, three.language_differs) == (None, None)


def test_an_instant_takes_the_stretch_holding_it() -> None:
    [instant] = tag_languages([sentence(2, 2)], [(0, 3, "en")], "auto")
    assert (instant.language, instant.language_differs) == ("en", None)
