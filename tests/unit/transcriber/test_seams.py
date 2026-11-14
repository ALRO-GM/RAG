"""Passages a recogniser returned twice at its window seams."""

from typing import List, Tuple

from transcriber.domain.seams import resolve_seams
from transcriber.domain.models import Word


def words(*items: Tuple[str, float, float]) -> List[Word]:
    """Words from ``(text, start, end)``."""
    return [Word(text=t, start=s, end=e) for t, s, e in items]


def texts(found: List[Word]) -> List[str]:
    """The texts of ``found``."""
    return [w.text for w in found]


def test_words_in_order_come_back_unchanged() -> None:
    given = words(("uno", 0.0, 0.5), ("dos", 0.5, 1.0), ("tres", 1.2, 1.5))
    assert resolve_seams(given) == given


def test_nothing_gives_nothing() -> None:
    assert resolve_seams([]) == []


def test_a_passage_heard_twice_is_kept_once() -> None:
    """The defect seen on a real recording, with made-up words.

    The first window runs to 29.3 s; the next one starts again at its
    own start (20 s, the first word glued to it) and repeats the
    overlap with slightly different times.
    """
    first = words(
        ("antes", 17.0, 18.0), ("de", 18.0, 19.0), ("hacer", 20.5, 21.3),
        ("una", 21.3, 22.0), ("prueba", 22.0, 23.9), ("larga", 24.0, 25.3),
        ("con", 25.6, 26.0), ("calma", 26.0, 26.6), ("y", 27.5, 27.6),
        ("luego", 27.6, 28.0), ("otra", 28.5, 29.3),
    )
    second = words(
        ("hacer", 20.0, 21.3), ("una", 21.3, 22.0), ("prueba", 22.0, 23.9),
        ("larga", 23.9, 25.3), ("con", 25.6, 26.0), ("calma", 26.0, 26.6),
        ("y", 27.5, 27.6), ("luego", 27.6, 28.1), ("otra", 28.5, 29.3),
        ("más", 30.0, 30.4),
    )
    found = resolve_seams(first + second)
    assert texts(found) == [
        "antes", "de", "hacer", "una", "prueba", "larga", "con", "calma",
        "y", "luego", "otra", "más",
    ]
    starts = [w.start for w in found]
    assert starts == sorted(starts)


def test_the_overlap_is_cut_at_its_middle() -> None:
    """Different words on the two sides: each window keeps its half."""
    first = words(("a", 10.0, 11.0), ("b", 12.0, 13.0), ("c", 18.0, 19.0))
    # Back to 11: the overlap runs from 11 to 19, cut at 15.
    second = words(("x", 11.0, 12.0), ("y", 16.0, 17.0), ("z", 20.0, 21.0))
    assert texts(resolve_seams(first + second)) == ["a", "b", "y", "z"]


def test_a_word_past_the_cut_goes_to_the_later_window() -> None:
    first = words(("a", 10.0, 11.0), ("hola,", 14.0, 15.2))
    second = words(("x", 13.0, 13.2), ("Hola", 14.9, 16.0), ("b", 17, 18))
    # The cut is at (13.0 + 15.2) / 2 = 14.1: "hola," (middle 14.6) is
    # past it and goes, "x" (middle 13.1) is before it and goes.
    assert texts(resolve_seams(first + second)) == ["a", "Hola", "b"]


def test_the_same_word_across_the_cut_is_not_repeated() -> None:
    first = words(("a", 10.0, 11.0), ("dijo", 12.0, 12.6), ("e", 13.5, 14))
    second = words(("x", 11.0, 11.5), ("dijo", 12.5, 13.0), ("b", 15, 16))
    # Cut at (11.0 + 14.0) / 2 = 12.5: the first "dijo" (middle 12.3)
    # stays; the second (middle 12.75) is the same word heard twice.
    assert texts(resolve_seams(first + second)) == ["a", "dijo", "b"]


def test_several_seams_in_a_row() -> None:
    """Back twice, as on one real recording (made-up words)."""
    found = resolve_seams(words(
        ("a", 78.0, 79.0), ("b", 84.0, 85.0), ("c", 87.9, 88.6),
        ("b2", 82.0, 82.5), ("c2", 86.9, 88.1),
        ("a3", 80.0, 81.4), ("b3", 84.1, 85.0), ("c3", 87.9, 88.5),
        ("d", 90.0, 91.0),
    ))
    assert texts(found) == ["a", "b3", "c3", "d"]
