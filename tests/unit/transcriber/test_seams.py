"""Passages a recogniser returned twice at its window seams."""

import random
from typing import List, Tuple

import pytest

from transcriber.domain.seams import (
    MAX_WORD_SECONDS,
    longest_plausible,
    resolve_seams,
)
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


def test_the_overlap_keeps_words_the_other_copy_does_not_contradict(
) -> None:
    """Different words at different moments: nothing was said twice."""
    first = words(("a", 10.0, 11.0), ("b", 12.0, 13.0), ("c", 18.0, 19.0))
    second = words(("x", 11.0, 12.0), ("y", 16.0, 17.0), ("z", 20.0, 21.0))
    assert texts(resolve_seams(first + second)) == [
        "a", "x", "b", "y", "c", "z",
    ]


def test_different_words_at_one_moment_keep_the_better_window() -> None:
    """The overlap runs 13-15.2 s, middle 14.1 s.

    At 13 s the earlier window (far from its end) wins over the later
    one (at its very start); at 15 s the later one wins.
    """
    first = words(("a", 10.0, 11.0), ("uno", 13.0, 13.4),
                  ("hola,", 14.8, 15.2))
    second = words(("one", 13.0, 13.3), ("ola", 14.7, 15.1),
                   ("b", 17, 18))
    assert texts(resolve_seams(first + second)) == [
        "a", "uno", "ola", "b",
    ]


def test_a_word_past_the_middle_comes_from_the_later_window() -> None:
    first = words(("a", 10.0, 11.0), ("hola,", 14.0, 15.2))
    second = words(("x", 12.8, 13.0), ("Hola", 14.9, 16.0), ("b", 17, 18))
    # "x" has no counterpart and nothing else is said at 13 s: kept.
    assert texts(resolve_seams(first + second)) == ["a", "x", "Hola", "b"]


def test_the_same_word_across_the_cut_is_not_repeated() -> None:
    first = words(("a", 10.0, 11.0), ("dijo", 12.0, 12.6), ("e", 13.5, 14))
    second = words(("x", 11.0, 11.5), ("dijo", 12.5, 13.0), ("b", 15, 16))
    assert texts(resolve_seams(first + second)) == [
        "a", "x", "dijo", "e", "b",
    ]


def test_several_seams_in_a_row() -> None:
    """Back twice, as on one real recording (made-up words)."""
    found = resolve_seams(words(
        ("uno", 78.0, 79.0), ("dos", 84.0, 85.0), ("tres", 87.9, 88.6),
        ("dos", 84.2, 85.0), ("tres", 87.8, 88.6),
        ("uno", 78.1, 79.0), ("dos", 84.1, 85.0), ("tres", 87.9, 88.5),
        ("cuatro", 90.0, 91.0),
    ))
    assert texts(found) == ["uno", "dos", "tres", "cuatro"]


class TestReviewOf24:
    """The inputs the review of #24 found losing or doubling words."""

    def test_jitter_is_not_a_seam(self) -> None:
        found = resolve_seams(words(
            ("a", 0, 1), ("b", 1.0, 1.5), ("c", 0.99, 2.0), ("d", 2.1, 2.5),
        ))
        assert texts(found) == ["a", "b", "c", "d"]
        assert [w.start for w in found] == [0, 1.0, 1.0, 2.1]

    def test_a_word_starting_inside_the_previous_one(self) -> None:
        found = resolve_seams(words(
            ("w1", 100, 100.4), ("w2", 100.4, 100.9), ("w3", 100.9, 101.3),
            ("w4", 100.85, 101.6), ("w5", 102, 103),
        ))
        assert texts(found) == ["w1", "w2", "w3", "w4", "w5"]

    def test_an_instant_word(self) -> None:
        found = resolve_seams(words(
            ("a", 9, 9.5), ("b", 10, 10), ("c", 9.98, 10.5), ("d", 11, 12),
        ))
        assert texts(found) == ["a", "b", "c", "d"]

    def test_a_jump_back_with_nothing_said_twice(self) -> None:
        found = resolve_seams(words(
            ("a", 10, 11), ("b", 12, 13), ("x", 5, 6), ("y", 14, 15),
        ))
        assert texts(found) == ["x", "a", "b", "y"]

    def test_a_short_second_copy_does_not_erase_the_first(self) -> None:
        first = words(("a", 20.0, 21.8), ("b", 22.0, 23.8),
                      ("c", 24.0, 25.8), ("d", 26.0, 27.8),
                      ("e", 28.0, 29.0))
        second = words(("x", 20.0, 21.0), ("f", 30.0, 31.0))
        assert texts(resolve_seams(first + second)) == [
            "a", "b", "c", "d", "e", "f",
        ]

    def test_a_repeated_word_is_not_doubled(self) -> None:
        first = words(("x", 10, 11), ("sí", 19.5, 19.8), ("sí", 20.0, 20.3),
                      ("z", 25, 26))
        second = words(("sí", 19.7, 20.1), ("sí", 20.0, 20.3), ("z", 25, 26),
                       ("q", 31, 32))
        assert texts(resolve_seams(first + second)) == [
            "x", "sí", "sí", "z", "q",
        ]

    def test_the_step_that_makes_a_seam_is_a_parameter(self) -> None:
        """Another word at 10.2 s: a seam at 1 s, jitter at 5 s."""
        given = words(("a", 10, 11), ("b", 12, 13), ("x", 10.2, 11),
                      ("c", 14, 15))
        assert texts(resolve_seams(given)) == ["a", "b", "c"]
        assert texts(resolve_seams(given, min_step=5.0)) == [
            "a", "b", "x", "c",
        ]

    def test_the_same_words_again_are_a_seam_at_any_step(self) -> None:
        given = words(("a", 10, 11), ("b", 12, 13), ("a", 10.2, 11),
                      ("b", 12, 13), ("c", 14, 15))
        assert texts(resolve_seams(given)) == ["a", "b", "c"]
        assert texts(resolve_seams(given, min_step=5.0)) == ["a", "b", "c"]


def random_words(rng: random.Random, count: int) -> List[Word]:
    """``count`` words of distinct text, forward with jitter and seams."""
    out: List[Word] = []
    t = 0.0
    for n in range(count):
        roll = rng.random()
        if roll < 0.08 and out:
            t = max(0.0, t - rng.uniform(1.5, 10.0))  # a seam
        elif roll < 0.2 and out:
            t = max(0.0, t - rng.uniform(0.0, 0.3))  # jitter
        length = rng.choice([0.0, 0.1, 0.3, 0.6, 1.2])
        out.append(Word(text=f"w{n}", start=round(t, 2),
                        end=round(t + length, 2)))
        t += length + rng.uniform(0.0, 0.5)
    return out


class TestProperties:
    """Checked on many random streams (seeded, so reproducible)."""

    def test_the_result_is_sorted_and_idempotent(self) -> None:
        rng = random.Random(24)
        for _ in range(300):
            found = resolve_seams(random_words(rng, rng.randint(0, 60)))
            starts = [w.start for w in found]
            assert starts == sorted(starts)
            assert resolve_seams(found) == found

    def test_without_seams_no_word_is_lost_or_reordered(self) -> None:
        rng = random.Random(25)
        for _ in range(300):
            given = [w for w in random_words(rng, 40)]
            given = resolve_seams(given)  # no seam left in it
            jittered = [
                w.model_copy(update={
                    "start": max(0.0, w.start - rng.uniform(0, 0.5)),
                }) if rng.random() < 0.3 and w.end > 0 else w
                for w in given
            ]
            jittered = [
                w.model_copy(update={"end": max(w.end, w.start)})
                for w in jittered
            ]
            assert texts(resolve_seams(jittered)) == texts(given)

    def test_a_jump_back_over_silence_loses_nothing(self) -> None:
        """Seams whose second copy says nothing over the first's words."""
        rng = random.Random(26)
        for _ in range(300):
            first = [Word(text=f"a{n}", start=10.0 + 2 * n,
                          end=10.8 + 2 * n) for n in range(6)]
            second = [Word(text=f"b{n}", start=11.0 + 2 * n,
                           end=11.0 + 2 * n + rng.uniform(0, 0.9))
                      for n in range(rng.randint(1, 8))]
            found = resolve_seams(first + second)
            assert sorted(texts(found)) == sorted(texts(first + second))


class TestSecondReview:
    """The inputs the second review found erasing or doubling words."""

    @staticmethod
    def clean(count: int = 100) -> List[Word]:
        """``w0``, ``w1``... at ``i`` to ``i + 0.9`` s."""
        return [Word(text=f"w{i}", start=float(i), end=i + 0.9)
                for i in range(count)]

    def test_one_long_word_stepping_back_erases_nothing_else(self) -> None:
        given = self.clean() + words(("um", 85.0, 100.9),
                                     ("next", 101.0, 101.5))
        found = texts(resolve_seams(given))
        assert [t for t in found if t != "um"] == [
            f"w{i}" for i in range(100)] + ["next"]

    def test_a_word_over_the_whole_recording_erases_nothing_else(
        self,
    ) -> None:
        given = self.clean() + words(("bad", 0.0, 100.95),
                                     ("next", 101.0, 101.5))
        found = texts(resolve_seams(given))
        assert [t for t in found if t != "bad"] == [
            f"w{i}" for i in range(100)] + ["next"]

    def test_an_implausible_span_is_shortened(self) -> None:
        """A word that jumped back keeps its end, any other its start."""
        assert longest_plausible(Word(text="um", start=0, end=1)) == \
            MAX_WORD_SECONDS
        assert longest_plausible(
            Word(text="responsabilidades", start=0, end=1)
        ) == pytest.approx(6.8)
        found = resolve_seams(words(("a", 10.0, 10.5), ("um", 10.5, 40.0),
                                    ("b", 41.0, 41.5)))
        assert [(w.text, w.start, w.end) for w in found] == [
            ("a", 10.0, 10.5), ("um", 10.5, 13.5), ("b", 41.0, 41.5),
        ]
        jumped = resolve_seams(words(("a", 50.0, 50.5), ("b", 52.0, 52.5),
                                     ("eh", 0.0, 60.0), ("c", 61.0, 61.5)))
        assert [(w.text, w.start, w.end) for w in jumped] == [
            ("a", 50.0, 50.5), ("b", 52.0, 52.5), ("eh", 57.0, 60.0),
            ("c", 61.0, 61.5),
        ]

    def test_one_word_never_displaces_two(self) -> None:
        """The later copy has one word where the earlier has two.

        The doubled stretch runs 19-22.5 s (middle 20.75 s); ``eh``
        shares time with ``tres`` and ``y``, past the middle, where the
        later window would otherwise win.
        """
        first = words(("x", 10.0, 11.0), ("uno", 20.0, 20.5),
                      ("dos", 20.6, 21.0), ("tres", 21.1, 21.5),
                      ("y", 22.0, 22.5))
        second = words(("z0", 19.0, 19.2), ("eh", 21.0, 22.4),
                       ("z", 24.0, 24.5))
        assert texts(resolve_seams(first + second)) == [
            "x", "z0", "uno", "dos", "tres", "y", "z",
        ]

    def test_a_word_said_again_one_second_back_is_kept_once(self) -> None:
        given = words(("one", 0.0, 0.5), ("two", 9.0, 10.0),
                      ("three", 10.0, 10.5), ("two", 9.0, 9.5),
                      ("four", 11.0, 11.5))
        assert texts(resolve_seams(given)) == ["one", "two", "three", "four"]

    def test_the_last_word_just_past_the_reach_is_compared(self) -> None:
        pre = words(("x", 15.0, 16.0), ("y", 16.5, 17.5))
        first = words(("the", 18.0, 18.2), ("cat", 18.5, 18.7),
                      ("sat", 19.0, 19.2), ("down", 19.5, 19.7))
        second = words(("the", 18.05, 18.25), ("cat", 18.55, 18.75),
                       ("sat", 19.05, 19.25), ("down", 19.75, 19.95),
                       ("ok", 20.5, 21.0))
        assert texts(resolve_seams(pre + first + second)) == [
            "x", "y", "the", "cat", "sat", "down", "ok",
        ]

    def test_a_repetition_said_in_order_stays_whole(self) -> None:
        given = words(("no", 1.0, 1.3), ("no", 1.29, 1.6), ("no", 1.55, 1.9),
                      ("vale", 2.0, 2.4))
        assert texts(resolve_seams(given)) == ["no", "no", "no", "vale"]

    def test_one_long_word_anywhere_loses_at_most_itself(self) -> None:
        """Property: insert one word of any span into a clean stream."""
        rng = random.Random(27)
        for _ in range(500):
            clean = self.clean(rng.randint(2, 60))
            at = rng.randint(0, len(clean))
            start = rng.uniform(0, len(clean) + 2)
            length = rng.choice([
                rng.uniform(0, 3), rng.uniform(3, 20),
                rng.uniform(20, 120),
            ])
            text = rng.choice(["um", "eh", "bad", "w5", "palabra"])
            odd = Word(text=text, start=start, end=start + length)
            given = clean[:at] + [odd] + clean[at:]
            found = resolve_seams(given)
            assert len(found) >= len(clean)
            kept = [w.text for w in found]
            for word in clean:
                assert word.text in kept, (at, odd)
            starts = [w.start for w in found]
            assert starts == sorted(starts)
            assert resolve_seams(found) == found
